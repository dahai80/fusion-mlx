// FineTuneScreenVM — view model for the Fine-Tune screen.
// callers: AppServices.fineTune (owns lifecycle), FineTuneScreen (reads state)
// API: /admin/api/fine-tune/* via FusionClient, SSE stream for live progress
// User instruction: "开始做，注意设计方案需要有GUI的设计和落地方案，提交给macos app"

import SwiftUI

@MainActor
@Observable
final class FineTuneScreenVM {
    var selectedModelId: String = ""
    var datasetPath: String = ""
    var adapterName: String = ""
    var config = FineTuneConfigDTO()

    private(set) var models: [FineTuneModelDTO] = []
    private(set) var jobs: [FineTuneJobDTO] = []
    private(set) var adapters: [FineTuneAdapterDTO] = []

    private(set) var isSubmitting: Bool = false
    var lastError: String?

    var canSubmit: Bool {
        !selectedModelId.isEmpty && !datasetPath.isEmpty && !isSubmitting
    }

    var hasActiveJob: Bool {
        jobs.contains { $0.status == "running" || $0.status == "queued" }
    }

    @ObservationIgnored
    private weak var client: FusionClient?
    @ObservationIgnored
    private var pollTask: Task<Void, Never>?
    @ObservationIgnored
    private var sseTask: Task<Void, Never>?
    @ObservationIgnored
    private var streamedJobId: String?

    func start(client: FusionClient) async {
        self.client = client
        await loadModels()
        await pollOnce()
        startPolling()
    }

    func stop() {
        pollTask?.cancel()
        pollTask = nil
        stopSSE()
    }

    private func loadModels() async {
        guard let client else { return }
        do {
            self.models = try await client.listFineTuneModels()
        } catch {
            self.lastError = "Failed to load models: \(error.fusionDescription)"
        }
    }

    private func pollOnce() async {
        guard let client else { return }
        async let jobsFetch: [FineTuneJobDTO] = client.listFineTuneJobs()
        async let adaptersFetch: [FineTuneAdapterDTO] = client.listFineTuneAdapters()
        do { self.jobs = try await jobsFetch } catch {}
        do { self.adapters = try await adaptersFetch } catch {}
        maybeStartSSE()
    }

    private func startPolling() {
        pollTask?.cancel()
        pollTask = Task { [weak self] in
            while !Task.isCancelled {
                guard let self else { return }
                try? await Task.sleep(for: .seconds(self.hasActiveJob ? 2 : 8))
                if Task.isCancelled { return }
                await self.pollOnce()
            }
        }
    }

    // MARK: - SSE Live Progress

    private func stopSSE() {
        sseTask?.cancel()
        sseTask = nil
        streamedJobId = nil
    }

    private func maybeStartSSE() {
        guard let runningJob = jobs.first(where: { $0.status == "running" }) else {
            if let sJobId = streamedJobId,
               let job = jobs.first(where: { $0.job_id == sJobId }),
               job.status != "running" {
                os_log(.info, "SSE: job \(sJobId) left running state (\(job.status)), stopping stream")
                stopSSE()
            }
            return
        }

        if runningJob.job_id == streamedJobId { return }

        stopSSE()
        streamedJobId = runningJob.job_id
        guard let client else { return }

        os_log(.info, "SSE: attaching to job \(runningJob.job_id)")

        let jobId = runningJob.job_id
        sseTask = Task { [weak self] in
            let stream = client.streamFineTuneJob(jobId: jobId)
            do {
                for try await event in stream {
                    if Task.isCancelled { return }
                    guard let self else { return }
                    self.handleSSEEvent(event, jobId: jobId)
                }
            } catch {
                os_log(.info, "SSE: stream ended for job \(jobId): \(error.localizedDescription)")
            }
            guard let self else { return }
            await self.pollOnce()
        }
    }

    private func handleSSEEvent(_ event: [String: Any], jobId: String) {
        guard let idx = jobs.firstIndex(where: { $0.job_id == jobId }) else { return }

        var progress = jobs[idx].progress
        if let v = event["step"] as? Int { progress.step = v }
        if let v = event["total_steps"] as? Int { progress.total_steps = v }
        if let v = event["train_loss"] as? Double { progress.train_loss = v }
        if let v = event["val_loss"] as? Double { progress.val_loss = v }
        if let v = event["learning_rate"] as? Double { progress.learning_rate = v }
        if let v = event["tokens_per_second"] as? Double { progress.tokens_per_second = v }
        if let v = event["iterations_per_second"] as? Double { progress.iterations_per_second = v }
        if let v = event["trained_tokens"] as? Int { progress.trained_tokens = v }
        if let v = event["peak_memory_gb"] as? Double { progress.peak_memory_gb = v }
        if let v = event["elapsed_seconds"] as? Double { progress.elapsed_seconds = v }
        if let v = event["eta_seconds"] as? Double { progress.eta_seconds = v }

        var job = jobs[idx]
        job.progress = progress

        if let status = event["status"] as? String {
            job.status = status
        }

        jobs[idx] = job
    }

    // MARK: - Actions

    func submitJob(client: FusionClient) {
        guard canSubmit else { return }
        let req = CreateFineTuneJobRequest(
            model_id: selectedModelId,
            dataset: datasetPath,
            adapter_name: adapterName.isEmpty ? nil : adapterName,
            config: config
        )
        isSubmitting = true
        lastError = nil
        Task { [weak self] in
            defer { Task { @MainActor [weak self] in self?.isSubmitting = false } }
            do {
                _ = try await client.createFineTuneJob(req)
                await self?.pollOnce()
            } catch {
                await MainActor.run {
                    self?.lastError = "Failed to create job: \(error.fusionDescription)"
                }
            }
        }
    }

    func cancelJob(client: FusionClient, jobId: String) {
        Task { [weak self] in
            do {
                _ = try await client.cancelFineTuneJob(id: jobId)
                await self?.pollOnce()
            } catch {
                await MainActor.run {
                    self?.lastError = "Failed to cancel: \(error.fusionDescription)"
                }
            }
        }
    }

    func deleteJob(client: FusionClient, jobId: String) {
        Task { [weak self] in
            do {
                _ = try await client.deleteFineTuneJob(id: jobId)
                await self?.pollOnce()
            } catch {
                await MainActor.run {
                    self?.lastError = "Failed to delete job: \(error.fusionDescription)"
                }
            }
        }
    }

    func deleteAdapter(client: FusionClient, modelId: String, adapterName: String) {
        Task { [weak self] in
            do {
                _ = try await client.deleteFineTuneAdapter(modelId: modelId, adapterName: adapterName)
                await self?.pollOnce()
            } catch {
                await MainActor.run {
                    self?.lastError = "Failed to delete adapter: \(error.fusionDescription)"
                }
            }
        }
    }

    func serveAdapter(client: FusionClient, modelId: String, adapterName: String) {
        Task { [weak self] in
            do {
                let result = try await client.serveFineTuneAdapter(modelId: modelId, adapterName: adapterName)
                os_log(.info, "Adapter served: \(result.description)")
            } catch {
                await MainActor.run {
                    self?.lastError = "Failed to serve adapter: \(error.fusionDescription)"
                }
            }
        }
    }

    func unloadAdapter(client: FusionClient, modelId: String, adapterName: String) {
        Task { [weak self] in
            do {
                _ = try await client.unloadFineTuneAdapter(modelId: modelId, adapterName: adapterName)
            } catch {
                await MainActor.run {
                    self?.lastError = "Failed to unload adapter: \(error.fusionDescription)"
                }
            }
        }
    }
}
