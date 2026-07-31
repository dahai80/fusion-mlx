// FineTuneScreenVM — view model for the Fine-Tune screen.
// callers: AppServices.fineTune (owns lifecycle), FineTuneScreen (reads state)
// API: /admin/api/fine-tune/* via FusionClient
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

    func start(client: FusionClient) async {
        self.client = client
        await loadModels()
        await pollOnce()
        startPolling()
    }

    func stop() {
        pollTask?.cancel()
        pollTask = nil
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
}
