// FineTuneScreen — SwiftUI view for LoRA/DORA fine-tuning job management.
// callers: AppView (screen routing via case .fineTune), AppServices.fineTune (VM owner)
// API: /admin/api/fine-tune/* via FusionClient (jobs CRUD, SSE stream, adapters, models)
// Data schemas: FineTuneJobDTO, FineTuneConfigDTO, FineTuneProgressDTO, FineTuneAdapterDTO, FineTuneModelDTO
// User instruction: "开始做，注意设计方案需要有GUI的设计和落地方案，提交给macos app"

import SwiftUI

struct FineTuneScreen: View {
    @Environment(AppServices.self) private var services
    @Bindable var vm: FineTuneScreenVM

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            ScreenHeader(
                eyebrow: String(localized: "finetune.header.eyebrow",
                                defaultValue: "Fine-Tune",
                                comment: "Eyebrow above the Fine-Tune screen header"),
                title: String(localized: "finetune.header.title",
                              defaultValue: "Train LoRA / DORA adapters",
                              comment: "Fine-Tune screen primary header"),
                subtitle: String(localized: "finetune.header.subtitle",
                                 defaultValue: "Fine-tune loaded models with mlx_lm. Training evicts the inference model; it reloads after completion. Adapters are saved to ~/.fusion-mlx/adapters/.",
                                 comment: "Fine-Tune screen subtitle")
            )

            ConfigSection(vm: vm, client: services.client)

            if let activeJob = vm.jobs.first(where: { $0.status == "running" }) {
                ActiveJobSection(job: activeJob)
            }

            MessageBanner(error: vm.lastError)

            if !vm.jobs.isEmpty {
                JobsSection(
                    jobs: vm.jobs,
                    onCancel: { id in vm.cancelJob(client: services.client, jobId: id) },
                    onDelete: { id in vm.deleteJob(client: services.client, jobId: id) }
                )
            }

            if !vm.adapters.isEmpty {
                AdaptersSection(
                    adapters: vm.adapters,
                    onDelete: { mid, aname in vm.deleteAdapter(client: services.client, modelId: mid, adapterName: aname) }
                )
            }
        }
        .task { await vm.start(client: services.client) }
    }
}

// MARK: - Configuration

private struct ConfigSection: View {
    @Bindable var vm: FineTuneScreenVM
    let client: FusionClient
    @State private var advancedOpen = false
    @Environment(\.fusionTheme) private var theme

    var body: some View {
        SectionHeader(
            String(localized: "finetune.section.config",
                   defaultValue: "Configuration",
                   comment: "Section header for the Fine-Tune configuration block"),
            subtitle: vm.hasActiveJob
                ? String(localized: "finetune.config.busy",
                         defaultValue: "A job is already running",
                         comment: "Subtitle when a fine-tune job is active")
                : nil
        )

        ListGroup {
            Row(
                label: String(localized: "finetune.config.model",
                              defaultValue: "Model",
                              comment: "Label for the model picker"),
                sublabel: String(localized: "finetune.config.model.sub",
                                 defaultValue: "Models available in the engine pool",
                                 comment: "Sublabel for model picker")
            ) {
                Popup(
                    selection: $vm.selectedModelId,
                    width: 320,
                    options: vm.models.map { ($0.model_id, $0.model_id) }
                )
            }

            Row(
                label: String(localized: "finetune.config.dataset",
                              defaultValue: "Dataset path",
                              comment: "Label for the dataset path input"),
                sublabel: String(localized: "finetune.config.dataset.sub",
                                 defaultValue: "Local path or HuggingFace repo ID",
                                 comment: "Sublabel for dataset path")
            ) {
                TextField("", text: $vm.datasetPath)
                    .textFieldStyle(.roundedBorder)
                    .frame(width: 320)
            }

            Row(
                label: String(localized: "finetune.config.adapter",
                              defaultValue: "Adapter name",
                              comment: "Label for adapter name input"),
                sublabel: String(localized: "finetune.config.adapter.sub",
                                 defaultValue: "Optional; auto-generated if blank",
                                 comment: "Sublabel for adapter name")
            ) {
                TextField("", text: $vm.adapterName)
                    .textFieldStyle(.roundedBorder)
                    .frame(width: 320)
            }

            Row(
                label: String(localized: "finetune.config.type",
                              defaultValue: "Type",
                              comment: "Label for LoRA/DORA type picker"),
                sublabel: nil
            ) {
                Popup(
                    selection: $vm.config.fine_tune_type,
                    width: 120,
                    options: [("lora", "LoRA"), ("dora", "DORA")]
                )
            }
        }

        // Advanced toggle
        Button {
            withAnimation(.easeInOut(duration: 0.15)) {
                advancedOpen.toggle()
            }
        } label: {
            HStack(spacing: 4) {
                Image(systemName: advancedOpen ? "chevron.down" : "chevron.right")
                    .font(.system(size: 9, weight: .semibold))
                Text(String(localized: "finetune.config.advanced",
                            defaultValue: "Advanced",
                            comment: "Toggle label for advanced config"))
                    .font(.fusionText(11.5, weight: .medium))
            }
            .foregroundStyle(theme.accent)
        }
        .buttonStyle(.plain)
        .padding(.horizontal, 18)
        .padding(.top, 10)

        if advancedOpen {
            AdvancedConfigSection(config: $vm.config, theme: theme)
        }

        // Submit
        HStack {
            Spacer()
            Button {
                vm.submitJob(client: client)
            } label: {
                HStack(spacing: 5) {
                    if vm.isSubmitting {
                        ProgressView()
                            .controlSize(.small)
                    }
                    Text(vm.isSubmitting
                         ? String(localized: "finetune.submit.running",
                                  defaultValue: "Submitting…",
                                  comment: "Button label while submitting")
                         : String(localized: "finetune.submit.start",
                                  defaultValue: "Start Training",
                                  comment: "Button label to start training"))
                        .font(.fusionText(12, weight: .semibold))
                }
            }
            .buttonStyle(.borderedProminent)
            .disabled(!vm.canSubmit)
            .padding(.trailing, 18)
            .padding(.top, 10)
        }
    }
}

// MARK: - Advanced Config

private struct AdvancedConfigSection: View {
    @Binding var config: FineTuneConfigDTO
    let theme: FusionTheme

    var body: some View {
        ListGroup {
            Row(label: "LoRA Rank") {
                Stepper(value: $config.lora_rank, in: 1...64) {
                    Text("\(config.lora_rank)")
                        .font(.fusionMono(11.5))
                        .frame(width: 40, alignment: .trailing)
                }
            }
            Row(label: "LoRA Alpha") {
                TextField("", value: $config.lora_alpha, format: .number)
                    .textFieldStyle(.roundedBorder)
                    .frame(width: 80)
            }
            Row(label: "LoRA Layers") {
                Stepper(value: $config.lora_layers, in: 1...64) {
                    Text("\(config.lora_layers)")
                        .font(.fusionMono(11.5))
                        .frame(width: 40, alignment: .trailing)
                }
            }
            Row(label: "Learning Rate") {
                TextField("", value: $config.learning_rate, format: .scientific)
                    .textFieldStyle(.roundedBorder)
                    .frame(width: 120)
            }
            Row(label: "Optimizer") {
                Popup(
                    selection: $config.optimizer,
                    width: 120,
                    options: [("adamw", "AdamW"), ("adam", "Adam"), ("sgd", "SGD")]
                )
            }
            Row(label: "Batch Size") {
                Stepper(value: $config.batch_size, in: 1...32) {
                    Text("\(config.batch_size)")
                        .font(.fusionMono(11.5))
                        .frame(width: 40, alignment: .trailing)
                }
            }
            Row(label: "Iterations") {
                Stepper(value: $config.iters, in: 10...10000, step: 100) {
                    Text("\(config.iters)")
                        .font(.fusionMono(11.5))
                        .frame(width: 60, alignment: .trailing)
                }
            }
            Row(label: "Max Seq Length") {
                Stepper(value: $config.max_seq_length, in: 128...8192, step: 128) {
                    Text("\(config.max_seq_length)")
                        .font(.fusionMono(11.5))
                        .frame(width: 60, alignment: .trailing)
                }
            }
            Row(label: "Gradient Checkpointing") {
                Toggle("", isOn: $config.gradient_checkpointing)
                    .labelsHidden()
                    .toggleStyle(.switch)
            }
            Row(label: "Grad Accumulation") {
                Stepper(value: $config.grad_accumulation_steps, in: 1...16) {
                    Text("\(config.grad_accumulation_steps)")
                        .font(.fusionMono(11.5))
                        .frame(width: 40, alignment: .trailing)
                }
            }
            Row(label: "Steps per Report") {
                Stepper(value: $config.steps_per_report, in: 1...1000, step: 10) {
                    Text("\(config.steps_per_report)")
                        .font(.fusionMono(11.5))
                        .frame(width: 40, alignment: .trailing)
                }
            }
            Row(label: "Steps per Eval") {
                Stepper(value: $config.steps_per_eval, in: 1...1000, step: 50) {
                    Text("\(config.steps_per_eval)")
                        .font(.fusionMono(11.5))
                        .frame(width: 40, alignment: .trailing)
                }
            }
            Row(label: "Steps per Save") {
                Stepper(value: $config.steps_per_save, in: 1...1000, step: 50) {
                    Text("\(config.steps_per_save)")
                        .font(.fusionMono(11.5))
                        .frame(width: 40, alignment: .trailing)
                }
            }
            Row(label: "Mask Prompt") {
                Toggle("", isOn: $config.mask_prompt)
                    .labelsHidden()
                    .toggleStyle(.switch)
            }
            Row(label: "Seed") {
                Stepper(value: $config.seed, in: 0...99999) {
                    Text(config.seed == 0 ? "random" : "\(config.seed)")
                        .font(.fusionMono(11.5))
                        .frame(width: 60, alignment: .trailing)
                }
            }
        }
        .padding(.top, 6)
    }
}

// MARK: - Active Job Progress

private struct ActiveJobSection: View {
    let job: FineTuneJobDTO
    @Environment(\.fusionTheme) private var theme

    private var progress: FineTuneProgressDTO { job.progress }
    private var fraction: Double {
        progress.total_steps > 0
            ? Double(progress.step) / Double(progress.total_steps)
            : 0
    }

    var body: some View {
        SectionHeader(
            String(localized: "finetune.section.progress",
                   defaultValue: "Training Progress",
                   comment: "Section header for the live training progress"),
            subtitle: String(
                localized: "finetune.progress.step",
                defaultValue: "Step \(progress.step) / \(progress.total_steps)",
                comment: "Step counter in the progress section")
        )

        VStack(spacing: 10) {
            ProgressView(value: fraction)
                .progressViewStyle(.linear)
                .tint(theme.accent)

            HStack(spacing: 24) {
                MetricPill(label: "Train Loss", value: String(format: "%.4f", progress.train_loss))
                if let vl = progress.val_loss {
                    MetricPill(label: "Val Loss", value: String(format: "%.4f", vl))
                }
                MetricPill(label: "LR", value: String(format: "%.2e", progress.learning_rate))
                MetricPill(label: "tok/s", value: String(format: "%.1f", progress.tokens_per_second))
                if progress.peak_memory_gb > 0 {
                    MetricPill(label: "Peak Mem", value: String(format: "%.1f GB", progress.peak_memory_gb))
                }
                if progress.eta_seconds > 0 {
                    MetricPill(label: "ETA", value: formatDuration(progress.eta_seconds))
                }
            }
        }
        .padding(.horizontal, 18)
        .padding(.bottom, 12)
    }
}

private struct MetricPill: View {
    let label: String
    let value: String
    @Environment(\.fusionTheme) private var theme

    var body: some View {
        VStack(spacing: 2) {
            Text(label)
                .font(.fusionText(9.5, weight: .medium))
                .foregroundStyle(theme.textTertiary)
            Text(value)
                .font(.fusionMono(11))
                .foregroundStyle(theme.text)
        }
    }
}

// MARK: - Jobs List

private struct JobsSection: View {
    let jobs: [FineTuneJobDTO]
    let onCancel: (String) -> Void
    let onDelete: (String) -> Void
    @Environment(\.fusionTheme) private var theme

    var body: some View {
        SectionHeader(
            String(localized: "finetune.section.jobs",
                   defaultValue: "Jobs",
                   comment: "Section header for the jobs list"),
            subtitle: String(localized: "finetune.jobs.count",
                             defaultValue: "\(jobs.count) job(s)",
                             comment: "Job count subtitle")
        )

        ListGroup {
            ForEach(jobs) { job in
                JobRow(job: job, onCancel: onCancel, onDelete: onDelete)
            }
        }
    }
}

private struct JobRow: View {
    let job: FineTuneJobDTO
    let onCancel: (String) -> Void
    let onDelete: (String) -> Void
    @Environment(\.fusionTheme) private var theme

    private var statusColor: Color {
        switch job.status {
        case "queued": return theme.textTertiary
        case "running": return theme.accent
        case "completed": return .green
        case "failed", "cancelled": return .red
        default: return theme.textTertiary
        }
    }

    var body: some View {
        HStack(spacing: 10) {
            Circle()
                .fill(statusColor)
                .frame(width: 8, height: 8)

            VStack(alignment: .leading, spacing: 2) {
                HStack(spacing: 6) {
                    Text(job.model_id)
                        .font(.fusionText(12, weight: .medium))
                        .foregroundStyle(theme.text)
                    Text(job.config.fine_tune_type.uppercased())
                        .font(.fusionText(9.5, weight: .semibold))
                        .foregroundStyle(theme.accentText)
                        .padding(.horizontal, 5)
                        .padding(.vertical, 1)
                        .background(theme.accent.opacity(0.15))
                        .clipShape(Capsule())
                }
                Text(job.dataset)
                    .font(.fusionText(10.5))
                    .foregroundStyle(theme.textSecondary)
                    .lineLimit(1)
            }

            Spacer(minLength: 0)

            if !job.error.isEmpty {
                Text(job.error)
                    .font(.fusionText(10))
                    .foregroundStyle(.red)
                    .lineLimit(1)
                    .frame(maxWidth: 180, alignment: .trailing)
            }

            if job.status == "running" || job.status == "queued" {
                Button {
                    onCancel(job.id)
                } label: {
                    Text(String(localized: "finetune.job.cancel",
                                defaultValue: "Cancel",
                                comment: "Cancel job button"))
                        .font(.fusionText(10.5, weight: .medium))
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
            }

            if job.status == "completed" || job.status == "failed" || job.status == "cancelled" {
                Button {
                    onDelete(job.id)
                } label: {
                    Image(systemName: "trash")
                        .font(.system(size: 10))
                        .foregroundStyle(.red)
                }
                .buttonStyle(.plain)
            }
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 8)
    }
}

// MARK: - Adapters

private struct AdaptersSection: View {
    let adapters: [FineTuneAdapterDTO]
    let onDelete: (String, String) -> Void
    @Environment(\.fusionTheme) private var theme

    var body: some View {
        SectionHeader(
            String(localized: "finetune.section.adapters",
                   defaultValue: "Saved Adapters",
                   comment: "Section header for the adapters list"),
            subtitle: String(localized: "finetune.adapters.count",
                             defaultValue: "\(adapters.count) adapter(s)",
                             comment: "Adapter count subtitle")
        )

        ListGroup {
            ForEach(adapters) { adapter in
                AdapterRow(adapter: adapter, onDelete: onDelete)
            }
        }
    }
}

private struct AdapterRow: View {
    let adapter: FineTuneAdapterDTO
    let onDelete: (String, String) -> Void
    @Environment(\.fusionTheme) private var theme

    var body: some View {
        HStack(spacing: 10) {
            Image(systemName: "brain.head.profile")
                .font(.system(size: 12))
                .foregroundStyle(theme.accent)

            VStack(alignment: .leading, spacing: 2) {
                HStack(spacing: 6) {
                    Text(adapter.adapter_name)
                        .font(.fusionText(12, weight: .medium))
                        .foregroundStyle(theme.text)
                    Text(adapter.model_id)
                        .font(.fusionText(10.5))
                        .foregroundStyle(theme.textSecondary)
                }
                HStack(spacing: 8) {
                    if let ft = adapter.fine_tune_type {
                        Text(ft.uppercased())
                            .font(.fusionText(9, weight: .semibold))
                            .foregroundStyle(theme.accentText)
                            .padding(.horizontal, 4)
                            .padding(.vertical, 1)
                            .background(theme.accent.opacity(0.12))
                            .clipShape(Capsule())
                    }
                    if let rank = adapter.lora_rank {
                        Text("r=\(rank)")
                            .font(.fusionMono(9.5))
                            .foregroundStyle(theme.textTertiary)
                    }
                    if let layers = adapter.lora_layers {
                        Text("layers=\(layers)")
                            .font(.fusionMono(9.5))
                            .foregroundStyle(theme.textTertiary)
                    }
                    if adapter.has_weights {
                        Text("weights ✓")
                            .font(.fusionText(9.5))
                            .foregroundStyle(.green)
                    }
                }
            }

            Spacer(minLength: 0)

            Button {
                onDelete(adapter.model_id, adapter.adapter_name)
            } label: {
                Image(systemName: "trash")
                    .font(.system(size: 10))
                    .foregroundStyle(.red)
            }
            .buttonStyle(.plain)
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 8)
    }
}

// MARK: - Helpers

private func formatDuration(_ seconds: Double) -> String {
    let mins = Int(seconds) / 60
    let secs = Int(seconds) % 60
    if mins > 60 {
        let hrs = mins / 60
        return "\(hrs)h \(mins % 60)m"
    }
    return mins > 0 ? "\(mins)m \(secs)s" : "\(secs)s"
}
