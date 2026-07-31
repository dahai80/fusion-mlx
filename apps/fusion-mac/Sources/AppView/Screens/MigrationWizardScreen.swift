// Migration wizard screen — 7-step guided HF→MLX conversion pipeline.
// Callers: AppView.swift screen(for: .migration); API: /admin/api/migrate/*
// User instruction verbatim: "做一个端到端的功能，做模型迁移和量化的功能，以openpangu为例，把迁移的每个步骤展现在GUI上"

import SwiftUI

struct MigrationWizardScreen: View {
    @Bindable var vm: MigrationWizardVM
    @Environment(\.fusionTheme) private var theme
    @Environment(AppServices.self) private var services

    var body: some View {
        VStack(alignment: .leading, spacing: 20) {
            stepIndicator
            stepContent
            errorBanner
            actionBar
        }
        .padding(20)
    }

    // MARK: - Step Indicator

    private var stepIndicator: some View {
        HStack(spacing: 0) {
            ForEach(MigrationStep.allCases) { step in
                stepPill(step)
                if step != .validate {
                    Rectangle()
                        .fill(vm.completedSteps.contains(step) ? theme.accent : theme.groupBorder)
                        .frame(height: 2)
                        .frame(maxWidth: .infinity)
                }
            }
        }
    }

    private func stepPill(_ step: MigrationStep) -> some View {
        let isActive = step == vm.currentStep
        let isDone = vm.completedSteps.contains(step)
        return VStack(spacing: 4) {
            ZStack {
                Circle()
                    .fill(isDone ? theme.accent : (isActive ? theme.accent.opacity(0.3) : theme.groupBorder))
                    .frame(width: 28, height: 28)
                if isDone {
                    Image(systemName: "checkmark")
                        .font(.system(size: 12, weight: .bold))
                        .foregroundStyle(.white)
                } else {
                    Text("\(step.rawValue + 1)")
                        .font(.fusionText(12, weight: .semibold))
                        .foregroundStyle(isActive ? theme.accent : theme.textTertiary)
                }
            }
            Text(step.title)
                .font(.fusionText(10))
                .foregroundStyle(isActive ? theme.text : theme.textTertiary)
                .lineLimit(1)
        }
        .frame(maxWidth: .infinity)
        .onTapGesture { vm.goToStep(step) }
    }

    // MARK: - Step Content

    @ViewBuilder
    private var stepContent: some View {
        switch vm.currentStep {
        case .source:   sourceStep
        case .analyze:  analyzeStep
        case .download: downloadStep
        case .convert:  convertStep
        case .codegen:  codegenStep
        case .quantize: quantizeStep
        case .validate: validateStep
        }
    }

    private var sourceStep: some View {
        VStack(alignment: .leading, spacing: 16) {
            SectionHeader("Model Source", icon: "square.and.arrow.down")
            GroupBox {
                VStack(alignment: .leading, spacing: 12) {
                    HStack {
                        Text("HuggingFace ID:")
                            .font(.fusionText(13))
                            .foregroundStyle(theme.textSecondary)
                        TextField("e.g. openpangu/OpenPangu-Embedded-7B", text: $vm.hfId)
                            .textFieldStyle(.roundedBorder)
                            .font(.fusionText(13))
                    }
                    HStack {
                        Text("HF Token (optional):")
                            .font(.fusionText(13))
                            .foregroundStyle(theme.textSecondary)
                        SecureField("hf_xxxx...", text: $vm.hfToken)
                            .textFieldStyle(.roundedBorder)
                            .font(.fusionText(13))
                    }
                    Toggle("Use hf-mirror.com (China)", isOn: $vm.mirror)
                        .font(.fusionText(13))
                        .toggleStyle(.checkbox)
                }
                .padding(12)
            }
        }
    }

    private var analyzeStep: some View {
        VStack(alignment: .leading, spacing: 16) {
            SectionHeader("Architecture Analysis", icon: "magnifyingglass")
            if let analysis = vm.analysis {
                GroupBox {
                    VStack(alignment: .leading, spacing: 8) {
                        infoRow("Model Type", value: analysis.modelType)
                        infoRow("Architectures", value: analysis.architectures.joined(separator: ", "))
                        infoRow("Template", value: analysis.template ?? "auto-detect")
                        if let diff = analysis.diff, !diff.isEmpty {
                            infoRow("Differences", value: diff.joined(separator: "; "))
                        }
                        infoRow("Params", value: String(format: "%.1fB", analysis.numParamsB))
                        infoRow("Est. Size", value: String(format: "%.1f GB", analysis.estimatedSizeGb))
                        infoRow("Compatible", value: analysis.compatible ? "✓ Yes" : "✗ No")
                        if !analysis.warnings.isEmpty {
                            VStack(alignment: .leading, spacing: 4) {
                                Text("Warnings:")
                                    .font(.fusionText(12, weight: .semibold))
                                    .foregroundStyle(theme.textSecondary)
                                ForEach(analysis.warnings, id: \.self) { w in
                                    Text("• \(w)")
                                        .font(.fusionText(12))
                                        .foregroundStyle(.orange)
                                }
                            }
                        }
                    }
                    .padding(12)
                }
            } else {
                Text("Click Run to analyze the model architecture.")
                    .font(.fusionText(13))
                    .foregroundStyle(theme.textTertiary)
            }
        }
    }

    private func infoRow(_ label: String, value: String) -> some View {
        HStack {
            Text(label + ":")
                .font(.fusionText(13))
                .foregroundStyle(theme.textSecondary)
            Text(value)
                .font(.fusionText(13, weight: .medium))
                .foregroundStyle(theme.text)
            Spacer()
        }
    }

    private var downloadStep: some View {
        VStack(alignment: .leading, spacing: 16) {
            SectionHeader("Weight Download", icon: "icloud.and.arrow.down")
            if vm.completedSteps.contains(.download) {
                GroupBox {
                    HStack {
                        Image(systemName: "checkmark.circle.fill")
                            .foregroundStyle(.green)
                        Text("Download complete — Migration ID: \(vm.migrationId)")
                            .font(.fusionText(13))
                    }
                    .padding(12)
                }
            } else {
                Text("Click Run to download model weights from HuggingFace.")
                    .font(.fusionText(13))
                    .foregroundStyle(theme.textTertiary)
            }
        }
    }

    private var convertStep: some View {
        VStack(alignment: .leading, spacing: 16) {
            SectionHeader("Weight Conversion", icon: "arrow.triangle.2.circlepath")
            if let result = vm.convertResult {
                GroupBox {
                    VStack(alignment: .leading, spacing: 8) {
                        infoRow("Output", value: result.outputDir)
                        infoRow("Weights", value: "\(result.numWeights)")
                        infoRow("Params", value: String(format: "%.1fB", result.totalParamsB))
                        if !result.orphans.isEmpty {
                            infoRow("Orphan Keys", value: "\(result.orphans.count)")
                        }
                        if !result.missing.isEmpty {
                            infoRow("Missing Keys", value: "\(result.missing.count)")
                        }
                    }
                    .padding(12)
                }
            } else {
                Text("Click Run to convert HF weights to MLX format.")
                    .font(.fusionText(13))
                    .foregroundStyle(theme.textTertiary)
            }
        }
    }

    private var codegenStep: some View {
        VStack(alignment: .leading, spacing: 16) {
            SectionHeader("Code Generation", icon: "chevron.left.forwardslash.chevron.right")
            if let result = vm.codegenResult {
                GroupBox {
                    VStack(alignment: .leading, spacing: 8) {
                        infoRow("Output", value: result.outputPath)
                        ForEach(result.filesGenerated, id: \.self) { f in
                            infoRow("File", value: f)
                        }
                    }
                    .padding(12)
                }
            } else {
                Text("Click Run to generate MLX model class code.")
                    .font(.fusionText(13))
                    .foregroundStyle(theme.textTertiary)
            }
        }
    }

    private var quantizeStep: some View {
        VStack(alignment: .leading, spacing: 16) {
            SectionHeader("Quantization", icon: "sparkles")
            GroupBox {
                VStack(alignment: .leading, spacing: 12) {
                    HStack {
                        Text("Quantization Bits:")
                            .font(.fusionText(13))
                            .foregroundStyle(theme.textSecondary)
                        Picker("", selection: $vm.quantBits) {
                            Text("None (BF16)").tag(0)
                            Text("4-bit").tag(4)
                            Text("8-bit").tag(8)
                        }
                        .pickerStyle(.segmented)
                        .frame(width: 240)
                    }
                    if vm.quantBits > 0 {
                        HStack {
                            Text("Group Size:")
                                .font(.fusionText(13))
                                .foregroundStyle(theme.textSecondary)
                            Picker("", selection: $vm.quantGroupSize) {
                                Text("32").tag(32)
                                Text("64").tag(64)
                                Text("128").tag(128)
                            }
                            .pickerStyle(.segmented)
                            .frame(width: 180)
                        }
                    }
                }
                .padding(12)
            }
            Text("Quantization is applied during conversion. Click Next to proceed.")
                .font(.fusionText(12))
                .foregroundStyle(theme.textTertiary)
        }
    }

    private var validateStep: some View {
        VStack(alignment: .leading, spacing: 16) {
            SectionHeader("Validation & Registration", icon: "checkmark.shield")
            if let result = vm.validationResult {
                GroupBox {
                    VStack(alignment: .leading, spacing: 8) {
                        infoRow("Status", value: result.success ? "✓ Passed" : "✗ Failed")
                        infoRow("Output", value: String(result.outputText.prefix(100)))
                        infoRow("Speed", value: String(format: "%.1f tok/s", result.tokensPerSec))
                    }
                    .padding(12)
                }
            }
            GroupBox {
                VStack(alignment: .leading, spacing: 12) {
                    HStack {
                        Text("Model Name:")
                            .font(.fusionText(13))
                            .foregroundStyle(theme.textSecondary)
                        TextField("model-name", text: $vm.modelName)
                            .textFieldStyle(.roundedBorder)
                            .font(.fusionText(13))
                    }
                    Button("Register Model") {
                        Task { await vm.runRegister(client: services.client) }
                    }
                    .buttonStyle(.fusion(.primary))
                    .disabled(vm.modelName.isEmpty || vm.isLoading)
                }
                .padding(12)
            }
        }
    }

    // MARK: - Error Banner

    @ViewBuilder
    private var errorBanner: some View {
        if let err = vm.lastError {
            HStack {
                Image(systemName: "exclamationmark.triangle.fill")
                    .foregroundStyle(.orange)
                Text(err)
                    .font(.fusionText(12))
                    .foregroundStyle(theme.textSecondary)
                Spacer()
                Button { vm.lastError = nil } label: {
                    Image(systemName: "xmark")
                        .font(.system(size: 10))
                }
                .buttonStyle(.fusion(.plain, size: .small))
            }
            .padding(10)
            .background(theme.groupBg)
            .clipShape(RoundedRectangle(cornerRadius: 8))
        }
    }

    // MARK: - Action Bar

    private var actionBar: some View {
        HStack {
            Button("Reset") { vm.reset() }
                .buttonStyle(.fusion(.normal))

            Spacer()

            if vm.currentStep != .source {
                Button("Back") {
                    let all = MigrationStep.allCases
                    if let idx = all.firstIndex(of: vm.currentStep), idx > 0 {
                        vm.currentStep = all[idx - 1]
                    }
                }
                .buttonStyle(.fusion(.normal))
            }

            Button("Run") {
                Task { await runCurrentStep() }
            }
            .buttonStyle(.fusion(.primary))
            .disabled(vm.isLoading)

            if vm.canProceed && vm.currentStep != .validate {
                Button("Next") { vm.advance() }
                    .buttonStyle(.fusion(.primary))
            }
        }
    }

    private func runCurrentStep() async {
        let client = services.client
        switch vm.currentStep {
        case .source, .analyze:
            let ok = await vm.runAnalyze(client: client)
            if ok { vm.advance() }
        case .download:
            let ok = await vm.runDownload(client: client)
            if ok { vm.advance() }
        case .convert:
            let ok = await vm.runConvert(client: client)
            if ok { vm.advance() }
        case .codegen:
            let ok = await vm.runCodegen(client: client)
            if ok { vm.advance() }
        case .quantize:
            vm.advance()
        case .validate:
            let _ = await vm.runValidate(client: client)
        }
    }
}

// MARK: - Section Header Helper

private struct SectionHeader: View {
    let title: String
    let icon: String
    @Environment(\.fusionTheme) private var theme

    init(_ title: String, icon: String) {
        self.title = title
        self.icon = icon
    }

    var body: some View {
        HStack(spacing: 8) {
            Image(systemName: icon)
                .font(.system(size: 14, weight: .semibold))
                .foregroundStyle(theme.accent)
            Text(title)
                .font(.fusionText(15, weight: .semibold))
                .foregroundStyle(theme.text)
        }
    }
}
