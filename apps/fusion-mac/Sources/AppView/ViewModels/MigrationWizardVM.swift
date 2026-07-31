// Migration wizard view model — 7-step HF→MLX pipeline state machine.
// Callers: MigrationWizardScreen, AppServices; API: /admin/api/migrate/*
// User instruction verbatim: "做一个端到端的功能，做模型迁移和量化的功能，以openpangu为例，把迁移的每个步骤展现在GUI上"

import Foundation

enum MigrationStep: Int, CaseIterable, Identifiable {
    case source = 0, analyze, download, convert, codegen, quantize, validate

    var id: Int { rawValue }

    var title: String {
        switch self {
        case .source:   return "Source Selection"
        case .analyze:  return "Architecture Analysis"
        case .download: return "Weight Download"
        case .convert:  return "Weight Conversion"
        case .codegen:  return "Code Generation"
        case .quantize: return "Quantization"
        case .validate: return "Validation & Registration"
        }
    }
}

@MainActor
@Observable
final class MigrationWizardVM {
    var currentStep: MigrationStep = .source
    var hfId: String = ""
    var mirror: Bool = false
    var hfToken: String = ""
    var migrationId: String = ""
    var modelName: String = ""
    var quantBits: Int = 4
    var quantGroupSize: Int = 64

    var analysis: MigrationAnalysis?
    var convertResult: ConvertResult?
    var codegenResult: CodegenResult?
    var validationResult: ValidationResult?
    var downloadProgress: Double = 0
    var downloadStatus: String = ""

    var isLoading: Bool = false
    var lastError: String?
    var completedSteps: Set<MigrationStep> = []

    var canProceed: Bool {
        switch currentStep {
        case .source:   return !hfId.trimmingCharacters(in: .whitespaces).isEmpty
        case .analyze:  return analysis != nil && analysis?.compatible == true
        case .download: return downloadProgress >= 1.0 || completedSteps.contains(.download)
        case .convert:  return convertResult != nil && convertResult?.error == nil
        case .codegen:  return codegenResult != nil && codegenResult?.error == nil
        case .quantize: return true
        case .validate: return validationResult?.success == true
        }
    }

    func reset() {
        currentStep = .source
        hfId = ""
        migrationId = ""
        modelName = ""
        analysis = nil
        convertResult = nil
        codegenResult = nil
        validationResult = nil
        downloadProgress = 0
        downloadStatus = ""
        isLoading = false
        lastError = nil
        completedSteps = []
    }

    @discardableResult
    func runAnalyze(client: FusionClient) async -> Bool {
        guard !hfId.isEmpty else { return false }
        isLoading = true
        lastError = nil
        defer { isLoading = false }

        do {
            let result = try await client.migrateAnalyze(hfId: hfId, mirror: mirror)
            analysis = result
            if let err = result.error {
                lastError = err
                return false
            }
            completedSteps.insert(.source)
            completedSteps.insert(.analyze)
            return true
        } catch {
            lastError = error.fusionDescription
            return false
        }
    }

    @discardableResult
    func runDownload(client: FusionClient) async -> Bool {
        isLoading = true
        lastError = nil
        defer { isLoading = false }

        do {
            let info = try await client.migrateDownload(
                hfId: hfId, hfToken: hfToken, mirror: mirror
            )
            migrationId = info.migrationId
            if modelName.isEmpty {
                modelName = hfId.components(separatedBy: "/").last ?? hfId
            }
            downloadStatus = "Downloading..."
            completedSteps.insert(.download)
            downloadProgress = 1.0
            return true
        } catch {
            lastError = error.fusionDescription
            return false
        }
    }

    @discardableResult
    func runConvert(client: FusionClient) async -> Bool {
        guard !migrationId.isEmpty else { return false }
        isLoading = true
        lastError = nil
        defer { isLoading = false }

        do {
            let result = try await client.migrateConvert(
                migrationId: migrationId,
                quantBits: quantBits,
                quantGroupSize: quantGroupSize
            )
            convertResult = result
            if let err = result.error {
                lastError = err
                return false
            }
            completedSteps.insert(.convert)
            return true
        } catch {
            lastError = error.fusionDescription
            return false
        }
    }

    @discardableResult
    func runCodegen(client: FusionClient) async -> Bool {
        guard !migrationId.isEmpty else { return false }
        isLoading = true
        lastError = nil
        defer { isLoading = false }

        do {
            let result = try await client.migrateCodegen(migrationId: migrationId)
            codegenResult = result
            if let err = result.error {
                lastError = err
                return false
            }
            completedSteps.insert(.codegen)
            return true
        } catch {
            lastError = error.fusionDescription
            return false
        }
    }

    @discardableResult
    func runValidate(client: FusionClient) async -> Bool {
        guard !migrationId.isEmpty else { return false }
        isLoading = true
        lastError = nil
        defer { isLoading = false }

        do {
            let result = try await client.migrateValidate(migrationId: migrationId)
            validationResult = result
            if !result.success {
                lastError = result.error ?? "Validation failed"
                return false
            }
            completedSteps.insert(.validate)
            return true
        } catch {
            lastError = error.fusionDescription
            return false
        }
    }

    @discardableResult
    func runRegister(client: FusionClient) async -> Bool {
        guard !migrationId.isEmpty else { return false }
        isLoading = true
        lastError = nil
        defer { isLoading = false }

        do {
            let _ = try await client.migrateRegister(
                migrationId: migrationId, modelName: modelName
            )
            return true
        } catch {
            lastError = error.fusionDescription
            return false
        }
    }

    func advance() {
        guard canProceed else { return }
        let allSteps = MigrationStep.allCases
        if let idx = allSteps.firstIndex(of: currentStep), idx + 1 < allSteps.count {
            currentStep = allSteps[idx + 1]
        }
    }

    func goToStep(_ step: MigrationStep) {
        if completedSteps.contains(step) || step.rawValue <= (completedSteps.map(\.rawValue).max() ?? -1) + 1 {
            currentStep = step
        }
    }
}
