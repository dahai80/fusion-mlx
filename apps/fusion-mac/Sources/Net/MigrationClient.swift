// Migration wizard API methods for FusionClient.
// Callers: MigrationWizardVM; API: /admin/api/migrate/*
// User instruction verbatim: "做一个端到端的功能，做模型迁移和量化的功能，以openpangu为例，把迁移的每个步骤展现在GUI上"

import Foundation

// MARK: - Migration DTOs

struct MigrationAnalysis: Decodable, Sendable {
    let hfId: String
    let modelType: String
    let architectures: [String]
    let template: String?
    let diff: [String]?
    let estimatedSizeGb: Double
    let numParamsB: Double
    let compatible: Bool
    let warnings: [String]
    let error: String?

    enum CodingKeys: String, CodingKey {
        case hfId = "hf_id"
        case modelType = "model_type"
        case architectures
        case template
        case diff
        case estimatedSizeGb = "estimated_size_gb"
        case numParamsB = "num_params_b"
        case compatible
        case warnings
        case error
    }
}

struct MigrationTaskInfo: Decodable, Sendable {
    let migrationId: String
    let task: [String: AnyCodable]?

    enum CodingKeys: String, CodingKey {
        case migrationId = "migration_id"
        case task
    }
}

struct ConvertResult: Decodable, Sendable {
    let outputDir: String
    let numWeights: Int
    let totalParamsB: Double
    let orphans: [String]
    let missing: [String]
    let error: String?

    enum CodingKeys: String, CodingKey {
        case outputDir = "output_dir"
        case numWeights = "num_weights"
        case totalParamsB = "total_params_b"
        case orphans, missing, error
    }
}

struct ValidationResult: Decodable, Sendable {
    let success: Bool
    let outputText: String
    let tokensPerSec: Double
    let numTokens: Int
    let error: String?

    enum CodingKeys: String, CodingKey {
        case success
        case outputText = "output_text"
        case tokensPerSec = "tokens_per_sec"
        case numTokens = "num_tokens"
        case error
    }
}

struct CodegenResult: Decodable, Sendable {
    let outputPath: String
    let filesGenerated: [String]
    let error: String?

    enum CodingKeys: String, CodingKey {
        case outputPath = "output_path"
        case filesGenerated = "files_generated"
        case error
    }
}

/// Type-erasing Decodable for nested JSON dicts
struct AnyCodable: Decodable, Sendable {
    let value: Any

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let intVal = try? container.decode(Int.self) { value = intVal }
        else if let doubleVal = try? container.decode(Double.self) { value = doubleVal }
        else if let stringVal = try? container.decode(String.self) { value = stringVal }
        else if let boolVal = try? container.decode(Bool.self) { value = boolVal }
        else if let arrayVal = try? container.decode([AnyCodable].self) { value = arrayVal.map { $0.value } }
        else if let dictVal = try? container.decode([String: AnyCodable].self) { value = dictVal.mapValues { $0.value } }
        else { value = NSNull() }
    }
}

// MARK: - FusionClient Migration Methods

extension FusionClient {
    func migrateAnalyze(hfId: String, mirror: Bool = false) async throws -> MigrationAnalysis {
        struct Req: Encodable { let hf_id: String; let mirror: Bool }
        let resp: AnalyzeResponse = try await post(AdminAPI.migrateAnalyze, body: Req(hf_id: hfId, mirror: mirror))
        return resp.analysis
    }

    func migrateDownload(hfId: String, hfToken: String = "", mirror: Bool = false) async throws -> MigrationTaskInfo {
        struct Req: Encodable { let hf_id: String; let hf_token: String; let mirror: Bool }
        let resp: DownloadResponse = try await post(AdminAPI.migrateDownload, body: Req(hf_id: hfId, hf_token: hfToken, mirror: mirror))
        return MigrationTaskInfo(migrationId: resp.migrationId, task: resp.task)
    }

    func migrateDownloadStatus(migrationId: String) async throws -> [String: AnyCodable]? {
        let resp: StatusResponse = try await get(AdminAPI.migrateDownloadStatus(migrationId))
        return resp.task
    }

    func migrateConvert(migrationId: String, quantBits: Int = 0, quantGroupSize: Int = 64) async throws -> ConvertResult {
        struct Req: Encodable { let migration_id: String; let quant_bits: Int; let quant_group_size: Int }
        let resp: ConvertResponse = try await post(AdminAPI.migrateConvert, body: Req(migration_id: migrationId, quant_bits: quantBits, quant_group_size: quantGroupSize))
        return resp.result
    }

    func migrateCodegen(migrationId: String) async throws -> CodegenResult {
        struct Req: Encodable { let migration_id: String }
        let resp: CodegenResponse = try await post(AdminAPI.migrateCodegen, body: Req(migration_id: migrationId))
        return resp.result
    }

    func migrateValidate(migrationId: String, prompt: String = "Hello, how are you?", maxTokens: Int = 32) async throws -> ValidationResult {
        struct Req: Encodable { let migration_id: String; let prompt: String; let max_tokens: Int }
        let resp: ValidateResponse = try await post(AdminAPI.migrateValidate, body: Req(migration_id: migrationId, prompt: prompt, max_tokens: maxTokens))
        return resp.result
    }

    func migrateRegister(migrationId: String, modelName: String = "") async throws -> RegisterResponse {
        struct Req: Encodable { let migration_id: String; let model_name: String }
        return try await post(AdminAPI.migrateRegister, body: Req(migration_id: migrationId, model_name: modelName))
    }

    func migrateList() async throws -> [MigrationMeta] {
        let resp: ListResponse = try await get(AdminAPI.migrateList)
        return resp.migrations
    }
}

// MARK: - Response Wrappers

private struct AnalyzeResponse: Decodable {
    let success: Bool
    let analysis: MigrationAnalysis
}

private struct DownloadResponse: Decodable {
    let success: Bool
    let migrationId: String
    let task: [String: AnyCodable]?

    enum CodingKeys: String, CodingKey {
        case success
        case migrationId = "migration_id"
        case task
    }
}

private struct StatusResponse: Decodable {
    let success: Bool
    let task: [String: AnyCodable]?
}

private struct ConvertResponse: Decodable {
    let success: Bool
    let result: ConvertResult
}

private struct CodegenResponse: Decodable {
    let success: Bool
    let result: CodegenResult
}

private struct ValidateResponse: Decodable {
    let success: Bool
    let result: ValidationResult
}

struct RegisterResponse: Decodable {
    let success: Bool
    let path: String?
    let modelName: String?

    enum CodingKeys: String, CodingKey {
        case success, path
        case modelName = "model_name"
    }
}

struct MigrationMeta: Decodable, Identifiable {
    var id: String { migrationId }
    let migrationId: String
    let hfId: String?

    enum CodingKeys: String, CodingKey {
        case migrationId = "migration_id"
        case hfId = "hf_id"
    }
}

private struct ListResponse: Decodable {
    let success: Bool
    let migrations: [MigrationMeta]
}
