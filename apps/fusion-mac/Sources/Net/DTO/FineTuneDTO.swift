// Fine-Tune DTOs for /admin/api/fine-tune/* endpoints.
// callers: FusionClient (API methods), FineTuneScreenVM (state), FineTuneScreen (rendering)
// API: POST/GET /admin/api/fine-tune/jobs, GET .../stream, GET/DELETE .../adapters, GET .../models
// User instruction: "开始做，注意设计方案需要有GUI的设计和落地方案，提交给macos app"

import Foundation

struct FineTuneJobDTO: Codable, Identifiable, Sendable {
    let job_id: String
    let model_id: String
    let dataset: String
    let config: FineTuneConfigDTO
    let status: String
    let progress: FineTuneProgressDTO
    let created_at: Double
    let started_at: Double?
    let finished_at: Double?
    let adapter_path: String
    let adapter_name: String
    let error: String

    var id: String { job_id }
}

struct FineTuneConfigDTO: Codable, Equatable, Sendable {
    var lora_layers: Int = 16
    var lora_rank: Int = 8
    var lora_alpha: Double = 16.0
    var lora_dropout: Double = 0.0
    var fine_tune_type: String = "lora"
    var optimizer: String = "adamw"
    var learning_rate: Double = 1e-5
    var batch_size: Int = 4
    var iters: Int = 100
    var val_batches: Int = 25
    var steps_per_report: Int = 10
    var steps_per_eval: Int = 200
    var steps_per_save: Int = 100
    var max_seq_length: Int = 2048
    var gradient_checkpointing: Bool = false
    var grad_accumulation_steps: Int = 1
    var seed: Int = 0
    var mask_prompt: Bool = false
}

struct FineTuneProgressDTO: Codable, Equatable, Sendable {
    var step: Int = 0
    var total_steps: Int = 0
    var train_loss: Double = 0.0
    var val_loss: Double?
    var learning_rate: Double = 0.0
    var tokens_per_second: Double = 0.0
    var iterations_per_second: Double = 0.0
    var trained_tokens: Int = 0
    var peak_memory_gb: Double = 0.0
    var elapsed_seconds: Double = 0.0
    var eta_seconds: Double = 0.0
}

struct FineTuneAdapterDTO: Codable, Identifiable, Sendable {
    let model_id: String
    let adapter_name: String
    let adapter_path: String
    let has_weights: Bool
    let has_config: Bool
    let lora_layers: Int?
    let lora_rank: Int?
    let fine_tune_type: String?

    var id: String { "\(model_id)/\(adapter_name)" }
}

struct FineTuneModelDTO: Codable, Identifiable, Sendable {
    let model_id: String
    let model_type: String?
    let model_path: String
    let loaded: Bool

    var id: String { model_id }
}

struct CreateFineTuneJobRequest: Encodable, Sendable {
    let model_id: String
    let dataset: String
    let adapter_name: String?
    let config: FineTuneConfigDTO?
}

struct DeleteAdapterRequest: Encodable, Sendable {
    let model_id: String
    let adapter_name: String
}
