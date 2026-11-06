// WelcomeHardwareDTO — hardware detection + model recommendation payloads
// for the first-run wizard's hardware-detect / recommend steps.
// Wire format matches Resources/whichllm_bridge.py (_format_hardware and
// the recommendation rows it prints to stdout).
// callers: WelcomeViewModel (WelcomeWindow.swift); Tests: WelcomeViewModelTests

import Foundation

struct GPUInfoDTO: Codable, Equatable, Sendable {
    let name: String
    let vendor: String
    let vramBytes: Int
    let usableVramBytes: Int?
    let memoryBandwidthGbps: Double?
    let sharedMemory: Bool?

    enum CodingKeys: String, CodingKey {
        case name, vendor
        case vramBytes = "vram_bytes"
        case usableVramBytes = "usable_vram_bytes"
        case memoryBandwidthGbps = "memory_bandwidth_gbps"
        case sharedMemory
    }
}

struct HardwareInfoDTO: Codable, Equatable, Sendable {
    let gpus: [GPUInfoDTO]
    let cpu: String
    let cpuCores: Int
    let ramBytes: Int
    let ramBudgetBytes: Int?
    let budgetNotes: String?
    let diskFreeBytes: Int?
    let os: String

    enum CodingKeys: String, CodingKey {
        case gpus, cpu
        case cpuCores = "cpu_cores"
        case ramBytes = "ram_bytes"
        case ramBudgetBytes = "ram_budget_bytes"
        case budgetNotes = "budget_notes"
        case diskFreeBytes = "disk_free_bytes"
        case os
    }

    /// Unified-memory GiB (binary), the number the UI quotes.
    var ramGB: Double { Double(ramBytes) / 1_073_741_824.0 }

    /// ProcessInfo-based fallback used when the whichllm bridge is
    /// unavailable (dev builds, offline). Apple Silicon: one Apple GPU
    /// sharing system RAM.
    static func quickDetect() -> HardwareInfoDTO {
        let info = ProcessInfo.processInfo
        let ramBytes = Int(info.physicalMemory)
        var chip = "Unknown"
        var names: [String] = []
        let key = "machdep.cpu.brand_string"
        var size = 0
        sysctlbyname(key, nil, &size, nil, 0)
        if size > 0 {
            var buf = [CChar](repeating: 0, count: size)
            sysctlbyname(key, &buf, &size, nil, 0)
            chip = String(cString: buf)
            names.append(chip)
        }
        var diskFree: Int?
        let home = FileManager.default.homeDirectoryForCurrentUser
        let values = try? home.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey])
        diskFree = values.flatMap { $0.volumeAvailableCapacityForImportantUsage.map(Int.init) }
        let gpus = names.isEmpty
            ? []
            : [GPUInfoDTO(
                name: names[0], vendor: "apple",
                vramBytes: ramBytes, usableVramBytes: nil,
                memoryBandwidthGbps: nil, sharedMemory: true
            )]
        return HardwareInfoDTO(
            gpus: gpus, cpu: chip, cpuCores: info.processorCount,
            ramBytes: ramBytes, ramBudgetBytes: nil, budgetNotes: nil,
            diskFreeBytes: diskFree, os: "darwin"
        )
    }
}

struct ModelRecommendationDTO: Codable, Identifiable, Equatable, Sendable {
    let rank: Int
    let modelId: String
    let modelName: String
    let artifactRepoId: String?
    let artifactFilename: String?
    let parameterCount: Int
    let parameterCountActive: Int?
    let architecture: String
    let contextLength: Int
    let quantType: String?
    let fileSizeBytes: Int
    let vramRequiredBytes: Int
    let vramAvailableBytes: Int?
    let estimatedTokPerSec: Double?
    let speedConfidence: String?
    let qualityScore: Double?
    let fitType: String
    let benchmarkStatus: String?
    let usesMultiGpu: Bool

    enum CodingKeys: String, CodingKey {
        case rank
        case modelId = "model_id"
        case modelName = "model_name"
        case artifactRepoId = "artifact_repo_id"
        case artifactFilename = "artifact_filename"
        case parameterCount = "parameter_count"
        case parameterCountActive = "parameter_count_active"
        case architecture
        case contextLength = "context_length"
        case quantType = "quant_type"
        case fileSizeBytes = "file_size_bytes"
        case vramRequiredBytes = "vram_required_bytes"
        case vramAvailableBytes = "vram_available_bytes"
        case estimatedTokPerSec = "estimated_tok_per_sec"
        case speedConfidence = "speed_confidence"
        case qualityScore = "quality_score"
        case fitType = "fit_type"
        case benchmarkStatus = "benchmark_status"
        case usesMultiGpu = "uses_multi_gpu"
    }

    var id: String { modelId }
    /// Billions of parameters (decimal), e.g. 7.0 for a 7B model.
    var paramsB: Double { Double(parameterCount) / 1_000_000_000.0 }
    /// File size in binary GiB, matching what the Finder shows.
    var fileSizeGB: Double { Double(fileSizeBytes) / 1_073_741_824.0 }
    /// Human label for how the model fits the detected hardware.
    var fitDescription: String {
        switch fitType {
        case "full_gpu": return "Full GPU"
        case "partial_gpu", "partial": return "Partial GPU"
        case "multi_gpu": return "Multi GPU"
        case "cpu", "cpu_only": return "CPU"
        default: return fitType.replacingOccurrences(of: "_", with: " ").capitalized
        }
    }
}
