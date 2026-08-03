import Foundation

@MainActor
final class StatsStore {
    static let shared = StatsStore()

    private let defaults = UserDefaults.standard
    private static let cacheKey = "fusion_mlx_alltime_stats_cache"

    struct AlltimeStats: Codable, Sendable {
        var totalRequests: Int?
        var totalPromptTokens: Int?
        var totalCompletionTokens: Int?
        var totalCachedTokens: Int?
        var cacheEfficiency: Double?

        enum CodingKeys: String, CodingKey {
            case totalRequests = "total_requests"
            case totalPromptTokens = "total_prompt_tokens"
            case totalCompletionTokens = "total_completion_tokens"
            case totalCachedTokens = "total_cached_tokens"
            case cacheEfficiency = "cache_efficiency"
        }
    }

    private(set) var cached: AlltimeStats?

    private init() {
        loadFromCache()
    }

    func update(_ stats: AlltimeStats) {
        cached = stats
        saveToCache(stats)
    }

    private func loadFromCache() {
        if let data = defaults.data(forKey: Self.cacheKey),
           let decoded = try? JSONDecoder().decode(AlltimeStats.self, from: data) {
            cached = decoded
        }
    }

    private func saveToCache(_ stats: AlltimeStats) {
        if let data = try? JSONEncoder().encode(stats) {
            defaults.set(data, forKey: Self.cacheKey)
        }
    }
}
