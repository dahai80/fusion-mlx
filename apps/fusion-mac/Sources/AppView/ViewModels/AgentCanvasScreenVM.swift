// ViewModel for the Agent Canvas screen.
//
// Manages canvas state, graph list, and communication with the
// canvases_admin API endpoints.

import Foundation

@MainActor
@Observable
final class AgentCanvasScreenVM {
    var graphs: [CanvasGraphInfo] = []
    var selectedGraphId: String = ""
    var isLoading = false
    var lastError: String?

    /// The URL of the canvas page, built from the fusion-mlx server address.
    var canvasURL: URL {
        // Default to localhost:11435/admin/canvas
        URL(string: "http://127.0.0.1:11435/admin/canvas")!
    }

    /// Load the list of saved graphs from the admin API.
    func loadGraphs(client: FusionClient) {
        Task {
            do {
                let data = try await client.get("/admin/api/canvas/graphs")
                let decoder = JSONDecoder()
                if let json = try? JSONSerialization.jsonObject(with: data) as? [[String: Any]] {
                    self.graphs = json.map { CanvasGraphInfo(from: $0) }
                }
            } catch {
                lastError = error.fusionDescription
            }
        }
    }

    /// Save the current graph to the backend.
    func saveCurrentGraph() {
        // The Web canvas handles save internally via the admin API.
        // This native button triggers the same action via JavaScript.
        // Currently a no-op since the WebView's canvas.js handles persistence.
    }

    /// Load a selected graph by ID.
    func loadSelectedGraph() {
        // The Web canvas handles load internally.
        // This native button navigates the WebView to the graph.
    }

    /// Called when the user picks a graph from the native picker.
    func onGraphSelected(id: String, client: FusionClient) {
        guard !id.isEmpty else { return }
        // The WebView will load the graph via the admin API
        // when the canvas.js detects the graph selector change.
    }

    /// Refresh the graph list.
    func refresh(client: FusionClient) {
        loadGraphs(client: client)
    }
}

// MARK: - Graph Info Model

struct CanvasGraphInfo: Identifiable {
    let id: String
    let name: String?
    let description: String?
    let nodeCount: Int
    let edgeCount: Int

    init(from dict: [String: Any]) {
        self.id = dict["id"] as? String ?? ""
        self.name = dict["name"] as? String
        self.description = dict["description"] as? String
        self.nodeCount = dict["node_count"] as? Int ?? 0
        self.edgeCount = dict["edge_count"] as? Int ?? 0
    }
}