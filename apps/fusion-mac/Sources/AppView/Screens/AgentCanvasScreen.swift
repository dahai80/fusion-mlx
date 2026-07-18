// Agent Canvas screen — embeds the Web-based graph editor in a native SwiftUI view.
//
// Uses WKWebView to load the admin canvas page at /admin/canvas, providing
// native macOS feel with the full drag-and-drop graph editor from the Web UI.
// The canvas communicates with the Python backend via HTTP (same as the admin
// panel), so no additional bridge code is needed.

import SwiftUI
import WebKit

struct AgentCanvasScreen: View {
    @Environment(AppServices.self) private var services
    @State private var vm = AgentCanvasScreenVM()

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            // Toolbar
            HStack(spacing: 8) {
                Image(systemName: "square.grid.3x3.topleft.filled")
                    .font(.system(size: 14, weight: .semibold))
                    .foregroundStyle(.secondary)
                Text(String(localized: "canvas.title",
                            defaultValue: "Agent Canvas",
                            comment: "Title of the Agent Canvas screen"))
                    .font(.fusionText(13, weight: .semibold))

                Spacer()

                Button {
                    vm.loadGraphs(client: services.client)
                } label: {
                    Image(systemName: "arrow.clockwise")
                }
                .buttonStyle(.fusion(.plain, size: .small))
                .help("Refresh")

                Button {
                    vm.saveCurrentGraph()
                } label: {
                    Text(String(localized: "canvas.save", defaultValue: "Save"))
                }
                .buttonStyle(.fusion(.primary, size: .small))

                Button {
                    vm.loadSelectedGraph()
                } label: {
                    Text(String(localized: "canvas.load", defaultValue: "Load"))
                }
                .buttonStyle(.fusion(.plain, size: .small))

                // Graph selector
                if !vm.graphs.isEmpty {
                    Picker("", selection: $vm.selectedGraphId) {
                        Text("-- Select Graph --").tag("")
                        ForEach(vm.graphs, id: \.id) { graph in
                            Text(graph.name ?? graph.id).tag(graph.id)
                        }
                    }
                    .pickerStyle(.menu)
                    .frame(width: 180)
                    .onChange(of: vm.selectedGraphId) { _, newId in
                        vm.onGraphSelected(id: newId, client: services.client)
                    }
                }
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 10)
            .background(.regularMaterial)

            Divider()

            // WebView canvas
            WebView(url: vm.canvasURL, isLoading: $vm.isLoading)
                .overlay {
                    if vm.isLoading {
                        ProgressView()
                            .scaleEffect(0.8)
                    }
                }
        }
        .onAppear {
            vm.loadGraphs(client: services.client)
        }
    }
}

// MARK: - WebView wrapper

struct WebView: NSViewRepresentable {
    let url: URL
    @Binding var isLoading: Bool

    func makeCoordinator() -> Coordinator {
        Coordinator(self)
    }

    func makeNSView(context: Context) -> WKWebView {
        let config = WKWebViewConfiguration()
        config.websiteDataStore = .nonPersistent()

        let webView = WKWebView(frame: .zero, configuration: config)
        webView.navigationDelegate = context.coordinator
        webView.setValue(false, forKey: "drawsBackground")

        let request = URLRequest(url: url, cachePolicy: .reloadIgnoringLocalCacheData)
        webView.load(request)
        return webView
    }

    func updateNSView(_ webView: WKWebView, context: Context) {
        if webView.url != url {
            let request = URLRequest(url: url, cachePolicy: .reloadIgnoringLocalCacheData)
            webView.load(request)
        }
    }

    class Coordinator: NSObject, WKNavigationDelegate {
        var parent: WebView

        init(_ parent: WebView) {
            self.parent = parent
        }

        func webView(_ webView: WKWebView, didStartProvisionalNavigation navigation: WKNavigation!) {
            parent.isLoading = true
        }

        func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) {
            parent.isLoading = false
        }

        func webView(_ webView: WKWebView, didFail navigation: WKNavigation!, withError error: Error) {
            parent.isLoading = false
        }
    }
}

// MARK: - Settings sidebar integration

extension AppSection {
    static let agentCanvas = AppSection.agentCanvasSection
}

private extension AppSection {
    static let agentCanvasSection = AppSection(rawValue: "agentCanvas")!
}

// MARK: - Preview

#Preview {
    AgentCanvasScreen()
        .environment(AppServices(config: AppConfig()))
        .frame(width: 900, height: 600)
}