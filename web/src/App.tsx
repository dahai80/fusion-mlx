import { useEffect, useState } from "react";
import { LogsPage } from "./pages/LogsPage";
import { MetricsPage } from "./pages/MetricsPage";
import { ModelsPage } from "./pages/ModelsPage";
import { embeddedParam, isEmbedded } from "./sdk";

// Hash router (no router dep). Standalone shows a bottom tab bar; embedded
// (?embedded=1 or iframe) the shell owns the chrome so the module hides its
// own navigation.
type Route = "models" | "metrics" | "logs";

const ROUTES: ReadonlySet<string> = new Set<Route>([
    "models",
    "metrics",
    "logs",
]);

function parseHash(): Route {
    const h = window.location.hash.replace(/^#\/?/, "");
    if (ROUTES.has(h)) return h as Route;
    return "models";
}

export function App() {
    const [route, setRoute] = useState<Route>(parseHash());

    useEffect(() => {
        const onHash = () => setRoute(parseHash());
        window.addEventListener("hashchange", onHash);
        return () => window.removeEventListener("hashchange", onHash);
    }, []);

    const embedded = embeddedParam() || isEmbedded();

    return (
        <div className={embedded ? "h-full" : "flex h-full flex-col"}>
            <main className={`min-h-0 flex-1 ${embedded ? "" : "pb-16 pt-4"}`}>
                {route === "models" && <ModelsPage />}
                {route === "metrics" && <MetricsPage />}
                {route === "logs" && <LogsPage />}
            </main>
            {!embedded && <BottomNav route={route} onChange={setRoute} />}
        </div>
    );
}

function BottomNav({
    route,
    onChange,
}: { route: Route; onChange: (r: Route) => void }) {
    const items: Array<{ id: Route; label: string }> = [
        { id: "models", label: "Models" },
        { id: "metrics", label: "Metrics" },
        { id: "logs", label: "Logs" },
    ];
    return (
        <nav className="fixed inset-x-0 bottom-0 z-10 flex border-t border-border bg-bg-elevated">
            {items.map((it) => (
                <button
                    key={it.id}
                    type="button"
                    onClick={() => onChange(it.id)}
                    className={`flex-1 py-3 text-center text-sm ${
                        route === it.id ? "text-brand" : "text-text-secondary"
                    }`}
                >
                    {it.label}
                </button>
            ))}
        </nav>
    );
}
