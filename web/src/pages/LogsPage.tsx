import { type LogLine, LogStream, SkeletonLines } from "@fusion/web-ui";
import { useEffect, useRef, useState } from "react";
import { api, parseLogLine } from "../api";
import { PageHeader } from "../components/PageHeader";

const MAX_LINES = 2000;

// Live log tail: initial /admin/api/logs snapshot, then SSE
// /admin/api/logs/stream. Stream drop (server restart / gateway envelope)
// shows a reconnect banner and retries after backoff — never a hung view.
const RECONNECT_MS = 3000;

export function LogsPage() {
    const [lines, setLines] = useState<LogLine[]>([]);
    const [streamError, setStreamError] = useState<string | null>(null);
    const [loaded, setLoaded] = useState(false);
    const linesRef = useRef<LogLine[]>([]);
    linesRef.current = lines;

    useEffect(() => {
        let alive = true;
        let controller: AbortController | null = null;
        let reconnectTimer = 0;

        const push = (raw: string) => {
            // SSE framing: blank separator lines, "data: <json>" payloads.
            const trimmed = raw.trim();
            if (!trimmed || trimmed.startsWith(":")) return;
            const payload = trimmed.startsWith("data:")
                ? trimmed.slice(5).trim()
                : trimmed;
            if (payload === "[DONE]") return;
            let lineJson: { line?: string; ts?: string | number } | null = null;
            try {
                lineJson = JSON.parse(payload) as {
                    line?: string;
                    ts?: string | number;
                };
            } catch {
                lineJson = null;
            }
            const rawLine = lineJson?.line ?? (lineJson ? null : payload);
            if (rawLine === null) return;
            const parsed = parseLogLine(rawLine);
            if (!parsed) return;
            setLines((prev) => {
                const out = [...prev, parsed];
                return out.length > MAX_LINES
                    ? out.slice(out.length - MAX_LINES)
                    : out;
            });
        };

        const connect = async () => {
            if (!alive) return;
            controller = new AbortController();
            try {
                if (linesRef.current.length === 0) {
                    const tail = await api.logsTail(200);
                    if (!alive) return;
                    const initial = tail.logs
                        .split("\n")
                        .map(parseLogLine)
                        .filter((l): l is NonNullable<typeof l> => l !== null);
                    setLines(initial.slice(-MAX_LINES));
                }
                setStreamError(null);
                setLoaded(true);
                await api.logStream(push, controller.signal);
                // Stream ended cleanly — server closed it; reconnect.
                if (alive) scheduleReconnect();
            } catch (e) {
                if (!alive || controller?.signal.aborted) return;
                setStreamError(e instanceof Error ? e.message : String(e));
                setLoaded(true);
                scheduleReconnect();
            }
        };

        const scheduleReconnect = () => {
            if (!alive) return;
            reconnectTimer = window.setTimeout(
                () => void connect(),
                RECONNECT_MS,
            );
        };

        void connect();
        return () => {
            alive = false;
            window.clearTimeout(reconnectTimer);
            controller?.abort();
        };
    }, []);

    if (!loaded) return <SkeletonLines rows={10} />;

    return (
        <div className="mx-auto flex h-full max-w-5xl flex-col px-4">
            <PageHeader
                title="Logs"
                subtitle="server.log live tail (SSE)"
                actions={
                    streamError ? (
                        <span className="text-xs text-warning">
                            reconnecting…
                        </span>
                    ) : (
                        <span className="text-xs text-success">live</span>
                    )
                }
            />
            <div className="min-h-0 flex-1 pb-4" style={{ height: "70vh" }}>
                <LogStream lines={lines} maxLines={MAX_LINES} />
            </div>
        </div>
    );
}
