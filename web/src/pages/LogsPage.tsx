import { type LogLine, LogStream, SkeletonLines } from "@fusion/web-ui";
import { useEffect, useMemo, useRef, useState } from "react";
import { api, parseLogLine } from "../api";
import { PageHeader } from "../components/PageHeader";

const MAX_LINES = 2000;

// Live log tail: initial /admin/api/logs snapshot, then SSE
// /admin/api/logs/stream. Stream drop (server restart / gateway envelope)
// shows a reconnect banner and retries after backoff — never a hung view.
// Inspector controls (#1278 C4): level / keyword / time filters narrow the
// buffered tail client-side; the stream itself stays untouched.
const RECONNECT_MS = 3000;

const LEVELS = ["all", "debug", "info", "warn", "error"] as const;
type LevelFilter = (typeof LEVELS)[number];

const LEVEL_RANK: Record<LogLine["level"], number> = {
    debug: 0,
    info: 1,
    warn: 2,
    error: 3,
};

export function LogsPage() {
    const [lines, setLines] = useState<LogLine[]>([]);
    const [streamError, setStreamError] = useState<string | null>(null);
    const [loaded, setLoaded] = useState(false);
    const [level, setLevel] = useState<LevelFilter>("all");
    const [keyword, setKeyword] = useState("");
    const [since, setSince] = useState(""); // "HH:MM" — show lines from then
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

    // Inspector filters (#1278): level rank, case-insensitive keyword
    // substring, and a "HH:MM" floor (today-local) for time jump.
    const filtered = useMemo(() => {
        const kw = keyword.trim().toLowerCase();
        let sinceTs: number | null = null;
        if (/^\d{2}:\d{2}$/.test(since)) {
            const [h, m] = since.split(":").map(Number);
            const d = new Date();
            d.setHours(h, m, 0, 0);
            sinceTs = d.getTime();
        }
        return lines.filter((l) => {
            if (level !== "all" && LEVEL_RANK[l.level] < LEVEL_RANK[level])
                return false;
            if (kw && !`${l.ts} ${l.level} ${l.msg}`.toLowerCase().includes(kw))
                return false;
            if (sinceTs !== null) {
                const t = Date.parse(String(l.ts).replace(" ", "T"));
                if (Number.isFinite(t) && t < sinceTs) return false;
            }
            return true;
        });
    }, [lines, level, keyword, since]);

    if (!loaded) return <SkeletonLines rows={10} />;

    return (
        <div className="mx-auto flex h-full max-w-5xl flex-col px-4">
            <PageHeader
                title="Logs"
                subtitle={`server.log live tail (SSE) · ${filtered.length}/${lines.length} shown`}
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
            <div className="mb-2 flex flex-wrap items-center gap-2">
                <select
                    value={level}
                    onChange={(e) => setLevel(e.target.value as LevelFilter)}
                    className="rounded-md border border-border bg-bg-elevated px-2 py-1 text-xs text-text-primary"
                    aria-label="Minimum log level"
                >
                    {LEVELS.map((lv) => (
                        <option key={lv} value={lv}>
                            {lv === "all" ? "all levels" : `≥ ${lv}`}
                        </option>
                    ))}
                </select>
                <input
                    type="search"
                    value={keyword}
                    onChange={(e) => setKeyword(e.target.value)}
                    placeholder="keyword…"
                    className="w-44 rounded-md border border-border bg-bg-elevated px-2 py-1 text-xs text-text-primary placeholder:text-text-muted"
                    aria-label="Keyword filter"
                />
                <input
                    type="time"
                    value={since}
                    onChange={(e) => setSince(e.target.value)}
                    className="rounded-md border border-border bg-bg-elevated px-2 py-1 text-xs text-text-primary"
                    aria-label="Show lines from"
                />
                {level !== "all" || keyword || since ? (
                    <button
                        type="button"
                        onClick={() => {
                            setLevel("all");
                            setKeyword("");
                            setSince("");
                        }}
                        className="rounded-md border border-border px-2 py-1 text-xs text-text-secondary hover:bg-bg-muted"
                    >
                        clear
                    </button>
                ) : null}
            </div>
            <div className="min-h-0 flex-1 pb-4" style={{ height: "70vh" }}>
                <LogStream lines={filtered} maxLines={MAX_LINES} />
            </div>
        </div>
    );
}
