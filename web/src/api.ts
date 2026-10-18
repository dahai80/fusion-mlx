import { DEV_API_KEY, isEmbedded } from "./sdk";

// Typed client over the fusion-mlx engine surface (direct :11434 behind the
// Vite dev proxy, gateway /shell/fusion-mlx/ proxy when embedded — both
// same-origin). All engine endpoints are management-key gated:
// - Embedded: the gateway /shell proxy injects the module key upstream, so a
//   plain same-origin fetch carries no secret in this bundle (spec: no auth
//   logic in UI).
// - Standalone dev: VITE_API_KEY (dev convenience only) attaches
//   Authorization: Bearer — the engine auth middleware accepts Bearer only
//   (x-api-key is rejected with 401).
// Raw fetch (not the sdk bridge) so text bodies (/metrics) and SSE streams
// work in both modes.

export interface ModelEntry {
    id: string;
    model_path?: string;
    loaded: boolean;
    is_loading?: boolean;
    estimated_size_formatted?: string | null;
    actual_size_formatted?: string | null;
    pinned?: boolean;
    in_use?: number;
    ttl_seconds?: number;
    engine_type?: string;
    model_type?: string;
    config_model_type?: string;
    is_default?: boolean;
}

export interface HealthReport {
    status: string;
    version?: string;
    uptime_seconds?: number;
    active_models?: string[];
    memory?: {
        rss_bytes?: number;
        used_bytes?: number;
        free_bytes?: number;
        total_bytes?: number;
        mlx_active_bytes?: number | null;
        mlx_cache_bytes?: number | null;
        mlx_peak_bytes?: number | null;
        per_model?: Array<{ name: string; bytes: number }>;
    };
    oom_risk?: string;
}

export interface LogsTail {
    logs: string;
    total_lines: number;
    log_file: string;
    available_files: string[];
}

// Served under the gateway shell path (/shell/fusion-mlx/) requests must
// carry the module prefix so they land on the gateway /shell proxy; served
// standalone the prefix is empty. Derived from the document pathname — one
// build serves both topologies.
export function apiBase(): string {
    if (typeof window === "undefined") return "";
    const m = window.location.pathname.match(/^(\/shell\/[^/]+)/);
    return m ? m[1] : "";
}

function authHeaders(): Record<string, string> {
    if (!isEmbedded() && DEV_API_KEY)
        return { Authorization: `Bearer ${DEV_API_KEY}` };
    return {};
}

async function request(path: string, init?: RequestInit): Promise<Response> {
    const res = await fetch(`${apiBase()}${path}`, {
        ...init,
        headers: { ...authHeaders(), ...(init?.headers ?? {}) },
    });
    if (!res.ok) throw new Error(`request failed: ${res.status} ${path}`);
    return res;
}

async function getJson<T>(path: string): Promise<T> {
    const res = await request(path);
    return (await res.json()) as T;
}

export const api = {
    models: () => getJson<{ models: ModelEntry[] }>("/admin/api/models"),
    load: (id: string) =>
        request(`/admin/api/models/${encodeURIComponent(id)}/load`, {
            method: "POST",
        }),
    unload: (id: string) =>
        request(`/admin/api/models/${encodeURIComponent(id)}/unload`, {
            method: "POST",
        }),
    health: () => getJson<HealthReport>("/v1/health"),
    logsTail: (lines = 200) =>
        getJson<LogsTail>(`/admin/api/logs?lines=${lines}`),
    metricsText: async () => {
        const res = await request("/metrics");
        return res.text();
    },
    // SSE log stream via fetch (ReadableStream) so auth headers work in
    // standalone mode; EventSource cannot set headers.
    logStream: (onLine: (raw: string) => void, signal: AbortSignal) =>
        request("/admin/api/logs/stream", { signal }).then(async (res) => {
            const reader = res.body?.getReader();
            if (!reader) return;
            const decoder = new TextDecoder();
            let buf = "";
            for (;;) {
                const { done, value } = await reader.read();
                if (done) break;
                buf += decoder.decode(value, { stream: true });
                const parts = buf.split("\n");
                buf = parts.pop() ?? "";
                for (const part of parts) onLine(part);
            }
        }),
};

// --- Prometheus text parsing (/metrics is text/plain exposition format) ---

export interface MetricSample {
    requests_total: number;
    requests_failed_total: number;
    requests_running: number;
    requests_waiting: number;
    prompt_tokens_total: number;
    completion_tokens_total: number;
    models_loaded: number;
    models_discovered: number;
    uptime_seconds: number;
}

const METRIC_NAMES = [
    "fusion_mlx_requests_total",
    "fusion_mlx_requests_failed_total",
    "fusion_mlx_requests_running",
    "fusion_mlx_requests_waiting",
    "fusion_mlx_prompt_tokens_total",
    "fusion_mlx_completion_tokens_total",
    "fusion_mlx_models_loaded",
    "fusion_mlx_models_discovered",
    "fusion_mlx_uptime_seconds",
] as const;

const SHORT: Record<string, keyof MetricSample> = {
    fusion_mlx_requests_total: "requests_total",
    fusion_mlx_requests_failed_total: "requests_failed_total",
    fusion_mlx_requests_running: "requests_running",
    fusion_mlx_requests_waiting: "requests_waiting",
    fusion_mlx_prompt_tokens_total: "prompt_tokens_total",
    fusion_mlx_completion_tokens_total: "completion_tokens_total",
    fusion_mlx_models_loaded: "models_loaded",
    fusion_mlx_models_discovered: "models_discovered",
    fusion_mlx_uptime_seconds: "uptime_seconds",
};

export function parsePrometheus(text: string): Partial<MetricSample> {
    const out: Partial<MetricSample> = {};
    for (const raw of text.split("\n")) {
        const line = raw.trim();
        if (!line || line.startsWith("#")) continue;
        // "name{labels} value" or "name value"
        const m = line.match(/^([a-z_]+)(?:\{[^}]*\})?\s+([-\d.eE+]+)$/);
        if (!m) continue;
        const name = m[1] as (typeof METRIC_NAMES)[number];
        if (!(METRIC_NAMES as readonly string[]).includes(name)) continue;
        const v = Number(m[2]);
        if (Number.isFinite(v)) out[SHORT[name]] = v;
    }
    return out;
}

// Log line → LogLine. Engine format:
// "2026-10-17 12:00:00,123 - fusion_mlx.server - INFO - [reqid] - message"
// (request-id segment optional)
const LOG_RE =
    /^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}[,.]\d*)\s+-\s+([\w.]+)\s+-\s+(\w+)\s+-\s+(?:\[[^\]]*\]\s+-\s+)?(.*)$/;

export interface ParsedLogLine {
    ts: number | string;
    level: "debug" | "info" | "warn" | "error";
    msg: string;
}

export function parseLogLine(raw: string): ParsedLogLine | null {
    const line = raw.trimEnd();
    if (!line) return null;
    const m = line.match(LOG_RE);
    if (!m) return { ts: "", level: "info", msg: line };
    const lvl = m[3].toLowerCase();
    const level: ParsedLogLine["level"] =
        lvl === "warning" || lvl === "warn"
            ? "warn"
            : lvl === "error"
              ? "error"
              : lvl === "debug"
                ? "debug"
                : "info";
    return { ts: m[1], level, msg: `${m[2]}: ${m[4]}` };
}
