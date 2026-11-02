import {
    ErrorState,
    MetricChart,
    type MetricSeries,
    MetricTile,
    SkeletonLines,
    StatusBadge,
    formatBytes,
    formatTokenCount,
} from "@fusion/web-ui";
import { useEffect, useRef, useState } from "react";
import {
    type HealthReport,
    type MetricSample,
    api,
    parsePrometheus,
} from "../api";
import { PageHeader } from "../components/PageHeader";
import {
    type UsageSample,
    WINDOWS,
    appendSample,
    loadSamples,
    windowSeries,
} from "../components/usageStore";
import { useT } from "../i18n";

const POLL_MS = 3000;
const RING_MAX = 4000; // matches usageStore persistence cap

// QPS / capacity / retrospection surface: polls /metrics (Prometheus) and
// /v1/health (memory JSON) at POLL_MS. Capacity gauge (#1278) from health
// memory + /metrics gauges; usage windows (#1278) aggregate the local sample
// ring persisted to localStorage (components/usageStore.ts) — no backend.
export function MetricsPage() {
    const t = useT();
    const [health, setHealth] = useState<HealthReport | null>(null);
    const [last, setLast] = useState<Partial<MetricSample> | null>(null);
    const [samples, setSamples] = useState<UsageSample[]>([]);
    const [winKey, setWinKey] = useState("5m");
    const [error, setError] = useState<string | null>(null);
    const [loaded, setLoaded] = useState(false);
    const pollRef = useRef<() => void>(() => undefined);
    const samplesRef = useRef<UsageSample[]>([]);
    samplesRef.current = samples;

    useEffect(() => {
        setSamples(loadSamples());
        let alive = true;
        const poll = async () => {
            try {
                const [h, text] = await Promise.all([
                    api.health(),
                    api.metricsText(),
                ]);
                const m = parsePrometheus(text);
                if (!alive) return;
                setHealth(h);
                setLast(m);
                setError(null);
                const now = Date.now();
                const prev = samplesRef.current[samplesRef.current.length - 1];
                // Counter went backwards: server restarted, so the cumulative
                // TTFT histogram was reset. Carrying the previous p95 would
                // plot a pre-restart value against fresh counters.
                const reset =
                    prev !== undefined &&
                    (m.requests_total ?? 0) < prev.requests;
                const next: UsageSample = {
                    t: now,
                    requests: m.requests_total ?? prev?.requests ?? 0,
                    failed: m.requests_failed_total ?? prev?.failed ?? 0,
                    completion:
                        m.completion_tokens_total ?? prev?.completion ?? 0,
                    prompt: m.prompt_tokens_total ?? prev?.prompt ?? 0,
                    tpsSum: m.tps_sum ?? prev?.tpsSum,
                    tpsCount: m.tps_count ?? prev?.tpsCount,
                    p95: m.ttft_p95_seconds ?? (reset ? undefined : prev?.p95),
                };
                setSamples((prevSamples) => {
                    const out = [...prevSamples, next];
                    return out.length > RING_MAX
                        ? out.slice(out.length - RING_MAX)
                        : out;
                });
                appendSample(next);
                setLoaded(true);
            } catch (e) {
                if (!alive) return;
                setError(e instanceof Error ? e.message : String(e));
                setLoaded(true);
            }
        };
        void poll();
        pollRef.current = poll;
        const timer = window.setInterval(poll, POLL_MS);
        return () => {
            alive = false;
            window.clearInterval(timer);
        };
    }, []);

    if (!loaded) return <SkeletonLines rows={8} />;
    if (error && !health)
        return (
            <div className="mx-auto max-w-5xl px-4">
                <PageHeader title={t("指标", "Metrics")} />
                <ErrorState
                    message={error}
                    onRetry={() => {
                        setError(null);
                        pollRef.current();
                    }}
                />
            </div>
        );

    const oomRiskLabels: Record<string, string> = {
        none: t("无", "none"),
        imminent: t("迫近", "imminent"),
        high: t("高", "high"),
        moderate: t("中", "moderate"),
        low: t("低", "low"),
    };
    const mem = health?.memory;
    const perModel = mem?.per_model ?? [];
    const qpsSeries: MetricSeries[] = [
        {
            label: t("请求/秒", "requests/s"),
            points: derivePoints(samples, (s) => s.requests),
        },
    ];
    const tokSeries: MetricSeries[] = [
        {
            label: t("生成 tokens/秒", "completion tokens/s"),
            points: derivePoints(samples, (s) => s.completion),
        },
    ];
    const win = WINDOWS.find((w) => w.key === winKey) ?? WINDOWS[0];
    const winRes = windowSeries(samples, win.ms, Date.now());

    return (
        <div className="mx-auto max-w-5xl px-4">
            <PageHeader
                title={t("指标", "Metrics")}
                subtitle={
                    health
                        ? `v${health.version ?? "?"} · ${t("已运行", "up")} ${Math.floor((health.uptime_seconds ?? 0) / 60)}m`
                        : undefined
                }
                actions={
                    health?.oom_risk ? (
                        <StatusBadge
                            status={
                                health.oom_risk === "imminent"
                                    ? "error"
                                    : health.oom_risk === "high"
                                      ? "warning"
                                      : "success"
                            }
                            label={`OOM: ${oomRiskLabels[health.oom_risk] ?? health.oom_risk}`}
                        />
                    ) : undefined
                }
            />
            {error ? (
                <div className="mb-2 text-xs text-warning">
                    {error} — {t("重试中", "retrying")}
                </div>
            ) : null}
            <CapacityGauge mem={mem} last={last} perModel={perModel} />
            <div className="mt-4 grid grid-cols-2 gap-3 md:grid-cols-4">
                <MetricTile
                    label={t("请求数", "Requests")}
                    value={fmt(last?.requests_total)}
                />
                <MetricTile
                    label={t("失败数", "Failed")}
                    value={fmt(last?.requests_failed_total)}
                />
                <MetricTile
                    label={t("运行中", "Running")}
                    value={fmt(last?.requests_running)}
                />
                <MetricTile
                    label={t("排队中", "Waiting")}
                    value={fmt(last?.requests_waiting)}
                />
                <MetricTile
                    label={t("输入 tokens", "Prompt tokens")}
                    value={formatTokenCount(
                        last?.prompt_tokens_total ?? Number.NaN,
                    )}
                />
                <MetricTile
                    label={t("输出 tokens", "Completion tokens")}
                    value={formatTokenCount(
                        last?.completion_tokens_total ?? Number.NaN,
                    )}
                />
                <MetricTile
                    label={t("已加载模型", "Models loaded")}
                    value={fmt(last?.models_loaded)}
                />
                <MetricTile
                    label={t("进程内存 (RSS)", "RSS")}
                    value={formatBytes(mem?.rss_bytes ?? Number.NaN)}
                />
            </div>
            <UsageWindow winKey={winKey} onWin={setWinKey} res={winRes} />
            <div className="mt-4 grid gap-3 md:grid-cols-2">
                <MetricChart series={qpsSeries} unit="req/s" />
                <MetricChart series={tokSeries} unit="tok/s" />
            </div>
            <div className="mt-4 grid grid-cols-1 gap-3 md:grid-cols-3">
                <MetricTile
                    label={t("MLX 活跃", "MLX active")}
                    value={formatBytes(mem?.mlx_active_bytes ?? Number.NaN)}
                />
                <MetricTile
                    label={t("MLX 缓存", "MLX cache")}
                    value={formatBytes(mem?.mlx_cache_bytes ?? Number.NaN)}
                />
                <MetricTile
                    label={t("MLX 峰值", "MLX peak")}
                    value={formatBytes(mem?.mlx_peak_bytes ?? Number.NaN)}
                />
            </div>
            <div className="mt-4 rounded-card border border-border bg-bg-elevated p-4">
                <div className="text-sm text-text-primary">
                    {t("各模型内存", "Per-model memory")}
                </div>
                {perModel.length === 0 ? (
                    <div className="mt-2 text-xs text-text-muted">
                        {t("暂无常驻模型", "No resident models")}
                    </div>
                ) : (
                    <ul className="mt-2 space-y-1">
                        {perModel.map((p) => (
                            <li
                                key={p.name}
                                className="flex items-center justify-between font-mono text-xs tabular-nums text-text-secondary"
                            >
                                <span className="truncate">{p.name}</span>
                                <span className="ml-2 flex shrink-0 gap-4">
                                    <span className="text-text-muted">
                                        {formatDuration(p.resident_seconds)}
                                    </span>
                                    <span>{formatBytes(p.bytes)}</span>
                                </span>
                            </li>
                        ))}
                    </ul>
                )}
            </div>
        </div>
    );
}

// Capacity gauge (#1278 G8/C5): unified-memory pressure bar + engine
// capacity tiles. Numbers read straight from /v1/health memory and /metrics
// gauges so the UI matches the prometheus surface exactly.
function CapacityGauge({
    mem,
    last,
    perModel,
}: {
    mem?: HealthReport["memory"];
    last: Partial<MetricSample> | null;
    perModel: Array<{
        name: string;
        bytes: number;
        resident_seconds?: number | null;
    }>;
}) {
    const usedB = mem?.used_bytes ?? 0;
    const totalB = mem?.total_bytes ?? 0;
    const t = useT();
    const pct = totalB > 0 ? Math.min(100, (usedB / totalB) * 100) : null;
    const residentBytes = perModel.reduce((acc, p) => acc + p.bytes, 0);
    const barColor =
        pct === null
            ? "bg-bg-muted"
            : pct < 70
              ? "bg-success"
              : pct < 85
                ? "bg-warning"
                : "bg-error";
    return (
        <div className="mt-4 rounded-card border border-border bg-bg-elevated p-4">
            <div className="flex items-center justify-between text-sm">
                <span className="text-text-primary">
                    {t("内存压力", "Memory pressure")}
                </span>
                <span className="font-mono text-xs tabular-nums text-text-secondary">
                    {pct === null ? "-" : `${pct.toFixed(0)}%`} ·{" "}
                    {formatBytes(usedB)} / {formatBytes(totalB)}
                </span>
            </div>
            <div className="mt-2 h-2 overflow-hidden rounded-full bg-bg-muted">
                <div
                    className={`h-full rounded-full ${barColor}`}
                    style={{ width: `${pct ?? 0}%` }}
                />
            </div>
            <div className="mt-3 grid grid-cols-2 gap-3 md:grid-cols-3">
                <MetricTile
                    label={t("KV 利用率", "KV utilization")}
                    value={pctGauge(last?.paged_kv_utilization)}
                />
                <MetricTile
                    label={t("Metal 活跃", "Metal active")}
                    value={formatBytes(last?.metal_active_bytes ?? Number.NaN)}
                />
                <MetricTile
                    label={t("Metal 接线", "Metal wired")}
                    value={formatBytes(last?.metal_wired_bytes ?? Number.NaN)}
                />
                <MetricTile
                    label={t("物理占用", "Phys footprint")}
                    value={formatBytes(
                        last?.phys_footprint_bytes ?? Number.NaN,
                    )}
                />
                <MetricTile
                    label={t("常驻模型", "Resident models")}
                    value={`${perModel.length} · ${formatBytes(residentBytes)}`}
                />
                <MetricTile
                    label={t("缓存命中 / 投机接受", "Cache hit / spec accept")}
                    value={`${pctGauge(last?.response_cache_hit_rate)} / ${pctGauge(last?.spec_decode_accept_rate)}`}
                />
                <MetricTile
                    label={t("可用内存", "Available")}
                    value={formatBytes(mem?.free_bytes ?? Number.NaN)}
                />
                <MetricTile
                    label={t("已用 Swap", "Swap used")}
                    value={`${formatBytes(last?.swap_used_bytes ?? Number.NaN)} / ${formatBytes(last?.swap_total_bytes ?? Number.NaN)}`}
                />
                <MetricTile
                    label={t("Swap 速率", "Swap rate")}
                    value={formatRate(last?.swap_rate_bytes_per_second)}
                />
            </div>
        </div>
    );
}

// Usage retrospection card (#1278 G8): window totals + per-minute request
// chart. Data = local sample ring (localStorage), so 2h/24h windows fill in
// as the page stays open / across reloads; no backend aggregation.
function UsageWindow({
    winKey,
    onWin,
    res,
}: {
    winKey: string;
    onWin: (k: string) => void;
    res: {
        points: Array<{ t: number; v: number }>;
        latencyPoints: Array<{ t: number; v: number }>;
        totals: {
            req: number;
            failed: number;
            tok: number;
            tps: number | null;
            p95: number | null;
        };
    };
}) {
    const [view, setView] = useState<"req" | "p95">("req");
    const t = useT();
    return (
        <div className="mt-4 rounded-card border border-border bg-bg-elevated p-4">
            <div className="flex items-center justify-between">
                <span className="text-sm text-text-primary">
                    {t("用量回溯", "Usage retrospection")}
                </span>
                <div className="flex gap-1">
                    <button
                        type="button"
                        onClick={() => setView("req")}
                        className={`rounded-md px-2 py-1 text-xs ${
                            view === "req"
                                ? "bg-brand text-white"
                                : "border border-border text-text-secondary"
                        }`}
                    >
                        {t("请求", "Requests")}
                    </button>
                    <button
                        type="button"
                        onClick={() => setView("p95")}
                        className={`rounded-md px-2 py-1 text-xs ${
                            view === "p95"
                                ? "bg-brand text-white"
                                : "border border-border text-text-secondary"
                        }`}
                    >
                        {t("p95 首 token", "p95 TTFT")}
                    </button>
                    <span className="mx-1 w-px bg-border" />
                    {WINDOWS.map((w) => (
                        <button
                            key={w.key}
                            type="button"
                            onClick={() => onWin(w.key)}
                            className={`rounded-md px-2 py-1 text-xs ${
                                winKey === w.key
                                    ? "bg-brand text-white"
                                    : "border border-border text-text-secondary"
                            }`}
                        >
                            {w.label}
                        </button>
                    ))}
                </div>
            </div>
            <div className="mt-2 grid grid-cols-2 gap-3 md:grid-cols-5">
                <MetricTile
                    label={t("请求数", "Requests")}
                    value={String(res.totals.req)}
                />
                <MetricTile
                    label={t("失败数", "Failed")}
                    value={`${res.totals.failed} (${pctGauge(res.totals.req > 0 ? res.totals.failed / res.totals.req : null)})`}
                />
                <MetricTile
                    label={t("Token 数", "Tokens")}
                    value={formatTokenCount(res.totals.tok)}
                />
                <MetricTile
                    label={t("平均 TPS", "Avg TPS")}
                    value={
                        res.totals.tps === null
                            ? "-"
                            : res.totals.tps.toFixed(1)
                    }
                />
                <MetricTile
                    label={t(
                        "p95 首 token（自启动）",
                        "p95 TTFT (since start)",
                    )}
                    value={
                        res.totals.p95 === null
                            ? "-"
                            : `${res.totals.p95.toFixed(2)}s`
                    }
                />
            </div>
            <div className="mt-3">
                <MetricChart
                    series={
                        view === "req"
                            ? [
                                  {
                                      label: t("请求/分钟", "requests/min"),
                                      points: res.points,
                                  },
                              ]
                            : [
                                  {
                                      label: t(
                                          "p95 首 token（累计）",
                                          "p95 TTFT (cumulative)",
                                      ),
                                      points: res.latencyPoints,
                                  },
                              ]
                    }
                    unit={view === "req" ? "req/min" : "s"}
                />
            </div>
        </div>
    );
}

// Gauge/percent formatter: metrics expose 0..1 ratios (kv utilization,
// hit rates); null/NaN prints "-".
function pctGauge(v: number | undefined | null): string {
    if (!Number.isFinite(v ?? Number.NaN)) return "-";
    return `${((v as number) * 100).toFixed(1)}%`;
}

interface Point {
    t: number;
    v: number;
}

// Counter → per-second rate series: v[i] = (c[i] - c[i-1]) / dt.
function derivePoints(
    samples: UsageSample[],
    pick: (s: UsageSample) => number,
): Array<Point> {
    const out: Array<Point> = [];
    for (let i = 1; i < samples.length; i++) {
        const dt = (samples[i].t - samples[i - 1].t) / 1000;
        if (dt <= 0) continue;
        out.push({
            t: samples[i].t,
            v: Math.max(0, (pick(samples[i]) - pick(samples[i - 1])) / dt),
        });
    }
    return out;
}

function fmt(v: number | undefined): string {
    return Number.isFinite(v) ? String(v) : "-";
}

// #1296: residency duration from /v1/health per_model.resident_seconds;
// null means the engine load time is unknown (loaded before tracking).
function formatDuration(seconds?: number | null): string {
    if (seconds === null || seconds === undefined || !Number.isFinite(seconds))
        return "-";
    const s = Math.max(0, Math.floor(seconds));
    if (s < 60) return `${s}s`;
    if (s < 3600) return `${Math.floor(s / 60)}m`;
    if (s < 86400)
        return `${Math.floor(s / 3600)}h ${Math.floor((s % 3600) / 60)}m`;
    return `${Math.floor(s / 86400)}d ${Math.floor((s % 86400) / 3600)}h`;
}

// #1296: swap rate gauge is signed net page-in minus page-out (negative =
// actively paging out under memory pressure); null when the backend has no
// second sample yet or psutil is unavailable.
function formatRate(v?: number | null): string {
    if (v === null || v === undefined || !Number.isFinite(v)) return "-";
    return `${v < 0 ? "-" : ""}${formatBytes(Math.abs(v))}/s`;
}
