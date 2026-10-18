import {
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

interface Sample {
    t: number;
    requests: number;
    completion: number;
    prompt: number;
}

const POLL_MS = 3000;
const MAX_SAMPLES = 240; // 12 min at 3s cadence

// QPS / latency / memory surface: polls /metrics (Prometheus counters) and
// /v1/health (memory JSON), derives per-second rates from counter deltas.
export function MetricsPage() {
    const [health, setHealth] = useState<HealthReport | null>(null);
    const [last, setLast] = useState<Partial<MetricSample> | null>(null);
    const [samples, setSamples] = useState<Sample[]>([]);
    const [error, setError] = useState<string | null>(null);
    const [loaded, setLoaded] = useState(false);
    const samplesRef = useRef<Sample[]>([]);
    samplesRef.current = samples;

    useEffect(() => {
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
                setSamples((prevSamples) => {
                    const out = [
                        ...prevSamples,
                        {
                            t: now,
                            requests: m.requests_total ?? prev?.requests ?? 0,
                            completion:
                                m.completion_tokens_total ??
                                prev?.completion ??
                                0,
                            prompt: m.prompt_tokens_total ?? prev?.prompt ?? 0,
                        },
                    ];
                    return out.length > MAX_SAMPLES
                        ? out.slice(out.length - MAX_SAMPLES)
                        : out;
                });
                setLoaded(true);
            } catch (e) {
                if (!alive) return;
                setError(e instanceof Error ? e.message : String(e));
                setLoaded(true);
            }
        };
        void poll();
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
                <PageHeader title="Metrics" />
                <div className="py-12 text-center text-sm text-error">
                    {error}
                </div>
            </div>
        );

    const mem = health?.memory;
    const perModel = mem?.per_model ?? [];
    const qpsSeries: MetricSeries[] = [
        {
            label: "requests/s",
            points: derivePoints(samples, (s) => s.requests),
        },
    ];
    const tokSeries: MetricSeries[] = [
        {
            label: "completion tokens/s",
            points: derivePoints(samples, (s) => s.completion),
        },
    ];

    return (
        <div className="mx-auto max-w-5xl px-4">
            <PageHeader
                title="Metrics"
                subtitle={
                    health
                        ? `v${health.version ?? "?"} · up ${Math.floor((health.uptime_seconds ?? 0) / 60)}m`
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
                            label={`OOM: ${health.oom_risk}`}
                        />
                    ) : undefined
                }
            />
            {error ? (
                <div className="mb-2 text-xs text-warning">
                    {error} — retrying
                </div>
            ) : null}
            <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
                <MetricTile
                    label="Requests"
                    value={fmt(last?.requests_total)}
                />
                <MetricTile
                    label="Failed"
                    value={fmt(last?.requests_failed_total)}
                />
                <MetricTile
                    label="Running"
                    value={fmt(last?.requests_running)}
                />
                <MetricTile
                    label="Waiting"
                    value={fmt(last?.requests_waiting)}
                />
                <MetricTile
                    label="Prompt tokens"
                    value={formatTokenCount(
                        last?.prompt_tokens_total ?? Number.NaN,
                    )}
                />
                <MetricTile
                    label="Completion tokens"
                    value={formatTokenCount(
                        last?.completion_tokens_total ?? Number.NaN,
                    )}
                />
                <MetricTile
                    label="Models loaded"
                    value={fmt(last?.models_loaded)}
                />
                <MetricTile
                    label="RSS"
                    value={formatBytes(mem?.rss_bytes ?? Number.NaN)}
                />
            </div>
            <div className="mt-4 grid gap-3 md:grid-cols-2">
                <MetricChart series={qpsSeries} unit="req/s" />
                <MetricChart series={tokSeries} unit="tok/s" />
            </div>
            <div className="mt-4 grid grid-cols-1 gap-3 md:grid-cols-3">
                <MetricTile
                    label="MLX active"
                    value={formatBytes(mem?.mlx_active_bytes ?? Number.NaN)}
                />
                <MetricTile
                    label="MLX cache"
                    value={formatBytes(mem?.mlx_cache_bytes ?? Number.NaN)}
                />
                <MetricTile
                    label="MLX peak"
                    value={formatBytes(mem?.mlx_peak_bytes ?? Number.NaN)}
                />
            </div>
            <div className="mt-4 rounded-card border border-border bg-bg-elevated p-4">
                <div className="text-sm text-text-primary">
                    Per-model memory
                </div>
                {perModel.length === 0 ? (
                    <div className="mt-2 text-xs text-text-muted">
                        No resident models
                    </div>
                ) : (
                    <ul className="mt-2 space-y-1">
                        {perModel.map((p) => (
                            <li
                                key={p.name}
                                className="flex items-center justify-between font-mono text-xs tabular-nums text-text-secondary"
                            >
                                <span className="truncate">{p.name}</span>
                                <span className="ml-2 shrink-0">
                                    {formatBytes(p.bytes)}
                                </span>
                            </li>
                        ))}
                    </ul>
                )}
            </div>
        </div>
    );
}

interface Point {
    t: number;
    v: number;
}

// Counter → per-second rate series: v[i] = (c[i] - c[i-1]) / dt.
function derivePoints(
    samples: Sample[],
    pick: (s: Sample) => number,
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
