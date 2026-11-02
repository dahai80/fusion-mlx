import {
    ConfirmDialog,
    EmptyState,
    ErrorState,
    SkeletonLines,
    StatusBadge,
    useAutoRetry,
} from "@fusion/web-ui";
import { useCallback, useEffect, useMemo, useState } from "react";
import { type ModelEntry, api } from "../api";
import { PageHeader } from "../components/PageHeader";
import { useT } from "../i18n";
import { sdk } from "../sdk";

// Model registry cards (#1278 C4): grouped by engine/modality, groups
// collapsible (31+ flat rows were the old table), one-click load/unload
// per card. Polling + toast flow unchanged from the table version.
// Spec S9: group labels are zh/en pairs picked by the host language.
const GROUP_LABELS: Record<string, [string, string]> = {
    llm: ["对话 / LLM", "Chat / LLM"],
    vlm: ["视觉语言", "Vision-language"],
    embedding: ["向量", "Embedding"],
    reranker: ["重排", "Reranker"],
    image: ["图像", "Image"],
    video: ["视频", "Video"],
    audio: ["音频", "Audio"],
    speech: ["语音", "Speech"],
    tts: ["TTS 合成", "TTS"],
    stt: ["STT 识别", "STT"],
    ner: ["NER", "NER"],
    ocr: ["OCR", "OCR"],
};

function groupKey(m: ModelEntry): string {
    const t = (m.model_type ?? m.engine_type ?? "other").toLowerCase();
    return GROUP_LABELS[t] ? t : t.split("_")[0] || "other";
}

function groupLabel(key: string, lang: "zh" | "en"): string {
    const pair = GROUP_LABELS[key];
    return pair ? (lang === "en" ? pair[1] : pair[0]) : key.toUpperCase();
}

export function ModelsPage() {
    const t = useT();
    const { data, error, loading, manualRetry } = useAutoRetry(
        () => api.models(),
        [],
    );
    const [pending, setPending] = useState<string | null>(null);
    const [confirmUnload, setConfirmUnload] = useState<ModelEntry | null>(null);
    const [collapsed, setCollapsed] = useState<Set<string>>(new Set());

    // Poll refresh while mounted — load states transition in the background.
    useEffect(() => {
        const timer = window.setInterval(() => manualRetry(), 5000);
        return () => window.clearInterval(timer);
    }, [manualRetry]);

    const act = useCallback(
        async (id: string, op: "load" | "unload") => {
            setPending(id);
            const okMsg =
                op === "load"
                    ? t("加载成功", "Load ok")
                    : t("卸载成功", "Unload ok");
            const failMsg =
                op === "load"
                    ? t("加载失败", "Load failed")
                    : t("卸载失败", "Unload failed");
            try {
                if (op === "load") await api.load(id);
                else await api.unload(id);
                sdk.toast.show({
                    level: "success",
                    message: `${okMsg}: ${id}`,
                });
            } catch (e) {
                console.error("[web] model op failed", id, op, e);
                sdk.toast.show({
                    level: "error",
                    message: `${failMsg}: ${id}`,
                });
            } finally {
                setPending(null);
                manualRetry();
            }
        },
        [manualRetry, t],
    );

    const models = data?.models ?? [];
    const groups = useMemo(() => {
        const map = new Map<string, ModelEntry[]>();
        for (const m of models) {
            const k = groupKey(m);
            const arr = map.get(k);
            if (arr) arr.push(m);
            else map.set(k, [m]);
        }
        return [...map.entries()].sort((a, b) => a[0].localeCompare(b[0]));
    }, [models]);

    const toggle = (key: string) => {
        setCollapsed((prev) => {
            const next = new Set(prev);
            if (next.has(key)) next.delete(key);
            else next.add(key);
            return next;
        });
    };

    if (loading && !data) return <SkeletonLines rows={6} />;
    if (error)
        return (
            <div className="mx-auto max-w-5xl px-4">
                <PageHeader title={t("模型", "Models")} />
                <ErrorState message={error.message} onRetry={manualRetry} />
            </div>
        );
    if (models.length === 0)
        return (
            <div className="mx-auto max-w-5xl px-4">
                <PageHeader title={t("模型", "Models")} />
                <EmptyState
                    title={t("未发现模型", "No models discovered")}
                    description={t(
                        "请将模型下载到 ~/.fusion-mlx/models 后重启服务。",
                        "Download models to ~/.fusion-mlx/models and restart the server.",
                    )}
                />
            </div>
        );

    const loadedCount = models.filter((m) => m.loaded).length;
    return (
        <div className="mx-auto max-w-5xl px-4">
            <PageHeader
                title={t("模型", "Models")}
                subtitle={t(
                    `已加载 ${loadedCount} / 共发现 ${models.length}`,
                    `${loadedCount} loaded / ${models.length} discovered`,
                )}
            />
            {groups.map(([key, entries]) => {
                const isCollapsed = collapsed.has(key);
                const groupLoaded = entries.filter((m) => m.loaded).length;
                return (
                    <section key={key} className="mb-4">
                        <button
                            type="button"
                            onClick={() => toggle(key)}
                            className="flex w-full items-center gap-2 text-left"
                        >
                            <span
                                className={`text-xs text-text-muted transition-transform ${isCollapsed ? "" : "rotate-90"}`}
                            >
                                ▶
                            </span>
                            <span className="text-sm text-text-primary">
                                {groupLabel(key, sdk.i18n.lang())}
                            </span>
                            <span className="text-xs text-text-muted">
                                {t(
                                    `已加载 ${groupLoaded}/${entries.length}`,
                                    `${groupLoaded}/${entries.length} loaded`,
                                )}
                            </span>
                        </button>
                        {!isCollapsed && (
                            <div className="mt-2 grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
                                {entries.map((m) => (
                                    <ModelCard
                                        key={m.id}
                                        m={m}
                                        pending={pending === m.id}
                                        onLoad={() => act(m.id, "load")}
                                        onUnload={() => setConfirmUnload(m)}
                                    />
                                ))}
                            </div>
                        )}
                    </section>
                );
            })}
            <ConfirmDialog
                open={confirmUnload !== null}
                title={t("卸载模型", "Unload model")}
                description={t(
                    `将从引擎池卸载 ${confirmUnload?.id ?? ""}（释放 LRU 槽位）。`,
                    `Evict ${confirmUnload?.id ?? ""} from the engine pool (LRU-free slot).`,
                )}
                confirmLabel={t("卸载", "Unload")}
                danger
                onConfirm={() => {
                    const m = confirmUnload;
                    setConfirmUnload(null);
                    if (m) void act(m.id, "unload");
                }}
                onCancel={() => setConfirmUnload(null)}
            />
        </div>
    );
}

function ModelCard({
    m,
    pending,
    onLoad,
    onUnload,
}: {
    m: ModelEntry;
    pending: boolean;
    onLoad: () => void;
    onUnload: () => void;
}) {
    const t = useT();
    const busy = pending || m.is_loading;
    return (
        <div className="rounded-card border border-border bg-bg-elevated p-3">
            <div className="flex items-start justify-between gap-2">
                <div className="min-w-0">
                    <div className="truncate font-mono text-sm text-text-primary">
                        {m.id}
                        {m.pinned ? (
                            <span className="ml-2 text-xs text-warning">
                                {t("已固定", "pinned")}
                            </span>
                        ) : null}
                    </div>
                    <div className="mt-1 text-xs text-text-muted">
                        {m.engine_type ?? "-"} ·{" "}
                        {m.actual_size_formatted ??
                            m.estimated_size_formatted ??
                            "-"}
                    </div>
                </div>
                {m.is_loading ? (
                    <StatusBadge
                        status="running"
                        label={t("加载中", "loading")}
                    />
                ) : m.loaded ? (
                    <StatusBadge
                        status="success"
                        label={t("已加载", "loaded")}
                    />
                ) : (
                    <StatusBadge status="offline" label={t("空闲", "idle")} />
                )}
            </div>
            <div className="mt-2 flex justify-end">
                {m.loaded ? (
                    <button
                        type="button"
                        disabled={pending}
                        onClick={onUnload}
                        className="rounded-md border border-error px-3 py-1 text-sm text-error hover:bg-bg-muted disabled:cursor-not-allowed disabled:opacity-40"
                    >
                        {pending ? "…" : t("卸载", "Unload")}
                    </button>
                ) : (
                    <button
                        type="button"
                        disabled={busy}
                        onClick={onLoad}
                        className="rounded-md bg-brand px-3 py-1 text-sm text-white hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-40"
                    >
                        {busy ? "…" : t("加载", "Load")}
                    </button>
                )}
            </div>
        </div>
    );
}
