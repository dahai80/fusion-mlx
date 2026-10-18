import {
    ConfirmDialog,
    DataTable,
    type DataTableColumn,
    EmptyState,
    ErrorState,
    SkeletonLines,
    StatusBadge,
    useAutoRetry,
} from "@fusion/web-ui";
import { useCallback, useEffect, useState } from "react";
import { type ModelEntry, api } from "../api";
import { PageHeader } from "../components/PageHeader";
import { sdk } from "../sdk";

// Model inventory: discovered engines with load/unload control. Four states
// per spec S4/S6 (loading skeleton / empty / error+retry / data).
export function ModelsPage() {
    const { data, error, loading, manualRetry } = useAutoRetry(
        () => api.models(),
        [],
    );
    const [pending, setPending] = useState<string | null>(null);
    const [confirmUnload, setConfirmUnload] = useState<ModelEntry | null>(null);

    // Poll refresh while mounted — load states transition in the background.
    useEffect(() => {
        const timer = window.setInterval(() => manualRetry(), 5000);
        return () => window.clearInterval(timer);
    }, [manualRetry]);

    const act = useCallback(
        async (id: string, op: "load" | "unload") => {
            setPending(id);
            try {
                if (op === "load") await api.load(id);
                else await api.unload(id);
                sdk.toast.show({
                    level: "success",
                    message: `${op} ok: ${id}`,
                });
            } catch (e) {
                console.error("[web] model op failed", id, op, e);
                sdk.toast.show({
                    level: "error",
                    message: `${op} failed: ${id}`,
                });
            } finally {
                setPending(null);
                manualRetry();
            }
        },
        [manualRetry],
    );

    if (loading && !data) return <SkeletonLines rows={6} />;
    if (error)
        return (
            <div className="mx-auto max-w-5xl px-4">
                <PageHeader title="Models" />
                <ErrorState message={error.message} onRetry={manualRetry} />
            </div>
        );
    const models = data?.models ?? [];
    if (models.length === 0)
        return (
            <div className="mx-auto max-w-5xl px-4">
                <PageHeader title="Models" />
                <EmptyState
                    title="No models discovered"
                    description="Download models to ~/.fusion-mlx/models and restart the server."
                />
            </div>
        );

    const columns: DataTableColumn<ModelEntry>[] = [
        {
            key: "id",
            header: "Model",
            sortable: true,
            render: (m) => (
                <span className="font-mono text-sm text-text-primary">
                    {m.id}
                    {m.pinned ? (
                        <span className="ml-2 text-xs text-warning">
                            pinned
                        </span>
                    ) : null}
                </span>
            ),
        },
        {
            key: "engine_type",
            header: "Engine",
            sortable: true,
            render: (m) => (
                <span className="text-sm text-text-secondary">
                    {m.engine_type ?? "-"}
                </span>
            ),
        },
        {
            key: "estimated_size_formatted",
            header: "Size",
            sortable: true,
            render: (m) => (
                <span className="font-mono text-sm tabular-nums text-text-secondary">
                    {m.actual_size_formatted ??
                        m.estimated_size_formatted ??
                        "-"}
                </span>
            ),
        },
        {
            key: "loaded",
            header: "Status",
            sortable: true,
            render: (m) => {
                if (m.is_loading)
                    return <StatusBadge status="running" label="loading" />;
                return m.loaded ? (
                    <StatusBadge status="success" label="loaded" />
                ) : (
                    <StatusBadge status="offline" label="idle" />
                );
            },
        },
        {
            key: "actions",
            header: "",
            align: "right",
            render: (m) => {
                const busy = pending === m.id;
                return m.loaded ? (
                    <button
                        type="button"
                        disabled={busy}
                        onClick={() => setConfirmUnload(m)}
                        className="rounded-md border border-error px-3 py-1 text-sm text-error hover:bg-bg-muted disabled:cursor-not-allowed disabled:opacity-40"
                    >
                        {busy ? "…" : "Unload"}
                    </button>
                ) : (
                    <button
                        type="button"
                        disabled={busy || m.is_loading}
                        onClick={() => act(m.id, "load")}
                        className="rounded-md bg-brand px-3 py-1 text-sm text-white hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-40"
                    >
                        {busy || m.is_loading ? "…" : "Load"}
                    </button>
                );
            },
        },
    ];

    return (
        <div className="mx-auto max-w-5xl px-4">
            <PageHeader
                title="Models"
                subtitle={`${models.filter((m) => m.loaded).length} loaded / ${models.length} discovered`}
            />
            <DataTable
                columns={columns}
                rows={models}
                rowKey={(m) => m.id}
                renderCard={(m) => (
                    <div className="flex items-center justify-between gap-2">
                        <div className="min-w-0">
                            <div className="truncate font-mono text-sm text-text-primary">
                                {m.id}
                            </div>
                            <div className="text-xs text-text-muted">
                                {m.engine_type ?? "-"} ·{" "}
                                {m.actual_size_formatted ??
                                    m.estimated_size_formatted ??
                                    "-"}
                            </div>
                        </div>
                        {m.is_loading ? (
                            <StatusBadge status="running" label="loading" />
                        ) : m.loaded ? (
                            <StatusBadge status="success" label="loaded" />
                        ) : (
                            <StatusBadge status="offline" label="idle" />
                        )}
                    </div>
                )}
            />
            <ConfirmDialog
                open={confirmUnload !== null}
                title="Unload model"
                description={`Evict ${confirmUnload?.id ?? ""} from the engine pool (LRU-free slot).`}
                confirmLabel="Unload"
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
