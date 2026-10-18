import { useEffect, useState } from 'react';

export interface ConfirmDialogProps {
    open: boolean;
    title: string;
    description?: string;
    confirmLabel?: string;
    cancelLabel?: string;
    /** Destructive: red confirm button. */
    danger?: boolean;
    /** Deletion guard: user must type this name to enable confirm (spec §5.1). */
    confirmName?: string;
    onConfirm(): void;
    onCancel(): void;
}

/** Confirmation modal; destructive ops require danger + name-typing (spec S5). */
export function ConfirmDialog({
    open,
    title,
    description,
    confirmLabel = '确认',
    cancelLabel = '取消',
    danger,
    confirmName,
    onConfirm,
    onCancel,
}: ConfirmDialogProps) {
    const [typed, setTyped] = useState('');
    useEffect(() => {
        if (open) setTyped('');
    }, [open]);
    useEffect(() => {
        if (!open) return;
        const onKey = (e: KeyboardEvent) => {
            if (e.key === 'Escape') onCancel();
        };
        window.addEventListener('keydown', onKey);
        return () => window.removeEventListener('keydown', onKey);
    }, [open, onCancel]);
    if (!open) return null;
    const nameOk = !confirmName || typed === confirmName;
    return (
        <div
            className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4"
            role="dialog"
            aria-modal="true"
            onMouseDown={(e) => e.target === e.currentTarget && onCancel()}
        >
            <div className="w-full max-w-sm rounded-card border border-border bg-bg-elevated p-4 shadow-overlay">
                <div className={`text-base ${danger ? 'text-error' : 'text-text-primary'}`}>
                    {title}
                </div>
                {description ? (
                    <div className="mt-1 text-sm text-text-secondary">{description}</div>
                ) : null}
                {confirmName ? (
                    <input
                        autoFocus
                        value={typed}
                        onChange={(e) => setTyped(e.target.value)}
                        placeholder={`输入 ${confirmName} 以确认`}
                        className="mt-3 w-full rounded-md border border-border bg-bg px-3 py-2 font-mono text-sm text-text-primary outline-none focus:border-brand"
                    />
                ) : null}
                <div className="mt-4 flex justify-end gap-2">
                    <button
                        type="button"
                        onClick={onCancel}
                        className="rounded-md border border-border px-4 py-2 text-sm text-text-secondary hover:bg-bg-muted"
                    >
                        {cancelLabel}
                    </button>
                    <button
                        type="button"
                        disabled={!nameOk}
                        onClick={onConfirm}
                        className={`rounded-md px-4 py-2 text-sm text-white disabled:cursor-not-allowed disabled:opacity-40 ${danger ? 'bg-error' : 'bg-brand'}`}
                    >
                        {confirmLabel}
                    </button>
                </div>
            </div>
        </div>
    );
}
