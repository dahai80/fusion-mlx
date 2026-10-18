import { type ReactNode, useState } from 'react';

/** Form kit (spec §5.1): validation text below field, error color (S: 统一位置). */
export function FormField({
    label,
    error,
    required,
    children,
    htmlFor,
}: {
    label: string;
    error?: string | null;
    required?: boolean;
    children: ReactNode;
    htmlFor?: string;
}) {
    return (
        <div className="flex flex-col gap-1">
            <label htmlFor={htmlFor} className="text-xs text-text-muted">
                {label}
                {required ? <span className="ml-0.5 text-error">*</span> : null}
            </label>
            {children}
            {error ? <div className="text-xs text-error">{error}</div> : null}
        </div>
    );
}

export function TextInput({
    value,
    onChange,
    type = 'text',
    placeholder,
    disabled,
    id,
}: {
    value: string;
    onChange(value: string): void;
    type?: 'text' | 'password' | 'number' | 'email';
    placeholder?: string;
    disabled?: boolean;
    id?: string;
}) {
    return (
        <input
            id={id}
            type={type}
            value={value}
            placeholder={placeholder}
            disabled={disabled}
            onChange={(e) => onChange(e.target.value)}
            className="min-h-[36px] w-full rounded-md border border-border bg-bg px-3 py-2 text-sm text-text-primary outline-none focus:border-brand disabled:opacity-50"
        />
    );
}

export function SelectInput({
    value,
    onChange,
    options,
    id,
}: {
    value: string;
    onChange(value: string): void;
    options: Array<{ value: string; label: string }>;
    id?: string;
}) {
    return (
        <select
            id={id}
            value={value}
            onChange={(e) => onChange(e.target.value)}
            className="min-h-[36px] w-full rounded-md border border-border bg-bg px-3 py-2 text-sm text-text-primary outline-none focus:border-brand"
        >
            {options.map((o) => (
                <option key={o.value} value={o.value}>
                    {o.label}
                </option>
            ))}
        </select>
    );
}

/** Submit button with built-in loading state (spec §6.2: no dead clicks). */
export function SubmitButton({
    onAction,
    label,
    busyLabel = '提交中…',
    disabled,
}: {
    onAction(): Promise<void> | void;
    label: string;
    busyLabel?: string;
    disabled?: boolean;
}) {
    const [busy, setBusy] = useState(false);
    return (
        <button
            type="button"
            disabled={busy || disabled}
            onClick={() => {
                const r = onAction();
                if (r instanceof Promise) {
                    void r.finally(() => setBusy(false));
                    setBusy(true);
                }
            }}
            className="min-h-[36px] rounded-md bg-brand px-4 text-sm text-white disabled:opacity-50"
        >
            {busy ? busyLabel : label}
        </button>
    );
}
