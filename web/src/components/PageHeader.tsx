import type { ReactNode } from "react";

// Spec §4.1 page skeleton header: title + subtitle left, global actions
// right. Modules carry no side nav / top nav of their own.
export function PageHeader({
    title,
    subtitle,
    actions,
}: { title: string; subtitle?: string; actions?: ReactNode }) {
    return (
        <div className="mb-4 flex items-start justify-between gap-4 px-1">
            <div>
                <h1 className="text-lg text-text-primary">{title}</h1>
                {subtitle ? (
                    <p className="mt-0.5 text-sm text-text-muted">{subtitle}</p>
                ) : null}
            </div>
            {actions ? (
                <div className="flex shrink-0 gap-2">{actions}</div>
            ) : null}
        </div>
    );
}
