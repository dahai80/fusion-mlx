import type { ReactNode } from 'react';
import { useMemo, useState } from 'react';

export interface DataTableColumn<T> {
    key: string;
    header: string;
    render?: (row: T) => ReactNode;
    sortable?: boolean;
    align?: 'left' | 'right';
}

export interface DataTableProps<T> {
    columns: DataTableColumn<T>[];
    rows: T[];
    rowKey: (row: T) => string;
    pageSize?: number;
    selectable?: boolean;
    onSelectedChange?(selected: string[]): void;
    /** Card-mode row renderer for < md (spec §4.2); defaults to KV list of first 4 columns. */
    renderCard?: (row: T) => ReactNode;
    emptyState?: ReactNode;
}

/**
 * Sort / paginate / select table. Under md breakpoint renders card list
 * instead of table (spec §4.2) — module passes renderCard for proper cards.
 */
export function DataTable<T>({
    columns,
    rows,
    rowKey,
    pageSize = 20,
    selectable,
    onSelectedChange,
    renderCard,
    emptyState,
}: DataTableProps<T>) {
    const [sortKey, setSortKey] = useState<string | null>(null);
    const [page, setPage] = useState(0);
    const [selected, setSelected] = useState<Set<string>>(new Set());

    const sorted = useMemo(() => {
        if (!sortKey) return rows;
        const col = columns.find((c) => c.key === sortKey);
        if (!col?.sortable) return rows;
        return [...rows].sort((a, b) => {
            const av = (a as Record<string, unknown>)[sortKey];
            const bv = (b as Record<string, unknown>)[sortKey];
            if (typeof av === 'number' && typeof bv === 'number') return av - bv;
            return String(av ?? '').localeCompare(String(bv ?? ''), 'zh-CN');
        });
    }, [rows, sortKey, columns]);

    const pageCount = Math.max(1, Math.ceil(sorted.length / pageSize));
    const safePage = Math.min(page, pageCount - 1);
    const pageRows = sorted.slice(safePage * pageSize, safePage * pageSize + pageSize);

    const toggle = (key: string) => {
        setSelected((prev) => {
            const next = new Set(prev);
            if (next.has(key)) next.delete(key);
            else next.add(key);
            onSelectedChange?.([...next]);
            return next;
        });
    };

    const allSelected = pageRows.length > 0 && pageRows.every((r) => selected.has(rowKey(r)));
    const toggleAll = () => {
        setSelected((prev) => {
            const next = new Set(prev);
            if (allSelected) pageRows.forEach((r) => next.delete(rowKey(r)));
            else pageRows.forEach((r) => next.add(rowKey(r)));
            onSelectedChange?.([...next]);
            return next;
        });
    };

    const cardView = (row: T) => (
        <div className="rounded-card border border-border bg-bg-elevated p-3">
            {renderCard ? (
                renderCard(row)
            ) : (
                <dl className="grid grid-cols-2 gap-2">
                    {columns.slice(0, 4).map((c) => (
                        <div key={c.key} className="min-w-0">
                            <dt className="text-xs text-text-muted">{c.header}</dt>
                            <dd className="truncate text-sm text-text-primary">
                                {c.render
                                    ? c.render(row)
                                    : String((row as Record<string, unknown>)[c.key] ?? '-')}
                            </dd>
                        </div>
                    ))}
                </dl>
            )}
        </div>
    );

    return (
        <div>
            {/* Table for >= md */}
            <div className="hidden overflow-x-auto rounded-card border border-border bg-bg-elevated md:block">
                <table className="w-full text-sm">
                    <thead>
                        <tr className="border-b border-border text-left text-xs text-text-muted">
                            {selectable ? (
                                <th className="w-8 px-3 py-2">
                                    <input
                                        type="checkbox"
                                        checked={allSelected}
                                        onChange={toggleAll}
                                        aria-label="select all"
                                    />
                                </th>
                            ) : null}
                            {columns.map((c) => (
                                <th
                                    key={c.key}
                                    className={`px-3 py-2 ${c.align === 'right' ? 'text-right' : ''}`}
                                >
                                    {c.sortable ? (
                                        <button
                                            type="button"
                                            onClick={() =>
                                                setSortKey(sortKey === c.key ? null : c.key)
                                            }
                                            className="inline-flex items-center gap-1"
                                        >
                                            {c.header}
                                            {sortKey === c.key ? <span>↑</span> : null}
                                        </button>
                                    ) : (
                                        c.header
                                    )}
                                </th>
                            ))}
                        </tr>
                    </thead>
                    <tbody>
                        {pageRows.map((row) => {
                            const key = rowKey(row);
                            return (
                                <tr
                                    key={key}
                                    className="border-b border-border/60 last:border-0 hover:bg-bg-muted/50"
                                >
                                    {selectable ? (
                                        <td className="px-3 py-2">
                                            <input
                                                type="checkbox"
                                                checked={selected.has(key)}
                                                onChange={() => toggle(key)}
                                                aria-label={`select ${key}`}
                                            />
                                        </td>
                                    ) : null}
                                    {columns.map((c) => (
                                        <td
                                            key={c.key}
                                            className={`px-3 py-2 ${c.align === 'right' ? 'text-right' : ''}`}
                                        >
                                            {c.render
                                                ? c.render(row)
                                                : String(
                                                      (row as Record<string, unknown>)[c.key] ?? '-'
                                                  )}
                                        </td>
                                    ))}
                                </tr>
                            );
                        })}
                    </tbody>
                </table>
                {pageRows.length === 0
                    ? (emptyState ?? (
                          <div className="px-3 py-8 text-center text-sm text-text-muted">
                              暂无数据
                          </div>
                      ))
                    : null}
            </div>

            {/* Card list for < md */}
            <div className="flex flex-col gap-2 md:hidden">
                {pageRows.map((row) => (
                    <div key={rowKey(row)}>{cardView(row)}</div>
                ))}
                {pageRows.length === 0 ? (
                    <div className="py-8 text-center text-sm text-text-muted">暂无数据</div>
                ) : null}
            </div>

            {pageCount > 1 ? (
                <div className="mt-2 flex items-center justify-end gap-2 text-xs text-text-secondary">
                    <button
                        type="button"
                        onClick={() => setPage(Math.max(0, safePage - 1))}
                        disabled={safePage === 0}
                        className="rounded px-2 py-1 disabled:opacity-40"
                    >
                        上一页
                    </button>
                    <span className="font-mono tabular-nums">
                        {safePage + 1}/{pageCount}
                    </span>
                    <button
                        type="button"
                        onClick={() => setPage(Math.min(pageCount - 1, safePage + 1))}
                        disabled={safePage === pageCount - 1}
                        className="rounded px-2 py-1 disabled:opacity-40"
                    >
                        下一页
                    </button>
                </div>
            ) : null}
        </div>
    );
}
