/** mobile-readonly banner (spec §8.2): shown on mobile for readonly modules. */
export function ReadOnlyBanner({ text = '只读模式' }: { text?: string }) {
    return (
        <div className="bg-warning/15 px-3 py-2 text-center text-xs text-warning" role="status">
            {text}
        </div>
    );
}
