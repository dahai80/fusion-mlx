export interface PollerOptions {
    /** Poll interval while tab visible. Monitor-class default 5000. */
    intervalMs: number;
    /** Poll interval while hidden. Defaults to intervalMs * 6 (spec §6.3). */
    hiddenIntervalMs?: number;
    /** Max consecutive failures before onError fires. Default 3. */
    failureThreshold?: number;
    onTick(): Promise<void>;
    onError?(err: Error, failures: number): void;
}

/**
 * Visibility-aware poller: hidden tabs auto-throttle x6 (spec §6.3).
 * Never stacks ticks: each run schedules the next only after settle.
 */
export function createPoller(opts: PollerOptions) {
    let timer: ReturnType<typeof setTimeout> | null = null;
    let running = false;
    let failures = 0;

    function hiddenInterval(): number {
        return opts.hiddenIntervalMs ?? opts.intervalMs * 6;
    }

    function currentVisible(): boolean {
        return typeof document === 'undefined' || document.visibilityState === 'visible';
    }

    async function tick(): Promise<void> {
        if (!running) return;
        try {
            await opts.onTick();
            failures = 0;
        } catch (err) {
            failures++;
            console.warn('[fusion-sdk poller] tick failed', failures, err);
            if (failures >= (opts.failureThreshold ?? 3)) {
                opts.onError?.(err as Error, failures);
            }
        } finally {
            schedule();
        }
    }

    function schedule(): void {
        if (!running) return;
        clear();
        const delay = currentVisible() ? opts.intervalMs : hiddenInterval();
        timer = setTimeout(() => void tick(), delay);
    }

    function clear(): void {
        if (timer !== null) {
            clearTimeout(timer);
            timer = null;
        }
    }

    function onVisibilityChange(): void {
        if (!running) return;
        // Re-schedule at the new cadence; do not fire immediately to avoid bursts.
        schedule();
    }

    if (typeof document !== 'undefined') {
        document.addEventListener('visibilitychange', onVisibilityChange);
    }

    return {
        start(immediate = true): void {
            if (running) return;
            running = true;
            if (immediate) void tick();
            else schedule();
        },
        stop(): void {
            running = false;
            clear();
        },
    };
}
