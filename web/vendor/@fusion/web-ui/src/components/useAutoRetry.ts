import { useCallback, useEffect, useRef, useState } from 'react';

export interface AutoRetryOptions {
    retries?: number;
    backoffMs?: number;
    /** Skip auto-retry for these statuses (e.g. 401/403). */
    noRetryStatuses?: number[];
}

/**
 * Spec §6.1 Retry state: exponential backoff auto-retry with visible
 * "第 N 次重试中" count. Turns into error state after exhausting retries.
 */
export function useAutoRetry<T>(
    fn: () => Promise<T>,
    deps: unknown[],
    opts: AutoRetryOptions = {}
) {
    const { retries = 3, backoffMs = 1000, noRetryStatuses = [401, 403] } = opts;
    const [data, setData] = useState<T | null>(null);
    const [error, setError] = useState<Error | null>(null);
    const [loading, setLoading] = useState(true);
    const [attempt, setAttempt] = useState(0);
    const fnRef = useRef(fn);
    fnRef.current = fn;
    const runIdRef = useRef(0);

    const run = useCallback(
        async (currentAttempt: number) => {
            const runId = ++runIdRef.current;
            setLoading(true);
            setError(null);
            try {
                const result = await fnRef.current();
                if (runId !== runIdRef.current) return;
                setData(result);
                setAttempt(0);
                // #19: success must clear loading or loading-gated pages
                // (data ? content : skeleton) stay on the skeleton forever.
                setLoading(false);
            } catch (err) {
                if (runId !== runIdRef.current) return;
                const status = (err as { status?: number }).status;
                const lastTry = currentAttempt >= retries || noRetryStatuses.includes(status ?? 0);
                if (lastTry) {
                    // Reset attempt so consumers stop rendering "第 N 次重试中"
                    // and the manual retry button becomes available again.
                    setAttempt(0);
                    setError(err as Error);
                    setLoading(false);
                    return;
                }
                setAttempt(currentAttempt + 1);
                const delay = backoffMs * 2 ** currentAttempt;
                setTimeout(() => {
                    if (runId === runIdRef.current) void run(currentAttempt + 1);
                }, delay);
            }
        },
        [retries, backoffMs, noRetryStatuses]
    );

    // biome-ignore lint/correctness/useExhaustiveDependencies: caller-provided deps intentionally gate the re-run; run is a stable useCallback
    useEffect(() => {
        void run(0);
        return () => {
            runIdRef.current++;
        };
    }, deps);

    return { data, error, loading, attempt, manualRetry: () => void run(0) };
}
