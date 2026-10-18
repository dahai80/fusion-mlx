export interface RetryOptions {
    retries: number;
    timeoutMs: number;
    /** Backoff base in ms; attempt n waits base * 2^n. Default 300. */
    backoffMs?: number;
    onRetry?: (attempt: number, error: Error) => void;
}

export class TimeoutError extends Error {
    constructor(ms: number) {
        super(`operation timed out after ${ms}ms`);
        this.name = 'TimeoutError';
    }
}

export function isRetryableError(err: unknown): boolean {
    if (err instanceof TimeoutError) return true;
    const status = (err as { status?: number }).status;
    // 4xx are deterministic: retrying will not help.
    if (typeof status === 'number') return status >= 500 || status === 429;
    return true;
}

export function withTimeout<T>(promise: Promise<T>, timeoutMs: number): Promise<T> {
    return new Promise<T>((resolve, reject) => {
        const timer = setTimeout(() => reject(new TimeoutError(timeoutMs)), timeoutMs);
        promise.then(
            (value) => {
                clearTimeout(timer);
                resolve(value);
            },
            (err) => {
                clearTimeout(timer);
                reject(err);
            }
        );
    });
}

const sleep = (ms: number) => new Promise<void>((r) => setTimeout(r, ms));

export async function withRetry<T>(
    fn: (attempt: number) => Promise<T>,
    opts: RetryOptions
): Promise<T> {
    let lastError: unknown;
    for (let attempt = 0; attempt <= opts.retries; attempt++) {
        try {
            return await fn(attempt);
        } catch (err) {
            lastError = err;
            if (attempt === opts.retries || !isRetryableError(err)) break;
            opts.onRetry?.(attempt + 1, err as Error);
            const backoff = (opts.backoffMs ?? 300) * 2 ** attempt;
            await sleep(backoff);
        }
    }
    throw lastError;
}

/** Error shape surfaced by the http proxy: gateway error envelope passthrough. */
export class HttpError extends Error {
    readonly status: number;
    readonly code: string;
    constructor(status: number, code: string, message: string) {
        super(message);
        this.name = 'HttpError';
        this.status = status;
        this.code = code;
    }
}
