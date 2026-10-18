import { HttpError, withRetry } from './retry';
import type {
    BridgeMessage,
    DeviceInfo,
    FusionShellSdk,
    HttpRequestInit,
    HttpResult,
    ModuleRequest,
    ShellSdkOptions,
    ThemeMode,
    ToastInput,
} from './types';

export { withRetry, withTimeout, isRetryableError, TimeoutError, HttpError } from './retry';
export { createPoller, type PollerOptions } from './poll';
export { createHostBridge, type HostBridgeHandle, type HostBridgeOptions } from './host';
export * from './types';

const DEFAULT_TIMEOUT = 15000;
const DEFAULT_RETRIES = 2;

let requestSeq = 0;
const nextId = () => `req-${Date.now().toString(36)}-${++requestSeq}`;

interface BridgeDeps {
    isEmbedded: boolean;
    send(msg: ModuleRequest): void;
    onEvent(listener: (event: string, payload: unknown) => void): () => void;
    log(level: 'log' | 'warn' | 'error', ...args: unknown[]): void;
}

function createBridge(deps: BridgeDeps, opts: Required<ShellSdkOptions>) {
    const pending = new Map<
        string,
        {
            resolve: (v: unknown) => void;
            reject: (e: Error) => void;
            timer: ReturnType<typeof setTimeout>;
        }
    >();

    function handleMessage(msg: BridgeMessage): void {
        if (!msg || typeof msg !== 'object') return;
        if (msg.type === 'response') {
            const entry = pending.get(msg.id);
            if (!entry) return;
            pending.delete(msg.id);
            clearTimeout(entry.timer);
            if (msg.ok) entry.resolve(msg.result);
            else
                entry.reject(
                    new HttpError(
                        502,
                        msg.error?.code ?? 'bridge_error',
                        msg.error?.message ?? 'bridge request failed'
                    )
                );
            return;
        }
        if (msg.type === 'event') {
            eventListeners.get(msg.event)?.forEach((fn) => fn(msg.payload));
        }
    }

    const eventListeners = new Map<string, Set<(payload: unknown) => void>>();

    function call<T>(
        method: ModuleRequest['method'],
        payload: unknown,
        timeoutMs?: number
    ): Promise<T> {
        if (!deps.isEmbedded) {
            return Promise.reject(new Error('not embedded: bridge unavailable'));
        }
        const id = nextId();
        return withRetry(
            () =>
                new Promise<T>((resolve, reject) => {
                    const timer = setTimeout(() => {
                        pending.delete(id);
                        reject(new Error(`bridge timeout: ${method}`));
                    }, timeoutMs ?? opts.timeoutMs);
                    pending.set(id, { resolve: resolve as (v: unknown) => void, reject, timer });
                    deps.send({ type: 'request', id, method, payload });
                }),
            { retries: opts.retries, timeoutMs: timeoutMs ?? opts.timeoutMs }
        ).catch((err) => {
            deps.log('error', `[fusion-sdk] ${method} failed:`, err);
            throw err;
        }) as Promise<T>;
    }

    return { handleMessage, call, deps, eventListeners };
}

function standaloneSdk(log: BridgeDeps['log']): FusionShellSdk {
    // Standalone dev mode (module served by its own Vite server, not embedded).
    // Theme/devices derive from the browser; http goes same-origin direct.
    log('warn', '[fusion-sdk] standalone mode: http goes direct, no host APIs');
    const themeListeners = new Set<(mode: ThemeMode) => void>();
    const mql =
        typeof window !== 'undefined' && window.matchMedia
            ? window.matchMedia('(prefers-color-scheme: dark)')
            : null;
    mql?.addEventListener('change', (e) =>
        themeListeners.forEach((fn) => fn(e.matches ? 'dark' : 'light'))
    );
    const request = async <T>(path: string, init?: HttpRequestInit): Promise<HttpResult<T>> => {
        // #20: JSON bodies without an explicit content-type make FastAPI (and
        // friends) reject the request with 422. Default to application/json;
        // a caller-provided header always wins.
        const hasContentType = Object.keys(init?.headers ?? {}).some(
            (k) => k.toLowerCase() === 'content-type'
        );
        const headers =
            init?.body !== undefined && !hasContentType
                ? { 'content-type': 'application/json', ...init?.headers }
                : init?.headers;
        const res = await fetch(path, {
            method: init?.method ?? 'GET',
            headers,
            body: init?.body === undefined ? undefined : JSON.stringify(init.body),
        });
        const data = (await res.json().catch(() => null)) as T;
        if (!res.ok) throw new HttpError(res.status, 'http_error', `request failed: ${res.status}`);
        return { status: res.status, ok: true, data };
    };
    return {
        http: {
            request,
            get: <T>(path: string) => request<T>(path),
            post: <T>(path: string, body?: unknown) => request<T>(path, { method: 'POST', body }),
            put: <T>(path: string, body?: unknown) => request<T>(path, { method: 'PUT', body }),
            delete: <T>(path: string) => request<T>(path, { method: 'DELETE' }),
        },
        toast: {
            show(input: ToastInput) {
                log(
                    input.level === 'error' ? 'error' : 'log',
                    `[toast:${input.level}]`,
                    input.message
                );
            },
        },
        tab: {
            setActiveTask() {
                // No host to guard tabs in standalone mode.
            },
        },
        theme: {
            get: () => (mql?.matches ? 'dark' : 'light'),
            subscribe(listener) {
                themeListeners.add(listener);
                return () => themeListeners.delete(listener);
            },
        },
        devices: {
            get(): DeviceInfo {
                const mobile =
                    typeof window !== 'undefined' &&
                    window.matchMedia?.('(max-width: 767px)').matches;
                return { viewport: mobile ? 'mobile' : 'desktop', mode: null };
            },
        },
        visibility: {
            get: () => typeof document === 'undefined' || document.visibilityState === 'visible',
            subscribe(listener) {
                const fn = () => listener(document.visibilityState === 'visible');
                document.addEventListener('visibilitychange', fn);
                return () => document.removeEventListener('visibilitychange', fn);
            },
        },
        i18n: {
            lang: () =>
                typeof document !== 'undefined' && document.documentElement.lang.startsWith('en')
                    ? 'en'
                    : 'zh',
            t: (zh: string, en: string) =>
                typeof document !== 'undefined' && document.documentElement.lang.startsWith('en')
                    ? en
                    : zh,
            subscribe(listener) {
                if (typeof document === 'undefined') return () => undefined;
                const obs = new MutationObserver(() =>
                    listener(document.documentElement.lang.startsWith('en') ? 'en' : 'zh')
                );
                obs.observe(document.documentElement, {
                    attributes: true,
                    attributeFilter: ['lang'],
                });
                return () => obs.disconnect();
            },
        },
    };
}

export function createShellSdk(options?: ShellSdkOptions): FusionShellSdk {
    const opts: Required<ShellSdkOptions> = {
        origin: options?.origin ?? (typeof window !== 'undefined' ? window.location.origin : ''),
        timeoutMs: options?.timeoutMs ?? DEFAULT_TIMEOUT,
        retries: options?.retries ?? DEFAULT_RETRIES,
    };
    const log: BridgeDeps['log'] = (level, ...args) => {
        // eslint-disable-next-line no-console
        (console[level] as typeof console.log)(...args);
    };
    if (typeof window === 'undefined' || window.parent === window) {
        return standaloneSdk(log);
    }

    const themeListeners = new Set<(mode: ThemeMode) => void>();
    const visListeners = new Set<(visible: boolean) => void>();

    const bridge = createBridge(
        {
            isEmbedded: true,
            send: (msg) => window.parent.postMessage(msg, opts.origin),
            onEvent: (listener) => {
                // routed via eventListeners in handleMessage; wired below
                void listener;
                return () => undefined;
            },
            log,
        },
        opts
    );

    bridge.eventListeners.set(
        'theme',
        new Set([(payload) => themeListeners.forEach((fn) => fn(payload as ThemeMode))])
    );
    bridge.eventListeners.set(
        'visibility',
        new Set([(payload) => visListeners.forEach((fn) => fn(payload as boolean))])
    );

    window.addEventListener('message', (ev: MessageEvent) => {
        if (ev.origin !== opts.origin) {
            log('warn', '[fusion-sdk] dropped message from disallowed origin:', ev.origin);
            return;
        }
        if (ev.source !== window.parent) {
            log('warn', '[fusion-sdk] dropped message from non-parent source');
            return;
        }
        bridge.handleMessage(ev.data as BridgeMessage);
    });

    // Host heartbeat (PRD §7.2): answer pings so host liveness detection stays
    // accurate. Modules that never answer are treated as not SDK-integrated.
    let lang: 'zh' | 'en' = document.documentElement.lang.startsWith('en') ? 'en' : 'zh';
    const langListeners = new Set<(l: 'zh' | 'en') => void>();
    window.addEventListener('message', (ev: MessageEvent) => {
        if (ev.origin !== opts.origin || ev.source !== window.parent) return;
        const msg = ev.data as BridgeMessage;
        if (msg?.type !== 'request' || (msg as { method?: string }).method !== 'ping') return;
        window.parent.postMessage(
            { type: 'response', id: msg.id, ok: true, result: 'pong' },
            opts.origin
        );
    });
    window.addEventListener('message', (ev: MessageEvent) => {
        if (ev.origin !== opts.origin || ev.source !== window.parent) return;
        const msg = ev.data as BridgeMessage;
        if (msg?.type !== 'event' || (msg as { event?: string }).event !== 'lang') return;
        lang = (msg as { payload?: string }).payload === 'en' ? 'en' : 'zh';
        langListeners.forEach((fn) => fn(lang));
    });

    const http = {
        async request<T = unknown>(path: string, init?: HttpRequestInit): Promise<HttpResult<T>> {
            const res = await bridge.call<HttpResult<T>>(
                'http',
                {
                    path,
                    method: init?.method ?? 'GET',
                    headers: init?.headers,
                    body: init?.body,
                    timeoutMs: init?.timeoutMs,
                },
                init?.timeoutMs
            );
            return res;
        },
        get: <T>(path: string) => http.request<T>(path),
        post: <T>(path: string, body?: unknown) => http.request<T>(path, { method: 'POST', body }),
        put: <T>(path: string, body?: unknown) => http.request<T>(path, { method: 'PUT', body }),
        delete: <T>(path: string) => http.request<T>(path, { method: 'DELETE' }),
    };

    return {
        http,
        toast: {
            show(input: ToastInput) {
                void bridge.call('toast', input, 5000).catch(() => undefined);
            },
        },
        tab: {
            setActiveTask(task: { name: string } | null) {
                void bridge.call('tab', task, 5000).catch(() => undefined);
            },
        },
        theme: {
            get: () =>
                typeof document !== 'undefined' &&
                document.documentElement.classList.contains('light')
                    ? 'light'
                    : 'dark',
            subscribe(listener) {
                themeListeners.add(listener);
                return () => themeListeners.delete(listener);
            },
        },
        devices: {
            get(): DeviceInfo {
                const mobile = window.matchMedia('(max-width: 767px)').matches;
                let mode: DeviceInfo['mode'] = null;
                try {
                    mode = JSON.parse(
                        new URLSearchParams(window.location.search).get('devices') ?? 'null'
                    ) as DeviceInfo['mode'];
                } catch {
                    mode = null;
                }
                return { viewport: mobile ? 'mobile' : 'desktop', mode };
            },
        },
        visibility: {
            get: () => document.visibilityState === 'visible',
            subscribe(listener) {
                visListeners.add(listener);
                return () => visListeners.delete(listener);
            },
        },
        i18n: {
            lang: () => lang,
            t: (zh: string, en: string) => (lang === 'en' ? en : zh),
            subscribe(listener) {
                langListeners.add(listener);
                return () => langListeners.delete(listener);
            },
        },
    };
}
