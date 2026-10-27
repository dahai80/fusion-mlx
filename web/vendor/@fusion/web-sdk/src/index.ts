import { HttpError, withRetry } from './retry';
import { createThemeState } from './theme';
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
export { createThemeState } from './theme';
export { createPoller, type PollerOptions } from './poll';
export { createHostBridge, type HostBridgeHandle, type HostBridgeOptions } from './host';
export * from './types';

const DEFAULT_TIMEOUT = 15000;
const DEFAULT_RETRIES = 2;

let requestSeq = 0;
const nextId = () => `req-${Date.now().toString(36)}-${++requestSeq}`;

// #1298: the shell marks embedded deep links with ?embedded=1. A top-level
// open of such a link has no parent frame, so the iframe check alone reports
// standalone and skips the host APIs.
function embeddedFromLocation(): boolean {
    if (typeof window === 'undefined') return false;
    try {
        return new URLSearchParams(window.location.search).get('embedded') === '1';
    } catch {
        return false;
    }
}

async function directRequest<T>(path: string, init?: HttpRequestInit): Promise<HttpResult<T>> {
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
    const controller = new AbortController();
    const timer = init?.timeoutMs
        ? setTimeout(() => controller.abort(), init.timeoutMs)
        : undefined;
    let res: Response;
    try {
        res = await fetch(path, {
            method: init?.method ?? 'GET',
            headers,
            body: init?.body === undefined ? undefined : JSON.stringify(init.body),
            signal: init?.timeoutMs ? controller.signal : undefined,
        });
    } catch (err) {
        if (init?.timeoutMs && controller.signal.aborted) {
            throw new HttpError(
                0,
                'timeout',
                `request timed out after ${init.timeoutMs}ms`
            );
        }
        throw err;
    } finally {
        if (timer !== undefined) clearTimeout(timer);
    }
    const data = (await res.json().catch(() => null)) as T;
    if (!res.ok) throw new HttpError(res.status, 'http_error', `request failed: ${res.status}`);
    return { status: res.status, ok: true, data };
}

interface BridgeDeps {
    isEmbedded: boolean;
    send(msg: ModuleRequest): void;
    onEvent(listener: (event: string, payload: unknown) => void): () => void;
    log(level: 'log' | 'warn' | 'error', ...args: unknown[]): void;
}

function createBridge(
    deps: BridgeDeps,
    opts: Required<Omit<ShellSdkOptions, 'embedded'>>
) {
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
    const request = async <T>(path: string, init?: HttpRequestInit): Promise<HttpResult<T>> =>
        directRequest<T>(path, init);
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
    const opts: Required<Omit<ShellSdkOptions, 'embedded'>> = {
        origin: options?.origin ?? (typeof window !== 'undefined' ? window.location.origin : ''),
        timeoutMs: options?.timeoutMs ?? DEFAULT_TIMEOUT,
        retries: options?.retries ?? DEFAULT_RETRIES,
    };
    const log: BridgeDeps['log'] = (level, ...args) => {
        // eslint-disable-next-line no-console
        (console[level] as typeof console.log)(...args);
    };
    // #1298: embedded when framed by the shell OR opened as a shell deep
    // link (?embedded=1) at top level. The host APIs (postMessage bridge,
    // heartbeat responder, theme/lang events) are wired in both cases so a
    // framed deep link behaves identically; without a parent frame there is
    // no host to answer, so http/toast/tab degrade explicitly instead of
    // timing out through the bridge.
    const hasHostFrame = typeof window !== 'undefined' && window.parent !== window;
    const embedded = options?.embedded ?? (hasHostFrame || embeddedFromLocation());
    if (typeof window === 'undefined' || !embedded) {
        return standaloneSdk(log);
    }
    if (!hasHostFrame) {
        log(
            'log',
            '[fusion-sdk] embedded via ?embedded=1 without a parent frame: ' +
                'host relay unavailable, http goes direct'
        );
    }

    const themeState = createThemeState(
        document.documentElement.classList.contains('light') ? 'light' : 'dark'
    );
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
        new Set([(payload) => themeState.set(payload as ThemeMode)])
    );
    // Seed from the host (issue #77): the mount-time theme push can be lost
    // to about:blank before the module's listener exists, so pull once in the
    // background; themeState.set dedupes against a push that did land.
    void bridge
        .call<ThemeMode>('theme', null, 5000)
        .then((mode) => themeState.set(mode))
        .catch(() => undefined);
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

    if (!hasHostFrame) {
        // #1298: without a parent frame no host event can arrive, so derive
        // visibility and language from the document instead. Subscribers used
        // to register and never update (silent stale state).
        const emitVisibility = () =>
            visListeners.forEach((fn) => fn(document.visibilityState === 'visible'));
        document.addEventListener('visibilitychange', emitVisibility);
        if (typeof MutationObserver !== 'undefined') {
            const langObs = new MutationObserver(() => {
                const next = document.documentElement.lang.startsWith('en') ? 'en' : 'zh';
                if (next === lang) return;
                lang = next;
                langListeners.forEach((fn) => fn(lang));
            });
            langObs.observe(document.documentElement, {
                attributes: true,
                attributeFilter: ['lang'],
            });
        }
    }
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
            if (!hasHostFrame) {
                return directRequest<T>(path, init);
            }
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
                if (!hasHostFrame) {
                    log(
                        input.level === 'error' ? 'error' : 'log',
                        `[toast:${input.level}]`,
                        input.message
                    );
                    return;
                }
                void bridge.call('toast', input, 5000).catch(() => undefined);
            },
        },
        tab: {
            setActiveTask(task: { name: string } | null) {
                if (!hasHostFrame) return;
                void bridge.call('tab', task, 5000).catch(() => undefined);
            },
        },
        theme: {
            // Cached sync read (issue #77): seeded from the module document,
            // corrected by the background host pull above, kept fresh by
            // host push events.
            get: () => themeState.get(),
            subscribe(listener) {
                return themeState.subscribe(listener);
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
