import { HttpError } from './retry';
import type { DeviceInfo, HostEvent, HostResponse, ModuleRequest, ThemeMode } from './types';

export interface HostBridgeOptions {
    /** Allowed module origin. Default: same origin as the host document. */
    origin?: string;
    /** Token attached to proxied http calls. */
    getToken(): string | null;
    onToast(input: { level: string; message: string; durationMs?: number }): void;
    onActiveTask(tabId: string, task: { name: string } | null): void;
    /** Gateway-proxied fetch used for http relay; host attaches Bearer here. */
    proxyHttp(req: {
        path: string;
        method: string;
        headers?: Record<string, string>;
        body?: unknown;
    }): Promise<{ status: number; data: unknown }>;
}

/** Issue #75: handleRequest returns bare strings ('theme'), plain objects
 * (http results, devices) and __httpError envelopes — only the last kind is
 * an error. `in` on a non-object throws, so guard the type first. */
export function isHttpErrorResult(
    result: unknown
): result is { __httpError: true; status: number; code: string; message: string } {
    return result !== null && typeof result === 'object' && '__httpError' in result;
}

export interface HostBridgeHandle {
    iframeWindow: Window | null;
    destroy(): void;
    notifyTheme(mode: ThemeMode): void;
    notifyVisibility(visible: boolean): void;
    notifyLang(lang: string): void;
    /** True when the module answered at least one heartbeat (SDK integrated). */
    readonly sdkAlive: boolean;
    startHeartbeat(
        onLivenessChange: (alive: boolean) => void,
        intervalMs?: number,
        timeoutMs?: number
    ): void;
    stopHeartbeat(): void;
}

/**
 * Host-side bridge for one module iframe. Validates source/origin on every
 * message; iframe never touches host state directly.
 */
export function createHostBridge(
    tabId: string,
    iframeWindow: Window | null,
    options: HostBridgeOptions
): HostBridgeHandle {
    const origin = options.origin ?? window.location.origin;
    const disposers: Array<() => void> = [];
    let currentIframeWindow = iframeWindow;

    const pendingPings = new Set<(msgId: string) => void>();
    let sdkAlive = false;
    let pingSeq = 0;
    let heartbeatTimer: ReturnType<typeof setInterval> | null = null;

    /**
     * Heartbeat (PRD 7.2): host pings; module SDK auto-answers. Enforcement
     * (crash) only after first pong, so non-SDK modules never false-crash.
     */
    function startHeartbeat(
        onLivenessChange: (alive: boolean) => void,
        intervalMs = 30000,
        timeoutMs = 10000
    ): void {
        stopHeartbeat();
        heartbeatTimer = setInterval(() => {
            if (!currentIframeWindow) return;
            const id = `ping-${tabId}-${++pingSeq}`;
            let answered = false;
            const resolver = (msgId: string) => {
                if (msgId === id) {
                    answered = true;
                }
            };
            pendingPings.add(resolver);
            currentIframeWindow.postMessage(
                { type: 'request', id, method: 'ping', payload: null },
                origin
            );
            setTimeout(() => {
                // Drop the dead closure so long-lived tabs don't leak (review #8).
                pendingPings.delete(resolver);
                if (answered) {
                    onLivenessChange(true);
                } else if (sdkAlive) {
                    // Enforcement only once the module has answered before;
                    // non-SDK modules stay silent forever and never false-crash.
                    console.warn('[host-bridge] heartbeat missed for', tabId);
                    onLivenessChange(false);
                }
            }, timeoutMs);
        }, intervalMs);
    }

    function stopHeartbeat(): void {
        if (heartbeatTimer !== null) {
            clearInterval(heartbeatTimer);
            heartbeatTimer = null;
        }
    }

    async function handleRequest(msg: ModuleRequest): Promise<unknown> {
        switch (msg.method) {
            case 'http': {
                const req = msg.payload as {
                    path: string;
                    method?: string;
                    headers?: Record<string, string>;
                    body?: unknown;
                };
                try {
                    const { status, data } = await options.proxyHttp({
                        path: req.path,
                        method: req.method ?? 'GET',
                        headers: req.headers,
                        body: req.body,
                    });
                    if (status >= 400) {
                        const envelope = data as { code?: string; message?: string } | null;
                        return {
                            __httpError: true,
                            status,
                            code: envelope?.code ?? 'http_error',
                            message: envelope?.message ?? `status ${status}`,
                        };
                    }
                    return { status, ok: status < 400, data };
                } catch (err) {
                    console.error('[host-bridge] http proxy failed:', err);
                    throw new HttpError(502, 'gateway_unreachable', 'gateway request failed');
                }
            }
            case 'toast':
                options.onToast(
                    msg.payload as { level: string; message: string; durationMs?: number }
                );
                return { ok: true };
            case 'tab':
                options.onActiveTask(tabId, (msg.payload as { name: string } | null) ?? null);
                return { ok: true };
            case 'theme': {
                const mode: ThemeMode = document.documentElement.classList.contains('light')
                    ? 'light'
                    : 'dark';
                return mode;
            }
            case 'devices': {
                const mobile = window.matchMedia('(max-width: 767px)').matches;
                const info: DeviceInfo = { viewport: mobile ? 'mobile' : 'desktop', mode: null };
                return info;
            }
            default:
                throw new Error(`unknown bridge method: ${(msg as ModuleRequest).method}`);
        }
    }

    function onMessage(ev: MessageEvent): void {
        if (ev.source !== currentIframeWindow || ev.origin !== origin) return;
        const raw = ev.data as { type?: string; id?: string };
        if (raw.type === 'response') {
            // Module answering a host-initiated request (heartbeat pong).
            sdkAlive = true;
            if (typeof raw.id === 'string') pendingPings.forEach((fn) => fn(raw.id as string));
            return;
        }
        const msg = ev.data as ModuleRequest;
        if (!raw || raw.type !== 'request') return;
        if (typeof raw.id !== 'string' || typeof msg.method !== 'string') return;
        void handleRequest(msg).then(
            (result) => {
                if (isHttpErrorResult(result)) {
                    const e = result as { status: number; code: string; message: string };
                    respond({
                        type: 'response',
                        id: msg.id,
                        ok: false,
                        error: { code: e.code, message: e.message },
                    });
                } else {
                    respond({ type: 'response', id: msg.id, ok: true, result });
                }
            },
            (err: Error) => {
                respond({
                    type: 'response',
                    id: msg.id,
                    ok: false,
                    error: { code: err.name, message: err.message },
                });
            }
        );
    }

    function respond(msg: HostResponse): void {
        if (!currentIframeWindow) return;
        currentIframeWindow.postMessage(msg, origin);
    }

    function notify(event: HostEvent): void {
        if (!currentIframeWindow) return;
        currentIframeWindow.postMessage(event, origin);
    }

    window.addEventListener('message', onMessage);
    disposers.push(() => window.removeEventListener('message', onMessage));

    return {
        get iframeWindow() {
            return currentIframeWindow;
        },
        set iframeWindow(w: Window | null) {
            currentIframeWindow = w;
        },
        destroy() {
            disposers.forEach((d) => d());
            stopHeartbeat();
            currentIframeWindow = null;
        },
        notifyTheme(mode: ThemeMode) {
            notify({ type: 'event', event: 'theme', payload: mode });
        },
        notifyVisibility(visible: boolean) {
            notify({ type: 'event', event: 'visibility', payload: visible });
        },
        notifyLang(lang: string) {
            notify({ type: 'event', event: 'lang', payload: lang });
        },
        get sdkAlive() {
            return sdkAlive;
        },
        startHeartbeat,
        stopHeartbeat,
    };
}
