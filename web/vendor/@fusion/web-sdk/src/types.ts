export type DeviceMode = 'desktop' | 'mobile-ready' | 'mobile-readonly' | 'desktop-only';

export type ToastLevel = 'info' | 'success' | 'warning' | 'error';

export type ThemeMode = 'dark' | 'light';

export interface HttpRequestInit {
    method?: string;
    headers?: Record<string, string>;
    body?: unknown;
    timeoutMs?: number;
}

export interface HttpResult<T = unknown> {
    status: number;
    ok: boolean;
    data: T;
}

export interface ToastInput {
    level: ToastLevel;
    message: string;
    durationMs?: number;
}

export interface DeviceInfo {
    viewport: 'desktop' | 'mobile';
    mode: DeviceMode | null;
}

export interface ShellSdkOptions {
    /** Allowed parent origin. Default: same origin as the module document. */
    origin?: string;
    /** Request timeout in ms. Default 15000. */
    timeoutMs?: number;
    /** Retry attempts for failed bridge requests. Default 2. */
    retries?: number;
}

export interface FusionShellSdk {
    /** Relay an API call through the host, which attaches the Bearer token. */
    http: {
        request<T = unknown>(path: string, init?: HttpRequestInit): Promise<HttpResult<T>>;
        get<T = unknown>(path: string): Promise<HttpResult<T>>;
        post<T = unknown>(path: string, body?: unknown): Promise<HttpResult<T>>;
        put<T = unknown>(path: string, body?: unknown): Promise<HttpResult<T>>;
        delete<T = unknown>(path: string): Promise<HttpResult<T>>;
    };
    toast: {
        show(input: ToastInput): void;
    };
    tab: {
        /** Declare an active long-running task; host blocks tab close while set. */
        setActiveTask(task: { name: string } | null): void;
    };
    theme: {
        get(): ThemeMode;
        subscribe(listener: (mode: ThemeMode) => void): () => void;
    };
    devices: {
        get(): DeviceInfo;
    };
    visibility: {
        get(): boolean;
        subscribe(listener: (visible: boolean) => void): () => void;
    };
    /** Host language awareness (spec S9): t() picks zh/en copy. */
    i18n: {
        lang(): 'zh' | 'en';
        t(zh: string, en: string): string;
        subscribe(listener: (lang: 'zh' | 'en') => void): () => void;
    };
}

/** Messages sent from module iframe to host. */
export interface ModuleRequest {
    type: 'request';
    id: string;
    method: 'http' | 'toast' | 'tab' | 'theme' | 'devices';
    payload: unknown;
}

/** Host-to-module request (currently heartbeat ping). */
export interface HostRequest {
    type: 'request';
    id: string;
    method: 'ping';
    payload: null;
}

/** Messages sent from host to module. */
export interface HostResponse {
    type: 'response';
    id: string;
    ok: boolean;
    result?: unknown;
    error?: { code: string; message: string };
}

export interface HostEvent {
    type: 'event';
    event: 'theme' | 'visibility' | 'lang';
    payload: unknown;
}

export type HostMessage = HostResponse | HostEvent;
export type BridgeMessage = ModuleRequest | HostRequest | HostMessage;
