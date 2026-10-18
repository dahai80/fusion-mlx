import { createShellSdk } from "@fusion/web-sdk";

// Single SDK instance for the module (toast/tab/theme bridge to the shell
// host). Data calls use raw same-origin fetch in api.ts — embedded mode the
// gateway injects the module key upstream, standalone dev attaches
// VITE_API_KEY (dev convenience only) as x-api-key there.
const DEV_API_KEY: string | undefined = import.meta.env.VITE_API_KEY;

export { DEV_API_KEY };

export const isEmbedded = (): boolean =>
    typeof window !== "undefined" && window.parent !== window;

// ?embedded=1 is added by the shell; standalone keeps the module's own nav.
export const embeddedParam = (): boolean => {
    try {
        return (
            new URLSearchParams(window.location.search).get("embedded") === "1"
        );
    } catch {
        return false;
    }
};

export const sdk = createShellSdk();
