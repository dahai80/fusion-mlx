import { createShellSdk } from "@fusion/web-sdk";

// Single SDK instance for the module (toast/tab/theme bridge to the shell
// host). Data calls use raw same-origin fetch in api.ts — embedded mode the
// gateway injects the module key upstream, standalone dev attaches
// VITE_API_KEY (dev convenience only) as x-api-key there.
const DEV_API_KEY: string | undefined = import.meta.env.VITE_API_KEY;

export { DEV_API_KEY };

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

// #1298: a shell deep link opened at top level has no parent frame, so the
// iframe check alone reports standalone. The ?embedded=1 flag is
// authoritative for chrome and host APIs; auth stays keyed on the served
// topology (see api.ts authHeaders) so a flagged link outside /shell/ still
// authenticates.
export const isEmbedded = (): boolean =>
    typeof window !== "undefined" &&
    (window.parent !== window || embeddedParam());

export const sdk = createShellSdk({ embedded: isEmbedded() });
