import { useEffect, useState } from "react";
import { sdk } from "./sdk";

// Spec S9 (fusion-web-ui-spec-1015 §9.1): module copy follows the host
// language. Lang state binds to the SDK i18n channel — the shell pushes
// 'lang' events to embedded iframes; standalone falls back to <html lang>.
export function useLang(): "zh" | "en" {
    const [lang, setLang] = useState<"zh" | "en">(sdk.i18n.lang());
    useEffect(() => sdk.i18n.subscribe((next) => setLang(next)), []);
    return lang;
}

export function useT() {
    const lang = useLang();
    return (zh: string, en: string) => (lang === "en" ? en : zh);
}
