import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { App } from "./App";
import "@fusion/web-ui/tokens.css";
import "./index.css";

const rootEl = document.getElementById("root");
if (rootEl) {
    createRoot(rootEl).render(
        <StrictMode>
            <App />
        </StrictMode>,
    );
} else {
    console.error("[web] #root element missing");
}
