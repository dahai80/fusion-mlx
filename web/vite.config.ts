import react from '@vitejs/plugin-react';
import { defineConfig } from 'vite';

// #1220: dev server 11471 (architecture/port-registry.yaml). Proxies the
// fusion-mlx API surface to the engine on 11434 so standalone mode mirrors
// the embedded gateway /shell/fusion-mlx/ layout (same-origin, no CORS).
// FUSION_DEV_BACKEND / FUSION_MLX_WEB_PORT override for parity.
const backend = process.env.FUSION_DEV_BACKEND ?? 'http://127.0.0.1:11434';
const port = Number(process.env.FUSION_MLX_WEB_PORT ?? 11471);

const apiPaths = ['/v1', '/admin', '/metrics', '/health', '/stats'];

export default defineConfig({
    plugins: [react()],
    // Embedded deployment serves dist through the gateway sub-path
    // /shell/fusion-mlx/ — absolute /assets would 404 there. Relative base
    // + hash router (#/...) resolves assets against the module root in both
    // embedded and standalone modes.
    base: './',
    server: {
        port,
        strictPort: true,
        proxy: Object.fromEntries(
            apiPaths.map((p) => [p, { target: backend, changeOrigin: true }])
        ),
    },
    preview: {
        port,
        strictPort: true,
        proxy: Object.fromEntries(
            apiPaths.map((p) => [p, { target: backend, changeOrigin: true }])
        ),
    },
    envPrefix: ['VITE_', 'FUSION_'],
});
