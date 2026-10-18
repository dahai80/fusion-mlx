import type { Config } from 'tailwindcss';

/**
 * Tailwind preset for all fusion web UIs (spec §3/§4.2).
 * md (768px) is the desktop/H5 boundary.
 * Spacing stays Tailwind's 4px-multiple scale.
 */
export const fusionPreset: Partial<Config> = {
    darkMode: ['class', '.light'],
    theme: {
        extend: {
            colors: {
                bg: 'var(--fs-bg)',
                'bg-elevated': 'var(--fs-bg-elevated)',
                'bg-muted': 'var(--fs-bg-muted)',
                border: 'var(--fs-border)',
                'text-primary': 'var(--fs-text-primary)',
                'text-secondary': 'var(--fs-text-secondary)',
                'text-muted': 'var(--fs-text-muted)',
                success: 'var(--fs-success)',
                running: 'var(--fs-running)',
                warning: 'var(--fs-warning)',
                error: 'var(--fs-error)',
                offline: 'var(--fs-offline)',
                brand: 'var(--fs-brand)',
            },
            borderRadius: {
                card: '0.5rem',
            },
            boxShadow: {
                overlay: '0 8px 24px rgb(0 0 0 / 0.35)',
            },
        },
    },
};

export default fusionPreset;
