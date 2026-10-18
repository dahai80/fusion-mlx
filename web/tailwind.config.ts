import { fusionPreset } from '@fusion/web-ui/preset';
import type { Config } from 'tailwindcss';

// NOTE: `presets` (plural) is the Tailwind key; a singular `preset` key is
// silently dropped and the token colors never compile.
export default {
    presets: [fusionPreset as Config],
    content: ['./index.html', './src/**/*.{ts,tsx}'],
} satisfies Config;
