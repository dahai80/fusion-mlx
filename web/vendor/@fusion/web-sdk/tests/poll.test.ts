import { describe, expect, it, mock } from 'bun:test';
import { createPoller } from '../src/poll';

describe('createPoller', () => {
    it('stops cleanly and runs no further ticks', async () => {
        let ticks = 0;
        const poller = createPoller({
            intervalMs: 1,
            hiddenIntervalMs: 1,
            onTick: async () => {
                ticks++;
            },
        });
        poller.start();
        await Bun.sleep(30);
        poller.stop();
        const atStop = ticks;
        expect(atStop).toBeGreaterThan(0);
        await Bun.sleep(20);
        expect(ticks).toBe(atStop);
    });

    it('counts consecutive failures and calls onError', async () => {
        const onError = mock(() => undefined);
        const poller = createPoller({
            intervalMs: 1,
            hiddenIntervalMs: 1,
            failureThreshold: 2,
            onTick: async () => {
                throw new Error('boom');
            },
            onError,
        });
        poller.start();
        await Bun.sleep(30);
        poller.stop();
        expect(onError).toHaveBeenCalled();
    });
});
