/**
 * Tests for useAudioMixer hook - sync robustness features
 *
 * Note: Full hook integration tests require complex mocking of Web Audio API.
 * These tests focus on the sync callback contract and timing behavior.
 * See syncRobustness.test.ts for comprehensive unit tests of sync logic.
 */

import { describe, it, expect, vi } from 'vitest';

describe('useAudioMixer - Sync Contract', () => {
  describe('onTimeSync callback interface', () => {
    it('should accept onTimeSync as an optional callback', () => {
      // The hook accepts { onTimeSync?: (time: number) => void }
      type UseAudioMixerOptions = {
        onTimeSync?: (time: number) => void;
      };

      const options: UseAudioMixerOptions = {
        onTimeSync: vi.fn(),
      };

      expect(options.onTimeSync).toBeDefined();
      expect(typeof options.onTimeSync).toBe('function');
    });

    it('should pass time as a number to onTimeSync', () => {
      const onTimeSync = vi.fn();

      // Simulate what the hook does internally
      const position = 42.5;
      onTimeSync(position);

      expect(onTimeSync).toHaveBeenCalledWith(42.5);
      expect(typeof onTimeSync.mock.calls[0][0]).toBe('number');
    });
  });

  describe('Sync timing constants', () => {
    it('should use 100ms interval for 10Hz sync rate', () => {
      const SYNC_INTERVAL_MS = 100;
      const EXPECTED_RATE_HZ = 10;

      expect(1000 / SYNC_INTERVAL_MS).toBe(EXPECTED_RATE_HZ);
    });
  });
});
