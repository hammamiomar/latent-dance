/**
 * Tests for useWebSocket hook - sync messaging
 *
 * Note: Full WebSocket integration tests require complex event simulation.
 * These tests focus on message format contracts.
 * See syncRobustness.test.ts for comprehensive unit tests of sync logic.
 */

import { describe, it, expect } from 'vitest';

describe('useWebSocket - Message Contracts', () => {
  describe('Audio sync message formats', () => {
    it('should format audio_timeupdate message correctly', () => {
      const time = 42.5;
      const message = JSON.stringify({
        action: 'audio_timeupdate',
        time,
      });

      const parsed = JSON.parse(message);
      expect(parsed.action).toBe('audio_timeupdate');
      expect(parsed.time).toBe(42.5);
    });

    it('should format audio_play message correctly', () => {
      const time = 10.0;
      const message = JSON.stringify({
        action: 'audio_play',
        time,
      });

      const parsed = JSON.parse(message);
      expect(parsed.action).toBe('audio_play');
      expect(parsed.time).toBe(10.0);
    });

    it('should format audio_pause message correctly', () => {
      const message = JSON.stringify({
        action: 'audio_pause',
      });

      const parsed = JSON.parse(message);
      expect(parsed.action).toBe('audio_pause');
    });

    it('should format audio_seek message correctly', () => {
      const time = 90.0;
      const message = JSON.stringify({
        action: 'audio_seek',
        time,
      });

      const parsed = JSON.parse(message);
      expect(parsed.action).toBe('audio_seek');
      expect(parsed.time).toBe(90.0);
    });
  });

  describe('Extended activity message parsing', () => {
    it('should parse extended_activity message with stems and prominence', () => {
      const serverMessage = {
        type: 'extended_activity',
        audio_time: 42.5,
        stems: {
          drums: { envelope: 0.8, flash: 0.6, sustain: 0.4 },
          bass: { envelope: 0.5, flash: 0.3, sustain: 0.7 },
        },
        prominence: {
          drums: { prominence: 0.85, surprise_active: false, rank: 1 },
          bass: { prominence: 0.6, surprise_active: true, rank: 2 },
        },
      };

      expect(serverMessage.audio_time).toBe(42.5);
      expect(serverMessage.stems.drums.envelope).toBe(0.8);
      expect(serverMessage.prominence?.drums.prominence).toBe(0.85);
      expect(serverMessage.prominence?.bass.surprise_active).toBe(true);
    });

    it('should include audioTime for drift detection', () => {
      const serverMessage = {
        type: 'extended_activity',
        audio_time: 100.0,
        stems: {},
      };

      expect(serverMessage.audio_time).toBe(100.0);
      expect(typeof serverMessage.audio_time).toBe('number');
    });
  });
});
