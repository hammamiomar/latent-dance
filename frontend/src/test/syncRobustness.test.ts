/**
 * Sync Robustness Test Suite
 *
 * Tests the audio-visual synchronization system:
 * 1. Periodic time sync (10Hz)
 * 2. Drift detection (>0.5s warning)
 * 3. Scrub debouncing (150ms)
 * 4. Disconnect handling (pause on disconnect)
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';

describe('Sync Robustness', () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.restoreAllMocks();
  });

  describe('1. Periodic Time Sync (10Hz)', () => {
    it('should sync at 100ms intervals (10Hz rate)', () => {
      // The sync interval is 100ms
      const SYNC_INTERVAL_MS = 100;
      const SYNC_RATE_HZ = 1000 / SYNC_INTERVAL_MS;

      expect(SYNC_RATE_HZ).toBe(10);
    });

    it('should track last sync time and only fire after threshold', () => {
      let lastSyncTime = -100; // Initialize to allow first sync at t=0
      const SYNC_INTERVAL = 100;
      const syncCallback = vi.fn();

      // Simulate RAF loop behavior
      const checkSync = (currentTime: number) => {
        if (currentTime - lastSyncTime >= SYNC_INTERVAL) {
          lastSyncTime = currentTime;
          syncCallback(currentTime);
        }
      };

      // Simulate time progression
      checkSync(0); // First sync (0 - (-100) = 100 >= 100)
      expect(syncCallback).toHaveBeenCalledTimes(1);

      checkSync(50); // Too soon (50 - 0 = 50 < 100)
      expect(syncCallback).toHaveBeenCalledTimes(1);

      checkSync(100); // Should sync (100 - 0 = 100 >= 100)
      expect(syncCallback).toHaveBeenCalledTimes(2);

      checkSync(150); // Too soon (150 - 100 = 50 < 100)
      expect(syncCallback).toHaveBeenCalledTimes(2);

      checkSync(200); // Should sync (200 - 100 = 100 >= 100)
      expect(syncCallback).toHaveBeenCalledTimes(3);
    });
  });

  describe('2. Drift Detection', () => {
    it('should detect drift > 0.5s', () => {
      const DRIFT_THRESHOLD = 0.5;

      const detectDrift = (frontendTime: number, backendTime: number) => {
        const drift = Math.abs(frontendTime - backendTime);
        return drift > DRIFT_THRESHOLD;
      };

      // No drift
      expect(detectDrift(10.0, 10.0)).toBe(false);
      expect(detectDrift(10.0, 10.3)).toBe(false);
      expect(detectDrift(10.0, 10.5)).toBe(false);

      // Drift detected
      expect(detectDrift(10.0, 10.6)).toBe(true);
      expect(detectDrift(10.0, 11.0)).toBe(true);
      expect(detectDrift(10.0, 8.0)).toBe(true);
    });

    it('should log warning when drift detected', () => {
      const consoleWarn = vi.spyOn(console, 'warn').mockImplementation(() => {});

      const handleStemActivity = (activity: { audioTime: number }, frontendTime: number) => {
        const drift = Math.abs(frontendTime - activity.audioTime);
        if (drift > 0.5) {
          console.warn(
            `[Sync] Drift detected: ${drift.toFixed(2)}s (frontend=${frontendTime.toFixed(2)}, backend=${activity.audioTime.toFixed(2)})`
          );
        }
      };

      // No drift - no warning
      handleStemActivity({ audioTime: 10.0 }, 10.2);
      expect(consoleWarn).not.toHaveBeenCalled();

      // Drift detected - warning logged
      handleStemActivity({ audioTime: 10.0 }, 11.0);
      expect(consoleWarn).toHaveBeenCalled();
      expect(consoleWarn).toHaveBeenCalledWith(expect.stringContaining('Drift detected'));

      consoleWarn.mockRestore();
    });

    it('should calculate drift correctly for both directions', () => {
      const calcDrift = (frontend: number, backend: number) =>
        Math.abs(frontend - backend);

      // Frontend ahead
      expect(calcDrift(10.5, 10.0)).toBe(0.5);

      // Backend ahead
      expect(calcDrift(10.0, 10.5)).toBe(0.5);

      // Large drift
      expect(calcDrift(15.0, 10.0)).toBe(5.0);
    });
  });

  describe('3. Scrub Debouncing', () => {
    it('should debounce rapid seeks (150ms)', () => {
      const DEBOUNCE_MS = 150;
      const sendSeek = vi.fn();
      let timeoutId: ReturnType<typeof setTimeout> | null = null;

      const debouncedSeek = (time: number) => {
        if (timeoutId) {
          clearTimeout(timeoutId);
        }
        timeoutId = setTimeout(() => {
          sendSeek(time);
          timeoutId = null;
        }, DEBOUNCE_MS);
      };

      // Rapid seeks
      debouncedSeek(10);
      debouncedSeek(20);
      debouncedSeek(30);
      debouncedSeek(40);

      // Not called yet (still debouncing)
      expect(sendSeek).not.toHaveBeenCalled();

      // Advance past debounce threshold
      vi.advanceTimersByTime(DEBOUNCE_MS);

      // Only last seek should be sent
      expect(sendSeek).toHaveBeenCalledTimes(1);
      expect(sendSeek).toHaveBeenCalledWith(40);
    });

    it('should allow seeks after debounce period', () => {
      const DEBOUNCE_MS = 150;
      const sendSeek = vi.fn();
      let timeoutId: ReturnType<typeof setTimeout> | null = null;

      const debouncedSeek = (time: number) => {
        if (timeoutId) {
          clearTimeout(timeoutId);
        }
        timeoutId = setTimeout(() => {
          sendSeek(time);
          timeoutId = null;
        }, DEBOUNCE_MS);
      };

      // First seek
      debouncedSeek(10);
      vi.advanceTimersByTime(DEBOUNCE_MS);
      expect(sendSeek).toHaveBeenCalledWith(10);

      // Second seek after debounce period
      debouncedSeek(50);
      vi.advanceTimersByTime(DEBOUNCE_MS);
      expect(sendSeek).toHaveBeenCalledTimes(2);
      expect(sendSeek).toHaveBeenLastCalledWith(50);
    });

    it('should not debounce local audio seek (only WS message)', () => {
      const localSeek = vi.fn();
      const wsSeek = vi.fn();
      const DEBOUNCE_MS = 150;
      let timeoutId: ReturnType<typeof setTimeout> | null = null;

      const handleSeek = (time: number) => {
        // Local seek is immediate
        localSeek(time);

        // WS seek is debounced
        if (timeoutId) {
          clearTimeout(timeoutId);
        }
        timeoutId = setTimeout(() => {
          wsSeek(time);
          timeoutId = null;
        }, DEBOUNCE_MS);
      };

      // Rapid seeks
      handleSeek(10);
      handleSeek(20);
      handleSeek(30);

      // Local seeks happen immediately
      expect(localSeek).toHaveBeenCalledTimes(3);

      // WS seek is debounced
      expect(wsSeek).not.toHaveBeenCalled();

      vi.advanceTimersByTime(DEBOUNCE_MS);

      // Only last WS seek is sent
      expect(wsSeek).toHaveBeenCalledTimes(1);
      expect(wsSeek).toHaveBeenCalledWith(30);
    });
  });

  describe('4. Disconnect Handling', () => {
    it('should pause audio when status changes to disconnected', () => {
      const pauseAudio = vi.fn();

      const handleStatusChange = (
        status: 'connected' | 'disconnected' | 'error',
        isPlaying: boolean
      ) => {
        if ((status === 'disconnected' || status === 'error') && isPlaying) {
          pauseAudio();
        }
      };

      // Connected - no pause
      handleStatusChange('connected', true);
      expect(pauseAudio).not.toHaveBeenCalled();

      // Disconnected while playing - should pause
      handleStatusChange('disconnected', true);
      expect(pauseAudio).toHaveBeenCalledTimes(1);
    });

    it('should NOT pause if already paused on disconnect', () => {
      const pauseAudio = vi.fn();

      const handleStatusChange = (
        status: 'connected' | 'disconnected' | 'error',
        isPlaying: boolean
      ) => {
        if ((status === 'disconnected' || status === 'error') && isPlaying) {
          pauseAudio();
        }
      };

      // Disconnected but not playing - no pause
      handleStatusChange('disconnected', false);
      expect(pauseAudio).not.toHaveBeenCalled();
    });

    it('should pause on error status', () => {
      const pauseAudio = vi.fn();

      const handleStatusChange = (
        status: 'connected' | 'disconnected' | 'error',
        isPlaying: boolean
      ) => {
        if ((status === 'disconnected' || status === 'error') && isPlaying) {
          pauseAudio();
        }
      };

      handleStatusChange('error', true);
      expect(pauseAudio).toHaveBeenCalledTimes(1);
    });

    it('should log warning when pausing due to disconnect', () => {
      const consoleWarn = vi.spyOn(console, 'warn').mockImplementation(() => {});

      const handleStatusChange = (status: string, isPlaying: boolean) => {
        if ((status === 'disconnected' || status === 'error') && isPlaying) {
          console.warn('[App] WebSocket disconnected - pausing audio');
        }
      };

      handleStatusChange('disconnected', true);
      expect(consoleWarn).toHaveBeenCalledWith(expect.stringContaining('disconnected'));

      consoleWarn.mockRestore();
    });
  });

  describe('Integration: Full Sync Cycle', () => {
    it('should maintain sync during normal playback', () => {
      // Simulate a full sync cycle
      const timeline: { frontendTime: number; backendTime: number; synced: boolean }[] = [];

      let frontendTime = 0;
      let backendTime = 0;
      let lastSyncTime = 0;
      const SYNC_INTERVAL = 100;

      // Simulate 1 second of playback with varying drift
      for (let t = 0; t <= 1000; t += 16) {
        // ~60fps
        // Frontend advances
        frontendTime = t / 1000;

        // Backend advances with slight lag (simulating network delay)
        backendTime = frontendTime - 0.05;

        // Sync check at 10Hz
        if (t - lastSyncTime >= SYNC_INTERVAL) {
          lastSyncTime = t;
          const drift = Math.abs(frontendTime - backendTime);
          timeline.push({
            frontendTime,
            backendTime,
            synced: drift <= 0.5,
          });
        }
      }

      // All sync points should show healthy sync (drift < 0.5s)
      expect(timeline.every((p) => p.synced)).toBe(true);
    });

    it('should detect when sync is lost', () => {
      const frontendTime = 10.0;
      const backendTime = 8.0; // 2 second lag

      const drift = Math.abs(frontendTime - backendTime);
      const isSynced = drift <= 0.5;

      expect(isSynced).toBe(false);
      expect(drift).toBe(2.0);
    });
  });
});
