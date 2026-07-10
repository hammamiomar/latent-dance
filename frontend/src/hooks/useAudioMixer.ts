/**
 * useAudioMixer - Web Audio API stem mixer hook
 *
 * Loads all 4 stems as AudioBuffers and routes through GainNodes
 * for real-time volume control, mute, and solo functionality.
 *
 * Architecture:
 *   [Stem AudioBuffer] → [GainNode] → [MasterGain] → [Destination]
 *
 * Synchronized playback is achieved by starting all stems at the same
 * AudioContext time. Seek is handled by stopping and restarting.
 */

import { useEffect, useRef, useCallback } from 'react';
import { useShallow } from 'zustand/react/shallow';
import { useAudioStore, getEffectiveStemVolume } from '../stores/useAudioStore';
import type { PhysicalStem } from '../types/sae';

const STEMS: PhysicalStem[] = ['bass', 'drums', 'vocals', 'other'];

interface StemNodes {
  buffer: AudioBuffer | null;
  source: AudioBufferSourceNode | null;
  gain: GainNode;
}

interface UseAudioMixerOptions {
  /** Callback for periodic time sync (called at ~10Hz during playback) */
  onTimeSync?: (time: number) => void;
}

interface UseAudioMixerReturn {
  load: (audioId: string) => Promise<void>;
  play: (startTime?: number) => void;
  pause: () => void;
  seek: (time: number) => void;
  stop: () => void;
}

export function useAudioMixer(options: UseAudioMixerOptions = {}): UseAudioMixerReturn {
  const { onTimeSync } = options;

  // AudioContext and nodes
  const audioContextRef = useRef<AudioContext | null>(null);
  const masterGainRef = useRef<GainNode | null>(null);
  const stemNodesRef = useRef<Record<PhysicalStem, StemNodes> | null>(null);

  // Playback state (local tracking for timing)
  const playStartTimeRef = useRef<number>(0); // AudioContext time when playback started
  const playOffsetRef = useRef<number>(0); // Position in audio when playback started
  const isPlayingRef = useRef<boolean>(false);

  // Concurrent load guard — prevent double-fetching the same stems
  const loadingAudioIdRef = useRef<string | null>(null);
  const loadedAudioIdRef = useRef<string | null>(null);

  // Animation frame for time updates
  const rafIdRef = useRef<number | null>(null);

  // Time sync + store update (~10Hz = every 100ms, smooth enough for progress bar)
  const lastSyncTimeRef = useRef<number>(0);
  const TIME_SYNC_INTERVAL_MS = 100;
  const onTimeSyncRef = useRef(onTimeSync);
  onTimeSyncRef.current = onTimeSync;

  // Store state (selective subscription — avoids re-render on currentTime changes)
  const {
    audioId,
    isPlaying,
    stemVolumes,
    stemMuted,
    stemSolo,
    masterVolume,
    setCurrentTime,
    setIsPlaying,
    duration,
  } = useAudioStore(useShallow((s) => ({
    audioId: s.audioId,
    isPlaying: s.isPlaying,
    stemVolumes: s.stemVolumes,
    stemMuted: s.stemMuted,
    stemSolo: s.stemSolo,
    masterVolume: s.masterVolume,
    setCurrentTime: s.setCurrentTime,
    setIsPlaying: s.setIsPlaying,
    duration: s.duration,
  })));

  // ============================================================================
  // Initialize AudioContext
  // ============================================================================

  const getAudioContext = useCallback(() => {
    if (!audioContextRef.current) {
      audioContextRef.current = new AudioContext();
      masterGainRef.current = audioContextRef.current.createGain();
      masterGainRef.current.connect(audioContextRef.current.destination);
    }
    return audioContextRef.current;
  }, []);

  // ============================================================================
  // Load stems from backend
  // ============================================================================

  const load = useCallback(async (audioIdToLoad: string) => {
    // Concurrent load guard — skip if already loading or loaded for this audioId
    if (loadingAudioIdRef.current === audioIdToLoad) return;
    if (loadedAudioIdRef.current === audioIdToLoad) return;

    loadingAudioIdRef.current = audioIdToLoad;
    useAudioStore.getState().setUploadPhase('loading_stems');

    const ctx = getAudioContext();

    // Resume if suspended (browser autoplay policy)
    if (ctx.state === 'suspended') {
      await ctx.resume();
    }

    // Initialize stem nodes if needed
    if (!stemNodesRef.current) {
      stemNodesRef.current = {} as Record<PhysicalStem, StemNodes>;
      for (const stem of STEMS) {
        const gain = ctx.createGain();
        gain.connect(masterGainRef.current!);
        stemNodesRef.current[stem] = {
          buffer: null,
          source: null,
          gain,
        };
      }
    }

    // Load all stems in parallel
    const loadPromises = STEMS.map(async (stem) => {
      const url = `/api/audio/${audioIdToLoad}/stem/${stem}`;
      try {
        const response = await fetch(url);
        if (!response.ok) {
          throw new Error(`Failed to fetch ${stem}: ${response.status}`);
        }
        const arrayBuffer = await response.arrayBuffer();
        const audioBuffer = await ctx.decodeAudioData(arrayBuffer);
        stemNodesRef.current![stem].buffer = audioBuffer;
      } catch (err) {
        console.error(`Failed to load stem ${stem}:`, err);
        // Continue with other stems
      }
    });

    await Promise.all(loadPromises);

    // Staleness check: if a newer load started while we were fetching, bail
    if (loadingAudioIdRef.current !== audioIdToLoad) {
      return;
    }

    // Check if any stems actually loaded — total failure = error, not silent "ready"
    const loadedCount = STEMS.filter(
      (stem) => stemNodesRef.current![stem].buffer !== null
    ).length;

    loadingAudioIdRef.current = null;

    if (loadedCount === 0) {
      // Don't set loadedAudioIdRef — allows retry on next attempt
      useAudioStore.getState().setUploadError('Failed to load audio stems. Try again.');
      console.error('[AudioMixer] All stem fetches failed for', audioIdToLoad);
      return;
    }

    loadedAudioIdRef.current = audioIdToLoad;
    useAudioStore.getState().setUploadPhase('ready');
  }, [getAudioContext]);

  // ============================================================================
  // Update gain nodes when volumes change
  // ============================================================================

  useEffect(() => {
    if (!stemNodesRef.current) return;

    const state = { stemVolumes, stemMuted, stemSolo, masterVolume };

    for (const stem of STEMS) {
      const effectiveVolume = getEffectiveStemVolume(stem, state);
      const gain = stemNodesRef.current[stem].gain;
      // Smooth transition
      gain.gain.setTargetAtTime(effectiveVolume, getAudioContext().currentTime, 0.05);
    }
  }, [stemVolumes, stemMuted, stemSolo, masterVolume, getAudioContext]);

  // ============================================================================
  // Stop playback (internal)
  // ============================================================================

  const stopInternal = useCallback(() => {
    if (!stemNodesRef.current) return;

    for (const stem of STEMS) {
      const node = stemNodesRef.current[stem];
      if (node.source) {
        try {
          node.source.stop();
          node.source.disconnect();
        } catch {
          // Ignore
        }
        node.source = null;
      }
    }

    isPlayingRef.current = false;

    if (rafIdRef.current) {
      cancelAnimationFrame(rafIdRef.current);
      rafIdRef.current = null;
    }
  }, []);

  // ============================================================================
  // Start playback
  // ============================================================================

  const play = useCallback((startTime = 0) => {
    if (!stemNodesRef.current || !masterGainRef.current) return;

    const ctx = getAudioContext();

    // Stop any existing sources
    stopInternal();

    // Create new sources and start synchronized
    const now = ctx.currentTime;
    playStartTimeRef.current = now;
    playOffsetRef.current = startTime;

    for (const stem of STEMS) {
      const node = stemNodesRef.current[stem];
      if (!node.buffer) continue;

      const source = ctx.createBufferSource();
      source.buffer = node.buffer;
      source.connect(node.gain);

      // Handle playback end
      source.onended = () => {
        if (isPlayingRef.current) {
          // Check if we actually reached the end
          const elapsed = ctx.currentTime - playStartTimeRef.current;
          const position = playOffsetRef.current + elapsed;
          if (duration > 0 && position >= duration - 0.1) {
            // Reached end, stop playback
            stopInternal();
            setIsPlaying(false);
            setCurrentTime(0);
          }
        }
      };

      source.start(now, startTime);
      node.source = source;
    }

    isPlayingRef.current = true;
    lastSyncTimeRef.current = performance.now();

    // Time update loop — throttled to ~10Hz for store + server sync
    const updateTime = () => {
      if (!isPlayingRef.current) return;

      const now = performance.now();
      if (now - lastSyncTimeRef.current >= TIME_SYNC_INTERVAL_MS) {
        lastSyncTimeRef.current = now;
        const elapsed = ctx.currentTime - playStartTimeRef.current;
        const position = playOffsetRef.current + elapsed;
        setCurrentTime(Math.min(position, duration));
        onTimeSyncRef.current?.(position);
      }

      rafIdRef.current = requestAnimationFrame(updateTime);
    };
    rafIdRef.current = requestAnimationFrame(updateTime);
  }, [getAudioContext, duration, setCurrentTime, setIsPlaying, stopInternal]);

  // ============================================================================
  // Pause
  // ============================================================================

  const pause = useCallback(() => {
    if (!isPlayingRef.current) return;

    const ctx = getAudioContext();
    const elapsed = ctx.currentTime - playStartTimeRef.current;
    const position = playOffsetRef.current + elapsed;

    stopInternal();
    setCurrentTime(position);
  }, [getAudioContext, stopInternal, setCurrentTime]);

  // ============================================================================
  // Seek
  // ============================================================================

  const seek = useCallback((time: number) => {
    const wasPlaying = isPlayingRef.current;

    if (wasPlaying) {
      stopInternal();
    }

    setCurrentTime(time);

    if (wasPlaying) {
      play(time);
    }
  }, [stopInternal, setCurrentTime, play]);

  // ============================================================================
  // Stop (public)
  // ============================================================================

  const stop = useCallback(() => {
    stopInternal();
    setCurrentTime(0);
    setIsPlaying(false);
    // Clear load refs so a new song can reload (or same song can retry)
    loadedAudioIdRef.current = null;
    loadingAudioIdRef.current = null;
  }, [stopInternal, setCurrentTime, setIsPlaying]);

  // ============================================================================
  // React to store isPlaying changes
  // ============================================================================

  useEffect(() => {
    const { currentTime } = useAudioStore.getState();

    if (isPlaying && !isPlayingRef.current) {
      // Store said play, but we're not playing
      if (loadedAudioIdRef.current === audioId) {
        play(currentTime);
      }
    } else if (!isPlaying && isPlayingRef.current) {
      // Store said pause, but we're playing
      pause();
    }
  }, [isPlaying, audioId, play, pause]);

  // ============================================================================
  // Cleanup on unmount
  // ============================================================================

  useEffect(() => {
    return () => {
      stopInternal();
      if (audioContextRef.current) {
        audioContextRef.current.close();
        audioContextRef.current = null;
      }
    };
  }, [stopInternal]);

  // ============================================================================
  // Return
  // ============================================================================

  return {
    load,
    play,
    pause,
    seek,
    stop,
  };
}
