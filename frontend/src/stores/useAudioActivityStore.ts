/**
 * Audio Activity Store
 *
 * Zustand store for real-time audio feature data from backend.
 * Receives ExtendedStemActivity messages at ~10Hz and provides
 * convenient accessors for audio-reactive UI components.
 *
 * Supports all 8 stems:
 *   - Physical (from Demucs): bass, drums, vocals, other
 *   - Virtual (bandpass filtered): drums_low, drums_high, other_mid, other_high
 *
 * Data flow:
 *   Backend (Python/librosa) → WebSocket → Store → Components
 */

import { create } from 'zustand';
import type { StemChannelData, ExtendedStemActivityMessage, AllStems, StemProminence, BlockActivityData, BlockCode } from '../types/sae';

// Default channel values (silence)
const DEFAULT_CHANNEL_DATA: StemChannelData = {
  envelope: 0,
  energy_smooth: 0,
  transient: 0,
  flux: 0,
  brightness: 0,
  flash: 0,
  sustain: 0,
};

// Create default state for all 8 stems
const createDefaultStems = (): Record<AllStems, StemChannelData> => ({
  // Physical stems
  bass: { ...DEFAULT_CHANNEL_DATA },
  drums: { ...DEFAULT_CHANNEL_DATA },
  vocals: { ...DEFAULT_CHANNEL_DATA },
  other: { ...DEFAULT_CHANNEL_DATA },
  // Virtual stems (bandpass filtered)
  drums_low: { ...DEFAULT_CHANNEL_DATA },
  drums_mid: { ...DEFAULT_CHANNEL_DATA },
  drums_high: { ...DEFAULT_CHANNEL_DATA },
  other_mid: { ...DEFAULT_CHANNEL_DATA },
  other_high: { ...DEFAULT_CHANNEL_DATA },
});

interface AudioActivityState {
  /** Current audio playback time */
  audioTime: number;

  /** Per-stem channel data (all 8 stems) */
  stems: Record<AllStems, StemChannelData>;

  /** Computed prominence per stem (optional) */
  prominence?: Record<string, StemProminence>;

  /** Per-block physics activity (optional) */
  blocks?: Record<BlockCode, BlockActivityData>;

  /** Timestamp of last update (for staleness detection) */
  lastUpdateTime: number;

  /** Whether we're receiving data */
  isReceiving: boolean;

  /** Update from WebSocket message */
  updateFromMessage: (msg: ExtendedStemActivityMessage) => void;

  /** Reset to silent state */
  reset: () => void;
}

export const useAudioActivityStore = create<AudioActivityState>((set) => ({
  audioTime: 0,
  stems: createDefaultStems(),
  prominence: undefined,
  lastUpdateTime: 0,
  isReceiving: false,

  updateFromMessage: (msg) => {
    set({
      audioTime: msg.audio_time,
      stems: msg.stems,
      prominence: msg.prominence,
      blocks: msg.blocks,
      lastUpdateTime: Date.now(),
      isReceiving: true,
    });
  },

  reset: () => {
    set({
      audioTime: 0,
      stems: createDefaultStems(),
      prominence: undefined,
      blocks: undefined,
      lastUpdateTime: 0,
      isReceiving: false,
    });
  },
}));
