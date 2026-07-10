/**
 * Audio state management store.
 *
 * Handles audio upload, playback state, and stem mixing controls.
 * Used in conjunction with useAudioMixer for Web Audio API integration.
 */

import { create } from 'zustand';
import type { PhysicalStem, Stem } from '../types/sae';

// Upload lifecycle phases — single source of truth for the entire flow
export type UploadPhase = 'idle' | 'uploading' | 'processing' | 'loading_stems' | 'ready' | 'error';

interface AudioState {
  // Audio metadata
  audioId: string | null;
  librarySongId: string | null;
  filename: string | null;
  stems: Stem[];
  duration: number;

  // Upload lifecycle (replaces scattered local state)
  uploadPhase: UploadPhase;
  uploadStatusLabel: string;
  uploadProgress: number;       // 0-1
  uploadError: string | null;

  // Playback state
  currentTime: number;
  isPlaying: boolean;

  // Stem mixing (only 4 physical stems from Demucs)
  stemVolumes: Record<PhysicalStem, number>;  // 0-1
  stemMuted: Record<PhysicalStem, boolean>;
  stemSolo: PhysicalStem | null;
  masterVolume: number;

  // Actions
  setAudioData: (
    audioId: string,
    stems: Stem[],
    duration: number,
    filename?: string,
    librarySongId?: string,
  ) => void;
  setCurrentTime: (time: number) => void;
  setIsPlaying: (playing: boolean) => void;

  // Upload lifecycle actions
  startUpload: (filename: string) => void;
  setUploadStatus: (label: string, progress: number) => void;
  setUploadPhase: (phase: UploadPhase) => void;
  setUploadError: (error: string) => void;

  // Stem mixing actions (physical stems only)
  setStemVolume: (stem: PhysicalStem, volume: number) => void;
  toggleStemMute: (stem: PhysicalStem) => void;
  setStemSolo: (stem: PhysicalStem | null) => void;
  setMasterVolume: (volume: number) => void;

  // Playback actions
  play: () => void;
  pause: () => void;
  togglePlayPause: () => void;
  seek: (time: number) => void;

  clearAudio: () => void;
}

const initialStemVolumes: Record<PhysicalStem, number> = {
  bass: 1,
  drums: 1,
  vocals: 1,
  other: 1,
};

const initialStemMuted: Record<PhysicalStem, boolean> = {
  bass: false,
  drums: false,
  vocals: false,
  other: false,
};

const initialState = {
  audioId: null,
  librarySongId: null,
  filename: null,
  stems: [] as Stem[],
  duration: 0,
  uploadPhase: 'idle' as UploadPhase,
  uploadStatusLabel: '',
  uploadProgress: 0,
  uploadError: null as string | null,
  currentTime: 0,
  isPlaying: false,
  stemVolumes: { ...initialStemVolumes },
  stemMuted: { ...initialStemMuted },
  stemSolo: null as PhysicalStem | null,
  masterVolume: 0.8,
};

export const useAudioStore = create<AudioState>((set) => ({
  ...initialState,

  setAudioData: (audioId, stems, duration, filename, librarySongId) =>
    set({
      audioId,
      librarySongId: librarySongId ?? null,
      filename: filename ?? null,
      stems,
      duration,
      currentTime: 0,
      isPlaying: false,
    }),

  setCurrentTime: (time) => set({ currentTime: time }),

  setIsPlaying: (playing) => set({ isPlaying: playing }),

  // Upload lifecycle actions
  startUpload: (filename) =>
    set({
      uploadPhase: 'uploading',
      uploadStatusLabel: 'Uploading...',
      uploadProgress: 0,
      uploadError: null,
      librarySongId: null,
      filename,
    }),

  setUploadStatus: (label, progress) =>
    set({ uploadStatusLabel: label, uploadProgress: Math.max(0, Math.min(1, progress)) }),

  setUploadPhase: (phase) => set({ uploadPhase: phase }),

  setUploadError: (error) =>
    set({ uploadPhase: 'error', uploadError: error, uploadStatusLabel: '', uploadProgress: 0 }),

  setStemVolume: (stem, volume) =>
    set((state) => ({
      stemVolumes: {
        ...state.stemVolumes,
        [stem]: Math.max(0, Math.min(1, volume)),
      },
    })),

  toggleStemMute: (stem) =>
    set((state) => ({
      stemMuted: {
        ...state.stemMuted,
        [stem]: !state.stemMuted[stem],
      },
    })),

  setStemSolo: (stem) =>
    set((state) => ({
      // If clicking same stem, unsolo it
      stemSolo: state.stemSolo === stem ? null : stem,
    })),

  setMasterVolume: (volume) =>
    set({ masterVolume: Math.max(0, Math.min(1, volume)) }),

  play: () => set({ isPlaying: true }),

  pause: () => set({ isPlaying: false }),

  togglePlayPause: () =>
    set((state) => ({ isPlaying: !state.isPlaying })),

  seek: (time) =>
    set((state) => ({
      currentTime: Math.max(0, Math.min(state.duration, time)),
    })),

  clearAudio: () => set(initialState),
}));

// ============================================================================
// Selectors (atomic for efficient re-renders)
// ============================================================================

export const useAudioId = () => useAudioStore((s) => s.audioId);
export const useAudioLibrarySongId = () => useAudioStore((s) => s.librarySongId);
export const useAudioFilename = () => useAudioStore((s) => s.filename);
export const useAudioDuration = () => useAudioStore((s) => s.duration);
export const useAudioCurrentTime = () => useAudioStore((s) => s.currentTime);
export const useAudioIsPlaying = () => useAudioStore((s) => s.isPlaying);
export const useUploadPhase = () => useAudioStore((s) => s.uploadPhase);
export const useUploadStatusLabel = () => useAudioStore((s) => s.uploadStatusLabel);
export const useUploadProgress = () => useAudioStore((s) => s.uploadProgress);
export const useUploadError = () => useAudioStore((s) => s.uploadError);
export const useStemVolumes = () => useAudioStore((s) => s.stemVolumes);
export const useStemMuted = () => useAudioStore((s) => s.stemMuted);
export const useStemSolo = () => useAudioStore((s) => s.stemSolo);
export const useMasterVolume = () => useAudioStore((s) => s.masterVolume);

/**
 * Get effective volume for a stem (considering solo/mute).
 */
export function getEffectiveStemVolume(
  stem: PhysicalStem,
  state: Pick<AudioState, 'stemVolumes' | 'stemMuted' | 'stemSolo' | 'masterVolume'>
): number {
  const { stemVolumes, stemMuted, stemSolo, masterVolume } = state;

  // If soloed, only that stem plays
  if (stemSolo !== null && stemSolo !== stem) {
    return 0;
  }

  // If muted, volume is 0
  if (stemMuted[stem]) {
    return 0;
  }

  return stemVolumes[stem] * masterVolume;
}
