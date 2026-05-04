import { create } from 'zustand';
import type { LinkTarget } from '../types/sae';

export interface StemProfile {
  name: string;
  role: string;
  texture: string;
  frequency_range: string;
  mean_energy: number;
  hpss_ratio: number;
  has_pitch: boolean;
  has_tension: boolean;
  onset_density: number;
}

export interface TensionArcPoint {
  time: number;
  tension: number;
  trend: 'rising' | 'falling' | 'stable';
}

export interface SongProfile {
  bpm: number;
  estimated_key: string | null;
  key_confidence: number;
  duration: number;
  stems: Record<string, StemProfile>;
  coupling: Record<string, number>;
  sections: number[];
  tension_arc: TensionArcPoint[];
  overall_character: string;
}

export interface SongTargetProfile {
  target: string;
  group: string;
  source: string;
  stats: Record<string, number>;
  movement_words: string[];
  good_for: string[];
  preferred_intensity_source: string;
  position_source_affordances: string[];
  coupled_targets: Array<Record<string, number | string>>;
}

export interface SongAnalysis {
  version: string;
  anonymous: boolean;
  metadata_policy: string;
  metadata?: Record<string, string>;
  duration: number;
  bpm: number;
  target_count: number;
  link_targets: Record<string, SongTargetProfile>;
  curve_catalog?: {
    format: string;
    target_count: number;
    targets: Record<string, string[]>;
  };
  ranked_drivers: Record<string, Array<{ target: string; score: number; reasons: string[] }>>;
  structure?: Record<string, unknown> & {
    section_target_summary?: Array<Record<string, unknown>>;
  };
  entry_planning?: Record<string, unknown>;
}

export interface DecodedSongCurves {
  timestamps?: Float32Array;
  tension?: Float32Array;
  tonal_distance?: Float32Array;
  novelty_long?: Float32Array;
  lock_index: Record<string, Float32Array>;
  target_curves: Partial<Record<LinkTarget, Record<string, Float32Array>>>;
}

interface SongIntelligenceCurves {
  tension: Float32Array | null;
  tonal_distance: Float32Array | null;
  novelty_long: Float32Array | null;
  lock_index: Record<string, Float32Array>;
  targetCurves: Partial<Record<LinkTarget, Record<string, Float32Array>>>;
}

interface SongIntelligenceState {
  audioId: string | null;
  profile: SongProfile | null;
  analysis: SongAnalysis | null;
  timestamps: Float32Array | null;
  sections: number[];
  curves: SongIntelligenceCurves;
  receivedAtWallTimeMs: number | null;
  setProfile: (audioId: string, profile: SongProfile, sections: number[], analysis?: SongAnalysis | null) => void;
  setCurves: (curves: DecodedSongCurves) => void;
  clear: () => void;
}

export interface SongIntelligencePayload {
  song_profile?: SongProfile | null;
  song_analysis?: SongAnalysis | null;
  song_sections?: number[] | null;
}

const emptyCurves = (): SongIntelligenceCurves => ({
  tension: null,
  tonal_distance: null,
  novelty_long: null,
  lock_index: {},
  targetCurves: {},
});

export const useSongIntelligenceStore = create<SongIntelligenceState>((set) => ({
  audioId: null,
  profile: null,
  analysis: null,
  timestamps: null,
  sections: [],
  curves: emptyCurves(),
  receivedAtWallTimeMs: null,

  setProfile: (audioId, profile, sections, analysis = null) =>
    set({
      audioId,
      profile,
      analysis,
      sections,
      receivedAtWallTimeMs: Date.now(),
    }),

  setCurves: (decoded) =>
    set((state) => ({
      timestamps: decoded.timestamps ?? state.timestamps,
      curves: {
        tension: decoded.tension ?? state.curves.tension,
        tonal_distance: decoded.tonal_distance ?? state.curves.tonal_distance,
        novelty_long: decoded.novelty_long ?? state.curves.novelty_long,
        lock_index: {
          ...state.curves.lock_index,
          ...decoded.lock_index,
        },
        targetCurves: mergeTargetCurves(state.curves.targetCurves, decoded.target_curves),
      },
      receivedAtWallTimeMs: Date.now(),
    })),

  clear: () =>
    set({
      audioId: null,
      profile: null,
      analysis: null,
      timestamps: null,
      sections: [],
      curves: emptyCurves(),
      receivedAtWallTimeMs: null,
    }),
}));

export const songIntelligenceActions = {
  setProfile: (...args: Parameters<SongIntelligenceState['setProfile']>) =>
    useSongIntelligenceStore.getState().setProfile(...args),
  setCurves: (...args: Parameters<SongIntelligenceState['setCurves']>) =>
    useSongIntelligenceStore.getState().setCurves(...args),
  hydrateFromPayload: (audioId: string, payload: SongIntelligencePayload) => {
    if (!payload.song_profile) return false;
    const sections = Array.isArray(payload.song_sections)
      ? payload.song_sections
      : payload.song_profile.sections;
    useSongIntelligenceStore.getState().setProfile(
      audioId,
      payload.song_profile,
      sections,
      payload.song_analysis ?? null,
    );
    return true;
  },
  clear: () => useSongIntelligenceStore.getState().clear(),
};

function mergeTargetCurves(
  current: Partial<Record<LinkTarget, Record<string, Float32Array>>>,
  incoming: Partial<Record<LinkTarget, Record<string, Float32Array>>>,
) {
  const merged: Partial<Record<LinkTarget, Record<string, Float32Array>>> = { ...current };
  for (const [target, channels] of Object.entries(incoming) as Array<[LinkTarget, Record<string, Float32Array>]>) {
    merged[target] = {
      ...(merged[target] ?? {}),
      ...channels,
    };
  }
  return merged;
}
