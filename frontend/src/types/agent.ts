import type {
  IntensitySource,
  LinkTarget,
  PositionSource,
} from "./sae";

export type AgentMode = "off" | "directive" | "dj";

export type AgentPhase =
  | "off"
  | "armed"
  | "listening"
  | "transcribing"
  | "thinking"
  | "searching_features"
  | "planning"
  | "applying"
  | "watching"
  | "dj_deciding"
  | "cooldown"
  | "error";

export type AgentIntentClauseKind =
  | "subject"
  | "transformation"
  | "effect"
  | "driver"
  | "target"
  | "timing"
  | "strength"
  | "style"
  | "composition";

export type AgentIntentTiming =
  | "persistent"
  | "section"
  | "on_hits"
  | "on_transients"
  | "on_tension"
  | "on_release"
  | "ambient";

export type AgentIntentStrength = "subtle" | "medium" | "strong" | "extreme";

export interface AgentIntentDriver {
  link_target?: LinkTarget | null;
  intensity_source?: IntensitySource | null;
  position_source?: PositionSource | null;
  aliases?: string[];
}

export interface AgentDirectiveClause {
  text: string;
  kind: AgentIntentClauseKind;
  subject?: string | null;
  transformation?: string | null;
  effect?: string | null;
  target_blocks?: string[];
  drivers?: AgentIntentDriver[];
  timing: AgentIntentTiming;
  strength: AgentIntentStrength;
  confidence: number;
}

export interface AgentIntentIR {
  directive: string;
  clauses: AgentDirectiveClause[];
  summary?: string | null;
  source?: "user" | "dj";
}

export interface AgentToolEvent {
  name: string;
  status: "started" | "completed" | "failed";
  arguments?: Record<string, unknown>;
  result_summary?: Record<string, unknown>;
}

export interface AgentEvent {
  type: "agent_event";
  event_id: string;
  timestamp: string;
  mode: AgentMode;
  phase: AgentPhase;
  provider?: string | null;
  model?: string | null;
  transcript?: string | null;
  summary?: string | null;
  tool?: AgentToolEvent | null;
  feature_candidates?: Record<string, unknown>[];
  changes?: Record<string, unknown>[];
  intent?: AgentIntentIR | null;
  error?: string | null;
}

export type AgentEntrySituation =
  | "no_song_loaded"
  | "song_processing"
  | "song_loaded_idle"
  | "visualizer_paused"
  | "visualizer_playing";

export interface AgentEntryContext {
  situation: AgentEntrySituation;
  summary: string;
  recommended_next_step: string;
  audio: {
    audio_id_present: boolean;
    upload_phase: string;
    duration: number | null;
    current_time: number;
    is_playing: boolean;
  };
  generation: {
    status: string;
    is_generating: boolean;
  };
  song_intelligence: {
    profile_available: boolean;
    analysis_available: boolean;
  };
  control: {
    enabled_block_count: number;
    prompt_empty: boolean;
    latent_empty: boolean;
    fresh_blank_setup: boolean;
    composition: Record<string, unknown>;
  };
}

export interface AgentStateResponse {
  armed: boolean;
  mode: AgentMode;
  active_session: boolean;
  entry_context?: AgentEntryContext;
  block_configs: Record<string, unknown>;
  destinations: Record<string, unknown>;
  song_profile?: Record<string, unknown> | null;
  song_analysis_available?: boolean;
  latest_event: AgentEvent | null;
  event_log?: AgentEvent[];
}

export interface AgentMusicWindowResponse {
  active_session: boolean;
  current_time: number;
  sampled_at_audio_time: number;
  sampled_at_wall_time_ms: number;
  duration?: number | null;
  bpm?: number | null;
  is_playing: boolean;
  lookback: number;
  lookahead: number;
  song_intelligence_available?: boolean;
  song_profile?: Record<string, unknown> | null;
  section?: Record<string, unknown> | null;
  at_current_time?: Record<string, unknown> | null;
  lookahead_context?: Record<string, unknown> | null;
  window_summary?: Record<string, unknown> | null;
  aggregate_windows?: Record<string, unknown> | null;
  target_windows?: Record<string, unknown> | null;
  ranked_window_targets?: Record<string, unknown> | null;
  auto_dance_hints?: Record<string, unknown> | null;
  dominant_targets?: Record<string, unknown> | null;
  stems?: Record<string, unknown>;
  prominence?: Record<string, unknown>;
  block_configs?: Record<string, unknown>;
  snapshots: Record<string, unknown>[];
}

export interface AgentSongAnalysisResponse {
  available: boolean;
  audio_id?: string | null;
  analysis?: Record<string, unknown> | null;
}
