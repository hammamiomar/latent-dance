export type CompositionMode = 'auto' | 'pulse' | 'continuous';

export interface CompositionStateSnapshot {
  distance: number;
  mode: CompositionMode;
}
