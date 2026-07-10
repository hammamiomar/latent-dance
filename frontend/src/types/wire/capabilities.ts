/**
 * Capabilities wire contract — TypeScript mirror of the backend manifest.
 *
 * The Python side lives in app/backends.py (BackendCapabilities.to_dict()),
 * served at GET /api/capabilities and as the first WebSocket message. Both
 * languages are locked to the same golden fixture,
 * tests/fixtures/capabilities.sae_steering.json: pytest asserts the backend
 * still emits it, vitest asserts parseBackendCapabilities() still accepts it
 * and covers every field. Contract change = regenerate the fixture
 * (scripts/dev/dump_capabilities.py) and let both suites point at the drift.
 *
 * parseBackendCapabilities() is the single entry point for manifests: a
 * malformed backend fails loudly at the boundary with a field path, never as
 * undefined reads deep in render code.
 */

/** How a backend produces frames: fresh per tick, or an evolving optimization. */
export const TEMPORAL_CONTRACTS = ['per_frame', 'evolving_canvas'] as const;
export type TemporalContract = (typeof TEMPORAL_CONTRACTS)[number];

/** Value shapes a control input accepts from the signal-routing layer. */
export const CONTROL_KINDS = ['scalar', 'id', 'mask2d', 'text', 'event'] as const;
export type ControlKind = (typeof CONTROL_KINDS)[number];

/** Display metadata for one steering slot. */
export interface SlotInfo {
  name: string; // canonical id ("down.2.1", "slot_0", ...)
  display_name: string; // "Composition"
  short_name: string; // "COMP"
  color: string; // hex, muted earthy palette
  description: string;
}

/** One named input the backend accepts from the routing layer (e.g. "slot.strength"). */
export interface ControlInput {
  name: string;
  kind: ControlKind;
  count: number;
  id_range: [number, number] | null; // inclusive, for kind="id"
  shape: [number, number] | null; // (H, W), for kind="mask2d"
  description: string;
}

/** What the active backend accepts and produces. */
export interface BackendCapabilities {
  mode: string;
  temporal: TemporalContract;
  slots: SlotInfo[];
  slot_count: number;
  feature_id_range: [number, number]; // inclusive
  feature_label: string; // "Feature" | "Concept" | "Unit"
  spatial_mask_shape: [number, number];
  has_prompts: boolean;
  has_destinations: boolean;
  output_resolution: [number, number]; // (width, height)
  control_inputs: ControlInput[];
}

// =============================================================================
// Runtime validation
// =============================================================================

function fail(path: string, expected: string, value: unknown): never {
  throw new Error(
    `Invalid capabilities manifest at ${path}: expected ${expected}, got ${JSON.stringify(value)}`,
  );
}

function asRecord(value: unknown, path: string): Record<string, unknown> {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) {
    fail(path, 'an object', value);
  }
  return value as Record<string, unknown>;
}

function asString(value: unknown, path: string): string {
  if (typeof value !== 'string') fail(path, 'a string', value);
  return value;
}

function asBoolean(value: unknown, path: string): boolean {
  if (typeof value !== 'boolean') fail(path, 'a boolean', value);
  return value;
}

function asInt(value: unknown, path: string): number {
  if (typeof value !== 'number' || !Number.isInteger(value)) fail(path, 'an integer', value);
  return value;
}

function asIntPair(value: unknown, path: string): [number, number] {
  if (!Array.isArray(value) || value.length !== 2) fail(path, 'a pair of integers', value);
  return [asInt(value[0], `${path}[0]`), asInt(value[1], `${path}[1]`)];
}

function asEnum<T extends string>(value: unknown, options: readonly T[], path: string): T {
  if (typeof value !== 'string' || !(options as readonly string[]).includes(value)) {
    fail(path, options.join(' | '), value);
  }
  return value as T;
}

function parseSlotInfo(value: unknown, path: string): SlotInfo {
  const raw = asRecord(value, path);
  return {
    name: asString(raw.name, `${path}.name`),
    display_name: asString(raw.display_name, `${path}.display_name`),
    short_name: asString(raw.short_name, `${path}.short_name`),
    color: asString(raw.color, `${path}.color`),
    description: asString(raw.description, `${path}.description`),
  };
}

function parseControlInput(value: unknown, path: string): ControlInput {
  const raw = asRecord(value, path);
  return {
    name: asString(raw.name, `${path}.name`),
    kind: asEnum(raw.kind, CONTROL_KINDS, `${path}.kind`),
    count: asInt(raw.count, `${path}.count`),
    id_range: raw.id_range == null ? null : asIntPair(raw.id_range, `${path}.id_range`),
    shape: raw.shape == null ? null : asIntPair(raw.shape, `${path}.shape`),
    description: asString(raw.description, `${path}.description`),
  };
}

/**
 * Validate an untrusted manifest payload into a BackendCapabilities.
 *
 * Returns a fresh object built only from validated fields — unknown extra
 * keys are tolerated (a newer backend may add fields) but never passed
 * through. Throws with a dotted field path on the first mismatch.
 */
export function parseBackendCapabilities(payload: unknown): BackendCapabilities {
  const raw = asRecord(payload, 'capabilities');
  if (!Array.isArray(raw.slots)) fail('capabilities.slots', 'an array', raw.slots);
  if (!Array.isArray(raw.control_inputs)) {
    fail('capabilities.control_inputs', 'an array', raw.control_inputs);
  }

  const slots = raw.slots.map((slot, i) => parseSlotInfo(slot, `capabilities.slots[${i}]`));
  const slotCount = asInt(raw.slot_count, 'capabilities.slot_count');
  if (slotCount !== slots.length) {
    fail('capabilities.slot_count', `the number of slots (${slots.length})`, slotCount);
  }

  return {
    mode: asString(raw.mode, 'capabilities.mode'),
    temporal: asEnum(raw.temporal, TEMPORAL_CONTRACTS, 'capabilities.temporal'),
    slots,
    slot_count: slotCount,
    feature_id_range: asIntPair(raw.feature_id_range, 'capabilities.feature_id_range'),
    feature_label: asString(raw.feature_label, 'capabilities.feature_label'),
    spatial_mask_shape: asIntPair(raw.spatial_mask_shape, 'capabilities.spatial_mask_shape'),
    has_prompts: asBoolean(raw.has_prompts, 'capabilities.has_prompts'),
    has_destinations: asBoolean(raw.has_destinations, 'capabilities.has_destinations'),
    output_resolution: asIntPair(raw.output_resolution, 'capabilities.output_resolution'),
    control_inputs: raw.control_inputs.map((input, i) =>
      parseControlInput(input, `capabilities.control_inputs[${i}]`),
    ),
  };
}
