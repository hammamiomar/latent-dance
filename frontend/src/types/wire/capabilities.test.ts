/**
 * Golden-fixture lock for the capabilities wire contract.
 *
 * pytest (tests/test_backends.py) asserts the backend emits exactly the
 * fixture; this suite asserts parseBackendCapabilities() reproduces the
 * fixture field-for-field. Together they pin backend output ≡ TS mirror.
 */

import { describe, expect, it } from 'vitest';
import { parseBackendCapabilities } from './capabilities';
import goldenFixture from '../../../../tests/fixtures/capabilities.sae_steering.json';

/** Fresh mutable copy per test — the imported module object is shared. */
function loadFixture(): Record<string, unknown> {
  return structuredClone(goldenFixture) as Record<string, unknown>;
}

describe('capabilities wire contract', () => {
  it('parses the golden SAE manifest field-for-field', () => {
    const caps = parseBackendCapabilities(loadFixture());

    // Deep equality means the mirror covers every field the backend emits;
    // a new backend field lands in the fixture and fails here until typed.
    expect(caps).toEqual(goldenFixture);

    expect(caps.mode).toBe('sae_steering');
    expect(caps.slot_count).toBe(4);
    expect(caps.slots.map((s) => s.name)).toEqual(['down.2.1', 'mid.0', 'up.0.0', 'up.0.1']);
    expect(caps.feature_id_range).toEqual([0, 5119]);
    expect(caps.control_inputs.map((c) => c.name)).toContain('slot.strength');
  });

  it('rejects an unknown control kind with the field path', () => {
    const broken = loadFixture() as { control_inputs: { kind: string }[] };
    broken.control_inputs[0].kind = 'tensor4d';
    expect(() => parseBackendCapabilities(broken)).toThrow(/control_inputs\[0\]\.kind/);
  });

  it('rejects a slot_count that disagrees with the slot list', () => {
    const broken = loadFixture();
    broken.slot_count = 7;
    expect(() => parseBackendCapabilities(broken)).toThrow(/slot_count/);
  });

  it('rejects a manifest missing a required field', () => {
    const broken = loadFixture();
    delete broken.has_prompts;
    expect(() => parseBackendCapabilities(broken)).toThrow(/has_prompts/);
  });
});
