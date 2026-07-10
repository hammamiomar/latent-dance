/**
 * HelpDialog — Onboarding guide for hambajuba2ba.
 *
 * Explains the overall flow, audio stems, slot config params,
 * and scene controls. Open/close controlled by parent (ModeBar [?] button).
 * The slot list comes from the capability manifest, like everything else.
 */

import { useState } from "react";
import { useCapabilities } from "../stores/useSessionStore";

type Tab = "flow" | "stems" | "config" | "scenes";

const TABS: { key: Tab; label: string }[] = [
  { key: "flow", label: "Flow" },
  { key: "stems", label: "Stems" },
  { key: "config", label: "Config" },
  { key: "scenes", label: "Scenes" },
];

interface HelpDialogProps {
  isOpen: boolean;
  onClose: () => void;
}

export function HelpDialog({ isOpen, onClose }: HelpDialogProps) {
  const [activeTab, setActiveTab] = useState<Tab>("flow");

  if (!isOpen) return null;

  return (
    <div
      className="absolute bottom-14 left-4 z-50 win95-panel"
      style={{ width: 380, maxHeight: "75vh", overflow: "hidden" }}
    >
      {/* Header */}
      <div className="win95-title-bar flex items-center justify-between">
        <span className="text-xs font-bold tracking-wider">GUIDE</span>
        <button onClick={onClose} className="win95-title-btn">X</button>
      </div>

      {/* Tabs */}
      <div className="flex border-b border-[var(--color-panel-border)]">
        {TABS.map(({ key, label }) => (
          <button
            key={key}
            onClick={() => setActiveTab(key)}
            className="flex-1 text-xs py-1.5 transition-colors"
            style={{
              background: activeTab === key ? "var(--color-panel-bg)" : "var(--color-void-deep)",
              color: activeTab === key ? "var(--color-text-primary)" : "var(--color-text-dim)",
              borderBottom: activeTab === key ? "2px solid var(--color-accent)" : "2px solid transparent",
            }}
          >
            {label}
          </button>
        ))}
      </div>

      {/* Content — scrollable */}
      <div className="p-3 overflow-y-auto" style={{ maxHeight: "calc(75vh - 64px)" }}>
        {activeTab === "flow" && <FlowTab />}
        {activeTab === "stems" && <StemsTab />}
        {activeTab === "config" && <ConfigTab />}
        {activeTab === "scenes" && <ScenesTab />}
      </div>
    </div>
  );
}

// =============================================================================
// SHARED COMPONENTS
// =============================================================================

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="mb-3">
      <h4 className="text-xs font-bold mb-1.5 tracking-wide" style={{ color: "var(--color-text-primary)" }}>
        {title}
      </h4>
      {children}
    </div>
  );
}

function Item({ name, desc, accent }: { name: string; desc: string; accent?: string }) {
  return (
    <div className="p-2 rounded mb-1.5" style={{ background: "var(--color-void-elevated)", borderLeft: `3px solid ${accent || "var(--color-accent)"}` }}>
      <div className="font-mono font-bold text-xs" style={{ color: "var(--color-text-primary)" }}>{name}</div>
      <div className="text-xs mt-0.5" style={{ color: "var(--color-text-muted)" }}>{desc}</div>
    </div>
  );
}

function Mono({ children }: { children: React.ReactNode }) {
  return (
    <div className="p-2 rounded font-mono mb-2" style={{ background: "var(--color-void-deep)", fontSize: "9px", lineHeight: 1.5, color: "var(--color-text-dim)" }}>
      <pre className="whitespace-pre">{children}</pre>
    </div>
  );
}

// =============================================================================
// FLOW TAB — Onboarding overview
// =============================================================================

function FlowTab() {
  return (
    <div className="text-xs">
      <Section title="How It Works">
        <p className="mb-2" style={{ color: "var(--color-text-muted)", lineHeight: 1.6 }}>
          Music is split into <b>stems</b> (bass, drums, vocals, other).
          Audio features are extracted, run through <b>physics simulation</b>,
          then steer learned visual concepts inside the image generator.
          The music literally <em>sculpts</em> the generation process.
        </p>
      </Section>

      <Section title="Four Control Axes">
        <Item name="SAE Steering" desc="WHAT is expressed — visual concepts driven by audio" accent="var(--color-stem-bass)" />
        <Item name="Spatial Masks" desc="WHERE in the frame — pitch maps to vertical position" accent="var(--color-stem-vocals)" />
        <Item name="Destinations" desc="WHERE in prompt space — A/B scene blending" accent="var(--color-stem-drums)" />
        <Item name="Composition" desc="HOW noise evolves — beat-synced circular walk" accent="var(--color-stem-other)" />
      </Section>

      <Section title="Auto-Config">
        <p style={{ color: "var(--color-text-muted)", lineHeight: 1.6 }}>
          Pick a stem. Everything else is <b>derived automatically</b> from audio analysis:
          physics character, spatial region, intensity source, response timing.
          Switch to Manual to override any parameter.
        </p>
      </Section>

      <Mono>{
`Audio → Demucs (4 stems)
  → Feature extraction (7 channels)
    → Physics simulation (spring/osc/drift)
      → SAE steering (strength × feature direction)
        → SDXL-Turbo → 50 FPS output`
      }</Mono>
    </div>
  );
}

// =============================================================================
// STEMS TAB — Audio sources (link targets)
// =============================================================================

function StemsTab() {
  return (
    <div className="text-xs">
      <Section title="Physical Stems">
        <p className="mb-1.5" style={{ color: "var(--color-text-dim)" }}>From Demucs source separation</p>
        <Item name="bass" desc="Low-end instruments, sub-bass" accent="var(--color-stem-bass)" />
        <Item name="drums" desc="Full drum kit, percussion" accent="var(--color-stem-drums)" />
        <Item name="vocals" desc="Voices, lead melodies" accent="var(--color-stem-vocals)" />
        <Item name="other" desc="Everything else — synths, guitars, FX" accent="var(--color-stem-other)" />
      </Section>

      <Section title="Sub-bands">
        <p className="mb-1.5" style={{ color: "var(--color-text-dim)" }}>Frequency splits of physical stems</p>
        <Item name="drums_low" desc="Kick drum, floor tom" />
        <Item name="drums_mid" desc="Snare body, toms" />
        <Item name="drums_high" desc="Hi-hats, cymbals, shakers" />
        <Item name="other_mid" desc="Guitars, keys, mid-range synths" />
        <Item name="other_high" desc="High synths, effects, shimmer" />
      </Section>

      <Section title="HPSS Components">
        <p className="mb-1.5" style={{ color: "var(--color-text-dim)" }}>Harmonic/percussive separation per stem</p>
        <p style={{ color: "var(--color-text-muted)" }}>
          Each stem splits into <b>_harmonic</b> (tonal content) and <b>_percussive</b> (transient content).
          E.g. <span className="font-mono">drums_harmonic</span> = ringing tones, <span className="font-mono">drums_percussive</span> = sharp attacks.
        </p>
      </Section>

      <Section title="Derived">
        <Item name="tension" desc="Harmonic tension — dissonance level per stem" />
        <Item name="tonal_distance" desc="How far from the home key" />
        <Item name="global" desc="Aggregate of the full mix" />
      </Section>
    </div>
  );
}

// =============================================================================
// CONFIG TAB — Slot config panel params
// =============================================================================

function ConfigTab() {
  const capabilities = useCapabilities();
  return (
    <div className="text-xs">
      <Section title="Steering Slots">
        <p className="mb-1.5" style={{ color: "var(--color-text-dim)" }}>
          Which part of the generation each orb steers — declared by the connected backend
        </p>
        {(capabilities?.slots ?? []).map((slot) => (
          <Item
            key={slot.name}
            name={`${slot.display_name} (${slot.name})`}
            desc={slot.description}
            accent={slot.color}
          />
        ))}
        {!capabilities && (
          <p style={{ color: "var(--color-text-dim)" }}>Waiting for the backend manifest…</p>
        )}
      </Section>

      <Section title="Intensity Source">
        <p className="mb-1.5" style={{ color: "var(--color-text-dim)" }}>What audio feature drives the visual response</p>
        <Item name="energy" desc="Smoothed loudness — fast attack, slow decay (default)" />
        <Item name="transient" desc="Onset/peak detection — spikes on drum hits" />
        <Item name="flux" desc="Spectral change rate — high on attacks, low on sustains" />
        <Item name="envelope" desc="Raw RMS energy — unsmoothed, jittery" />
      </Section>

      <Section title="Spatial Mode">
        <Item name="draw" desc="Paint a 16×16 mask — control exactly where" />
        <Item name="pitch_aligned" desc="Auto from pitch: low notes → bottom, high → top" />
        <Mono>{
`┌───────────────────┐
│   ~ high pitch ~  │  pitch_aligned
│                   │  maps pitch to
│    ◉ mid range    │  vertical position
│                   │  in the 16×16
│ ▓▓▓ low pitch ▓▓ │  latent grid
└───────────────────┘`
        }</Mono>
      </Section>

      <Section title="Strength Range">
        <p style={{ color: "var(--color-text-muted)", lineHeight: 1.6 }}>
          <b>Min/Max</b> bounds for the SAE steering strength. Physics output
          [0,1] maps to [min, max]. Wider range = more dramatic visual effect.
          <b> Stage Home</b> = rest position when audio is silent.
        </p>
      </Section>

      <Section title="Dancer Rank">
        <p className="mb-1.5" style={{ color: "var(--color-text-dim)" }}>Prominence in the Dancer Ensemble — who leads the visual show</p>
        <Item name="Rank 1" desc="Main dancer — full prominence (1.0)" />
        <Item name="Rank 2" desc="Backup — 60% prominence" />
        <Item name="Rank 3" desc="Background — 30% prominence" />
        <Item name="Rank 4" desc="Barely visible — 10% prominence" />
        <Item name="Auto" desc="Unranked — eligible for surprise promotion on novelty" />
      </Section>

      <Section title="Intensity Curve">
        <Item name="linear" desc="Direct mapping — no shaping" />
        <Item name="gamma" desc="Power curve — reshape response (adjustable exponent)" />
        <Item name="clip" desc="Boost 1.5× — more aggressive, clips at max" />
      </Section>
    </div>
  );
}

// =============================================================================
// SCENES TAB — Destinations + Composition
// =============================================================================

function ScenesTab() {
  return (
    <div className="text-xs">
      <Section title="Prompt Destinations">
        <p className="mb-2" style={{ color: "var(--color-text-muted)", lineHeight: 1.6 }}>
          Two text prompts (A and B) define the visual scene space.
          The system <b>SLERPs</b> between them — smoothly blending
          in the model's prompt embedding space.
        </p>
        <Item name="Slider" desc="Manual crossfader — you control the blend" />
        <Item name="Reactive" desc="Audio drives the blend — weighted average of active stems" />
        <Item name="Linked" desc="A specific stem drives the blend — uses the Dance Model" />
        <p className="mt-1.5" style={{ color: "var(--color-text-dim)", lineHeight: 1.6 }}>
          Loading a new prompt freezes current state as A, sets the new prompt as B, and
          resets blend to 0. You always travel FROM where you are TO somewhere new.
        </p>
      </Section>

      <Section title="Composition (Noise Walk)">
        <p className="mb-2" style={{ color: "var(--color-text-muted)", lineHeight: 1.6 }}>
          Controls the latent noise that seeds generation. A <b>circular walk</b> between
          two noise tensors (seeds A and B) continuously explores visual space.
        </p>
        <Item name="Auto" desc="Beat energy + tonal drift blended (default)" />
        <Item name="Pulse" desc="Pure beat-driven rotation — rhythmic" />
        <Item name="Continuous" desc="Pure tonal drift — smooth, evolving" />
        <p className="mt-1.5" style={{ color: "var(--color-text-muted)", lineHeight: 1.6 }}>
          <b>Distance</b> controls how far each beat moves through noise space.
          Higher = more visual variation per beat. <b>Seeds</b> define the two anchor
          points of the circular walk.
        </p>
      </Section>

      <Section title="Physics Presets">
        <p className="mb-1.5" style={{ color: "var(--color-text-dim)" }}>Mass-spring-damper between audio and visuals — auto-selected per stem</p>
        <Item name="drums" desc="Punchy bounce (zeta 0.45) — overshoots, settles" accent="var(--color-stem-drums)" />
        <Item name="drums_high" desc="Snappy hats (zeta 0.40) — fastest bounce" accent="var(--color-stem-drums)" />
        <Item name="bass" desc="Weighty, minimal bounce (zeta 0.85)" accent="var(--color-stem-bass)" />
        <Item name="vocals" desc="Expressive breathing (zeta 0.70)" accent="var(--color-stem-vocals)" />
        <Item name="ambient" desc="Overdamped, dreamy (zeta 1.25) — never bounces" accent="var(--color-stem-other)" />
        <Item name="linear" desc="Near-instant (zeta 1.0) — raw, no physics" />
        <Mono>{
`zeta < 1  Underdamped — bounce, overshoot
zeta = 1  Critical — fastest, no overshoot
zeta > 1  Overdamped — slow, weighty approach`
        }</Mono>
      </Section>
    </div>
  );
}
