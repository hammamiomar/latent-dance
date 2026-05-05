# Hermes Operator Guide

This guide expands the compact Hermes skill with the real Hamba instrument
vocabulary. It is for operators and agents that need to translate creative
language into typed visual controls without touching runtime code.

## Control Model

Hermes should plan in two layers:

1. Intent IR: what the user means.
2. Visual plan: the sparse set of legal frontend controls that implements it.

Intent IR concepts:

- `subject`: the visual world, object, figure, or scene to establish.
- `transformation`: a change between states, often prompt A -> prompt B.
- `effect`: local accents such as sparkles, edges, faces, stripes, lighting, or
  texture.
- `driver`: the audio link target that animates the clause.
- `target block`: which SAE block owns the visual role.
- `timing`: hit, sustain, phrase, section, tension, or manual motion.
- `strength`: subtle, medium, strong, or extreme.

The IR is not a backend payload. It is a planning and audit artifact that helps
Hermes choose blocks, link targets, feature searches, ranks, stage bounds,
destination settings, and composition motion.

## Conversation Modes

Directive mode is collaborative. Hermes should not treat every user utterance
as an immediate apply request. If the user is sketching an idea, comparing
directions, or giving an underspecified vibe, it should talk through what it
would control and ask one useful question. Apply controls when the user clearly
asks to set, change, try, make, route, or confirms a direction.

DJ mode is autonomous. Hermes can make sparse section-level moves without
asking first, but it should still report the audible cue and the visual reason
for the move.

Good directive behavior:

- "I can make the Nigerian travel idea mostly prompt-driven, then use
  `other_high` for whistle traces. Do you want the A/B journey to move from
  daytime road sketch into night market, or from calm landscape into chaotic
  dance floor?"
- After the user answers, apply one sparse plan.
- If the user asks "what do you see?" or "what is notable to visualize?" and
  whole-song analysis is not available, Hermes should say it cannot inspect the
  song's DSP yet. It may still stage a conservative starter rig from the blank
  control surface if the user is clearly trying to begin.

Bad directive behavior:

- Searching four SAE blocks immediately for a broad vibe before setting prompts.
- Claiming to know BPM, sections, or stem activity when no song profile exists.
- Filling every block because the user gave a vivid idea.

## Freshness And Staleness

`hamba_get_music_window` returns a snapshot of the local frontend music state.
The snapshot includes:

- `sampled_at_audio_time`: the audio element time used for curve sampling.
- `sampled_at_wall_time_ms`: the wall-clock time when the frontend sampled it.

Use sampled times as context, not as a beat-perfect contract. Normal Hamba
plans are durable steering changes, so do not include `based_on_audio_time`,
`based_on_wall_time_ms`, or `max_staleness_sec` in `hamba_apply_visual_plan`.
The MCP server ignores those legacy timing fields when they are received.

Music windows are for choosing mappings, not triggering individual frames.
Prefer persistent audio-reactive mappings such as stems, link targets, ranks,
feature IDs, prompt journeys, response curves, and composition behavior over
frame-precise "do this now" actions. If the track has moved too far from the
sampled window, call `hamba_get_music_window` again and choose durable mappings
for the current playback context.

Idle staging is allowed. When `hamba_get_state` reports `active_session=false`
but `armed=true` and `has_control_state=true`, Hermes may still stage prompts,
latent seeds, composition settings, and SAE block mappings for the next
generation/playback session. Do not reject durable setup just because frames are
not currently streaming. The exception is `freeze_blend`, which captures a live
backend blend and therefore needs an active visualizer session.

Empty prompt or latent slots are not a blocker. For a first visual setup,
Hermes should create both prompt slots, both latent seed slots, composition
settings, and any needed SAE block mappings while idle. When generation starts,
the frontend syncs that staged control state to the backend before playback
continues.

ARM entry context is part of `hamba_get_state`. ARM is order-independent: the
user may arm before upload, during processing, after a song is ready, or during
playback. Treat ARM as permission for Hermes to listen/control, not as proof
that the song is ready. When the user hits ARM, the frontend records the
situation in `entry_context`:

- `song_processing`: upload/stem separation/analysis is still running.
- `song_loaded_idle`: a song is ready but generation is not active.
- `visualizer_paused`: generation is active but playback is paused.
- `visualizer_playing`: generation and playback are both active.
- `no_song_loaded`: no audio context exists yet.

Hermes should use that context before deciding whether to wait, ask for visual
direction, stage a first setup, or read a live music window. If the user arms
before upload, the frontend emits a newer `agent_entry_context` event as upload,
analysis, stem loading, or song readiness changes. Hermes should always prefer
the newest context. The entry context is a state snapshot, not a song-metadata
source. If filename context matters, call `hamba_get_song_analysis`; it may
include upload or library filename metadata when available.

Default blank state:

- All SAE blocks disabled.
- Blank/default parameters are not a plan. When enabling an SAE block in a first
  setup, choose its link target, rank, intensity source/curve, and strength
  bounds from song analysis plus the visual role. Do not copy default block
  values through just because the canvas started blank.
- SAE stage bounds: `-30, 0, 30`.
- Default links/ranks: `down.2.1` -> `bass` rank 1, `mid.0` -> `vocals` rank
  2, `up.0.0` -> `drums` rank 1, `up.0.1` -> `other_high` rank null.
- Default block intensity: `energy_smooth` except `up.0.0` starts as
  `transient`; all curves are `linear`, gamma 1.
- Prompt defaults: GLOBAL/reactive mode, empty A/B prompts, stage `-30, 0, 30`,
  position `auto`, intensity `energy_smooth`, rankings `drums=1`, `bass=2`,
  `vocals=null`, `other=null`, blend slew 1.5.
- Latent defaults: no seed A/B. Composition defaults: distance 1.0, mode
  `auto`.

If `has_song_profile=false`, Hermes should not pretend to know the loaded
song's measured structure. It can use the user's genre or sound description as
creative intent, but should avoid claims about current sections, BPM, drops, or
stem activity until a song profile or music window is available.

## Whole-Song Analysis

`hamba_get_song_analysis` is the entry-planning read. It returns DSP evidence
for each available link target, not creative prescriptions. The response may
include `metadata.filename` and `metadata_policy` when an upload or library
filename is available. Use filename as lightweight taste/reference context only:
do not infer artist, title, or genre beyond what the filename literally says,
and keep control decisions grounded in DSP plus the user's directive.

The response includes:

- `link_targets[target].movement_words`: compact signal descriptors such as
  `punchy`, `sustained`, `wide_swing`, `pitched`, `bright`, or `distinct`.
- `link_targets[target].preferred_intensity_source`: a suggested control input
  from the measured behavior, usually `transient`, `flux`, `energy_smooth`, or
  `envelope`.
- `link_targets[target].position_source_affordances`: prompt position sources
  that have signal support, such as `pitch`, `chroma`, `brightness`, `tension`,
  or `auto`.
- `link_targets[target].good_for`: control roles like `primary_driver`,
  `rhythmic_hits`, `texture_motion`, `bright_air`, `prompt_position`, or
  `section_or_harmony_arc`.
- `ranked_drivers`: global rankings by role. Use these as evidence, not as an
  autopilot. The user's visual story still decides whether a strong target
  should drive prompt, composition, or an SAE block.

Fresh entry flow:

1. Read state and control surface.
2. If the user has not given a visual direction, ask one focused question.
3. After a direction exists, read whole-song analysis and then the current music
   window if timing matters.
4. Build prompts A/B as the semantic story, set latent/composition motion, then
   add SAE layers for scene, structure, detail, and style only where they have a
   clear role.

Hard guard: when `hamba_get_state` reports
`entry_context.situation="song_loaded_idle"`, `has_song_analysis=true`, and a
blank/fresh setup, Hermes must call `hamba_get_song_analysis` before describing
song traits, browsing/searching SAE features, or applying the first plan. The
MCP server enforces this order for feature browse/search and plan apply.

## Musical Gesture Directives

Treat words like `triplets`, `hats`, `percussion`, `kick`, `snare`, `shaker`,
`bass hits`, or syllables like `tititi` as audio-control intent before visual
subject matter. Usually the right first move is to focus an existing visual
layer on measured targets such as `drums_high`, `drums_mid`,
`drums_percussive`, or `bass_percussive` with `intensity_source="transient"`,
not to search SAE labels for literal dotted or repeated objects. Only add
dotted, bead-chain, or grid motifs when the user asks for that visual motif.

Prompt crossfade should stay in GLOBAL/reactive mode by default. Use
`set_destination_mode(space="prompt", mode="reactive")` and
`set_reactive_config` for global prompt motion. Avoid `set_destination_link`
unless the user explicitly asks the prompt crossfade itself to follow one
specific link target.

## Prompt-First Broad Vision

When the user gives a broad visual direction, prompt destinations are the main
semantic surface. SAE features are supporting controls for scene anchors,
structure, details, texture, lighting, and style. Do not turn a broad request
into only feature searches.

Use this order for broad visual setup:

1. Write prompt A as the home state and prompt B as the transformed/intense
   state. Keep them in the same visual world unless the user asks for a hard
   cut.
2. Set latent seeds and composition distance/mode so the image has the right
   amount of frame-to-frame motion.
3. Set prompt mode and reactive/link config if the user describes how the
   journey should move.
4. Search SAE features only for specific accents or structural reinforcements.
5. Apply one sparse plan.

Prompt writing for SDXL-Turbo should be concrete and sensory: subject, place,
materials, palette, lighting, framing, motion, and style. Vary prompt wording
and latent seeds creatively; do not route common simple requests to the same
deterministic prompt every time. If the user gives a vibe but no clear A/B
journey, ask one concise clarifying question instead of filling every block.

Planning hierarchy:

1. Prompt A/B is the image story and whole-image semantic change.
2. Prompt reactive config and composition decide the dominant motion.
3. `down.2.1` supports global world/composition.
4. `up.0.1` supports global style/texture/light.
5. `up.0.0` and `mid.0` add detail/structure.

If the user asks to change the whole image, Hermes should rewrite prompts
first, then decide whether any SAE layer should be changed. Feature search is
evidence for specific layers, not the default answer to every directive.

## Divergence And Creative Style

The desktop bridge adds two creative instructions to each Brain directive:

- creative style: public, repo-bundled Hamba taste/personality instructions
  from the desktop source tree. The Brain path uses the instructions shipped
  with the repository; it does not read local user files or style env overrides.
- `divergence`: `0..1` creative distance across prompts, latent/noise motion,
  and SAE channels. Default is `0.85`. This is not the LLM API sampling
  temperature.

Divergence does not change DSP truth. It changes how far Hermes may move from
literal semantic matching across channels:

- `<0.35`: lower chaos, not lower authorship. Honor the user's named anchor in
  one clear channel while keeping the other channels non-redundant.
- `0.35..0.70`: anchored weird. Preserve the scene as one anchor, then use
  several lateral moves across SAE features, latent seeds, composition distance,
  material, structure, texture, color, or mood.
- `>=0.70`: exploratory. Preserve the important user anchors, then choose
  orthogonal prompts, SAE features, unusual seeds, shifted ranks, composition
  mode, composition distance, and stage bounds that create model-native
  collisions.

The noise/latent path is a creative channel. Seed A, seed B, composition
distance, and composition mode should express song motion or transformation,
not merely fill default slots.

High-divergence plans should avoid redundant literal features. If the prompt
names a familiar scene, one SAE block may support that scene, but the others
should explore useful collisions such as map grids, facade edges, textile
density, fluorescent sheen, symbols, borders, or abstract structure.

"Randomize everything" means curated chaos: keep whole-song analysis and strong
measured drivers, but resample prompts, latent seeds, SAE features, ranks,
links, and stage bounds. Hermes may shift focus to another strong measured
backbone if the song analysis supports it.

## Criticism And Diagnosis

Criticism about the current visualizer is a diagnostic task. When the user says
that a section is not responding, not showing enough, too much, too static, or
too chaotic, Hermes should compare the current control setup against measured
song behavior before applying.

Diagnostic flow:

1. Read `hamba_get_state` for current prompts, composition, enabled blocks,
   link targets, ranks, bounds, and intensity sources.
2. Read `hamba_get_song_analysis` for whole-song driver affordances.
3. Read `hamba_get_music_window` around the problem time.
4. Identify which current drivers are weak or absent in that window.
5. Identify measured alternatives that are alive, such as `tension`,
   `tonal_distance`, `other_mid`, `other_high`, or HPSS/sub-band targets.
6. Apply the smallest useful change: retarget prompt/composition, adjust
   bounds/ranks/curves, or swap one feature only if the feature itself is the
   issue.

Example: if the opening is visually flat because prompt and blocks are tied to
quiet drums/bass, but tonal distance is moving, route prompt motion or one
supporting layer to `tonal_distance`/`tension` before searching for new scene
features.

Example:

```text
Directive: "punchy watercolor Van Gogh travel through Nigeria, and show the
amapiano whistles."

Prompt A: watercolor travel sketch through Nigeria, winding road by lagoon and
green roadside fields, ochre dust, indigo evening sky, loose wet paint.

Prompt B: swirling post-impressionist Nigerian night journey, yellow-blue
strokes, lanterns, market road, high silver whistle trails in the sky.

Then add latent seeds, composition pulse/auto, and optional SAE accents for
road/world, painterly texture, and high-air shimmer.
```

## Steering And Auto Dance

Brain Window directives are steerable. Submitting a new message while a Hermes
request is still running does not cancel the older request. The desktop bridge
serializes those messages: the current request can finish, then the newer
message runs against the latest frontend state as steering/correction. The STOP
button is the explicit hard-cancel path and also invalidates queued steering.

Hermes should treat a queued or corrective message as collaboration, not as a
fresh unrelated reset. Read state again, preserve what the user liked, and make
the smallest coherent revision unless the user asks for a full image change.

AUTO DANCE is the autonomous DJ loop. It should not live in the render hot
path. The loop should:

1. Check armed/mode/playback state.
2. Poll `hamba_get_music_window` at a coarse interval.
3. Make at most one sparse, durable section-level change per interval.
4. Avoid timing/staleness metadata in normal plans; apply durable mappings.
5. Stop when disarmed, playback stops, or mode changes away from DJ.

## SAE Blocks

Use the four real block IDs:

| Block | Role | Best for |
|-------|------|----------|
| `down.2.1` | Composition | Scene, object, mood, character, global world |
| `mid.0` | Abstract | Structure, spatial layout, density, contrast, symmetry, depth |
| `up.0.0` | Details | Object details, faces, body parts, accessories, edges, shapes |
| `up.0.1` | Style | Style, texture, pattern, lighting, material, color |

Route visual clauses by what they control. A directive like "haunted cathedral
with glass sparkles on hats" should search `down.2.1` for the cathedral/world
and `up.0.1` or `up.0.0` for the sparkles.

## Link Targets

Link targets are the legal audio sources for block steering and linked prompt
destinations. Treat them as measured signals, not fixed genre labels. The
source separator and mix decide what lands in `bass`, `drums`, `vocals`, and
`other`; a synthesizer can land in `other` in one song and bleed into `drums` or
`bass` behavior in another. When the directive is ambiguous, inspect state,
song profile, activity, or a music window and choose the target whose measured
behavior matches the requested motion.

| Group | Targets | How calculated | Use |
|-------|---------|----------------|-----|
| Physical | `bass`, `drums`, `vocals`, `other` | Source-separated stems estimated from the uploaded mix | Broad whole-stem behavior |
| HPSS | `drums_harmonic`, `drums_percussive`, `bass_harmonic`, `bass_percussive`, `vocals_harmonic`, `vocals_percussive`, `other_harmonic`, `other_percussive` | Harmonic-percussive split of each physical stem | Separate sustained tonal body from attacks/noise |
| Sub-band | `drums_low`, `drums_mid`, `drums_high`, `other_mid`, `other_high` | Offline bandpass-filtered virtual stems | Frequency-specific routing |
| Derived | `tension`, `tonal_distance`, `global` | Aggregate song-analysis curves | Harmonic journey or whole-track behavior |

Calculation notes:

- Physical targets are separator estimates. Use them when the whole estimated
  source should move together.
- HPSS `_harmonic` means sustained, tonal, ringing, pitched, or washed energy
  inside the parent stem. HPSS `_percussive` means transient, noisy, struck,
  plucked, consonant, or attack energy inside the parent stem.
- Sub-bands are frequency slices: `drums_low` = 20-200 Hz, `drums_mid` =
  200-5000 Hz, `drums_high` = 5000-16000 Hz, `other_mid` = 200-4000 Hz, and
  `other_high` = 4000-16000 Hz.
- `tension` is energy-weighted aggregate harmonic tension.
  `tonal_distance` is energy-weighted departure from the track tonal center.
  `global` is average activity across physical stems.

Common aliases:

| User language | Prefer |
|---------------|--------|
| kick, thump, low drums | `drums_low` with `transient` |
| snare, clap | `drums_mid` with `transient` or `flux` |
| hi-hats, cymbals, shimmer | `drums_high` with `transient` or `flux` |
| bass hits | `bass` or `bass_percussive` |
| bassline movement | `bass_harmonic` with `energy_smooth` or `pitch` |
| keys, guitar, harmonic body | `other_mid` |
| air, sparkle, reverb tail | `other_high` |
| vocal phrase or melody | `vocals` or `vocals_harmonic` |
| tension, release, key change | `tension` or `tonal_distance` |

Use the most specific target that matches the request. Do not invent stems.

## Sources And Timing

Current boundary: `position_source`, smoothing, silence behavior, and drift are
prompt destination controls in the frontend/Hermes contract. SAE block steering
currently exposes link target, intensity source, intensity curve, gamma, rank,
spatial controls, feature, enable, and strength bounds. Do not rely on
`position_source` as an SAE block action until that contract is explicitly
added.

For prompt destinations, position sources control where the dance model is in
its range:

- `auto`: stem-based defaults.
- `pitch`: monophonic pitch height.
- `chroma`: polyphonic chroma centroid.
- `brightness`: spectral centroid position.
- `tension`: per-stem harmonic tension.
- `tension_global`: aggregate harmonic tension.

Intensity sources control how far the signal moves from home:

- `energy_smooth`: stable musical loudness; default for sustained motion.
- `transient`: onset/hit detector; best for kicks, snares, hats, punctuation.
- `flux`: spectral change; useful for timbral motion and busy textures.
- `envelope`: raw RMS; direct but jittery.

Intensity curves:

- `linear`: direct proportional response.
- `gamma`: power curve; pair with `intensity_gamma`.
- `clip`: aggressive boosted response.

`impulse` is a removed legacy response curve. Do not teach it or send it in
Hermes plans.

## Ranks And Stage Bounds

SAE rank controls ensemble prominence:

| Rank | Role |
|------|------|
| `1` | Lead visual focus |
| `2` | Visible support |
| `3` | Background presence |
| `4` | Subtle texture |
| `null` | Auto/available for surprise promotion |

Default to ranks 1-3 for visible layers. Rank 4 is very weak in practice; use
it only when the user asks for a barely-there texture or when deliberately
parking a layer in the background.

For SAE blocks, stage bounds currently act as a strength range. Use ordered
anchors when sending them, but the current SAE steering runtime maps physics
directly from min to max:

```text
strength = strength_min + physics_value * (strength_max - strength_min)
final_strength = strength * prominence
```

The frontend stores and sends `stage_home` for block configs, but from the
audited runtime code it does not affect SAE block strength today. Do not depend
on block `stage_home` for behavior until the steering computation is changed.
SAE strength is literal feature addition into the UNet block: positive values
push toward the chosen feature direction, negative values push the inverse
direction, and narrow ranges reduce that block's gain without changing its
driver or feature.

For prompt destinations, stage anchors are active runtime controls:

```text
pos_value = stage_left + position * (stage_right - stage_left)
output = stage_home + intensity * (pos_value - stage_home)
blend = (output - stage_left) / (stage_right - stage_left)
```

Use ordered anchors: `stage_left <= stage_home <= stage_right`.

Guidelines:

- Narrow range, such as `-10` to `10`, means subtle.
- Wide range, such as `-40` to `40`, means dramatic.
- Positive-only range, such as `0` to `35`, adds a feature without subtracting
  it on quiet parts.
- Negative-only range, such as `-35` to `0`, intentionally steers toward the
  inverse feature direction on active moments.
- Rank and stage bounds should agree. A rank 1 block with a tiny range is a
  restrained lead; a rank 4 block with a huge range is usually incoherent.

## Spatial Masks

`spatial_mode` has only two legal values:

- `draw`: use a 16x16 `spatial_mask` with exactly 256 numeric values.
- `pitch_aligned`: Hamba generates dynamic masks from pitch; low notes move
  lower in the frame, high notes move higher.

Useful `draw` mask patterns:

- floor: bottom half on, for bass, kick, grounding.
- ceiling: top half on, for hats, shimmer, air.
- center: middle band on, for vocals, melody, focal details.
- fill: all 256 cells on, for uniform steering.
- clear: all 256 cells off, effectively no spatial steering.

These names are presets/patterns, not `spatial_mode` enum values.

## Prompt Destinations

Prompt destinations define a journey between slots A and B.

Actions:

- `set_destination`: set slot `a` or `b` in `prompt` or `latent` space.
  Prompt space is legal only with `destination_type="prompt"` and a `prompt`
  string. Latent space is legal only with `destination_type="seed"` and a
  numeric `seed`.
- `clear_destination`: clear a destination slot. Include `space` and `slot`.
- `set_destination_mode`: choose `slider` or `reactive`. Include
  `space="prompt"`. The UI label `GLOBAL` means contract mode `reactive`. Do
  not send `mode="linked"` here; use `set_destination_link`.
- `set_destination_link`: choose the link target for linked mode; this also
  switches the prompt destination to `linked`. Include `space="prompt"` and
  `link_target`. This is not the default prompt behavior.
- `set_reactive_config`: configure anchors, sources, smoothing, silence
  behavior, rankings, rank weights, and blend slew for prompt space. Include
  `space="prompt"`.
- `set_blend_position`: manual crossfader position in slider mode. Include
  `space="prompt"`.
- `freeze_blend`: ask the backend to capture the current blended destination
  into target slot `a` or `b`. Do not mutate local destination state yourself;
  the frontend updates labels from the next destination status. Include
  `space="prompt"` and use only during an active visualizer session.

Modes:

- `slider`: manual 0-1 blend position.
- `reactive`: ranked stem activity pushes between A and B. This is labeled
  `GLOBAL` in the frontend.
- `linked`: one link target, often `tension` or `tonal_distance`, drives the
  blend.

Default to `reactive`/GLOBAL for prompt motion. A user asking to hear a drum
detail, bass hit, shimmer, or other musical gesture should usually change SAE
block links/intensity or composition, not switch the prompt crossfade into
linked mode.

Prompt stage anchors apply in both `reactive` and `linked`. In `reactive`, the
position/intensity pair is computed from ranked stems. In `linked`, it is sampled
from the selected `link_target`. `slider` mode bypasses stage anchors and uses
the manual blend position.

Prompt-only boundary:

- `set_reactive_config` is meaningful for `space="prompt"`.
- `set_destination_link` is meaningful for `space="prompt"`.
- The backend ignores prompt-style mode, reactive config, and link controls for
  `space="latent"`. Use latent seeds plus composition controls instead.

Prompt convention:

- A is home, calm, grounded, or low-intensity.
- B is far, intense, transformed, or high-tension.
- A and B should usually share a visual world so the journey feels like a
  transformation, not unrelated cuts.

## Composition

Noise composition controls frame-to-frame variation.

| Control | Values | Use |
|---------|--------|-----|
| `distance` | `0` to `4` | Circular walk radius per beat or drift |
| `mode` | `auto`, `pulse`, `continuous` | Adaptive, beat-synced, or drifting motion |

Distance guide:

- `0`: static noise.
- `0.3`: subtle shimmer.
- `1.0`: strong musical movement.
- `1.5` to `2.5`: dramatic performance.
- `3` to `4`: extreme churn; use sparingly.

Mode guide:

- `pulse`: percussive tracks, drops, obvious beats.
- `continuous`: ambient, drone, legato, harmonic motion.
- `auto`: default when the track has both hits and sustained energy.

Hermes can read current composition `distance` and `mode` from
`hamba_get_state.control_state.composition` and set them with
`set_composition_config`.

## Browse And Search

Feature retrieval is local to Hermes/MCP. Labels are decision data, not
generation data. The GPU backend receives numeric controls, not label searches.

Recommended workflow:

1. Browse for orientation with `hamba_browse_catalog`. Use it to see category
   counts and seeded samples for a block or category.
2. Search each Intent IR clause with `hamba_search_features`. Include synonyms
   and a likely category.
3. Prefer hybrid results when available: lexical score, optional semantic
   embeddings, confidence/activation, duplicate avoidance, and seeded diversity.
4. Use `seed` for reproducibility, feature-search `temperature` for retrieval
   novelty, and `avoid_feature_ids` to avoid recent repeats when supported.
5. Select only IDs returned by browse/search or already present in current
   state. Never invent feature IDs.

Semantic retrieval is local and opportunistic. The MCP process uses the tracked
feature embedding artifact plus the optional `hermes-semantic` runtime to embed
queries. If either piece is missing, search still returns lexical results and
reports the fallback reason in `retrieval`.

Example decomposition:

```text
Directive: "Make the bass turn the room into a storm, with glitter on hats."

Intent:
  subject: room/world -> down.2.1, prompt A
  transformation: storm -> prompt B, driver bass or tension
  effect: glitter -> up.0.1 or up.0.0, driver drums_high
  timing: bass sustain plus hat hits
  strength: medium/strong

Retrieval:
  browse down.2.1 scene or mood
  search down.2.1 "room interior storm rain lightning"
  search up.0.1 "glitter sparkle shimmer bright particles"
```

## Safety Boundary

Hermes may use only the Hamba MCP tool surface:

- `hamba_get_state`
- `hamba_get_control_surface`
- `hamba_get_music_window`
- `hamba_browse_catalog`
- `hamba_search_features`
- `hamba_report_phase`
- `hamba_apply_visual_plan`

Hermes must not use shell, filesystem, browser, web tools, arbitrary network,
direct backend endpoints, upload controls, playback ownership, stop/disconnect,
or frame-by-frame control. Plans go to the local frontend bridge, which updates
stores and sends the existing validated WebSocket controls to the backend.
