import { DEFAULT_HAMBA_SOUL } from "./hambaSoul";

type Env = Record<string, string | undefined>;

interface SubmitDirectiveOptions {
  signal?: AbortSignal;
}

interface CreativePayload {
  divergence?: unknown;
  // Legacy aliases kept so older Brain windows and env files still work.
  temperature?: unknown;
  wildness?: unknown;
}

const DEFAULT_HERMES_URL = "http://127.0.0.1:8642/v1/responses";
const DEFAULT_HERMES_MODEL = "hermes-agent";
const DEFAULT_DIVERGENCE = 0.85;
const DEFAULT_HERMES_INSTRUCTIONS =
  "You are controlling the Hamba visualizer. Inspect state, use song analysis for fresh/global setup when available, build prompt and latent direction before SAE accents for broad visual requests, and apply a sparse visual plan through the Hamba MCP tools. Treat new user messages as steering/refinement unless the user asks for a reset.";
const BRAIN_OPERATING_DOCTRINE = [
  "# Brain Operating Doctrine",
  "Non-negotiable first-tool rule: after `hamba_get_state`, if `entry_context.situation=\"song_loaded_idle\"`, `has_song_analysis=true`, and the setup is blank/fresh, call `hamba_get_song_analysis` immediately.",
  "Before that song-analysis call returns, do not describe song traits, visual opinions, SAE ideas, feature searches, or first-rig plans. Say only that you are reading the whole-song analysis.",
  "After song analysis, build the visual hierarchy in this order: prompt A/B story, latent/noise and composition motion, then sparse SAE features.",
  "Use `hamba_get_song_analysis` for whole-song target/section evidence, filename metadata when available, and `hamba_get_music_window` for near-future per-target windows. The filename can color taste, but DSP is the control truth. Do not over-index on tension when drums, bass, other, subbands, or HPSS targets carry stronger local evidence.",
  "A user request is one thread, not the whole image. Honor the user's named subject in at most one prompt and at most one SAE block; the other channels should answer the song with lateral structure, texture, color, material, mood, or motion.",
  "Musical gesture words are control intent, not visual nouns. If the user says triplets, hats, percussion, kick, snare, shimmer, bass hits, or a sung/drummed syllable like tititi, first map it to link targets and intensity sources (`drums_high`/`drums_mid`/`drums_percussive` with `transient` for fast percussion) before searching visual SAE concepts.",
  "Prompt mode policy: keep prompt in `reactive`/GLOBAL mode by default. Use `set_destination_mode` with `space=\"prompt\"` and `mode=\"reactive\"` plus `set_reactive_config`; do not call `set_destination_link` unless the user explicitly asks the prompt crossfade itself to follow one specific stem/link target.",
  "SAE rank policy: every enabled Hermes `update_block_config` should use `sae_rank: 1`. Make support/subtlety with strength ranges, spatial masks, link targets, and intensity curves, not lower ranks.",
  "Intensity source vocabulary is limited: use `energy_smooth` for sustained/body motion, `transient` for hits/attacks, `flux` for texture/change, and `envelope` only when explicitly needed.",
  "Schema vocabulary is exact: set prompt A/B with two `set_destination` actions using `space=\"prompt\"`, `slot=\"a\"|\"b\"`, and `destination_type=\"prompt\"`; never use `set_prompt`, `set_prompt_destinations`, `prompt_a`, or `prompt_b` in an apply plan. In `update_block_config`, write `sae_rank`, never `rank`; write `intensity_gamma`, never `gamma`. In `set_reactive_config`, write flat `stage_left`, `stage_home`, `stage_right`, never a nested `stage` object; write `position_smoothing_ms`, never `smoothing`; write `blend_slew_rate`, never `blend_slew`; omit null `rank_weights` entries; do not send `target`.",
  "Do not send `based_on_audio_time`, `based_on_wall_time_ms`, or `max_staleness_sec` for ordinary live steering. These are durable rig changes applied when ready, not beat-perfect cues; rewinds and loops are valid.",
  "If a feature search or apply is blocked because song analysis has not been read, recover by calling `hamba_get_song_analysis` next, then retry with the analysis in hand.",
  "Prepared palette policy: during first setup or after a major vibe change, call `hamba_prepare_feature_palette` with the current theme and divergence so Auto Dance has ready candidates. At high divergence this palette should be deliberately diverse, not just similar labels.",
  "Auto Dance policy: call `hamba_get_feature_palette` after the music window and choose from prepared unused candidates before doing live feature search. If the palette is empty, prepare it once; do not spend every live checkpoint searching from scratch.",
  "Feature lookup budget: once you have usable candidates or after 4-8 total `hamba_search_features`/`hamba_browse_catalog` calls, stop looking and call `hamba_apply_visual_plan`. If a lookup says the budget is exhausted, apply now using remembered candidates.",
  "`hamba_search_features` is relevance-biased; it should find a requested clause. For orthogonal surprise, first-rig breadth, and high-divergence channels, use `hamba_browse_catalog` with different categories/seeds to cast a wide net, then converge.",
].join("\n");

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object";
}

function clamp01(value: number) {
  if (!Number.isFinite(value)) return null;
  return Math.max(0, Math.min(1, value));
}

function parseCreativeDivergence(value: unknown) {
  if (typeof value === "number") return clamp01(value);
  if (typeof value !== "string" || !value.trim()) return null;
  return clamp01(Number(value));
}

function randomizeDirective(directive: string) {
  const normalized = directive.toLowerCase();
  return (
    /randomi[sz]e\s+(everything|all|it)/.test(normalized) ||
    /(everything|all)\s+random/.test(normalized) ||
    /screw it[, ]+.*random/.test(normalized)
  );
}

function divergenceLabel(divergence: number) {
  if (divergence < 0.35) return "anchored";
  if (divergence < 0.7) return "lateral";
  return "exploratory";
}

function resolveCreativeDivergence(directive: string, env: Env, creative?: CreativePayload) {
  const requested =
    parseCreativeDivergence(creative?.divergence) ??
    parseCreativeDivergence(creative?.temperature) ??
    parseCreativeDivergence(creative?.wildness);
  const configured =
    parseCreativeDivergence(env.HAMBA_HERMES_DIVERGENCE) ??
    parseCreativeDivergence(env.HAMBA_HERMES_TEMPERATURE) ??
    parseCreativeDivergence(env.HAMBA_HERMES_WILDNESS);
  const base = requested ?? configured ?? DEFAULT_DIVERGENCE;
  return randomizeDirective(directive) ? Math.max(base, 0.95) : base;
}

export function buildHermesInstructions(
  directive: string,
  env: Env = process.env,
  creative?: CreativePayload,
) {
  const base = env.HAMBA_HERMES_INSTRUCTIONS || DEFAULT_HERMES_INSTRUCTIONS;
  const divergence = resolveCreativeDivergence(directive, env, creative);
  const label = divergenceLabel(divergence);

  return [
    BRAIN_OPERATING_DOCTRINE,
    "",
    base,
    "",
    "# Hamba Soul",
    DEFAULT_HAMBA_SOUL,
    "",
    "# Creative Divergence",
    `Divergence: ${divergence.toFixed(2)} (${label}). This is Hamba channel divergence, not LLM sampling temperature. DSP stays grounded; divergence changes how far prompts, latent seeds, feature selection, ranks, bounds, and focus move away from the user's literal noun.`,
    "Low divergence (<0.35): reduce chaos, not authorship. Honor the user's named anchor in one clear channel while keeping the remaining channels non-redundant.",
    "Medium divergence (0.35-0.70): keep the user scene as one anchor, then make several channels lateral across structure, texture, color, material, or mood.",
    "High divergence (>=0.70): preserve only the important user anchors, then choose orthogonal prompts, seeds, links, ranks, and SAE features that create model-native collisions.",
    "Treat the noise/latent path as its own authorship channel: seed A, seed B, composition distance, and composition mode should express song motion or transformation, not merely fill defaults.",
    "Do not put literal versions of the same noun on every block. In normal creative use, at most one SAE block and at most one prompt should directly match the user's noun unless the user asks for faithful depiction.",
    "Every non-debug plan should have channel divergence: prompts A and B must not paraphrase each other, and SAE searches should not be four versions of the same concept.",
    "For broad visual requests, prompts establish the world, but SAE features should often be surprising internal concepts rather than redundant prompt labels.",
    "For a first visualization, start wider than the user noun: combine one relevant searched anchor with catalog/browse surprises from different kingdoms, then converge on later revisions if the user steers or the music window narrows.",
    "At roughly 0.75 divergence, prepare a mixed Auto Dance palette: relevant anchors, adjacent materials/structures, orthogonal browse candidates, and wildcards. At 1.0, push more of the palette toward orthogonal and wildcard collisions.",
    "If the user asks to randomize everything, use curated chaos: keep song-analysis/DSP grounding, but resample prompts, seeds, features, ranks, links, and stage bounds with high novelty.",
  ].join("\n");
}

export function hermesResponsesUrl(env: Env = process.env) {
  const raw = env.HAMBA_HERMES_API_URL || DEFAULT_HERMES_URL;
  const trimmed = raw.replace(/\/+$/, "");
  if (trimmed.endsWith("/v1/responses") || trimmed.endsWith("/v1/chat/completions")) {
    return trimmed;
  }
  return `${trimmed}/v1/responses`;
}

export function hermesModelName(env: Env = process.env) {
  return env.HAMBA_HERMES_MODEL?.trim() || DEFAULT_HERMES_MODEL;
}

export function extractHermesText(data: unknown): string {
  if (!isRecord(data)) return "";
  const outputText = data.output_text;
  if (typeof outputText === "string") return outputText;

  const choices = data.choices;
  if (Array.isArray(choices)) {
    const first = choices[0];
    if (isRecord(first) && isRecord(first.message) && typeof first.message.content === "string") {
      return first.message.content;
    }
  }

  const output = data.output;
  if (!Array.isArray(output)) return "";

  const chunks: string[] = [];
  for (const item of output) {
    if (!isRecord(item) || !Array.isArray(item.content)) continue;
    for (const part of item.content) {
      if (isRecord(part) && typeof part.text === "string") chunks.push(part.text);
    }
  }
  return chunks.join("\n").trim();
}

export function buildHermesRequest(
  directive: string,
  url: string,
  env: Env = process.env,
  creative?: CreativePayload,
) {
  const apiKey = env.HAMBA_HERMES_API_KEY || env.API_SERVER_KEY || "";
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  if (apiKey) headers.Authorization = `Bearer ${apiKey}`;

  const instructions = buildHermesInstructions(directive, env, creative);
  const model = hermesModelName(env);
  const isChatCompletions = url.endsWith("/v1/chat/completions");
  const body = isChatCompletions
    ? {
        model,
        messages: [
          { role: "system", content: instructions },
          { role: "user", content: directive },
        ],
        stream: false,
      }
    : {
        model,
        input: directive,
        instructions,
        store: false,
      };

  return { headers, body };
}

export async function submitDirectiveToHermes(
  payload: unknown,
  env: Env = process.env,
  fetchImpl: typeof fetch = fetch,
  options: SubmitDirectiveOptions = {},
) {
  if (!isRecord(payload)) {
    throw new Error("Directive payload must be an object");
  }
  const directive = String(payload.directive || "").trim();
  if (!directive) {
    throw new Error("Directive cannot be empty");
  }
  const creative = isRecord(payload.creative) ? payload.creative : undefined;

  const url = hermesResponsesUrl(env);
  const request = buildHermesRequest(directive, url, env, creative);
  const response = await fetchImpl(url, {
    method: "POST",
    headers: request.headers,
    body: JSON.stringify(request.body),
    signal: options.signal,
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `Hermes API request failed: ${response.status}`);
  }

  const data = await response.json();
  return {
    accepted: true,
    response: extractHermesText(data),
    raw_id: isRecord(data) && typeof data.id === "string" ? data.id : null,
  };
}
