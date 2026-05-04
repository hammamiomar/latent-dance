# Hamba Brain / Hermes Agent

Hamba Brain is the optional Hermes agent layer for latent-dance. It does not
replace the visualizer or change the project identity: it gives the existing
instrument a conversational and autonomous operator.

The visualizer still runs the same SDXL-Turbo SAE steering path. Hermes only
talks to Hamba through local MCP tools that can read state, inspect song
analysis, search feature labels, prepare Auto Dance palettes, and apply
validated visual plans.

## What The Agent Can Do

- Build a first visual rig from a song and a user direction: prompt A/B, latent
  seeds, composition motion, and sparse SAE feature controls.
- Interpret musical language as control intent. For example, “hats,” “kick,”
  “bass hits,” “shimmer,” and sung syllables are mapped to measured link
  targets before they become visual motifs.
- Use whole-song analysis and a local music window to choose durable drivers
  such as `drums_high`, `bass_percussive`, `other_mid`, `tonal_distance`, or
  `tension`.
- Search and browse the public SAE feature labels, then apply only returned
  feature IDs through the validated visual-plan contract.
- Run Auto Dance checkpoints during playback. Auto Dance keeps evolving the
  visual chapter, uses a prepared feature palette for speed, and preserves
  recent human steering instead of undoing it.
- Explain what it changed in the Brain Window without polluting the main
  visualizer surface.

## Pieces

- **MCP tool server:** `hambajuba-hermes-mcp`, installed from this package with
  the `hermes` extra. It exposes Hamba state, song analysis, feature lookup,
  palette preparation, and visual-plan apply tools.
- **Hermes skill:** `skills/creative/hambajuba-dance-director/SKILL.md`. It
  teaches Hermes the instrument vocabulary and tool workflow.
- **Brain Window:** the desktop talkback panel that sends directives and shows
  agent/tool/action logs.
- **Optional semantic search:** the `hermes-semantic` extra enables local
  embedding search over the public feature labels. Without it, feature lookup
  falls back to lexical search and seeded diversity.

## Install

Install the Hamba tool server:

```bash
uv tool install "hambajuba2ba[hermes,hermes-semantic] @ git+https://github.com/hammamiomar/latent-dance"
```

Install the Hermes skill:

```bash
hermes skills install hammamiomar/latent-dance/skills/creative/hambajuba-dance-director
```

Then configure Hermes to run `hambajuba-hermes-mcp` as a local MCP server and
start the desktop app with the Hermes environment loaded. The exact gateway
configuration depends on your local Hermes setup, but the required pieces are:

- a local Hamba desktop/frontend bridge,
- a Hermes API/gateway model configuration,
- an MCP server entry that runs `hambajuba-hermes-mcp`,
- tool access for the Hamba MCP tools,
- the `hambajuba-dance-director` skill installed.

## Tool Workflow

For a fresh song setup, Hermes should:

1. call `hamba_get_state`,
2. call `hamba_get_song_analysis`,
3. prepare or read a feature palette when Auto Dance will be used,
4. set prompt and latent destinations,
5. apply sparse block configs with `sae_rank: 1`.

For live playback or user steering, Hermes should:

1. call `hamba_get_state`,
2. call `hamba_get_music_window` when current or upcoming musical behavior
   matters,
3. use prepared palette candidates before live feature hunting in Auto Dance,
4. apply durable steering without beat-perfect staleness metadata.

See [HERMES_OPERATOR_GUIDE.md](HERMES_OPERATOR_GUIDE.md) for the detailed
instrument vocabulary, legal action schema, and planning guidance.
