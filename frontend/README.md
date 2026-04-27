# hambajuba2ba frontend

React + TypeScript frontend for the audio-reactive visualizer. Physics-based UI with 3D orbs, living tendrils, and real-time WebSocket frames.

## Stack

- **React 18** + **TypeScript** — UI framework
- **Vite** — Build tool
- **Zustand** — State management (8 stores)
- **React Three Fiber** — 3D rendering (Three.js)
- **Matter.js** — 2D physics simulation
- **Web Audio API** — Stem playback + mixing

## Structure

```
src/
├── App.tsx                       # Main orchestrator (~750 lines)
│
├── components/
│   ├── AudioPlayerWindow.tsx     # Upload, playback, stem mixer
│   ├── Canvas.tsx                # Video frame display (JPEG from WS)
│   ├── OrbSystem.tsx             # Matter.js stem orbs + heart layout
│   ├── FlowerOrb.tsx             # 3D orb with GLSL shaders (R3F)
│   ├── CrystalHeart.tsx          # Central crystal (R3F)
│   ├── PlantStems.tsx            # Organic tendril connections
│   ├── Win95Window.tsx           # Retro window chrome
│   ├── HelpDialog.tsx            # Help overlay
│   ├── Notifications.tsx         # Toast notifications
│   ├── ErrorBoundary.tsx         # React error boundary
│   │
│   ├── destinations/             # Destination control panels
│   │   ├── CompositionPanel.tsx  # Latent: seeds + circle radius + mode
│   │   ├── PromptDestinationPanel.tsx  # Prompt: A/B + blend + reactive config
│   │   └── shared/               # BlendSlider, ReactiveConfigSection, Win95Select
│   │
│   └── v2/                       # SAE block config UI
│       ├── BlockConfigPanel.tsx  # Per-block: link target, strength, rank
│       ├── LinkTargetSelect.tsx  # Audio source dropdown
│       ├── StrengthRangeSlider.tsx
│       └── AutoManualToggle.tsx
│
├── hooks/
│   ├── useWebSocket.ts           # WS connection, frame/JSON dispatch
│   ├── useAudioMixer.ts          # Web Audio stem playback + mixing
│   ├── useMatterPhysics.ts       # Matter.js engine + useSyncExternalStore
│   ├── useBlockConfigHandlers.ts # Block config event handlers
│   ├── useDestinationHandlers.ts # Destination event handlers
│   ├── useCanvasSampling.ts      # Canvas color sampling for lighting
│   └── animations.ts             # Animation utility functions
│
├── stores/
│   ├── useAudioStore.ts          # Audio state (duration, time, playing)
│   ├── useAudioActivityStore.ts  # Per-stem activity levels
│   ├── useBlockStore.ts          # SAE block configs (link target, strength, rank)
│   ├── useDestinationStore.ts    # Destination A/B state
│   ├── usePerfStore.ts           # Performance telemetry
│   ├── useCanvasLightingStore.ts # Canvas color → lighting
│   └── useNotificationStore.ts   # Toast notifications
│
├── data/
│   ├── features.ts               # SAE feature definitions + STEM_COLORS
│   └── options.ts                # Shared select options (position, intensity, etc.)
│
├── types/
│   ├── sae.ts                    # SAE block types, TrackInfo, ExtendedStemActivity
│   └── destinations.ts           # Destination types, ReactiveConfig
│
├── types.ts                      # Core types (ConnectionStatus, etc.)
├── constants.ts                  # WS config, perf config
└── shaders/
    └── shaderUtils.ts            # Shared GLSL noise functions
```

## WebSocket Protocol

### Client → Server

```typescript
{ action: "start_sae_steering", audio_id: "abc123" }
{ action: "audio_timeupdate", time: 45.2 }          // 10Hz sync
{ action: "update_block_config", block: "down.2.1", link_target: "drums", ... }
{ action: "set_destination", space: "prompt", slot: "b", destination_type: "prompt", prompt: "..." }
{ action: "set_composition_config", distance: 1.5, mode: "auto" }
{ action: "set_blend_position", space: "prompt", position: 0.7 }
```

### Server → Client

```typescript
// Binary: JPEG frames (~30KB each, ~50 FPS)
// JSON:
{ type: "extended_activity", audio_time: 45.2, stems: {...}, prominence: {...} }
{ type: "destination_status", space: "prompt", blend_position: 0.7, mode: "slider" }
{ type: "block_configs", configs: { "down.2.1": {...}, ... } }
{ type: "track_info", audio_id: "abc", duration: 180, bpm: 120, stems: [...] }
```

## Development

```bash
npm install       # Install dependencies
npm run dev       # Start dev server
npm run build     # Production build
npm run typecheck # TypeScript check
npm run test:run  # Run tests (vitest)
```

## Key Patterns

- **useSyncExternalStore** for Matter.js → avoids 60fps setState re-renders
- **Callback refs** in useWebSocket → avoids reconnections on parent re-render
- **Handler extraction** (useBlockConfigHandlers, useDestinationHandlers) → keeps App.tsx manageable
- **Binary + JSON on same WS** → frames as ArrayBuffer, telemetry as JSON text
- **10Hz time sync** → frontend is audio authority, backend uses PLL clock to stay in sync
