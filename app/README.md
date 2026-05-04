# app/ — FastAPI server

WebSocket-driven backend for real-time audio-reactive generation. Pure server concerns — all core logic lives in `src/hambajuba2ba/`.

```bash
uv run uvicorn app.main:app --reload --port 8080
```

## Structure

```
app/
├── main.py                   # FastAPI app, lifespan, CORS
├── schemas.py                # Pydantic message schemas (Client ↔ Server)
├── websocket_manager.py      # WebSocket producer-consumer (RuthlessConsumer)
├── pipeline.py               # Pipeline lifecycle management
├── generation.py             # FrameItem transport type
├── caching.py                # Runtime TTL cache for active audio_id sessions
├── dependencies.py           # FastAPI dependency injection
│
├── routers/
│   ├── audio.py              # Upload, song library, activation, stem serving
│   └── streaming.py          # WS /ws — WebSocket endpoint
│
└── strategies/
    ├── base.py               # BaseStrategy ABC
    ├── sae_steering_strategy.py  # Main strategy: generation loop, audio→steering
    ├── handlers/
    │   ├── audio_handlers.py     # play/pause/seek/timeupdate
    │   ├── stem_handlers.py      # UpdateBlockConfig
    │   ├── destination_handlers.py # set_destination, blend, mode, composition
    │   └── modulation_handlers.py  # steering mode, seed
    └── managers/
        └── frame_manager.py      # Async JPEG encoding + WebSocket dispatch
```

## Message Flow

```
Frontend                    Backend
   |                           |
   |-- audio_timeupdate ------>|  (10Hz sync)
   |-- update_block_config --->|  (SAE steering config)
   |-- set_destination ------->|  (prompt/latent destinations)
   |-- set_composition_config->|  (circle radius, mode)
   |                           |
   |<---- binary JPEG frame ---|  (~50 FPS)
   |<---- extended_activity ---|  (~10Hz telemetry)
   |<---- destination_status --|  (~2Hz status)
   |<---- block_configs -------|  (on change)
```

## Key Design

- **RuthlessConsumer**: Drops old frames if frontend can't keep up. No buffering.
- **Strategy pattern**: `SaeSteeringStrategy` owns the generation loop. Handlers dispatch messages by type.
- **Persistent song library**: SQLite indexes cached song artifacts, but runtime
  generation still uses an in-memory `audio_id` payload.
- **No core logic here**: Physics, audio, SAE, composition all live in `src/hambajuba2ba/`. This is just routing and lifecycle.
