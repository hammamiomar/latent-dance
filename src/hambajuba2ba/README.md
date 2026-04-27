# hambajuba2ba core library

Real-time SDXL-Turbo inference with SAE steering, audio-reactive generation, and physics-driven spatial control.

## Structure

```
hambajuba2ba/
├── generation/               # ML inference (compiled CUDA graph)
│   ├── engine.py             # SDXLTurboEngine — compiled UNet+VAE, ~14ms/frame
│   ├── pipeline.py           # SAESteerablePipeline — model loading, prompt encoding
│   ├── encoding.py           # GPU→CPU tensor transfer + JPEG encoding
│   └── sae/                  # Sparse Autoencoder steering
│       ├── model.py          # SparseAutoencoder — weight loading (decoder-only)
│       └── inline.py         # InlineSAEManager, SteeredModule wrappers
│
├── audio/                    # Offline DSP (all at upload, O(1) at runtime)
│   ├── features.py           # StemAnalyzer (5-phase extraction), StemFeatures
│   ├── sampler.py            # AudioSampler — runtime O(1) feature lookup
│   ├── perceptual.py         # Asymmetric envelope, onset strength, spectral flux
│   ├── harmonic.py           # Tension, tonal distance, roughness, spectral entropy
│   ├── pitch.py              # PESTO/CREPE pitch tracking, interval features
│   ├── hpss.py               # GPU-accelerated harmonic-percussive separation
│   ├── virtual_stems.py      # Sub-band stems (drums_low/mid/high, other_mid/high)
│   ├── coupling.py           # Cross-stem: PLV, spectral overlap, call-response
│   ├── prominence.py         # Dancer Ensemble prominence engine (ranking + surprise)
│   ├── structure.py          # Multi-timescale novelty, layer detection
│   ├── classification.py     # Component classification (physics/spatial selection)
│   ├── focus_config.py       # BlockLinkConfig, DANCE_MODEL_DEFAULTS
│   ├── separator.py          # Demucs stem separation
│   ├── util.py               # Shared utilities (align_1d)
│   └── youtube.py            # YouTube audio download (optional)
│
├── bridge/                   # Audio → generation connection
│   ├── composition.py        # CompositionEngine — noise circular walk
│   ├── destinations.py       # DestinationModulator — prompt SLERP (slider/reactive/linked)
│   ├── steering.py           # SteeringComputation — audio → SAE strengths
│   ├── physics.py            # Spring/oscillator/perlin physics simulations
│   ├── physics_manager.py    # Per-stem BlendedPhysics orchestration
│   ├── spatial.py            # Pitch-indexed Gaussian masks for spatial steering
│   ├── spatial_manager.py    # Per-block spatial mask management
│   └── clock.py              # BPM-synced audio clock with drift correction
│
├── config/                   # All configuration dataclasses
│   ├── base.py               # PipelineConfig
│   ├── sae.py                # SAEConfig
│   ├── audio.py              # AudioConfig
│   ├── strategy.py           # StrategyConfig
│   ├── streaming.py          # StreamingConfig (fps=60)
│   └── loader.py             # Config loading from env/YAML
│
└── presets/                  # YAML preset files
    ├── envelope.yaml         # Per-stem envelope configs
    ├── brightness.yaml       # Spectral brightness presets
    ├── physics.yaml          # Spring/oscillator physics params (12 presets)
    └── dual_layer.yaml       # Flash/sustain dual-layer configs
```

## Key Concepts

**Three control axes:**

| Axis | What | Driven By | Module |
|------|------|-----------|--------|
| Composition | Noise buffer (~95% of image) | Beat grid + tonal drift | `CompositionEngine` |
| Semantics | Prompt embeddings | Crossfader / audio-reactive | `DestinationModulator` |
| Expression | SAE per-layer steering | Per-stem physics + prominence | `SteeringComputation` |

**SAE Steering**: Intercepts UNet attention blocks via `SteeredModule` wrappers. torch.compile-safe — no hooks, no graph breaks. Decoder-only loading saves ~104MB VRAM.

**Audio Pipeline**: All DSP (HPSS, onset detection, spectral features) runs at upload time. Runtime is O(1) array lookup via `AudioSampler`. Cross-stem coupling feeds the prominence engine.

**Composition**: `noise(theta) = cos(theta)*a + sin(theta)*b` — full circular walk driven by beat energy and tonal drift. `distance` parameter controls circle radius.

**Bridge Layer**: Translates audio features into steering strengths (physics-smoothed), spatial masks, and SLERP positions.

## Credits

SAE architecture from [sdxl-unbox](https://github.com/surkovv/sdxl-unbox) and [Interpreting and Steering Features in Images](https://www.lesswrong.com/posts/Quqekpvx8BGMMcaem/interpreting-and-steering-features-in-images).
