"""Strategy configuration."""
from dataclasses import dataclass


@dataclass
class StrategyConfig:
    """SAE steering strategy settings."""

    # Audio clock
    audio_clock_alpha: float = 0.5
    audio_clock_rate: float = 1.0

    # Estimated client-side latency (JPEG decode + browser paint).
    # Only manually-tuned constant in the lookahead system; everything
    # else is measured from FrameManager.
    client_render_ms: float = 15.0
