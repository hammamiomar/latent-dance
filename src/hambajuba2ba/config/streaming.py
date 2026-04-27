"""Streaming configuration."""
from dataclasses import dataclass


@dataclass
class StreamingConfig:
    """Real-time streaming settings."""

    # Maximum allowed FPS (ceiling). Used for queue sizing and producer pacing.
    # The actual scheduling rate comes from FrameManager.measured_fps, which
    # adapts to hardware throughput. Also sets DSP feature extraction rate.
    # At 60 fps, hop_length = 735 samples (~17ms) - captures most musical events.
    fps: float = 60.0
    jpeg_quality: int = 75
    queue_timeout: float = 0.2
    late_tolerance: float = 0.05
    control_queue_size: int = 32
    max_queue_frames_multiplier: float = 2.0
    drop_to_frames_multiplier: float = 1.0
