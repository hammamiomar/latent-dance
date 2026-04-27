"""Bridge between audio and generation.

Connects audio features to visual generation:
- SteeringComputation: audio → SAE feature strengths
- PhysicsManager: per-stem physics simulation
- SpatialManager: per-block 16x16 spatial masks
- DestinationModulator: SLERP travel in prompt space
- CompositionEngine: noise circular walk (latent composition)
- AudioClock: BPM-synced frame pacing
"""

from .steering import SteeringComputation
from .physics_manager import PhysicsManager
from .spatial_manager import SpatialManager
from .destinations import DestinationModulator
from .composition import CompositionEngine
from .clock import AudioClock

__all__ = [
    "SteeringComputation",
    "PhysicsManager",
    "SpatialManager",
    "DestinationModulator",
    "CompositionEngine",
    "AudioClock",
]
