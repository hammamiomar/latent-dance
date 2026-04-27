"""Strategy managers.

FrameManager handles async frame encoding (server concern).
Other managers have moved to src/hambajuba2ba/:
- AudioSampler → hambajuba2ba.audio.sampler
- SteeringComputation → hambajuba2ba.bridge.steering
- PhysicsManager → hambajuba2ba.bridge.physics_manager
- SpatialManager → hambajuba2ba.bridge.spatial_manager
"""

from .frame_manager import FrameManager

__all__ = ["FrameManager"]
