"""Frame transport types for WebSocket delivery."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional, Union


@dataclass
class FrameItem:
    """Container for a generated frame or telemetry message.

    Strategies produce lists of FrameItems that the consumer sends
    over WebSocket. Binary frames go directly, JSON is serialized.

    Attributes:
        kind: "frame" for binary image data, "json" for telemetry
        payload: JPEG bytes (frame) or dict (json)
        due_ts: Optional scheduled send time (for rhythmic pacing)
    """

    kind: Literal["frame", "json"]
    payload: Union[bytes, dict, Any]
    due_ts: Optional[float] = None
    produced_at: Optional[float] = None  # perf_counter() when frame began generation
