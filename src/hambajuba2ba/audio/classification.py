"""Component classification for audio-to-physics mapping.

Each stem/virtual-stem gets classified along multiple axes:
1. Texture: harmonic vs percussive (from HPSS)
2. Role: rhythm vs melody vs texture
3. Frequency range: bass vs mid vs high

These classifications drive physics model selection and blending.
A percussive bass stem (like kick drum) needs different physics
than a harmonic high stem (like strings).

Design:
- Multi-label with confidence scores (not mutually exclusive)
- Immutable dataclass for thread safety
- physics_weights() translates classification → physics blending
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .features import StemFeatures


@dataclass(frozen=True)
class ComponentClassification:
    """Multi-label classification with confidence scores.

    Texture (should sum to ~1.0):
        percussive_confidence: Transient, noise-like content
        harmonic_confidence: Sustained, tonal content

    Role (can overlap, no sum constraint):
        rhythm_confidence: Acts as timekeeper (drums, percussion)
        melody_confidence: Carries pitched contour (vocals, lead)
        harmony_confidence: Chordal, static pitch (pads, sustained chords)
        texture_confidence: Provides ambient fill (noise, reverb tails)

    Frequency Range (can overlap):
        is_bass: Significant energy below 250Hz
        is_mid: Significant energy 250Hz-4kHz
        is_high: Significant energy above 4kHz
    """

    # Texture (HPSS-derived)
    percussive_confidence: float
    harmonic_confidence: float

    # Role
    rhythm_confidence: float
    melody_confidence: float
    harmony_confidence: float
    texture_confidence: float

    # Frequency range
    is_bass: bool
    is_mid: bool
    is_high: bool

    def __post_init__(self):
        """Validate confidence scores are in [0, 1]."""
        for field_name in (
            "percussive_confidence",
            "harmonic_confidence",
            "rhythm_confidence",
            "melody_confidence",
            "harmony_confidence",
            "texture_confidence",
        ):
            value = getattr(self, field_name)
            if not (0.0 <= value <= 1.0):
                raise ValueError(f"{field_name} must be in [0, 1], got {value}")

    def physics_weights(self) -> dict[str, float]:
        """Compute weights for blending physics models.

        Returns weights for four physics behaviors:
        - "spring": SteeringPhysics (percussive, impulse-driven)
        - "pitch_follow": PitchFollowingPhysics (melodic contour)
        - "oscillator": CoupledOscillatorPhysics (harmonic chords, pads)
        - "perlin": PerlinDriftPhysics (texture, ambient)

        Weights sum to 1.0.

        Key design: Harmonic roles (pitch_follow, oscillator, perlin) are gated
        by harmonic_confidence. A purely percussive hit routes 100% to spring
        regardless of role confidences.
        """
        # Raw contributions: percussive goes to spring, harmonic roles are gated
        spring = self.percussive_confidence
        pitch_follow = self.melody_confidence * self.harmonic_confidence
        oscillator = self.harmony_confidence * self.harmonic_confidence
        perlin = self.texture_confidence * self.harmonic_confidence

        total = spring + pitch_follow + oscillator + perlin
        if total < 0.01:
            # Fallback: pure spring (safe default for any content)
            return {"spring": 1.0, "pitch_follow": 0.0, "oscillator": 0.0, "perlin": 0.0}

        return {
            "spring": spring / total,
            "pitch_follow": pitch_follow / total,
            "oscillator": oscillator / total,
            "perlin": perlin / total,
        }

    @property
    def dominant_texture(self) -> str:
        """Return the dominant texture type."""
        if self.percussive_confidence > self.harmonic_confidence:
            return "percussive"
        return "harmonic"

    @property
    def dominant_role(self) -> str:
        """Return the dominant role."""
        roles = {
            "rhythm": self.rhythm_confidence,
            "melody": self.melody_confidence,
            "harmony": self.harmony_confidence,
            "texture": self.texture_confidence,
        }
        return max(roles, key=roles.get)

    @property
    def frequency_range(self) -> str:
        """Return frequency range as string (e.g., 'bass', 'mid-high')."""
        parts = []
        if self.is_bass:
            parts.append("bass")
        if self.is_mid:
            parts.append("mid")
        if self.is_high:
            parts.append("high")
        return "-".join(parts) if parts else "full"


def classify_component(features: "StemFeatures") -> ComponentClassification:
    """Auto-classify an audio component based on its extracted features.

    Uses HPSS ratio, onset density, pitch confidence, and spectral features
    to determine texture (percussive/harmonic), role (rhythm/melody/harmony/texture),
    and frequency range (bass/mid/high).

    Args:
        features: Extracted audio features from StemAnalyzer

    Returns:
        ComponentClassification with confidence scores for physics selection
    """
    import numpy as np

    # === Texture: HPSS ratio (0=harmonic, 1=percussive) ===
    percussive_conf = features.hpss_ratio if features.hpss_ratio is not None else 0.5
    harmonic_conf = 1.0 - percussive_conf

    # === Role: Rhythm (onset density, >4/sec = rhythmic) ===
    onset_density = len(features.onsets) / max(features.duration, 0.1)
    rhythm_conf = float(np.clip(onset_density / 4.0, 0, 1))

    # === Role: Melody (pitch confidence, high = melodic content) ===
    if features.pitch_confidence is not None and len(features.pitch_confidence) > 0:
        melody_conf = float(np.mean(features.pitch_confidence > 0.5))
    else:
        melody_conf = 0.0

    # === Role: Texture (spectral flatness, high = noise-like) ===
    if features.flatness is not None and len(features.flatness) > 0:
        texture_conf = float(np.mean(features.flatness > 0.4))
    else:
        texture_conf = 0.0

    # === Role: Harmony (harmonic but NOT melodic, NOT texture) ===
    # Pads, sustained chords: harmonic content without melodic movement or noise
    harmony_conf = harmonic_conf * (1.0 - melody_conf) * (1.0 - texture_conf)

    # === Frequency range from brightness (spectral centroid) ===
    # brightness is normalized [0,1], denormalize assuming ~20kHz max
    if features.brightness is not None and len(features.brightness) > 0:
        brightness_hz = float(np.mean(features.brightness)) * 20000
    else:
        brightness_hz = 1000.0  # Default to mid-range

    is_bass = brightness_hz < 250
    is_high = brightness_hz > 4000
    is_mid = not is_bass and not is_high

    return ComponentClassification(
        percussive_confidence=percussive_conf,
        harmonic_confidence=harmonic_conf,
        rhythm_confidence=rhythm_conf,
        melody_confidence=melody_conf,
        harmony_confidence=harmony_conf,
        texture_confidence=texture_conf,
        is_bass=is_bass,
        is_mid=is_mid,
        is_high=is_high,
    )
