"""Audio feature sampling utilities.

Stateless utilities for sampling pre-computed audio features at runtime.
All heavy DSP runs at upload time; runtime is O(1) array lookups.

v3: Includes structural novelty/layer sampling for ProminenceEngine.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional, Tuple

if TYPE_CHECKING:
    from hambajuba2ba.audio.features import StemFeatures
    from hambajuba2ba.audio.coupling import CrossStemFeatures
    from hambajuba2ba.audio.classification import ComponentClassification


class AudioSampler:
    """Stateless audio feature sampling.

    All methods are pure functions that sample from pre-computed
    StemFeatures arrays. No state is modified.

    Usage:
        sampler = AudioSampler(stem_features, fps=60)
        value = sampler.sample_stem("bass", 1.5, "energy_smooth", "combined")
    """

    def __init__(
        self,
        stem_features: Dict[str, "StemFeatures"],
        fps: float = 60.0,
        cross_stem_features: Optional["CrossStemFeatures"] = None,
        classifications: Optional[Dict[str, "ComponentClassification"]] = None,
    ):
        """Initialize audio sampler.

        Args:
            stem_features: Pre-computed features per stem
            fps: Frames per second (for time→index conversion)
            cross_stem_features: Pre-computed cross-stem coupling (optional)
            classifications: Pre-computed stem classifications (optional)
        """
        self.stem_features = stem_features
        self.fps = fps
        self.cross_stem_features = cross_stem_features
        self._classifications = classifications or {}

    def sample_stem(
        self,
        stem: str,
        audio_time: float,
        layer: str = "combined",
    ) -> float:
        """Sample feature value for a stem using layer (flash/sustain/combined).

        Args:
            stem: Stem name ("bass", "drums", etc.)
            audio_time: Playback time in seconds
            layer: Which layer ("flash", "sustain", "combined")

        Returns:
            Feature value in [0, 1]
        """
        if stem not in self.stem_features:
            return 0.0

        features = self.stem_features[stem]

        if layer == "flash":
            return features.sample_at_time(audio_time, "flash")
        elif layer == "sustain":
            return features.sample_at_time(audio_time, "sustain")
        else:
            # Combined: average of flash and sustain (comet-tail effect)
            flash = features.sample_at_time(audio_time, "flash")
            sustain = features.sample_at_time(audio_time, "sustain")
            return (flash + sustain) / 2.0

    def sample_position(
        self,
        stem: Optional[str],
        audio_time: float,
        source: str = "auto",
        pitch_conf_threshold: float = 0.3,
    ) -> Tuple[Optional[float], bool]:
        """Sample position value (where in range this stem wants to be).

        Returns (value, valid). If valid is False, caller may treat as silence.
        """
        if source == "tension_global":
            return self.sample_aggregate_tension(audio_time), True

        if source == "tonal_distance_global":
            return self.sample_aggregate_tonal_distance(audio_time), True

        if stem is None or stem not in self.stem_features:
            return None, False

        features = self.stem_features[stem]

        if source == "auto":
            # Per-stem defaults
            if stem == "other" and features.chroma_centroid is not None:
                value = features.sample_at_time(audio_time, "chroma_centroid")
                return value, True

            conf = features.sample_at_time(audio_time, "pitch_confidence") if features.pitch_confidence is not None else 0.0
            if conf > pitch_conf_threshold and features.pitch_normalized is not None:
                value = features.sample_at_time(audio_time, "pitch_normalized")
                return value, True

            # Fallback to brightness
            value = features.sample_at_time(audio_time, "brightness")
            return value, True

        if source == "pitch":
            conf = features.sample_at_time(audio_time, "pitch_confidence") if features.pitch_confidence is not None else 0.0
            if conf > pitch_conf_threshold and features.pitch_normalized is not None:
                return features.sample_at_time(audio_time, "pitch_normalized"), True
            return None, False

        if source == "brightness":
            return features.sample_at_time(audio_time, "brightness"), True

        if source == "chroma":
            if features.chroma_centroid is None:
                return None, False
            return features.sample_at_time(audio_time, "chroma_centroid"), True

        if source == "tension":
            if features.tension is None:
                return None, False
            return features.sample_at_time(audio_time, "tension"), True

        if source == "tonal_distance":
            if features.tonal_distance is None:
                return None, False
            return features.sample_at_time(audio_time, "tonal_distance"), True

        # Unknown source
        return None, False

    def sample_intensity(
        self,
        stem: Optional[str],
        audio_time: float,
        source: str = "energy_smooth",
    ) -> float:
        """Sample intensity value (how strongly this stem manifests)."""
        if stem is None or stem not in self.stem_features:
            return 0.0

        channel = source or "energy_smooth"
        return self.stem_features[stem].sample_at_time(audio_time, channel)

    def get_extended_activity(self, audio_time: float) -> Dict[str, Dict[str, float]]:
        """Get all channels for all stems (for audio-reactive UI).

        Returns all 8 stems (4 physical + 4 virtual bandpass filtered).

        Args:
            audio_time: Current audio playback time

        Returns:
            Nested dict: {stem: {channel: value}}
        """
        channels = [
            "envelope", "energy_smooth", "transient",
            "flux", "brightness", "flash", "sustain"
        ]
        all_stems = [
            "bass", "drums", "vocals", "other",
            "drums_low", "drums_mid", "drums_high", "other_mid", "other_high"
        ]

        result = {}
        for stem in all_stems:
            if stem in self.stem_features:
                features = self.stem_features[stem]
                result[stem] = {
                    ch: features.sample_at_time(audio_time, ch)
                    for ch in channels
                }
            else:
                result[stem] = {ch: 0.0 for ch in channels}

        return result

    def time_to_frame(self, audio_time: float) -> int:
        """Convert audio time to frame index, clamped to valid range.

        Args:
            audio_time: Time in seconds

        Returns:
            Frame index in [0, max_frames]
        """
        frame = round(audio_time * self.fps)
        if frame < 0:
            return 0
        # Clamp to longest stem's array length
        max_frames = 0
        for features in self.stem_features.values():
            if features.energy_smooth is not None:
                max_frames = max(max_frames, len(features.energy_smooth) - 1)
                break  # All stems have same length
        if max_frames > 0 and frame > max_frames:
            return max_frames
        return frame

    def sample_pitch(self, stem: str, audio_time: float) -> Tuple[float, float]:
        """Sample pitch_hz and pitch_confidence for a stem.

        Args:
            stem: Stem name
            audio_time: Playback time in seconds

        Returns:
            Tuple of (pitch_hz, confidence). Returns (0.0, 0.0) if unavailable.
        """
        features = self.stem_features.get(stem)
        if features is None:
            return 0.0, 0.0

        if features.pitch_hz is None or features.pitch_confidence is None:
            return 0.0, 0.0

        pitch_hz = features.sample_at_time(audio_time, "pitch_hz")
        confidence = features.sample_at_time(audio_time, "pitch_confidence")
        return float(pitch_hz), float(confidence)

    def sample_tension(self, stem: str, audio_time: float) -> float:
        """Sample tension value for a stem.

        Tension measures harmonic dissonance/roughness: higher = more tension.
        Useful for modulating SLERP blend position or visual intensity.
        """
        if stem not in self.stem_features:
            return 0.5
        features = self.stem_features[stem]
        if features.tension is None:
            return 0.5
        return features.sample_at_time(audio_time, "tension")

    def sample_pitch_normalized(self, stem: str, audio_time: float) -> float:
        """Sample normalized pitch (0-1 scaled for spatial mapping).

        Useful for driving spatial mask Y-offset (high pitch = up).

        Args:
            stem: Stem name
            audio_time: Playback time in seconds

        Returns:
            Pitch in [0, 1]. Returns 0.5 (middle) if unavailable.
        """
        if stem not in self.stem_features:
            return 0.5
        features = self.stem_features[stem]
        if features.pitch_normalized is None:
            return 0.5
        return features.sample_at_time(audio_time, "pitch_normalized")

    def sample_lock_index(
        self, stem1: str, stem2: str, frame_idx: int
    ) -> float:
        """Get coupling strength between two stems.

        Lock index combines envelope correlation, PLV, and onset synchrony.
        > 0.7 = "in the pocket", < 0.3 = independent.

        Args:
            stem1: First stem name
            stem2: Second stem name
            frame_idx: Frame index (not time)

        Returns:
            Lock index in [0, 1]. Returns 0.0 if unavailable.
        """
        if self.cross_stem_features is None:
            return 0.0
        return self.cross_stem_features.get_lock_index(stem1, stem2, frame_idx)

    def sample_call_response(
        self, stem1: str, stem2: str, frame_idx: int
    ) -> float:
        """Get call-response score between two stems.

        High when one stem is active while the other is quiet (alternation).
        Useful for focus alternation visual effects.

        Args:
            stem1: First stem name
            stem2: Second stem name
            frame_idx: Frame index (not time)

        Returns:
            Call-response score in [0, 1]. Returns 0.0 if unavailable.
        """
        if self.cross_stem_features is None:
            return 0.0
        return self.cross_stem_features.get_call_response(stem1, stem2, frame_idx)

    def get_classification(self, stem: str) -> Optional["ComponentClassification"]:
        """Get the pre-computed classification for a stem.

        Classifications describe the stem's musical character:
        - percussive_confidence: how rhythmic/transient-heavy
        - harmonic_confidence: how tonal/melodic
        - is_bass, is_high: frequency range flags

        Useful for auto-config derivation (channel, layer, spatial selection).

        Args:
            stem: Stem name

        Returns:
            ComponentClassification or None if not available.
        """
        return self._classifications.get(stem)

    def sample_harmonic_energy(self, stem: str, audio_time: float) -> float:
        """Sample harmonic component energy for a stem.

        Used for HPSS component link targets (e.g., drums_harmonic).

        Args:
            stem: Base stem name ("drums", "bass", etc.)
            audio_time: Playback time in seconds

        Returns:
            Harmonic energy in [0, 1]. Returns 0.0 if unavailable.
        """
        if stem not in self.stem_features:
            return 0.0
        features = self.stem_features[stem]
        if features.harmonic_energy is None:
            return 0.0
        return features.sample_at_time(audio_time, "harmonic_energy")

    def sample_percussive_energy(self, stem: str, audio_time: float) -> float:
        """Sample percussive component energy for a stem.

        Used for HPSS component link targets (e.g., drums_percussive).

        Args:
            stem: Base stem name ("drums", "bass", etc.)
            audio_time: Playback time in seconds

        Returns:
            Percussive energy in [0, 1]. Returns 0.0 if unavailable.
        """
        if stem not in self.stem_features:
            return 0.0
        features = self.stem_features[stem]
        if features.percussive_energy is None:
            return 0.0
        return features.sample_at_time(audio_time, "percussive_energy")

    def sample_aggregate_tension(self, audio_time: float) -> float:
        """Sample aggregate tension across all stems.

        Used for the 'tension' virtual link target.
        Computes weighted average of tension across stems with harmonic content.

        Args:
            audio_time: Playback time in seconds

        Returns:
            Aggregate tension in [0, 1].
        """
        tensions = []
        weights = []

        for stem in self.stem_features.keys():
            features = self.stem_features[stem]
            if features.tension is not None:
                t = features.sample_at_time(audio_time, "tension")
                # Weight by energy (louder stems contribute more)
                e = features.sample_at_time(audio_time, "energy_smooth")
                tensions.append(t)
                weights.append(e + 0.1)  # Add small base weight

        if not tensions:
            return 0.5

        total_weight = sum(weights)
        weighted_sum = sum(t * w for t, w in zip(tensions, weights))
        return weighted_sum / total_weight

    def sample_tonal_distance(self, stem: str, audio_time: float) -> float:
        """Sample tonal distance for a stem.

        Tonal distance measures harmonic departure from track average (JSD).
        Higher = more distant from the song's tonal center.
        Drives prompt SLERP: home key → prompt A, departure → prompt B.
        """
        if stem not in self.stem_features:
            return 0.0
        features = self.stem_features[stem]
        if features.tonal_distance is None:
            return 0.0
        return features.sample_at_time(audio_time, "tonal_distance")

    def sample_aggregate_tonal_distance(self, audio_time: float) -> float:
        """Sample aggregate tonal distance across stems with harmonic content.

        Energy-weighted average: louder stems contribute more.
        """
        distances = []
        weights = []
        for stem, features in self.stem_features.items():
            if features.tonal_distance is not None:
                d = features.sample_at_time(audio_time, "tonal_distance")
                e = features.sample_at_time(audio_time, "energy_smooth")
                distances.append(d)
                weights.append(e + 0.1)

        if not distances:
            return 0.0

        total = sum(weights)
        return sum(d * w for d, w in zip(distances, weights)) / total

    def sample_link_target(
        self,
        link_target: str,
        audio_time: float,
        layer: str = "combined",
    ) -> float:
        """Sample audio value for any LinkTarget type.

        Unified sampling method for v2 system. Handles:
        - Physical stems: bass, drums, vocals, other
        - HPSS components: *_harmonic, *_percussive
        - Sub-bands: drums_low, drums_high, other_mid, other_high
        - Derived: tension, global

        Args:
            link_target: LinkTarget value as string
            audio_time: Playback time in seconds
            layer: Response layer (flash, sustain, combined)

        Returns:
            Audio value in [0, 1]
        """
        # Physical stems (direct lookup)
        if link_target in ("bass", "drums", "vocals", "other"):
            return self.sample_stem(link_target, audio_time, layer)

        # Sub-bands (virtual stems from bandpass filtering)
        if link_target in ("drums_low", "drums_mid", "drums_high", "other_mid", "other_high"):
            return self.sample_stem(link_target, audio_time, layer)

        # HPSS components (harmonic/percussive variants)
        if link_target.endswith("_harmonic"):
            base_stem = link_target.replace("_harmonic", "")
            return self.sample_harmonic_energy(base_stem, audio_time)

        if link_target.endswith("_percussive"):
            base_stem = link_target.replace("_percussive", "")
            return self.sample_percussive_energy(base_stem, audio_time)

        # Derived: aggregate tension
        if link_target == "tension":
            return self.sample_aggregate_tension(audio_time)

        # Derived: aggregate tonal distance
        if link_target == "tonal_distance":
            return self.sample_aggregate_tonal_distance(audio_time)

        # Derived: global (average of all physical stems)
        if link_target == "global":
            total = 0.0
            count = 0
            for stem in ("bass", "drums", "vocals", "other"):
                if stem in self.stem_features:
                    total += self.sample_stem(stem, audio_time, layer)
                    count += 1
            return total / max(1, count)

        # Unknown target - return 0
        return 0.0

    def compute_coupling_bonus(
        self,
        stem: str,
        frame_idx: int,
        focused_stems: list,
        max_bonus: float = 0.5,
    ) -> float:
        """Compute coupling bonus for a stem based on lock_index with other focused stems.

        When multiple stems are focused and they're rhythmically locked (lock_index > 0.5),
        they reinforce each other visually. This creates a cohesive "pulse together" effect.

        Args:
            stem: The stem to compute bonus for
            frame_idx: Current frame index
            focused_stems: List of all stems with focus_weight > 0
            max_bonus: Maximum total bonus (default 0.5 = +50%)

        Returns:
            Coupling bonus in [0, max_bonus]
        """
        if self.cross_stem_features is None:
            return 0.0

        coupling_bonus = 0.0

        for other in focused_stems:
            if other == stem:
                continue

            lock_idx = self.sample_lock_index(stem, other, frame_idx)
            if lock_idx > 0.5:
                # Proportional bonus based on lock strength
                coupling_bonus += (lock_idx - 0.5) * 2.0 * max_bonus

        # Cap at max_bonus
        return min(coupling_bonus, max_bonus)

    # ─────────────────────────────────────────────────────────────────────────
    # Structural Awareness: Novelty & Layer Detection
    # ─────────────────────────────────────────────────────────────────────────

    def sample_novelty(
        self,
        stem: str,
        audio_time: float,
        timescale: str = "short",
    ) -> float:
        """Sample novelty at a specific timescale.

        Novelty measures how different the current moment is from the past.
        - short: 0.5-2s window, detects transients/fills
        - medium: 4-8s window, detects phrase boundaries
        - long: 16-32s window, detects section changes
        """
        if stem not in self.stem_features:
            return 0.0

        channel = f"novelty_{timescale}"
        return self.stem_features[stem].sample_at_time(audio_time, channel)

    def sample_novelty_derivative(
        self,
        stem: str,
        audio_time: float,
        timescale: str = "short",
    ) -> float:
        """Sample novelty derivative (rising/falling edge detection).

        Positive derivative = novelty increasing = event starting
        Use for triggering SAE bursts or SLERP movements.

        Args:
            stem: Stem name
            audio_time: Playback time in seconds
            timescale: 'short', 'medium', or 'long'

        Returns:
            Derivative in roughly [-1, 1]. Positive = rising novelty.
        """
        if stem not in self.stem_features:
            return 0.0

        channel = f"novelty_{timescale}_deriv"
        return self.stem_features[stem].sample_at_time(audio_time, channel)

    def sample_layer_entry(self, stem: str, frame_idx: int) -> bool:
        """Check if a new instrumental layer enters at this frame.

        Layer entries are detected via normalized flux + flatness change.
        Use for triggering SLERP movements or SAE feature activation.

        Args:
            stem: Stem name
            frame_idx: Frame index (not time)

        Returns:
            True if a layer entry is detected at this frame.
        """
        features = self.stem_features.get(stem)
        if features is None or features.layer_entry_mask is None:
            return False

        if frame_idx < 0 or frame_idx >= len(features.layer_entry_mask):
            return False

        return bool(features.layer_entry_mask[frame_idx])

    def compute_activity_gate(
        self,
        stem: str,
        audio_time: float,
        threshold_db: float = -40.0,
    ) -> float:
        """Compute activity gate for a stem (silent stems fade out).

        Returns a value in [0, 1] where:
        - 0 = stem is silent (below threshold)
        - 1 = stem is active (at or above 0 dB reference)

        Args:
            stem: Stem name
            audio_time: Playback time in seconds
            threshold_db: dB threshold below which gate closes (default -40)

        Returns:
            Activity gate in [0, 1]
        """
        if stem not in self.stem_features:
            return 0.0

        features = self.stem_features[stem]
        if features.energy_db is None:
            # Fallback to energy_smooth
            return features.sample_at_time(audio_time, "energy_smooth")

        energy_db = features.sample_at_time(audio_time, "energy_db")

        # Map from [threshold_db, 0] to [0, 1]
        if energy_db <= threshold_db:
            return 0.0
        elif energy_db >= 0:
            return 1.0
        else:
            return (energy_db - threshold_db) / (0 - threshold_db)
