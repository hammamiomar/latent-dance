"""SAE steering computation from audio features.

Core computation that converts audio features into SAE steering configs.
v5: Physics-driven + auto-prominence model.

Each SAE block gets one value from BlendedPhysics (spring/pitch_follow/oscillator/perlin
weighted by ComponentClassification), mapped to [strength_min, strength_max],
then scaled by ProminenceEngine for automatic musical emphasis.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict, Optional

from hambajuba2ba.audio.focus_config import BlockLinkConfig, get_base_stem
from hambajuba2ba.audio.prominence import ProminenceEngine, compute_all_prominences
from hambajuba2ba.bridge.destinations import apply_intensity_curve

if TYPE_CHECKING:
    from hambajuba2ba.audio.classification import ComponentClassification
    from hambajuba2ba.audio.sampler import AudioSampler
    from hambajuba2ba.bridge.physics_manager import PhysicsManager

logger = logging.getLogger("uvicorn")


class SteeringComputation:
    """Computes SAE steering from audio features via physics + auto-prominence.

    v5 (Physics + Auto-Prominence):
    - BlendedPhysics provides ONE value per block with the right motion character
      (spring for drums, pitch_follow for vocals, oscillator for pads, perlin for texture)
    - Value is mapped to [strength_min, strength_max]
    - ProminenceEngine scales by musical context (novelty, coupling, activity gate, surprise)
    - No more PositionFollower or apply_intensity_curve for SAE blocks

    Usage:
        computation = SteeringComputation()
        computation.initialize(["bass", "drums", "vocals", "other"])

        # In frame loop:
        steering, activity = computation.compute(
            block_configs=configs,
            audio_sampler=sampler,
            physics=physics_manager,
            audio_time=1.5,
            dt=0.033,
            classifications=classifications,
        )
    """

    def __init__(self):
        """Initialize steering computation."""
        self._prominence_engine = ProminenceEngine()

    def initialize(self, stems: list) -> None:
        """Initialize state for all stems.

        Args:
            stems: List of stem names
        """
        self._prominence_engine.reset()

    def compute(
        self,
        block_configs: Dict[str, BlockLinkConfig],
        audio_sampler: "AudioSampler",
        physics: "PhysicsManager",
        audio_time: float,
        dt: float,
        classifications: Optional[Dict[str, "ComponentClassification"]] = None,
    ) -> tuple[Dict[str, tuple], Dict[str, Dict[str, float]]]:
        """Compute steering using BlendedPhysics + auto-prominence.

        For each enabled SAE block:
        1. Sample raw audio features (energy, pitch) for the linked stem
        2. Step BlendedPhysics → one value [0,1] with correct motion character
        3. Map to strength range: strength_min + physics_value * (max - min)
        4. Scale by auto-prominence for dynamic musical emphasis

        Args:
            block_configs: Block name → BlockLinkConfig
            audio_sampler: Audio feature sampler
            physics: Physics simulation manager
            audio_time: Current audio playback time
            dt: Time since last frame
            classifications: Optional stem classifications for auto-config

        Returns:
            Tuple:
                - Dict of {block: (feature_id, strength)} for pipeline
                - Dict of {block: activity_dict} for UI telemetry
        """
        steering: Dict[str, tuple] = {}
        block_activity: Dict[str, Dict[str, float]] = {}

        active_configs = {b: c for b, c in block_configs.items() if c.enabled}
        if not active_configs:
            return {}, {}

        # Compute auto-prominence for all stems (one call, reused across blocks)
        stem_ranks: Dict[str, Optional[int]] = {}
        for config in active_configs.values():
            base = get_base_stem(config.link_target)
            if base:
                stem_ranks[base] = config.sae_rank

        frame_idx = audio_sampler.time_to_frame(audio_time)
        prominence_map = compute_all_prominences(
            engine=self._prominence_engine,
            stem_ranks=stem_ranks,
            audio_sampler=audio_sampler,
            audio_time=audio_time,
            frame_idx=frame_idx,
            dt=dt,
        )

        for block, config in active_configs.items():
            stem = config.link_target
            base_stem = get_base_stem(stem)

            # 1. Sample raw audio features
            energy = audio_sampler.sample_intensity(stem, audio_time, config.intensity_source)
            if base_stem:
                pitch_hz, pitch_conf = audio_sampler.sample_pitch(base_stem, audio_time)
            else:
                pitch_hz, pitch_conf = 0.0, 0.0

            # 2. BlendedPhysics — one value with the right motion character
            #    PhysicsManager routes to BlendedPhysics (multi-model) or
            #    SteeringPhysics (legacy fallback) based on classification
            physics_value = physics.step(
                stem, energy, dt,
                pitch_hz=pitch_hz,
                pitch_confidence=pitch_conf,
                energy=energy,
            )

            # 2b. Apply intensity curve if non-linear
            if config.intensity_curve in ("gamma", "clip"):
                physics_value, _ = apply_intensity_curve(
                    physics_value, config.intensity_curve, config.intensity_gamma, dt, None
                )

            # 3. Map to strength range
            strength = config.strength_min + physics_value * (config.strength_max - config.strength_min)

            # 4. Scale by auto-prominence
            prominence = prominence_map.get(base_stem, 0.5) if base_stem else 0.5
            final_strength = strength * prominence

            steering[block] = (config.feature_id, final_strength)
            block_activity[block] = {
                "raw": float(energy),
                "physics": float(physics_value),
                "prominence": float(prominence),
                "strength": float(final_strength),
            }

        return steering, block_activity

    def get_prominence_engine(self) -> ProminenceEngine:
        """Get the prominence engine for external queries (e.g., telemetry)."""
        return self._prominence_engine

    def reset(self) -> None:
        """Reset all computation state."""
        self._prominence_engine.reset()
