"""Per-stem physics simulation manager.

Manages physics simulations for each audio stem.
Supports two modes:
- BlendedPhysics: Multi-model blend based on classification (melodic, harmonic, texture)
- SteeringPhysics: Simple spring physics (legacy fallback)

Physics provides smooth, musical response to audio features.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict, Optional, Union

from hambajuba2ba.config.slots import get_base_stem
from hambajuba2ba.bridge.physics import (
    SteeringPhysics,
    BlendedPhysics,
    get_physics_preset,
    create_physics_from_classification,
)

if TYPE_CHECKING:
    from hambajuba2ba.config.slots import BlockLinkConfig
    from hambajuba2ba.audio import ComponentClassification, StemFeatures

logger = logging.getLogger("uvicorn")


class PhysicsManager:
    """Manages per-stem physics simulations.

    Two physics modes:
    - BlendedPhysics: Classification-driven multi-model blend for stems with HPSS data
    - SteeringPhysics: Simple spring physics (legacy fallback)

    Usage:
        manager = PhysicsManager(bpm=120)
        manager.initialize(mappings, stem_features, classifications)

        # In frame loop:
        smoothed = manager.step("bass", raw_value, dt, pitch_hz, pitch_conf, energy)
    """

    def __init__(self, bpm: float = 120.0):
        """Initialize physics manager.

        Args:
            bpm: Beats per minute for physics scaling
        """
        self.bpm = bpm
        self._simulations: Dict[str, Union[SteeringPhysics, BlendedPhysics]] = {}
        self._classifications: Dict[str, Optional["ComponentClassification"]] = {}

    def initialize(
        self,
        block_configs: Dict[str, "BlockLinkConfig"],
        stem_features: Optional[Dict[str, "StemFeatures"]] = None,
        classifications: Optional[Dict[str, Optional["ComponentClassification"]]] = None,
    ) -> None:
        """Initialize physics simulations for each stem.

        If classifications provided, uses BlendedPhysics with multi-model blending.
        Otherwise falls back to SteeringPhysics (legacy behavior).

        Args:
            block_configs: BlockLinkConfig mapping (block → config)
            stem_features: Pre-computed features per stem (optional)
            classifications: Component classifications per stem (optional)
        """
        self._simulations.clear()
        self._classifications = classifications or {}

        for config in block_configs.values():
            link_target = config.link_target
            if link_target in self._simulations:
                continue

            base_stem = get_base_stem(link_target)
            classification = self._classifications.get(base_stem) if base_stem else None

            if classification is not None:
                # New path: BlendedPhysics with classification-driven weights
                self._simulations[link_target] = create_physics_from_classification(
                    classification, self.bpm
                )
                weights = classification.physics_weights()
                logger.debug(
                    f"Physics for {link_target}: BlendedPhysics, "
                    f"spring={weights['spring']:.2f}, pitch={weights['pitch_follow']:.2f}, "
                    f"osc={weights['oscillator']:.2f}, perlin={weights['perlin']:.2f}"
                )
            else:
                # Legacy path: single SteeringPhysics
                preset = get_physics_preset(config.physics_preset, self.bpm)
                self._simulations[link_target] = SteeringPhysics(preset)
                logger.debug(
                    f"Physics for {link_target}: SteeringPhysics, preset={config.physics_preset}, "
                    f"zeta={preset.damping_ratio:.2f}"
                )

    def step(
        self,
        stem: str,
        target: float,
        dt: float,
        pitch_hz: float = 0.0,
        pitch_confidence: float = 0.0,
        energy: float = 0.5,
    ) -> float:
        """Step physics simulation for a stem.

        For BlendedPhysics, routes inputs to appropriate sub-models.
        For SteeringPhysics, uses target only (legacy).

        Args:
            stem: Stem name
            target: Target value (0-1) for spring physics
            dt: Time step in seconds
            pitch_hz: Detected pitch in Hz (for pitch_follow)
            pitch_confidence: Pitch confidence 0-1 (for pitch_follow)
            energy: Energy level 0-1 (for oscillator/perlin)

        Returns:
            Smoothed value after physics simulation
        """
        sim = self._simulations.get(stem)
        if sim is None:
            return target

        if isinstance(sim, BlendedPhysics):
            # Multi-modal inputs for BlendedPhysics
            inputs = {
                "spring": target,
                "pitch_follow": (pitch_hz, pitch_confidence),
                "oscillator": energy,
                "perlin": energy,
            }
            value = sim.step(inputs, dt)
        else:
            # Legacy SteeringPhysics
            value = sim.step(target, dt)

        # Clamp (physics can overshoot in underdamped mode)
        return max(0.0, min(1.0, value))

    def reset_for_seek(
        self,
        stem: str,
        position: float,
        velocity: float = 0.0,
    ) -> None:
        """Reset physics state after audio seek.

        Eliminates lag by setting position directly.
        Only affects SteeringPhysics; BlendedPhysics continues naturally.

        Args:
            stem: Stem name
            position: New position (typically current audio value)
            velocity: Initial velocity (typically 0)
        """
        sim = self._simulations.get(stem)
        if sim is not None and isinstance(sim, SteeringPhysics):
            sim.reset(position=position, velocity=velocity)

    def update_preset(self, stem: str, preset: str) -> None:
        """Update physics preset for a stem.

        Preserves position and velocity for smooth transitions.
        Note: This forces SteeringPhysics even if BlendedPhysics was active.
        To restore BlendedPhysics, reinitialize with classifications.

        Args:
            stem: Stem name
            preset: New physics preset name
        """
        config = get_physics_preset(preset, self.bpm)

        old = self._simulations.get(stem)
        self._simulations[stem] = SteeringPhysics(config)

        # Preserve position for smooth transition (only from SteeringPhysics)
        if old is not None and isinstance(old, SteeringPhysics):
            self._simulations[stem].position = old.position
            self._simulations[stem].velocity = old.velocity

        logger.info(
            f"UpdateStemPhysics: {stem} -> preset={preset}, "
            f"zeta={config.damping_ratio:.2f}"
        )

    def clear(self) -> None:
        """Clear all physics simulations."""
        self._simulations.clear()
        self._classifications.clear()

    @property
    def stems(self) -> list:
        """List of stems with physics simulations."""
        return list(self._simulations.keys())
