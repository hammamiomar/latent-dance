"""Mass-spring-damper physics for organic steering motion.

Instead of directly mapping audio features to steering values, we use
a physics simulation where:
- Audio features set the "target" position
- A virtual mass-spring-damper system moves toward the target
- The spring/damper parameters control the motion character

This creates natural-feeling motion with:
- Underdamped (ζ < 1): overshoot and bounce (drums, impacts)
- Critically damped (ζ = 1): fastest approach, no overshoot (vocals)
- Overdamped (ζ > 1): slow, weighty approach (bass, sustains)

Physics can be scaled by BPM so motion feels natural at any tempo.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from opensimplex import OpenSimplex

if TYPE_CHECKING:
    from hambajuba2ba.audio import ComponentClassification


@dataclass
class PhysicsConfig:
    """Configuration for mass-spring-damper system.

    The damping ratio ζ = c / (2 * sqrt(k * m)) determines behavior:
    - ζ < 1: underdamped (bouncy, overshoot)
    - ζ = 1: critically damped (snappy, no overshoot)
    - ζ > 1: overdamped (sluggish, weighty)

    Natural frequency ω₀ = sqrt(k / m) controls speed.

    Attributes:
        mass: Virtual mass (higher = more inertia)
        stiffness: Spring constant k (higher = faster response)
        damping: Damping coefficient c (higher = less oscillation)
    """

    mass: float = 1.0
    stiffness: float = 200.0
    damping: float = 20.0

    @property
    def natural_frequency(self) -> float:
        """Natural frequency ω₀ = sqrt(k/m) in rad/s."""
        return (self.stiffness / self.mass) ** 0.5

    @property
    def damping_ratio(self) -> float:
        """Damping ratio ζ = c / (2 * sqrt(k * m))."""
        return self.damping / (2 * (self.stiffness * self.mass) ** 0.5)

    def scaled_by_bpm(self, bpm: float, reference_bpm: float = 120.0) -> "PhysicsConfig":
        """Scale physics for different tempos while preserving damping ratio.

        Faster tempos need faster physics to keep up with the beat.
        We scale stiffness by tempo ratio and damping by √(tempo ratio)
        to preserve the damping ratio ζ (bounce/weight character).

        Math:
            ω₀' = √(k×scale/m) = √scale × ω₀  (faster at higher BPM)
            ζ' = c×√scale / (2√(k×scale×m)) = ζ  (preserved)

        Args:
            bpm: Target tempo
            reference_bpm: Reference tempo (120 BPM = no scaling)

        Returns:
            New PhysicsConfig with scaled parameters
        """
        scale = bpm / reference_bpm
        return PhysicsConfig(
            mass=self.mass,
            stiffness=self.stiffness * scale,
            damping=self.damping * (scale ** 0.5),  # √scale preserves ζ
        )


# Physics presets for different stem types
# Designed for 120 BPM reference - scale for other tempos
#
# Damping ratios (ζ) tuned for organic motion:
# - ζ < 1: underdamped (overshoot, bounce, settle) = ALIVE feeling
# - ζ = 1: critically damped (fastest, no overshoot) = mechanical
# - ζ > 1: overdamped (slow approach) = heavy/dreamy
#
# Research: ζ = 0.4-0.7 feels most organic for music visualization
#
# Presets are loaded from presets/physics.json
def _load_physics_presets() -> dict[str, PhysicsConfig]:
    """Load physics presets from JSON file."""
    from hambajuba2ba.presets import load_physics_presets

    json_presets = load_physics_presets()
    return {
        name: PhysicsConfig(
            mass=params["mass"],
            stiffness=params["stiffness"],
            damping=params["damping"],
        )
        for name, params in json_presets.items()
    }


# Lazily loaded on first access
_PHYSICS_PRESETS_CACHE: dict[str, PhysicsConfig] | None = None


def _get_physics_presets() -> dict[str, PhysicsConfig]:
    """Get physics presets (lazy load from JSON)."""
    global _PHYSICS_PRESETS_CACHE
    if _PHYSICS_PRESETS_CACHE is None:
        _PHYSICS_PRESETS_CACHE = _load_physics_presets()
    return _PHYSICS_PRESETS_CACHE


def get_physics_preset(
    name: str,
    bpm: Optional[float] = None,
) -> PhysicsConfig:
    """Get physics preset, optionally scaled by BPM.

    Args:
        name: Preset name (kick, bass, vocals, etc.)
        bpm: Optional tempo for scaling (120 BPM = no scaling)

    Returns:
        PhysicsConfig for the preset
    """
    presets = _get_physics_presets()
    config = presets.get(name, presets.get("other", PhysicsConfig()))
    if bpm is not None:
        config = config.scaled_by_bpm(bpm)
    return config


class SteeringPhysics:
    """Mass-spring-damper simulation for steering values.

    Simulates a 1D mass on a spring with damping, driven by a target position.
    Uses semi-implicit Euler integration for stability.

    State:
        position: Current value (what gets used for steering)
        velocity: Rate of change

    Usage:
        physics = SteeringPhysics(PhysicsConfig(...))
        for each_frame:
            target = audio_feature_value
            steering_value = physics.step(target, dt)
    """

    def __init__(self, config: PhysicsConfig):
        """Initialize physics simulation.

        Args:
            config: Physics parameters
        """
        self.config = config
        self.position = 0.0
        self.velocity = 0.0

    def step(self, target: float, dt: float) -> float:
        """Advance simulation by dt seconds toward target.

        Uses semi-implicit Euler integration:
        1. Update velocity using spring force
        2. Update position using new velocity

        This is more stable than explicit Euler for oscillatory systems.

        Args:
            target: Target position (from audio features)
            dt: Time step in seconds

        Returns:
            New position value (use this for steering)
        """
        # Spring force: F = -k * (x - target) - c * v
        # Acceleration: a = F / m
        displacement = self.position - target
        spring_force = -self.config.stiffness * displacement
        damping_force = -self.config.damping * self.velocity
        acceleration = (spring_force + damping_force) / self.config.mass

        # Semi-implicit Euler: update velocity first, then position
        self.velocity += acceleration * dt
        self.position += self.velocity * dt

        return self.position

    def reset(self, position: float = 0.0, velocity: float = 0.0) -> None:
        """Reset state to given values."""
        self.position = position
        self.velocity = velocity


# =============================================================================
# Sustained Physics Models
# =============================================================================
#
# These physics models are designed for sustained content (pads, vocals, strings)
# where the mass-spring-damper would settle to equilibrium. They provide
# continuous motion that never settles while maintaining organic feel.
#
# Wired in Phase 5 (Physics Integration):
# - PhysicsManager uses BlendedPhysics for stems with ComponentClassification
# - PitchFollowingPhysics receives pitch_hz from Phase 3 pitch tracking
# - CoupledOscillatorPhysics/PerlinDriftPhysics receive energy from audio
# - Chord changes inject impulses via inject_impulse()
#
# Interaction: Percussive hits inject impulses that momentarily tighten motion
# via increased damping, creating rhythmic coupling between percussive and
# sustained elements.


class SustainedPhysicsBase:
    """Base class for sustained physics with percussion interaction.

    All sustained physics models inherit from this to get:
    - Impulse injection from percussive hits
    - Temporary damping spikes (motion tightens on hits)
    - Velocity injection (adds energy from percussion)

    Subclasses implement step() for their specific motion type and
    _apply_velocity_impulse() for how impulses affect their state.
    """

    def __init__(self, base_damping: float = 0.05):
        """Initialize sustained physics base.

        Args:
            base_damping: Default damping coefficient (0-1, lower = looser)
        """
        self.base_damping = base_damping
        self.current_damping = base_damping
        self.damping_spike_frames = 0
        self.impulse_velocity = 0.0  # Accumulated velocity from impulses

    def inject_impulse(
        self, magnitude: float, spike_ms: float = 200, fps: float = 60.0
    ) -> None:
        """Inject energy from a percussion hit.

        Percussion hits affect sustained motion in two ways:
        1. Velocity injection: adds energy in current direction
        2. Damping spike: temporarily tightens motion (more controlled)

        Args:
            magnitude: Impulse strength (0-1 typical)
            spike_ms: How long the damping spike lasts
            fps: Frame rate for converting ms to frames
        """
        self._apply_velocity_impulse(magnitude)
        self.current_damping = self.base_damping * 1.5
        self.damping_spike_frames = int(spike_ms / 1000 * fps)

    def _apply_velocity_impulse(self, magnitude: float) -> None:
        """Add velocity from percussion hit. Override in subclasses."""
        self.impulse_velocity += magnitude * 0.5

    def _decay_damping_spike(self) -> None:
        """Decay damping spike back to base. Call each frame."""
        if self.damping_spike_frames > 0:
            self.damping_spike_frames -= 1
            if self.damping_spike_frames == 0:
                self.current_damping = self.base_damping

    def _decay_impulse_velocity(self, decay: float = 0.95) -> float:
        """Decay and return impulse velocity contribution."""
        contribution = self.impulse_velocity
        self.impulse_velocity *= decay
        return contribution


class PitchFollowingPhysics(SustainedPhysicsBase):
    """Output follows melodic contour. For vocals, leads, bass lines.

    Maps pitch (Hz) to a 0-1 value using log scale, then smoothly
    interpolates toward the target. Confidence gates the tracking
    to avoid jumps during unvoiced segments.

    Receives pitch_hz and pitch_confidence from Phase 3 pitch tracking
    via BlendedPhysics.step().

    Motion character:
    - Smooth glides between notes
    - Velocity adds overshoot on large jumps
    - Impulses add temporary "bounce" energy
    """

    # Pitch range for 0-1 mapping (log scale)
    MIN_PITCH_HZ = 80.0   # Low bass / vocal
    MAX_PITCH_HZ = 2000.0  # High vocal / lead

    def __init__(
        self,
        smoothing: float = 0.15,
        base_damping: float = 0.05,
    ):
        """Initialize pitch-following physics.

        Args:
            smoothing: Interpolation speed (0-1, higher = faster tracking)
            base_damping: Base damping for impulse decay
        """
        super().__init__(base_damping)
        self.smoothing = smoothing
        self.position = 0.5  # Start at middle
        self.velocity = 0.0

    def step(self, pitch_hz: float, confidence: float, dt: float) -> float:
        """Track pitch with smooth interpolation.

        Args:
            pitch_hz: Detected pitch in Hz
            confidence: Pitch detection confidence (0-1)
            dt: Time step in seconds

        Returns:
            Position value (0-1) tracking the pitch
        """
        self._decay_damping_spike()

        # Map pitch to 0-1 (log scale)
        if pitch_hz > 0:
            log_min = math.log(self.MIN_PITCH_HZ)
            log_max = math.log(self.MAX_PITCH_HZ)
            log_pitch = math.log(max(self.MIN_PITCH_HZ, min(pitch_hz, self.MAX_PITCH_HZ)))
            target = (log_pitch - log_min) / (log_max - log_min)
        else:
            target = self.position  # Hold position when no pitch

        # Gate tracking by confidence
        effective_smoothing = self.smoothing * confidence

        # Smooth interpolation with velocity
        error = target - self.position
        self.velocity += error * effective_smoothing * 60 * dt  # Scale by ~60fps reference
        self.velocity *= (1.0 - self.current_damping)  # Apply damping

        # Add impulse velocity contribution
        self.velocity += self._decay_impulse_velocity() * dt

        self.position += self.velocity * dt
        self.position = max(0.0, min(1.0, self.position))

        return self.position

    def _apply_velocity_impulse(self, magnitude: float) -> None:
        """Impulses add bounce in current velocity direction."""
        direction = 1.0 if self.velocity >= 0 else -1.0
        self.impulse_velocity += magnitude * 0.3 * direction


class CoupledOscillatorPhysics(SustainedPhysicsBase):
    """Two coupled oscillators create breathing motion. For pads, drones.

    Assigned to stems classified as harmonic with high harmony_confidence
    via ComponentClassification.physics_weights().

    Two virtual oscillators are connected by a spring-like coupling.
    Energy input increases coupling strength, making them "breathe"
    together. With energy > 0, this never settles.

    Motion character:
    - Slow, organic breathing
    - Energy controls intensity
    - Impulses add momentary "pulse"
    """

    def __init__(
        self,
        freq_a: float = 0.3,    # Hz, slow breathing
        freq_b: float = 0.47,   # Hz, slightly different for phasing
        coupling: float = 0.3,
        base_damping: float = 0.02,
    ):
        """Initialize coupled oscillators.

        Args:
            freq_a: Frequency of oscillator A (Hz)
            freq_b: Frequency of oscillator B (Hz)
            coupling: How strongly oscillators influence each other
            base_damping: Base damping (low for sustained oscillation)
        """
        super().__init__(base_damping)
        self.freq_a = freq_a
        self.freq_b = freq_b
        self.coupling = coupling

        # Oscillator state (position, velocity)
        self.pos_a = 0.3
        self.pos_b = -0.3
        self.vel_a = 0.1
        self.vel_b = -0.1

    def step(self, energy: float, dt: float) -> float:
        """Advance coupled oscillators.

        Args:
            energy: Input energy level (0-1) — controls oscillation amplitude
            dt: Time step in seconds

        Returns:
            Blended position (0-1) from both oscillators
        """
        self._decay_damping_spike()

        # Energy modulates coupling strength
        effective_coupling = self.coupling * (0.3 + energy * 0.7)

        # Angular frequencies
        omega_a = 2 * math.pi * self.freq_a
        omega_b = 2 * math.pi * self.freq_b

        # Coupling forces (each oscillator pulled toward the other)
        coupling_force_a = effective_coupling * (self.pos_b - self.pos_a)
        coupling_force_b = effective_coupling * (self.pos_a - self.pos_b)

        # Restoring force (toward center at 0)
        restore_a = -omega_a * omega_a * self.pos_a
        restore_b = -omega_b * omega_b * self.pos_b

        # Update velocities
        self.vel_a += (restore_a + coupling_force_a) * dt
        self.vel_b += (restore_b + coupling_force_b) * dt

        # Velocity clamp: prevent divergence if energy pump outpaces damping
        max_vel = 10.0
        self.vel_a = max(-max_vel, min(max_vel, self.vel_a))
        self.vel_b = max(-max_vel, min(max_vel, self.vel_b))

        # Energy pump: add energy when amplitude drops below target
        # This ensures oscillation never dies out with energy > 0
        current_amplitude = (self.vel_a**2 + self.vel_b**2) ** 0.5
        target_amplitude = 0.5 * energy
        if current_amplitude < target_amplitude and current_amplitude > 0:
            boost = (target_amplitude / current_amplitude - 1.0) * 0.1
            self.vel_a *= (1.0 + boost)
            self.vel_b *= (1.0 + boost)

        # Apply damping inversely proportional to energy
        # High energy = no damping (perpetual motion)
        # Low energy = strong damping (settles to rest)
        effective_damping = self.current_damping * (1.0 - energy)
        # Time-step aware: damping^(dt/reference_dt) for consistent behavior
        damping_per_frame = (1.0 - effective_damping) ** (dt / 0.02)
        self.vel_a *= damping_per_frame
        self.vel_b *= damping_per_frame

        # Add impulse velocity
        impulse = self._decay_impulse_velocity()
        self.vel_a += impulse * dt
        self.vel_b -= impulse * dt  # Opposite direction for interesting motion

        # Update positions
        self.pos_a += self.vel_a * dt
        self.pos_b += self.vel_b * dt

        # Blend both oscillators and normalize to 0-1
        raw_output = 0.6 * self.pos_a + 0.4 * self.pos_b
        return 0.5 + raw_output * 0.5  # Map [-1, 1] to [0, 1]

    def _apply_velocity_impulse(self, magnitude: float) -> None:
        """Impulses kick both oscillators outward."""
        self.impulse_velocity += magnitude * 0.4


class PerlinDriftPhysics(SustainedPhysicsBase):
    """Perlin noise drift. For ambient, atmospheric content.

    Assigned to stems classified with high texture_confidence and
    harmonic_confidence via ComponentClassification.physics_weights().

    Uses simplex noise (Perlin-like) for smooth, organic drift.
    Energy controls drift speed — more energy = faster movement
    through noise space.

    Motion character:
    - Organic, flowing, never repeats
    - Energy controls pace
    - Impulses add velocity bumps
    """

    def __init__(
        self,
        base_speed: float = 0.5,
        energy_scale: float = 2.0,
        base_damping: float = 0.05,
        seed: int = 42,
    ):
        """Initialize Perlin drift physics.

        Args:
            base_speed: Minimum drift speed through noise space
            energy_scale: How much energy increases drift speed
            base_damping: Damping for impulse velocity decay
            seed: Random seed for noise generator
        """
        super().__init__(base_damping)
        self.base_speed = base_speed
        self.energy_scale = energy_scale

        # Noise generator
        self.noise = OpenSimplex(seed=seed)

        # Position in noise space
        self.noise_pos = 0.0
        self.drift_velocity = 0.0

    def step(self, energy: float, dt: float) -> float:
        """Sample noise at current position, advance by energy-scaled speed.

        Args:
            energy: Input energy (0-1) — controls drift speed
            dt: Time step in seconds

        Returns:
            Noise value normalized to 0-1
        """
        self._decay_damping_spike()

        # Energy controls drift speed through noise space
        speed = self.base_speed + energy * self.energy_scale

        # Add impulse contribution (decays over time)
        impulse = self._decay_impulse_velocity()
        self.drift_velocity += impulse
        self.drift_velocity *= (1.0 - self.current_damping)

        # Advance through noise space
        self.noise_pos += (speed + self.drift_velocity) * dt

        # Sample 1D simplex noise (returns -1 to 1)
        raw_noise = self.noise.noise2(self.noise_pos, 0.0)

        # Normalize to 0-1
        return 0.5 + raw_noise * 0.5

    def _apply_velocity_impulse(self, magnitude: float) -> None:
        """Impulses speed up drift temporarily."""
        self.drift_velocity += magnitude * 1.5


class BlendedPhysics:
    """Weighted blend of multiple physics models.

    Created by PhysicsManager.initialize() for stems with ComponentClassification.
    Weights come from ComponentClassification.physics_weights().

    Combines spring (percussive), pitch_follow (melodic), oscillator (harmonic),
    and perlin (texture) physics based on weights from physics_weights().

    Each model runs independently, outputs are blended by weight.
    Impulses (from chord changes) propagate to all sustained models.
    """

    def __init__(self, weights: dict[str, float], bpm: float = 120.0):
        """Initialize blended physics with component weights.

        Args:
            weights: Dict with "spring", "pitch_follow", "oscillator", "perlin"
                     weights (should sum to 1.0)
            bpm: BPM for scaling spring physics
        """
        self.weights = weights
        self.bpm = bpm

        # Create physics models matching ComponentClassification.physics_weights() keys
        self.models: dict[str, SteeringPhysics | SustainedPhysicsBase] = {
            "spring": SteeringPhysics(get_physics_preset("kick", bpm)),
            "pitch_follow": PitchFollowingPhysics(),
            "oscillator": CoupledOscillatorPhysics(),
            "perlin": PerlinDriftPhysics(),
        }

    def step(
        self,
        inputs: dict[str, float | tuple[float, ...]],
        dt: float,
    ) -> float:
        """Advance all models and blend outputs.

        Args:
            inputs: Model inputs:
                - "spring": target value for SteeringPhysics
                - "pitch_follow": (pitch_hz, confidence) for PitchFollowingPhysics
                - "oscillator": energy for CoupledOscillatorPhysics
                - "perlin": energy for PerlinDriftPhysics
            dt: Time step in seconds

        Returns:
            Weighted blend of all model outputs
        """
        outputs: dict[str, float] = {}

        # Step spring (percussive, impulse-driven)
        spring_target = inputs.get("spring", 0.0)
        if not isinstance(spring_target, (int, float)):
            spring_target = 0.0
        spring_model = self.models["spring"]
        if isinstance(spring_model, SteeringPhysics):
            outputs["spring"] = spring_model.step(float(spring_target), dt)

        # Step pitch_follow (melodic contour)
        pitch_input = inputs.get("pitch_follow", (440.0, 0.5))
        if isinstance(pitch_input, tuple) and len(pitch_input) >= 2:
            pitch_hz, confidence = pitch_input[0], pitch_input[1]
        else:
            pitch_hz, confidence = 440.0, 0.5
        pitch_model = self.models["pitch_follow"]
        if isinstance(pitch_model, PitchFollowingPhysics):
            outputs["pitch_follow"] = pitch_model.step(float(pitch_hz), float(confidence), dt)

        # Step oscillator (harmonic chords, pads)
        oscillator_energy = inputs.get("oscillator", 0.5)
        if not isinstance(oscillator_energy, (int, float)):
            oscillator_energy = 0.5
        oscillator_model = self.models["oscillator"]
        if isinstance(oscillator_model, CoupledOscillatorPhysics):
            outputs["oscillator"] = oscillator_model.step(float(oscillator_energy), dt)

        # Step perlin (texture, ambient drift)
        perlin_energy = inputs.get("perlin", 0.5)
        if not isinstance(perlin_energy, (int, float)):
            perlin_energy = 0.5
        perlin_model = self.models["perlin"]
        if isinstance(perlin_model, PerlinDriftPhysics):
            outputs["perlin"] = perlin_model.step(float(perlin_energy), dt)

        # Weighted blend
        result = sum(outputs[key] * self.weights.get(key, 0.0) for key in outputs)
        return result

    def inject_impulse(self, magnitude: float) -> None:
        """Propagate percussion impulse to all sustained models.

        Args:
            magnitude: Impulse strength (0-1 typical)
        """
        for model in self.models.values():
            if isinstance(model, SustainedPhysicsBase):
                model.inject_impulse(magnitude)


def create_physics_from_classification(
    classification: "ComponentClassification",
    bpm: float = 120.0,
) -> BlendedPhysics:
    """Create physics blend from component classification.

    Factory function that translates a ComponentClassification into
    an appropriately-weighted BlendedPhysics instance.

    Called by PhysicsManager.initialize() for each stem that has
    a classification (stems with HPSS data from Phase 1).

    Args:
        classification: Audio component classification with texture/role/range
        bpm: BPM for scaling spring physics

    Returns:
        BlendedPhysics with weights derived from classification
    """
    weights = classification.physics_weights()
    return BlendedPhysics(weights, bpm)
