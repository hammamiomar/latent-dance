"""Intelligent prominence computation for the Dancer Ensemble architecture.

The ProminenceEngine transforms user-defined ranks into dynamic prominence values
based on musical context. Instead of static weights, users pick WHAT to focus on
(the ensemble), and the system decides HOW MUCH to show each element.

Ranking Semantics (compressed 4x range):
    Rank 1: Lead - base prominence 1.0
    Rank 2: Support - base prominence 0.65
    Rank 3: Background - base prominence 0.40
    Rank 4: Subtle - base prominence 0.25
    None:   Auto/available - base 0.05, can be promoted on surprise

Dynamic Modulation:
    - Novelty boost: stems doing something new get temporary attention (+30%)
    - Coupling bonus: locked stems dance together (up to +50%)
    - Call-response: alternating focus ping-pong (boost/fade)
    - Activity gate: silent stems fade out gracefully
    - Surprise promotion: unranked stems get promoted on layer entry

Design:
    - Stateless computation (all context passed in)
    - Uses AudioSampler for feature queries
    - Per-axis rankings (SAE, SLERP latent, SLERP prompt)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, Optional


if TYPE_CHECKING:
    from hambajuba2ba.audio.sampler import AudioSampler


# Rank to base prominence mapping (compressed 4x range, perceptually even steps)
RANK_TO_BASE: Dict[Optional[int], float] = {
    1: 1.0,    # Lead
    2: 0.65,   # Support
    3: 0.40,   # Background
    4: 0.25,   # Subtle (visible, not invisible)
    None: 0.05,  # Auto/available for surprise
}


@dataclass
class SurpriseState:
    """Tracks surprise promotion state for gradual decay.

    When an unranked stem is promoted (layer entry, strong novelty),
    the boost decays smoothly over ~3 seconds for organic feel.
    """
    boost: float = 0.0
    decay_rate: float = 0.33  # ~3 second decay (time-based via dt)

    def update(self, new_boost: float, dt: float) -> float:
        """Update boost with new value or decay existing.

        Args:
            new_boost: Fresh boost from surprise detection (0 if none)
            dt: Time delta in seconds

        Returns:
            Current boost value after update
        """
        if new_boost > self.boost:
            # New surprise event - jump to new level
            self.boost = new_boost
        else:
            # Decay existing boost
            self.boost = max(0.0, self.boost - self.decay_rate * dt)
        return self.boost


@dataclass
class ProminenceEngine:
    """Computes dynamic prominence for each stem based on rank + musical context.

    The engine transforms static rankings into dynamic prominence values that
    respond to the music. This creates a visualizer that "understands" the music
    rather than just reacting to it.

    Usage:
        engine = ProminenceEngine()

        # In frame loop:
        prominence = engine.compute_prominence(
            stem="drums",
            rank=1,
            audio_sampler=sampler,
            audio_time=1.5,
            frame_idx=45,
            dt=0.033,
            all_stem_ranks={"drums": 1, "bass": 2, "vocals": None},
        )
    """

    # Per-stem surprise state for smooth promotion decay
    _surprise_states: Dict[str, SurpriseState] = field(default_factory=dict)

    # Modulation parameters (can be tuned)
    novelty_boost_max: float = 0.3      # Up to +30% for novel moments
    coupling_bonus_max: float = 0.5     # Up to +50% for locked stems
    call_response_boost: float = 1.3    # 30% boost for calling stem
    call_response_fade: float = 0.6     # 40% fade for responding stem

    # Surprise promotion thresholds
    layer_entry_boost: float = 0.7      # Promote to near rank-1 on layer entry
    novelty_threshold: float = 0.5      # Strong novelty derivative threshold
    novelty_boost_amount: float = 0.5   # Promotion amount for strong novelty

    def compute_prominence(
        self,
        stem: str,
        rank: Optional[int],
        audio_sampler: "AudioSampler",
        audio_time: float,
        frame_idx: int,
        dt: float,
        all_stem_ranks: Dict[str, Optional[int]],
    ) -> float:
        """Compute actual visual prominence for a stem.

        The algorithm:
        1. Base prominence from rank (1->1.0, 2->0.65, 3->0.40, 4->0.25, None->0.05)
        2. Novelty boost: stem doing something new gets attention
        3. Coupling bonus: locked stems dance together
        4. Call-response mod: alternating focus ping-pong
        5. Activity gate: silent stems fade out
        6. Surprise promotion: unranked stems get promoted on events

        Args:
            stem: Stem name (bass, drums, vocals, other)
            rank: User-assigned rank (1-4) or None for auto
            audio_sampler: AudioSampler for feature queries
            audio_time: Current audio playback time in seconds
            frame_idx: Current frame index
            dt: Time delta since last frame in seconds
            all_stem_ranks: Dict of all stem rankings for this axis

        Returns:
            Computed prominence in [0, 1]
        """
        # 1. Base prominence from rank
        base = self._rank_to_base(rank)

        # 2. Get stems at same rank (competitors for attention within tier)
        same_rank_stems = [
            s for s, r in all_stem_ranks.items()
            if r == rank and s != stem
        ]

        # 3. Novelty boost: stem doing something new gets attention
        novelty_boost = self._compute_novelty_boost(stem, audio_sampler, audio_time)

        # 4. Coupling bonus: locked stems dance together
        coupling_bonus = self._compute_coupling_bonus(
            stem, same_rank_stems, frame_idx, audio_sampler
        )

        # 5. Call-response modulation: alternating focus ping-pong
        call_response_mod = self._compute_call_response_mod(
            stem, same_rank_stems, frame_idx, audio_sampler, audio_time
        )

        # 6. Activity gate: silent stems fade out
        activity_gate = audio_sampler.compute_activity_gate(stem, audio_time)

        # 7. Surprise promotion for unranked stems
        surprise_boost = 0.0
        if rank is None:
            surprise_boost = self._compute_surprise_promotion(
                stem, audio_sampler, audio_time, frame_idx, dt
            )

        # Combine: base * (1 + boosts) * modifiers * gate + surprise
        prominence = base * (1 + novelty_boost + coupling_bonus)
        prominence *= call_response_mod
        prominence *= activity_gate
        prominence += surprise_boost  # Surprise adds on top

        return min(prominence, 1.0)

    def _rank_to_base(self, rank: Optional[int]) -> float:
        """Convert rank to base prominence."""
        return RANK_TO_BASE.get(rank, 0.05)

    def _compute_novelty_boost(
        self,
        stem: str,
        audio_sampler: "AudioSampler",
        audio_time: float,
    ) -> float:
        """Compute novelty boost from rising edge detection.

        Positive novelty derivative = stem doing something new = attention.
        Uses short timescale for immediate transient response.

        Returns:
            Boost in [0, novelty_boost_max]
        """
        novelty_deriv = audio_sampler.sample_novelty_derivative(
            stem, audio_time, "short"
        )
        # Only positive derivatives (rising novelty) get boost
        if novelty_deriv > 0:
            return min(novelty_deriv, 1.0) * self.novelty_boost_max
        return 0.0

    def _compute_coupling_bonus(
        self,
        stem: str,
        same_rank_stems: list,
        frame_idx: int,
        audio_sampler: "AudioSampler",
    ) -> float:
        """Compute coupling bonus for stems that are rhythmically locked.

        When stems at the same rank have high lock_index (>0.5),
        they reinforce each other visually - dancing together.

        Returns:
            Bonus in [0, coupling_bonus_max]
        """
        # Delegate to AudioSampler's existing method
        return audio_sampler.compute_coupling_bonus(
            stem, frame_idx, same_rank_stems, max_bonus=self.coupling_bonus_max
        )

    def _compute_call_response_mod(
        self,
        stem: str,
        same_rank_stems: list,
        frame_idx: int,
        audio_sampler: "AudioSampler",
        audio_time: float,
    ) -> float:
        """Compute call-response modulation for alternating focus.

        When two stems have high call-response score, the currently
        active stem gets boosted while the other fades.

        Returns:
            Multiplier around 1.0 (boost > 1, fade < 1)
        """
        mod = 1.0

        for other in same_rank_stems:
            cr = audio_sampler.sample_call_response(stem, other, frame_idx)
            if cr > 0.5:
                # Determine who's "calling" (louder)
                stem_energy = audio_sampler.sample_stem(stem, audio_time, "energy_smooth")
                other_energy = audio_sampler.sample_stem(other, audio_time, "energy_smooth")

                if stem_energy > other_energy:
                    # This stem is calling - boost
                    mod *= self.call_response_boost
                else:
                    # This stem is responding - fade
                    mod *= self.call_response_fade

        return mod

    def _compute_surprise_promotion(
        self,
        stem: str,
        audio_sampler: "AudioSampler",
        audio_time: float,
        frame_idx: int,
        dt: float,
    ) -> float:
        """Compute surprise promotion for unranked stems.

        Unranked stems can be temporarily promoted on:
        1. Layer entry: new instrumental section starts
        2. Strong novelty: something significant happening

        The boost decays smoothly over ~3 seconds.

        Returns:
            Surprise boost to add to base prominence
        """
        # Initialize surprise state if needed
        if stem not in self._surprise_states:
            self._surprise_states[stem] = SurpriseState()

        state = self._surprise_states[stem]
        new_boost = 0.0

        # Check for layer entry (new section driven by this stem)
        if audio_sampler.sample_layer_entry(stem, frame_idx):
            new_boost = max(new_boost, self.layer_entry_boost)

        # Check for strong rising novelty (something important happening)
        novelty_deriv = audio_sampler.sample_novelty_derivative(
            stem, audio_time, "medium"
        )
        if novelty_deriv > self.novelty_threshold:
            new_boost = max(new_boost, self.novelty_boost_amount)

        # Update state with decay
        return state.update(new_boost, dt)

    def reset(self) -> None:
        """Reset all internal state (call when track changes)."""
        self._surprise_states.clear()

    def get_surprise_active(self, stem: str) -> bool:
        """Return True if a stem currently has a surprise boost active."""
        state = self._surprise_states.get(stem)
        if state is None:
            return False
        return state.boost > 1e-3


def compute_all_prominences(
    engine: ProminenceEngine,
    stem_ranks: Dict[str, Optional[int]],
    audio_sampler: "AudioSampler",
    audio_time: float,
    frame_idx: int,
    dt: float,
) -> Dict[str, float]:
    """Compute prominence for all stems in one call.

    Convenience function for computing the full prominence map.

    Args:
        engine: ProminenceEngine instance
        stem_ranks: Dict of stem name to rank (1-4 or None)
        audio_sampler: AudioSampler for feature queries
        audio_time: Current audio playback time
        frame_idx: Current frame index
        dt: Time delta since last frame

    Returns:
        Dict of stem name to computed prominence [0, 1]
    """
    return {
        stem: engine.compute_prominence(
            stem=stem,
            rank=rank,
            audio_sampler=audio_sampler,
            audio_time=audio_time,
            frame_idx=frame_idx,
            dt=dt,
            all_stem_ranks=stem_ranks,
        )
        for stem, rank in stem_ranks.items()
    }
