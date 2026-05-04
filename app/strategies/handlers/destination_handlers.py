"""Destination modulation and composition handlers.

Handles: SetDestination, FreezeBlend, SetBlendPosition, SetDestinationMode,
         SetReactiveConfig, SetCompositionConfig
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

from pydantic import BaseModel

from app.schemas import (
    SetDestination,
    ClearDestination,
    FreezeBlend,
    SetBlendPosition,
    SetDestinationMode,
    SetReactiveConfig,
    SetDestinationLink,
    SetCompositionConfig,
)
from hambajuba2ba.bridge.destinations import Destination, ReactiveConfig

if TYPE_CHECKING:
    from app.strategies.sae_steering_strategy import SAESteeringStrategy

logger = logging.getLogger("uvicorn")


def handle_destination_message(
    strategy: "SAESteeringStrategy",
    message: BaseModel,
) -> Optional[dict]:
    """Handle destination modulation messages.

    May queue messages if modulators not yet initialized.

    Args:
        strategy: The SAE steering strategy instance
        message: One of the destination-related message types

    Returns:
        Optional response dict (currently None)
    """
    if isinstance(message, SetDestination):
        return _handle_set_destination(strategy, message)

    if isinstance(message, ClearDestination):
        return _handle_clear_destination(strategy, message)

    if isinstance(message, FreezeBlend):
        return _handle_freeze_blend(strategy, message)

    if isinstance(message, SetBlendPosition):
        return _handle_set_blend_position(strategy, message)

    if isinstance(message, SetDestinationMode):
        return _handle_set_mode(strategy, message)

    if isinstance(message, SetReactiveConfig):
        return _handle_set_reactive_config(strategy, message)

    if isinstance(message, SetDestinationLink):
        return _handle_set_destination_link(strategy, message)

    if isinstance(message, SetCompositionConfig):
        return _handle_set_composition_config(strategy, message)

    return None


def _get_modulator(strategy: "SAESteeringStrategy", space: str):
    """Get the prompt destination modulator.

    Latent destinations are dead — composition is driven by
    CompositionEngine. Only prompt SLERP remains as a modulator.
    For latent space, returns None (handled via CompositionEngine).
    """
    if space == "prompt":
        return strategy.prompt_destinations
    return None


def _handle_set_destination(
    strategy: "SAESteeringStrategy",
    message: SetDestination,
) -> None:
    """Load a destination into slot A or B.

    For latent space: loads noise into CompositionEngine (seed only).
    For prompt space: loads into DestinationModulator (prompt SLERP).
    """
    # Latent space → CompositionEngine (noise circular walk)
    if message.space == "latent":
        if message.destination_type != "seed" or message.seed is None:
            logger.warning(
                "Latent destinations only accept seed type, got %s",
                message.destination_type,
            )
            return None

        if strategy._composition is None:
            if message.slot == "a":
                strategy._noise_seed_a = message.seed
            else:
                strategy._noise_seed_b = message.seed
            logger.info(
                "Deferred SetDestination: latent/%s -> seed %s",
                message.slot,
                message.seed,
            )
            return None

        if strategy.pipeline.engine is None:
            strategy._pending_destination_messages.append(message)
            logger.info(
                f"Queued SetDestination(latent/{message.slot}) for after setup()"
            )
            return None

        noise_tensor = strategy.pipeline.engine.make_noise(message.seed)
        strategy._composition.load_noise(message.slot, noise_tensor, message.seed)
        if message.slot == "a":
            strategy._noise_seed_a = message.seed
        else:
            strategy._noise_seed_b = message.seed
        logger.info(
            f"SetDestination: latent/{message.slot} -> seed {message.seed} "
            f"(loaded into CompositionEngine)"
        )
        return None

    # Prompt space → DestinationModulator (SLERP)
    modulator = _get_modulator(strategy, message.space)

    if modulator is None:
        # Queue for replay after setup()
        strategy._pending_destination_messages.append(message)
        logger.info(
            f"Queued SetDestination({message.space}/{message.slot}) for after setup()"
        )
        return None

    # Load the destination
    destination = _load_destination(
        strategy,
        space=message.space,
        destination_type=message.destination_type,
        seed=message.seed,
        prompt=message.prompt,
    )

    # Route based on replace_mode
    replace_mode = message.replace_mode

    if replace_mode == "from_blend":
        # Legacy behavior: freeze current blend as A, new dest as B
        modulator.replace_from_blend(destination)
    else:
        # Direct replacement: load into the specified slot, no blend freeze
        if message.slot == "a":
            modulator.load_a(destination)
        else:
            modulator.load_b(destination)

    # Log with shape info
    pooled_shape = (
        destination.tensor_pooled.shape
        if destination.tensor_pooled is not None
        else "None"
    )
    logger.info(
        f"SetDestination({replace_mode}): {message.space}/{message.slot} -> "
        f"{destination.label}, tensor={destination.tensor.shape}, pooled={pooled_shape}"
    )
    return None


def _handle_clear_destination(
    strategy: "SAESteeringStrategy",
    message: ClearDestination,
) -> None:
    """Clear a destination slot in latent or prompt space."""
    if message.space == "latent":
        if strategy._composition is None:
            if message.slot == "a":
                if strategy._noise_seed_b is not None:
                    strategy._noise_seed_a = strategy._noise_seed_b
                    strategy._noise_seed_b = None
                else:
                    strategy._noise_seed_a = None
            else:
                strategy._noise_seed_b = None
            logger.info(
                "Deferred ClearDestination: latent/%s -> seed_a=%s, seed_b=%s",
                message.slot,
                strategy._noise_seed_a,
                strategy._noise_seed_b,
            )
            return None

        strategy._composition.clear_noise(message.slot)
        strategy._noise_seed_a = strategy._composition.seed_a
        strategy._noise_seed_b = strategy._composition.seed_b
        logger.info(
            "ClearDestination: latent/%s -> seed_a=%s, seed_b=%s",
            message.slot,
            strategy._noise_seed_a,
            strategy._noise_seed_b,
        )
        return None

    modulator = _get_modulator(strategy, message.space)

    if modulator is None:
        strategy._pending_destination_messages.append(message)
        logger.info(
            f"Queued ClearDestination({message.space}/{message.slot}) for after setup()"
        )
        return None

    modulator.clear_destination(message.slot)
    logger.info(
        "ClearDestination: %s/%s -> A=%s, B=%s",
        message.space,
        message.slot,
        modulator.destination_a.label if modulator.destination_a else None,
        modulator.destination_b.label if modulator.destination_b else None,
    )
    return None


def _handle_freeze_blend(
    strategy: "SAESteeringStrategy",
    message: FreezeBlend,
) -> None:
    """Freeze current blend position into a specific slot."""
    if message.space == "latent":
        logger.debug("FreezeBlend ignored for latent space (use CompositionEngine)")
        return None

    modulator = _get_modulator(strategy, message.space)

    if modulator is None:
        strategy._pending_destination_messages.append(message)
        logger.info(
            f"Queued FreezeBlend({message.space}/{message.target_slot}) for after setup()"
        )
        return None

    modulator.freeze_blend_to(message.target_slot)
    logger.info(
        f"FreezeBlend: {message.space} -> froze blend into slot {message.target_slot}, "
        f"blend_position now {modulator.blend_position:.3f}"
    )
    return None


def _handle_set_blend_position(
    strategy: "SAESteeringStrategy",
    message: SetBlendPosition,
) -> None:
    """Set blend position (slider mode)."""
    if message.space == "latent":
        logger.debug("SetBlendPosition ignored for latent space (use CompositionEngine)")
        return None

    modulator = _get_modulator(strategy, message.space)

    if modulator is None:
        strategy._pending_destination_messages.append(message)
        logger.info(f"Queued SetBlendPosition({message.space}) for after setup()")
        return None

    modulator.set_blend(message.position)
    logger.debug(f"SetBlendPosition: {message.space} -> {message.position:.2f}")
    return None


def _handle_set_mode(
    strategy: "SAESteeringStrategy",
    message: SetDestinationMode,
) -> None:
    """Set modulation mode (slider/reactive)."""
    if message.space == "latent":
        logger.debug("SetDestinationMode ignored for latent space (use CompositionEngine)")
        return None

    modulator = _get_modulator(strategy, message.space)

    if modulator is None:
        strategy._pending_destination_messages.append(message)
        logger.info(f"Queued SetDestinationMode({message.space}) for after setup()")
        return None

    old_mode = modulator.mode
    old_blend = modulator.blend_position
    modulator.set_mode(message.mode)
    logger.info(
        f"[MODE CHANGE] {message.space}: {old_mode} -> {message.mode} "
        f"(blend_pos={old_blend:.3f}, now_mode={modulator.mode})"
    )
    return None


def _handle_set_reactive_config(
    strategy: "SAESteeringStrategy",
    message: SetReactiveConfig,
) -> None:
    """Set reactive mode configuration (does NOT change mode)."""
    if message.space == "latent":
        logger.debug("SetReactiveConfig ignored for latent space (use CompositionEngine)")
        return None

    modulator = _get_modulator(strategy, message.space)

    if modulator is None:
        strategy._pending_destination_messages.append(message)
        logger.info(f"Queued SetReactiveConfig({message.space}) for after setup()")
        return None

    config = ReactiveConfig(
        driver=message.driver,
        stem=message.stem,
        blend_min=message.blend_min,
        blend_max=message.blend_max,
        stage_left=message.stage_left if message.stage_left is not None else ReactiveConfig().stage_left,
        stage_home=message.stage_home if message.stage_home is not None else ReactiveConfig().stage_home,
        stage_right=message.stage_right if message.stage_right is not None else ReactiveConfig().stage_right,
        position_source=message.position_source or ReactiveConfig().position_source,
        intensity_source=message.intensity_source or ReactiveConfig().intensity_source,
        position_smoothing_ms=message.position_smoothing_ms or ReactiveConfig().position_smoothing_ms,
        silence_behavior=message.silence_behavior or ReactiveConfig().silence_behavior,
        drift_ms=message.drift_ms or ReactiveConfig().drift_ms,
        intensity_curve=message.intensity_curve or ReactiveConfig().intensity_curve,
        intensity_gamma=message.intensity_gamma if message.intensity_gamma is not None else ReactiveConfig().intensity_gamma,
        stem_rankings=message.stem_rankings,
        rank_weights=message.rank_weights,
    )
    # Only update config, don't change mode - mode is controlled by SetDestinationMode
    modulator.reactive_config = config
    # Update blend slew rate if provided
    if message.blend_slew_rate is not None:
        modulator.blend_slew_rate = message.blend_slew_rate
    # Refresh follower parameters if currently active
    if modulator.mode in ("reactive", "linked"):
        modulator._ensure_followers()
    slew = getattr(modulator, "blend_slew_rate", 1.5)
    logger.info(
        f"SetReactiveConfig: {message.space} -> driver={message.driver} "
        f"slew_rate={slew:.1f} (mode unchanged: {modulator.mode})"
    )
    return None


def _handle_set_destination_link(
    strategy: "SAESteeringStrategy",
    message: SetDestinationLink,
) -> None:
    """Set link target for destination (enables linked mode)."""
    if message.space == "latent":
        logger.debug("SetDestinationLink ignored for latent space (use CompositionEngine)")
        return None

    modulator = _get_modulator(strategy, message.space)

    if modulator is None:
        strategy._pending_destination_messages.append(message)
        logger.info(f"Queued SetDestinationLink({message.space}) for after setup()")
        return None

    modulator.set_link_target(message.link_target)
    logger.info(
        f"SetDestinationLink: {message.space} -> link_target={message.link_target}, "
        f"mode now 'linked'"
    )
    return None


def _load_destination(
    strategy: "SAESteeringStrategy",
    space: str,
    destination_type: str,
    seed: Optional[int] = None,
    prompt: Optional[str] = None,
) -> Destination:
    """Create a Destination from user input (prompt space only).

    Latent/seed destinations are handled directly by CompositionEngine
    in _handle_set_destination(). This function is only called for
    prompt space destinations.

    Args:
        strategy: Strategy instance (for pipeline access)
        space: "prompt" (latent handled separately)
        destination_type: "prompt" (seed handled by CompositionEngine)
        seed: Unused (latent seeds go to CompositionEngine)
        prompt: Prompt text (for type="prompt")

    Returns:
        Destination with tensor and metadata
    """
    if destination_type == "prompt":
        if prompt is None:
            raise ValueError("prompt required for destination_type='prompt'")
        embeds, pooled = strategy.pipeline.encode_prompt(prompt, use_cache=False)
        label = prompt[:20] + "..." if len(prompt) > 20 else prompt

        return Destination(
            tensor=embeds,
            tensor_pooled=pooled,
            label=label,
            prompt=prompt,
        )

    raise ValueError(
        f"destination_type='{destination_type}' not supported for prompt space. "
        "Use 'prompt' type for prompt space, or 'seed' type for latent space."
    )


def _handle_set_composition_config(
    strategy: "SAESteeringStrategy",
    message: SetCompositionConfig,
) -> None:
    """Set composition engine properties (distance, mode)."""
    if strategy._composition is None:
        logger.info("Composition not initialized, ignoring SetCompositionConfig")
        return None

    if message.distance is not None:
        strategy._composition.distance = message.distance
    if message.mode is not None:
        strategy._composition.mode = message.mode

    logger.info(
        "SetCompositionConfig: distance=%.2f, mode=%s",
        strategy._composition.distance,
        strategy._composition.mode,
    )
    return None


def is_destination_message(message: BaseModel) -> bool:
    """Check if message is a destination or composition message."""
    return isinstance(message, (
        SetDestination,
        ClearDestination,
        FreezeBlend,
        SetBlendPosition,
        SetDestinationMode,
        SetReactiveConfig,
        SetDestinationLink,
        SetCompositionConfig,
    ))
