"""SAE steering strategy for audio-driven feature control.

Orchestrates audio-to-visual translation:
- Audio stem features → SAE steering strengths
- Physics simulation for musical response
- Spatial masks for regional control
- Destination modulation for scene travel

This is the orchestrator - see managers/ for implementation details.
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import replace
from typing import Any, Dict, List, Optional

import torch
from pydantic import BaseModel

from app.caching import CacheManager
from app.generation import FrameItem
from hambajuba2ba.generation.encoding import gpu_to_cpu_tensor
from app.schemas import (
    DestinationStatus,
    BlockConfigs,
    BlockConfigSnapshot,
)
from app.strategies.base import GenerationStrategy

# Handlers
from app.strategies.handlers import (
    handle_audio_message,
    handle_stem_message,
    handle_destination_message,
    handle_modulation_message,
)
from app.strategies.handlers.audio_handlers import is_audio_message
from app.strategies.handlers.slot_handlers import is_stem_message
from app.strategies.handlers.destination_handlers import is_destination_message
from app.strategies.handlers.modulation_handlers import is_modulation_message

from hambajuba2ba.audio.focus_config import DEFAULT_BLOCK_CONFIGS
from hambajuba2ba.config import PipelineConfig
from hambajuba2ba.generation.pipeline import SAESteerablePipeline
from hambajuba2ba.bridge.destinations import DestinationModulator, Destination

logger = logging.getLogger("uvicorn")


class SAESteeringStrategy(GenerationStrategy):
    """Real-time SAE feature steering driven by audio stems.

    Orchestrates the audio→visual pipeline:
    1. AudioSampler reads pre-computed features
    2. PhysicsManager smooths values musically
    3. SteeringComputation builds SAE config
    4. SpatialManager animates regional masks
    5. FrameManager handles encode pipeline
    """

    def __init__(
        self,
        pipeline: SAESteerablePipeline,
        config: PipelineConfig,
        websocket: Any,
        audio_cache: CacheManager,
        **kwargs,
    ):
        super().__init__(pipeline, config, websocket, audio_cache, **kwargs)

        # SAE-specific state (everything else is in base)
        self._prompt_embeds: Optional[torch.Tensor] = None
        self._pooled_embeds: Optional[torch.Tensor] = None
        self._base_latent: Optional[torch.Tensor] = None
        self.prompt_destinations: Optional[DestinationModulator] = None
        self._pending_destination_messages: List[BaseModel] = []
        self._last_destination_status_time: float = 0.0

    def _init_default_slot_configs(self) -> None:
        """Initialize default block configs with random feature IDs."""
        for block, default_cfg in DEFAULT_BLOCK_CONFIGS.items():
            cfg = replace(
                default_cfg,
                feature_id=random.randint(0, 5119),
            )
            self.slot_configs[block] = cfg

    def _auto_derive_slot_configs(self) -> None:
        """Derive intensity_source and spatial_mask from stem classification."""
        from hambajuba2ba.bridge.spatial_manager import PRESET_MASKS

        for block, config in self.slot_configs.items():
            if not config.auto_config:
                continue
            classification = self.stem_classifications.get(config.link_target)
            is_percussive = (
                (classification and classification.percussive_confidence > 0.6)
                or config.link_target.startswith("drums")
                or config.link_target.endswith("_percussive")
            )
            config.intensity_source = "transient" if is_percussive else "energy_smooth"

            config.spatial_mask = PRESET_MASKS["uniform"]
            if self._spatial is not None:
                self._spatial.set_block_mask(block, config.spatial_mask)

    async def _setup_backend(self, params: BaseModel, cached: dict) -> None:
        """SAE-specific setup: encode prompt, init destinations, replay queue."""
        # Send block config snapshot for UI sync
        await self.websocket.send_json(
            BlockConfigs(configs=self._serialize_slot_configs()).model_dump()
        )

        # Encode prompt embeddings
        if self._prompt_embeds is None:
            if hasattr(params, 'prompt') and params.prompt:
                self._prompt_embeds, self._pooled_embeds = self.pipeline.encode_prompt(
                    params.prompt
                )
                logger.info(f"Encoded prompt: {self._prompt_embeds.shape}")
            else:
                fallback = "abstract art, flowing colors, ethereal"
                self._prompt_embeds, self._pooled_embeds = self.pipeline.encode_prompt(fallback)
                logger.info(f"Using fallback prompt: {fallback}")

            self._base_latent = self.pipeline._base_latent

        # Initialize prompt destination modulator
        if self._base_latent is not None:
            self._init_destination_modulators()
            logger.info("Initialized prompt destination modulator")

            if self.prompt_destinations:
                self.prompt_destinations.warmup()
            logger.info("Warmed up prompt destination SLERP path")

        # Replay queued destination messages
        if self._pending_destination_messages:
            to_replay = list(self._pending_destination_messages)
            self._pending_destination_messages.clear()
            logger.info(f"Replaying {len(to_replay)} queued destination messages")
            for msg in to_replay:
                await self.handle_message(msg)

    def _cleanup_backend(self) -> None:
        """Clear SAE-specific embeddings and hooks."""
        self._prompt_embeds = None
        self._pooled_embeds = None
        self._base_latent = None
        if self.pipeline.steering_manager:
            self.pipeline.steering_manager.clear_hooks()

    def _init_destination_modulators(self) -> None:
        """Initialize prompt destination modulator.

        Latent destinations are dead — composition is driven entirely by
        CompositionEngine (noise circular walk). Only prompt SLERP
        (driven by tonal_distance) remains as a destination modulator.
        """
        device = self.pipeline.device
        dtype = self._base_latent.dtype if self._base_latent is not None else torch.float16

        self.prompt_destinations = DestinationModulator(device, dtype, self._bpm, self._fps)

        if self._prompt_embeds is not None and self._pooled_embeds is not None:
            self.prompt_destinations.load_a(Destination(
                tensor=self._prompt_embeds,
                tensor_pooled=self._pooled_embeds,
                label="Current prompt",
            ))

    def _serialize_slot_configs(self) -> Dict[str, BlockConfigSnapshot]:
        """Serialize current block configs for frontend sync."""
        payload: Dict[str, BlockConfigSnapshot] = {}
        for block, cfg in self._get_slot_configs().items():
            physics_preset = cfg.physics_preset
            if physics_preset == "default":
                physics_preset = "ambient"
            payload[block] = BlockConfigSnapshot(
                block=cfg.block,
                link_target=cfg.link_target,
                strength_min=cfg.strength_min,
                strength_max=cfg.strength_max,
                feature_id=cfg.feature_id,
                enabled=cfg.enabled,
                auto_config=cfg.auto_config,
                sae_rank=cfg.sae_rank,
                spatial_mode=cfg.spatial_mode,
                spatial_mask=cfg.spatial_mask,
                channel=cfg.channel,
                layer=cfg.layer,
                physics_preset=physics_preset,
                stage_left=cfg.stage_left,
                stage_home=cfg.stage_home,
                stage_right=cfg.stage_right,
                position_source=cfg.position_source,
                intensity_source=cfg.intensity_source,
                position_smoothing_ms=cfg.position_smoothing_ms,
                silence_behavior=cfg.silence_behavior,
                drift_ms=cfg.drift_ms,
                intensity_curve=cfg.intensity_curve,
                intensity_gamma=cfg.intensity_gamma,
            )
        return payload

    # ------------------------------------------------------------------
    # Template method hooks (called by base next_frame_batch)
    # ------------------------------------------------------------------

    def _is_ready(self) -> bool:
        """SAE needs prompt embeddings and a base latent to generate."""
        has_prompt = (
            self._prompt_embeds is not None
            or (self.prompt_destinations and self.prompt_destinations.has_destinations())
        )
        return has_prompt and self._base_latent is not None

    def _pre_generate(self, audio_time: float, dt: float) -> float:
        """Compute destination blend position before GPU forward."""
        frame_idx = self.audio_sampler.time_to_frame(audio_time)
        return self._compute_destination_blends(audio_time, dt, frame_idx)

    def _compute_destination_blends(
        self, audio_time: float, dt: float, frame_idx: int
    ) -> float:
        """Pre-compute prompt destination blend position.

        Latent destinations are dead — composition is handled by
        CompositionEngine. Only prompt SLERP remains.

        Returns:
            prompt_blend_pos (float)
        """
        prompt_pos = 0.0

        # Cache enabled stems list
        enabled_stems = self._get_active_base_stems()

        # Prompt destinations
        if self.prompt_destinations and self.prompt_destinations.has_destinations():
            if self.prompt_destinations.mode == "linked" and self.prompt_destinations.link_target:
                position, intensity, is_silent = self._sample_linked_destination(
                    self.prompt_destinations, audio_time
                )
                self.prompt_destinations.update_from_link(position, intensity, dt, is_silent)
            elif self.prompt_destinations.mode == "reactive" and self.prompt_destinations.reactive_config:
                position, intensity, is_silent = self._sample_global_destination(
                    self.prompt_destinations, audio_time, enabled_stems
                )
                self.prompt_destinations.update_from_reactive(position, intensity, dt, is_silent)
            prompt_pos = self.prompt_destinations.blend_position

        return prompt_pos

    def _sample_linked_destination(
        self,
        modulator,
        audio_time: float,
    ) -> tuple[Optional[float], float, bool]:
        cfg = modulator.reactive_config
        link_target = modulator.link_target

        # Handle derived/global targets
        if link_target in ("tension", "global"):
            pos, _ = self.audio_sampler.sample_position(None, audio_time, "tension_global")
            intensity = pos if pos is not None else 0.0
            return pos, intensity, intensity < 0.02

        # Tonal distance: JSD-based harmonic departure drives prompt SLERP
        if link_target == "tonal_distance":
            pos, _ = self.audio_sampler.sample_position(None, audio_time, "tonal_distance_global")
            intensity = pos if pos is not None else 0.0
            return pos, intensity, intensity < 0.02

        position, valid = self.audio_sampler.sample_position(
            link_target, audio_time, cfg.position_source
        )
        intensity = self.audio_sampler.sample_intensity(
            link_target, audio_time, cfg.intensity_source
        )
        is_silent = intensity < 0.02 or not valid
        return position, intensity, is_silent

    def _sample_global_destination(
        self,
        modulator,
        audio_time: float,
        enabled_stems: list,
    ) -> tuple[Optional[float], float, bool]:
        cfg = modulator.reactive_config

        # Default to the 4 base stems for global mode (simple).
        base_stems = [s for s in ("drums", "bass", "vocals", "other") if s in self.stem_features]
        stems = enabled_stems or base_stems
        if cfg.driver == "stem" and cfg.stem:
            stems = [cfg.stem]

        default_ranks = {"drums": 1, "bass": 2, "vocals": None, "other": None}
        stem_ranks = cfg.stem_rankings or default_ranks

        default_weights = {1: 1.0, 2: 0.75, 3: 0.5, 4: 0.25, None: 0.6}
        raw_weights = cfg.rank_weights or {}

        def weight_for(rank: Optional[int]) -> float:
            key = "auto" if rank is None else str(rank)
            if key in raw_weights:
                return float(raw_weights[key])
            if rank in default_weights:
                return default_weights[rank]
            return default_weights[None]

        total_weight = 0.0
        total_rank_weight = 0.0
        pos_weight = 0.0
        pos_sum = 0.0

        for stem in stems:
            if stem not in self.stem_features:
                continue

            position, valid = self.audio_sampler.sample_position(
                stem, audio_time, cfg.position_source
            )
            intensity = self.audio_sampler.sample_intensity(
                stem, audio_time, cfg.intensity_source
            )
            rank = stem_ranks.get(stem)
            weight = weight_for(rank)
            w = intensity * weight

            total_weight += w
            total_rank_weight += weight

            if valid and position is not None:
                pos_sum += position * w
                pos_weight += w

        if total_weight < 1e-6 or total_rank_weight <= 0:
            return None, 0.0, True

        position = pos_sum / pos_weight if pos_weight > 1e-6 else None
        intensity = total_weight / max(total_rank_weight, 1e-6)
        is_silent = total_weight < 0.02
        return position, intensity, is_silent

    def _gpu_forward(
        self,
        steering: dict,
        prompt_blend_pos: float,
        audio_time: float = 0.0,
    ) -> Optional[tuple]:
        """GPU forward + D2H transfer (runs entirely in executor).

        Combining these keeps the event loop free during the ~15ms inference
        that previously blocked the consumer from delivering frames.

        Returns:
            (cpu_tensor, prep_ms, infer_ms, d2h_ms) or None if generation failed
        """
        t_prep = time.perf_counter()

        # Update spatial masks (pass audio_sampler for intelligent spatial mode)
        self._spatial.update_masks(
            self.pipeline.steering_manager,
            self._get_slot_configs(),
            audio_sampler=self.audio_sampler,
            audio_time=audio_time,
        )

        # Get prompt embeds from destinations or base
        prompt_embeds = self._prompt_embeds
        pooled_embeds = self._pooled_embeds
        if self.prompt_destinations and self.prompt_destinations.has_destinations():
            prompt_embeds, pooled_embeds = self.prompt_destinations.step_dual(
                blend_position=prompt_blend_pos
            )

        # Composition (Axis 1: noise circular walk)
        noise = None
        if self._composition is not None and self._composition.has_both():
            noise = self._composition.get_noise(audio_time)

        # Validate shapes
        if len(prompt_embeds.shape) != 3 or len(pooled_embeds.shape) != 2:
            logger.error(f"Invalid embed shapes: {prompt_embeds.shape}, {pooled_embeds.shape}")
            return None

        prep_ms = (time.perf_counter() - t_prep) * 1000

        # GPU forward + explicit sync to separate inference from D2H
        t_infer = time.perf_counter()
        with torch.inference_mode():
            gpu_img = self.pipeline.generate_steered(
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_embeds,
                latents=self._base_latent,
                noise=noise,
                steerings=steering,
            )
        if self.pipeline.device == "cuda":
            torch.cuda.synchronize()
        infer_ms = (time.perf_counter() - t_infer) * 1000

        if gpu_img is None:
            return None

        # D2H copy (GPU already synced, this is pure memcpy)
        t_d2h = time.perf_counter()
        cpu_tensor = gpu_to_cpu_tensor(gpu_img)
        d2h_ms = (time.perf_counter() - t_d2h) * 1000

        return cpu_tensor, prep_ms, infer_ms, d2h_ms

    def _build_telemetry(
        self, t_start: float, audio_time: float, pre_ctx: Any
    ) -> List[FrameItem]:
        """Add destination status telemetry to base stem activity."""
        items = super()._build_telemetry(t_start, audio_time, pre_ctx)

        # Destination status (~2Hz) — prompt SLERP position
        prompt_blend_pos = pre_ctx  # pre_ctx is prompt_blend_pos for SAE
        if t_start - self._last_destination_status_time > 0.5:
            if self.prompt_destinations and self.prompt_destinations.has_destinations():
                items.append(FrameItem(
                    kind="json",
                    payload=DestinationStatus(
                        space="prompt",
                        destination_a=self.prompt_destinations.destination_a.label if self.prompt_destinations.destination_a else None,
                        destination_b=self.prompt_destinations.destination_b.label if self.prompt_destinations.destination_b else None,
                        blend_position=prompt_blend_pos,
                        mode=self.prompt_destinations.mode,
                    ).model_dump(),
                    due_ts=None,
                ))
            self._last_destination_status_time = t_start

        return items

    async def handle_message(self, message: BaseModel) -> Optional[dict]:
        """Handle control messages via grouped handlers."""
        if is_audio_message(message):
            return handle_audio_message(self, message)

        if is_stem_message(message):
            response = handle_stem_message(self, message)
            if response is not None:
                return response
            return BlockConfigs(configs=self._serialize_slot_configs()).model_dump()

        if is_destination_message(message):
            return handle_destination_message(self, message)

        if is_modulation_message(message):
            return handle_modulation_message(self, message)

        logger.warning(f"Unhandled message type: {type(message).__name__}")
        return None

