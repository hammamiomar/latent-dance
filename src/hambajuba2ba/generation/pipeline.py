"""SDXL-Turbo pipeline with SAE feature steering.

Loads SDXL-Turbo, applies InlineSAE wrappers to UNet attention blocks,
and builds a compiled SDXLTurboEngine for inference. The diffusers pipeline
is only used for model loading and prompt encoding — all frame generation
goes through the compiled CUDA graph.

Usage:
    from hambajuba2ba import PipelineConfig
    from hambajuba2ba.generation.pipeline import SAESteerablePipeline

    config = PipelineConfig(pipeline_type="sdxl_sae", device="cuda")
    pipe = SAESteerablePipeline(config)
    pipe.load()

    image = pipe.generate_steered(
        prompt="portrait of a woman",
        steerings={"down.2.1": (2301, 15.0)},
    )
"""

from __future__ import annotations

import logging
import torch
from diffusers import AutoPipelineForImage2Image, AutoencoderTiny

from hambajuba2ba.artifacts import resolve_sae_weights_dir
from hambajuba2ba.config import PipelineConfig
from .engine import SDXLTurboEngine
from .sae import InlineSAEManager

logger = logging.getLogger(__name__)


class SAESteerablePipeline:
    """SDXL-Turbo pipeline with SAE feature steering.

    Loads models, applies SAE wrappers, and builds a compiled inference engine.
    The diffusers pipeline is used only for loading — all inference goes through
    SDXLTurboEngine (compiled UNet + Euler + VAE as one CUDA graph).

    Attributes:
        config: Pipeline configuration
        pipe: Underlying diffusers pipeline (used for loading/encoding only)
        steering_manager: SAE steering manager (InlineSAEManager on CUDA)
    """

    def __init__(self, config: PipelineConfig):
        """Initialize pipeline with config.

        Args:
            config: Pipeline configuration (should have pipeline_type="sdxl_sae")
        """
        self.config = config
        self.pipe = None
        self._steering_manager: InlineSAEManager | None = None
        self.device = config.device

        # Compiled inference engine (built in load())
        self._engine = None

        # Cached embeddings
        self._cached_prompt: str | None = None
        self._cached_embeds: tuple[torch.Tensor, torch.Tensor] | None = None

        # Base latent: generated once at init, never changes.
        # Contributes ~5% to composition via: noisy = latent + noise * sigma
        self._base_latent: torch.Tensor | None = None

        # Feature ID cache for optimized set_steering vs update_strengths dispatch
        self._last_feature_ids: dict = {}

        logger.info(f"Initializing SAESteerablePipeline on {self.device}")

    def load(self) -> None:
        """Load SDXL-Turbo model and SAE weights.

        This is expensive (~2 minutes first time for model download).
        """
        logger.info(f"Loading SDXL-Turbo from {self.config.sdxl_model_id}")

        # Load SDXL-Turbo pipeline
        self.pipe = AutoPipelineForImage2Image.from_pretrained(
            self.config.sdxl_model_id,
            torch_dtype=self.config.get_torch_dtype(),
            variant=self.config.variant,
            safety_checker=None,
        ).to(self.device)

        # Apply optimizations (VAE, eval mode, etc.)
        self._optimize()

        # Initialize InlineSAEManager — wraps UNet attention blocks for
        # torch.compile-safe steering (must happen BEFORE engine creation)
        sae_weights_dir = resolve_sae_weights_dir(self.config.sae)
        logger.info("Loading SAE weights from %s", sae_weights_dir)
        self._steering_manager = InlineSAEManager(
            unet=self.pipe.unet,
            sae_weights_dir=str(sae_weights_dir),
            device=self.device,
            dtype=self.config.get_torch_dtype(),
            blocks=self.config.sae.blocks,
        )

        # Pre-allocate spatial activation map buffers (compile-safe copy_() targets)
        if self._steering_manager is not None and \
           hasattr(self._steering_manager, "init_activation_maps"):
            self._steering_manager.init_activation_maps(
                self.config.latent_height, self.config.latent_width
            )
            logger.info("Pre-allocated spatial activation maps")

        # Build and compile inference engine (UNet + Euler + VAE as one CUDA graph)
        self._engine = SDXLTurboEngine(self.pipe, self.config)
        self._engine.compile(
            warmup_iters=self.config.warmup_iterations,
        )

        # Generate base latent once — contributes ~5% to composition.
        # This never changes during a session.
        self._base_latent = self._engine.make_noise(self.config.seed)
        self._engine.init_seed(self.config.seed)

        logger.info("Pipeline loaded and compiled")

    def _optimize(self) -> None:
        """Apply performance optimizations."""
        # Disable progress bars
        self.pipe.set_progress_bar_config(disable=True)

        # Explicitly set to evaluation mode (disables dropout, uses running stats for batchnorm)
        self.pipe.unet.train(False)

        # TinyVAE: 2x faster decode
        if self.config.use_tiny_vae:
            logger.info("Using TinyVAE")
            self.pipe.vae = AutoencoderTiny.from_pretrained(
                self.config.sdxl_vae_id,
                torch_dtype=self.config.get_torch_dtype(),
            ).to(self.device)

        self.pipe.vae.train(False)

        # CUDA optimizations
        if self.device == "cuda":
            self.pipe.unet = self.pipe.unet.to(memory_format=torch.channels_last)
            self.pipe.vae = self.pipe.vae.to(memory_format=torch.channels_last)
            torch.backends.cudnn.benchmark = True

    @property
    def steering_manager(self) -> InlineSAEManager | None:
        """Access the InlineSAEManager for direct steering control."""
        return self._steering_manager

    @property
    def engine(self) -> SDXLTurboEngine | None:
        """Access the compiled inference engine."""
        return self._engine

    def encode_prompt(
        self,
        prompt: str,
        use_cache: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode text prompt to SDXL embeddings.

        SDXL uses dual text encoders, returning both standard
        embeddings and pooled embeddings.

        Args:
            prompt: Text prompt
            use_cache: Whether to cache and reuse embeddings

        Returns:
            Tuple of (prompt_embeds, pooled_prompt_embeds)
        """
        # Check cache
        if use_cache and prompt == self._cached_prompt and self._cached_embeds is not None:
            return self._cached_embeds

        # Encode with both SDXL text encoders
        with torch.no_grad():
            # Get embeddings from pipeline's encode method
            (
                prompt_embeds,
                _negative_prompt_embeds,
                pooled_prompt_embeds,
                _negative_pooled,
            ) = self.pipe.encode_prompt(
                prompt=prompt,
                device=self.device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=False,
            )

        result = (prompt_embeds, pooled_prompt_embeds)

        # Cache for reuse
        if use_cache:
            self._cached_prompt = prompt
            self._cached_embeds = result

        return result

    def generate_steered(
        self,
        prompt: str | None = None,
        prompt_embeds: torch.Tensor | None = None,
        pooled_prompt_embeds: torch.Tensor | None = None,
        latents: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
        steerings: dict[str, tuple[int, float]] | None = None,
    ) -> torch.Tensor:
        """Generate a single frame with SAE steering.

        Args:
            prompt: Text prompt (optional if embeddings provided)
            prompt_embeds: Pre-computed prompt embeddings
            pooled_prompt_embeds: Pre-computed pooled embeddings
            latents: Base latent (defaults to _base_latent, ~5% of composition)
            noise: Noise tensor from CompositionEngine (~95% of composition).
                   If None, engine falls back to its internal noise buffer.
            steerings: {block: (feature_id, strength)} steering config

        Returns:
            (H, W, 3) uint8 tensor on GPU
        """
        # Encode prompt if needed
        if prompt_embeds is None:
            if prompt is None:
                prompt = "portrait of a person"
            prompt_embeds, pooled_prompt_embeds = self.encode_prompt(prompt)

        # Use base latent if none provided
        if latents is None:
            latents = self._base_latent

        # Apply steering — optimized to only call set_steering when features change
        if self._steering_manager is not None:
            if steerings:
                current_features = {block: fid for block, (fid, _) in steerings.items()}
                if current_features != self._last_feature_ids:
                    # Feature IDs changed — full set_steering
                    self._steering_manager.set_steering(
                        steerings,
                        use_mean_scaling=self.config.sae.use_mean_scaling,
                    )
                    self._last_feature_ids = current_features.copy()
                else:
                    # Only strengths changed — fast path
                    self._steering_manager.update_strengths(
                        steerings,
                        use_mean_scaling=self.config.sae.use_mean_scaling,
                    )
            elif self._last_feature_ids:
                self._steering_manager.clear_hooks()
                self._last_feature_ids = {}

        return self._engine.gen_frame(latents, noise, prompt_embeds, pooled_prompt_embeds)

    def cleanup(self) -> None:
        """Free GPU memory and clear hooks."""
        # Clear any remaining hooks
        if self._steering_manager is not None:
            self._steering_manager.clear_hooks()

        # Clear embedding cache
        self._cached_prompt = None
        self._cached_embeds = None

        # Clear GPU cache
        torch.cuda.empty_cache()

        logger.info("GPU cache cleared")
