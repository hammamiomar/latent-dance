"""Direct SDXL-Turbo inference engine — bypasses diffusers Pipeline.__call__().

Compiles the entire inference path (UNet + Euler step + VAE decode + uint8)
as a single torch.compile graph for maximum GPU throughput.

For SDXL-Turbo 1-step inference, the scheduler reduces to pure tensor math:

    noisy  = latent + noise * sigma              # add_noise
    scaled = noisy * input_scale                  # scale_model_input
    pred   = unet(scaled, timestep, embeds, cond) # denoise
    x0     = noisy - sigma * pred                 # euler step (sigma_next=0)
    image  = vae.decode(x0 * vae_scale)           # decode

All constants are pre-computed at init. The compiled graph has zero Python overhead.
"""

from __future__ import annotations

import logging
import time

import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global Dynamo settings — must be set BEFORE any torch.compile call
# ---------------------------------------------------------------------------
torch.set_float32_matmul_precision("medium")  # TF32 matmuls (3.5x on Blackwell)
torch._dynamo.config.capture_scalar_outputs = True  # avoid graph breaks from scalars


class SDXLTurboEngine:
    """Direct inference engine for SDXL-Turbo 1-step generation.

    Replaces the diffusers pipeline path with a single compiled graph
    covering UNet forward, Euler step, VAE decode, and uint8 conversion.

    Usage:
        engine = SDXLTurboEngine(pipe, config)
        engine.compile()                          # compile + warmup
        engine.init_seed(42)                      # one-time base noise buffer
        img = engine.gen_frame(latent, noise, pe, pool)  # (H,W,3) uint8 on GPU
    """

    def __init__(self, pipe, config):
        """Extract models and pre-compute all scheduler constants.

        Must be called AFTER InlineSAEManager has wrapped UNet attention blocks,
        but BEFORE any individual torch.compile on UNet/VAE.

        Args:
            pipe: Loaded diffusers pipeline (with InlineSAE wrappers applied).
            config: PipelineConfig instance.
        """
        self.device = config.device
        self.dtype = config.get_torch_dtype()
        self._latent_shape = (1, 4, config.latent_height, config.latent_width)

        # ---- Models (references, not copies) --------------------------------
        self.unet = pipe.unet
        self.vae = pipe.vae

        # ---- Scheduler constants --------------------------------------------
        scheduler = pipe.scheduler
        scheduler.set_timesteps(config.num_inference_steps, device=config.device)

        timestep = scheduler.timesteps[0]
        sigma = scheduler.sigmas[0].item()

        # Store as GPU tensors (consumed inside compiled graph)
        self._timestep = timestep.clone().detach().to(device=config.device)
        self._sigma = torch.tensor(sigma, device=config.device, dtype=self.dtype)
        self._input_scale = torch.tensor(
            1.0 / (sigma**2 + 1) ** 0.5,
            device=config.device,
            dtype=self.dtype,
        )
        self._vae_scale = torch.tensor(
            1.0 / pipe.vae.config.scaling_factor,
            device=config.device,
            dtype=self.dtype,
        )

        # SDXL micro-conditioning time_ids:
        # [orig_h, orig_w, crop_top, crop_left, target_h, target_w]
        self._time_ids = torch.tensor(
            [config.height, config.width, 0, 0, config.height, config.width],
            device=config.device,
            dtype=self.dtype,
        ).unsqueeze(0)  # (1, 6)

        # ---- Base noise buffer (one-time init per session) --------------------
        self._noise_buffer: torch.Tensor | None = None
        self._base_seed: int | None = None

        # ---- Compiled function (populated by compile()) ----------------------
        self._compiled_generate = None

        logger.info(
            "Engine constants: timestep=%s  sigma=%.4f  input_scale=%.6f  vae_scale=%.6f",
            self._timestep.item(),
            sigma,
            self._input_scale.item(),
            self._vae_scale.item(),
        )

    # ------------------------------------------------------------------
    # Compilation & warmup
    # ------------------------------------------------------------------

    def compile(self, warmup_iters: int = 10) -> None:
        """Compile the unified inference graph and run warmup.

        Uses fullgraph=True + reduce-overhead + dynamic=False for a single
        CUDA graph covering UNet → Euler step → VAE decode → uint8.

        Args:
            warmup_iters: Warmup iterations for CUDA graph capture (~10).
        """
        # reduce-overhead: CUDA graphs without Triton kernel autotuning.
        # max-autotune would be ideal but TritonGPUAccelerateMatmul crashes
        # on Blackwell (cc=120) — Triton's MLIR pass pipeline fails for
        # matmul shapes in SDXL UNet. cuBLAS handles these fine via
        # reduce-overhead. Revisit when Triton ships Blackwell fixes.
        mode = "reduce-overhead"
        logger.info(
            "Compiling unified inference graph "
            "(fullgraph=True, mode=%s, dynamic=False) ...",
            mode,
        )

        self._compiled_generate = torch.compile(
            self._generate_impl,
            fullgraph=True,
            mode=mode,
            dynamic=False,
        )

        # Warmup with dummy tensors that match real shapes
        dummy_latent = torch.randn(
            self._latent_shape, device=self.device, dtype=self.dtype
        )
        dummy_noise = torch.randn_like(dummy_latent)
        # SDXL prompt_embeds: (1, 77, 2048), pooled: (1, 1280)
        dummy_pe = torch.randn(1, 77, 2048, device=self.device, dtype=self.dtype)
        dummy_pool = torch.randn(1, 1280, device=self.device, dtype=self.dtype)

        logger.info(
            "Running %d warmup iterations; first iteration performs torch.compile",
            warmup_iters,
        )
        warmup_start = time.perf_counter()
        with torch.inference_mode():
            for i in range(warmup_iters):
                iter_start = time.perf_counter()
                logger.info("  warmup %d/%d starting", i + 1, warmup_iters)
                _ = self._compiled_generate(
                    dummy_latent, dummy_noise, dummy_pe, dummy_pool
                )
                elapsed = time.perf_counter() - iter_start
                suffix = " (graph compiled)" if i == 0 else ""
                logger.info(
                    "  warmup %d/%d complete in %.1fs%s",
                    i + 1,
                    warmup_iters,
                    elapsed,
                    suffix,
                )
        torch.cuda.synchronize()
        logger.info("Engine warmup complete in %.1fs", time.perf_counter() - warmup_start)

    # ------------------------------------------------------------------
    # Seed / noise management
    # ------------------------------------------------------------------

    def init_seed(self, seed: int) -> None:
        """One-time base noise buffer init for this session.

        Called once during setup. The noise buffer provides a default for
        gen_frame() when no external noise is supplied. For the composition
        system, CompositionEngine supplies noise directly via gen_frame().
        """
        if seed == self._base_seed and self._noise_buffer is not None:
            return
        gen = torch.Generator(device=self.device)
        gen.manual_seed(seed)
        self._noise_buffer = torch.randn(
            self._latent_shape,
            generator=gen,
            device=self.device,
            dtype=self.dtype,
        )
        self._base_seed = seed
        logger.info("Initialized base noise buffer for seed=%d", seed)

    def make_noise(self, seed: int) -> torch.Tensor:
        """Generate a noise tensor for a given seed (standalone factory).

        Used by CompositionEngine to pre-compute noise_a/noise_b for
        circular blending. Does not modify the engine's internal state.

        Args:
            seed: Random seed for noise generation

        Returns:
            Noise tensor of shape (1, 4, H/8, W/8) on GPU
        """
        gen = torch.Generator(device=self.device)
        gen.manual_seed(seed)
        return torch.randn(
            self._latent_shape,
            generator=gen,
            device=self.device,
            dtype=self.dtype,
        )

    # ------------------------------------------------------------------
    # Core inference (compiled)
    # ------------------------------------------------------------------

    def _generate_impl(
        self,
        latent: torch.Tensor,
        noise: torch.Tensor,
        prompt_embeds: torch.Tensor,
        pooled_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """Entire inference path — compiled into one CUDA graph.

        Args:
            latent:        (1, 4, H/8, W/8) — noise or SLERP'd latent
            noise:         (1, 4, H/8, W/8) — pre-computed noise buffer
            prompt_embeds: (1, seq, dim)     — text encoder output
            pooled_embeds: (1, dim)          — pooled text output

        Returns:
            (H, W, 3) uint8 tensor on GPU
        """
        # 1. Add noise (replicates scheduler.add_noise at highest sigma)
        #    sigma ≈ 14.6 → noise dominates ~95%, latent nudges composition.
        #    See notes/cleanup_journal.md for implications on destination SLERP.
        noisy = latent + noise * self._sigma

        # 2. Scale for UNet input (replicates scheduler.scale_model_input)
        scaled = noisy * self._input_scale

        # 3. UNet forward with SDXL conditioning
        #    return_dict=False → tuple output, avoids BaseOutput.__getattr__ graph breaks
        noise_pred = self.unet(
            scaled,
            self._timestep,
            encoder_hidden_states=prompt_embeds,
            added_cond_kwargs={
                "text_embeds": pooled_embeds,
                "time_ids": self._time_ids,
            },
            return_dict=False,
        )[0]

        # 4. Euler step (sigma_next = 0 for 1-step)
        denoised = noisy - self._sigma * noise_pred

        # 5. VAE decode
        decoded = self.vae.decode(denoised * self._vae_scale, return_dict=False)[0]

        # 6. Float → uint8 on GPU (fused into graph, avoids CPU conversion)
        decoded = (decoded * 0.5 + 0.5).clamp_(0.0, 1.0)
        decoded = (decoded * 255.0).to(torch.uint8)

        # 7. (1, 3, H, W) → (H, W, 3) contiguous for JPEG encoding
        return decoded.squeeze(0).permute(1, 2, 0).contiguous()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def gen_frame(
        self,
        latent: torch.Tensor,
        noise: torch.Tensor | None,
        prompt_embeds: torch.Tensor,
        pooled_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """Generate one frame.

        Args:
            latent:        (1, 4, H/8, W/8) — base latent (nudges composition ~5%)
            noise:         (1, 4, H/8, W/8) — from CompositionEngine (dominates ~95%).
                           If None, falls back to internal _noise_buffer.
            prompt_embeds: (1, seq, dim)     — from encode_prompt()
            pooled_embeds: (1, dim)          — from encode_prompt()

        Returns:
            (H, W, 3) uint8 tensor on GPU — ready for gpu_to_cpu_tensor()
        """
        if self._compiled_generate is None:
            raise RuntimeError("Call engine.compile() first")

        if noise is None:
            if self._noise_buffer is None:
                raise RuntimeError("Call engine.init_seed() first")
            noise = self._noise_buffer

        return self._compiled_generate(
            latent, noise, prompt_embeds, pooled_embeds
        )
