#!/usr/bin/env python3
"""Profile SDXL-Turbo inference with PyTorch Profiler.

Exports trace to Perfetto format for visualization.

Usage:
    python scripts/profile_inference.py

Then open chrome://tracing or https://ui.perfetto.dev and load the trace.
"""

import torch
from torch.profiler import profile, ProfilerActivity, schedule

from hambajuba2ba.config import PipelineConfig
from hambajuba2ba.generation.pipeline import SAESteerablePipeline

# Detect device
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
dtype = "bfloat16" if device == "cuda" else "float32"

print(f"Device: {device}, dtype: {dtype}")
print(f"PyTorch: {torch.__version__}")
if device == "cuda":
    print(f"CUDA: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name()}")

# Load pipeline
print("\nLoading pipeline...")
config = PipelineConfig(
    pipeline_type="sdxl_sae",
    device=device,
    dtype=dtype,
)
pipeline = SAESteerablePipeline(config)
pipeline.load()

print(f"UNet dtype: {pipeline.pipe.unet.dtype}")
print(f"VAE dtype: {pipeline.pipe.vae.dtype}")
print(f"VAE type: {type(pipeline.pipe.vae).__name__}")

# Prepare inputs
print("\nPreparing inputs...")
prompt_embeds, pooled_embeds = pipeline.encode_prompt("a portrait of a person")
latents = pipeline._base_latent

# Warmup
print("Warming up...")
for _ in range(3):
    with torch.inference_mode():
        _ = pipeline.generate_steered(
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_embeds,
            latents=latents,
            steerings={"down.2.1": (2301, 10.0)},  # With steering
        )
if device == "cuda":
    torch.cuda.synchronize()

# Profile
print("\nProfiling (20 frames)...")
trace_path = "inference_trace.json"

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=schedule(wait=1, warmup=2, active=10, repeat=1),
    on_trace_ready=lambda p: p.export_chrome_trace(trace_path),
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
) as prof:
    for i in range(20):
        with torch.inference_mode():
            _ = pipeline.generate_steered(
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_embeds,
                latents=latents,
                steerings={"down.2.1": (2301, 10.0), "up.0.0": (2937, 5.0)},
            )
        if device == "cuda":
            torch.cuda.synchronize()
        prof.step()

print(f"\nTrace saved to: {trace_path}")
print("Open in: chrome://tracing or https://ui.perfetto.dev")

# Print summary table
print("\n" + "=" * 60)
print("PROFILER SUMMARY (sorted by CUDA time)")
print("=" * 60)
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

print("\n" + "=" * 60)
print("PROFILER SUMMARY (sorted by CPU time)")
print("=" * 60)
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))
