#!/usr/bin/env python3
"""Profile the SDXLTurboEngine vs diffusers pipeline.

Verifies:
1. No graph breaks in the compiled engine
2. Steady-state FPS after warmup
3. Per-component timing breakdown

Usage:
    python scripts/profile_engine.py
    python scripts/profile_engine.py --trace   # also export Perfetto trace
"""

import argparse
import time

import torch
from torch.profiler import profile, ProfilerActivity

from hambajuba2ba.config import PipelineConfig
from hambajuba2ba.generation.pipeline import SAESteerablePipeline

FRAMES = 100  # steady-state benchmark
STEERING = {
    "down.2.1": (2301, 10.0),
    "up.0.0": (2937, 5.0),
    "mid.0": (1388, 8.0),
    "up.0.1": (4977, 3.0),
}


def check_device():
    if not torch.cuda.is_available():
        print("CUDA not available — engine requires CUDA. Exiting.")
        raise SystemExit(1)

    print(f"PyTorch {torch.__version__}  CUDA {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"TF32 matmul: {torch.get_float32_matmul_precision()}")


def load_pipeline():
    print("\nLoading pipeline + engine ...")
    config = PipelineConfig(device="cuda")
    pipeline = SAESteerablePipeline(config)
    pipeline.load()

    print(f"Engine active: {pipeline._engine is not None}")
    print(f"UNet dtype: {pipeline.pipe.unet.dtype}")
    print(f"VAE type: {type(pipeline.pipe.vae).__name__}")
    return pipeline


def verify_graph(pipeline):
    """Check for graph breaks in the compiled engine."""
    engine = pipeline._engine
    if engine is None:
        print("\nNo engine — skipping graph break check")
        return

    print("\nChecking for graph breaks ...")
    dummy_latent = torch.randn(
        engine._latent_shape, device="cuda", dtype=engine.dtype
    )
    dummy_noise = torch.randn_like(dummy_latent)
    dummy_pe = torch.randn(1, 77, 2048, device="cuda", dtype=engine.dtype)
    dummy_pool = torch.randn(1, 1280, device="cuda", dtype=engine.dtype)

    try:
        explanation = torch._dynamo.explain(engine._generate_impl)(
            dummy_latent, dummy_noise, dummy_pe, dummy_pool
        )
        print(f"  Graph breaks: {explanation.graph_break_count}")
        if explanation.graph_break_count > 0:
            print("  Break reasons:")
            for reason in explanation.break_reasons:
                print(f"    - {reason}")
    except Exception as e:
        print(f"  explain() failed: {e}")
        print("  (This is OK — graph may still compile correctly)")


def benchmark_fps(pipeline):
    """Measure steady-state FPS."""
    print(f"\nBenchmarking {FRAMES} frames ...")
    prompt_embeds, pooled_embeds = pipeline.encode_prompt("a portrait of a person")
    latents = pipeline._base_latent

    # Apply steering
    if pipeline.steering_manager is not None:
        pipeline.steering_manager.set_steering(
            STEERING, use_mean_scaling=pipeline.config.sae.use_mean_scaling,
        )

    # Warmup (engine already warmed up during load, but do a few with steering)
    for _ in range(5):
        with torch.inference_mode():
            _ = pipeline.generate_steered(
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_embeds,
                latents=latents,
                steerings=STEERING,
            )
    torch.cuda.synchronize()

    # Timed run
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(FRAMES):
        with torch.inference_mode():
            result = pipeline.generate_steered(
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_embeds,
                latents=latents,
                steerings=STEERING,
            )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    fps = FRAMES / elapsed
    ms_per_frame = elapsed / FRAMES * 1000

    print(f"  Result shape: {result.shape}, dtype: {result.dtype}")
    print(f"  {FRAMES} frames in {elapsed:.2f}s")
    print(f"  FPS: {fps:.1f}")
    print(f"  ms/frame: {ms_per_frame:.2f}")
    return fps


def profile_trace(pipeline, output_path: str):
    """Export a Perfetto trace."""
    print(f"\nProfiling 20 frames → {output_path} ...")
    prompt_embeds, pooled_embeds = pipeline.encode_prompt("a portrait of a person")
    latents = pipeline._base_latent

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_flops=True,
    ) as prof:
        for _ in range(20):
            with torch.inference_mode():
                _ = pipeline.generate_steered(
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_embeds,
                    latents=latents,
                    steerings=STEERING,
                )
            torch.cuda.synchronize()

    prof.export_chrome_trace(output_path)
    print(f"  Trace saved to {output_path}")
    print("\n" + prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))


def main():
    parser = argparse.ArgumentParser(description="Profile SDXLTurboEngine")
    parser.add_argument("--trace", action="store_true", help="Export Perfetto trace")
    args = parser.parse_args()

    check_device()
    pipeline = load_pipeline()
    verify_graph(pipeline)
    fps = benchmark_fps(pipeline)

    if args.trace:
        profile_trace(pipeline, "engine_trace.json")

    print("\n" + "=" * 50)
    if fps >= 30:
        print(f"TARGET MET: {fps:.1f} FPS (>= 30)")
    else:
        print(f"BELOW TARGET: {fps:.1f} FPS (target: 30+)")
    print("=" * 50)


if __name__ == "__main__":
    main()
