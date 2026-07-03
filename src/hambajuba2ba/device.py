"""Device policy — the single home for device-conditional decisions.

Every "what do we do on cuda vs mps vs cpu?" question is answered here so
that the execution layer (generation, strategies, audio) never branches on
device strings itself. The config layer decides WHICH device to run on
(config/base.py autodetect + dtype rules); this module decides HOW to
drive it.

Import constraint: torch + stdlib only. config/ imports this module, and
config must remain an import-cycle-free leaf package. `torch.mps` is
present on every platform build (only `is_available()` differs), and all
dispatch happens inside function bodies, so importing this module is
side-effect-free everywhere.
"""

from __future__ import annotations

import torch


def autodetect() -> str:
    """Pick the best available device: cuda → mps → cpu."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def synchronize(device: str) -> None:
    """Block until all queued work on the device completes.

    GPU ops are async — call this before wall-clock timing or the numbers
    lie. CPU execution is synchronous by nature, so cpu is a no-op.
    """
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def empty_cache(device: str) -> None:
    """Release cached allocator blocks back to the system.

    Boundary-time relief only (scene start, shutdown): after a cache clear,
    the next transient allocation pays a slow cudaMalloc/OS round-trip, so
    never call this per-frame.
    """
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()


def compile_mode(device: str) -> str | None:
    """torch.compile mode for the frame-generation graph; None means eager.

    cuda: "reduce-overhead" — CUDA graphs without Triton kernel autotuning.
    max-autotune would be ideal but Triton's MLIR passes crash on Blackwell
    (cc=120) for SDXL UNet matmul shapes; cuBLAS via reduce-overhead handles
    them fine. Revisit when Triton ships Blackwell fixes.

    mps/cpu: eager — at seconds-per-frame throughput, inductor's non-CUDA
    backends don't repay their compile latency. Revisit if MPS becomes a
    performance target rather than a correctness target.
    """
    return "reduce-overhead" if device == "cuda" else None


def configure_backend(device: str) -> None:
    """Apply process-wide torch settings, once, at pipeline load time.

    Must run BEFORE any torch.compile call — dynamo reads its config at
    compile time. pipeline.load() calls this ahead of engine creation.
    """
    if device == "cuda":
        torch.set_float32_matmul_precision("medium")  # TF32 matmuls (3.5x on Blackwell)
        torch.backends.cudnn.benchmark = True  # autotune conv algos for our fixed shapes
    # Scalar constants inside the compiled graph (sigma, scales) must not
    # cause graph breaks. Harmless in eager mode, so set unconditionally.
    torch._dynamo.config.capture_scalar_outputs = True
