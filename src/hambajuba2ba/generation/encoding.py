"""GPU-to-CPU transfer and JPEG encoding utilities.

Isolates the D2H synchronization and CPU-bound JPEG encoding
from the generation strategy, enabling double-buffered pipelining.
"""

from __future__ import annotations

import torch
import turbojpeg

# Module-level TurboJPEG encoder (releases GIL, 5-10x faster than PIL)
_jpeg_encoder = turbojpeg.TurboJPEG()


def gpu_to_cpu_tensor(gpu_tensor: torch.Tensor) -> torch.Tensor:
    """Move GPU tensor to CPU with proper format for encoding.

    This function isolates the cudaDeviceSynchronize that happens during
    the D2H copy. Call this immediately after GPU work completes, then
    pass the CPU tensor to the encoder in a separate executor call.

    Handles two input formats:
    - Engine path: (H, W, 3) uint8 on GPU — already converted, just D2H copy
    - Legacy path: (C, H, W) float in [0, 1] on GPU — permute + convert

    Args:
        gpu_tensor: Image tensor on GPU (either format above)

    Returns:
        (H, W, C) uint8 tensor on CPU, ready for encoding
    """
    # Engine path: already (H, W, 3) uint8 — just copy to CPU
    if gpu_tensor.dtype == torch.uint8:
        return gpu_tensor.cpu()

    # Legacy path: (C, H, W) float in [0, 1]
    # Permute on GPU (essentially free), then contiguous for efficient D2H copy
    if gpu_tensor.dim() == 3 and gpu_tensor.shape[0] in (1, 3, 4):
        img = gpu_tensor.permute(1, 2, 0).contiguous().cpu()
    else:
        img = gpu_tensor.cpu()

    # In-place ops to avoid allocations
    img.clamp_(0, 1).mul_(255)
    return img.to(torch.uint8)


def encode_cpu_tensor(
    cpu_tensor: torch.Tensor,
    quality: int = 75,
) -> bytes:
    """Encode a single CPU tensor to JPEG bytes.

    This function takes a CPU tensor (already transferred from GPU).
    Use with double-buffering: GPU work can proceed while this encodes.

    Args:
        cpu_tensor: (H, W, C) uint8 tensor on CPU
        quality: JPEG quality (1-100)

    Returns:
        JPEG-encoded bytes
    """
    img_np = cpu_tensor.numpy()

    # Handle grayscale
    if img_np.shape[-1] == 1:
        img_np = img_np.squeeze(-1)
        return _jpeg_encoder.encode(
            img_np,
            quality=quality,
            pixel_format=turbojpeg.TJPF_GRAY,
        )
    else:
        return _jpeg_encoder.encode(
            img_np,
            quality=quality,
            pixel_format=turbojpeg.TJPF_RGB,
        )
