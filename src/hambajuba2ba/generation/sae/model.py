"""Sparse Autoencoder for interpretable feature extraction.

Ported from sdxl-unbox/SAE/sae.py, optimized for inference-only use.
The SAE learns to decompose neural network activations into sparse,
interpretable features using Top-K activation.

Architecture:
    latents = relu(topk(encoder(x - pre_bias) + latent_bias))
    recons = decoder(latents) + pre_bias

The decoder weights are constrained to unit norm per feature direction,
making feature strengths directly interpretable as magnitudes.
"""

from __future__ import annotations

import json
import logging
import os

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class SparseAutoencoder(nn.Module):
    """Top-K Sparse Autoencoder for SDXL UNet activations.

    This autoencoder learns sparse, interpretable representations of
    diffusion model activations. Only the top-k features activate for
    any input, making the representation highly sparse.

    Attributes:
        n_dirs_local: Number of sparse features (typically 5120)
        d_model: Input/output dimension (typically 1280 for SDXL)
        k: Number of top activations to keep (sparsity level)
    """

    def __init__(
        self,
        n_dirs_local: int,
        d_model: int,
        k: int,
        auxk: int | None = None,
        dead_steps_threshold: int = 0,
    ):
        """Initialize the Sparse Autoencoder.

        Args:
            n_dirs_local: Number of feature directions (sparse dimension)
            d_model: Model dimension (input/output size)
            k: Top-K sparsity level (number of active features)
            auxk: Auxiliary K for dead neuron handling (training only)
            dead_steps_threshold: Steps before neuron considered dead (training only)
        """
        super().__init__()
        self.n_dirs_local = n_dirs_local
        self.d_model = d_model
        self.k = k
        self.auxk = auxk
        self.dead_steps_threshold = dead_steps_threshold

        # Encoder: project input to sparse feature space
        self.encoder = nn.Linear(d_model, n_dirs_local, bias=False)

        # Decoder: reconstruct from sparse features
        self.decoder = nn.Linear(n_dirs_local, d_model, bias=False)

        # Biases
        self.pre_bias = nn.Parameter(torch.zeros(d_model))
        self.latent_bias = nn.Parameter(torch.zeros(n_dirs_local))

        # Training statistics buffer (not used in inference)
        self.register_buffer(
            "stats_last_nonzero",
            torch.zeros(n_dirs_local, dtype=torch.long),
        )

        # Initialize with tied weights
        self.decoder.weight.data = self.encoder.weight.data.T.clone()
        self.decoder.weight.data = self.decoder.weight.data.T.contiguous().T

        # Normalize decoder to unit norm
        unit_norm_decoder_(self)

    @classmethod
    def load_from_disk(
        cls,
        path: str,
        device: str | torch.device = "cuda",
        dtype: torch.dtype | None = None,
        decoder_only: bool = False,
    ) -> SparseAutoencoder:
        """Load a trained SAE from disk.

        Args:
            path: Directory containing config.json and state_dict.pth
            device: Target device.
            dtype: Target dtype. If None, keeps saved dtype.
            decoder_only: If True, discard encoder weights after loading
                         to save VRAM. Only decoder.weight is retained.

        Returns:
            Loaded SparseAutoencoder instance.
        """
        config_path = os.path.join(path, "config.json")
        weights_path = os.path.join(path, "state_dict.pth")

        with open(config_path, "r") as f:
            cfg = json.load(f)

        ae = cls(
            n_dirs_local=cfg["n_dirs_local"],
            d_model=cfg["d_model"],
            k=cfg["k"],
            auxk=cfg.get("auxk"),
            dead_steps_threshold=cfg.get("dead_steps_threshold", 0),
        )

        state_dict = torch.load(weights_path, map_location=device, weights_only=True)
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        ae.load_state_dict(state_dict)

        if dtype is not None:
            ae.to(device=device, dtype=dtype)
        else:
            ae.to(device)

        if decoder_only:
            # Discard encoder + training-only parameters to free VRAM.
            # encoder.weight alone is ~104MB per block (d_model × n_dirs × dtype).
            del ae.encoder
            del ae.pre_bias
            del ae.latent_bias
            del ae.stats_last_nonzero
            logger.debug("Discarded encoder weights for %s (decoder_only=True)", path)

        return ae

    def save_to_disk(self, path: str) -> None:
        """Save the SAE to disk.

        Args:
            path: Directory to save config.json and state_dict.pth
        """
        os.makedirs(path, exist_ok=True)

        cfg = {
            "n_dirs_local": self.n_dirs_local,
            "d_model": self.d_model,
            "k": self.k,
            "auxk": self.auxk,
            "dead_steps_threshold": self.dead_steps_threshold,
        }

        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(cfg, f)

        torch.save(
            {"state_dict": self.state_dict()},
            os.path.join(path, "state_dict.pth"),
        )

    @property
    def n_dirs(self) -> int:
        """Total number of feature directions."""
        return self.n_dirs_local

    @property
    def device(self) -> torch.device:
        """Device where the model parameters reside."""
        return next(self.parameters()).device

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to sparse latent representation.

        Applies Top-K selection followed by ReLU to produce a sparse
        activation tensor where at most k features are non-zero.

        Args:
            x: Input tensor of shape (..., d_model)

        Returns:
            Sparse latent tensor of shape (..., n_dirs_local)
            with at most k non-zero values per position
        """
        # Remove pre-bias and project to feature space
        x = x - self.pre_bias
        latents_pre_act = self.encoder(x) + self.latent_bias

        # Top-K selection for sparsity
        vals, inds = torch.topk(latents_pre_act, k=self.k, dim=-1)

        # Create sparse output via scatter
        latents = torch.zeros_like(latents_pre_act)
        latents.scatter_(-1, inds, torch.relu(vals))

        return latents

    def encode_topk(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode and return top-k indices and values.

        More efficient than full encode when you only need the
        sparse representation without the full dense tensor.

        Args:
            x: Input tensor of shape (..., d_model)

        Returns:
            Tuple of (indices, values) each of shape (..., k)
        """
        x = x - self.pre_bias
        latents_pre_act = self.encoder(x) + self.latent_bias
        vals, inds = torch.topk(latents_pre_act, k=self.k, dim=-1)
        return inds, torch.relu(vals)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode from dense latent representation.

        Args:
            latents: Latent tensor of shape (..., n_dirs_local)

        Returns:
            Reconstructed tensor of shape (..., d_model)
        """
        return latents @ self.decoder.weight.T + self.pre_bias

    def decode_sparse(
        self,
        inds: torch.Tensor,
        vals: torch.Tensor,
    ) -> torch.Tensor:
        """Decode from sparse indices and values.

        More memory-efficient than decode() when working with
        the output of encode_topk().

        Args:
            inds: Feature indices of shape (batch, k)
            vals: Feature values of shape (batch, k)

        Returns:
            Reconstructed tensor of shape (batch, d_model)
        """
        rows = inds.shape[0]
        cols = self.n_dirs

        row_indices = (
            torch.arange(rows, device=inds.device)
            .unsqueeze(1)
            .expand(-1, inds.shape[1])
            .reshape(-1)
        )
        vals_flat = vals.reshape(-1)
        inds_flat = inds.reshape(-1)

        indices = torch.stack([row_indices, inds_flat])
        sparse_tensor = torch.sparse_coo_tensor(
            indices, vals_flat, torch.Size([rows, cols])
        )

        recons = torch.sparse.mm(sparse_tensor, self.decoder.weight.T) + self.pre_bias
        return recons

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full forward pass: encode then decode.

        Args:
            x: Input tensor of shape (..., d_model)

        Returns:
            Reconstructed tensor of shape (..., d_model)
        """
        latents = self.encode(x)
        return self.decode(latents)


def unit_norm_decoder_(autoencoder: SparseAutoencoder) -> None:
    """Normalize decoder weights to unit norm per feature direction.

    This in-place operation ensures each feature direction has unit
    magnitude, making activation strengths directly interpretable.

    Args:
        autoencoder: SAE instance to normalize
    """
    autoencoder.decoder.weight.data /= autoencoder.decoder.weight.data.norm(dim=0)
