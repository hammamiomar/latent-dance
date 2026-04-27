"""Tests for PipelineConfig and SAEConfig.

These tests validate configuration handling, especially the MPS dtype fix
that was causing runtime errors with float16 on Apple Silicon.
"""

import pytest
import torch

from hambajuba2ba.artifacts import (
    UPSTREAM_SDXL_SAE_DIRS,
    find_sae_block_dir,
    has_sae_weights,
    resolve_sae_weights_dir,
)
from hambajuba2ba.config import PipelineConfig, SAEConfig


class TestPipelineConfig:
    """Tests for PipelineConfig dataclass."""

    def test_mps_forces_float32(self):
        """MPS device should override float16 to float32 for stability.

        This test catches the bug where MPS + float16 caused:
        "Input type (float) and bias type (c10::Half) should be the same"
        """
        config = PipelineConfig(device="mps", dtype="float16")
        assert config.dtype == "float32"
        assert config.get_torch_dtype() == torch.float32

    def test_cuda_keeps_float16(self):
        """CUDA should preserve float16 for performance."""
        config = PipelineConfig(device="cuda", dtype="float16")
        assert config.dtype == "float16"
        assert config.get_torch_dtype() == torch.float16

    def test_cpu_allows_float16(self):
        """CPU should allow float16 (though slower)."""
        config = PipelineConfig(device="cpu", dtype="float16")
        assert config.dtype == "float16"

    def test_cpu_allows_float32(self):
        """CPU should allow float32."""
        config = PipelineConfig(device="cpu", dtype="float32")
        assert config.dtype == "float32"
        assert config.get_torch_dtype() == torch.float32

    def test_cpu_allows_bfloat16(self):
        """CPU should allow bfloat16."""
        config = PipelineConfig(device="cpu", dtype="bfloat16")
        assert config.dtype == "bfloat16"
        assert config.get_torch_dtype() == torch.bfloat16

    def test_latent_dimensions_512(self):
        """Latent height/width should be 1/8 of image dimensions."""
        config = PipelineConfig(height=512, width=512)
        assert config.latent_height == 64
        assert config.latent_width == 64

    def test_latent_dimensions_1024(self):
        """Latent dimensions for 1024x1024 images."""
        config = PipelineConfig(height=1024, width=1024)
        assert config.latent_height == 128
        assert config.latent_width == 128

    def test_default_sae_blocks(self):
        """SAE config should have all 4 blocks by default."""
        config = PipelineConfig()
        assert len(config.sae.blocks) == 4
        assert "down.2.1" in config.sae.blocks
        assert "mid.0" in config.sae.blocks
        assert "up.0.0" in config.sae.blocks
        assert "up.0.1" in config.sae.blocks

    def test_default_sae_weights_dir(self):
        """SAE weights dir should default to ./data/sdxl/sae_weights."""
        config = PipelineConfig()
        assert config.sae.weights_dir == "./data/sdxl/sae_weights"

    def test_default_artifact_repo(self):
        """Public artifact repo should point to upstream SAE weights."""
        config = PipelineConfig()
        assert config.sae.artifact_repo_id == "surokpro2/sdxl-saes"
        assert config.sae.artifact_repo_type == "model"
        assert config.sae.artifact_weights_subdir == ""

    def test_default_num_inference_steps(self):
        """Default to 1-step inference for SDXL-Turbo."""
        config = PipelineConfig()
        assert config.num_inference_steps == 1

class TestSAEConfig:
    """Tests for SAEConfig dataclass."""

    def test_use_mean_scaling_default(self):
        """Mean scaling should be enabled by default."""
        sae_config = SAEConfig()
        assert sae_config.use_mean_scaling is True


class TestArtifacts:
    """Tests for public artifact resolution."""

    def test_has_sae_weights_accepts_final_layout(self, mock_sae_weights):
        assert has_sae_weights(mock_sae_weights, ["down.2.1"])

    def test_has_sae_weights_accepts_upstream_layout(self, tmp_path, mock_sae):
        block_dir = tmp_path / UPSTREAM_SDXL_SAE_DIRS["down.2.1"]
        block_dir.mkdir(parents=True)
        mock_sae.save_to_disk(str(block_dir))

        assert has_sae_weights(tmp_path, ["down.2.1"])
        assert find_sae_block_dir(tmp_path, "down.2.1") == block_dir

    def test_resolve_sae_weights_dir_prefers_local(self, mock_sae_weights):
        config = SAEConfig(
            weights_dir=str(mock_sae_weights),
            blocks=["down.2.1"],
            auto_download_weights=False,
        )
        assert resolve_sae_weights_dir(config) == mock_sae_weights

    def test_resolve_sae_weights_dir_errors_when_disabled(self, tmp_path):
        config = SAEConfig(
            weights_dir=str(tmp_path / "missing"),
            blocks=["down.2.1"],
            auto_download_weights=False,
        )
        with pytest.raises(FileNotFoundError):
            resolve_sae_weights_dir(config)
