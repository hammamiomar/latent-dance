"""Tests for PipelineConfig and SAEConfig.

These tests validate configuration handling, especially the MPS dtype fix
that was causing runtime errors with float16 on Apple Silicon.
"""

import pytest
import torch

from hambajuba2ba.artifacts import (
    RUNTIME_SAE_FILES,
    UPSTREAM_SDXL_SAE_DIRS,
    _allow_patterns,
    find_sae_block_dir,
    has_sae_weights,
    resolve_sae_weights_dir,
)
from hambajuba2ba.config import PipelineConfig, SAEConfig


class TestPipelineConfig:
    """Tests for PipelineConfig dataclass."""

    def test_mps_derives_float32_by_default(self):
        """MPS defaults to float32 (conservative), but explicit float16
        is respected — validated on torch 2.10 / M1 Max (Jul 2026)."""
        derived = PipelineConfig(device="mps")
        assert derived.dtype == "float32"

        explicit = PipelineConfig(device="mps", dtype="float16")
        assert explicit.dtype == "float16"
        assert explicit.get_torch_dtype() == torch.float16

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

    def test_allow_patterns_download_only_runtime_files(self):
        config = SAEConfig(blocks=["down.2.1"])

        patterns = _allow_patterns(config)

        assert len(patterns) == 3 * len(RUNTIME_SAE_FILES)
        assert (
            f"{UPSTREAM_SDXL_SAE_DIRS['down.2.1']}/state_dict.pth"
            in patterns
        )
        assert all(not pattern.endswith("/*") for pattern in patterns)
        assert all(
            pattern.rsplit("/", 1)[-1] in RUNTIME_SAE_FILES
            for pattern in patterns
        )


class TestVariant:
    """The fp16 checkpoint variant is always the download target."""

    def test_variant_is_fp16_for_every_dtype(self):
        # diffusers upcasts fp16 files to torch_dtype at load, so float32
        # devices (mps/cpu) get correct weights from half the download
        assert PipelineConfig(device="cuda").variant == "fp16"
        assert PipelineConfig(device="cpu").variant == "fp16"
        assert PipelineConfig(device="cpu", dtype="float32").variant == "fp16"


class TestEnvLoader:
    """load_from_env applies HAMBAJUBA_* overrides coherently."""

    @pytest.fixture(autouse=True)
    def clean_hambajuba_env(self, monkeypatch):
        """Strip stray HAMBAJUBA_* vars so tests see only what they set."""
        import os

        for key in list(os.environ):
            if key.startswith("HAMBAJUBA_") or key.startswith("HAMBA_"):
                monkeypatch.delenv(key, raising=False)

    def test_device_override_rederives_dtype(self, monkeypatch):
        """HAMBAJUBA_DEVICE=cpu must not keep a float16 derived for CUDA."""
        from hambajuba2ba.config import base as config_base
        from hambajuba2ba.config import load_from_env

        # Simulate a CUDA box: construction derives float16, then the env
        # override to cpu must re-derive float32 (the staleness bug)
        monkeypatch.setattr(config_base, "autodetect", lambda: "cuda")
        monkeypatch.setenv("HAMBAJUBA_DEVICE", "cpu")
        config = load_from_env()
        assert config.device == "cpu"
        assert config.dtype == "float32"

    def test_explicit_dtype_override_wins(self, monkeypatch):
        from hambajuba2ba.config import load_from_env

        monkeypatch.setenv("HAMBAJUBA_DEVICE", "cpu")
        monkeypatch.setenv("HAMBAJUBA_DTYPE", "float16")
        config = load_from_env()
        assert config.device == "cpu"
        assert config.dtype == "float16"

    def test_explicit_mps_float16_respected_via_env(self, monkeypatch):
        """HAMBAJUBA_DTYPE=float16 on MPS is honored (fast local mode);
        only the DERIVED default stays float32."""
        from hambajuba2ba.config import load_from_env

        monkeypatch.setenv("HAMBAJUBA_DEVICE", "mps")
        monkeypatch.setenv("HAMBAJUBA_DTYPE", "float16")
        config = load_from_env()
        assert config.device == "mps"
        assert config.dtype == "float16"

    def test_feature_device_env_reaches_audio_config(self, monkeypatch):
        from hambajuba2ba.config import load_from_env

        monkeypatch.setenv("HAMBAJUBA_AUDIO_FEATURE_DEVICE", "mps")
        monkeypatch.setenv("HAMBAJUBA_AUDIO_FEATURE_BACKEND", "torch")
        config = load_from_env()
        assert config.audio.feature_device == "mps"
        assert config.audio.feature_backend == "torch"

    def test_every_env_map_path_shape_applies(self, monkeypatch):
        """All four path shapes: top-level, top-level+converter,
        nested, nested+converter. The (attr, converter) shape used to
        crash _set_nested (it was parsed as (section, attr))."""
        from hambajuba2ba.config import load_from_env

        monkeypatch.setenv("HAMBAJUBA_DEVICE", "cpu")  # (attr,)
        monkeypatch.setenv("HAMBAJUBA_WARMUP_ITERATIONS", "7")  # (attr, int)
        monkeypatch.setenv("HAMBAJUBA_SERVER_HOST", "0.0.0.0")  # (section, attr)
        monkeypatch.setenv("HAMBAJUBA_SERVER_PORT", "9100")  # (section, attr, int)
        config = load_from_env()
        assert config.device == "cpu"
        assert config.warmup_iterations == 7
        assert config.server.host == "0.0.0.0"
        assert config.server.port == 9100
