"""Tests for the device policy module (Phase 5).

hambajuba2ba.device is the single home for device-conditional decisions.
These tests pin the dispatch table, prove configure_backend touches the
right process globals (and restores cleanly), and guard against CUDA-isms
creeping back into the execution layer.
"""

from pathlib import Path

import pytest
import torch

from hambajuba2ba import device as device_policy

REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Dispatch: synchronize / empty_cache route to the right torch namespace
# ---------------------------------------------------------------------------


class TestDispatch:
    def test_synchronize_dispatches_cuda(self, monkeypatch):
        calls = []
        monkeypatch.setattr(torch.cuda, "synchronize", lambda: calls.append("cuda"))
        device_policy.synchronize("cuda")
        assert calls == ["cuda"]

    def test_synchronize_dispatches_mps(self, monkeypatch):
        calls = []
        monkeypatch.setattr(torch.mps, "synchronize", lambda: calls.append("mps"))
        device_policy.synchronize("mps")
        assert calls == ["mps"]

    def test_synchronize_cpu_is_noop(self, monkeypatch):
        def boom():  # pragma: no cover - must never run
            raise AssertionError("dispatched on cpu")

        monkeypatch.setattr(torch.cuda, "synchronize", boom)
        monkeypatch.setattr(torch.mps, "synchronize", boom)
        device_policy.synchronize("cpu")  # no error, no dispatch

    def test_empty_cache_dispatches_cuda(self, monkeypatch):
        calls = []
        monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append("cuda"))
        device_policy.empty_cache("cuda")
        assert calls == ["cuda"]

    def test_empty_cache_dispatches_mps(self, monkeypatch):
        calls = []
        monkeypatch.setattr(torch.mps, "empty_cache", lambda: calls.append("mps"))
        device_policy.empty_cache("mps")
        assert calls == ["mps"]

    def test_empty_cache_cpu_is_noop(self, monkeypatch):
        def boom():  # pragma: no cover - must never run
            raise AssertionError("dispatched on cpu")

        monkeypatch.setattr(torch.cuda, "empty_cache", boom)
        monkeypatch.setattr(torch.mps, "empty_cache", boom)
        device_policy.empty_cache("cpu")


# ---------------------------------------------------------------------------
# Compile mode: CUDA graphs on cuda, eager everywhere else
# ---------------------------------------------------------------------------


class TestCompileMode:
    def test_cuda_uses_reduce_overhead(self):
        # reduce-overhead, not max-autotune: Triton crashes on Blackwell cc=120
        assert device_policy.compile_mode("cuda") == "reduce-overhead"

    @pytest.mark.parametrize("device", ["mps", "cpu"])
    def test_non_cuda_is_eager(self, device):
        assert device_policy.compile_mode(device) is None


# ---------------------------------------------------------------------------
# Autodetect: cuda -> mps -> cpu probe order
# ---------------------------------------------------------------------------


class TestAutodetect:
    def test_prefers_cuda(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        assert device_policy.autodetect() == "cuda"

    def test_falls_back_to_mps(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
        assert device_policy.autodetect() == "mps"

    def test_falls_back_to_cpu(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
        assert device_policy.autodetect() == "cpu"


# ---------------------------------------------------------------------------
# configure_backend: process-wide globals, applied per device
# ---------------------------------------------------------------------------


@pytest.fixture
def restore_torch_globals():
    """Save/restore process-wide torch state this test mutates."""
    precision = torch.get_float32_matmul_precision()
    scalar_capture = torch._dynamo.config.capture_scalar_outputs
    cudnn_benchmark = torch.backends.cudnn.benchmark
    yield
    torch.set_float32_matmul_precision(precision)
    torch._dynamo.config.capture_scalar_outputs = scalar_capture
    torch.backends.cudnn.benchmark = cudnn_benchmark


class TestConfigureBackend:
    def test_cuda_sets_all_globals(self, restore_torch_globals):
        torch.set_float32_matmul_precision("highest")
        torch.backends.cudnn.benchmark = False
        torch._dynamo.config.capture_scalar_outputs = False

        device_policy.configure_backend("cuda")

        assert torch.get_float32_matmul_precision() == "medium"
        assert torch.backends.cudnn.benchmark is True
        assert torch._dynamo.config.capture_scalar_outputs is True

    @pytest.mark.parametrize("device", ["mps", "cpu"])
    def test_non_cuda_leaves_cuda_globals_alone(self, device, restore_torch_globals):
        torch.set_float32_matmul_precision("highest")
        torch.backends.cudnn.benchmark = False
        torch._dynamo.config.capture_scalar_outputs = False

        device_policy.configure_backend(device)

        assert torch.get_float32_matmul_precision() == "highest"
        assert torch.backends.cudnn.benchmark is False
        # dynamo scalar capture is compile-time config, set on every device
        assert torch._dynamo.config.capture_scalar_outputs is True


# ---------------------------------------------------------------------------
# Migration guards: CUDA-isms must not creep back into the execution layer
# ---------------------------------------------------------------------------


class TestMigrationGuards:
    def _sources(self, *rel_dirs: str):
        for rel in rel_dirs:
            for path in sorted((REPO_ROOT / rel).rglob("*.py")):
                yield path, path.read_text()

    def test_engine_has_no_import_time_torch_globals(self):
        src = (REPO_ROOT / "src/hambajuba2ba/generation/engine.py").read_text()
        assert "set_float32_matmul_precision" not in src
        assert "torch._dynamo" not in src

    def test_no_bare_cuda_calls_outside_device_policy(self):
        offenders = []
        for path, src in self._sources(
            "src/hambajuba2ba/generation", "app/strategies"
        ):
            if "torch.cuda.synchronize" in src or "torch.cuda.empty_cache" in src:
                offenders.append(str(path.relative_to(REPO_ROOT)))
        assert offenders == [], f"route through hambajuba2ba.device: {offenders}"


# ---------------------------------------------------------------------------
# Spectral resolution: explicit mps is honored, auto stays conservative
# ---------------------------------------------------------------------------


class TestSpectralResolution:
    def test_explicit_mps_uses_torch_backend(self):
        from hambajuba2ba.audio.spectral import _resolve_backend

        assert _resolve_backend("auto", "mps") == ("torch", "mps")
        assert _resolve_backend("torch", "mps") == ("torch", "mps")

    def test_auto_never_self_selects_mps(self, monkeypatch):
        """On a CUDA-less machine, auto must stay librosa/cpu (MPS DSP is opt-in)."""
        from hambajuba2ba.audio import spectral

        monkeypatch.setattr(spectral.torch.cuda, "is_available", lambda: False)
        assert spectral._resolve_backend("auto", "auto") == ("librosa", "cpu")
