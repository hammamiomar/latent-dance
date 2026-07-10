"""Tests for the backend registry (app/backends.py).

The registry is the pluggability seam: main.py, create_strategy, and the
capabilities endpoint all resolve backends through it. These tests pin the
contract a new backend must satisfy, and check the SAE manifest against
independent runtime sources of truth (not against backends.py itself).
"""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from app.backends import get_backend, register_backend

GOLDEN_FIXTURE = Path(__file__).parent / "fixtures" / "capabilities.sae_steering.json"
MOCK_FIXTURE = Path(__file__).parent / "fixtures" / "capabilities.mock.json"


class TestRegistry:
    def test_sae_steering_is_registered(self):
        spec = get_backend("sae_steering")
        assert spec.mode == "sae_steering"
        assert spec.capabilities.mode == "sae_steering"
        assert spec.mode_label

    def test_sae_spec_wires_strategy_and_pipeline(self):
        from app.strategies.sae_steering_strategy import SAESteeringStrategy
        from hambajuba2ba.generation.pipeline import SAESteerablePipeline

        spec = get_backend("sae_steering")
        assert spec.strategy_class is SAESteeringStrategy
        assert spec.pipeline_factory is SAESteerablePipeline

    def test_unknown_mode_lists_registered_backends(self):
        with pytest.raises(ValueError, match="Unknown mode"):
            get_backend("looping")
        # The error must name what IS available
        with pytest.raises(ValueError, match="sae_steering"):
            get_backend("looping")

    def test_duplicate_registration_rejected(self):
        spec = get_backend("sae_steering")
        with pytest.raises(ValueError, match="already registered"):
            register_backend(spec)

    def test_spec_capabilities_mode_mismatch_rejected(self):
        broken = replace(get_backend("sae_steering"), mode="not_sae")
        with pytest.raises(ValueError, match="capabilities.mode"):
            register_backend(broken)


class TestSAECapabilities:
    def test_manifest_shape(self):
        caps = get_backend("sae_steering").capabilities
        assert caps.temporal == "per_frame"
        assert caps.slot_count == 4
        assert caps.feature_id_range == (0, 5119)
        assert caps.feature_label == "Feature"
        assert caps.spatial_mask_shape == (16, 16)
        assert caps.has_prompts
        assert caps.has_destinations

    def test_slot_names_match_default_block_configs(self):
        """Manifest slot ids must agree with the runtime slot defaults."""
        from hambajuba2ba.config.slots import DEFAULT_BLOCK_CONFIGS

        caps = get_backend("sae_steering").capabilities
        assert [s.name for s in caps.slots] == list(DEFAULT_BLOCK_CONFIGS.keys())

    def test_slot_names_match_steered_unet_blocks(self):
        """Manifest slot ids must agree with the blocks the SAE manager wraps."""
        from hambajuba2ba.generation.sae.inline import InlineSAEManager

        caps = get_backend("sae_steering").capabilities
        assert [s.name for s in caps.slots] == list(
            InlineSAEManager.DEFAULT_BLOCK_PATHS.keys()
        )

    def test_control_input_names_unique_and_slot_bound(self):
        caps = get_backend("sae_steering").capabilities
        names = [c.name for c in caps.control_inputs]
        assert len(names) == len(set(names))
        assert "slot.strength" in names
        strength = next(c for c in caps.control_inputs if c.name == "slot.strength")
        assert strength.count == caps.slot_count

    def test_to_dict_is_json_serializable(self):
        caps = get_backend("sae_steering").capabilities
        payload = caps.to_dict()
        round_tripped = json.loads(json.dumps(payload))  # raises if not JSON-safe
        assert round_tripped["mode"] == "sae_steering"
        assert round_tripped["slot_count"] == 4
        assert round_tripped["slots"][0]["name"] == "down.2.1"
        assert round_tripped["slots"][0]["color"].startswith("#")

    def test_to_dict_matches_golden_fixture(self):
        """Cross-language contract lock: the frontend validates this same
        fixture through its TS mirror (frontend/src/types/wire/). A failure
        here means the manifest changed — regenerate deliberately with
        scripts/dev/dump_capabilities.py and update the mirror to match."""
        fixture = json.loads(GOLDEN_FIXTURE.read_text())
        wire = json.loads(  # the wire shape: tuples become arrays
            json.dumps(get_backend("sae_steering").capabilities.to_dict())
        )
        assert wire == fixture


class TestMockBackend:
    """The mock is the second real backend: it must satisfy every contract
    a GPU backend does, on any machine, with no weights."""

    def test_registered_with_pipeline_and_strategy(self):
        from app.mock_backend import MockPipeline, MockStrategy

        spec = get_backend("mock")
        assert spec.pipeline_factory is MockPipeline
        assert spec.strategy_class is MockStrategy

    def test_manifest_shape_differs_from_sae_on_purpose(self):
        """Six slots and a non-SAE feature range, so the frontend's
        capability-driven paths are exercised beyond the shape they grew
        up with."""
        caps = get_backend("mock").capabilities
        assert caps.slot_count == 6
        assert caps.feature_id_range == (0, 999)
        names = [s.name for s in caps.slots]
        assert len(names) == len(set(names))
        strength = next(c for c in caps.control_inputs if c.name == "slot.strength")
        assert strength.count == caps.slot_count

    def test_to_dict_matches_golden_fixture(self):
        fixture = json.loads(MOCK_FIXTURE.read_text())
        wire = json.loads(
            json.dumps(get_backend("mock").capabilities.to_dict())
        )
        assert wire == fixture

    def test_pipeline_boots_without_gpu_or_weights(self):
        from app.mock_backend import MockPipeline

        pipeline = MockPipeline(config=None)
        pipeline.load()  # must not download, compile, or touch a GPU
        assert pipeline.device == "cpu"
        assert pipeline.engine is None  # composition deliberately skipped
        pipeline.cleanup()


class TestMockFrame:
    def test_frame_shape_and_dtype(self):
        import numpy as np

        from app.mock_backend import _SLOT_RGB, render_mock_frame

        frame = render_mock_frame(
            64, 48, _SLOT_RGB,
            activities=np.zeros(6), enabled=np.zeros(6, dtype=bool), t=0.0,
        )
        assert frame.shape == (48, 64, 3)
        assert frame.dtype == np.uint8

    def test_enabled_slot_band_lights_with_activity(self):
        import numpy as np

        from app.mock_backend import _SLOT_RGB, render_mock_frame

        activities = np.zeros(6)
        activities[0] = 1.0
        enabled = np.zeros(6, dtype=bool)
        enabled[0] = True
        # t chosen so the sweep column sits in the last band, not the ones
        # under comparison
        frame = render_mock_frame(60, 60, _SLOT_RGB, activities, enabled, t=3.9)

        band = 60 // 6
        lit = frame[:band].mean()
        disabled = frame[band : 2 * band].mean()
        assert lit > disabled * 3

    def test_sweep_column_moves_with_time(self):
        import numpy as np

        from app.mock_backend import _SLOT_RGB, render_mock_frame

        args = (_SLOT_RGB, np.zeros(6), np.zeros(6, dtype=bool))
        early = render_mock_frame(64, 48, *args, t=0.5)
        late = render_mock_frame(64, 48, *args, t=1.5)
        assert not np.array_equal(early, late)
