"""Tests for the backend registry (app/backends.py).

The registry is the Phase 2 seam: main.py, create_strategy, and the
capabilities endpoint all resolve backends through it. These tests pin the
contract a new backend must satisfy, and check the SAE manifest against
independent runtime sources of truth (not against backends.py itself).
"""

import json
from dataclasses import replace

import pytest

from app.backends import get_backend, register_backend


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
