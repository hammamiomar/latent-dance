"""Tests for the unified slot-config message.

UpdateSlotConfig is the one message every backend's slots ride;
UpdateBlockConfig is its legacy alias (old action literal, `block` key
accepted). Both must validate through the ClientMessage discriminated
union and drive the same shared slot handler.
"""

from pydantic import TypeAdapter

from app.schemas import (
    ClientMessage,
    SlotConfigSnapshot,
    UpdateBlockConfig,
    UpdateSlotConfig,
)

adapter = TypeAdapter(ClientMessage)


class FakeStrategy:
    """Minimal StrategyProtocol surface for the slot handler."""

    def __init__(self):
        self.slot_configs = {}
        self.stem_features = {}
        self.stem_classifications = {}
        self._physics = None
        self._spatial = None


class TestUnifiedMessage:
    def test_update_slot_config_validates(self):
        msg = adapter.validate_python(
            {"action": "update_slot_config", "slot": "mid.0", "enabled": True}
        )
        assert isinstance(msg, UpdateSlotConfig)
        assert msg.slot == "mid.0"
        assert msg.enabled is True

    def test_legacy_block_payload_still_validates(self):
        """v2 clients send action=update_block_config with a `block` key."""
        msg = adapter.validate_python(
            {"action": "update_block_config", "block": "mid.0", "feature_id": 42}
        )
        assert isinstance(msg, UpdateBlockConfig)
        assert isinstance(msg, UpdateSlotConfig)  # subclass → one handler
        assert msg.slot == "mid.0"

    def test_slot_key_accepted_on_legacy_action(self):
        msg = adapter.validate_python(
            {"action": "update_block_config", "slot": "up.0.0"}
        )
        assert msg.slot == "up.0.0"


class TestSharedHandler:
    def test_handler_applies_both_message_forms(self):
        from app.strategies.handlers.slot_handlers import (
            handle_slot_message,
            is_slot_message,
        )

        strategy = FakeStrategy()
        legacy = adapter.validate_python(
            {
                "action": "update_block_config",
                "block": "mid.0",
                "feature_id": 7,
                "enabled": True,
                "strength_min": -5.0,
                "strength_max": 12.0,
            }
        )
        unified = adapter.validate_python(
            {"action": "update_slot_config", "slot": "up.0.1", "feature_id": 9}
        )

        assert is_slot_message(legacy)
        assert is_slot_message(unified)
        handle_slot_message(strategy, legacy)
        handle_slot_message(strategy, unified)

        assert strategy.slot_configs["mid.0"].feature_id == 7
        assert strategy.slot_configs["mid.0"].enabled is True
        assert strategy.slot_configs["mid.0"].strength_min == -5.0
        assert strategy.slot_configs["mid.0"].strength_max == 12.0
        assert strategy.slot_configs["up.0.1"].feature_id == 9


class TestSnapshot:
    def test_snapshot_serializes_slot_and_legacy_block(self):
        snap = SlotConfigSnapshot(
            slot="mid.0",
            block="mid.0",
            link_target="vocals",
            strength_min=-30.0,
            strength_max=30.0,
            feature_id=1,
            enabled=True,
            auto_config=True,
            spatial_mode="draw",
            channel="energy_smooth",
            layer="combined",
            physics_preset="ambient",
        )
        payload = snap.model_dump()
        assert payload["slot"] == "mid.0"
        assert payload["block"] == "mid.0"  # legacy duplicate for pre-slot clients


class TestShim:
    def test_focus_config_shim_reexports_slot_contract(self):
        from hambajuba2ba.audio import focus_config
        from hambajuba2ba.config import slots

        assert focus_config.BlockLinkConfig is slots.BlockLinkConfig
        assert focus_config.get_base_stem is slots.get_base_stem
        assert focus_config.DEFAULT_BLOCK_CONFIGS is slots.DEFAULT_BLOCK_CONFIGS
