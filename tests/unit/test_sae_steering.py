"""Tests for InlineSAEManager.

Validates block wrapping, steering set/clear, and strength updates for
torch.compile-safe SAE feature steering via SteeredModule wrappers.
"""

import pytest
import torch
import torch.nn as nn

from hambajuba2ba.generation.sae import InlineSAEManager


class SimpleAttention(nn.Module):
    """Minimal attention-like module for testing.

    Returns (hidden_states,) tuple to match diffusers convention.
    """

    def forward(self, hidden_states, *args, **kwargs):
        return (hidden_states,)


def create_mock_unet():
    """Create a mock UNet with real nn.Module structure.

    Provides down_blocks[2].attentions[1] — the block path for "down.2.1".
    """
    unet = nn.Module()
    down_block_2 = nn.Module()
    down_block_2.attentions = nn.ModuleList([SimpleAttention(), SimpleAttention()])
    unet.down_blocks = nn.ModuleList([nn.Module(), nn.Module(), down_block_2])
    return unet


class TestInlineSAEManagerInit:
    """Tests for InlineSAEManager initialization."""

    def test_init_loads_blocks(self, mock_sae_weights):
        """Should wrap attention module with SteeredModule for loaded blocks."""
        unet = create_mock_unet()
        manager = InlineSAEManager(
            unet, str(mock_sae_weights), device="cpu", dtype=torch.float32,
            blocks=["down.2.1"],
        )
        assert "down.2.1" in manager.steered_modules

    def test_init_steering_is_zero(self, mock_sae_weights):
        """Steering strength should be zero after init (no active steering)."""
        unet = create_mock_unet()
        manager = InlineSAEManager(
            unet, str(mock_sae_weights), device="cpu", dtype=torch.float32,
            blocks=["down.2.1"],
        )
        assert manager.steered_modules["down.2.1"].strength.item() == 0.0


class TestSetSteering:
    """Tests for set_steering method."""

    def test_set_steering_sets_strength(self, mock_sae_weights):
        """set_steering should load feature direction and set strength."""
        unet = create_mock_unet()
        manager = InlineSAEManager(
            unet, str(mock_sae_weights), device="cpu", dtype=torch.float32,
            blocks=["down.2.1"],
        )
        manager.set_steering({"down.2.1": (0, 10.0)}, use_mean_scaling=False)
        assert manager.steered_modules["down.2.1"].strength.item() == pytest.approx(10.0)

    def test_set_steering_clears_previous(self, mock_sae_weights):
        """set_steering should clear previous steering then set new values."""
        unet = create_mock_unet()
        manager = InlineSAEManager(
            unet, str(mock_sae_weights), device="cpu", dtype=torch.float32,
            blocks=["down.2.1"],
        )
        manager.set_steering({"down.2.1": (0, 10.0)}, use_mean_scaling=False)
        manager.set_steering({"down.2.1": (1, 5.0)}, use_mean_scaling=False)
        assert manager.steered_modules["down.2.1"].strength.item() == pytest.approx(5.0)

    def test_skip_unknown_block(self, mock_sae_weights):
        """Should silently skip blocks that aren't loaded."""
        unet = create_mock_unet()
        manager = InlineSAEManager(
            unet, str(mock_sae_weights), device="cpu", dtype=torch.float32,
            blocks=["down.2.1"],
        )
        # Try to steer unloaded block — should not raise
        manager.set_steering({"up.0.0": (0, 10.0)})
        # Loaded block should remain at zero
        assert manager.steered_modules["down.2.1"].strength.item() == 0.0


class TestUpdateStrengths:
    """Tests for update_strengths (per-frame fast path)."""

    def test_update_strengths_changes_amplitude(self, mock_sae_weights):
        """update_strengths should change strength without reloading direction."""
        unet = create_mock_unet()
        manager = InlineSAEManager(
            unet, str(mock_sae_weights), device="cpu", dtype=torch.float32,
            blocks=["down.2.1"],
        )
        manager.set_steering({"down.2.1": (0, 10.0)}, use_mean_scaling=False)
        direction_before = manager.steered_modules["down.2.1"].direction.clone()

        manager.update_strengths({"down.2.1": (0, 3.0)}, use_mean_scaling=False)

        assert manager.steered_modules["down.2.1"].strength.item() == pytest.approx(3.0)
        # Direction should be unchanged
        assert torch.equal(manager.steered_modules["down.2.1"].direction, direction_before)


class TestActivationMaps:
    """Tests for fixed-size spatial activation map setup."""

    def test_init_activation_maps_replaces_placeholder_buffer(self, mock_sae_weights):
        """init_activation_maps should resize the constructor placeholder once."""
        unet = create_mock_unet()
        manager = InlineSAEManager(
            unet, str(mock_sae_weights), device="cpu", dtype=torch.float32,
            blocks=["down.2.1"],
        )

        module = manager.steered_modules["down.2.1"]
        assert tuple(module.activation_map.shape) == (1, 1, 1)

        manager.init_activation_maps(latent_h=64, latent_w=64)

        assert tuple(module.activation_map.shape) == (1, 16, 16)
        assert tuple(dict(module.named_buffers())["activation_map"].shape) == (1, 16, 16)

    def test_init_activation_maps_is_idempotent_for_same_shape(self, mock_sae_weights):
        """Same-shape init is safe when setup is replayed before compile."""
        unet = create_mock_unet()
        manager = InlineSAEManager(
            unet, str(mock_sae_weights), device="cpu", dtype=torch.float32,
            blocks=["down.2.1"],
        )

        manager.init_activation_maps(latent_h=64, latent_w=64)
        manager.init_activation_maps(latent_h=64, latent_w=64)

        assert tuple(manager.steered_modules["down.2.1"].activation_map.shape) == (1, 16, 16)

    def test_init_activation_maps_rejects_shape_change(self, mock_sae_weights):
        """Changing the fixed map shape after init would break compiled assumptions."""
        unet = create_mock_unet()
        manager = InlineSAEManager(
            unet, str(mock_sae_weights), device="cpu", dtype=torch.float32,
            blocks=["down.2.1"],
        )

        manager.init_activation_maps(latent_h=64, latent_w=64)

        with pytest.raises(RuntimeError, match="activation_map already initialized"):
            manager.init_activation_maps(latent_h=128, latent_w=128)


class TestClearHooks:
    """Tests for clear_hooks method."""

    def test_clear_hooks_zeros_all(self, mock_sae_weights):
        """clear_hooks should zero steering for all blocks."""
        unet = create_mock_unet()
        manager = InlineSAEManager(
            unet, str(mock_sae_weights), device="cpu", dtype=torch.float32,
            blocks=["down.2.1"],
        )
        manager.set_steering({"down.2.1": (0, 10.0)}, use_mean_scaling=False)
        manager.clear_hooks()
        assert manager.steered_modules["down.2.1"].strength.item() == 0.0

    def test_clear_hooks_idempotent(self, mock_sae_weights):
        """clear_hooks should be safe to call multiple times."""
        unet = create_mock_unet()
        manager = InlineSAEManager(
            unet, str(mock_sae_weights), device="cpu", dtype=torch.float32,
            blocks=["down.2.1"],
        )
        manager.clear_hooks()
        manager.clear_hooks()
        manager.clear_hooks()
        assert manager.steered_modules["down.2.1"].strength.item() == 0.0
