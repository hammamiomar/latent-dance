"""Tests for SparseAutoencoder.

These tests validate the SAE encoding/decoding pipeline, ensuring
sparse representations are correct and persistence works properly.
"""

import torch

from hambajuba2ba.generation.sae import SparseAutoencoder


class TestSparseAutoencoderInit:
    """Tests for SAE initialization."""

    def test_init_dimensions(self, mock_sae):
        """SAE should have correct dimensions."""
        assert mock_sae.n_dirs_local == 64
        assert mock_sae.d_model == 32
        assert mock_sae.k == 4

    def test_init_encoder_shape(self, mock_sae):
        """Encoder should have correct weight shape."""
        assert mock_sae.encoder.weight.shape == (64, 32)

    def test_init_decoder_shape(self, mock_sae):
        """Decoder should have correct weight shape."""
        assert mock_sae.decoder.weight.shape == (32, 64)

    def test_decoder_weights_are_unit_norm(self, mock_sae):
        """Decoder columns should have unit norm.

        This is enforced by unit_norm_decoder_() to make feature
        strengths directly interpretable as magnitudes.
        """
        norms = mock_sae.decoder.weight.data.norm(dim=0)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_pre_bias_initialized_to_zero(self, mock_sae):
        """Pre-bias should be initialized to zeros."""
        assert torch.allclose(mock_sae.pre_bias, torch.zeros_like(mock_sae.pre_bias))

    def test_latent_bias_initialized_to_zero(self, mock_sae):
        """Latent bias should be initialized to zeros."""
        assert torch.allclose(mock_sae.latent_bias, torch.zeros_like(mock_sae.latent_bias))


class TestSparseAutoencoderEncode:
    """Tests for SAE encoding."""

    def test_encode_output_shape(self, mock_sae):
        """Encode should produce correct output shape."""
        x = torch.randn(1, 32)
        latents = mock_sae.encode(x)

        assert latents.shape == (1, 64)

    def test_encode_produces_sparse_output(self, mock_sae):
        """Encode should produce at most k non-zero values.

        Top-K selection ensures exactly k features are active
        per input position.
        """
        x = torch.randn(1, 32)
        latents = mock_sae.encode(x)

        non_zero = (latents != 0).sum().item()
        assert non_zero <= mock_sae.k

    def test_encode_output_is_non_negative(self, mock_sae):
        """Encoded values should be non-negative (ReLU)."""
        x = torch.randn(10, 32)
        latents = mock_sae.encode(x)

        assert (latents >= 0).all()

    def test_encode_batch(self, mock_sae):
        """Encode should handle batched inputs."""
        x = torch.randn(8, 32)
        latents = mock_sae.encode(x)

        assert latents.shape == (8, 64)

    def test_encode_topk_shapes(self, mock_sae):
        """encode_topk should return correct shapes."""
        x = torch.randn(4, 32)
        inds, vals = mock_sae.encode_topk(x)

        assert inds.shape == (4, mock_sae.k)
        assert vals.shape == (4, mock_sae.k)

    def test_encode_topk_values_non_negative(self, mock_sae):
        """encode_topk values should be non-negative."""
        x = torch.randn(4, 32)
        inds, vals = mock_sae.encode_topk(x)

        assert (vals >= 0).all()

    def test_encode_topk_indices_valid(self, mock_sae):
        """encode_topk indices should be in valid range."""
        x = torch.randn(4, 32)
        inds, vals = mock_sae.encode_topk(x)

        assert (inds >= 0).all()
        assert (inds < mock_sae.n_dirs_local).all()


class TestSparseAutoencoderDecode:
    """Tests for SAE decoding."""

    def test_decode_output_shape(self, mock_sae):
        """Decode should produce correct output shape."""
        latents = torch.randn(1, 64)
        recons = mock_sae.decode(latents)

        assert recons.shape == (1, 32)

    def test_decode_batch(self, mock_sae):
        """Decode should handle batched inputs."""
        latents = torch.randn(8, 64)
        recons = mock_sae.decode(latents)

        assert recons.shape == (8, 32)

    def test_decode_sparse_matches_dense(self, mock_sae):
        """decode_sparse should match decode for same input.

        Both decoding methods should produce identical results
        for the same sparse representation.
        """
        x = torch.randn(4, 32)
        inds, vals = mock_sae.encode_topk(x)

        # Create dense latents from sparse
        latents = torch.zeros(4, 64)
        for i in range(4):
            latents[i, inds[i]] = vals[i]

        dense_recons = mock_sae.decode(latents)
        sparse_recons = mock_sae.decode_sparse(inds, vals)

        assert torch.allclose(dense_recons, sparse_recons, atol=1e-5)


class TestSparseAutoencoderRoundtrip:
    """Tests for encode-decode roundtrip."""

    def test_forward_shape(self, mock_sae):
        """Forward should maintain input shape."""
        x = torch.randn(1, 32)
        recons = mock_sae(x)

        assert recons.shape == x.shape

    def test_forward_batch(self, mock_sae):
        """Forward should handle batched inputs."""
        x = torch.randn(8, 32)
        recons = mock_sae(x)

        assert recons.shape == x.shape


class TestSparseAutoencoderPersistence:
    """Tests for save/load functionality."""

    def test_save_and_load(self, mock_sae, tmp_path):
        """SAE should save and load correctly."""
        save_path = str(tmp_path / "sae")
        mock_sae.save_to_disk(save_path)

        loaded = SparseAutoencoder.load_from_disk(save_path, device="cpu")

        assert loaded.n_dirs_local == mock_sae.n_dirs_local
        assert loaded.d_model == mock_sae.d_model
        assert loaded.k == mock_sae.k

    def test_loaded_weights_match(self, mock_sae, tmp_path):
        """Loaded weights should match saved weights."""
        save_path = str(tmp_path / "sae")
        mock_sae.save_to_disk(save_path)

        loaded = SparseAutoencoder.load_from_disk(save_path, device="cpu")

        assert torch.allclose(loaded.encoder.weight, mock_sae.encoder.weight)
        assert torch.allclose(loaded.decoder.weight, mock_sae.decoder.weight)

    def test_loaded_produces_same_output(self, mock_sae, tmp_path):
        """Loaded SAE should produce same output as original."""
        save_path = str(tmp_path / "sae")
        mock_sae.save_to_disk(save_path)

        loaded = SparseAutoencoder.load_from_disk(save_path, device="cpu")

        x = torch.randn(4, 32)
        orig_out = mock_sae(x)
        loaded_out = loaded(x)

        assert torch.allclose(orig_out, loaded_out)
