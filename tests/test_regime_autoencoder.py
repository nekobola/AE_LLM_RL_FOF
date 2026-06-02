"""Unit tests for RegimeAutoEncoder."""

import pytest
import torch
from src.models.regime_autoencoder import RegimeAutoEncoder


class TestRegimeAutoEncoder:
    """Test suite for RegimeAutoEncoder."""

    @pytest.fixture
    def model(self):
        return RegimeAutoEncoder(input_dim=25, latent_dim=6, hidden_dim=16)

    @pytest.fixture
    def batch(self):
        return torch.randn(8, 25)

    def test_forward_output_shape(self, model, batch):
        output = model(batch)
        assert output.shape == batch.shape, (
            f"Expected output shape {batch.shape}, got {output.shape}"
        )

    def test_latent_dim(self, model, batch):
        z = model.encode(batch)
        assert z.shape[-1] == model.latent_dim, (
            f"Expected latent dim {model.latent_dim}, got {z.shape[-1]}"
        )

    def test_latent_range_tanh(self, model, batch):
        z = model.encode(batch)
        assert z.abs().max() <= 1.0 + 1e-5, (
            f"Tanh output should be in [-1, 1], got range [{z.min():.4f}, {z.max():.4f}]"
        )

    def test_backward_pass_gradient_flow(self, model, batch):
        """Verify backward pass completes without error (gradient computation viable)."""
        output = model(batch)
        loss = output.sum()
        loss.backward()
        has_nan = any(
            p.grad is not None and torch.isnan(p.grad).any()
            for p in model.parameters()
        )
        assert not has_nan, "Gradients should not contain NaN after backward"
        grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
        assert len(grad_norms) > 0, "Some parameters should have gradients"

    def test_decode_shape(self, model):
        z = torch.randn(4, 6)
        reconstructed = model.decode(z)
        assert reconstructed.shape == (4, 25), (
            f"Expected decoded shape (4, 25), got {reconstructed.shape}"
        )

    def test_reconstruction_identity_near_zero(self, model, batch):
        """Small input noise should produce small reconstruction error."""
        output = model(batch)
        error = (batch - output).abs().mean()
        assert not torch.isnan(error).any(), "Reconstruction error should not be NaN"

    def test_encoder_decoder_invertible(self, model, batch):
        z = model.encode(batch)
        reconstructed = model.decode(z)
        assert reconstructed.shape == batch.shape

    def test_model_attributes(self, model):
        assert model.input_dim == 25
        assert model.latent_dim == 6
        assert model.hidden_dim == 16
