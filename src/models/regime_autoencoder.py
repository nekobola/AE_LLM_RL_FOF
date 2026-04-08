"""Regime autoencoder: encodes market regime features into a compact latent space."""

import torch
import torch.nn as nn


class RegimeAutoEncoder(nn.Module):
    """Autoencoder for compressing market regime features into a low-dimensional latent space.

    Architecture:
        input_dim -> Linear(25, 16) -> LeakyReLU(0.01) -> Linear(16, 6) -> Tanh() -> [Latent 6-dim]
        -> Linear(6, 16) -> LeakyReLU(0.01) -> Linear(16, 25) -> output
    """

    def __init__(self, input_dim: int = 25, latent_dim: int = 6, hidden_dim: int = 16, leakyrelu_negative_slope: float = 0.01):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(leakyrelu_negative_slope),
            nn.Linear(hidden_dim, latent_dim),
            nn.Tanh(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(leakyrelu_negative_slope),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)
