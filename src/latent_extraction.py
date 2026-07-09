"""Latent feature extraction helpers."""

import torch


def encode_features(encoder: torch.nn.Module, features: torch.Tensor) -> torch.Tensor:
    """Transform normalized features into latent vectors using a trained encoder."""
    encoder.eval()
    with torch.no_grad():
        return encoder(features)
