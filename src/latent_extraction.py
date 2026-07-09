"""Latent feature extraction from the trained Autoencoder encoder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.lib.format import open_memmap

from src.autoencoder import Autoencoder, batch_tensor, get_device
from src.data_loading import find_project_root, load_config, resolve_project_path


def load_torch_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    """Load a PyTorch checkpoint with compatibility across torch versions."""
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def load_encoder(
    encoder_path: Path,
    input_dim: int,
    latent_dim: int,
    device: torch.device,
) -> torch.nn.Module:
    """Load the trained encoder module saved during Phase 3."""
    if not encoder_path.exists():
        raise FileNotFoundError(f"Encoder checkpoint not found: {encoder_path}")

    checkpoint = load_torch_checkpoint(encoder_path, device)
    metadata = checkpoint.get("metadata", {})
    if metadata.get("input_dim") != input_dim or metadata.get("latent_dim") != latent_dim:
        raise ValueError(
            "Encoder checkpoint metadata does not match the configured Autoencoder dimensions."
        )

    model = Autoencoder(input_dim=input_dim, latent_dim=latent_dim).to(device)
    model.encoder.load_state_dict(checkpoint["encoder_state_dict"])
    model.encoder.eval()
    return model.encoder


def encode_array(
    encoder: torch.nn.Module,
    source: np.ndarray,
    output_path: Path,
    latent_dim: int,
    batch_size: int,
    device: torch.device,
) -> Path:
    """Encode a normalized feature array into latent vectors using batched memmap writes."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    latent = open_memmap(
        output_path,
        mode="w+",
        dtype=np.float32,
        shape=(source.shape[0], latent_dim),
    )

    with torch.no_grad():
        for start in range(0, source.shape[0], batch_size):
            end = min(start + batch_size, source.shape[0])
            inputs = batch_tensor(source[start:end], device)
            encoded = encoder(inputs).detach().cpu().numpy().astype(np.float32)
            latent[start:end] = encoded

    latent.flush()
    return output_path


def summarize_latent_array(path: Path, batch_size: int) -> dict[str, Any]:
    """Summarize a latent feature array without fully materializing it in memory."""
    latent = np.load(path, mmap_mode="r")
    min_value = float("inf")
    max_value = float("-inf")
    finite = True

    for start in range(0, latent.shape[0], batch_size):
        block = latent[start : start + batch_size]
        finite = finite and bool(np.isfinite(block).all())
        min_value = min(min_value, float(block.min()))
        max_value = max(max_value, float(block.max()))

    return {
        "path": str(path),
        "shape": tuple(latent.shape),
        "dtype": str(latent.dtype),
        "finite": finite,
        "min": min_value,
        "max": max_value,
    }


def validate_latent_outputs(
    z_train_path: Path,
    z_test_path: Path,
    x_train: np.ndarray,
    x_test: np.ndarray,
    latent_dim: int,
    batch_size: int,
) -> dict[str, Any]:
    """Validate latent array shape, finite values, and ReLU non-negative range."""
    train_summary = summarize_latent_array(z_train_path, batch_size)
    test_summary = summarize_latent_array(z_test_path, batch_size)

    expected_train_shape = (x_train.shape[0], latent_dim)
    expected_test_shape = (x_test.shape[0], latent_dim)
    if tuple(train_summary["shape"]) != expected_train_shape:
        raise ValueError(
            f"Z_train shape mismatch: expected {expected_train_shape}, "
            f"found {train_summary['shape']}."
        )
    if tuple(test_summary["shape"]) != expected_test_shape:
        raise ValueError(
            f"Z_test shape mismatch: expected {expected_test_shape}, found {test_summary['shape']}."
        )
    if not train_summary["finite"] or not test_summary["finite"]:
        raise ValueError("Latent arrays contain NaN or infinite values.")
    if train_summary["min"] < 0.0 or test_summary["min"] < 0.0:
        raise ValueError("Latent arrays contain negative values despite ReLU latent activation.")

    return {
        "Z_train": train_summary,
        "Z_test": test_summary,
    }


def extract_latent_features(
    config_path: str | Path = "configs/config.yaml",
    batch_size: int | None = None,
    device_name: str = "auto",
) -> dict[str, Any]:
    """Run Phase 4 latent feature extraction for train and test arrays."""
    project_root = find_project_root()
    config = load_config(config_path)
    ae_config = config["autoencoder"]
    processed_dir = resolve_project_path(config["paths"]["processed_dir"], project_root)
    models_dir = resolve_project_path(config["paths"]["models_dir"], project_root)
    metrics_dir = resolve_project_path(config["paths"]["metrics_dir"], project_root)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    input_dim = int(ae_config["input_dim"])
    latent_dim = int(ae_config["latent_dim"])
    effective_batch_size = int(batch_size if batch_size is not None else ae_config["batch_size"])
    device = get_device(device_name)

    x_train_path = processed_dir / "X_train.npy"
    x_test_path = processed_dir / "X_test.npy"
    if not x_train_path.exists() or not x_test_path.exists():
        raise FileNotFoundError("Processed feature arrays were not found. Run Phase 2 first.")

    x_train = np.load(x_train_path, mmap_mode="r")
    x_test = np.load(x_test_path, mmap_mode="r")
    if x_train.shape[1] != input_dim or x_test.shape[1] != input_dim:
        raise ValueError("Processed feature count does not match configured Autoencoder input_dim.")

    encoder = load_encoder(models_dir / "encoder.pt", input_dim, latent_dim, device)
    z_train_path = processed_dir / "Z_train.npy"
    z_test_path = processed_dir / "Z_test.npy"

    encode_array(encoder, x_train, z_train_path, latent_dim, effective_batch_size, device)
    encode_array(encoder, x_test, z_test_path, latent_dim, effective_batch_size, device)
    validation = validate_latent_outputs(
        z_train_path,
        z_test_path,
        x_train,
        x_test,
        latent_dim,
        effective_batch_size,
    )

    report = {
        "input_dim": input_dim,
        "latent_dim": latent_dim,
        "batch_size": effective_batch_size,
        "device": str(device),
        "encoder_path": str(models_dir / "encoder.pt"),
        **validation,
    }
    with open(metrics_dir / "latent_extraction_report.json", "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    return report


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for latent feature extraction."""
    parser = argparse.ArgumentParser(description="Extract Autoencoder latent features.")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to the YAML config file.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override extraction batch size.")
    parser.add_argument("--device", default="auto", help="Extraction device: auto, cpu, or cuda.")
    return parser.parse_args()


def main() -> None:
    """Run Phase 4 latent feature extraction from the command line."""
    args = parse_args()
    report = extract_latent_features(args.config, args.batch_size, args.device)
    print("Phase 4 latent feature extraction completed.")
    for key, value in report.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
