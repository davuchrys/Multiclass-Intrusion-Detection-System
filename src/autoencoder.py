"""Autoencoder training for dimensionality reduction."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from src.data_loading import find_project_root, load_config, resolve_project_path


class Autoencoder(nn.Module):
    """Symmetric autoencoder: input -> 64 -> 32 -> latent -> 32 -> 64 -> output."""

    def __init__(self, input_dim: int = 69, latent_dim: int = 16) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruct normalized input features."""
        z = self.encoder(x)
        return self.decoder(z)


def set_reproducibility(seed: int) -> None:
    """Set deterministic seeds for Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(preferred: str = "auto") -> torch.device:
    """Resolve the training device."""
    if preferred == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(preferred)


def make_validation_mask(n_rows: int, validation_split: float, seed: int) -> np.ndarray:
    """Create a deterministic validation mask over X_train rows."""
    if not 0 < validation_split < 1:
        raise ValueError("validation_split must be between 0 and 1.")
    rng = np.random.default_rng(seed)
    validation_count = int(round(n_rows * validation_split))
    validation_indices = rng.choice(n_rows, size=validation_count, replace=False)
    validation_mask = np.zeros(n_rows, dtype=bool)
    validation_mask[validation_indices] = True
    return validation_mask


def batch_tensor(block: np.ndarray, device: torch.device) -> torch.Tensor:
    """Convert a NumPy block into a float tensor on the target device."""
    writable_block = np.array(block, dtype=np.float32, copy=True)
    return torch.as_tensor(writable_block, dtype=torch.float32, device=device)


def run_epoch(
    model: Autoencoder,
    data: np.ndarray,
    validation_mask: np.ndarray,
    batch_size: int,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> float:
    """Run one train or validation epoch and return average MSE."""
    is_training = optimizer is not None
    model.train(mode=is_training)
    total_loss = 0.0
    total_values = 0
    criterion = nn.MSELoss(reduction="sum")

    for start in range(0, data.shape[0], batch_size):
        end = min(start + batch_size, data.shape[0])
        local_validation_mask = validation_mask[start:end]
        local_mask = ~local_validation_mask if is_training else local_validation_mask
        if not local_mask.any():
            continue

        inputs = batch_tensor(data[start:end][local_mask], device)
        if is_training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_training):
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            if is_training:
                loss.backward()
                optimizer.step()

        total_loss += float(loss.detach().cpu().item())
        total_values += int(inputs.numel())

    if total_values == 0:
        raise ValueError("No rows were available for this epoch.")
    return total_loss / total_values


def compute_reconstruction_error(
    model: Autoencoder,
    data: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> float:
    """Compute full-array reconstruction MSE."""
    model.eval()
    criterion = nn.MSELoss(reduction="sum")
    total_loss = 0.0
    total_values = 0

    with torch.no_grad():
        for start in range(0, data.shape[0], batch_size):
            end = min(start + batch_size, data.shape[0])
            inputs = batch_tensor(data[start:end], device)
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            total_loss += float(loss.detach().cpu().item())
            total_values += int(inputs.numel())

    return total_loss / total_values


def save_history_csv(history: list[dict[str, float]], path: Path) -> None:
    """Save training history as CSV."""
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["epoch", "train_loss", "val_loss"])
        writer.writeheader()
        writer.writerows(history)


def save_loss_curve(history: list[dict[str, float]], path: Path) -> None:
    """Save an Autoencoder training/validation loss curve."""
    epochs = [row["epoch"] for row in history]
    train_loss = [row["train_loss"] for row in history]
    val_loss = [row["val_loss"] for row in history]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, label="Train loss")
    plt.plot(epochs, val_loss, label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title("Autoencoder Reconstruction Loss")
    plt.legend()
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close()


def save_models(
    model: Autoencoder,
    config: dict[str, Any],
    models_dir: Path,
) -> tuple[Path, Path]:
    """Save Autoencoder and encoder checkpoints."""
    models_dir.mkdir(parents=True, exist_ok=True)
    autoencoder_path = models_dir / "autoencoder.pt"
    encoder_path = models_dir / "encoder.pt"
    metadata = {
        "input_dim": config["autoencoder"]["input_dim"],
        "latent_dim": config["autoencoder"]["latent_dim"],
        "architecture": (
            f"{config['autoencoder']['input_dim']}-64-32-"
            f"{config['autoencoder']['latent_dim']}-32-64-"
            f"{config['autoencoder']['output_dim']}"
        ),
    }
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "metadata": metadata,
        },
        autoencoder_path,
    )
    torch.save(
        {
            "encoder_state_dict": model.encoder.state_dict(),
            "metadata": metadata,
        },
        encoder_path,
    )
    return autoencoder_path, encoder_path


def train_autoencoder(
    config_path: str | Path = "configs/config.yaml",
    epochs: int | None = None,
    batch_size: int | None = None,
    device_name: str = "auto",
) -> dict[str, Any]:
    """Train the Autoencoder on normalized training features."""
    project_root = find_project_root()
    config = load_config(config_path)
    data_config = config["data"]
    ae_config = config["autoencoder"]
    random_state = int(config["project"]["random_state"])
    set_reproducibility(random_state)

    processed_dir = resolve_project_path(config["paths"]["processed_dir"], project_root)
    models_dir = resolve_project_path(config["paths"]["models_dir"], project_root)
    metrics_dir = resolve_project_path(config["paths"]["metrics_dir"], project_root)
    figures_dir = resolve_project_path(config["paths"]["figures_dir"], project_root)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    x_train_path = processed_dir / "X_train.npy"
    x_test_path = processed_dir / "X_test.npy"
    if not x_train_path.exists() or not x_test_path.exists():
        raise FileNotFoundError("Processed train/test arrays were not found. Run Phase 2 first.")

    x_train = np.load(x_train_path, mmap_mode="r")
    x_test = np.load(x_test_path, mmap_mode="r")
    input_dim = int(ae_config["input_dim"])
    latent_dim = int(ae_config["latent_dim"])
    if x_train.shape[1] != input_dim or x_test.shape[1] != input_dim:
        raise ValueError("Processed feature count does not match Autoencoder input_dim.")
    if input_dim != data_config["expected_feature_count"]:
        raise ValueError("Autoencoder input_dim does not match configured feature count.")

    effective_epochs = int(epochs if epochs is not None else ae_config["epochs"])
    effective_batch_size = int(batch_size if batch_size is not None else ae_config["batch_size"])
    validation_split = float(ae_config["validation_split"])
    device = get_device(device_name)

    model = Autoencoder(input_dim=input_dim, latent_dim=latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(ae_config["learning_rate"]))
    validation_mask = make_validation_mask(x_train.shape[0], validation_split, random_state)

    history: list[dict[str, float]] = []
    for epoch in range(1, effective_epochs + 1):
        train_loss = run_epoch(
            model,
            x_train,
            validation_mask,
            effective_batch_size,
            device,
            optimizer=optimizer,
        )
        val_loss = run_epoch(
            model,
            x_train,
            validation_mask,
            effective_batch_size,
            device,
            optimizer=None,
        )
        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
        history.append(row)
        print(
            f"Epoch {epoch:03d}/{effective_epochs} - "
            f"train_loss={train_loss:.8f} - val_loss={val_loss:.8f}"
        )

    history_path = metrics_dir / "autoencoder_history.csv"
    curve_path = figures_dir / "ae_loss_curve.png"
    save_history_csv(history, history_path)
    save_loss_curve(history, curve_path)
    autoencoder_path, encoder_path = save_models(model, config, models_dir)

    train_reconstruction_error = compute_reconstruction_error(
        model,
        x_train,
        effective_batch_size,
        device,
    )
    test_reconstruction_error = compute_reconstruction_error(
        model,
        x_test,
        effective_batch_size,
        device,
    )
    reconstruction_report = {
        "train_reconstruction_mse": train_reconstruction_error,
        "test_reconstruction_mse": test_reconstruction_error,
        "epochs": effective_epochs,
        "batch_size": effective_batch_size,
        "device": str(device),
        "input_dim": input_dim,
        "latent_dim": latent_dim,
        "autoencoder_path": str(autoencoder_path),
        "encoder_path": str(encoder_path),
        "history_path": str(history_path),
        "loss_curve_path": str(curve_path),
    }
    with open(metrics_dir / "autoencoder_reconstruction_error.json", "w", encoding="utf-8") as file:
        json.dump(reconstruction_report, file, indent=2)

    return reconstruction_report


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for Autoencoder training."""
    parser = argparse.ArgumentParser(description="Train the Autoencoder feature extractor.")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to the YAML config file.")
    parser.add_argument("--epochs", type=int, default=None, help="Override Autoencoder epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override Autoencoder batch size.")
    parser.add_argument("--device", default="auto", help="Training device: auto, cpu, or cuda.")
    return parser.parse_args()


def main() -> None:
    """Run Autoencoder training from the command line."""
    args = parse_args()
    report = train_autoencoder(args.config, args.epochs, args.batch_size, args.device)
    print("Phase 3 Autoencoder training completed.")
    for key, value in report.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
