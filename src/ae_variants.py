"""Autoencoder variant study: latent activation and imbalance-aware reconstruction loss.

Each variant changes only the Autoencoder stage. The Phase 2 split, the MinMax scaler,
the LightGBM configuration, and the Phase 5 class weights stay fixed, so any metric
difference is attributable to the learned latent representation.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from lightgbm import LGBMClassifier
from torch import nn

from src.autoencoder import make_validation_mask, set_reproducibility
from src.data_loading import find_project_root, load_config, resolve_project_path
from src.evaluation import macro_metrics


LATENT_ACTIVATIONS = {
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "linear": None,
}


class VariantAutoencoder(nn.Module):
    """Symmetric autoencoder with a configurable latent activation."""

    def __init__(
        self,
        input_dim: int = 69,
        latent_dim: int = 16,
        latent_activation: str = "relu",
    ) -> None:
        super().__init__()
        if latent_activation not in LATENT_ACTIVATIONS:
            raise ValueError(f"Unsupported latent activation: {latent_activation}")

        encoder_layers: list[nn.Module] = [
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim),
        ]
        activation = LATENT_ACTIVATIONS[latent_activation]
        if activation is not None:
            encoder_layers.append(activation())

        self.encoder = nn.Sequential(*encoder_layers)
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
        return self.decoder(self.encoder(x))


def class_weight_lookup(labels: np.ndarray, num_classes: int) -> np.ndarray:
    """Build the Equation 2.8 weight vector w_c = n / (C * n_c).

    The weights already average to 1.0 across the training set, so the gradient
    scale stays comparable with the unweighted baseline.
    """
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=num_classes)
    if np.any(counts == 0):
        raise ValueError("Every class must be present in the training labels.")
    total = int(counts.sum())
    return total / (num_classes * counts.astype(np.float64))


def run_epoch(
    model: VariantAutoencoder,
    features: np.ndarray,
    weights: np.ndarray | None,
    row_indices: np.ndarray,
    batch_size: int,
    optimizer: torch.optim.Optimizer | None,
    rng: np.random.Generator | None,
) -> float:
    """Run one train or validation epoch and return the mean per-value MSE."""
    is_training = optimizer is not None
    model.train(mode=is_training)
    total_loss = 0.0
    total_values = 0

    epoch_indices = rng.permutation(row_indices) if is_training else row_indices
    for start in range(0, epoch_indices.shape[0], batch_size):
        batch_indices = np.sort(epoch_indices[start : start + batch_size])
        inputs = torch.as_tensor(
            np.array(features[batch_indices], dtype=np.float32),
            dtype=torch.float32,
        )
        if is_training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_training):
            outputs = model(inputs)
            squared_error = (outputs - inputs) ** 2
            if weights is None:
                loss = squared_error.mean()
            else:
                batch_weights = torch.as_tensor(
                    weights[batch_indices],
                    dtype=torch.float32,
                ).unsqueeze(1)
                # Weighted mean keeps the loss on the same scale as the unweighted case.
                loss = (squared_error * batch_weights).sum() / (
                    batch_weights.sum() * squared_error.shape[1]
                )
            if is_training:
                loss.backward()
                optimizer.step()

        # Report unweighted MSE so variants stay comparable.
        total_loss += float(squared_error.detach().sum().item())
        total_values += int(inputs.numel())

    return total_loss / total_values


def encode_all(
    model: VariantAutoencoder,
    features: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    """Encode a feature array into latent vectors."""
    model.eval()
    latent_dim = model.encoder[-1].out_features if isinstance(
        model.encoder[-1], nn.Linear
    ) else model.encoder[-2].out_features
    latent = np.empty((features.shape[0], latent_dim), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, features.shape[0], batch_size):
            end = min(start + batch_size, features.shape[0])
            block = torch.as_tensor(
                np.array(features[start:end], dtype=np.float32),
                dtype=torch.float32,
            )
            latent[start:end] = model.encoder(block).numpy()
    return latent


def active_latent_dimensions(latent: np.ndarray, stride: int = 50) -> dict[str, Any]:
    """Report how many latent dimensions carry variation."""
    sample = latent[::stride]
    standard_deviation = sample.std(axis=0)
    return {
        "latent_dim": int(latent.shape[1]),
        "dead_dimensions": int((standard_deviation == 0).sum()),
        "effective_dimensions": int((standard_deviation > 1e-6).sum()),
    }


def train_variant(
    name: str,
    latent_dim: int,
    latent_activation: str,
    class_weighted_loss: bool,
    config: dict[str, Any],
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    sample_weight: np.ndarray,
    epochs: int,
    models_dir: Path,
) -> dict[str, Any]:
    """Train one Autoencoder variant, then evaluate LightGBM on its latent features."""
    ae_config = config["autoencoder"]
    random_state = int(config["project"]["random_state"])
    num_classes = int(config["data"]["expected_num_classes"])
    batch_size = int(ae_config["batch_size"])
    set_reproducibility(random_state)

    weights = None
    if class_weighted_loss:
        lookup = class_weight_lookup(y_train, num_classes)
        weights = lookup[np.asarray(y_train, dtype=np.int64)]

    model = VariantAutoencoder(
        input_dim=int(ae_config["input_dim"]),
        latent_dim=latent_dim,
        latent_activation=latent_activation,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=float(ae_config["learning_rate"]))
    validation_mask = make_validation_mask(
        x_train.shape[0],
        float(ae_config["validation_split"]),
        random_state,
    )
    train_indices = np.flatnonzero(~validation_mask)
    validation_indices = np.flatnonzero(validation_mask)
    epoch_rng = np.random.default_rng(random_state + 1)

    best_val = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    ae_start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        train_loss = run_epoch(
            model, x_train, weights, train_indices, batch_size, optimizer, epoch_rng
        )
        val_loss = run_epoch(
            model, x_train, None, validation_indices, batch_size, None, None
        )
        if val_loss < best_val:
            best_val = val_loss
            best_state = {
                key: value.detach().clone() for key, value in model.state_dict().items()
            }
        if epoch % 10 == 0 or epoch == 1:
            print(
                f"  [{name}] epoch {epoch:02d}/{epochs} "
                f"train_mse={train_loss:.8f} val_mse={val_loss:.8f}"
            )

    if best_state is None:
        raise ValueError(f"No checkpoint was selected for variant {name}.")
    model.load_state_dict(best_state)
    ae_seconds = time.perf_counter() - ae_start
    # ReLU and linear latent variants share identical state_dict keys, so the
    # activation must travel with the checkpoint to prevent silent mis-loading.
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "metadata": {
                "variant": name,
                "latent_dim": latent_dim,
                "latent_activation": latent_activation,
                "class_weighted_reconstruction": class_weighted_loss,
            },
        },
        models_dir / f"ae_{name}.pt",
    )

    z_train = encode_all(model, x_train, batch_size)
    z_test = encode_all(model, x_test, batch_size)
    latent_health = active_latent_dimensions(z_train)

    model_config = dict(config["lightgbm"])
    model_config.pop("prediction_batch_size", None)
    lgbm_start = time.perf_counter()
    classifier = LGBMClassifier(**model_config)
    classifier.fit(z_train, y_train, sample_weight=sample_weight)
    lgbm_seconds = time.perf_counter() - lgbm_start
    y_pred = np.argmax(classifier.booster_.predict(z_test), axis=1)

    metrics = macro_metrics(y_test, y_pred)
    return {
        "variant": name,
        "latent_dim": latent_dim,
        "latent_activation": latent_activation,
        "class_weighted_reconstruction": class_weighted_loss,
        "epochs": epochs,
        **metrics,
        "reconstruction_val_mse": best_val,
        "latent_health": latent_health,
        "autoencoder_seconds": ae_seconds,
        "lightgbm_seconds": lgbm_seconds,
        "y_pred": y_pred,
    }


def run_study(
    config_path: str | Path = "configs/config.yaml",
    epochs: int = 50,
) -> dict[str, Any]:
    """Run the full Autoencoder variant study under the fixed S2 class-weight scenario."""
    project_root = find_project_root()
    config = load_config(config_path)
    processed_dir = resolve_project_path(config["paths"]["processed_dir"], project_root)
    metrics_dir = resolve_project_path(config["paths"]["metrics_dir"], project_root)
    models_dir = resolve_project_path(config["paths"]["models_dir"], project_root)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    x_train = np.load(processed_dir / "X_train.npy", mmap_mode="r")
    y_train = np.array(np.load(processed_dir / "y_train.npy", mmap_mode="r"))
    x_test = np.load(processed_dir / "X_test.npy", mmap_mode="r")
    y_test = np.array(np.load(processed_dir / "y_test.npy", mmap_mode="r"))
    sample_weight = np.array(
        np.load(processed_dir / "sample_weight_s2_class_weight.npy", mmap_mode="r")
    )

    variants = [
        ("v1_relu_16_unweighted", 16, "relu", False),
        ("v2_linear_16_unweighted", 16, "linear", False),
        ("v3_linear_16_weighted", 16, "linear", True),
        ("v4_linear_32_weighted", 32, "linear", True),
    ]

    results: list[dict[str, Any]] = []
    for name, latent_dim, activation, weighted in variants:
        print(f"\n=== {name} ===")
        result = train_variant(
            name,
            latent_dim,
            activation,
            weighted,
            config,
            x_train,
            y_train,
            x_test,
            y_test,
            sample_weight,
            epochs,
            models_dir,
        )
        np.save(processed_dir / f"y_pred_{name}.npy", result.pop("y_pred"))
        print(
            f"  -> macro_f1={result['macro_f1']:.4f} "
            f"macro_recall={result['macro_recall']:.4f} "
            f"accuracy={result['accuracy']:.4f} "
            f"effective_latent={result['latent_health']['effective_dimensions']}"
            f"/{latent_dim}"
        )
        results.append(result)

    output_path = metrics_dir / "ae_variant_study.json"
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump({"scenario": "s2_class_weight", "variants": results}, file, indent=2)
    return {"variants": results, "output_path": str(output_path)}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the Autoencoder variant study."""
    parser = argparse.ArgumentParser(description="Run the Autoencoder variant study.")
    parser.add_argument("--config", default="configs/config.yaml", help="YAML config path.")
    parser.add_argument("--epochs", type=int, default=50, help="Epochs per variant.")
    return parser.parse_args()


def main() -> None:
    """Run the variant study from the command line."""
    args = parse_args()
    run_study(args.config, args.epochs)


if __name__ == "__main__":
    main()
