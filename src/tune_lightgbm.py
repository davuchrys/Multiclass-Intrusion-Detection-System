"""LightGBM tuning within the ranges stated in proposal Table 3.7.

Table 3.7 lists learning_rate as "0.05 or 0.1", n_estimators as "100 or 200", and
max_depth as "-1 or tuned value". Phase 6 only ever used (0.05, 200). This module
evaluates the remaining in-proposal combinations on three feature representations,
all under the fixed S2 class-weight scenario, so the tuning itself stays inside the
proposal's stated configuration space.
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
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

from src.ae_variants import VariantAutoencoder, encode_all
from src.data_loading import find_project_root, load_config, resolve_project_path


IN_PROPOSAL_GRID = [
    {"learning_rate": 0.05, "n_estimators": 200},
    {"learning_rate": 0.1, "n_estimators": 200},
    {"learning_rate": 0.1, "n_estimators": 100},
    {"learning_rate": 0.05, "n_estimators": 100},
]


def full_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute accuracy, macro metrics, and weighted F1 for literature comparison."""
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
    }


def load_variant_latents(
    checkpoint_path: Path,
    latent_dim: int,
    latent_activation: str,
    x_train: np.ndarray,
    x_test: np.ndarray,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Re-encode train/test features with a saved variant Autoencoder.

    ReLU and linear latent variants have identical state_dict keys, so a wrong
    `latent_activation` would load silently and produce wrong latents. When the
    checkpoint carries metadata (newer saves), it is verified against the caller;
    older checkpoints rely on the caller passing the training-time activation.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    metadata = checkpoint.get("metadata")
    if metadata is not None:
        if (
            int(metadata["latent_dim"]) != latent_dim
            or metadata["latent_activation"] != latent_activation
        ):
            raise ValueError(
                f"Checkpoint {checkpoint_path.name} was trained with "
                f"latent_dim={metadata['latent_dim']}, "
                f"latent_activation={metadata['latent_activation']}; "
                f"caller requested {latent_dim}/{latent_activation}."
            )
    model = VariantAutoencoder(
        input_dim=x_train.shape[1],
        latent_dim=latent_dim,
        latent_activation=latent_activation,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    return (
        encode_all(model, x_train, batch_size),
        encode_all(model, x_test, batch_size),
    )


def run_tuning(config_path: str | Path = "configs/config.yaml") -> dict[str, Any]:
    """Run the in-proposal LightGBM grid over the candidate feature representations."""
    project_root = find_project_root()
    config = load_config(config_path)
    processed_dir = resolve_project_path(config["paths"]["processed_dir"], project_root)
    metrics_dir = resolve_project_path(config["paths"]["metrics_dir"], project_root)
    models_dir = resolve_project_path(config["paths"]["models_dir"], project_root)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    base_config = dict(config["lightgbm"])
    base_config.pop("prediction_batch_size", None)
    batch_size = int(config["autoencoder"]["batch_size"])

    y_train = np.array(np.load(processed_dir / "y_train.npy", mmap_mode="r"))
    y_test = np.array(np.load(processed_dir / "y_test.npy", mmap_mode="r"))
    sample_weight = np.array(
        np.load(processed_dir / "sample_weight_s2_class_weight.npy", mmap_mode="r")
    )
    x_train = np.load(processed_dir / "X_train.npy", mmap_mode="r")
    x_test = np.load(processed_dir / "X_test.npy", mmap_mode="r")

    print("Re-encoding latent-32 features from the V4 checkpoint...")
    z32_train, z32_test = load_variant_latents(
        models_dir / "ae_v4_linear_32_weighted.pt",
        32,
        "linear",
        x_train,
        x_test,
        batch_size,
    )

    representations: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "latent16_proposal": (
            np.load(processed_dir / "Z_train.npy", mmap_mode="r"),
            np.load(processed_dir / "Z_test.npy", mmap_mode="r"),
        ),
        "latent32_v4": (z32_train, z32_test),
        "original69_diagnostic": (x_train, x_test),
    }

    results: list[dict[str, Any]] = []
    for rep_name, (train_features, test_features) in representations.items():
        for grid_point in IN_PROPOSAL_GRID:
            model_config = {**base_config, **grid_point}
            start = time.perf_counter()
            classifier = LGBMClassifier(**model_config)
            classifier.fit(train_features, y_train, sample_weight=sample_weight)
            y_pred = np.argmax(classifier.booster_.predict(test_features), axis=1)
            seconds = time.perf_counter() - start
            row = {
                "representation": rep_name,
                "scenario": "s2_class_weight",
                **grid_point,
                **full_metrics(y_test, y_pred),
                "seconds": seconds,
            }
            results.append(row)
            print(
                f"[{rep_name}] lr={grid_point['learning_rate']} "
                f"n={grid_point['n_estimators']} -> "
                f"macro_f1={row['macro_f1']:.4f} "
                f"weighted_f1={row['weighted_f1']:.4f} "
                f"acc={row['accuracy']:.4f} ({seconds:.0f}s)"
            )

    output_path = metrics_dir / "lightgbm_tuning_in_proposal.json"
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump({"grid": IN_PROPOSAL_GRID, "results": results}, file, indent=2)
    print(f"Saved: {output_path}")
    return {"results": results, "output_path": str(output_path)}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for in-proposal LightGBM tuning."""
    parser = argparse.ArgumentParser(description="Tune LightGBM within proposal Table 3.7.")
    parser.add_argument("--config", default="configs/config.yaml", help="YAML config path.")
    return parser.parse_args()


def main() -> None:
    """Run the tuning study from the command line."""
    args = parse_args()
    run_tuning(args.config)


if __name__ == "__main__":
    main()
