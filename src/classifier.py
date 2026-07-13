"""LightGBM training and prediction for the four imbalance scenarios."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from lightgbm import LGBMClassifier
from numpy.lib.format import open_memmap

from src.data_loading import find_project_root, load_config, resolve_project_path
from src.imbalance import SCENARIO_ALIASES


def build_lightgbm(config: dict[str, Any]) -> LGBMClassifier:
    """Build a LightGBM multiclass classifier from a config dictionary."""
    return LGBMClassifier(**config)


def train_lightgbm(
    z_train: np.ndarray,
    y_train: np.ndarray,
    scenario: str,
    sample_weight: np.ndarray | None = None,
    model_config: dict[str, Any] | None = None,
) -> LGBMClassifier:
    """Train one fixed-iteration LightGBM model on latent training features."""
    if z_train.ndim != 2:
        raise ValueError("Latent training features must be a two-dimensional array.")
    if y_train.ndim != 1 or y_train.shape[0] != z_train.shape[0]:
        raise ValueError("Training features and labels must contain the same row count.")
    if sample_weight is not None:
        if sample_weight.shape != y_train.shape:
            raise ValueError(f"Sample weights do not match y_train for {scenario}.")
        if not np.isfinite(sample_weight).all() or float(sample_weight.min()) <= 0.0:
            raise ValueError(f"Sample weights are invalid for {scenario}.")

    if model_config is None:
        model_config = dict(load_config("configs/config.yaml")["lightgbm"])
        model_config.pop("prediction_batch_size", None)

    expected_classes = int(model_config["num_class"])
    observed_classes = np.unique(y_train)
    if not np.array_equal(observed_classes, np.arange(expected_classes)):
        raise ValueError(
            f"{scenario} training labels must contain every class from 0 to "
            f"{expected_classes - 1}."
        )

    model = build_lightgbm(model_config)
    model.fit(z_train, y_train, sample_weight=sample_weight)
    return model


def predict(
    model: LGBMClassifier,
    z_test: np.ndarray,
    output_path: Path | None = None,
    batch_size: int = 250_000,
    output_dtype: np.dtype = np.dtype(np.int16),
) -> np.ndarray | Path:
    """Predict test labels in blocks, optionally persisting them as a memmap."""
    if output_path is None:
        predictions = np.empty(z_test.shape[0], dtype=output_dtype)
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        predictions = open_memmap(
            output_path,
            mode="w+",
            dtype=output_dtype,
            shape=(z_test.shape[0],),
        )
    for start in range(0, z_test.shape[0], batch_size):
        end = min(start + batch_size, z_test.shape[0])
        block_probabilities = model.booster_.predict(z_test[start:end])
        block_predictions = np.argmax(block_probabilities, axis=1)
        predictions[start:end] = np.asarray(block_predictions, dtype=output_dtype)
    if isinstance(predictions, np.memmap):
        predictions.flush()
        return output_path
    return predictions


def file_sha256(path: Path, block_size: int = 1024 * 1024) -> str:
    """Return a SHA256 digest without loading the complete file into memory."""
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        while block := file.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def load_imbalance_report(metrics_dir: Path, scenario: str) -> dict[str, Any]:
    """Load and validate the canonical Phase 5 report for one scenario."""
    report_path = metrics_dir / f"imbalance_report_{scenario}.json"
    if not report_path.exists():
        raise FileNotFoundError(
            f"Phase 5 report not found for {scenario}: {report_path}. Run Phase 5 first."
        )
    with open(report_path, encoding="utf-8") as file:
        report = json.load(file)
    if report.get("scenario") != scenario or report.get("test_data_modified") is not False:
        raise ValueError(f"Invalid Phase 5 report for {scenario}.")
    return report


def validate_prediction_file(
    prediction_path: Path,
    expected_rows: int,
    num_classes: int,
) -> list[int]:
    """Validate prediction shape and return the represented class indices."""
    predictions = np.load(prediction_path, mmap_mode="r")
    if predictions.shape != (expected_rows,):
        raise ValueError(
            f"Prediction shape mismatch in {prediction_path}: {predictions.shape}."
        )
    if not np.issubdtype(predictions.dtype, np.integer):
        raise ValueError(f"Predictions in {prediction_path} are not integer labels.")
    predicted_classes = np.unique(predictions).astype(int).tolist()
    if not predicted_classes:
        raise ValueError(f"No predictions were written to {prediction_path}.")
    if predicted_classes[0] < 0 or predicted_classes[-1] >= num_classes:
        raise ValueError(f"Predictions in {prediction_path} contain invalid class indices.")
    return predicted_classes


def source_signature(
    scenario: str,
    model_config: dict[str, Any],
    z_train_path: Path,
    y_train_path: Path,
    sample_weight_path: Path | None,
) -> str:
    """Build a lightweight signature for cached model artifact reuse."""
    source_paths = [z_train_path, y_train_path]
    if sample_weight_path is not None:
        source_paths.append(sample_weight_path)
    payload = {
        "scenario": scenario,
        "model_config": model_config,
        "sources": [
            {
                "path": str(path.resolve()),
                "size": path.stat().st_size,
                "mtime_ns": path.stat().st_mtime_ns,
            }
            for path in source_paths
        ],
    }
    serialized = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def load_reusable_report(
    report_path: Path,
    model_path: Path,
    prediction_path: Path,
    signature: str,
    expected_rows: int,
    num_classes: int,
) -> dict[str, Any] | None:
    """Return a cached report only when all related artifacts remain valid."""
    if not report_path.exists() or not model_path.exists() or not prediction_path.exists():
        return None
    with open(report_path, encoding="utf-8") as file:
        report = json.load(file)
    if report.get("source_signature") != signature:
        return None
    validate_prediction_file(prediction_path, expected_rows, num_classes)
    report["reused_existing_artifacts"] = True
    return report


def train_scenario(
    scenario: str,
    config_path: str | Path = "configs/config.yaml",
    force: bool = False,
) -> dict[str, Any]:
    """Train, save, and predict for one configured imbalance scenario."""
    canonical_scenario = SCENARIO_ALIASES.get(scenario.lower())
    if canonical_scenario is None:
        raise ValueError(f"Unknown imbalance scenario: {scenario}")

    project_root = find_project_root()
    config = load_config(config_path)
    paths = config["paths"]
    processed_dir = resolve_project_path(paths["processed_dir"], project_root)
    metrics_dir = resolve_project_path(paths["metrics_dir"], project_root)
    models_dir = resolve_project_path(paths["models_dir"], project_root)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    phase5_report = load_imbalance_report(metrics_dir, canonical_scenario)
    z_train_path = Path(phase5_report["z_train_path"])
    y_train_path = Path(phase5_report["y_train_path"])
    sample_weight_path = (
        Path(phase5_report["sample_weight_path"])
        if phase5_report["sample_weight_path"] is not None
        else None
    )
    z_test_path = processed_dir / "Z_test.npy"
    y_test_path = processed_dir / "y_test.npy"
    required_paths = [z_train_path, y_train_path, z_test_path, y_test_path]
    if sample_weight_path is not None:
        required_paths.append(sample_weight_path)
    missing_paths = [path for path in required_paths if not path.exists()]
    if missing_paths:
        raise FileNotFoundError(f"Required classifier artifacts are missing: {missing_paths}")

    model_config = dict(config["lightgbm"])
    num_classes = int(model_config["num_class"])
    latent_dim = int(config["autoencoder"]["latent_dim"])
    prediction_batch_size = int(model_config.pop("prediction_batch_size", 250_000))
    model_path = models_dir / f"lgbm_{canonical_scenario}.txt"
    prediction_path = processed_dir / f"y_pred_{canonical_scenario}.npy"
    report_path = metrics_dir / f"classifier_report_{canonical_scenario}.json"
    signature = source_signature(
        canonical_scenario,
        model_config,
        z_train_path,
        y_train_path,
        sample_weight_path,
    )

    z_test = np.load(z_test_path, mmap_mode="r")
    if z_test.ndim != 2 or z_test.shape[1] != latent_dim:
        raise ValueError("Z_test does not match the configured latent dimension.")
    if not force:
        reusable_report = load_reusable_report(
            report_path,
            model_path,
            prediction_path,
            signature,
            z_test.shape[0],
            num_classes,
        )
        if reusable_report is not None:
            return reusable_report

    z_train = np.load(z_train_path, mmap_mode="r")
    y_train = np.load(y_train_path, mmap_mode="r")
    sample_weight = (
        np.load(sample_weight_path, mmap_mode="r") if sample_weight_path is not None else None
    )
    if z_train.shape != (y_train.shape[0], latent_dim):
        raise ValueError(f"Training artifact shape mismatch for {canonical_scenario}.")
    if z_train.shape[0] != int(phase5_report["output_rows"]):
        raise ValueError(f"Training row count differs from Phase 5 for {canonical_scenario}.")

    start_time = time.perf_counter()
    model = train_lightgbm(
        z_train,
        y_train,
        canonical_scenario,
        sample_weight=sample_weight,
        model_config=model_config,
    )
    training_seconds = time.perf_counter() - start_time
    model.booster_.save_model(str(model_path))

    prediction_start = time.perf_counter()
    predict(model, z_test, prediction_path, prediction_batch_size, y_train.dtype)
    prediction_seconds = time.perf_counter() - prediction_start
    predicted_classes = validate_prediction_file(
        prediction_path,
        z_test.shape[0],
        num_classes,
    )

    report = {
        "scenario": canonical_scenario,
        "model_path": str(model_path),
        "prediction_path": str(prediction_path),
        "z_train_path": str(z_train_path),
        "y_train_path": str(y_train_path),
        "sample_weight_path": str(sample_weight_path) if sample_weight_path else None,
        "training_rows": int(z_train.shape[0]),
        "test_rows": int(z_test.shape[0]),
        "latent_dim": latent_dim,
        "num_classes": num_classes,
        "training_classes": np.unique(y_train).astype(int).tolist(),
        "predicted_classes": predicted_classes,
        "all_configured_classes_predicted": predicted_classes == list(range(num_classes)),
        "training_seconds": training_seconds,
        "prediction_seconds": prediction_seconds,
        "model_iterations": int(model.booster_.current_iteration()),
        "model_config": model_config,
        "model_selection": "fixed_n_estimators",
        "internal_validation": False,
        "validation_split_strategy": "not_applicable",
        "source_signature": signature,
        "reused_existing_artifacts": False,
    }
    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
    return report


def train_all_scenarios(
    config_path: str | Path = "configs/config.yaml",
    force: bool = False,
) -> dict[str, dict[str, Any]]:
    """Train all scenarios and verify that Phase 2/4 test artifacts remain unchanged."""
    project_root = find_project_root()
    config = load_config(config_path)
    processed_dir = resolve_project_path(config["paths"]["processed_dir"], project_root)
    metrics_dir = resolve_project_path(config["paths"]["metrics_dir"], project_root)
    test_paths = {
        "Z_test": processed_dir / "Z_test.npy",
        "y_test": processed_dir / "y_test.npy",
    }
    test_hashes_before = {name: file_sha256(path) for name, path in test_paths.items()}

    reports = {
        scenario: train_scenario(scenario, config_path, force)
        for scenario in config["imbalance"]["scenarios"]
    }

    test_hashes_after = {name: file_sha256(path) for name, path in test_paths.items()}
    if test_hashes_after != test_hashes_before:
        raise ValueError("Test artifacts changed during classifier training or prediction.")
    integrity_report = {
        "test_hashes_before": test_hashes_before,
        "test_hashes_after": test_hashes_after,
        "test_artifacts_unchanged": True,
        "scenarios": list(reports),
    }
    with open(metrics_dir / "classifier_test_integrity.json", "w", encoding="utf-8") as file:
        json.dump(integrity_report, file, indent=2)
    return reports


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for Phase 6."""
    parser = argparse.ArgumentParser(description="Train LightGBM scenario classifiers.")
    parser.add_argument(
        "--scenario",
        default="all",
        choices=["all", *SCENARIO_ALIASES],
        help="Scenario to train.",
    )
    parser.add_argument("--config", default="configs/config.yaml", help="YAML config path.")
    parser.add_argument("--force", action="store_true", help="Retrain existing model artifacts.")
    return parser.parse_args()


def main() -> None:
    """Run Phase 6 from the command line."""
    args = parse_args()
    if args.scenario == "all":
        reports = train_all_scenarios(args.config, args.force)
    else:
        report = train_scenario(args.scenario, args.config, args.force)
        reports = {report["scenario"]: report}

    print("Phase 6 LightGBM classification completed.")
    for scenario, report in reports.items():
        print(
            f"{scenario}: {report['training_rows']:,} training rows, "
            f"{report['test_rows']:,} predictions, "
            f"{report['model_iterations']} boosting iterations"
        )


if __name__ == "__main__":
    main()
