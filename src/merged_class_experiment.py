"""Corrected 9-class experiment: DoS and DDoS merged into one class.

The dataset stores every DoS flow a second time under the DDoS label with
identical features and identifiers (see dataset_label_conflicts.md), so the
two labels are not learnable as separate classes. Following supervisor
direction, this module reruns the classification stage under a corrected
9-class protocol in which DoS and DDoS form a single class.

The Autoencoder never sees labels, so the Phase 4 latent features are reused
unchanged. Only the label mapping, the class weights, the resampling, the
LightGBM head (num_class=9), and the evaluation change. Results are reported
alongside, never in place of, the 10-class results: macro metrics with
different class counts are not directly comparable.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import confusion_matrix

from src.data_loading import find_project_root, load_config, resolve_project_path
from src.evaluation import macro_metrics, plot_confusion_matrices, save_confusion_csv
from src.imbalance import (
    build_sampling_counts,
    compute_class_weights,
    count_classes,
    create_sample_weights,
    write_resampled_training_data,
)


MERGED_CLASS_NAMES = [
    "Benign",
    "Backdoor",
    "DoS/DDoS",
    "Injection",
    "MITM",
    "Password",
    "Ransomware",
    "Scanning",
    "XSS",
]
MERGED_NUM_CLASSES = 9


def merge_labels(labels: np.ndarray) -> np.ndarray:
    """Map 10-class indices to 9 classes: DoS (3) joins DDoS (2), others shift down."""
    merged = np.asarray(labels, dtype=np.int16).copy()
    merged[merged == 3] = 2
    merged[merged > 3] -= 1
    return merged


def predict_in_batches(
    model: LGBMClassifier,
    features: np.ndarray,
    batch_size: int = 250_000,
) -> np.ndarray:
    """Predict class labels in blocks so memory-mapped inputs stay bounded."""
    predictions = np.empty(features.shape[0], dtype=np.int16)
    for start in range(0, features.shape[0], batch_size):
        end = min(start + batch_size, features.shape[0])
        probabilities = model.booster_.predict(features[start:end])
        predictions[start:end] = np.argmax(probabilities, axis=1).astype(np.int16)
    return predictions


def evaluate_run(
    name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metrics_dir: Path,
    figures_dir: Path,
) -> dict[str, Any]:
    """Compute metrics and persist the confusion matrix for one merged-class run."""
    metrics = macro_metrics(y_true, y_pred)
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(MERGED_NUM_CLASSES)))
    save_confusion_csv(
        matrix,
        MERGED_CLASS_NAMES,
        metrics_dir / f"confusion_matrix_merged9_{name}.csv",
    )
    plot_confusion_matrices(
        matrix,
        MERGED_CLASS_NAMES,
        f"merged9_{name}",
        figures_dir / f"confusion_matrix_merged9_{name}.png",
    )
    return {**metrics, "confusion_matrix": matrix.astype(int).tolist()}


def train_and_evaluate(
    name: str,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    model_config: dict[str, Any],
    metrics_dir: Path,
    figures_dir: Path,
    sample_weight: np.ndarray | None = None,
) -> dict[str, Any]:
    """Train one 9-class LightGBM run and return its evaluation report."""
    observed = np.unique(np.asarray(train_labels))
    if not np.array_equal(observed, np.arange(MERGED_NUM_CLASSES)):
        raise ValueError(f"{name}: training labels must cover all 9 merged classes.")

    start = time.perf_counter()
    model = LGBMClassifier(**model_config)
    model.fit(train_features, train_labels, sample_weight=sample_weight)
    training_seconds = time.perf_counter() - start
    y_pred = predict_in_batches(model, test_features)

    report = {
        "run": name,
        "training_rows": int(np.asarray(train_labels).shape[0]),
        "test_rows": int(test_labels.shape[0]),
        "num_classes": MERGED_NUM_CLASSES,
        "training_seconds": training_seconds,
        **evaluate_run(name, test_labels, y_pred, metrics_dir, figures_dir),
    }
    print(
        f"[{name}] macro_f1={report['macro_f1']:.4f} "
        f"macro_recall={report['macro_recall']:.4f} "
        f"accuracy={report['accuracy']:.4f} ({training_seconds:.0f}s)"
    )
    return report


def run_experiment(
    config_path: str | Path = "configs/config.yaml",
    include_original: bool = True,
) -> dict[str, Any]:
    """Run the corrected 9-class experiment across all four imbalance scenarios."""
    project_root = find_project_root()
    config = load_config(config_path)
    processed_dir = resolve_project_path(config["paths"]["processed_dir"], project_root)
    metrics_dir = resolve_project_path(config["paths"]["metrics_dir"], project_root)
    figures_dir = resolve_project_path(config["paths"]["figures_dir"], project_root)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    random_state = int(config["project"]["random_state"])
    batch_size = int(config["imbalance"].get("batch_size", 250_000))
    model_config = dict(config["lightgbm"])
    model_config.pop("prediction_batch_size", None)
    model_config["num_class"] = MERGED_NUM_CLASSES

    z_train = np.load(processed_dir / "Z_train.npy", mmap_mode="r")
    z_test = np.load(processed_dir / "Z_test.npy", mmap_mode="r")
    y_train9 = merge_labels(np.load(processed_dir / "y_train.npy", mmap_mode="r"))
    y_test9 = merge_labels(np.load(processed_dir / "y_test.npy", mmap_mode="r"))
    counts = count_classes(y_train9, MERGED_NUM_CLASSES, batch_size)
    print("Merged training class counts:", dict(enumerate(counts.tolist())))

    reports: dict[str, dict[str, Any]] = {}

    reports["s1_none"] = train_and_evaluate(
        "s1_none", z_train, y_train9, z_test, y_test9,
        model_config, metrics_dir, figures_dir,
    )

    class_weights = compute_class_weights(counts)
    weight_path = processed_dir / "sample_weight_merged9_s2.npy"
    create_sample_weights(y_train9, class_weights, weight_path, batch_size)
    s2_weights = np.load(weight_path, mmap_mode="r")
    reports["s2_class_weight"] = train_and_evaluate(
        "s2_class_weight", z_train, y_train9, z_test, y_test9,
        model_config, metrics_dir, figures_dir, sample_weight=s2_weights,
    )
    reports["s2_class_weight"]["class_weights"] = {
        MERGED_CLASS_NAMES[index]: weight for index, weight in class_weights.items()
    }

    for scenario, target in [
        ("s3_upsampling", int(counts.max())),
        ("s4_downsampling", int(counts.min())),
    ]:
        output_counts = build_sampling_counts(counts, scenario, target)
        feature_path = processed_dir / f"Z_train_merged9_{scenario}.npy"
        label_path = processed_dir / f"y_train_merged9_{scenario}.npy"
        write_resampled_training_data(
            z_train, y_train9, feature_path, label_path,
            output_counts, random_state, batch_size,
        )
        resampled_features = np.load(feature_path, mmap_mode="r")
        resampled_labels = np.load(label_path, mmap_mode="r")
        reports[scenario] = train_and_evaluate(
            scenario, resampled_features, np.asarray(resampled_labels),
            z_test, y_test9, model_config, metrics_dir, figures_dir,
        )

    if include_original:
        x_train = np.load(processed_dir / "X_train.npy", mmap_mode="r")
        x_test = np.load(processed_dir / "X_test.npy", mmap_mode="r")
        reports["original69_s2"] = train_and_evaluate(
            "original69_s2", x_train, y_train9, x_test, y_test9,
            model_config, metrics_dir, figures_dir, sample_weight=s2_weights,
        )

    summary = pd.DataFrame(
        [
            {
                "run": name,
                "accuracy": report["accuracy"],
                "macro_precision": report["macro_precision"],
                "macro_recall": report["macro_recall"],
                "macro_f1": report["macro_f1"],
                "training_rows": report["training_rows"],
                "training_seconds": report["training_seconds"],
            }
            for name, report in reports.items()
        ]
    )
    summary_path = metrics_dir / "merged9_summary.csv"
    summary.to_csv(summary_path, index=False)

    output = {
        "protocol": "9 classes: DoS and DDoS merged due to contradictory duplicate records",
        "comparability_note": (
            "Macro metrics over 9 classes are not directly comparable with "
            "10-class results or with 10-class numbers from the literature."
        ),
        "merged_class_names": MERGED_CLASS_NAMES,
        "runs": {
            name: {k: v for k, v in report.items() if k != "confusion_matrix"}
            for name, report in reports.items()
        },
        "summary_path": str(summary_path),
    }
    with open(metrics_dir / "merged9_experiment.json", "w", encoding="utf-8") as file:
        json.dump(output, file, indent=2)
    print(f"Saved: {metrics_dir / 'merged9_experiment.json'}")
    return output


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the merged-class experiment."""
    parser = argparse.ArgumentParser(description="Run the corrected 9-class experiment.")
    parser.add_argument("--config", default="configs/config.yaml", help="YAML config path.")
    parser.add_argument(
        "--skip-original",
        action="store_true",
        help="Skip the original-69-feature diagnostic run.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the corrected 9-class experiment from the command line."""
    args = parse_args()
    run_experiment(args.config, include_original=not args.skip_original)


if __name__ == "__main__":
    main()
