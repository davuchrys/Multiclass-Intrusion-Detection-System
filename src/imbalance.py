"""Memory-conscious class imbalance handling for latent training features."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
from numpy.lib.format import open_memmap

from src.data_loading import find_project_root, load_config, resolve_project_path


SCENARIO_ALIASES = {
    "s1": "s1_none",
    "s1_none": "s1_none",
    "s2": "s2_class_weight",
    "s2_class_weight": "s2_class_weight",
    "s3": "s3_upsampling",
    "s3_upsampling": "s3_upsampling",
    "s4": "s4_downsampling",
    "s4_downsampling": "s4_downsampling",
}


def class_counts(labels: Iterable[int]) -> dict[int, int]:
    """Return per-class counts as a regular dictionary."""
    if isinstance(labels, np.ndarray):
        unique, counts = np.unique(labels, return_counts=True)
        return {int(label): int(count) for label, count in zip(unique, counts)}
    return dict(Counter(int(label) for label in labels))


def count_classes(
    labels: np.ndarray,
    num_classes: int,
    batch_size: int = 250_000,
) -> np.ndarray:
    """Count integer labels in blocks without copying a full memory-mapped array."""
    counts = np.zeros(num_classes, dtype=np.int64)
    for start in range(0, labels.shape[0], batch_size):
        block = np.asarray(labels[start : start + batch_size], dtype=np.int64)
        if block.size and (block.min() < 0 or block.max() >= num_classes):
            raise ValueError("Training labels contain values outside the configured class range.")
        counts += np.bincount(block, minlength=num_classes)

    if np.any(counts == 0):
        missing = np.flatnonzero(counts == 0).tolist()
        raise ValueError(f"Training labels are missing configured classes: {missing}")
    return counts


def compute_class_weights(counts: np.ndarray) -> dict[int, float]:
    """Compute w_c = n / (C * n_c) for every class."""
    if counts.ndim != 1 or counts.size == 0 or np.any(counts <= 0):
        raise ValueError("Class counts must be a non-empty, positive one-dimensional array.")
    total = int(counts.sum())
    num_classes = int(counts.size)
    return {
        class_index: total / (num_classes * int(class_count))
        for class_index, class_count in enumerate(counts)
    }


def resolve_sampling_target(target: str | int, counts: np.ndarray) -> int:
    """Resolve a named or integer sampling target against current class counts."""
    if isinstance(target, str):
        normalized = target.strip().lower()
        named_targets = {
            "majority": int(counts.max()),
            "minority": int(counts.min()),
            "median": int(np.median(counts)),
        }
        if normalized not in named_targets:
            raise ValueError(
                "Sampling target must be 'majority', 'minority', 'median', or a positive integer."
            )
        return named_targets[normalized]

    resolved = int(target)
    if resolved <= 0:
        raise ValueError("Sampling target must be positive.")
    return resolved


def create_sample_weights(
    labels: np.ndarray,
    class_weights: Mapping[int, float],
    output_path: Path,
    batch_size: int,
) -> Path:
    """Persist per-row training weights without changing features or labels."""
    lookup = np.asarray(
        [class_weights[class_index] for class_index in range(len(class_weights))],
        dtype=np.float64,
    )
    sample_weights = open_memmap(
        output_path,
        mode="w+",
        dtype=np.float64,
        shape=(labels.shape[0],),
    )
    for start in range(0, labels.shape[0], batch_size):
        end = min(start + batch_size, labels.shape[0])
        sample_weights[start:end] = lookup[np.asarray(labels[start:end], dtype=np.int64)]
    sample_weights.flush()
    return output_path


def build_sampling_counts(
    counts: np.ndarray,
    scenario: str,
    target: int,
) -> np.ndarray:
    """Build per-class output counts for random over- or undersampling."""
    if scenario == "s3_upsampling":
        return np.maximum(counts, target)
    if scenario == "s4_downsampling":
        return np.minimum(counts, target)
    raise ValueError(f"Sampling counts are not defined for scenario: {scenario}")


def write_resampled_training_data(
    features: np.ndarray,
    labels: np.ndarray,
    output_feature_path: Path,
    output_label_path: Path,
    output_counts: np.ndarray,
    random_state: int,
    batch_size: int,
) -> tuple[Path, Path]:
    """Write deterministic random resamples directly to disk-backed NumPy arrays."""
    total_rows = int(output_counts.sum())
    output_features = open_memmap(
        output_feature_path,
        mode="w+",
        dtype=features.dtype,
        shape=(total_rows, features.shape[1]),
    )
    output_labels = open_memmap(
        output_label_path,
        mode="w+",
        dtype=labels.dtype,
        shape=(total_rows,),
    )

    rng = np.random.default_rng(random_state)
    output_start = 0
    for class_index, requested_count in enumerate(output_counts):
        source_indices = np.flatnonzero(np.asarray(labels) == class_index)
        source_count = int(source_indices.size)
        target_count = int(requested_count)

        if target_count > source_count:
            extra_indices = rng.choice(
                source_indices,
                size=target_count - source_count,
                replace=True,
            )
            selected_indices = np.concatenate((source_indices, extra_indices))
        elif target_count < source_count:
            selected_indices = rng.choice(source_indices, size=target_count, replace=False)
        else:
            selected_indices = source_indices.copy()

        rng.shuffle(selected_indices)
        for local_start in range(0, target_count, batch_size):
            local_end = min(local_start + batch_size, target_count)
            output_end = output_start + (local_end - local_start)
            batch_indices = np.sort(selected_indices[local_start:local_end])
            output_features[output_start:output_end] = features[batch_indices]
            output_labels[output_start:output_end] = class_index
            output_start = output_end

    output_features.flush()
    output_labels.flush()
    return output_feature_path, output_label_path


def validate_training_arrays(
    feature_path: Path,
    label_path: Path,
    expected_counts: np.ndarray,
    latent_dim: int,
    batch_size: int,
) -> None:
    """Validate shape, labels, and finite latent values for a scenario output."""
    features = np.load(feature_path, mmap_mode="r")
    labels = np.load(label_path, mmap_mode="r")
    expected_rows = int(expected_counts.sum())
    if features.shape != (expected_rows, latent_dim):
        raise ValueError(
            f"Unexpected feature shape in {feature_path}: {features.shape}; "
            f"expected {(expected_rows, latent_dim)}."
        )
    if labels.shape != (expected_rows,):
        raise ValueError(f"Unexpected label shape in {label_path}: {labels.shape}.")

    actual_counts = count_classes(labels, expected_counts.size, batch_size)
    if not np.array_equal(actual_counts, expected_counts):
        raise ValueError(
            f"Resampled class counts do not match the target for {label_path.name}."
        )
    for start in range(0, features.shape[0], batch_size):
        if not np.isfinite(features[start : start + batch_size]).all():
            raise ValueError(f"Non-finite latent values found in {feature_path}.")


def write_distribution_csv(
    output_path: Path,
    class_names: Mapping[int, str],
    before_counts: np.ndarray,
    after_counts: np.ndarray,
    class_weights: Mapping[int, float] | None,
) -> Path:
    """Write before/after training distributions for one scenario."""
    before_total = int(before_counts.sum())
    after_total = int(after_counts.sum())
    with open(output_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "class_index",
                "class_name",
                "before_count",
                "before_percentage",
                "after_count",
                "after_percentage",
                "class_weight",
            ],
        )
        writer.writeheader()
        for class_index in range(before_counts.size):
            writer.writerow(
                {
                    "class_index": class_index,
                    "class_name": class_names[class_index],
                    "before_count": int(before_counts[class_index]),
                    "before_percentage": 100.0 * before_counts[class_index] / before_total,
                    "after_count": int(after_counts[class_index]),
                    "after_percentage": 100.0 * after_counts[class_index] / after_total,
                    "class_weight": (
                        class_weights[class_index] if class_weights is not None else ""
                    ),
                }
            )
    return output_path


def prepare_imbalance_scenario(
    scenario: str,
    config_path: str | Path = "configs/config.yaml",
    force: bool = False,
) -> dict[str, Any]:
    """Prepare one Phase 5 scenario using training artifacts only."""
    canonical_scenario = SCENARIO_ALIASES.get(scenario.lower())
    if canonical_scenario is None:
        raise ValueError(f"Unknown imbalance scenario: {scenario}")

    project_root = find_project_root()
    config = load_config(config_path)
    paths = config["paths"]
    data_config = config["data"]
    imbalance_config = config["imbalance"]
    processed_dir = resolve_project_path(paths["processed_dir"], project_root)
    metrics_dir = resolve_project_path(paths["metrics_dir"], project_root)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    z_train_path = processed_dir / "Z_train.npy"
    y_train_path = processed_dir / "y_train.npy"
    if not z_train_path.exists() or not y_train_path.exists():
        raise FileNotFoundError("Phase 4 training artifacts were not found.")

    features = np.load(z_train_path, mmap_mode="r")
    labels = np.load(y_train_path, mmap_mode="r")
    latent_dim = int(config["autoencoder"]["latent_dim"])
    num_classes = int(data_config["expected_num_classes"])
    batch_size = int(imbalance_config.get("batch_size", 250_000))
    random_state = int(config["project"]["random_state"])
    if features.ndim != 2 or features.shape[1] != latent_dim:
        raise ValueError("Z_train does not match the configured latent dimension.")
    if labels.ndim != 1 or labels.shape[0] != features.shape[0]:
        raise ValueError("Z_train and y_train row counts do not match.")

    before_counts = count_classes(labels, num_classes, batch_size)
    after_counts = before_counts.copy()
    scenario_feature_path = z_train_path
    scenario_label_path = y_train_path
    sample_weight_path: Path | None = None
    class_weights: dict[int, float] | None = None
    sampling_target: int | None = None

    if canonical_scenario == "s2_class_weight":
        class_weights = compute_class_weights(before_counts)
        sample_weight_path = processed_dir / "sample_weight_s2_class_weight.npy"
        if force or not sample_weight_path.exists():
            create_sample_weights(labels, class_weights, sample_weight_path, batch_size)
        stored_weights = np.load(sample_weight_path, mmap_mode="r")
        if stored_weights.shape != labels.shape or not np.isfinite(stored_weights).all():
            raise ValueError("Stored S2 sample weights are invalid.")
        if float(stored_weights.min()) <= 0.0:
            raise ValueError("S2 sample weights must all be positive.")

    if canonical_scenario in {"s3_upsampling", "s4_downsampling"}:
        target_key = (
            "upsampling_target"
            if canonical_scenario == "s3_upsampling"
            else "downsampling_target"
        )
        default_target = "majority" if canonical_scenario == "s3_upsampling" else "minority"
        sampling_target = resolve_sampling_target(
            imbalance_config.get(target_key, default_target),
            before_counts,
        )
        after_counts = build_sampling_counts(
            before_counts,
            canonical_scenario,
            sampling_target,
        )
        scenario_feature_path = processed_dir / f"Z_train_{canonical_scenario}.npy"
        scenario_label_path = processed_dir / f"y_train_{canonical_scenario}.npy"
        if force or not scenario_feature_path.exists() or not scenario_label_path.exists():
            write_resampled_training_data(
                features,
                labels,
                scenario_feature_path,
                scenario_label_path,
                after_counts,
                random_state,
                batch_size,
            )
        validate_training_arrays(
            scenario_feature_path,
            scenario_label_path,
            after_counts,
            latent_dim,
            batch_size,
        )

    class_mapping = data_config["class_mapping"]
    class_names = {int(index): name for name, index in class_mapping.items()}
    distribution_path = metrics_dir / f"imbalance_distribution_{canonical_scenario}.csv"
    write_distribution_csv(
        distribution_path,
        class_names,
        before_counts,
        after_counts,
        class_weights,
    )

    report = {
        "scenario": canonical_scenario,
        "random_state": random_state,
        "latent_dim": latent_dim,
        "input_rows": int(before_counts.sum()),
        "output_rows": int(after_counts.sum()),
        "sampling_target": sampling_target,
        "z_train_path": str(scenario_feature_path),
        "y_train_path": str(scenario_label_path),
        "sample_weight_path": str(sample_weight_path) if sample_weight_path else None,
        "distribution_path": str(distribution_path),
        "before_counts": before_counts.tolist(),
        "after_counts": after_counts.tolist(),
        "class_weights": (
            {str(index): weight for index, weight in class_weights.items()}
            if class_weights is not None
            else None
        ),
        "test_data_modified": False,
    }
    report_path = metrics_dir / f"imbalance_report_{canonical_scenario}.json"
    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
    report["report_path"] = str(report_path)
    return report


def prepare_all_imbalance_scenarios(
    config_path: str | Path = "configs/config.yaml",
    force: bool = False,
) -> dict[str, dict[str, Any]]:
    """Prepare every configured Phase 5 scenario."""
    config = load_config(config_path)
    scenarios = config["imbalance"]["scenarios"]
    return {
        scenario: prepare_imbalance_scenario(scenario, config_path, force)
        for scenario in scenarios
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for Phase 5."""
    parser = argparse.ArgumentParser(description="Prepare class imbalance scenarios.")
    parser.add_argument(
        "--scenario",
        default="all",
        choices=["all", *SCENARIO_ALIASES],
        help="Scenario to prepare.",
    )
    parser.add_argument("--config", default="configs/config.yaml", help="YAML config path.")
    parser.add_argument("--force", action="store_true", help="Regenerate existing artifacts.")
    return parser.parse_args()


def main() -> None:
    """Run Phase 5 from the command line."""
    args = parse_args()
    if args.scenario == "all":
        reports = prepare_all_imbalance_scenarios(args.config, args.force)
    else:
        report = prepare_imbalance_scenario(args.scenario, args.config, args.force)
        reports = {report["scenario"]: report}

    print("Phase 5 class imbalance handling completed.")
    for scenario, report in reports.items():
        print(
            f"{scenario}: {report['input_rows']:,} -> {report['output_rows']:,} training rows"
        )


if __name__ == "__main__":
    main()
