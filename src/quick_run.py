"""Isolated stratified quick-run support for pipeline performance checks."""

from __future__ import annotations

import copy
import csv
import json
import logging
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from src.data_loading import find_project_root, load_config, resolve_project_path


LOGGER = logging.getLogger("pipeline")

PHASE1_ARTIFACTS = [
    "class_distribution.csv",
    "feature_summary.csv",
    "dropped_features.csv",
    "class_signal_summary.csv",
    "quasi_constant_review.csv",
    "candidate_features.txt",
    "data_inspection_report.txt",
]


def file_signature(path: Path) -> dict[str, Any]:
    """Return inexpensive immutable-source metadata for cache validation."""
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": stat.st_size,
        "modified_time_ns": stat.st_mtime_ns,
    }


def copy_if_changed(source: Path, destination: Path) -> None:
    """Copy a support artifact only when its size or timestamp changed."""
    if destination.exists():
        source_stat = source.stat()
        destination_stat = destination.stat()
        if (
            source_stat.st_size == destination_stat.st_size
            and source_stat.st_mtime_ns == destination_stat.st_mtime_ns
        ):
            return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def invalidate_quick_downstream_artifacts(
    quick_root: Path,
    processed_dir: Path,
    models_dir: Path,
    metrics_dir: Path,
    figures_dir: Path,
) -> None:
    """Remove stale quick-run outputs while preserving samples and benchmark history."""
    preserved_processed = {
        "X_train.npy",
        "X_test.npy",
        "y_train.npy",
        "y_test.npy",
        "train_source_indices.npy",
        "test_source_indices.npy",
        "minmax_scaler.joblib",
        "label_mapping.json",
    }
    preserved_metrics = {
        *PHASE1_ARTIFACTS,
        "preprocessing_report.json",
        "split_class_distribution.csv",
        "quick_run_history.csv",
    }
    cleanup_rules = (
        (processed_dir, preserved_processed),
        (models_dir, set()),
        (metrics_dir, preserved_metrics),
        (figures_dir, set()),
    )
    resolved_root = quick_root.resolve()
    for directory, preserved_names in cleanup_rules:
        resolved_directory = directory.resolve()
        if not resolved_directory.is_relative_to(resolved_root):
            raise ValueError(f"Refusing to clean a path outside quick_root: {directory}")
        for path in directory.iterdir():
            if path.is_file() and path.name not in preserved_names:
                path.unlink()


def allocate_stratified_counts(
    counts: np.ndarray,
    target_rows: int,
    minimum_per_class: int = 1,
) -> np.ndarray:
    """Allocate an exact sample size proportionally while retaining every class."""
    if counts.ndim != 1 or np.any(counts <= 0):
        raise ValueError("Class counts must be a positive one-dimensional array.")
    if minimum_per_class <= 0:
        raise ValueError("minimum_per_class must be positive.")
    if target_rows > int(counts.sum()):
        raise ValueError("The requested sample exceeds the available rows.")

    minimum_allocations = np.minimum(counts, minimum_per_class)
    if target_rows < int(minimum_allocations.sum()):
        raise ValueError(
            "The quick sample is too small for the configured per-class minimum."
        )
    targets = counts.astype(np.float64) / counts.sum() * target_rows
    allocations = np.maximum(minimum_allocations, np.floor(targets).astype(np.int64))
    allocations = np.minimum(allocations, counts)

    while int(allocations.sum()) < target_rows:
        deficits = targets - allocations
        deficits[allocations >= counts] = -np.inf
        allocations[int(np.argmax(deficits))] += 1

    while int(allocations.sum()) > target_rows:
        excess = allocations - targets
        excess[allocations <= minimum_allocations] = -np.inf
        allocations[int(np.argmax(excess))] -= 1

    if int(allocations.sum()) != target_rows or np.any(allocations > counts):
        raise ValueError("Could not construct a valid stratified sample allocation.")
    return allocations


def stratified_sample_indices(
    labels: np.ndarray,
    target_rows: int,
    random_state: int,
    expected_num_classes: int,
    minimum_per_class: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select deterministic shuffled indices with approximately original proportions."""
    if labels.ndim != 1:
        raise ValueError("Labels must be one-dimensional.")
    effective_rows = min(int(target_rows), labels.shape[0])
    classes, counts = np.unique(labels, return_counts=True)
    expected_classes = np.arange(expected_num_classes, dtype=classes.dtype)
    if not np.array_equal(classes, expected_classes):
        raise ValueError("Source labels do not contain every configured class.")

    allocations = allocate_stratified_counts(counts, effective_rows, minimum_per_class)
    rng = np.random.default_rng(random_state)
    selected: list[np.ndarray] = []
    for class_index, allocation in zip(classes, allocations):
        class_positions = np.flatnonzero(labels == class_index)
        chosen = rng.choice(class_positions, size=int(allocation), replace=False)
        selected.append(chosen)

    indices = np.concatenate(selected).astype(np.int64, copy=False)
    rng.shuffle(indices)
    return indices, counts.astype(np.int64), allocations.astype(np.int64)


def write_sample_array(source: np.ndarray, indices: np.ndarray, destination: Path) -> None:
    """Write selected rows to a standalone NumPy array."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    read_order = np.argsort(indices)
    sampled = np.empty((indices.shape[0], *source.shape[1:]), dtype=source.dtype)
    sampled[read_order] = np.asarray(source[indices[read_order]], dtype=source.dtype)
    np.save(destination, sampled, allow_pickle=False)


def write_split_distribution(
    path: Path,
    class_mapping: dict[str, int],
    train_source_counts: np.ndarray,
    train_sample_counts: np.ndarray,
    test_source_counts: np.ndarray,
    test_sample_counts: np.ndarray,
) -> None:
    """Persist source and quick-sample class distributions for both splits."""
    index_to_class = {int(index): label for label, index in class_mapping.items()}
    rows: list[dict[str, Any]] = []
    for split, source_counts, sample_counts in (
        ("train", train_source_counts, train_sample_counts),
        ("test", test_source_counts, test_sample_counts),
    ):
        for class_index, (source_count, sample_count) in enumerate(
            zip(source_counts, sample_counts)
        ):
            rows.append(
                {
                    "split": split,
                    "class_index": class_index,
                    "class": index_to_class[class_index],
                    "source_count": int(source_count),
                    "sample_count": int(sample_count),
                    "sample_percentage": float(sample_count / sample_counts.sum() * 100.0),
                }
            )

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def validate_quick_arrays(
    processed_dir: Path,
    train_rows: int,
    test_rows: int,
    feature_count: int,
    num_classes: int,
) -> None:
    """Validate sampled arrays before downstream training starts."""
    arrays = {
        "X_train": np.load(processed_dir / "X_train.npy", mmap_mode="r"),
        "X_test": np.load(processed_dir / "X_test.npy", mmap_mode="r"),
        "y_train": np.load(processed_dir / "y_train.npy", mmap_mode="r"),
        "y_test": np.load(processed_dir / "y_test.npy", mmap_mode="r"),
    }
    if arrays["X_train"].shape != (train_rows, feature_count):
        raise ValueError("Quick X_train has an unexpected shape.")
    if arrays["X_test"].shape != (test_rows, feature_count):
        raise ValueError("Quick X_test has an unexpected shape.")
    if arrays["y_train"].shape != (train_rows,) or arrays["y_test"].shape != (test_rows,):
        raise ValueError("Quick label arrays have unexpected shapes.")
    if not np.isfinite(arrays["X_train"]).all() or not np.isfinite(arrays["X_test"]).all():
        raise ValueError("Quick feature arrays contain NaN or infinite values.")
    expected_classes = np.arange(num_classes)
    if not np.array_equal(np.unique(arrays["y_train"]), expected_classes):
        raise ValueError("Quick training labels do not contain every class.")
    if not np.array_equal(np.unique(arrays["y_test"]), expected_classes):
        raise ValueError("Quick test labels do not contain every class.")


def append_quick_run_history(path: Path, report: dict[str, Any]) -> None:
    """Append concise performance measurements for cold and cached quick runs."""
    fields = [
        "completed_at",
        "scenario",
        "sample_regenerated",
        "train_rows",
        "test_rows",
        "autoencoder_epochs",
        "lightgbm_estimators",
        "train_row_reduction_percentage",
        "test_row_reduction_percentage",
        "sample_preparation_seconds",
        "pipeline_seconds",
        "total_seconds",
        "source_artifacts_unchanged",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        if write_header:
            writer.writeheader()
        writer.writerow({field: report[field] for field in fields})


def prepare_quick_artifacts(
    config_path: str | Path = "configs/config.yaml",
    train_rows: int | None = None,
    test_rows: int | None = None,
    autoencoder_epochs: int | None = None,
    lightgbm_estimators: int | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Create or reuse an isolated deterministic quick-run workspace."""
    preparation_started = time.perf_counter()
    project_root = find_project_root()
    config = load_config(config_path)
    quick_defaults = config["pipeline"]["quick_run"]
    requested_train_rows = int(
        quick_defaults["train_rows"] if train_rows is None else train_rows
    )
    requested_test_rows = int(
        quick_defaults["test_rows"] if test_rows is None else test_rows
    )
    effective_epochs = int(
        quick_defaults["autoencoder_epochs"]
        if autoencoder_epochs is None
        else autoencoder_epochs
    )
    effective_estimators = int(
        quick_defaults["lightgbm_estimators"]
        if lightgbm_estimators is None
        else lightgbm_estimators
    )
    minimum_train_per_class = int(quick_defaults["minimum_train_rows_per_class"])
    minimum_test_per_class = int(quick_defaults["minimum_test_rows_per_class"])
    if min(
        requested_train_rows,
        requested_test_rows,
        effective_epochs,
        effective_estimators,
        minimum_train_per_class,
        minimum_test_per_class,
    ) <= 0:
        raise ValueError("Quick-run row counts and model iterations must be positive.")

    full_processed_dir = resolve_project_path(config["paths"]["processed_dir"], project_root)
    full_metrics_dir = resolve_project_path(config["paths"]["metrics_dir"], project_root)
    quick_root = resolve_project_path(quick_defaults["root_dir"], project_root)
    processed_dir = quick_root / "processed"
    models_dir = quick_root / "models"
    metrics_dir = quick_root / "results" / "metrics"
    figures_dir = quick_root / "results" / "figures"
    for directory in (quick_root, processed_dir, models_dir, metrics_dir, figures_dir):
        directory.mkdir(parents=True, exist_ok=True)

    source_paths = {
        "X_train": full_processed_dir / "X_train.npy",
        "X_test": full_processed_dir / "X_test.npy",
        "y_train": full_processed_dir / "y_train.npy",
        "y_test": full_processed_dir / "y_test.npy",
        "minmax_scaler": full_processed_dir / "minmax_scaler.joblib",
        "label_mapping": full_processed_dir / "label_mapping.json",
    }
    missing_sources = [str(path) for path in source_paths.values() if not path.exists()]
    missing_sources.extend(
        str(full_metrics_dir / name)
        for name in PHASE1_ARTIFACTS
        if not (full_metrics_dir / name).exists()
    )
    if missing_sources:
        raise FileNotFoundError(
            "Quick-run source artifacts are incomplete. Run Phases 1 and 2 first: "
            f"{missing_sources}"
        )

    source_arrays = {
        name: np.load(source_paths[name], mmap_mode="r")
        for name in ("X_train", "X_test", "y_train", "y_test")
    }
    effective_train_rows = min(requested_train_rows, source_arrays["y_train"].shape[0])
    effective_test_rows = min(requested_test_rows, source_arrays["y_test"].shape[0])
    num_classes = int(config["data"]["expected_num_classes"])
    feature_count = int(config["data"]["expected_feature_count"])
    random_state = int(config["project"]["random_state"])
    if effective_train_rows < num_classes or effective_test_rows < num_classes:
        raise ValueError("Quick train and test samples must each include at least 10 rows.")
    if source_arrays["X_train"].shape != (
        source_arrays["y_train"].shape[0],
        feature_count,
    ):
        raise ValueError("Full X_train and y_train artifacts have incompatible shapes.")
    if source_arrays["X_test"].shape != (
        source_arrays["y_test"].shape[0],
        feature_count,
    ):
        raise ValueError("Full X_test and y_test artifacts have incompatible shapes.")

    quick_config = copy.deepcopy(config)
    quick_config["paths"].update(
        {
            "interim_dir": str(quick_root.resolve()),
            "processed_dir": str(processed_dir.resolve()),
            "models_dir": str(models_dir.resolve()),
            "metrics_dir": str(metrics_dir.resolve()),
            "figures_dir": str(figures_dir.resolve()),
        }
    )
    quick_config["autoencoder"]["epochs"] = effective_epochs
    quick_config["lightgbm"]["n_estimators"] = effective_estimators
    quick_config["pipeline"]["quick_run"].update(
        {
            "active": True,
            "effective_train_rows": effective_train_rows,
            "effective_test_rows": effective_test_rows,
            "scientific_use": "smoke_test_only_not_final_results",
        }
    )
    quick_config_path = quick_root / "config.yaml"
    manifest_path = quick_root / "quick_run_manifest.json"
    source_signatures = {name: file_signature(path) for name, path in source_paths.items()}
    signature = {
        "source_artifacts": source_signatures,
        "train_rows": effective_train_rows,
        "test_rows": effective_test_rows,
        "random_state": random_state,
        "feature_count": feature_count,
        "num_classes": num_classes,
        "minimum_train_rows_per_class": minimum_train_per_class,
        "minimum_test_rows_per_class": minimum_test_per_class,
        "class_mapping": quick_config["data"]["class_mapping"],
        "autoencoder": quick_config["autoencoder"],
        "lightgbm": quick_config["lightgbm"],
        "imbalance": quick_config["imbalance"],
    }
    required_quick_paths = [
        quick_config_path,
        processed_dir / "X_train.npy",
        processed_dir / "X_test.npy",
        processed_dir / "y_train.npy",
        processed_dir / "y_test.npy",
        processed_dir / "train_source_indices.npy",
        processed_dir / "test_source_indices.npy",
        processed_dir / "minmax_scaler.joblib",
        processed_dir / "label_mapping.json",
        metrics_dir / "preprocessing_report.json",
        metrics_dir / "split_class_distribution.csv",
        *[metrics_dir / name for name in PHASE1_ARTIFACTS],
    ]
    reusable = False
    if not force and manifest_path.exists() and all(path.exists() for path in required_quick_paths):
        with open(manifest_path, "r", encoding="utf-8") as file:
            reusable = json.load(file).get("signature") == signature

    if not reusable:
        invalidate_quick_downstream_artifacts(
            quick_root,
            processed_dir,
            models_dir,
            metrics_dir,
            figures_dir,
        )
        train_indices, train_source_counts, train_sample_counts = stratified_sample_indices(
            source_arrays["y_train"],
            effective_train_rows,
            random_state,
            num_classes,
            minimum_train_per_class,
        )
        test_indices, test_source_counts, test_sample_counts = stratified_sample_indices(
            source_arrays["y_test"],
            effective_test_rows,
            random_state + 1,
            num_classes,
            minimum_test_per_class,
        )
        write_sample_array(source_arrays["X_train"], train_indices, processed_dir / "X_train.npy")
        write_sample_array(source_arrays["X_test"], test_indices, processed_dir / "X_test.npy")
        write_sample_array(source_arrays["y_train"], train_indices, processed_dir / "y_train.npy")
        write_sample_array(source_arrays["y_test"], test_indices, processed_dir / "y_test.npy")
        np.save(processed_dir / "train_source_indices.npy", train_indices, allow_pickle=False)
        np.save(processed_dir / "test_source_indices.npy", test_indices, allow_pickle=False)
        copy_if_changed(
            source_paths["minmax_scaler"],
            processed_dir / "minmax_scaler.joblib",
        )
        copy_if_changed(
            source_paths["label_mapping"],
            processed_dir / "label_mapping.json",
        )
        for name in PHASE1_ARTIFACTS:
            copy_if_changed(full_metrics_dir / name, metrics_dir / name)

        write_split_distribution(
            metrics_dir / "split_class_distribution.csv",
            config["data"]["class_mapping"],
            train_source_counts,
            train_sample_counts,
            test_source_counts,
            test_sample_counts,
        )
        preprocessing_report = {
            "quick_run": True,
            "sampling_strategy": (
                "deterministic_stratified_without_replacement_with_class_minimum"
            ),
            "scientific_use": "smoke_test_only_not_final_results",
            "train_rows": effective_train_rows,
            "test_rows": effective_test_rows,
            "feature_count": feature_count,
            "random_state": random_state,
            "minimum_train_rows_per_class": minimum_train_per_class,
            "minimum_test_rows_per_class": minimum_test_per_class,
            "scaler_fit_scope": "reused_from_full_training_pipeline",
            "source_processed_dir": str(full_processed_dir),
            "split_distribution_path": str(metrics_dir / "split_class_distribution.csv"),
        }
        with open(metrics_dir / "preprocessing_report.json", "w", encoding="utf-8") as file:
            json.dump(preprocessing_report, file, indent=2)
        with open(quick_config_path, "w", encoding="utf-8") as file:
            yaml.safe_dump(quick_config, file, sort_keys=False)
        with open(manifest_path, "w", encoding="utf-8") as file:
            json.dump(
                {
                    "signature": signature,
                    "train_source_counts": train_source_counts.tolist(),
                    "train_sample_counts": train_sample_counts.tolist(),
                    "test_source_counts": test_source_counts.tolist(),
                    "test_sample_counts": test_sample_counts.tolist(),
                },
                file,
                indent=2,
            )

    validate_quick_arrays(
        processed_dir,
        effective_train_rows,
        effective_test_rows,
        feature_count,
        num_classes,
    )
    return {
        "quick_config_path": str(quick_config_path),
        "quick_root": str(quick_root),
        "processed_dir": str(processed_dir),
        "metrics_dir": str(metrics_dir),
        "train_rows": effective_train_rows,
        "test_rows": effective_test_rows,
        "source_train_rows": int(source_arrays["y_train"].shape[0]),
        "source_test_rows": int(source_arrays["y_test"].shape[0]),
        "autoencoder_epochs": effective_epochs,
        "lightgbm_estimators": effective_estimators,
        "regenerated": not reusable,
        "preparation_seconds": time.perf_counter() - preparation_started,
        "source_signatures": source_signatures,
    }


def run_quick_pipeline(
    config_path: str | Path = "configs/config.yaml",
    scenario: str = "all",
    train_rows: int | None = None,
    test_rows: int | None = None,
    autoencoder_epochs: int | None = None,
    lightgbm_estimators: int | None = None,
    device: str = "auto",
    force: bool = False,
) -> dict[str, Any]:
    """Prepare an isolated sample and run Phases 3 through 7 as a smoke test."""
    started = time.perf_counter()
    preparation = prepare_quick_artifacts(
        config_path=config_path,
        train_rows=train_rows,
        test_rows=test_rows,
        autoencoder_epochs=autoencoder_epochs,
        lightgbm_estimators=lightgbm_estimators,
        force=force,
    )
    LOGGER.info(
        "Quick-run sample ready: %d train rows and %d test rows (%s).",
        preparation["train_rows"],
        preparation["test_rows"],
        "regenerated" if preparation["regenerated"] else "reused",
    )

    from run_pipeline import run_pipeline

    pipeline_report = run_pipeline(
        config_path=preparation["quick_config_path"],
        scenario=scenario,
        skip_preprocessing=True,
        force_phases={3} if preparation["regenerated"] or force else set(),
        device=device,
    )
    source_signatures_after = {
        name: file_signature(Path(details["path"]))
        for name, details in preparation["source_signatures"].items()
    }
    if source_signatures_after != preparation["source_signatures"]:
        raise ValueError("Full-data source artifacts changed during the isolated quick run.")

    total_seconds = time.perf_counter() - started
    report = {
        "mode": "isolated_stratified_quick_run",
        "scientific_use": "smoke_test_only_not_final_results",
        "completed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "scenario": scenario,
        "train_rows": preparation["train_rows"],
        "test_rows": preparation["test_rows"],
        "source_train_rows": preparation["source_train_rows"],
        "source_test_rows": preparation["source_test_rows"],
        "train_row_reduction_percentage": float(
            (1.0 - preparation["train_rows"] / preparation["source_train_rows"]) * 100.0
        ),
        "test_row_reduction_percentage": float(
            (1.0 - preparation["test_rows"] / preparation["source_test_rows"]) * 100.0
        ),
        "autoencoder_epochs": preparation["autoencoder_epochs"],
        "lightgbm_estimators": preparation["lightgbm_estimators"],
        "sample_regenerated": preparation["regenerated"],
        "sample_preparation_seconds": preparation["preparation_seconds"],
        "pipeline_seconds": pipeline_report["total_duration_seconds"],
        "total_seconds": total_seconds,
        "source_artifacts_unchanged": True,
        "pipeline_report_path": pipeline_report["report_path"],
        "quick_config_path": preparation["quick_config_path"],
    }
    report_path = Path(preparation["metrics_dir"]) / "quick_run_report.json"
    history_path = Path(preparation["metrics_dir"]) / "quick_run_history.csv"
    report["report_path"] = str(report_path)
    report["history_path"] = str(history_path)
    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
    append_quick_run_history(history_path, report)
    LOGGER.info("Quick run completed in %.2f seconds. Report: %s", total_seconds, report_path)
    return report
