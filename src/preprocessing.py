"""Chunked preprocessing pipeline for CIC-ToN-IoT data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from numpy.lib.format import open_memmap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from src.data_loading import find_project_root, load_config, normalize_attack_label, resolve_project_path


def map_labels(labels: pd.Series, aliases: dict[str, str], class_mapping: dict[str, int]) -> np.ndarray:
    """Normalize raw attack labels and map them to configured numeric class IDs."""
    canonical = labels.map(lambda value: normalize_attack_label(value, aliases))
    mapped = canonical.map(class_mapping)
    return mapped.to_numpy(dtype=np.float64)


def features_to_numeric(frame: pd.DataFrame, feature_columns: list[str]) -> np.ndarray:
    """Convert selected feature columns to a numeric NumPy array and replace infinities with NaN."""
    numeric = frame[feature_columns].apply(pd.to_numeric, errors="coerce")
    values = numeric.to_numpy(dtype=np.float64, copy=True)
    values[~np.isfinite(values)] = np.nan
    return values


def collect_valid_rows_and_labels(
    dataset_path: Path,
    feature_columns: list[str],
    label_column: str,
    aliases: dict[str, str],
    class_mapping: dict[str, int],
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Collect valid original row indices and labels after invalid-row filtering."""
    valid_indices: list[np.ndarray] = []
    valid_labels: list[np.ndarray] = []
    total_rows = 0
    invalid_rows = 0

    use_columns = feature_columns + [label_column]
    reader = pd.read_csv(dataset_path, usecols=use_columns, chunksize=chunk_size, low_memory=False)

    for chunk in reader:
        chunk.columns = chunk.columns.str.strip()
        labels = map_labels(chunk[label_column], aliases, class_mapping)
        features = features_to_numeric(chunk, feature_columns)

        valid_label_mask = ~np.isnan(labels)
        valid_feature_mask = ~np.isnan(features).any(axis=1)
        valid_mask = valid_label_mask & valid_feature_mask

        row_indices = np.arange(total_rows, total_rows + len(chunk), dtype=np.int64)
        valid_indices.append(row_indices[valid_mask])
        valid_labels.append(labels[valid_mask].astype(np.int16))

        invalid_rows += int((~valid_mask).sum())
        total_rows += len(chunk)

    return (
        np.concatenate(valid_indices),
        np.concatenate(valid_labels),
        total_rows,
        invalid_rows,
    )


def build_split_masks(
    valid_indices: np.ndarray,
    valid_labels: np.ndarray,
    total_rows: int,
    test_size: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create deterministic stratified train/test masks over original row positions."""
    train_indices, test_indices, y_train, y_test = train_test_split(
        valid_indices,
        valid_labels,
        test_size=test_size,
        random_state=random_state,
        stratify=valid_labels,
    )

    train_mask = np.zeros(total_rows, dtype=bool)
    test_mask = np.zeros(total_rows, dtype=bool)
    train_mask[train_indices] = True
    test_mask[test_indices] = True
    return train_mask, test_mask, y_train.astype(np.int16), y_test.astype(np.int16)


def compute_train_minmax(
    dataset_path: Path,
    feature_columns: list[str],
    train_mask: np.ndarray,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute feature min/max using training rows only."""
    n_features = len(feature_columns)
    data_min = np.full(n_features, np.inf, dtype=np.float64)
    data_max = np.full(n_features, -np.inf, dtype=np.float64)
    start = 0

    reader = pd.read_csv(dataset_path, usecols=feature_columns, chunksize=chunk_size, low_memory=False)
    for chunk in reader:
        chunk.columns = chunk.columns.str.strip()
        end = start + len(chunk)
        local_train_mask = train_mask[start:end]
        if local_train_mask.any():
            features = features_to_numeric(chunk, feature_columns)[local_train_mask]
            data_min = np.minimum(data_min, np.nanmin(features, axis=0))
            data_max = np.maximum(data_max, np.nanmax(features, axis=0))
        start = end

    if not np.isfinite(data_min).all() or not np.isfinite(data_max).all():
        raise ValueError("Scaler min/max contains non-finite values.")
    return data_min, data_max


def build_minmax_scaler(data_min: np.ndarray, data_max: np.ndarray, feature_columns: list[str]) -> MinMaxScaler:
    """Build a fitted MinMaxScaler from precomputed train-only min/max values."""
    scaler = MinMaxScaler()
    data_range = data_max - data_min
    safe_range = np.where(data_range == 0, 1.0, data_range)
    scaler.data_min_ = data_min
    scaler.data_max_ = data_max
    scaler.data_range_ = data_range
    scaler.scale_ = 1.0 / safe_range
    scaler.min_ = -data_min * scaler.scale_
    scaler.n_features_in_ = len(feature_columns)
    return scaler


def write_processed_arrays(
    dataset_path: Path,
    processed_dir: Path,
    feature_columns: list[str],
    label_column: str,
    aliases: dict[str, str],
    class_mapping: dict[str, int],
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    scaler: MinMaxScaler,
    chunk_size: int,
) -> tuple[Path, Path, Path, Path]:
    """Write normalized train/test arrays as memory-mapped NPY files."""
    train_count = int(train_mask.sum())
    test_count = int(test_mask.sum())
    n_features = len(feature_columns)

    processed_dir.mkdir(parents=True, exist_ok=True)
    x_train_path = processed_dir / "X_train.npy"
    x_test_path = processed_dir / "X_test.npy"
    y_train_path = processed_dir / "y_train.npy"
    y_test_path = processed_dir / "y_test.npy"

    x_train = open_memmap(x_train_path, mode="w+", dtype=np.float32, shape=(train_count, n_features))
    x_test = open_memmap(x_test_path, mode="w+", dtype=np.float32, shape=(test_count, n_features))
    y_train = open_memmap(y_train_path, mode="w+", dtype=np.int16, shape=(train_count,))
    y_test = open_memmap(y_test_path, mode="w+", dtype=np.int16, shape=(test_count,))

    train_cursor = 0
    test_cursor = 0
    start = 0
    use_columns = feature_columns + [label_column]
    reader = pd.read_csv(dataset_path, usecols=use_columns, chunksize=chunk_size, low_memory=False)

    for chunk in reader:
        chunk.columns = chunk.columns.str.strip()
        end = start + len(chunk)
        local_train_mask = train_mask[start:end]
        local_test_mask = test_mask[start:end]

        if local_train_mask.any() or local_test_mask.any():
            features = features_to_numeric(chunk, feature_columns)
            labels = map_labels(chunk[label_column], aliases, class_mapping).astype(np.int16)

            if local_train_mask.any():
                train_features = scaler.transform(features[local_train_mask]).astype(np.float32)
                count = train_features.shape[0]
                x_train[train_cursor : train_cursor + count] = train_features
                y_train[train_cursor : train_cursor + count] = labels[local_train_mask]
                train_cursor += count

            if local_test_mask.any():
                test_features = scaler.transform(features[local_test_mask]).astype(np.float32)
                count = test_features.shape[0]
                x_test[test_cursor : test_cursor + count] = test_features
                y_test[test_cursor : test_cursor + count] = labels[local_test_mask]
                test_cursor += count

        start = end

    x_train.flush()
    x_test.flush()
    y_train.flush()
    y_test.flush()

    if train_cursor != train_count or test_cursor != test_count:
        raise ValueError("Written row counts do not match split masks.")

    return x_train_path, x_test_path, y_train_path, y_test_path


def save_label_encoder(processed_dir: Path, class_mapping: dict[str, int]) -> None:
    """Save label mapping artifacts in JSON and joblib formats."""
    index_to_class = {index: label for label, index in class_mapping.items()}
    payload = {
        "class_to_index": class_mapping,
        "index_to_class": index_to_class,
    }
    with open(processed_dir / "label_mapping.json", "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
    joblib.dump(payload, processed_dir / "label_encoder.joblib")


def write_split_distribution(
    metrics_dir: Path,
    y_train: np.ndarray,
    y_test: np.ndarray,
    class_mapping: dict[str, int],
) -> Path:
    """Write train/test class distributions for stratification checks."""
    rows: list[dict[str, Any]] = []
    index_to_class = {index: label for label, index in class_mapping.items()}
    for split_name, labels in [("train", y_train), ("test", y_test)]:
        counts = pd.Series(labels).value_counts().to_dict()
        total = int(labels.shape[0])
        for index in sorted(index_to_class):
            count = int(counts.get(index, 0))
            rows.append(
                {
                    "split": split_name,
                    "class_index": index,
                    "class": index_to_class[index],
                    "count": count,
                    "percentage": count / total * 100,
                }
            )
    output_path = metrics_dir / "split_class_distribution.csv"
    pd.DataFrame(rows).to_csv(output_path, index=False)
    return output_path


def validate_stratified_distribution(
    y_train: np.ndarray,
    y_test: np.ndarray,
    class_mapping: dict[str, int],
    tolerance_percentage_points: float = 0.05,
) -> float:
    """Assert that train/test class percentages remain close after stratified splitting."""
    train_total = y_train.shape[0]
    test_total = y_test.shape[0]
    max_delta = 0.0

    for class_index in sorted(class_mapping.values()):
        train_count = int((y_train == class_index).sum())
        test_count = int((y_test == class_index).sum())
        train_percentage = train_count / train_total * 100
        test_percentage = test_count / test_total * 100
        delta = abs(train_percentage - test_percentage)
        max_delta = max(max_delta, delta)

        if delta > tolerance_percentage_points:
            raise ValueError(
                "Stratified split class percentage drift exceeded tolerance: "
                f"class_index={class_index}, "
                f"train={train_percentage:.6f}%, "
                f"test={test_percentage:.6f}%, "
                f"delta={delta:.6f} percentage points, "
                f"tolerance={tolerance_percentage_points:.6f}"
            )

    return max_delta


def validate_processed_arrays(
    x_train_path: Path,
    x_test_path: Path,
    y_train_path: Path,
    y_test_path: Path,
    expected_features: int,
    expected_classes: int,
) -> dict[str, Any]:
    """Validate processed arrays without loading everything into memory."""
    x_train = np.load(x_train_path, mmap_mode="r")
    x_test = np.load(x_test_path, mmap_mode="r")
    y_train = np.load(y_train_path, mmap_mode="r")
    y_test = np.load(y_test_path, mmap_mode="r")

    if x_train.shape[1] != expected_features or x_test.shape[1] != expected_features:
        raise ValueError("Processed feature count does not match config.")
    if y_train.ndim != 1 or y_test.ndim != 1:
        raise ValueError("Labels must be one-dimensional arrays.")
    if x_train.shape[0] != y_train.shape[0] or x_test.shape[0] != y_test.shape[0]:
        raise ValueError("Feature/label row counts do not match.")

    train_classes = set(np.unique(y_train).tolist())
    test_classes = set(np.unique(y_test).tolist())
    expected_class_set = set(range(expected_classes))
    if train_classes != expected_class_set or test_classes != expected_class_set:
        raise ValueError("Train/test splits do not contain all expected classes.")

    # Streaming finite check to avoid materializing the arrays.
    for name, array in [("X_train", x_train), ("X_test", x_test)]:
        step = 250_000
        for start in range(0, array.shape[0], step):
            block = array[start : start + step]
            if not np.isfinite(block).all():
                raise ValueError(f"{name} contains NaN or infinite values.")

    return {
        "X_train_shape": tuple(x_train.shape),
        "X_test_shape": tuple(x_test.shape),
        "y_train_shape": tuple(y_train.shape),
        "y_test_shape": tuple(y_test.shape),
    }


def preprocess_dataset(
    config_path: str | Path = "configs/config.yaml",
    chunk_size: int = 100_000,
) -> dict[str, Any]:
    """Run Phase 2 preprocessing from raw CSV to normalized train/test arrays."""
    project_root = find_project_root()
    config = load_config(config_path)
    data_config = config["data"]
    dataset_path = resolve_project_path(config["paths"]["raw_dataset"], project_root)
    processed_dir = resolve_project_path(config["paths"]["processed_dir"], project_root)
    metrics_dir = resolve_project_path(config["paths"]["metrics_dir"], project_root)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    feature_columns = data_config["feature_columns"]
    label_column = data_config["label_column"]
    aliases = data_config["raw_label_aliases"]
    class_mapping = data_config["class_mapping"]
    test_size = float(data_config["test_size"])
    random_state = int(config["project"]["random_state"])

    valid_indices, valid_labels, total_rows, invalid_rows = collect_valid_rows_and_labels(
        dataset_path,
        feature_columns,
        label_column,
        aliases,
        class_mapping,
        chunk_size,
    )
    train_mask, test_mask, y_train_split, y_test_split = build_split_masks(
        valid_indices,
        valid_labels,
        total_rows,
        test_size,
        random_state,
    )
    data_min, data_max = compute_train_minmax(dataset_path, feature_columns, train_mask, chunk_size)
    scaler = build_minmax_scaler(data_min, data_max, feature_columns)
    x_train_path, x_test_path, y_train_path, y_test_path = write_processed_arrays(
        dataset_path,
        processed_dir,
        feature_columns,
        label_column,
        aliases,
        class_mapping,
        train_mask,
        test_mask,
        scaler,
        chunk_size,
    )

    joblib.dump(scaler, processed_dir / "minmax_scaler.joblib")
    save_label_encoder(processed_dir, class_mapping)
    split_distribution_path = write_split_distribution(
        metrics_dir,
        y_train_split,
        y_test_split,
        class_mapping,
    )
    max_split_percentage_delta = validate_stratified_distribution(
        y_train_split,
        y_test_split,
        class_mapping,
    )
    validation = validate_processed_arrays(
        x_train_path,
        x_test_path,
        y_train_path,
        y_test_path,
        data_config["expected_feature_count"],
        data_config["expected_num_classes"],
    )

    report = {
        "total_rows": total_rows,
        "valid_rows": int(valid_indices.shape[0]),
        "invalid_rows_dropped": invalid_rows,
        "train_rows": int(train_mask.sum()),
        "test_rows": int(test_mask.sum()),
        "feature_count": len(feature_columns),
        "test_size": test_size,
        "random_state": random_state,
        "scaler_fit_scope": "train_only",
        "max_split_percentage_delta": max_split_percentage_delta,
        "split_distribution_path": str(split_distribution_path),
        **validation,
    }
    with open(metrics_dir / "preprocessing_report.json", "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    return report


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for Phase 2 preprocessing."""
    parser = argparse.ArgumentParser(description="Preprocess CIC-ToN-IoT for modeling.")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to the YAML config file.")
    parser.add_argument("--chunk-size", type=int, default=100_000, help="Rows per processing chunk.")
    return parser.parse_args()


def main() -> None:
    """Run Phase 2 preprocessing from the command line."""
    args = parse_args()
    report = preprocess_dataset(args.config, args.chunk_size)
    print("Phase 2 preprocessing completed.")
    for key, value in report.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
