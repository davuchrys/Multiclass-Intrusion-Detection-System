"""Dataset loading and Phase 1 inspection utilities."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


def find_project_root(start: Path | None = None) -> Path:
    """Find the repository root from a script or notebook working directory."""
    start = (start or Path.cwd()).resolve()
    for candidate in [start, *start.parents]:
        if (candidate / "configs" / "config.yaml").exists() and (candidate / "data").exists():
            return candidate
    raise FileNotFoundError("Could not locate the project root.")


def load_config(config_path: str | Path = "configs/config.yaml") -> dict[str, Any]:
    """Load the project YAML configuration."""
    path = Path(config_path)
    if not path.is_absolute():
        path = find_project_root() / path
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def resolve_project_path(path: str | Path, project_root: Path | None = None) -> Path:
    """Resolve a path relative to the project root."""
    resolved = Path(path)
    if resolved.is_absolute():
        return resolved
    return (project_root or find_project_root()) / resolved


def load_csv(path: str | Path, **kwargs: Any) -> pd.DataFrame:
    """Load a CSV dataset from disk."""
    return pd.read_csv(path, **kwargs)


def normalize_attack_label(value: object, aliases: dict[str, str]) -> str:
    """Normalize raw attack labels to canonical proposal label names."""
    key = str(value).strip().lower()
    return aliases.get(key, str(value).strip())


def read_csv_sample(path: Path, sample_size: int) -> tuple[pd.DataFrame, str]:
    """Read a bounded CSV sample while trying common encodings."""
    encodings = ["utf-8", "latin1", "ISO-8859-1"]
    last_error: Exception | None = None
    for encoding in encodings:
        try:
            sample = pd.read_csv(path, nrows=sample_size, encoding=encoding, low_memory=False)
            sample.columns = sample.columns.str.strip()
            return sample, encoding
        except Exception as exc:  # pragma: no cover - diagnostic fallback
            last_error = exc
    raise RuntimeError(f"Failed to read CSV sample. Last error: {last_error}")


def numeric_convert_ratio(series: pd.Series) -> float:
    """Estimate how much of a series can safely be interpreted as numeric."""
    non_null = series.dropna()
    if len(non_null) == 0:
        return 0.0
    if pd.api.types.is_numeric_dtype(non_null):
        return 1.0
    converted = pd.to_numeric(non_null, errors="coerce")
    return float(converted.notna().mean())


def build_sample_feature_profile(sample_df: pd.DataFrame, label_column: str) -> pd.DataFrame:
    """Build sample-level dtype, uniqueness, and quasi-constant statistics."""
    rows: list[dict[str, Any]] = []
    for column in [col for col in sample_df.columns if col != label_column]:
        series = sample_df[column]
        value_counts = series.value_counts(dropna=False, normalize=True)
        top_value_frequency = float(value_counts.iloc[0]) if len(value_counts) else 1.0
        rows.append(
            {
                "feature": column,
                "sample_dtype": str(series.dtype),
                "sample_unique_count": int(series.nunique(dropna=False)),
                "sample_unique_ratio": float(series.nunique(dropna=False) / max(len(series), 1)),
                "sample_top_value_frequency": top_value_frequency,
                "numeric_convert_ratio": numeric_convert_ratio(series),
            }
        )
    profile = pd.DataFrame(rows)
    profile["is_numeric_candidate"] = profile["numeric_convert_ratio"] >= 0.95
    return profile


def profile_full_dataset(
    path: Path,
    encoding: str,
    label_column: str,
    feature_columns: list[str],
    numeric_columns: list[str],
    aliases: dict[str, str],
    chunk_size: int,
) -> tuple[int, Counter[str], Counter[str], Counter[str]]:
    """Profile row count, class distribution, missing values, and infinite values in chunks."""
    total_rows = 0
    class_counts: Counter[str] = Counter()
    missing_counts: Counter[str] = Counter()
    infinite_counts: Counter[str] = Counter()

    reader = pd.read_csv(path, chunksize=chunk_size, encoding=encoding, low_memory=False)
    for chunk in reader:
        chunk.columns = chunk.columns.str.strip()
        total_rows += len(chunk)

        labels = chunk[label_column].map(lambda value: normalize_attack_label(value, aliases))
        class_counts.update(labels.value_counts(dropna=False).to_dict())

        missing_counts.update(chunk[feature_columns].isna().sum().to_dict())

        for column in numeric_columns:
            numeric_series = pd.to_numeric(chunk[column], errors="coerce")
            infinite_counts[column] += int(np.isinf(numeric_series).sum())

    return total_rows, class_counts, missing_counts, infinite_counts


def build_feature_summary(
    sample_profile: pd.DataFrame,
    total_rows: int,
    missing_counts: Counter[str],
    infinite_counts: Counter[str],
    feature_columns: list[str],
    drop_columns: list[str],
    identifier_columns: list[str],
    quasi_constant_threshold: float,
) -> pd.DataFrame:
    """Combine sample and full-dataset statistics into a Phase 1 feature summary."""
    summary = sample_profile.copy()
    summary["missing_count"] = summary["feature"].map(lambda value: missing_counts.get(value, 0))
    summary["missing_percent"] = summary["missing_count"] / total_rows * 100
    summary["inf_count"] = summary["feature"].map(lambda value: infinite_counts.get(value, 0))
    summary["inf_percent"] = summary["inf_count"] / total_rows * 100
    summary["is_identifier"] = summary["feature"].isin(identifier_columns)
    summary["is_quasi_constant"] = summary["sample_top_value_frequency"] >= quasi_constant_threshold
    summary["configured_drop"] = summary["feature"].isin(drop_columns)
    summary["configured_model_feature"] = summary["feature"].isin(feature_columns)

    def reasons(row: pd.Series) -> str:
        values: list[str] = []
        if row["is_identifier"]:
            values.append("identifier or metadata column")
        if not row["is_numeric_candidate"]:
            values.append("non-numeric or not safely convertible to numeric")
        if row["is_quasi_constant"] or (row["configured_drop"] and not row["is_identifier"]):
            values.append("quasi-constant or uninformative feature")
        if row["missing_percent"] > 0:
            values.append("contains missing values")
        if row["inf_percent"] > 0:
            values.append("contains infinite values")
        return "; ".join(dict.fromkeys(values))

    summary["recommended_drop_reason"] = summary.apply(reasons, axis=1)
    summary["recommended_for_modeling"] = summary["configured_model_feature"]
    return summary.sort_values(
        by=["recommended_for_modeling", "feature"],
        ascending=[False, True],
    )


def inspect_dataset(
    config_path: str | Path = "configs/config.yaml",
    sample_size: int = 100_000,
    chunk_size: int = 100_000,
) -> dict[str, Any]:
    """Run Phase 1 dataset inspection and write reproducible output files."""
    project_root = find_project_root()
    config = load_config(config_path)
    data_config = config["data"]
    preprocessing_config = config["preprocessing"]

    dataset_path = resolve_project_path(config["paths"]["raw_dataset"], project_root)
    metrics_dir = resolve_project_path(config["paths"]["metrics_dir"], project_root)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    label_column = data_config["label_column"]
    feature_columns = data_config["feature_columns"]
    drop_columns = data_config["drop_columns"]
    identifier_columns = data_config["identifier_columns"]
    aliases = data_config["raw_label_aliases"]
    class_mapping = data_config["class_mapping"]
    quasi_constant_threshold = float(preprocessing_config["quasi_constant_threshold"])

    sample_df, encoding = read_csv_sample(dataset_path, sample_size)
    header_columns = sample_df.columns.tolist()
    expected_columns = set(feature_columns) | set(drop_columns) | {label_column}

    missing_expected = sorted(expected_columns - set(header_columns))
    unmapped_columns = sorted(set(header_columns) - expected_columns)
    feature_drop_overlap = sorted(set(feature_columns) & set(drop_columns))
    if missing_expected or unmapped_columns or feature_drop_overlap:
        raise ValueError(
            "Dataset column mapping does not match config.yaml: "
            f"missing_expected={missing_expected}, "
            f"unmapped_columns={unmapped_columns}, "
            f"feature_drop_overlap={feature_drop_overlap}"
        )

    sample_profile = build_sample_feature_profile(sample_df, label_column)
    numeric_columns = sample_profile.loc[sample_profile["is_numeric_candidate"], "feature"].tolist()
    total_rows, class_counts, missing_counts, infinite_counts = profile_full_dataset(
        dataset_path,
        encoding,
        label_column,
        [col for col in header_columns if col != label_column],
        numeric_columns,
        aliases,
        chunk_size,
    )

    class_distribution = pd.DataFrame(
        [{"class": name, "count": count} for name, count in class_counts.items()]
    )
    class_distribution["percentage"] = (
        class_distribution["count"] / class_distribution["count"].sum() * 100
    )
    class_distribution["class_index"] = class_distribution["class"].map(class_mapping)
    class_distribution = class_distribution.sort_values("class_index").reset_index(drop=True)

    feature_summary = build_feature_summary(
        sample_profile,
        total_rows,
        missing_counts,
        infinite_counts,
        feature_columns,
        drop_columns,
        identifier_columns,
        quasi_constant_threshold,
    )
    dropped_features = feature_summary.loc[
        feature_summary["feature"].isin(drop_columns),
        ["feature", "recommended_drop_reason"],
    ].sort_values("feature")

    class_distribution.to_csv(metrics_dir / "class_distribution.csv", index=False)
    feature_summary.to_csv(metrics_dir / "feature_summary.csv", index=False)
    dropped_features.to_csv(metrics_dir / "dropped_features.csv", index=False)
    with open(metrics_dir / "candidate_features.txt", "w", encoding="utf-8") as file:
        for column in feature_columns:
            file.write(f"{column}\n")

    report_path = metrics_dir / "data_inspection_report.txt"
    with open(report_path, "w", encoding="utf-8") as file:
        file.write("DATA INSPECTION REPORT\n")
        file.write("======================\n\n")
        file.write(f"Dataset path: {dataset_path}\n")
        file.write(f"Rows: {total_rows}\n")
        file.write(f"Columns: {len(header_columns)}\n")
        file.write(f"Label column: {label_column}\n")
        file.write(f"Classes: {class_distribution.shape[0]}\n")
        file.write(f"Configured model features: {len(feature_columns)}\n")
        file.write(f"Configured dropped columns: {len(drop_columns)}\n")
        file.write(f"Identifier columns: {', '.join(identifier_columns)}\n")
        file.write(f"Quasi-constant threshold: {quasi_constant_threshold}\n\n")
        file.write("Class distribution:\n")
        for _, row in class_distribution.iterrows():
            file.write(
                f"- {row['class_index']}: {row['class']}: "
                f"{row['count']} ({row['percentage']:.4f}%)\n"
            )

    result = {
        "dataset_path": dataset_path,
        "total_rows": total_rows,
        "total_columns": len(header_columns),
        "num_classes": class_distribution.shape[0],
        "feature_count": len(feature_columns),
        "drop_count": len(drop_columns),
        "metrics_dir": metrics_dir,
    }

    expected_rows = data_config["expected_rows"]
    expected_total_columns = data_config["expected_total_columns"]
    expected_feature_count = data_config["expected_feature_count"]
    expected_num_classes = data_config["expected_num_classes"]
    if total_rows != expected_rows:
        raise ValueError(f"Expected {expected_rows} rows, found {total_rows}.")
    if len(header_columns) != expected_total_columns:
        raise ValueError(f"Expected {expected_total_columns} columns, found {len(header_columns)}.")
    if len(feature_columns) != expected_feature_count:
        raise ValueError(f"Expected {expected_feature_count} features, found {len(feature_columns)}.")
    if class_distribution.shape[0] != expected_num_classes:
        raise ValueError(f"Expected {expected_num_classes} classes, found {class_distribution.shape[0]}.")

    return result


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for Phase 1 inspection."""
    parser = argparse.ArgumentParser(description="Inspect the CIC-ToN-IoT raw dataset.")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to the YAML config file.")
    parser.add_argument("--sample-size", type=int, default=100_000, help="Rows to read for sample profiling.")
    parser.add_argument("--chunk-size", type=int, default=100_000, help="Rows per chunk for full profiling.")
    return parser.parse_args()


def main() -> None:
    """Run Phase 1 inspection from the command line."""
    args = parse_args()
    result = inspect_dataset(args.config, args.sample_size, args.chunk_size)
    print("Phase 1 data inspection completed.")
    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
