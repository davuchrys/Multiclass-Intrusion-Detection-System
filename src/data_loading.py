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
    signal_check_columns: list[str],
    aliases: dict[str, str],
    chunk_size: int,
) -> tuple[
    int,
    Counter[str],
    Counter[str],
    Counter[str],
    dict[str, Counter[object]],
    dict[tuple[str, str], dict[str, Any]],
]:
    """Profile row count, class distribution, missing values, infinite values, and class signal."""
    total_rows = 0
    class_counts: Counter[str] = Counter()
    missing_counts: Counter[str] = Counter()
    infinite_counts: Counter[str] = Counter()
    signal_value_counts: dict[str, Counter[object]] = {
        column: Counter() for column in signal_check_columns
    }
    class_signal_stats: dict[tuple[str, str], dict[str, Any]] = {}

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

        signal_numeric = {
            column: pd.to_numeric(chunk[column], errors="coerce")
            for column in signal_check_columns
            if column in chunk.columns
        }
        for column, series in signal_numeric.items():
            signal_value_counts[column].update(series.value_counts(dropna=False).to_dict())

        for class_name in labels.dropna().unique():
            class_mask = labels == class_name
            for column, series in signal_numeric.items():
                class_series = series.loc[class_mask].dropna()
                key = (column, str(class_name))
                stats = class_signal_stats.setdefault(
                    key,
                    {
                        "count": 0,
                        "non_zero_count": 0,
                        "sum": 0.0,
                        "min": None,
                        "max": None,
                        "unique_values": set(),
                    },
                )
                if class_series.empty:
                    continue

                stats["count"] += int(class_series.shape[0])
                stats["non_zero_count"] += int((class_series != 0).sum())
                stats["sum"] += float(class_series.sum())
                current_min = float(class_series.min())
                current_max = float(class_series.max())
                stats["min"] = current_min if stats["min"] is None else min(stats["min"], current_min)
                stats["max"] = current_max if stats["max"] is None else max(stats["max"], current_max)
                stats["unique_values"].update(class_series.unique().tolist())

    return (
        total_rows,
        class_counts,
        missing_counts,
        infinite_counts,
        signal_value_counts,
        class_signal_stats,
    )


def build_class_signal_summary(
    class_signal_stats: dict[tuple[str, str], dict[str, Any]],
    class_mapping: dict[str, int],
) -> pd.DataFrame:
    """Convert class-conditional signal statistics into a tabular summary."""
    rows: list[dict[str, Any]] = []
    for (feature, class_name), stats in class_signal_stats.items():
        count = int(stats["count"])
        rows.append(
            {
                "feature": feature,
                "class": class_name,
                "class_index": class_mapping.get(class_name),
                "count": count,
                "non_zero_count": int(stats["non_zero_count"]),
                "non_zero_ratio": float(stats["non_zero_count"] / count) if count else 0.0,
                "mean": float(stats["sum"] / count) if count else 0.0,
                "min": stats["min"],
                "max": stats["max"],
                "unique_count": len(stats["unique_values"]),
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["feature", "class_index"]).reset_index(drop=True)


def build_feature_signal_flags(class_signal_summary: pd.DataFrame) -> pd.DataFrame:
    """Summarize whether a feature has class-conditional signal despite global sparsity."""
    if class_signal_summary.empty:
        return pd.DataFrame(
            columns=[
                "feature",
                "max_class_non_zero_ratio",
                "max_class_unique_count",
                "class_mean_range",
                "has_class_conditional_signal",
            ]
        )

    grouped = class_signal_summary.groupby("feature", as_index=False).agg(
        max_class_non_zero_ratio=("non_zero_ratio", "max"),
        max_class_unique_count=("unique_count", "max"),
        min_class_mean=("mean", "min"),
        max_class_mean=("mean", "max"),
    )
    grouped["class_mean_range"] = grouped["max_class_mean"] - grouped["min_class_mean"]
    grouped["has_class_conditional_signal"] = (
        (grouped["max_class_unique_count"] > 1)
        & (grouped["max_class_non_zero_ratio"] >= 0.01)
    )
    return grouped[
        [
            "feature",
            "max_class_non_zero_ratio",
            "max_class_unique_count",
            "class_mean_range",
            "has_class_conditional_signal",
        ]
    ]


def build_feature_summary(
    sample_profile: pd.DataFrame,
    total_rows: int,
    missing_counts: Counter[str],
    infinite_counts: Counter[str],
    feature_columns: list[str],
    drop_columns: list[str],
    identifier_columns: list[str],
    secondary_label_column: str,
    quasi_constant_threshold: float,
    full_top_value_frequency: dict[str, float],
    feature_signal_flags: pd.DataFrame,
) -> pd.DataFrame:
    """Combine sample and full-dataset statistics into a Phase 1 feature summary."""
    summary = sample_profile.copy()
    summary["missing_count"] = summary["feature"].map(lambda value: missing_counts.get(value, 0))
    summary["missing_percent"] = summary["missing_count"] / total_rows * 100
    summary["inf_count"] = summary["feature"].map(lambda value: infinite_counts.get(value, 0))
    summary["inf_percent"] = summary["inf_count"] / total_rows * 100
    summary["is_identifier"] = summary["feature"].isin(identifier_columns)
    summary["is_secondary_label"] = summary["feature"] == secondary_label_column
    summary["full_top_value_frequency"] = summary["feature"].map(full_top_value_frequency)
    summary["is_globally_quasi_constant"] = (
        summary["full_top_value_frequency"].fillna(summary["sample_top_value_frequency"])
        >= quasi_constant_threshold
    )
    summary["configured_drop"] = summary["feature"].isin(drop_columns)
    summary["configured_model_feature"] = summary["feature"].isin(feature_columns)

    signal_lookup = feature_signal_flags.set_index("feature") if not feature_signal_flags.empty else None
    summary["has_class_conditional_signal"] = summary["feature"].map(
        lambda value: bool(signal_lookup.loc[value, "has_class_conditional_signal"])
        if signal_lookup is not None and value in signal_lookup.index
        else False
    )

    def reasons(row: pd.Series) -> str:
        values: list[str] = []
        if row["is_identifier"]:
            values.append("identifier or metadata column")
        if row["is_secondary_label"]:
            values.append("secondary target label; leakage risk")
        if not row["is_numeric_candidate"]:
            values.append("non-numeric or not safely convertible to numeric")
        if (
            row["is_globally_quasi_constant"]
            and not row["has_class_conditional_signal"]
            and not row["is_secondary_label"]
            and not row["is_identifier"]
        ):
            values.append("globally and class-conditionally quasi-constant")
        if row["configured_drop"] and row["has_class_conditional_signal"] and not row["is_secondary_label"]:
            values.append("configured for exclusion but class-conditional signal detected")
        elif row["configured_drop"] and not values:
            values.append("configured for exclusion")
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
    secondary_label_column = data_config["secondary_label_column"]
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
    signal_check_columns = sample_profile.loc[
        sample_profile["is_numeric_candidate"]
        & (
            (sample_profile["sample_top_value_frequency"] >= quasi_constant_threshold)
            | (sample_profile["feature"].isin(drop_columns))
        ),
        "feature",
    ].tolist()
    (
        total_rows,
        class_counts,
        missing_counts,
        infinite_counts,
        signal_value_counts,
        class_signal_stats,
    ) = profile_full_dataset(
        dataset_path,
        encoding,
        label_column,
        [col for col in header_columns if col != label_column],
        numeric_columns,
        signal_check_columns,
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

    full_top_value_frequency = {
        column: (max(counter.values()) / total_rows if counter else np.nan)
        for column, counter in signal_value_counts.items()
    }
    class_signal_summary = build_class_signal_summary(class_signal_stats, class_mapping)
    feature_signal_flags = build_feature_signal_flags(class_signal_summary)

    feature_summary = build_feature_summary(
        sample_profile,
        total_rows,
        missing_counts,
        infinite_counts,
        feature_columns,
        drop_columns,
        identifier_columns,
        secondary_label_column,
        quasi_constant_threshold,
        full_top_value_frequency,
        feature_signal_flags,
    )
    dropped_features = feature_summary.loc[
        feature_summary["feature"].isin(drop_columns),
        ["feature", "recommended_drop_reason"],
    ].sort_values("feature")
    quasi_constant_review = feature_summary.loc[
        feature_summary["is_globally_quasi_constant"],
        [
            "feature",
            "configured_model_feature",
            "configured_drop",
            "full_top_value_frequency",
            "has_class_conditional_signal",
            "recommended_drop_reason",
        ],
    ].sort_values(["configured_drop", "feature"], ascending=[False, True])

    class_distribution.to_csv(metrics_dir / "class_distribution.csv", index=False)
    feature_summary.to_csv(metrics_dir / "feature_summary.csv", index=False)
    dropped_features.to_csv(metrics_dir / "dropped_features.csv", index=False)
    class_signal_summary.to_csv(metrics_dir / "class_signal_summary.csv", index=False)
    quasi_constant_review.to_csv(metrics_dir / "quasi_constant_review.csv", index=False)
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
        file.write("Class-conditional quasi-constant review:\n")
        reviewed = quasi_constant_review[
            [
                "feature",
                "configured_model_feature",
                "configured_drop",
                "full_top_value_frequency",
                "has_class_conditional_signal",
                "recommended_drop_reason",
            ]
        ]
        for _, row in reviewed.iterrows():
            file.write(
                f"- {row['feature']}: model_feature={row['configured_model_feature']}, "
                f"drop={row['configured_drop']}, "
                f"full_top_frequency={row['full_top_value_frequency']:.6f}, "
                f"class_signal={row['has_class_conditional_signal']}, "
                f"reason={row['recommended_drop_reason']}\n"
            )
        file.write("\n")
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
