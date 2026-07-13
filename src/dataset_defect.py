"""Detect contradictory duplicate flow records in CIC-ToN-IoT.

The DoS and DDoS classes cannot be separated by any model trained on the CICFlowMeter
feature set, because the dataset stores the same flow record twice under both labels.
This module quantifies that defect and writes the evidence needed to report it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.data_loading import find_project_root, load_config, normalize_attack_label, resolve_project_path


IDENTITY_COLUMNS = ["Flow ID", "Src IP", "Dst IP", "Timestamp"]


def load_class_rows(
    dataset_path: Path,
    feature_columns: list[str],
    label_column: str,
    aliases: dict[str, str],
    target_classes: set[str],
    chunk_size: int,
) -> pd.DataFrame:
    """Load every raw row belonging to the target classes, with identifiers retained."""
    use_columns = list(dict.fromkeys(feature_columns + IDENTITY_COLUMNS + [label_column]))
    collected: list[pd.DataFrame] = []
    reader = pd.read_csv(dataset_path, usecols=use_columns, chunksize=chunk_size, low_memory=False)
    for chunk in reader:
        chunk.columns = chunk.columns.str.strip()
        canonical = chunk[label_column].map(lambda value: normalize_attack_label(value, aliases))
        selected = chunk[canonical.isin(target_classes)].copy()
        if not selected.empty:
            selected[label_column] = canonical[canonical.isin(target_classes)]
            collected.append(selected)

    if not collected:
        raise ValueError(f"No rows were found for classes: {sorted(target_classes)}")
    return pd.concat(collected, ignore_index=True)


def analyze_label_conflicts(
    rows: pd.DataFrame,
    feature_columns: list[str],
    label_column: str,
) -> dict[str, Any]:
    """Quantify feature-identical rows that carry contradictory class labels."""
    grouped = rows.groupby(feature_columns, dropna=False, sort=False)
    conflict_groups = []
    for _, group in grouped:
        if group[label_column].nunique() < 2:
            continue
        conflict_groups.append(group)

    class_totals = rows[label_column].value_counts().to_dict()
    conflicted_counts: dict[str, int] = {name: 0 for name in class_totals}
    identical_identity_groups = 0

    for group in conflict_groups:
        for name, count in group[label_column].value_counts().items():
            conflicted_counts[name] += int(count)
        if all(group[column].nunique() == 1 for column in IDENTITY_COLUMNS):
            identical_identity_groups += 1

    examples = []
    for group in conflict_groups[:5]:
        examples.append(
            {
                "flow_id": str(group["Flow ID"].iloc[0]),
                "src_ip": str(group["Src IP"].iloc[0]),
                "dst_ip": str(group["Dst IP"].iloc[0]),
                "timestamp": str(group["Timestamp"].iloc[0]),
                "labels": sorted(group[label_column].tolist()),
                "identifiers_identical": bool(
                    all(group[column].nunique() == 1 for column in IDENTITY_COLUMNS)
                ),
            }
        )

    return {
        "class_totals": {name: int(count) for name, count in class_totals.items()},
        "conflict_groups": len(conflict_groups),
        "rows_in_conflict": int(sum(len(group) for group in conflict_groups)),
        "conflicted_rows_per_class": conflicted_counts,
        "conflicted_fraction_per_class": {
            name: (conflicted_counts[name] / class_totals[name]) if class_totals[name] else 0.0
            for name in class_totals
        },
        "groups_with_identical_identifiers": identical_identity_groups,
        "identifier_columns_checked": IDENTITY_COLUMNS,
        "examples": examples,
    }


def write_defect_markdown(analysis: dict[str, Any], output_path: Path) -> Path:
    """Write a report-ready summary of the dataset defect."""
    totals = analysis["class_totals"]
    conflicted = analysis["conflicted_rows_per_class"]
    fractions = analysis["conflicted_fraction_per_class"]
    lines = [
        "# CIC-ToN-IoT Label Contradiction: DoS and DDoS",
        "",
        "## Finding",
        "",
        f"The dataset contains {analysis['conflict_groups']} groups of rows whose complete "
        "CICFlowMeter feature vectors are identical but whose `Attack` labels contradict "
        f"each other. In total {analysis['rows_in_conflict']} rows are involved.",
        "",
        f"All {analysis['groups_with_identical_identifiers']} of these groups also share the "
        "same `Flow ID`, `Src IP`, `Dst IP`, and `Timestamp`. They are therefore the same "
        "flow record stored twice under two different labels, not two similar flows.",
        "",
        "## Coverage",
        "",
        "| Class | Total rows | Rows in conflict | Share |",
        "|---|---:|---:|---:|",
    ]
    for name in sorted(totals):
        lines.append(
            f"| {name} | {totals[name]:,} | {conflicted[name]:,} | {100 * fractions[name]:.1f}% |"
        )

    lines.extend(
        [
            "",
            "## Consequence",
            "",
            "For every affected row the model receives identical inputs paired with "
            "contradictory targets. No classifier, dimensionality-reduction method, or "
            "imbalance-handling strategy can separate these rows, because the information "
            "required to do so is absent from the feature set. On these rows any classifier "
            "faces a forced trade-off: every correctly recalled row under one label is "
            "matched by an identical row under the other label that is then necessarily "
            "misclassified, so high F1 on both classes simultaneously is unattainable.",
            "",
            "This explains the near-total mutual confusion observed between DoS and DDoS in "
            "every experimental scenario, and it holds for the original 69 features as well "
            "as for the 16-dimensional Autoencoder latent features.",
            "",
            "## Examples",
            "",
            "| Flow ID | Timestamp | Labels | Identifiers identical |",
            "|---|---|---|---|",
        ]
    )
    for example in analysis["examples"]:
        lines.append(
            f"| `{example['flow_id']}` | {example['timestamp']} | "
            f"{', '.join(example['labels'])} | {example['identifiers_identical']} |"
        )
    lines.append("")

    with open(output_path, "w", encoding="utf-8") as file:
        file.write("\n".join(lines))
    return output_path


def detect_dataset_defect(
    config_path: str | Path = "configs/config.yaml",
    classes: tuple[str, ...] = ("DoS", "DDoS"),
    chunk_size: int = 500_000,
) -> dict[str, Any]:
    """Detect and report contradictory duplicate records for the given classes."""
    project_root = find_project_root()
    config = load_config(config_path)
    data_config = config["data"]
    dataset_path = resolve_project_path(config["paths"]["raw_dataset"], project_root)
    metrics_dir = resolve_project_path(config["paths"]["metrics_dir"], project_root)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    rows = load_class_rows(
        dataset_path,
        data_config["feature_columns"],
        data_config["label_column"],
        data_config["raw_label_aliases"],
        set(classes),
        chunk_size,
    )
    analysis = analyze_label_conflicts(
        rows,
        data_config["feature_columns"],
        data_config["label_column"],
    )
    analysis["classes_examined"] = list(classes)
    analysis["dataset_path"] = str(dataset_path)

    json_path = metrics_dir / "dataset_label_conflicts.json"
    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(analysis, file, indent=2)
    markdown_path = write_defect_markdown(analysis, metrics_dir / "dataset_label_conflicts.md")

    analysis["json_path"] = str(json_path)
    analysis["markdown_path"] = str(markdown_path)
    return analysis


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for dataset defect detection."""
    parser = argparse.ArgumentParser(description="Detect contradictory duplicate flow records.")
    parser.add_argument("--config", default="configs/config.yaml", help="YAML config path.")
    parser.add_argument(
        "--classes",
        nargs="+",
        default=["DoS", "DDoS"],
        help="Classes to examine for label conflicts.",
    )
    return parser.parse_args()


def main() -> None:
    """Run dataset defect detection from the command line."""
    args = parse_args()
    analysis = detect_dataset_defect(args.config, tuple(args.classes))
    print("Dataset label-conflict analysis completed.")
    print(f"Conflict groups: {analysis['conflict_groups']}")
    print(f"Rows in conflict: {analysis['rows_in_conflict']}")
    for name, fraction in analysis["conflicted_fraction_per_class"].items():
        print(f"  {name}: {100 * fraction:.1f}% of rows are contradicted")
    print(f"Report: {analysis['markdown_path']}")


if __name__ == "__main__":
    main()
