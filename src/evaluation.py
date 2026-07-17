"""Evaluation and result analysis for LightGBM imbalance scenarios."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from src.data_loading import find_project_root, load_config, resolve_project_path
from src.imbalance import SCENARIO_ALIASES


MINORITY_CLASS_NAMES = ["DoS", "DDoS", "MITM", "Ransomware", "Backdoor"]


def macro_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute accuracy and macro precision, recall, and F1-score."""
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
    }


def file_sha256(path: Path, block_size: int = 1024 * 1024) -> str:
    """Return a SHA256 digest without loading the complete file into memory."""
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        while block := file.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def validate_labels(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
    scenario: str,
    require_all_predicted_classes: bool = True,
) -> None:
    """Validate shape and class range before metric calculation."""
    if y_true.ndim != 1 or y_pred.ndim != 1 or y_true.shape != y_pred.shape:
        raise ValueError(f"Label shape mismatch for {scenario}: {y_true.shape} vs {y_pred.shape}.")
    expected_classes = np.arange(num_classes)
    if not np.array_equal(np.unique(y_true), expected_classes):
        raise ValueError("y_test does not contain every configured class.")
    predicted_classes = np.unique(y_pred)
    if require_all_predicted_classes and not np.array_equal(
        predicted_classes,
        expected_classes,
    ):
        raise ValueError(
            f"{scenario} predictions do not contain every configured class: "
            f"{predicted_classes.tolist()}."
        )


def normalize_confusion_rows(matrix: np.ndarray) -> np.ndarray:
    """Normalize each confusion-matrix row by its true-class support."""
    row_totals = matrix.sum(axis=1, keepdims=True)
    return np.divide(
        matrix,
        row_totals,
        out=np.zeros_like(matrix, dtype=np.float64),
        where=row_totals != 0,
    )


def plot_confusion_matrices(
    matrix: np.ndarray,
    class_names: list[str],
    scenario: str,
    output_path: Path,
) -> Path:
    """Save raw-count and row-normalized confusion heatmaps in one figure."""
    normalized = normalize_confusion_rows(matrix)
    figure, axes = plt.subplots(1, 2, figsize=(22, 9))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=",d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axes[0],
        cbar_kws={"label": "Count"},
    )
    axes[0].set_title("Raw Counts")
    axes[0].set_xlabel("Predicted class")
    axes[0].set_ylabel("True class")

    sns.heatmap(
        normalized,
        annot=True,
        fmt=".1%",
        cmap="YlGnBu",
        vmin=0.0,
        vmax=1.0,
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axes[1],
        cbar_kws={"label": "Row proportion"},
    )
    axes[1].set_title("Normalized by True Class")
    axes[1].set_xlabel("Predicted class")
    axes[1].set_ylabel("True class")

    for axis in axes:
        axis.tick_params(axis="x", rotation=45)
        axis.tick_params(axis="y", rotation=0)
    figure.suptitle(f"Confusion Matrix: {scenario}", fontsize=16)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(figure)
    return output_path


def save_confusion_csv(
    matrix: np.ndarray,
    class_names: list[str],
    output_path: Path,
) -> Path:
    """Persist an exact confusion matrix with explicit true/predicted labels."""
    frame = pd.DataFrame(
        matrix,
        index=[f"true_{name}" for name in class_names],
        columns=[f"pred_{name}" for name in class_names],
    )
    frame.to_csv(output_path, index=True)
    return output_path


def load_classifier_report(metrics_dir: Path, scenario: str) -> dict[str, Any]:
    """Load the canonical Phase 6 report for a scenario."""
    path = metrics_dir / f"classifier_report_{scenario}.json"
    if not path.exists():
        raise FileNotFoundError(f"Phase 6 classifier report not found: {path}")
    with open(path, encoding="utf-8") as file:
        report = json.load(file)
    if report.get("scenario") != scenario:
        raise ValueError(f"Classifier report scenario mismatch in {path}.")
    return report


def evaluate_scenario(
    scenario: str,
    config_path: str | Path = "configs/config.yaml",
) -> dict[str, Any]:
    """Evaluate one saved scenario prediction and persist its complete report."""
    canonical_scenario = SCENARIO_ALIASES.get(scenario.lower())
    if canonical_scenario is None:
        raise ValueError(f"Unknown evaluation scenario: {scenario}")

    project_root = find_project_root()
    config = load_config(config_path)
    processed_dir = resolve_project_path(config["paths"]["processed_dir"], project_root)
    metrics_dir = resolve_project_path(config["paths"]["metrics_dir"], project_root)
    figures_dir = resolve_project_path(config["paths"]["figures_dir"], project_root)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    classifier_report_data = load_classifier_report(metrics_dir, canonical_scenario)
    y_test_path = processed_dir / "y_test.npy"
    prediction_path = Path(classifier_report_data["prediction_path"])
    if not y_test_path.exists() or not prediction_path.exists():
        raise FileNotFoundError(
            f"Evaluation labels or predictions are missing for {canonical_scenario}."
        )

    y_true = np.load(y_test_path, mmap_mode="r")
    y_pred = np.load(prediction_path, mmap_mode="r")
    num_classes = int(config["data"]["expected_num_classes"])
    class_mapping = config["data"]["class_mapping"]
    class_names = [
        name
        for name, _ in sorted(class_mapping.items(), key=lambda item: int(item[1]))
    ]
    labels = list(range(num_classes))
    quick_run_active = bool(
        config.get("pipeline", {}).get("quick_run", {}).get("active", False)
    )
    validate_labels(
        y_true,
        y_pred,
        num_classes,
        canonical_scenario,
        require_all_predicted_classes=not quick_run_active,
    )

    aggregate = macro_metrics(y_true, y_pred)
    per_class_report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    if matrix.shape != (num_classes, num_classes) or int(matrix.sum()) != y_true.shape[0]:
        raise ValueError(f"Invalid confusion matrix for {canonical_scenario}.")

    confusion_csv_path = metrics_dir / f"confusion_matrix_{canonical_scenario}.csv"
    confusion_figure_path = figures_dir / f"confusion_matrix_{canonical_scenario}.png"
    save_confusion_csv(matrix, class_names, confusion_csv_path)
    plot_confusion_matrices(matrix, class_names, canonical_scenario, confusion_figure_path)

    report = {
        "scenario": canonical_scenario,
        "quick_run": quick_run_active,
        "test_rows": int(y_true.shape[0]),
        "class_names": class_names,
        "predicted_classes": np.unique(y_pred).astype(int).tolist(),
        "all_configured_classes_predicted": bool(
            np.array_equal(np.unique(y_pred), np.arange(num_classes))
        ),
        "aggregate_metrics": aggregate,
        "classification_report": per_class_report,
        "confusion_matrix": matrix.astype(int).tolist(),
        "confusion_matrix_csv": str(confusion_csv_path),
        "confusion_matrix_figure": str(confusion_figure_path),
        "y_test_path": str(y_test_path),
        "prediction_path": str(prediction_path),
        "y_test_sha256": file_sha256(y_test_path),
        "prediction_sha256": file_sha256(prediction_path),
        "training_seconds": float(classifier_report_data["training_seconds"]),
        "prediction_seconds": float(classifier_report_data["prediction_seconds"]),
        "model_iterations": int(classifier_report_data["model_iterations"]),
    }
    report_path = metrics_dir / f"report_{canonical_scenario}.json"
    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
    report["report_path"] = str(report_path)
    return report


def build_summary(reports: dict[str, dict[str, Any]]) -> pd.DataFrame:
    """Build the required aggregate scenario comparison table."""
    rows = []
    for report in reports.values():
        metrics = report["aggregate_metrics"]
        rows.append(
            {
                "scenario": report["scenario"],
                "accuracy": metrics["accuracy"],
                "macro_precision": metrics["macro_precision"],
                "macro_recall": metrics["macro_recall"],
                "macro_f1": metrics["macro_f1"],
                "training_seconds": report["training_seconds"],
                "prediction_seconds": report["prediction_seconds"],
            }
        )
    return pd.DataFrame(rows)


def build_per_class_comparison(
    reports: dict[str, dict[str, Any]],
    class_names: list[str],
) -> pd.DataFrame:
    """Build a tidy per-class metric table across all scenarios."""
    rows = []
    for report in reports.values():
        for class_name in class_names:
            metrics = report["classification_report"][class_name]
            rows.append(
                {
                    "scenario": report["scenario"],
                    "class_name": class_name,
                    "precision": float(metrics["precision"]),
                    "recall": float(metrics["recall"]),
                    "f1_score": float(metrics["f1-score"]),
                    "support": int(metrics["support"]),
                    "is_minority_focus": class_name in MINORITY_CLASS_NAMES,
                }
            )
    return pd.DataFrame(rows)


def plot_metric_comparison(summary: pd.DataFrame, output_path: Path) -> Path:
    """Save a grouped comparison of aggregate metrics for all scenarios."""
    metric_columns = ["accuracy", "macro_precision", "macro_recall", "macro_f1"]
    plot_data = summary.melt(
        id_vars="scenario",
        value_vars=metric_columns,
        var_name="metric",
        value_name="score",
    )
    figure, axis = plt.subplots(figsize=(12, 6))
    sns.barplot(data=plot_data, x="scenario", y="score", hue="metric", ax=axis)
    axis.set_ylim(0.0, 1.0)
    axis.set_xlabel("Scenario")
    axis.set_ylabel("Score")
    axis.set_title("Aggregate Evaluation Metrics by Imbalance Scenario")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(title="Metric", ncol=2)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(figure)
    return output_path


def build_qualitative_analysis(
    reports: dict[str, dict[str, Any]],
    summary: pd.DataFrame,
    per_class: pd.DataFrame,
    metrics_dir: Path,
) -> dict[str, Any]:
    """Summarize minority-class effects and known scenario tradeoffs."""
    minority = per_class[per_class["is_minority_focus"]].copy()
    minority_averages = (
        minority.groupby("scenario", sort=False)[["precision", "recall", "f1_score"]]
        .mean()
        .reset_index()
    )
    best_by_class = []
    minority_scenario_effects = []
    for class_name in MINORITY_CLASS_NAMES:
        class_rows = minority[minority["class_name"] == class_name]
        best_recall = class_rows.loc[class_rows["recall"].idxmax()]
        best_f1 = class_rows.loc[class_rows["f1_score"].idxmax()]
        baseline = class_rows[class_rows["scenario"] == "s1_none"].iloc[0]
        best_by_class.append(
            {
                "class_name": class_name,
                "best_recall_scenario": str(best_recall["scenario"]),
                "best_recall": float(best_recall["recall"]),
                "best_f1_scenario": str(best_f1["scenario"]),
                "best_f1": float(best_f1["f1_score"]),
                "baseline_recall": float(baseline["recall"]),
                "baseline_f1": float(baseline["f1_score"]),
            }
        )
        for _, row in class_rows.iterrows():
            minority_scenario_effects.append(
                {
                    "class_name": class_name,
                    "scenario": str(row["scenario"]),
                    "recall": float(row["recall"]),
                    "recall_delta_from_s1": float(row["recall"] - baseline["recall"]),
                    "f1": float(row["f1_score"]),
                    "f1_delta_from_s1": float(row["f1_score"] - baseline["f1_score"]),
                }
            )

    class_names = next(iter(reports.values()))["class_names"]
    class_to_index = {name: index for index, name in enumerate(class_names)}
    top_confusions = []
    for report in reports.values():
        matrix = np.asarray(report["confusion_matrix"], dtype=np.int64)
        for class_name in MINORITY_CLASS_NAMES:
            class_index = class_to_index[class_name]
            row = matrix[class_index].copy()
            support = int(row.sum())
            row[class_index] = 0
            confused_index = int(np.argmax(row))
            confused_count = int(row[confused_index])
            top_confusions.append(
                {
                    "scenario": report["scenario"],
                    "true_class": class_name,
                    "most_confused_with": class_names[confused_index],
                    "misclassified_count": confused_count,
                    "percentage_of_true_class": (
                        float(100.0 * confused_count / support) if support else 0.0
                    ),
                }
            )

    s2 = summary[summary["scenario"] == "s2_class_weight"].iloc[0]
    s3 = summary[summary["scenario"] == "s3_upsampling"].iloc[0]
    s1 = summary[summary["scenario"] == "s1_none"].iloc[0]
    s4 = summary[summary["scenario"] == "s4_downsampling"].iloc[0]
    s1_phase5 = load_phase5_report(metrics_dir, "s1_none")
    s3_phase5 = load_phase5_report(metrics_dir, "s3_upsampling")
    s4_phase5 = load_phase5_report(metrics_dir, "s4_downsampling")
    input_rows = int(s1_phase5["output_rows"])
    majority_count = int(max(s1_phase5["after_counts"]))
    num_classes = len(s1_phase5["after_counts"])
    relative_weight_scale = majority_count * num_classes / input_rows
    retained_fraction = int(s4_phase5["output_rows"]) / input_rows

    best_macro_f1 = summary.loc[summary["macro_f1"].idxmax()]
    best_macro_recall = summary.loc[summary["macro_recall"].idxmax()]
    best_minority_recall = minority_averages.loc[minority_averages["recall"].idxmax()]
    best_minority_f1 = minority_averages.loc[minority_averages["f1_score"].idxmax()]
    s4_minority = minority_averages[
        minority_averages["scenario"] == "s4_downsampling"
    ].iloc[0]
    best_f1_by_class = {
        result["class_name"]: result["best_f1"] for result in best_by_class
    }
    key_findings = [
        (
            f"S1 has the highest accuracy ({s1['accuracy']:.4f}) but a substantially lower "
            f"macro F1-score ({s1['macro_f1']:.4f}) than S2 and S3."
        ),
        (
            f"S2 provides the best aggregate macro recall ({s2['macro_recall']:.4f}) and "
            f"macro F1-score ({s2['macro_f1']:.4f})."
        ),
        (
            f"S2 and S3 differ by only {abs(s2['macro_f1'] - s3['macro_f1']):.4f} macro F1, "
            "which is consistent with their equivalent relative class contribution."
        ),
        (
            f"S4 reaches the highest minority mean recall ({s4_minority['recall']:.4f}) but "
            f"low minority mean precision ({s4_minority['precision']:.4f}) and F1 "
            f"({s4_minority['f1_score']:.4f}), indicating extensive false positives."
        ),
        (
            "DoS and DDoS remain the least reliable minority classes across all scenarios; "
            f"their best per-class F1-scores are {best_f1_by_class['DoS']:.4f} and "
            f"{best_f1_by_class['DDoS']:.4f}, respectively."
        ),
    ]
    analysis = {
        "best_accuracy": {
            "scenario": str(summary.loc[summary["accuracy"].idxmax(), "scenario"]),
            "score": float(summary["accuracy"].max()),
        },
        "best_macro_f1": {
            "scenario": str(best_macro_f1["scenario"]),
            "score": float(best_macro_f1["macro_f1"]),
        },
        "best_macro_recall": {
            "scenario": str(best_macro_recall["scenario"]),
            "score": float(best_macro_recall["macro_recall"]),
        },
        "best_minority_mean_recall": {
            "scenario": str(best_minority_recall["scenario"]),
            "score": float(best_minority_recall["recall"]),
        },
        "best_minority_mean_f1": {
            "scenario": str(best_minority_f1["scenario"]),
            "score": float(best_minority_f1["f1_score"]),
        },
        "minority_class_best_results": best_by_class,
        "minority_scenario_effects": minority_scenario_effects,
        "minority_scenario_averages": minority_averages.to_dict(orient="records"),
        "top_minority_confusions": top_confusions,
        "key_findings": key_findings,
        "s1_accuracy_macro_f1_tradeoff": {
            "accuracy": float(s1["accuracy"]),
            "macro_f1": float(s1["macro_f1"]),
            "macro_recall": float(s1["macro_recall"]),
        },
        "s2_s3_equivalence": {
            "macro_f1_absolute_delta": float(abs(s2["macro_f1"] - s3["macro_f1"])),
            "macro_recall_absolute_delta": float(
                abs(s2["macro_recall"] - s3["macro_recall"])
            ),
            "training_time_ratio_s3_over_s2": float(
                s3["training_seconds"] / s2["training_seconds"]
            ),
            "relative_weight_global_scale_s3_over_s2": float(relative_weight_scale),
            "explanation": (
                "Random duplication and class weighting give LightGBM the same relative "
                "per-class contribution up to a constant global scale. Their similar metrics "
                "are therefore an expected experimental finding, not an implementation defect."
            ),
            "s3_method_retained": "random_duplication",
        },
        "s4_information_loss": {
            "training_rows": int(s4_phase5["output_rows"]),
            "retained_fraction": float(retained_fraction),
            "discarded_percentage": float(100.0 * (1.0 - retained_fraction)),
            "accuracy": float(s4["accuracy"]),
            "macro_f1": float(s4["macro_f1"]),
        },
    }
    return analysis


def write_analysis_markdown(
    analysis: dict[str, Any],
    summary: pd.DataFrame,
    output_path: Path,
) -> Path:
    """Write a concise, report-ready qualitative Phase 7 summary."""
    lines = [
        "# Phase 7 Evaluation Analysis",
        "",
        "## Aggregate Metrics",
        "",
        "| Scenario | Accuracy | Macro Precision | Macro Recall | Macro F1 | Training (s) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['scenario']} | {row['accuracy']:.6f} | "
            f"{row['macro_precision']:.6f} | {row['macro_recall']:.6f} | "
            f"{row['macro_f1']:.6f} | {row['training_seconds']:.3f} |"
        )

    lines.extend(["", "## Key Findings", ""])
    lines.extend(f"- {finding}" for finding in analysis["key_findings"])
    lines.extend(
        [
            "",
            "## Minority-Class Best Results",
            "",
            "| Class | Best Recall Scenario | Recall | Best F1 Scenario | F1 |",
            "|---|---|---:|---|---:|",
        ]
    )
    for result in analysis["minority_class_best_results"]:
        lines.append(
            f"| {result['class_name']} | {result['best_recall_scenario']} | "
            f"{result['best_recall']:.6f} | {result['best_f1_scenario']} | "
            f"{result['best_f1']:.6f} |"
        )

    equivalence = analysis["s2_s3_equivalence"]
    loss = analysis["s4_information_loss"]
    lines.extend(
        [
            "",
            "## Methodological Interpretation",
            "",
            f"- S3 took {equivalence['training_time_ratio_s3_over_s2']:.3f} times as long "
            "as S2 while producing nearly identical aggregate metrics.",
            f"- The S3-to-S2 relative-weight scale is a constant "
            f"{equivalence['relative_weight_global_scale_s3_over_s2']:.6f}; global scale "
            "does not change their relative class contribution.",
            f"- S4 retained only {100.0 * loss['retained_fraction']:.6f}% of training rows "
            f"and discarded {loss['discarded_percentage']:.6f}%.",
            "- S3 remains random duplication, as permitted by the proposal; the S2-S3 "
            "similarity is retained as an experimental finding rather than replaced with SMOTE.",
            "",
        ]
    )
    with open(output_path, "w", encoding="utf-8") as file:
        file.write("\n".join(lines))
    return output_path


def load_phase5_report(metrics_dir: Path, scenario: str) -> dict[str, Any]:
    """Load one Phase 5 report for cross-phase qualitative analysis."""
    path = metrics_dir / f"imbalance_report_{scenario}.json"
    if not path.exists():
        raise FileNotFoundError(f"Phase 5 report not found: {path}")
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def evaluate_all_scenarios(
    config_path: str | Path = "configs/config.yaml",
) -> dict[str, Any]:
    """Evaluate all scenarios and save aggregate plus per-class comparisons."""
    project_root = find_project_root()
    config = load_config(config_path)
    metrics_dir = resolve_project_path(config["paths"]["metrics_dir"], project_root)
    figures_dir = resolve_project_path(config["paths"]["figures_dir"], project_root)
    scenarios = config["imbalance"]["scenarios"]
    class_names = [
        name
        for name, _ in sorted(
            config["data"]["class_mapping"].items(),
            key=lambda item: int(item[1]),
        )
    ]

    reports = {
        scenario: evaluate_scenario(scenario, config_path)
        for scenario in scenarios
    }
    summary = build_summary(reports)
    summary_path = metrics_dir / "summary_comparison.csv"
    summary.to_csv(summary_path, index=False)

    per_class = build_per_class_comparison(reports, class_names)
    per_class_path = metrics_dir / "per_class_comparison.csv"
    per_class.to_csv(per_class_path, index=False)
    minority_path = metrics_dir / "minority_class_comparison.csv"
    per_class[per_class["is_minority_focus"]].to_csv(minority_path, index=False)

    metric_figure_path = figures_dir / "metrics_comparison.png"
    plot_metric_comparison(summary, metric_figure_path)
    qualitative = build_qualitative_analysis(reports, summary, per_class, metrics_dir)
    qualitative_path = metrics_dir / "phase7_qualitative_analysis.json"
    with open(qualitative_path, "w", encoding="utf-8") as file:
        json.dump(qualitative, file, indent=2)
    analysis_markdown_path = metrics_dir / "phase7_analysis.md"
    write_analysis_markdown(qualitative, summary, analysis_markdown_path)

    return {
        "reports": reports,
        "summary": summary,
        "per_class": per_class,
        "qualitative_analysis": qualitative,
        "summary_path": str(summary_path),
        "per_class_path": str(per_class_path),
        "minority_path": str(minority_path),
        "metric_figure_path": str(metric_figure_path),
        "qualitative_path": str(qualitative_path),
        "analysis_markdown_path": str(analysis_markdown_path),
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for Phase 7."""
    parser = argparse.ArgumentParser(description="Evaluate saved LightGBM predictions.")
    parser.add_argument(
        "--scenario",
        default="all",
        choices=["all", *SCENARIO_ALIASES],
        help="Scenario to evaluate.",
    )
    parser.add_argument("--config", default="configs/config.yaml", help="YAML config path.")
    return parser.parse_args()


def main() -> None:
    """Run Phase 7 from the command line."""
    args = parse_args()
    if args.scenario == "all":
        result = evaluate_all_scenarios(args.config)
        print("Phase 7 evaluation completed.")
        print(result["summary"].to_string(index=False))
    else:
        report = evaluate_scenario(args.scenario, args.config)
        print(f"Phase 7 evaluation completed for {report['scenario']}.")
        print(json.dumps(report["aggregate_metrics"], indent=2))


if __name__ == "__main__":
    main()
