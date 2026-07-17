"""Generate full-data Phase 9 tables, figures, and Chapter IV draft material."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.data_loading import find_project_root, load_config, resolve_project_path


SCENARIO_LABELS = {
    "s1_none": "S1 - No handling",
    "s2_class_weight": "S2 - Class weight",
    "s3_upsampling": "S3 - Upsampling",
    "s4_downsampling": "S4 - Downsampling",
}

SCENARIO_COLORS = {
    "S1 - No handling": "#3366CC",
    "S2 - Class weight": "#109D76",
    "S3 - Upsampling": "#F4B400",
    "S4 - Downsampling": "#DB4437",
}


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON artifact."""
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def require_paths(paths: list[Path]) -> None:
    """Fail with one actionable message when report inputs are incomplete."""
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Phase 9 requires completed full-data Phase 1-7 and baseline artifacts: "
            f"{missing}"
        )


def markdown_table(frame: pd.DataFrame, digits: int = 4) -> str:
    """Render a compact Markdown table without an optional tabulate dependency."""
    headers = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    for row in frame.itertuples(index=False, name=None):
        cells = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                cell = f"{float(value):.{digits}f}"
            else:
                cell = str(value)
            cells.append(cell.replace("|", "\\|"))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def validate_full_data_inputs(
    config: dict[str, Any],
    metrics_dir: Path,
    summary: pd.DataFrame,
    reports: dict[str, dict[str, Any]],
    preprocessing: dict[str, Any],
    baseline: pd.DataFrame,
    imbalance_reports: dict[str, dict[str, Any]],
    classifier_integrity: dict[str, Any],
) -> None:
    """Reject quick-run, stale, or internally inconsistent report inputs."""
    scenarios = list(config["imbalance"]["scenarios"])
    if summary["scenario"].tolist() != scenarios:
        raise ValueError("Full-data summary scenarios do not match config order.")
    expected_test_rows = int(preprocessing["test_rows"])
    for scenario, report in reports.items():
        if report.get("quick_run") is not False:
            raise ValueError(f"Refusing to report non-full evaluation artifact: {scenario}.")
        if int(report["test_rows"]) != expected_test_rows:
            raise ValueError(f"Unexpected test row count for {scenario}.")
        if report.get("all_configured_classes_predicted") is not True:
            raise ValueError(f"Full-data predictions are incomplete for {scenario}.")
        summary_row = summary.loc[summary["scenario"] == scenario].iloc[0]
        for metric in ("accuracy", "macro_precision", "macro_recall", "macro_f1"):
            if not np.isclose(
                float(summary_row[metric]),
                float(report["aggregate_metrics"][metric]),
                atol=1e-12,
                rtol=0.0,
            ):
                raise ValueError(f"Stale summary value for {scenario} {metric}.")

    if preprocessing.get("scaler_fit_scope") != "train_only":
        raise ValueError("Preprocessing report does not prove train-only scaler fitting.")
    if int(preprocessing["feature_count"]) != int(config["data"]["expected_feature_count"]):
        raise ValueError("Preprocessing feature count does not match config.")
    if baseline.shape[0] != 4:
        raise ValueError("The required latent/original S1-S2 baseline is incomplete.")
    if set(baseline["representation"]) != {"latent_16", "original_69"}:
        raise ValueError("Unexpected representations in the Phase 9 baseline.")
    if classifier_integrity.get("test_artifacts_unchanged") is not True:
        raise ValueError("Classifier integrity report does not preserve test artifacts.")
    if set(classifier_integrity.get("scenarios", [])) != set(scenarios):
        raise ValueError("Classifier integrity report does not cover all scenarios.")
    for scenario, report in imbalance_reports.items():
        if report.get("test_data_modified") is not False:
            raise ValueError(f"Phase 5 modified test data for {scenario}.")

    quick_root = resolve_project_path(
        config["pipeline"]["quick_run"]["root_dir"],
        find_project_root(),
    ).resolve()
    for scenario in scenarios:
        prediction_path = Path(reports[scenario]["prediction_path"]).resolve()
        if prediction_path.is_relative_to(quick_root):
            raise ValueError("Quick-run predictions cannot be used in the Chapter IV draft.")
    if not (metrics_dir / "summary_comparison.csv").exists():
        raise FileNotFoundError("Canonical full-data summary is missing.")


def plot_autoencoder_loss(history: pd.DataFrame, output_path: Path) -> None:
    """Plot training and validation reconstruction loss."""
    fig, axis = plt.subplots(figsize=(8.2, 4.8))
    axis.plot(history["epoch"], history["train_loss"], label="Training loss", color="#3366CC")
    axis.plot(history["epoch"], history["val_loss"], label="Validation loss", color="#DB4437")
    axis.set_yscale("log")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Mean squared error (log scale)")
    axis.set_title("Autoencoder Reconstruction Loss")
    axis.legend(frameon=False)
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_scenario_metrics(summary: pd.DataFrame, output_path: Path) -> None:
    """Plot aggregate metrics for the four imbalance scenarios."""
    metric_columns = ["accuracy", "macro_precision", "macro_recall", "macro_f1"]
    labels = ["Accuracy", "Macro precision", "Macro recall", "Macro F1"]
    x_positions = np.arange(len(summary))
    width = 0.19
    fig, axis = plt.subplots(figsize=(10.5, 5.2))
    colors = ["#3366CC", "#7B61A8", "#109D76", "#DB4437"]
    for index, (metric, label, color) in enumerate(zip(metric_columns, labels, colors)):
        offset = (index - 1.5) * width
        axis.bar(x_positions + offset, summary[metric], width, label=label, color=color)
    axis.set_xticks(x_positions)
    axis.set_xticklabels(summary["scenario_label"])
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("Score")
    axis.set_title("Full-Data Performance by Imbalance Scenario")
    axis.legend(ncol=2, frameon=False)
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_minority_f1(minority: pd.DataFrame, output_path: Path) -> None:
    """Plot per-scenario F1-scores for the five focus minority classes."""
    class_order = ["Backdoor", "DDoS", "DoS", "MITM", "Ransomware"]
    pivot = minority.pivot(index="class_name", columns="scenario_label", values="f1_score")
    pivot = pivot.reindex(class_order)
    colors = [SCENARIO_COLORS[column] for column in pivot.columns]
    axis = pivot.plot(kind="bar", figsize=(10.5, 5.2), color=colors, width=0.82)
    axis.set_ylim(0.0, 1.0)
    axis.set_xlabel("Minority class")
    axis.set_ylabel("F1-score")
    axis.set_title("Minority-Class F1 by Imbalance Scenario")
    axis.legend(title=None, frameon=False, ncol=2)
    axis.grid(axis="y", alpha=0.25)
    axis.figure.tight_layout()
    axis.figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(axis.figure)


def plot_representation_baseline(baseline: pd.DataFrame, output_path: Path) -> None:
    """Compare latent and original representations under controlled scenarios."""
    display = baseline.copy()
    display["representation_label"] = display["representation"].map(
        {"latent_16": "Latent 16", "original_69": "Original 69"}
    )
    display["scenario_label"] = display["scenario"].map(
        {"s1_none": "S1", "s2_class_weight": "S2"}
    )
    melted = display.melt(
        id_vars=["representation_label", "scenario_label"],
        value_vars=["macro_recall", "macro_f1"],
        var_name="metric",
        value_name="score",
    )
    melted["group"] = melted["scenario_label"] + " / " + melted["representation_label"]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8), sharey=True)
    palette = {"Latent 16": "#3366CC", "Original 69": "#109D76"}
    for axis, metric, title in zip(
        axes,
        ["macro_recall", "macro_f1"],
        ["Macro recall", "Macro F1"],
    ):
        subset = melted[melted["metric"] == metric]
        sns.barplot(
            data=subset,
            x="scenario_label",
            y="score",
            hue="representation_label",
            palette=palette,
            ax=axis,
        )
        axis.set_title(title)
        axis.set_xlabel("Scenario")
        axis.set_ylabel("Score" if axis is axes[0] else "")
        axis.set_ylim(0.0, 0.75)
        axis.grid(axis="y", alpha=0.25)
        axis.legend(title=None, frameon=False)
    fig.suptitle("Latent-16 versus Original-69 LightGBM Baseline")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_s2_confusion(report: dict[str, Any], output_path: Path) -> None:
    """Plot a row-normalized S2 confusion matrix for discussion."""
    matrix = np.asarray(report["confusion_matrix"], dtype=np.float64)
    row_totals = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(
        matrix,
        row_totals,
        out=np.zeros_like(matrix),
        where=row_totals != 0,
    )
    fig, axis = plt.subplots(figsize=(9.0, 7.2))
    sns.heatmap(
        normalized,
        cmap="YlGnBu",
        vmin=0.0,
        vmax=1.0,
        xticklabels=report["class_names"],
        yticklabels=report["class_names"],
        ax=axis,
    )
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("True class")
    axis.set_title("S2 Row-Normalized Confusion Matrix")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_chapter_draft(
    output_path: Path,
    preprocessing: dict[str, Any],
    reconstruction: dict[str, Any],
    summary: pd.DataFrame,
    minority: pd.DataFrame,
    per_class: pd.DataFrame,
    baseline: pd.DataFrame,
    baseline_diagnostic: dict[str, Any],
    hypothesis_table: pd.DataFrame,
    optional_results: dict[str, Any],
) -> None:
    """Write the reproducible Chapter IV results and discussion draft."""
    s1 = summary.loc[summary["scenario"] == "s1_none"].iloc[0]
    s2 = summary.loc[summary["scenario"] == "s2_class_weight"].iloc[0]
    s3 = summary.loc[summary["scenario"] == "s3_upsampling"].iloc[0]
    s4 = summary.loc[summary["scenario"] == "s4_downsampling"].iloc[0]
    baseline_s2_latent = baseline[
        (baseline["representation"] == "latent_16")
        & (baseline["scenario"] == "s2_class_weight")
    ].iloc[0]
    baseline_s2_original = baseline[
        (baseline["representation"] == "original_69")
        & (baseline["scenario"] == "s2_class_weight")
    ].iloc[0]
    xss_s1 = per_class[
        (per_class["scenario"] == "s1_none") & (per_class["class_name"] == "XSS")
    ].iloc[0]
    xss_s2 = per_class[
        (per_class["scenario"] == "s2_class_weight")
        & (per_class["class_name"] == "XSS")
    ].iloc[0]
    dimension_reduction = (1.0 - reconstruction["latent_dim"] / reconstruction["input_dim"]) * 100
    test_gap = (
        reconstruction["test_reconstruction_mse"]
        / reconstruction["train_reconstruction_mse"]
        - 1.0
    ) * 100
    s2_s3_f1_delta = abs(float(s2["macro_f1"]) - float(s3["macro_f1"]))
    s3_time_ratio = float(s3["training_seconds"]) / float(s2["training_seconds"])
    baseline_f1_delta = float(baseline_s2_original["macro_f1"]) - float(
        baseline_s2_latent["macro_f1"]
    )
    classifier_time_reduction = (
        1.0
        - float(baseline_s2_latent["training_seconds"])
        / float(baseline_s2_original["training_seconds"])
    ) * 100

    scenario_display = summary[
        [
            "scenario_label",
            "accuracy",
            "macro_precision",
            "macro_recall",
            "macro_f1",
            "training_seconds",
        ]
    ].rename(
        columns={
            "scenario_label": "Scenario",
            "accuracy": "Accuracy",
            "macro_precision": "Macro precision",
            "macro_recall": "Macro recall",
            "macro_f1": "Macro F1",
            "training_seconds": "Training seconds",
        }
    )
    baseline_display = baseline[
        [
            "representation",
            "scenario",
            "accuracy",
            "macro_recall",
            "macro_f1",
            "training_seconds",
        ]
    ].rename(
        columns={
            "representation": "Representation",
            "scenario": "Scenario",
            "accuracy": "Accuracy",
            "macro_recall": "Macro recall",
            "macro_f1": "Macro F1",
            "training_seconds": "Classifier seconds",
        }
    )

    lines = [
        "# Chapter IV Draft - Results and Discussion",
        "",
        "> **Data provenance.** Every primary value in this draft is generated from the "
        "full-data artifacts in `results/metrics`. Quick-run artifacts are explicitly rejected "
        "by `src/reporting.py` and must not be reported as thesis results.",
        "",
        "## 4.1 Experimental Data and Preprocessing",
        "",
        f"The raw dataset contained {preprocessing['total_rows']:,} rows. Preprocessing removed "
        f"{preprocessing['invalid_rows_dropped']:,} rows with invalid numeric values, leaving "
        f"{preprocessing['valid_rows']:,} valid observations and "
        f"{preprocessing['feature_count']} model features. The stratified 80:20 split produced "
        f"{preprocessing['train_rows']:,} training rows and {preprocessing['test_rows']:,} test "
        f"rows. The maximum class-percentage difference between the two splits was "
        f"{preprocessing['max_split_percentage_delta']:.6f} percentage points.",
        "",
        "The scaler was fitted only on training rows (`scaler_fit_scope = train_only`) and then "
        "applied unchanged to both splits. Resampling and class weighting were applied only to "
        "training artifacts, while the classifier integrity report verified that test artifacts "
        "remained unchanged. These controls support H1 methodologically: they demonstrate the "
        "intended ordering and reduce leakage risk, but they do not claim that leakage prevention "
        "is itself an empirical performance gain.",
        "",
        "See [Table 4.1](tables/table_4_1_dataset_pipeline.csv).",
        "",
        "## 4.2 Autoencoder Training and Latent Representation",
        "",
        f"The Autoencoder reduced {reconstruction['input_dim']} normalized features to "
        f"{reconstruction['latent_dim']} latent features, a {dimension_reduction:.2f}% dimension "
        f"reduction. The selected checkpoint was epoch {reconstruction['best_epoch']} with "
        f"validation MSE {reconstruction['best_val_loss']:.8f}. Full-array reconstruction MSE was "
        f"{reconstruction['train_reconstruction_mse']:.8f} on train and "
        f"{reconstruction['test_reconstruction_mse']:.8f} on test, a relative gap of "
        f"{test_gap:.2f}%. The small gap indicates no coarse reconstruction overfitting.",
        "",
        "![Autoencoder loss](figures/figure_4_1_autoencoder_loss.png)",
        "",
        "## 4.3 Imbalance-Handling Scenario Results",
        "",
        markdown_table(scenario_display),
        "",
        f"S1 achieved the highest accuracy ({s1['accuracy']:.4f}) but only "
        f"{s1['macro_f1']:.4f} macro F1. S2 achieved the best macro recall "
        f"({s2['macro_recall']:.4f}) and macro F1 ({s2['macro_f1']:.4f}). S3 was nearly "
        f"equivalent to S2, differing by only {s2_s3_f1_delta:.4f} macro F1 while requiring "
        f"{s3_time_ratio:.2f} times the classifier training time. This similarity is consistent "
        "with random duplication and class weighting producing equivalent relative class "
        "contributions for tree training. S4 produced the lowest aggregate macro F1 "
        f"({s4['macro_f1']:.4f}) after discarding nearly all majority-class training rows.",
        "",
        "![Scenario metrics](figures/figure_4_2_scenario_metrics.png)",
        "",
        "## 4.4 Per-Class Effects and Confusion Patterns",
        "",
        "The imbalance strategies improved several minority classes but did not improve every "
        "class. Backdoor and Ransomware benefited strongly under S2/S3, while DoS and DDoS "
        "remained unreliable in all four scenarios. The complete focus-class values are provided "
        "in [Table 4.4](tables/table_4_4_minority_class_metrics.csv).",
        "",
        f"The trade-off was substantial for XSS: recall fell from {xss_s1['recall']:.4f} under "
        f"S1 to {xss_s2['recall']:.4f} under S2. Therefore, S2 is the preferred scenario under "
        "the pre-specified macro-metric objective, not a universal winner for every class or "
        "operating requirement.",
        "",
        "![Minority F1](figures/figure_4_3_minority_f1.png)",
        "",
        "![S2 confusion matrix](figures/figure_4_4_s2_confusion_matrix.png)",
        "",
        "## 4.5 Optional Original-Feature Baseline and H2",
        "",
        markdown_table(baseline_display),
        "",
        f"Under S2, the original 69-feature baseline improved macro F1 by "
        f"{baseline_f1_delta:.4f} ({baseline_s2_latent['macro_f1']:.4f} to "
        f"{baseline_s2_original['macro_f1']:.4f}) and improved macro recall from "
        f"{baseline_s2_latent['macro_recall']:.4f} to "
        f"{baseline_s2_original['macro_recall']:.4f}. In contrast, the latent representation "
        f"reduced LightGBM training time by {classifier_time_reduction:.1f}% for S2. This timing "
        "comparison covers the classifier only and must not be interpreted as an end-to-end "
        "runtime advantage because Autoencoder training is an additional cost.",
        "",
        "H2 is therefore partially supported. The Autoencoder clearly provides a compact "
        "representation from which LightGBM can distinguish several classes, but the controlled "
        "baseline shows that the 16-dimensional representation does not preserve all useful "
        "discriminative information. It should not be claimed that the Autoencoder improves "
        "classification performance over the original features.",
        "",
        f"For DoS and DDoS, the original-feature baseline did not improve the best F1 for either "
        f"class. This supports the diagnostic conclusion that their failure is not primarily an "
        f"Autoencoder artifact: {baseline_diagnostic['diagnosis']}",
        "",
        "![Representation baseline](figures/figure_4_5_representation_baseline.png)",
        "",
        "## 4.6 Hypothesis Assessment",
        "",
        markdown_table(hypothesis_table),
        "",
        "H3 is supported because imbalance handling materially changed macro recall, macro F1, "
        "and class-level confusion. The contrast between S1 accuracy and S2 macro performance "
        "also confirms that accuracy alone is insufficient for this dataset.",
        "",
        "## 4.7 Supplementary Diagnostics",
        "",
    ]

    dataset_defect = optional_results.get("dataset_defect")
    if dataset_defect:
        lines.extend(
            [
                f"A post-hoc data-quality audit found {dataset_defect['conflict_groups']} "
                "byte-identical DoS/DDoS feature groups with contradictory labels. All "
                f"{dataset_defect['class_totals']['DoS']} DoS rows were involved, together with "
                f"{dataset_defect['conflicted_fraction_per_class']['DDoS'] * 100:.2f}% of DDoS "
                "rows. This label contradiction provides a direct explanation for persistent "
                "mutual confusion and must be stated as a dataset limitation.",
                "",
            ]
        )

    ae_variants = optional_results.get("ae_variants")
    if ae_variants:
        best_variant = max(ae_variants["variants"], key=lambda item: item["macro_f1"])
        lines.extend(
            [
                f"The exploratory Autoencoder sensitivity study reached its best macro F1 "
                f"({best_variant['macro_f1']:.4f}) with {best_variant['variant']}. This remains "
                "below the original-69 S2 baseline and is supplementary rather than a replacement "
                "for the pre-registered latent-16 pipeline.",
                "",
            ]
        )

    tuning = optional_results.get("tuning")
    if tuning:
        latent_results = [
            item for item in tuning["results"] if item["representation"] == "latent16_proposal"
        ]
        best_latent_tuning = max(latent_results, key=lambda item: item["macro_f1"])
        lines.extend(
            [
                f"Within the proposal's LightGBM sensitivity grid, the strongest latent-16 point "
                f"was learning_rate={best_latent_tuning['learning_rate']} and "
                f"n_estimators={best_latent_tuning['n_estimators']} with macro F1 "
                f"{best_latent_tuning['macro_f1']:.4f}. Because these sensitivity values were "
                "evaluated on the held-out test set, they are descriptive and must not be used as "
                "post-hoc test-set model selection.",
                "",
            ]
        )

    merged = optional_results.get("merged9")
    if merged:
        merged_s2 = merged["runs"]["s2_class_weight"]
        lines.extend(
            [
                "A separate nine-class diagnostic merged DoS and DDoS to remove the contradictory "
                f"decision boundary. Its S2 macro F1 was {merged_s2['macro_f1']:.4f}. This value "
                "is not directly comparable with the primary ten-class metrics and must remain a "
                "supplementary dataset-correction experiment.",
                "",
            ]
        )

    lines.extend(
        [
            "## 4.8 Limitations",
            "",
            "- DoS and DDoS have extremely small support and contradictory duplicate labels.",
            "- S3 uses random duplication, so its effective class contribution is mathematically "
            "close to S2 while requiring substantially more storage and training time.",
            "- S4 removes 99.97% of training rows and is intentionally an extreme comparison.",
            "- The original-feature baseline improves predictive metrics, so dimensionality "
            "reduction should be justified by compactness and classifier cost, not accuracy gain.",
            "- Exploratory variant and tuning results are sensitivity analyses, not confirmation "
            "from an independent second test set.",
            "- Quick-run metrics are smoke-test outputs and are excluded from every table and "
            "conclusion in this chapter.",
            "",
            "## 4.9 Chapter Conclusion",
            "",
            "The experiment supports the leakage-aware preprocessing procedure in H1 and the "
            "imbalance-sensitivity claim in H3. H2 is partially supported: latent-16 is compact "
            "and useful, but the optional original-69 baseline is stronger under the primary "
            "macro metrics. S2 class weighting is the preferred ten-class latent scenario for "
            "macro recall and macro F1, while S1 remains preferable only when aggregate accuracy "
            "is prioritized over balanced class performance.",
            "",
            "## Generated Materials",
            "",
            "- `tables/`: CSV tables for direct import into the thesis source.",
            "- `figures/`: full-data PNG figures at 180 DPI.",
            "- `report_manifest.json`: machine-readable provenance and hypothesis conclusions.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_phase9_report(
    config_path: str | Path = "configs/config.yaml",
    output_dir: str | Path = "reports/phase9",
) -> dict[str, Any]:
    """Generate the canonical Phase 9 report package from full-data artifacts."""
    project_root = find_project_root()
    config = load_config(config_path)
    metrics_dir = resolve_project_path(config["paths"]["metrics_dir"], project_root)
    report_dir = resolve_project_path(output_dir, project_root)
    tables_dir = report_dir / "tables"
    figures_dir = report_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    scenarios = list(config["imbalance"]["scenarios"])
    required_paths = [
        metrics_dir / "preprocessing_report.json",
        metrics_dir / "autoencoder_reconstruction_error.json",
        metrics_dir / "autoencoder_history.csv",
        metrics_dir / "summary_comparison.csv",
        metrics_dir / "per_class_comparison.csv",
        metrics_dir / "minority_class_comparison.csv",
        metrics_dir / "baseline_original_vs_latent_summary.csv",
        metrics_dir / "baseline_dos_ddos_diagnostic.json",
        metrics_dir / "classifier_test_integrity.json",
        *[metrics_dir / f"report_{scenario}.json" for scenario in scenarios],
        *[metrics_dir / f"imbalance_report_{scenario}.json" for scenario in scenarios],
    ]
    require_paths(required_paths)

    preprocessing = load_json(metrics_dir / "preprocessing_report.json")
    reconstruction = load_json(metrics_dir / "autoencoder_reconstruction_error.json")
    history = pd.read_csv(metrics_dir / "autoencoder_history.csv")
    summary = pd.read_csv(metrics_dir / "summary_comparison.csv")
    per_class = pd.read_csv(metrics_dir / "per_class_comparison.csv")
    minority = pd.read_csv(metrics_dir / "minority_class_comparison.csv")
    baseline = pd.read_csv(metrics_dir / "baseline_original_vs_latent_summary.csv")
    baseline_diagnostic = load_json(metrics_dir / "baseline_dos_ddos_diagnostic.json")
    reports = {
        scenario: load_json(metrics_dir / f"report_{scenario}.json")
        for scenario in scenarios
    }
    imbalance_reports = {
        scenario: load_json(metrics_dir / f"imbalance_report_{scenario}.json")
        for scenario in scenarios
    }
    classifier_integrity = load_json(metrics_dir / "classifier_test_integrity.json")
    validate_full_data_inputs(
        config,
        metrics_dir,
        summary,
        reports,
        preprocessing,
        baseline,
        imbalance_reports,
        classifier_integrity,
    )

    summary["scenario_label"] = summary["scenario"].map(SCENARIO_LABELS)
    minority["scenario_label"] = minority["scenario"].map(SCENARIO_LABELS)
    dataset_table = pd.DataFrame(
        [
            {
                "raw_rows": preprocessing["total_rows"],
                "valid_rows": preprocessing["valid_rows"],
                "invalid_rows_dropped": preprocessing["invalid_rows_dropped"],
                "train_rows": preprocessing["train_rows"],
                "test_rows": preprocessing["test_rows"],
                "feature_count": preprocessing["feature_count"],
                "split_strategy": "80:20 stratified",
                "scaler_fit_scope": preprocessing["scaler_fit_scope"],
            }
        ]
    )
    reconstruction_table = pd.DataFrame(
        [
            {
                "input_dim": reconstruction["input_dim"],
                "latent_dim": reconstruction["latent_dim"],
                "best_epoch": reconstruction["best_epoch"],
                "best_validation_mse": reconstruction["best_val_loss"],
                "train_reconstruction_mse": reconstruction["train_reconstruction_mse"],
                "test_reconstruction_mse": reconstruction["test_reconstruction_mse"],
            }
        ]
    )
    hypothesis_table = pd.DataFrame(
        [
            {
                "hypothesis": "H1",
                "assessment": "Supported methodologically",
                "evidence": (
                    "Split precedes normalization; scaler scope is train_only; imbalance handling "
                    "uses training artifacts; test integrity hashes remain unchanged."
                ),
            },
            {
                "hypothesis": "H2",
                "assessment": "Partially supported",
                "evidence": (
                    "Latent-16 reduces dimension by 76.81% and remains classifiable, but "
                    "original-69 "
                    "S2 macro F1 is 0.5406 versus 0.4249 for latent-16."
                ),
            },
            {
                "hypothesis": "H3",
                "assessment": "Supported",
                "evidence": (
                    "S1 has the best accuracy, while S2 has the best macro recall/F1; per-class "
                    "recall and confusion change materially across scenarios."
                ),
            },
        ]
    )

    table_outputs = {
        "dataset_pipeline": tables_dir / "table_4_1_dataset_pipeline.csv",
        "autoencoder_reconstruction": tables_dir / "table_4_2_autoencoder_reconstruction.csv",
        "scenario_metrics": tables_dir / "table_4_3_scenario_metrics.csv",
        "minority_metrics": tables_dir / "table_4_4_minority_class_metrics.csv",
        "baseline": tables_dir / "table_4_5_representation_baseline.csv",
        "hypotheses": tables_dir / "table_4_6_hypothesis_assessment.csv",
    }
    dataset_table.to_csv(table_outputs["dataset_pipeline"], index=False)
    reconstruction_table.to_csv(table_outputs["autoencoder_reconstruction"], index=False)
    summary.to_csv(table_outputs["scenario_metrics"], index=False, float_format="%.12g")
    minority.to_csv(table_outputs["minority_metrics"], index=False, float_format="%.12g")
    baseline.to_csv(table_outputs["baseline"], index=False, float_format="%.12g")
    hypothesis_table.to_csv(table_outputs["hypotheses"], index=False)

    figure_outputs = {
        "autoencoder_loss": figures_dir / "figure_4_1_autoencoder_loss.png",
        "scenario_metrics": figures_dir / "figure_4_2_scenario_metrics.png",
        "minority_f1": figures_dir / "figure_4_3_minority_f1.png",
        "s2_confusion": figures_dir / "figure_4_4_s2_confusion_matrix.png",
        "representation_baseline": figures_dir / "figure_4_5_representation_baseline.png",
    }
    plot_autoencoder_loss(history, figure_outputs["autoencoder_loss"])
    plot_scenario_metrics(summary, figure_outputs["scenario_metrics"])
    plot_minority_f1(minority, figure_outputs["minority_f1"])
    plot_s2_confusion(reports["s2_class_weight"], figure_outputs["s2_confusion"])
    plot_representation_baseline(baseline, figure_outputs["representation_baseline"])

    optional_paths = {
        "dataset_defect": metrics_dir / "dataset_label_conflicts.json",
        "ae_variants": metrics_dir / "ae_variant_study.json",
        "tuning": metrics_dir / "lightgbm_tuning_in_proposal.json",
        "merged9": metrics_dir / "merged9_experiment.json",
    }
    optional_results = {
        name: load_json(path)
        for name, path in optional_paths.items()
        if path.exists()
    }
    draft_path = report_dir / "chapter4_results_and_discussion.md"
    write_chapter_draft(
        draft_path,
        preprocessing,
        reconstruction,
        summary,
        minority,
        per_class,
        baseline,
        baseline_diagnostic,
        hypothesis_table,
        optional_results,
    )

    manifest = {
        "full_data_only": True,
        "quick_run_artifacts_excluded": True,
        "valid_rows": int(preprocessing["valid_rows"]),
        "train_rows": int(preprocessing["train_rows"]),
        "test_rows": int(preprocessing["test_rows"]),
        "feature_count": int(preprocessing["feature_count"]),
        "latent_dim": int(reconstruction["latent_dim"]),
        "scenarios": scenarios,
        "hypothesis_assessments": {
            row["hypothesis"]: row["assessment"]
            for row in hypothesis_table.to_dict(orient="records")
        },
        "optional_baseline_included": True,
        "test_artifacts_unchanged": True,
        "supplementary_diagnostics_included": sorted(optional_results),
        "draft_path": draft_path.relative_to(project_root).as_posix(),
        "tables": {
            name: path.relative_to(project_root).as_posix()
            for name, path in table_outputs.items()
        },
        "figures": {
            name: path.relative_to(project_root).as_posix()
            for name, path in figure_outputs.items()
        },
    }
    manifest_path = report_dir / "report_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)

    return {
        "draft_path": str(draft_path),
        "manifest_path": str(manifest_path),
        "table_paths": {name: str(path) for name, path in table_outputs.items()},
        "figure_paths": {name: str(path) for name, path in figure_outputs.items()},
        "summary": summary,
        "hypothesis_table": hypothesis_table,
        "quick_run_artifacts_excluded": True,
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for Phase 9 reporting."""
    parser = argparse.ArgumentParser(description="Generate full-data Chapter IV report materials.")
    parser.add_argument("--config", default="configs/config.yaml", help="YAML config path.")
    parser.add_argument("--output-dir", default="reports/phase9", help="Report output directory.")
    return parser.parse_args()


def main() -> None:
    """Generate Phase 9 report materials from the command line."""
    args = parse_args()
    result = generate_phase9_report(args.config, args.output_dir)
    print("Phase 9 report materials generated from full-data artifacts.")
    print(f"Draft: {result['draft_path']}")
    print(f"Manifest: {result['manifest_path']}")


if __name__ == "__main__":
    main()
