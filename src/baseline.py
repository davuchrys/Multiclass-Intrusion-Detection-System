"""Original-feature LightGBM baseline for diagnosing Autoencoder information loss."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

from src.classifier import (
    load_reusable_report,
    predict,
    source_signature,
    train_lightgbm,
    validate_prediction_file,
)
from src.data_loading import find_project_root, load_config, resolve_project_path
from src.evaluation import (
    file_sha256,
    macro_metrics,
    plot_confusion_matrices,
    save_confusion_csv,
    validate_labels,
)


BASELINE_SCENARIOS = ["s1_none", "s2_class_weight"]
FOCUS_CLASS_NAMES = ["DoS", "DDoS", "XSS"]


def configured_baseline_scenarios(config: dict[str, Any]) -> list[str]:
    """Validate and return the configured diagnostic baseline scenarios."""
    scenarios = list(config.get("baseline", {}).get("scenarios", BASELINE_SCENARIOS))
    if set(scenarios) != set(BASELINE_SCENARIOS) or len(scenarios) != len(BASELINE_SCENARIOS):
        raise ValueError(
            "The diagnostic baseline must contain exactly s1_none and s2_class_weight."
        )
    return scenarios


def load_phase5_report(metrics_dir: Path, scenario: str) -> dict[str, Any]:
    """Load a Phase 5 report to reuse the canonical S2 sample weights."""
    path = metrics_dir / f"imbalance_report_{scenario}.json"
    if not path.exists():
        raise FileNotFoundError(f"Phase 5 report not found: {path}")
    with open(path, encoding="utf-8") as file:
        report = json.load(file)
    if report.get("scenario") != scenario:
        raise ValueError(f"Phase 5 scenario mismatch in {path}.")
    return report


def train_original_feature_baseline(
    scenario: str,
    config_path: str | Path = "configs/config.yaml",
    force: bool = False,
) -> dict[str, Any]:
    """Train one LightGBM baseline directly on the normalized 69-feature arrays."""
    if scenario not in BASELINE_SCENARIOS:
        raise ValueError(f"Original-feature baseline scenario is not supported: {scenario}")

    project_root = find_project_root()
    config = load_config(config_path)
    configured_baseline_scenarios(config)
    processed_dir = resolve_project_path(config["paths"]["processed_dir"], project_root)
    metrics_dir = resolve_project_path(config["paths"]["metrics_dir"], project_root)
    models_dir = resolve_project_path(config["paths"]["models_dir"], project_root)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    x_train_path = processed_dir / "X_train.npy"
    x_test_path = processed_dir / "X_test.npy"
    y_train_path = processed_dir / "y_train.npy"
    sample_weight_path: Path | None = None
    if scenario == "s2_class_weight":
        phase5_report = load_phase5_report(metrics_dir, scenario)
        sample_weight_path = Path(phase5_report["sample_weight_path"])

    required_paths = [x_train_path, x_test_path, y_train_path]
    if sample_weight_path is not None:
        required_paths.append(sample_weight_path)
    missing_paths = [path for path in required_paths if not path.exists()]
    if missing_paths:
        raise FileNotFoundError(f"Baseline artifacts are missing: {missing_paths}")

    model_config = dict(config["lightgbm"])
    prediction_batch_size = int(model_config.pop("prediction_batch_size", 250_000))
    expected_features = int(config["data"]["expected_feature_count"])
    num_classes = int(model_config["num_class"])
    model_path = models_dir / f"lgbm_original_69_{scenario}.txt"
    prediction_path = processed_dir / f"y_pred_original_69_{scenario}.npy"
    report_path = metrics_dir / f"baseline_classifier_report_{scenario}.json"
    signature = source_signature(
        f"original_69_{scenario}",
        model_config,
        x_train_path,
        y_train_path,
        sample_weight_path,
    )

    x_test = np.load(x_test_path, mmap_mode="r")
    if x_test.ndim != 2 or x_test.shape[1] != expected_features:
        raise ValueError("X_test does not contain the configured 69 original features.")
    if not force:
        reusable = load_reusable_report(
            report_path,
            model_path,
            prediction_path,
            signature,
            x_test.shape[0],
            num_classes,
        )
        if reusable is not None:
            return reusable

    x_train = np.load(x_train_path, mmap_mode="r")
    y_train = np.load(y_train_path, mmap_mode="r")
    sample_weight = (
        np.load(sample_weight_path, mmap_mode="r") if sample_weight_path is not None else None
    )
    if x_train.shape != (y_train.shape[0], expected_features):
        raise ValueError("Original-feature training arrays have inconsistent shapes.")

    training_start = time.perf_counter()
    model = train_lightgbm(
        x_train,
        y_train,
        f"original_69_{scenario}",
        sample_weight=sample_weight,
        model_config=model_config,
    )
    training_seconds = time.perf_counter() - training_start
    model.booster_.save_model(str(model_path))

    prediction_start = time.perf_counter()
    predict(
        model,
        x_test,
        prediction_path,
        prediction_batch_size,
        y_train.dtype,
    )
    prediction_seconds = time.perf_counter() - prediction_start
    predicted_classes = validate_prediction_file(
        prediction_path,
        x_test.shape[0],
        num_classes,
    )

    report = {
        "scenario": scenario,
        "representation": "original_69",
        "model_path": str(model_path),
        "prediction_path": str(prediction_path),
        "x_train_path": str(x_train_path),
        "y_train_path": str(y_train_path),
        "sample_weight_path": str(sample_weight_path) if sample_weight_path else None,
        "training_rows": int(x_train.shape[0]),
        "test_rows": int(x_test.shape[0]),
        "feature_count": expected_features,
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
        "source_signature": signature,
        "reused_existing_artifacts": False,
    }
    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
    return report


def train_original_feature_baselines(
    config_path: str | Path = "configs/config.yaml",
    force: bool = False,
) -> dict[str, dict[str, Any]]:
    """Train S1 and S2 original-feature baselines and protect test artifacts."""
    project_root = find_project_root()
    config = load_config(config_path)
    scenarios = configured_baseline_scenarios(config)
    processed_dir = resolve_project_path(config["paths"]["processed_dir"], project_root)
    metrics_dir = resolve_project_path(config["paths"]["metrics_dir"], project_root)
    test_paths = {
        "X_test": processed_dir / "X_test.npy",
        "y_test": processed_dir / "y_test.npy",
    }
    before_hashes = {name: file_sha256(path) for name, path in test_paths.items()}
    reports = {
        scenario: train_original_feature_baseline(scenario, config_path, force)
        for scenario in scenarios
    }
    after_hashes = {name: file_sha256(path) for name, path in test_paths.items()}
    if before_hashes != after_hashes:
        raise ValueError("Original-feature baseline modified test artifacts.")

    integrity = {
        "test_hashes_before": before_hashes,
        "test_hashes_after": after_hashes,
        "test_artifacts_unchanged": True,
        "scenarios": scenarios,
    }
    with open(metrics_dir / "baseline_test_integrity.json", "w", encoding="utf-8") as file:
        json.dump(integrity, file, indent=2)
    return reports


def evaluate_original_feature_baseline(
    scenario: str,
    config_path: str | Path = "configs/config.yaml",
) -> dict[str, Any]:
    """Evaluate one saved original-feature baseline prediction."""
    if scenario not in BASELINE_SCENARIOS:
        raise ValueError(f"Original-feature baseline scenario is not supported: {scenario}")

    project_root = find_project_root()
    config = load_config(config_path)
    configured_baseline_scenarios(config)
    processed_dir = resolve_project_path(config["paths"]["processed_dir"], project_root)
    metrics_dir = resolve_project_path(config["paths"]["metrics_dir"], project_root)
    figures_dir = resolve_project_path(config["paths"]["figures_dir"], project_root)
    classifier_path = metrics_dir / f"baseline_classifier_report_{scenario}.json"
    if not classifier_path.exists():
        raise FileNotFoundError(f"Baseline classifier report not found: {classifier_path}")
    with open(classifier_path, encoding="utf-8") as file:
        classifier_data = json.load(file)

    y_test_path = processed_dir / "y_test.npy"
    prediction_path = Path(classifier_data["prediction_path"])
    y_true = np.load(y_test_path, mmap_mode="r")
    y_pred = np.load(prediction_path, mmap_mode="r")
    num_classes = int(config["data"]["expected_num_classes"])
    class_names = [
        name
        for name, _ in sorted(
            config["data"]["class_mapping"].items(),
            key=lambda item: int(item[1]),
        )
    ]
    labels = list(range(num_classes))
    validate_labels(y_true, y_pred, num_classes, f"original_69_{scenario}")
    aggregate = macro_metrics(y_true, y_pred)
    per_class = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred, labels=labels)

    confusion_csv_path = metrics_dir / f"confusion_matrix_original_69_{scenario}.csv"
    confusion_figure_path = figures_dir / f"confusion_matrix_original_69_{scenario}.png"
    save_confusion_csv(matrix, class_names, confusion_csv_path)
    plot_confusion_matrices(
        matrix,
        class_names,
        f"original_69_{scenario}",
        confusion_figure_path,
    )
    report = {
        "scenario": scenario,
        "representation": "original_69",
        "test_rows": int(y_true.shape[0]),
        "class_names": class_names,
        "aggregate_metrics": aggregate,
        "classification_report": per_class,
        "confusion_matrix": matrix.astype(int).tolist(),
        "confusion_matrix_csv": str(confusion_csv_path),
        "confusion_matrix_figure": str(confusion_figure_path),
        "prediction_path": str(prediction_path),
        "training_seconds": float(classifier_data["training_seconds"]),
        "prediction_seconds": float(classifier_data["prediction_seconds"]),
        "model_iterations": int(classifier_data["model_iterations"]),
    }
    report_path = metrics_dir / f"baseline_report_original_69_{scenario}.json"
    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
    report["report_path"] = str(report_path)
    return report


def load_latent_evaluation(metrics_dir: Path, scenario: str) -> dict[str, Any]:
    """Load the canonical Phase 7 latent-feature evaluation report."""
    path = metrics_dir / f"report_{scenario}.json"
    if not path.exists():
        raise FileNotFoundError(f"Latent evaluation report not found: {path}")
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def plot_baseline_comparison(
    aggregate: pd.DataFrame,
    focus: pd.DataFrame,
    output_path: Path,
) -> Path:
    """Plot aggregate scores and DoS/DDoS class F1 across representations."""
    figure, axes = plt.subplots(1, 2, figsize=(17, 6))
    aggregate_plot = aggregate.melt(
        id_vars=["representation", "scenario"],
        value_vars=["accuracy", "macro_recall", "macro_f1"],
        var_name="metric",
        value_name="score",
    )
    aggregate_plot["model"] = (
        aggregate_plot["representation"] + " / " + aggregate_plot["scenario"]
    )
    sns.barplot(data=aggregate_plot, x="model", y="score", hue="metric", ax=axes[0])
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_title("Aggregate Metrics")
    axes[0].set_xlabel("Representation / scenario")
    axes[0].tick_params(axis="x", rotation=30)
    axes[0].grid(axis="y", alpha=0.25)

    attack_focus = focus[focus["class_name"].isin(["DoS", "DDoS"])].copy()
    attack_focus["model"] = attack_focus["representation"] + " / " + attack_focus["scenario"]
    sns.barplot(data=attack_focus, x="model", y="f1_score", hue="class_name", ax=axes[1])
    axes[1].set_ylim(0.0, max(0.1, float(attack_focus["f1_score"].max()) * 1.15))
    axes[1].set_title("DoS and DDoS F1-Score")
    axes[1].set_xlabel("Representation / scenario")
    axes[1].tick_params(axis="x", rotation=30)
    axes[1].grid(axis="y", alpha=0.25)

    figure.suptitle("Original 69 Features vs Autoencoder Latent 16 Features")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(figure)
    return output_path


def compare_original_and_latent_baselines(
    config_path: str | Path = "configs/config.yaml",
) -> dict[str, Any]:
    """Compare S1/S2 original features against their latent-feature counterparts."""
    project_root = find_project_root()
    config = load_config(config_path)
    scenarios = configured_baseline_scenarios(config)
    metrics_dir = resolve_project_path(config["paths"]["metrics_dir"], project_root)
    figures_dir = resolve_project_path(config["paths"]["figures_dir"], project_root)
    baseline_reports = {
        scenario: evaluate_original_feature_baseline(scenario, config_path)
        for scenario in scenarios
    }
    latent_reports = {
        scenario: load_latent_evaluation(metrics_dir, scenario)
        for scenario in scenarios
    }

    aggregate_rows = []
    focus_rows = []
    for representation, reports in (
        ("latent_16", latent_reports),
        ("original_69", baseline_reports),
    ):
        for scenario, report in reports.items():
            aggregate_rows.append(
                {
                    "representation": representation,
                    "scenario": scenario,
                    **report["aggregate_metrics"],
                    "training_seconds": report["training_seconds"],
                }
            )
            matrix = np.asarray(report["confusion_matrix"], dtype=np.int64)
            class_names = report["class_names"]
            for class_name in FOCUS_CLASS_NAMES:
                class_index = class_names.index(class_name)
                class_metrics = report["classification_report"][class_name]
                paired_class = "DDoS" if class_name == "DoS" else "DoS"
                paired_index = class_names.index(paired_class) if class_name != "XSS" else None
                focus_rows.append(
                    {
                        "representation": representation,
                        "scenario": scenario,
                        "class_name": class_name,
                        "precision": float(class_metrics["precision"]),
                        "recall": float(class_metrics["recall"]),
                        "f1_score": float(class_metrics["f1-score"]),
                        "support": int(class_metrics["support"]),
                        "paired_confusion_count": (
                            int(matrix[class_index, paired_index])
                            if paired_index is not None
                            else None
                        ),
                        "paired_confusion_rate": (
                            float(matrix[class_index, paired_index] / matrix[class_index].sum())
                            if paired_index is not None
                            else None
                        ),
                    }
                )

    aggregate = pd.DataFrame(aggregate_rows)
    focus = pd.DataFrame(focus_rows)
    aggregate_path = metrics_dir / "baseline_original_vs_latent_summary.csv"
    focus_path = metrics_dir / "baseline_original_vs_latent_focus_classes.csv"
    aggregate.to_csv(aggregate_path, index=False)
    focus.to_csv(focus_path, index=False)

    dos_ddos = focus[focus["class_name"].isin(["DoS", "DDoS"])]
    gain_rows = []
    for scenario in scenarios:
        for class_name in ["DoS", "DDoS"]:
            latent = dos_ddos[
                (dos_ddos["representation"] == "latent_16")
                & (dos_ddos["scenario"] == scenario)
                & (dos_ddos["class_name"] == class_name)
            ].iloc[0]
            original = dos_ddos[
                (dos_ddos["representation"] == "original_69")
                & (dos_ddos["scenario"] == scenario)
                & (dos_ddos["class_name"] == class_name)
            ].iloc[0]
            gain_rows.append(
                {
                    "scenario": scenario,
                    "class_name": class_name,
                    "latent_recall": float(latent["recall"]),
                    "original_recall": float(original["recall"]),
                    "recall_delta_original_minus_latent": float(
                        original["recall"] - latent["recall"]
                    ),
                    "latent_f1": float(latent["f1_score"]),
                    "original_f1": float(original["f1_score"]),
                    "f1_delta_original_minus_latent": float(
                        original["f1_score"] - latent["f1_score"]
                    ),
                    "latent_paired_confusion_rate": float(latent["paired_confusion_rate"]),
                    "original_paired_confusion_rate": float(original["paired_confusion_rate"]),
                }
            )

    gain_frame = pd.DataFrame(gain_rows)
    best_f1_comparison = []
    for class_name in ["DoS", "DDoS"]:
        class_rows = dos_ddos[dos_ddos["class_name"] == class_name]
        latent_best = float(
            class_rows[class_rows["representation"] == "latent_16"]["f1_score"].max()
        )
        original_best = float(
            class_rows[class_rows["representation"] == "original_69"]["f1_score"].max()
        )
        best_f1_comparison.append(
            {
                "class_name": class_name,
                "best_latent_f1": latent_best,
                "best_original_f1": original_best,
                "best_f1_delta_original_minus_latent": original_best - latent_best,
            }
        )

    best_f1_gains = np.asarray(
        [row["best_f1_delta_original_minus_latent"] for row in best_f1_comparison]
    )
    latent_s2 = aggregate[
        (aggregate["representation"] == "latent_16")
        & (aggregate["scenario"] == "s2_class_weight")
    ].iloc[0]
    original_s2 = aggregate[
        (aggregate["representation"] == "original_69")
        & (aggregate["scenario"] == "s2_class_weight")
    ].iloc[0]
    xss_s2 = focus[
        (focus["scenario"] == "s2_class_weight") & (focus["class_name"] == "XSS")
    ]
    latent_xss_s2 = xss_s2[xss_s2["representation"] == "latent_16"].iloc[0]
    original_xss_s2 = xss_s2[xss_s2["representation"] == "original_69"].iloc[0]

    if np.all(best_f1_gains > 0.0):
        diagnosis = (
            "The best DoS and DDoS F1-scores both improve when LightGBM uses the original 69 "
            "features. This is consistent with the Autoencoder discarding some class-separating "
            "information, although the very small class supports remain a confounding factor."
        )
    elif np.all(best_f1_gains <= 0.0):
        diagnosis = (
            f"The original 69 features improve S2 macro F1 from {latent_s2['macro_f1']:.4f} "
            f"to {original_s2['macro_f1']:.4f}, but they do not improve the best attainable "
            "F1-score for either DoS or DDoS. Their mutual confusion is therefore more "
            "consistent with intrinsic flow similarity and very small class supports than "
            "with Autoencoder information loss as the primary cause."
        )
    else:
        diagnosis = (
            "The original-feature baseline produces mixed DoS/DDoS changes across scenarios. "
            "The evidence does not isolate Autoencoder information loss as the sole cause; "
            "flow similarity and very small class supports remain plausible explanations."
        )

    diagnostic = {
        "comparison_scope": scenarios,
        "focus_classes": ["DoS", "DDoS", "XSS"],
        "dos_ddos_deltas": gain_rows,
        "dos_ddos_best_f1_by_representation": best_f1_comparison,
        "aggregate_s2_effect": {
            "latent_macro_f1": float(latent_s2["macro_f1"]),
            "original_macro_f1": float(original_s2["macro_f1"]),
            "macro_f1_delta_original_minus_latent": float(
                original_s2["macro_f1"] - latent_s2["macro_f1"]
            ),
            "latent_macro_recall": float(latent_s2["macro_recall"]),
            "original_macro_recall": float(original_s2["macro_recall"]),
        },
        "xss_s2_effect": {
            "latent_recall": float(latent_xss_s2["recall"]),
            "original_recall": float(original_xss_s2["recall"]),
            "recall_delta_original_minus_latent": float(
                original_xss_s2["recall"] - latent_xss_s2["recall"]
            ),
            "latent_f1": float(latent_xss_s2["f1_score"]),
            "original_f1": float(original_xss_s2["f1_score"]),
        },
        "diagnosis": diagnosis,
        "causal_limit": (
            "This controlled baseline changes the feature representation while keeping the "
            "split, LightGBM configuration, and S1/S2 handling fixed. It supports, but cannot "
            "alone prove, a causal explanation because DoS and DDoS supports remain extremely small."
        ),
    }
    diagnostic_path = metrics_dir / "baseline_dos_ddos_diagnostic.json"
    with open(diagnostic_path, "w", encoding="utf-8") as file:
        json.dump(diagnostic, file, indent=2)

    figure_path = figures_dir / "baseline_original_vs_latent.png"
    plot_baseline_comparison(aggregate, focus, figure_path)
    markdown_path = metrics_dir / "baseline_original_vs_latent_analysis.md"
    write_baseline_markdown(aggregate, gain_frame, diagnostic, markdown_path)
    return {
        "baseline_reports": baseline_reports,
        "latent_reports": latent_reports,
        "aggregate_comparison": aggregate,
        "focus_comparison": focus,
        "diagnostic": diagnostic,
        "aggregate_path": str(aggregate_path),
        "focus_path": str(focus_path),
        "diagnostic_path": str(diagnostic_path),
        "figure_path": str(figure_path),
        "markdown_path": str(markdown_path),
    }


def write_baseline_markdown(
    aggregate: pd.DataFrame,
    gain_frame: pd.DataFrame,
    diagnostic: dict[str, Any],
    output_path: Path,
) -> Path:
    """Write a concise original-versus-latent baseline analysis."""
    lines = [
        "# Original-Feature Baseline Analysis",
        "",
        "## Aggregate Results",
        "",
        "| Representation | Scenario | Accuracy | Macro Recall | Macro F1 | Training (s) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for _, row in aggregate.iterrows():
        lines.append(
            f"| {row['representation']} | {row['scenario']} | {row['accuracy']:.6f} | "
            f"{row['macro_recall']:.6f} | {row['macro_f1']:.6f} | "
            f"{row['training_seconds']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## DoS and DDoS Diagnostic",
            "",
            "| Scenario | Class | Latent Recall | Original Recall | Recall Delta | "
            "Latent F1 | Original F1 | F1 Delta |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in gain_frame.iterrows():
        lines.append(
            f"| {row['scenario']} | {row['class_name']} | {row['latent_recall']:.6f} | "
            f"{row['original_recall']:.6f} | "
            f"{row['recall_delta_original_minus_latent']:.6f} | "
            f"{row['latent_f1']:.6f} | {row['original_f1']:.6f} | "
            f"{row['f1_delta_original_minus_latent']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            diagnostic["diagnosis"],
            "",
            diagnostic["causal_limit"],
            "",
        ]
    )
    with open(output_path, "w", encoding="utf-8") as file:
        file.write("\n".join(lines))
    return output_path


def run_original_feature_baseline(
    config_path: str | Path = "configs/config.yaml",
    force: bool = False,
) -> dict[str, Any]:
    """Train and evaluate the complete S1/S2 original-feature diagnostic baseline."""
    training_reports = train_original_feature_baselines(config_path, force)
    comparison = compare_original_and_latent_baselines(config_path)
    return {"training_reports": training_reports, **comparison}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the original-feature baseline."""
    parser = argparse.ArgumentParser(description="Run the 69-feature LightGBM baseline.")
    parser.add_argument(
        "--scenario",
        default="all",
        choices=["all", *BASELINE_SCENARIOS],
        help="Baseline scenario to train.",
    )
    parser.add_argument("--config", default="configs/config.yaml", help="YAML config path.")
    parser.add_argument("--force", action="store_true", help="Retrain baseline artifacts.")
    return parser.parse_args()


def main() -> None:
    """Run the original-feature baseline from the command line."""
    args = parse_args()
    if args.scenario == "all":
        result = run_original_feature_baseline(args.config, args.force)
        print("Original-feature LightGBM baseline completed.")
        print(result["aggregate_comparison"].to_string(index=False))
        print(result["diagnostic"]["diagnosis"])
    else:
        report = train_original_feature_baseline(args.scenario, args.config, args.force)
        print(
            f"Original-feature baseline {args.scenario} completed: "
            f"{report['training_rows']:,} training rows."
        )


if __name__ == "__main__":
    main()
