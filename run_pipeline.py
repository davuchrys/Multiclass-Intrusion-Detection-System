"""End-to-end orchestration for Phases 1 through 7."""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from src.data_loading import find_project_root, load_config, resolve_project_path


LOGGER = logging.getLogger("pipeline")

SCENARIO_ALIASES = {
    "s1": "s1_none",
    "s2": "s2_class_weight",
    "s3": "s3_upsampling",
    "s4": "s4_downsampling",
}

PHASE_NAMES = {
    1: "Dataset inspection",
    2: "Preprocessing",
    3: "Autoencoder training",
    4: "Latent feature extraction",
    5: "Class imbalance handling",
    6: "LightGBM training",
    7: "Evaluation and analysis",
}


def configure_logging(level: str = "INFO") -> None:
    """Configure concise timestamped pipeline logging."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )


def _all_exist(paths: list[Path]) -> bool:
    """Return whether every required artifact exists."""
    return all(path.exists() for path in paths)


def _load_json(path: Path) -> dict[str, Any]:
    """Load a JSON report used to summarize a reused phase."""
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def _scenario_names(scenario: str, configured: list[str]) -> list[str]:
    """Resolve a CLI scenario alias to canonical configured names."""
    if scenario == "all":
        return list(configured)
    canonical = SCENARIO_ALIASES[scenario]
    if canonical not in configured:
        raise ValueError(f"Scenario {canonical} is not enabled in configs/config.yaml.")
    return [canonical]


def _expand_force_phases(requested: set[int]) -> set[int]:
    """Propagate forced rebuilds to phases that consume changed artifacts."""
    effective: set[int] = set()
    for phase in requested:
        if phase == 1:
            effective.add(1)
        else:
            effective.update(range(phase, 8))
    return effective


def _phase_artifacts(
    processed_dir: Path,
    models_dir: Path,
    metrics_dir: Path,
    figures_dir: Path,
) -> dict[int, list[Path]]:
    """Return the canonical artifacts required to reuse Phases 1 through 4."""
    return {
        1: [
            metrics_dir / "class_distribution.csv",
            metrics_dir / "feature_summary.csv",
            metrics_dir / "dropped_features.csv",
            metrics_dir / "class_signal_summary.csv",
            metrics_dir / "quasi_constant_review.csv",
            metrics_dir / "candidate_features.txt",
            metrics_dir / "data_inspection_report.txt",
        ],
        2: [
            processed_dir / "X_train.npy",
            processed_dir / "X_test.npy",
            processed_dir / "y_train.npy",
            processed_dir / "y_test.npy",
            processed_dir / "minmax_scaler.joblib",
            processed_dir / "label_mapping.json",
            metrics_dir / "preprocessing_report.json",
            metrics_dir / "split_class_distribution.csv",
        ],
        3: [
            models_dir / "autoencoder.pt",
            models_dir / "encoder.pt",
            metrics_dir / "autoencoder_history.csv",
            metrics_dir / "autoencoder_reconstruction_error.json",
            figures_dir / "ae_loss_curve.png",
        ],
        4: [
            processed_dir / "Z_train.npy",
            processed_dir / "Z_test.npy",
            metrics_dir / "latent_extraction_report.json",
        ],
    }


def _phase5_artifacts(
    scenarios: list[str],
    processed_dir: Path,
    metrics_dir: Path,
) -> list[Path]:
    """Return reports and generated training arrays required by Phase 5."""
    required: list[Path] = []
    for scenario in scenarios:
        required.extend(
            [
                metrics_dir / f"imbalance_report_{scenario}.json",
                metrics_dir / f"imbalance_distribution_{scenario}.csv",
            ]
        )
        if scenario == "s2_class_weight":
            required.append(processed_dir / "sample_weight_s2_class_weight.npy")
        if scenario in {"s3_upsampling", "s4_downsampling"}:
            required.extend(
                [
                    processed_dir / f"Z_train_{scenario}.npy",
                    processed_dir / f"y_train_{scenario}.npy",
                ]
            )
    return required


def _run_phase(
    phase: int,
    status: str,
    action: Callable[[], Any],
    records: list[dict[str, Any]],
) -> Any:
    """Execute one phase with consistent progress and duration logging."""
    name = PHASE_NAMES[phase]
    LOGGER.info("Phase %d/7 started: %s (%s).", phase, name, status)
    started = time.perf_counter()
    try:
        result = action()
    except Exception:
        LOGGER.exception("Phase %d/7 failed: %s.", phase, name)
        raise
    duration = time.perf_counter() - started
    records.append(
        {
            "phase": phase,
            "name": name,
            "status": status,
            "duration_seconds": round(duration, 6),
        }
    )
    LOGGER.info("Phase %d/7 completed in %.2f seconds.", phase, duration)
    return result


def run_pipeline(
    config_path: str | Path = "configs/config.yaml",
    scenario: str = "all",
    skip_preprocessing: bool = False,
    force_phases: set[int] | None = None,
    device: str = "auto",
    sample_size: int = 100_000,
    chunk_size: int = 100_000,
) -> dict[str, Any]:
    """Run or resume Phases 1 through 7 and persist an orchestration report."""
    if scenario not in {"all", *SCENARIO_ALIASES}:
        raise ValueError(f"Unknown scenario: {scenario}")
    if sample_size <= 0 or chunk_size <= 0:
        raise ValueError("sample_size and chunk_size must be positive integers.")

    requested_force = set(force_phases or set())
    invalid_phases = requested_force - set(PHASE_NAMES)
    if invalid_phases:
        raise ValueError(f"Invalid force phases: {sorted(invalid_phases)}")
    if skip_preprocessing and 2 in requested_force:
        raise ValueError("--skip-preprocessing cannot be combined with --force-phase 2.")

    project_root = find_project_root()
    config = load_config(config_path)
    config_file = Path(config_path)
    if not config_file.is_absolute():
        config_file = project_root / config_file

    paths = config["paths"]
    processed_dir = resolve_project_path(paths["processed_dir"], project_root)
    models_dir = resolve_project_path(paths["models_dir"], project_root)
    metrics_dir = resolve_project_path(paths["metrics_dir"], project_root)
    figures_dir = resolve_project_path(paths["figures_dir"], project_root)
    for directory in (processed_dir, models_dir, metrics_dir, figures_dir):
        directory.mkdir(parents=True, exist_ok=True)

    configured_scenarios = list(config["imbalance"]["scenarios"])
    target_scenarios = _scenario_names(scenario, configured_scenarios)
    artifacts = _phase_artifacts(processed_dir, models_dir, metrics_dir, figures_dir)
    if skip_preprocessing and not _all_exist(artifacts[2]):
        missing = [str(path) for path in artifacts[2] if not path.exists()]
        raise FileNotFoundError(
            "Phase 2 was skipped, but its required artifacts are incomplete: "
            f"{missing}"
        )
    effective_force = _expand_force_phases(requested_force)
    records: list[dict[str, Any]] = []
    started_at = datetime.now().astimezone().isoformat(timespec="seconds")
    pipeline_started = time.perf_counter()

    LOGGER.info(
        "Pipeline started for %s. Forced phases: %s.",
        ", ".join(target_scenarios),
        sorted(effective_force) or "none",
    )

    phase1_execute = 1 in effective_force or not _all_exist(artifacts[1])
    if phase1_execute:
        def execute_phase1() -> dict[str, Any]:
            from src.data_loading import inspect_dataset

            return inspect_dataset(config_path, sample_size, chunk_size)

        _run_phase(
            1,
            "executed",
            execute_phase1,
            records,
        )
    else:
        _run_phase(1, "reused", lambda: None, records)

    phase2_execute = 2 in effective_force or not _all_exist(artifacts[2])
    if phase2_execute:
        def execute_phase2() -> dict[str, Any]:
            from src.preprocessing import preprocess_dataset

            return preprocess_dataset(config_path, chunk_size)

        _run_phase(
            2,
            "executed",
            execute_phase2,
            records,
        )
        effective_force.update(range(3, 8))
    else:
        _run_phase(
            2,
            "reused",
            lambda: _load_json(metrics_dir / "preprocessing_report.json"),
            records,
        )

    phase3_execute = 3 in effective_force or not _all_exist(artifacts[3])
    if phase3_execute:
        def execute_phase3() -> dict[str, Any]:
            from src.autoencoder import train_autoencoder

            return train_autoencoder(config_path, device_name=device)

        _run_phase(
            3,
            "executed",
            execute_phase3,
            records,
        )
        effective_force.update(range(4, 8))
    else:
        _run_phase(
            3,
            "reused",
            lambda: _load_json(metrics_dir / "autoencoder_reconstruction_error.json"),
            records,
        )

    phase4_execute = 4 in effective_force or not _all_exist(artifacts[4])
    if phase4_execute:
        def execute_phase4() -> dict[str, Any]:
            from src.latent_extraction import extract_latent_features

            return extract_latent_features(config_path, device_name=device)

        _run_phase(
            4,
            "executed",
            execute_phase4,
            records,
        )
        effective_force.update(range(5, 8))
    else:
        _run_phase(
            4,
            "reused",
            lambda: _load_json(metrics_dir / "latent_extraction_report.json"),
            records,
        )

    phase5_artifacts = _phase5_artifacts(target_scenarios, processed_dir, metrics_dir)
    phase5_execute = 5 in effective_force or not _all_exist(phase5_artifacts)

    def execute_phase5() -> dict[str, Any]:
        from src.imbalance import (
            prepare_all_imbalance_scenarios,
            prepare_imbalance_scenario,
        )

        if scenario == "all":
            return prepare_all_imbalance_scenarios(
                config_path,
                force=5 in effective_force,
            )
        return prepare_imbalance_scenario(
            scenario,
            config_path,
            force=5 in effective_force,
        )

    _run_phase(
        5,
        "executed" if phase5_execute else "validated",
        execute_phase5,
        records,
    )
    if phase5_execute:
        effective_force.update(range(6, 8))

    phase6_artifacts = [
        artifact
        for name in target_scenarios
        for artifact in (
            models_dir / f"lgbm_{name}.txt",
            processed_dir / f"y_pred_{name}.npy",
            metrics_dir / f"classifier_report_{name}.json",
        )
    ]
    phase6_execute = 6 in effective_force or not _all_exist(phase6_artifacts)

    def execute_phase6() -> dict[str, Any]:
        from src.classifier import train_all_scenarios, train_scenario

        if scenario == "all":
            return train_all_scenarios(
                config_path,
                force=6 in effective_force,
            )
        return train_scenario(
            scenario,
            config_path,
            force=6 in effective_force,
        )

    _run_phase(
        6,
        "executed" if phase6_execute else "validated",
        execute_phase6,
        records,
    )

    def execute_phase7() -> dict[str, Any]:
        from src.evaluation import evaluate_all_scenarios, evaluate_scenario

        if scenario == "all":
            return evaluate_all_scenarios(config_path)
        return evaluate_scenario(scenario, config_path)

    evaluation_result = _run_phase(7, "executed", execute_phase7, records)

    total_seconds = time.perf_counter() - pipeline_started
    report_path = metrics_dir / "pipeline_report.json"
    report = {
        "success": True,
        "started_at": started_at,
        "completed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "total_duration_seconds": round(total_seconds, 6),
        "config_path": str(config_file.resolve()),
        "scenario": scenario,
        "target_scenarios": target_scenarios,
        "skip_preprocessing": skip_preprocessing,
        "requested_force_phases": sorted(requested_force),
        "effective_force_phases": sorted(effective_force),
        "device": device,
        "sample_size": sample_size,
        "chunk_size": chunk_size,
        "phases": records,
        "evaluation_artifacts": (
            {
                "summary_path": evaluation_result["summary_path"],
                "analysis_markdown_path": evaluation_result["analysis_markdown_path"],
            }
            if scenario == "all"
            else {"report_path": evaluation_result["report_path"]}
        ),
    }
    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    LOGGER.info("Pipeline completed successfully in %.2f seconds.", total_seconds)
    LOGGER.info("Orchestration report: %s", report_path)
    report["report_path"] = str(report_path)
    return report


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the end-to-end pipeline."""
    parser = argparse.ArgumentParser(description="Run or resume the CIC-ToN-IoT pipeline.")
    parser.add_argument(
        "--scenario",
        default="all",
        choices=["s1", "s2", "s3", "s4", "all"],
        help="Imbalance handling scenario to run through Phases 5 to 7.",
    )
    parser.add_argument("--config", default="configs/config.yaml", help="YAML config path.")
    parser.add_argument(
        "--skip-preprocessing",
        action="store_true",
        help="Require and reuse existing Phase 2 artifacts.",
    )
    parser.add_argument(
        "--force-phase",
        action="append",
        type=int,
        choices=range(1, 8),
        default=[],
        metavar="N",
        help="Rebuild phase N and its dependent phases. Repeat for multiple phases.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="PyTorch device used when Phases 3 or 4 must run.",
    )
    parser.add_argument("--sample-size", type=int, default=100_000)
    parser.add_argument("--chunk-size", type=int, default=100_000)
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def main() -> None:
    """Run the pipeline from the command line."""
    args = parse_args()
    configure_logging(args.log_level)
    run_pipeline(
        config_path=args.config,
        scenario=args.scenario,
        skip_preprocessing=args.skip_preprocessing,
        force_phases=set(args.force_phase),
        device=args.device,
        sample_size=args.sample_size,
        chunk_size=args.chunk_size,
    )


if __name__ == "__main__":
    main()
