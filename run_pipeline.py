"""End-to-end pipeline entry point.

Implementation will be filled phase by phase after preprocessing is validated
in notebooks.
"""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the CIC-ToN-IoT pipeline.")
    parser.add_argument(
        "--scenario",
        default="all",
        choices=["s1", "s2", "s3", "s4", "all"],
        help="Imbalance handling scenario to run.",
    )
    parser.add_argument(
        "--skip-preprocessing",
        action="store_true",
        help="Reuse existing processed artifacts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raise NotImplementedError(
        f"Pipeline orchestration is not implemented yet. Requested scenario: {args.scenario}"
    )


if __name__ == "__main__":
    main()
