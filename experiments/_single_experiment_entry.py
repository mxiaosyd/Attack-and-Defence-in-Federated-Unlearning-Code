"""Small entry helpers for one comprehensive experiment group."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

from experiments.comprehensive_config import IMPROVED_VALIDATION_METHODS, METHODS
from experiments.run_comprehensive_reproduction import run_comprehensive


def _run_group(
    *,
    label: str,
    experiment_name: str,
    methods: List[str],
    default_output_root: str,
    argv: List[str] | None,
) -> int:
    parser = argparse.ArgumentParser(description=f"Run {label} {experiment_name} experiment")
    parser.add_argument("--output_root", default=default_output_root)
    parser.add_argument("--timeout_per_scenario", type=int, default=14400)
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args(argv)

    try:
        output_root = Path(args.output_root)
        summary = run_comprehensive(
            output_root,
            experiment_names=[experiment_name],
            methods=list(methods),
            timeout=int(args.timeout_per_scenario),
            python=str(args.python),
        )
        exp_root = output_root / experiment_name
        print(f"[{label}] finished in {summary['elapsed_sec']:.2f}s")
        print(f"[{label}] methods={', '.join(methods)}")
        print(f"[{label}] experiment={experiment_name}")
        print(f"[{label}] summary={exp_root / 'summary.csv'}")
        print(f"[{label}] detailed={exp_root / 'detailed_results.csv'}")
        print(f"[{label}] manifest={exp_root / 'manifest.json'}")
    except Exception as exc:
        print(f"[{label}] ERROR: {exc}")
        return 1
    return 0


def run_original_7(experiment_name: str, argv: List[str] | None = None) -> int:
    return _run_group(
        label="original-7",
        experiment_name=experiment_name,
        methods=list(METHODS),
        default_output_root="outputs/comprehensive_reproduction",
        argv=argv,
    )


def run_improved_validation(experiment_name: str, argv: List[str] | None = None) -> int:
    return _run_group(
        label="improved-validation",
        experiment_name=experiment_name,
        methods=list(IMPROVED_VALIDATION_METHODS),
        default_output_root="outputs/improved_validation",
        argv=argv,
    )
