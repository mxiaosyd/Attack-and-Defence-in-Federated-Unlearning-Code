"""Run direct comprehensive experiments across datasets and data distributions."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.comprehensive_config import (
    DEFAULT_EXPERIMENTS,
    METHODS,
    get_experiment,
    get_scenario,
    list_experiment_names,
    list_scenario_names,
    repo_root,
    runtime_env,
    scenario_args,
)
from experiments.summarize_results import write_combined_outputs


def latest_run_dir(output_root: Path) -> Path | None:
    runs = [path for path in output_root.glob("run-*") if path.is_dir()]
    if not runs:
        return None
    return max(runs, key=lambda path: path.stat().st_mtime)


def run_scenario(
    scenario: Dict[str, object],
    output_root: Path,
    *,
    experiment: Dict[str, object],
    methods: List[str] | None = None,
    timeout: int,
    python: str,
) -> Dict[str, object]:
    root = repo_root()
    scenario_name = str(scenario["name"])
    scenario_root = output_root / scenario_name
    scenario_root.mkdir(parents=True, exist_ok=True)
    method_names = list(METHODS if methods is None else methods)

    command = [
        python,
        str(root / "common" / "experiment_runner.py"),
        *scenario_args(scenario),
        "--methods",
        *method_names,
        "--output_dir",
        str(scenario_root),
    ]

    started = time.time()
    completed = subprocess.run(
        command,
        cwd=str(root),
        env=runtime_env(root),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
    )
    elapsed = time.time() - started

    log_path = scenario_root / "comprehensive.log"
    log_path.write_text(completed.stdout, encoding="utf-8")
    (scenario_root / "command.json").write_text(
        json.dumps(
            {
                "command": command,
                "elapsed_sec": elapsed,
                "scenario": scenario_name,
                "methods": method_names,
                "returncode": completed.returncode,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    if completed.returncode != 0:
        raise RuntimeError(f"Scenario {scenario_name} failed. See {log_path}")

    run_dir = latest_run_dir(scenario_root)
    if run_dir is None:
        raise FileNotFoundError(f"No run-* directory found under {scenario_root}")

    scenario_payload = {
        "experiment": experiment.get("name", ""),
        "experiment_description": experiment.get("description", ""),
        "metric_focus": ", ".join(str(item) for item in experiment.get("focus_metrics", [])),
        "name": scenario_name,
        "description": scenario.get("description", ""),
    }
    (scenario_root / "scenario.json").write_text(
        json.dumps(scenario_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (run_dir / "scenario.json").write_text(
        json.dumps(scenario_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {
        "scenario": scenario_name,
        "elapsed_sec": elapsed,
        "log_path": str(log_path),
        "run_dir": str(run_dir),
    }


def run_experiment(
    output_root: Path,
    *,
    experiment_name: str,
    methods: List[str] | None = None,
    timeout: int,
    python: str,
) -> Dict[str, object]:
    experiment = get_experiment(experiment_name)
    experiment_root = output_root / experiment_name
    experiment_root.mkdir(parents=True, exist_ok=True)
    scenario_names = [str(name) for name in experiment["scenarios"]]
    method_names = list(METHODS if methods is None else methods)
    run_summaries: List[Dict[str, object]] = []
    run_dirs: List[Path] = []
    started = time.time()

    for name in scenario_names:
        scenario = get_scenario(name)
        summary = run_scenario(
            scenario,
            experiment_root,
            experiment=experiment,
            methods=method_names,
            timeout=timeout,
            python=python,
        )
        run_summaries.append(summary)
        run_dirs.append(Path(str(summary["run_dir"])))

    outputs = write_combined_outputs(experiment_root, run_dirs)
    elapsed = time.time() - started
    manifest = {
        "experiment": experiment_name,
        "description": experiment.get("description", ""),
        "focus_metrics": experiment.get("focus_metrics", []),
        "methods": method_names,
        "elapsed_sec": elapsed,
        "scenarios": scenario_names,
        "runs": run_summaries,
        "outputs": outputs,
    }
    (experiment_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {
        **manifest,
        **outputs,
    }


def run_custom_scenarios(
    output_root: Path,
    *,
    scenario_names: List[str],
    methods: List[str] | None = None,
    timeout: int,
    python: str,
) -> Dict[str, object]:
    method_names = list(METHODS if methods is None else methods)
    experiment = {
        "name": "custom",
        "description": "User-selected comprehensive scenarios.",
        "focus_metrics": ["clean_acc", "asr", "runtime_sec", "history_storage_mb"],
        "scenarios": scenario_names,
    }
    custom_root = output_root / "custom"
    custom_root.mkdir(parents=True, exist_ok=True)
    run_summaries: List[Dict[str, object]] = []
    run_dirs: List[Path] = []
    started = time.time()
    for name in scenario_names:
        scenario = get_scenario(name)
        summary = run_scenario(
            scenario,
            custom_root,
            experiment=experiment,
            methods=method_names,
            timeout=timeout,
            python=python,
        )
        run_summaries.append(summary)
        run_dirs.append(Path(str(summary["run_dir"])))
    outputs = write_combined_outputs(custom_root, run_dirs)
    elapsed = time.time() - started
    manifest = {
        "experiment": "custom",
        "description": experiment["description"],
        "focus_metrics": experiment["focus_metrics"],
        "methods": method_names,
        "elapsed_sec": elapsed,
        "scenarios": scenario_names,
        "runs": run_summaries,
        "outputs": outputs,
    }
    (custom_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_root / "manifest.json").write_text(
        json.dumps(
            {
                "elapsed_sec": elapsed,
                "experiments": ["custom"],
                "methods": method_names,
                "runs": [manifest],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        **manifest,
        **outputs,
    }


def run_comprehensive(
    output_root: Path,
    *,
    experiment_names: List[str],
    methods: List[str] | None = None,
    timeout: int,
    python: str,
) -> Dict[str, object]:
    output_root.mkdir(parents=True, exist_ok=True)
    started = time.time()
    method_names = list(METHODS if methods is None else methods)
    experiments: List[Dict[str, object]] = []
    for name in experiment_names:
        experiments.append(
            run_experiment(
                output_root,
                experiment_name=name,
                methods=method_names,
                timeout=timeout,
                python=python,
            )
        )
    manifest = {
        "elapsed_sec": time.time() - started,
        "experiments": experiment_names,
        "methods": method_names,
        "runs": experiments,
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run comprehensive experiments by metric-focused groups")
    parser.add_argument("--output_root", default="outputs/comprehensive_reproduction")
    parser.add_argument("--experiments", nargs="*", choices=list_experiment_names(), default=None)
    parser.add_argument("--scenarios", nargs="*", choices=list_scenario_names(), default=None)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--timeout_per_scenario", type=int, default=14400)
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args(argv)

    try:
        output_root = Path(args.output_root)
        if args.quick:
            experiment_names = ["quick"]
            summary = run_comprehensive(
                output_root,
                experiment_names=experiment_names,
                timeout=int(args.timeout_per_scenario),
                python=str(args.python),
            )
        elif args.scenarios:
            summary = run_custom_scenarios(
                output_root,
                scenario_names=list(args.scenarios),
                timeout=int(args.timeout_per_scenario),
                python=str(args.python),
            )
            experiment_names = ["custom"]
        else:
            experiment_names = list(args.experiments or DEFAULT_EXPERIMENTS)
            summary = run_comprehensive(
                output_root,
                experiment_names=experiment_names,
                timeout=int(args.timeout_per_scenario),
                python=str(args.python),
            )
        print(f"[comprehensive] finished in {summary['elapsed_sec']:.2f}s")
        print(f"[comprehensive] experiments={', '.join(experiment_names)}")
        for name in experiment_names:
            exp_root = output_root / name
            print(f"[comprehensive] {name}_summary={exp_root / 'summary.csv'}")
            print(f"[comprehensive] {name}_detailed={exp_root / 'detailed_results.csv'}")
        print(f"[comprehensive] manifest={output_root / 'manifest.json'}")
    except Exception as exc:
        print(f"[comprehensive] ERROR: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
