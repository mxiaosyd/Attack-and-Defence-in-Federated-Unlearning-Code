"""Configuration and runners for seven independent paper reproductions."""
from __future__ import annotations

import argparse
import csv
import os
import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List


@dataclass(frozen=True)
class MethodConfig:
    key: str
    method_name: str
    args: List[str]


METHOD_ORDER = [
    "fedrecover",
    "crab",
    "fedup",
    "fedsweep",
    "unlearning_backdoor",
    "fast",
    "mcc_fed",
]


METHOD_NAMES = {
    "fedrecover": "FedRecover",
    "crab": "Crab",
    "fedup": "FedUP",
    "fedsweep": "FedSweep",
    "unlearning_backdoor": "UnlearningBackdoor",
    "fast": "FAST",
    "mcc_fed": "MCCFed",
}


COMMON_ARGS = [
    "--dataset", "mnist",
    "--data_root", "./data",
    "--download",
    "--seed", "42",
    "--model", "smallcnn",
    "--train_size", "15000",
    "--test_size", "3000",
    "--num_clients", "20",
    "--malicious_clients", "4",
    "--client_fraction", "1.0",
    "--partition", "dirichlet",
    "--dirichlet_alpha", "0.5",
    "--global_rounds", "60",
    "--local_epochs", "2",
    "--batch_size", "64",
    "--test_batch_size", "256",
    "--lr", "0.08",
    "--server_lr", "1.0",
    "--aggregation", "fedavg",
    "--attack", "backdoor",
    "--attack_start_round", "0",
    "--target_label", "0",
    "--poison_fraction", "0.7",
    "--trigger_size", "4",
    "--trigger_pattern", "square",
    "--trigger_position", "bottom_right",
    "--public_source", "test_split",
    "--public_fraction", "0.5",
    "--repair_epochs", "4",
    "--repair_lr", "0.02",
    "--num_threads", "1",
]


DETECTED_MALICIOUS_ARGS = [
    "--detected_malicious_clients", "0,1,2,3",
]


METHOD_CONFIGS: Dict[str, MethodConfig] = {
    "fedrecover": MethodConfig(
        key="fedrecover",
        method_name="FedRecover",
        args=[
            *COMMON_ARGS,
            *DETECTED_MALICIOUS_ARGS,
            "--fedrecover_warmup_rounds", "20",
            "--fedrecover_correction_period", "10",
            "--fedrecover_final_tuning_rounds", "6",
            "--fedrecover_lbfgs_buffer_size", "2",
            "--fedrecover_tolerance_rate", "1e-6",
        ],
    ),
    "crab": MethodConfig(
        key="crab",
        method_name="Crab",
        args=[
            *COMMON_ARGS,
            *DETECTED_MALICIOUS_ARGS,
            "--crab_select_round_ratio", "0.6",
            "--crab_select_client_ratio", "0.7",
            "--crab_sensitivity_threshold", "0.05",
        ],
    ),
    "fedup": MethodConfig(
        key="fedup",
        method_name="FedUP",
        args=[
            *COMMON_ARGS,
            *DETECTED_MALICIOUS_ARGS,
            "--fedup_auto_prune_ratio",
            "--fedup_auto_recovery_rounds",
            "--fedup_similarity_mode", "model",
            "--fedup_recovery_optimizer", "adam",
            "--fedup_recovery_lr", "0.001",
        ],
    ),
    "fedsweep": MethodConfig(
        key="fedsweep",
        method_name="FedSweep",
        args=[
            *COMMON_ARGS,
            "--fedsweep_trigger_generator", "patch",
            "--fedsweep_patch_position", "search",
            "--fedsweep_trigger_steps", "150",
            "--fedsweep_generator_lr", "0.1",
            "--fedsweep_lambda_l1", "0.001",
            "--fedsweep_unet_base_channels", "16",
            "--fedsweep_trigger_sparse_ratio", "0",
            "--fedsweep_entropy_batches", "5",
            "--fedsweep_credit_weight", "0.5",
            "--fedsweep_credit_decay", "0.9",
            "--fedsweep_credit_reward", "0.1",
            "--fedsweep_unlearn_epochs", "2",
            "--fedsweep_unlearn_lr", "0.003",
            "--fedsweep_feature_weight", "0.3",
            "--fedsweep_clean_weight", "2.0",
        ],
    ),
    "unlearning_backdoor": MethodConfig(
        key="unlearning_backdoor",
        method_name="UnlearningBackdoor",
        args=[
            *COMMON_ARGS,
            *DETECTED_MALICIOUS_ARGS,
            "--public_per_client", "50",
            "--unlearning_backdoor_alpha", "1.0",
            "--unlearning_backdoor_kd_epochs", "4",
            "--unlearning_backdoor_kd_lr", "0.02",
            "--unlearning_backdoor_temperature", "2.0",
        ],
    ),
    "fast": MethodConfig(
        key="fast",
        method_name="FAST",
        args=[
            *COMMON_ARGS,
            *DETECTED_MALICIOUS_ARGS,
            "--fast_extra_epochs", "4",
            "--fast_extra_lr", "0.02",
            "--fast_over_unlearning_tolerance", "0.01",
            "--fast_overunlearning_patience", "3",
            "--fast_unlearning_order", "forward",
        ],
    ),
    "mcc_fed": MethodConfig(
        key="mcc_fed",
        method_name="MCCFed",
        args=[
            *COMMON_ARGS,
            "--num_clients", "10",
            "--malicious_clients", "4",
            "--partition", "iid",
            "--poison_fraction", "0.9",
            "--mcc_detection_ratio_threshold", "1.0",
            "--mcc_detection_policy", "minority_kmeans",
            "--mcc_contribution_init",
            "--mcc_unlearn_rounds", "10",
            "--mcc_alpha", "1.0",
            "--mcc_noise_std", "0.01",
            "--mcc_regularization_scale", "1.0",
            "--mcc_regularization_mode", "repel_malicious",
            "--mcc_optimizer", "sgd",
        ],
    ),
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def runtime_env(root: Path) -> Dict[str, str]:
    env = dict(os.environ)
    site_packages = root / ".venv" / "Lib" / "site-packages"
    parts = [str(root)]
    if site_packages.exists():
        parts.append(str(site_packages))
    if env.get("PYTHONPATH"):
        parts.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(parts)
    return env


def available_methods() -> List[str]:
    return list(METHOD_ORDER)


def get_method_config(method: str) -> MethodConfig:
    key = method.lower()
    if key not in METHOD_CONFIGS:
        raise KeyError(f"Unknown method {method!r}. Available: {', '.join(METHOD_ORDER)}")
    return METHOD_CONFIGS[key]


def merge_extra_args(base: List[str], extra: List[str]) -> List[str]:
    if not extra:
        return list(base)
    return [*base, *extra]


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _latest_run_dir(method_output: Path) -> Path | None:
    runs = [path for path in method_output.glob("run-*") if path.is_dir()]
    if not runs:
        return None
    return max(runs, key=lambda path: path.stat().st_mtime)


def _as_float(row: Dict[str, Any] | None, key: str) -> float | None:
    if not row or key not in row:
        return None
    try:
        return float(row[key])
    except (TypeError, ValueError):
        return None


def _compact_result_payload(
    *,
    method_key: str,
    config: MethodConfig,
    command: List[str],
    completed: subprocess.CompletedProcess[str],
    elapsed: float,
    method_output: Path,
) -> Dict[str, Any]:
    run_dir = _latest_run_dir(method_output)
    rows = _read_csv_rows(run_dir / "results.csv") if run_dir else []
    results_json = _read_json(run_dir / "results.json", rows) if run_dir else rows
    run_config = _read_json(run_dir / "config.json", {}) if run_dir else {}

    poisoned = next((row for row in rows if row.get("method") == "Poisoned"), None)
    recovered = rows[-1] if rows else None
    poisoned_asr = _as_float(poisoned, "asr")
    recovered_asr = _as_float(recovered, "asr")
    poisoned_acc = _as_float(poisoned, "clean_acc")
    recovered_acc = _as_float(recovered, "clean_acc")

    metrics: Dict[str, float] = {}
    if poisoned_acc is not None:
        metrics["poisoned_clean_acc"] = poisoned_acc
    if poisoned_asr is not None:
        metrics["poisoned_asr"] = poisoned_asr
    if recovered_acc is not None:
        metrics["recovered_clean_acc"] = recovered_acc
    if recovered_asr is not None:
        metrics["recovered_asr"] = recovered_asr
    if poisoned_asr is not None and recovered_asr is not None:
        metrics["asr_drop"] = poisoned_asr - recovered_asr
    if poisoned_acc is not None and recovered_acc is not None:
        metrics["clean_acc_delta"] = recovered_acc - poisoned_acc

    return {
        "method_key": method_key,
        "method": config.method_name,
        "status": "completed" if completed.returncode == 0 else "failed",
        "returncode": completed.returncode,
        "elapsed_sec": elapsed,
        "command": command,
        "run_dir_name": run_dir.name if run_dir else "",
        "config": run_config,
        "poisoned": poisoned or {},
        "recovered": recovered or {},
        "metrics": metrics,
        "results": results_json,
        "stdout": completed.stdout,
    }


def _cleanup_raw_outputs(method_output: Path) -> None:
    for path in method_output.glob("run-*"):
        if path.is_dir() and path.parent == method_output:
            shutil.rmtree(path)
    for path in method_output.glob("*.log"):
        if path.is_file() and path.parent == method_output:
            path.unlink()
    for path in method_output.glob("*.command.json"):
        if path.is_file() and path.parent == method_output:
            path.unlink()


def run_independent_method(method_key: str, output_root: Path, *, timeout: int, python: str,
                           extra_args: List[str] | None = None,
                           keep_raw_outputs: bool = False) -> Dict[str, object]:
    root = repo_root()
    config = get_method_config(method_key)
    method_output = output_root / method_key
    args = merge_extra_args(config.args, extra_args or [])
    command = [
        python,
        str(root / "common" / "experiment_runner.py"),
        *args,
        "--methods",
        config.method_name,
        "--output_dir",
        str(method_output),
    ]

    method_output.mkdir(parents=True, exist_ok=True)
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

    result_path = method_output / "result.json"
    payload = _compact_result_payload(
        method_key=method_key,
        config=config,
        command=command,
        completed=completed,
        elapsed=elapsed,
        method_output=method_output,
    )
    result_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if not keep_raw_outputs:
        _cleanup_raw_outputs(method_output)

    if completed.returncode != 0:
        raise RuntimeError(f"{config.method_name} failed. See {result_path}")

    return {
        "method_key": method_key,
        "method": config.method_name,
        "elapsed_sec": elapsed,
        "result_path": str(result_path),
        "output_root": str(method_output),
        "returncode": completed.returncode,
    }


def main_for_method(method_key: str, argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=f"Run {get_method_config(method_key).method_name} reproduction")
    parser.add_argument("--output_root", default="outputs/independent_reproduction")
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--keep_raw_outputs", action="store_true")
    parser.add_argument("--extra", nargs=argparse.REMAINDER, default=[])
    args = parser.parse_args(argv)

    output_root = Path(args.output_root)
    try:
        print(f"[independent] running {method_key}")
        summary = run_independent_method(
            method_key,
            output_root,
            timeout=int(args.timeout),
            python=str(args.python),
            extra_args=list(args.extra),
            keep_raw_outputs=bool(args.keep_raw_outputs),
        )
        print(f"[independent] {method_key} finished in {summary['elapsed_sec']:.2f}s")
        print(f"[independent] output_root={summary['output_root']}")
        print(f"[independent] result={summary['result_path']}")
    except Exception as exc:
        print(f"[independent] {method_key} ERROR: {exc}")
        return 1
    return 0
