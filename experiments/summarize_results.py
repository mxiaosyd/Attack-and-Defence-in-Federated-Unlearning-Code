"""Combine comprehensive experiment results into simple data files."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping


SCENARIO_FIELDS = [
    "experiment",
    "experiment_description",
    "metric_focus",
    "scenario",
    "dataset",
    "partition",
    "dirichlet_alpha",
    "model",
    "train_size",
    "test_size",
    "num_clients",
    "malicious_clients",
    "global_rounds",
    "local_epochs",
    "attack",
    "attack_scale",
    "poison_fraction",
    "trigger_pattern",
    "trigger_position",
    "public_source",
    "public_fraction",
]


SUMMARY_FIELDS = [
    *SCENARIO_FIELDS,
    "method",
    "clean_acc",
    "clean_loss",
    "asr",
    "runtime_sec",
    "history_storage_mb",
    "clean_acc_delta_vs_poisoned",
    "asr_drop_vs_poisoned",
    "clean_acc_gap_to_retrain",
    "asr_gap_to_retrain",
]


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: List[Mapping[str, Any]], fieldnames: List[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _safe_float(value: Any) -> float:
    try:
        if value == "":
            return math.nan
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _finite(value: float) -> bool:
    return not math.isnan(value) and not math.isinf(value)


def _diff(left: Any, right: Any) -> float:
    left_value = _safe_float(left)
    right_value = _safe_float(right)
    if not _finite(left_value) or not _finite(right_value):
        return math.nan
    return left_value - right_value


def _abs_gap(left: Any, right: Any) -> float:
    value = _diff(left, right)
    if not _finite(value):
        return math.nan
    return abs(value)


def _row_by_method(rows: Iterable[Mapping[str, Any]], method: str) -> Mapping[str, Any]:
    for row in rows:
        if row.get("method") == method:
            return row
    return {}


def _scenario_fields(config: Mapping[str, Any], scenario: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "experiment": scenario.get("experiment", ""),
        "experiment_description": scenario.get("experiment_description", ""),
        "metric_focus": scenario.get("metric_focus", ""),
        "scenario": scenario.get("name", ""),
        "dataset": config.get("dataset", ""),
        "partition": config.get("partition", ""),
        "dirichlet_alpha": config.get("dirichlet_alpha", ""),
        "model": config.get("model", ""),
        "train_size": config.get("train_size", ""),
        "test_size": config.get("test_size", ""),
        "num_clients": config.get("num_clients", ""),
        "malicious_clients": config.get("malicious_clients", ""),
        "global_rounds": config.get("global_rounds", ""),
        "local_epochs": config.get("local_epochs", ""),
        "attack": config.get("attack", ""),
        "attack_scale": config.get("attack_scale", ""),
        "poison_fraction": config.get("poison_fraction", ""),
        "trigger_pattern": config.get("trigger_pattern", ""),
        "trigger_position": config.get("trigger_position", ""),
        "public_source": config.get("public_source", ""),
        "public_fraction": config.get("public_fraction", ""),
    }


def rows_from_run(run_dir: Path) -> List[Dict[str, Any]]:
    rows = _load_json(run_dir / "results.json", [])
    config = _load_json(run_dir / "config.json", {})
    scenario = _load_json(run_dir / "scenario.json", {})
    if not isinstance(rows, list):
        raise ValueError(f"Expected list in {run_dir / 'results.json'}")

    poisoned = _row_by_method(rows, "Poisoned")
    retrain = _row_by_method(rows, "RetrainClean")
    scenario_base = _scenario_fields(config, scenario)

    out: List[Dict[str, Any]] = []
    for row in rows:
        combined = {
            **scenario_base,
            "method": row.get("method", ""),
            "clean_acc": row.get("clean_acc", ""),
            "clean_loss": row.get("clean_loss", ""),
            "asr": row.get("asr", ""),
            "runtime_sec": row.get("runtime_sec", ""),
            "history_storage_mb": row.get("history_storage_mb", ""),
            "clean_acc_delta_vs_poisoned": _diff(row.get("clean_acc"), poisoned.get("clean_acc")),
            "asr_drop_vs_poisoned": _diff(poisoned.get("asr"), row.get("asr")),
            "clean_acc_gap_to_retrain": _abs_gap(row.get("clean_acc"), retrain.get("clean_acc")),
            "asr_gap_to_retrain": _abs_gap(row.get("asr"), retrain.get("asr")),
        }
        out.append(combined)
    return out


def detailed_rows_from_run(run_dir: Path) -> List[Dict[str, Any]]:
    rows = _load_json(run_dir / "results.json", [])
    config = _load_json(run_dir / "config.json", {})
    scenario = _load_json(run_dir / "scenario.json", {})
    scenario_base = _scenario_fields(config, scenario)
    return [{**scenario_base, **dict(row)} for row in rows]


def _union_fields(rows: List[Mapping[str, Any]], base_fields: List[str]) -> List[str]:
    fields = list(base_fields)
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    return fields


def write_combined_outputs(output_root: Path, run_dirs: List[Path]) -> Dict[str, str]:
    summary_rows: List[Dict[str, Any]] = []
    detailed_rows: List[Dict[str, Any]] = []
    for run_dir in run_dirs:
        summary_rows.extend(rows_from_run(run_dir))
        detailed_rows.extend(detailed_rows_from_run(run_dir))

    summary_csv = output_root / "summary.csv"
    summary_json = output_root / "summary.json"
    detailed_csv = output_root / "detailed_results.csv"
    detailed_json = output_root / "detailed_results.json"

    _write_csv(summary_csv, summary_rows, SUMMARY_FIELDS)
    _write_json(summary_json, summary_rows)
    _write_csv(detailed_csv, detailed_rows, _union_fields(detailed_rows, SCENARIO_FIELDS))
    _write_json(detailed_json, detailed_rows)

    return {
        "summary_csv": str(summary_csv),
        "summary_json": str(summary_json),
        "detailed_csv": str(detailed_csv),
        "detailed_json": str(detailed_json),
    }
