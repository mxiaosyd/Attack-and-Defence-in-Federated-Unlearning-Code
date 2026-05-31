"""Shared experiment runner.

Example quick check:
    python common/experiment_runner.py --dataset synthetic --global_rounds 3 --train_size 300 --test_size 100 \
      --num_clients 6 --malicious_clients 1 --repair_epochs 1

Example real MNIST run:
    python common/experiment_runner.py --dataset mnist --download --model smallcnn --global_rounds 20 \
      --num_clients 20 --malicious_clients 4 --methods FedRecover Crab FedUP FedSweep FAST MCCFed
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Mapping, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import methods

from common.args import build_parser
from common.fl_engine import FLSimulator, evaluate_recovery_result
from common.metrics import evaluate_all
from common.recovery_base import get_method
from common.utils import Timer, ensure_dir, model_to_vector, save_json, timestamp, write_csv


def load_optional_method_packages(method_names):
    improved_aliases = {
        "fast+",
        "fast_plus",
        "fastplus",
        "fedsweep+",
        "fedsweep_plus",
        "fedsweepplus",
    }
    normalized = {str(name).strip().lower() for name in method_names}
    if normalized.intersection(improved_aliases):
        import improved_methods


def main(single_method: Optional[str] = None, default_overrides: Mapping[str, object] | None = None):
    parser = build_parser(single_method=single_method)
    if default_overrides:
        parser.set_defaults(**dict(default_overrides))
    args = parser.parse_args()
    load_optional_method_packages(args.methods)
    out_dir = ensure_dir(Path(args.output_dir) / f"run-{timestamp()}")
    save_json(out_dir / "config.json", vars(args))

    print(f"[setup] output_dir={out_dir}")
    print(f"[setup] methods={args.methods}")
    sim = FLSimulator(args)

    print("[train] training one shared poisoned FL history for all methods...")
    with Timer() as t_train:
        context = sim.build_context()
    poisoned_model = context.new_model_from_vector(context.poisoned_vector)
    poisoned_metrics = evaluate_all(poisoned_model, context.test_loader, context.device,
                                    target_label=context.target_label, trigger_size=context.trigger_size,
                                    trigger_pattern=context.trigger_pattern,
                                    trigger_position=context.trigger_position)
    print(f"[train] done in {t_train.elapsed:.2f}s | poisoned clean_acc={poisoned_metrics['clean_acc']:.4f}, ASR={poisoned_metrics['asr']:.4f}")

    rows = []
    rows.append({
        "method": "Poisoned",
        "clean_acc": poisoned_metrics["clean_acc"],
        "clean_loss": poisoned_metrics["clean_loss"],
        "asr": poisoned_metrics["asr"],
        "runtime_sec": t_train.elapsed,
        "method_note": "trained_with_malicious_clients",
    })

    if args.include_retrain_baseline:
        print("[baseline] training clean retrain-from-scratch baseline...")
        with Timer() as t_retrain:
            retrain_model = sim.train_retrain_baseline()
        row = evaluate_recovery_result(context, "RetrainClean", retrain_model, {"runtime_sec": t_retrain.elapsed})
        rows.append(row)
        print(f"[baseline] clean_acc={row['clean_acc']:.4f}, ASR={row['asr']:.4f}, time={t_retrain.elapsed:.2f}s")

    for method_name in args.methods:
        print(f"[recover] {method_name}...")
        method = get_method(method_name)
        with Timer() as t:
            result = method.recover(context)
        row = evaluate_recovery_result(context, method_name, result.model, result.extra)
        row["runtime_sec"] = t.elapsed
        rows.append(row)
        print(f"[recover] {method_name}: clean_acc={row['clean_acc']:.4f}, ASR={row['asr']:.4f}, time={t.elapsed:.2f}s")
        if args.save_models:
            import torch
            torch.save(result.model.state_dict(), out_dir / f"{method_name}.pt")

    write_csv(out_dir / "results.csv", rows)
    save_json(out_dir / "results.json", rows)
    print(f"[done] results saved to {out_dir / 'results.csv'}")
    return rows


if __name__ == "__main__":
    main()
