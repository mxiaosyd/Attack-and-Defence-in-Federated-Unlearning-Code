from __future__ import annotations

import copy
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import ConcatDataset, Subset

from common.utils import load_vector_into_model, model_to_vector
from attack_reproduction.common_attack import (
    batched_gradient_vector,
    ensure_dir,
    evaluate_accuracy,
    get_device,
    gradient_vector,
    last_linear,
    load_dataset,
    make_loader,
    make_model,
    now_stamp,
    partition_dataset,
    save_csv,
    save_json,
    set_seed,
    train_model,
)
from .config import ULIAConfig


def _label_set(dataset, indices: list[int]) -> set[int]:
    return {int(dataset[i][1]) for i in indices}


def _first_label_indices(dataset, labels: set[int], limit: int) -> list[int]:
    out: list[int] = []
    for i in range(len(dataset)):
        if int(dataset[i][1]) in labels:
            out.append(i)
            if len(out) >= int(limit):
                break
    return out


def _output_layer_abs_scores(model: nn.Module, grad_vec: torch.Tensor) -> torch.Tensor:
    linear = last_linear(model)
    params = [p for p in model.parameters() if p.requires_grad]
    offset = 0
    weight_scores = None
    bias_scores = None
    for p in params:
        n = p.numel()
        segment = grad_vec[offset:offset + n].detach().cpu()
        if p is linear.weight:
            g = segment.reshape_as(linear.weight).abs()
            weight_scores = g.mean(dim=1)
        elif linear.bias is not None and p is linear.bias:
            bias_scores = segment.reshape_as(linear.bias).abs()
        offset += n
    if weight_scores is None:
        raise ValueError("Could not locate classifier gradient")
    if bias_scores is None:
        bias_scores = torch.zeros_like(weight_scores)
    scores = weight_scores + bias_scores
    if not torch.isfinite(scores).all() or float(scores.sum().item()) <= 1e-12:
        scores = torch.ones_like(scores)
    return scores / scores.sum().clamp_min(1e-12)


def _infer_labels(scores: torch.Tensor, tau: float, known_count: int | None) -> set[int]:
    if known_count is not None:
        k = max(1, min(int(known_count), int(scores.numel())))
        return set(torch.topk(scores, k=k).indices.cpu().numpy().astype(int).tolist())
    z = (scores - scores.mean()) / scores.std(unbiased=False).clamp_min(1e-12)
    pred = set(torch.where(z > float(tau))[0].cpu().numpy().astype(int).tolist())
    if not pred:
        pred = {int(torch.argmax(scores).item())}
    return pred


def _iou(true: set[int], pred: set[int]) -> float:
    return float(len(true & pred) / max(1, len(true | pred)))


def _local_change(model, full_ds, retained_ds, device, config: ULIAConfig) -> torch.Tensor:
    before = copy.deepcopy(model)
    after = copy.deepcopy(model)
    train_model(before, make_loader(full_ds, config.batch_size, True), device, config.local_epochs, config.lr, config.momentum, config.weight_decay)
    if len(retained_ds) > 0:
        train_model(after, make_loader(retained_ds, config.batch_size, True), device, config.local_epochs, config.lr, config.momentum, config.weight_decay)
    return model_to_vector(after).detach().cpu() - model_to_vector(before).detach().cpu()


def _rapid_retrain(model, retained_ds, device, config: ULIAConfig):
    after = copy.deepcopy(model)
    train_model(after, make_loader(retained_ds, config.batch_size, True), device, config.unlearning_epochs, config.lr, config.momentum, config.weight_decay)
    return after


def _first_order_unlearn(model, forgotten_ds, retained_ds, device, config: ULIAConfig, ascent_scale: float, repair_epochs: int):
    grad = batched_gradient_vector(model, make_loader(forgotten_ds, config.batch_size, False), device)
    vec = model_to_vector(model).detach().cpu()
    after = copy.deepcopy(model)
    load_vector_into_model(after, vec + float(config.lr * config.unlearning_epochs * ascent_scale) * grad, model.state_dict())
    if repair_epochs > 0 and len(retained_ds) > 0:
        train_model(after, make_loader(retained_ds, config.batch_size, True), device, repair_epochs, config.lr, config.momentum, config.weight_decay)
    return after


def _federaser(model, forgotten_ds, device, config: ULIAConfig):
    grad = batched_gradient_vector(model, make_loader(forgotten_ds, config.batch_size, False), device)
    vec = model_to_vector(model).detach().cpu()
    after = copy.deepcopy(model)
    load_vector_into_model(after, vec + float(config.lr * config.unlearning_epochs) * grad, model.state_dict())
    return after


def _sga_ewc(model, forgotten_ds, retained_ds, device, config: ULIAConfig):
    return _first_order_unlearn(model, forgotten_ds, retained_ds, device, config, ascent_scale=1.5, repair_epochs=0)


def _unlearned_model(method: str, model, forgotten_ds, retained_ds, device, config: ULIAConfig):
    if method == "rapid_retrain":
        return _first_order_unlearn(model, forgotten_ds, retained_ds, device, config, ascent_scale=1.0, repair_epochs=0)
    if method == "federaser":
        return _federaser(model, forgotten_ds, device, config)
    if method == "sga_ewc":
        return _sga_ewc(model, forgotten_ds, retained_ds, device, config)
    raise ValueError(f"Unknown ULIA unlearning method: {method}")


def _trial_for_scenario(
    scenario: str,
    method: str,
    model,
    clients,
    train_ds,
    test_ds,
    device,
    config: ULIAConfig,
    rng: random.Random,
) -> dict[str, Any]:
    cid = rng.randrange(len(clients))
    client = clients[cid]
    labels_in_client = sorted({int(client[i][1]) for i in range(len(client))})
    chosen_labels = set(labels_in_client[: max(1, min(config.forgotten_label_count, len(labels_in_client)))])

    if scenario == "sample":
        target_positions = [i for i in range(len(client)) if int(client[i][1]) in chosen_labels]
        count = max(1, int(round(len(client) * float(config.forgotten_fraction))))
        target_positions = target_positions[:count]
        if not target_positions:
            target_positions = [0]
            chosen_labels = {int(client[0][1])}
        retained_positions = [i for i in range(len(client)) if i not in set(target_positions)]
        forgotten_ds = Subset(client, target_positions)
        retained_client = Subset(client, retained_positions)
        retained_global = ConcatDataset([retained_client] + [ds for j, ds in enumerate(clients) if j != cid])
        true_labels = _label_set(client, target_positions)
        full_for_local = client
        retained_for_local = retained_client
    elif scenario == "class":
        global_forget = _first_label_indices(train_ds, chosen_labels, max(1, int(len(train_ds) * float(config.forgotten_fraction))))
        retained_global_idx = [i for i in range(len(train_ds)) if i not in set(global_forget)]
        forgotten_ds = Subset(train_ds, global_forget)
        retained_global = Subset(train_ds, retained_global_idx)
        true_labels = _label_set(train_ds, global_forget)
        client_forget = [i for i in range(len(client)) if int(client[i][1]) in true_labels]
        retained_client_idx = [i for i in range(len(client)) if i not in set(client_forget)]
        full_for_local = client
        retained_for_local = Subset(client, retained_client_idx or list(range(len(client))))
    elif scenario == "client":
        target_positions = [i for i in range(len(client)) if int(client[i][1]) in chosen_labels]
        count = max(1, int(round(len(client) * float(config.forgotten_fraction))))
        target_positions = target_positions[:count] or [0]
        chosen_labels = {int(client[i][1]) for i in target_positions}
        retained_positions = [i for i in range(len(client)) if i not in set(target_positions)]
        forgotten_ds = Subset(client, target_positions)
        retained_client = Subset(client, retained_positions)
        retained_global = ConcatDataset([retained_client] + [ds for j, ds in enumerate(clients) if j != cid])
        true_labels = _label_set(client, target_positions)
        full_for_local = client
        retained_for_local = retained_client
    else:
        raise ValueError(scenario)

    after = _unlearned_model(method, model, forgotten_ds, retained_global, device, config)
    global_change = model_to_vector(after).detach().cpu() - model_to_vector(model).detach().cpu()
    local_change = _local_change(model, full_for_local, retained_for_local, device, config)
    eta = max(float(torch.norm(local_change).item() / (torch.norm(global_change).item() + 1e-12)), 1e-6)
    wk = len(full_for_local) / max(1, len(train_ds))
    approx_grad = (global_change - local_change) / max((1.0 - wk) * eta, 1e-6)
    if method in {"federaser", "rapid_retrain", "sga_ewc"}:
        scores = _output_layer_abs_scores(model, global_change)
    else:
        scores = _output_layer_abs_scores(model, approx_grad)
    known = len(true_labels) if config.attacker_knows_label_count else None
    pred = _infer_labels(scores, config.tau, known)
    return {
        "method": method,
        "scenario": scenario,
        "client_id": cid,
        "true_labels": sorted(true_labels),
        "predicted_labels": sorted(pred),
        "asr_iou": _iou(true_labels, pred),
        "eta_approx": eta,
        "post_clean_accuracy": evaluate_accuracy(after, make_loader(test_ds, config.batch_size, False), device),
        "scores": [float(x) for x in scores.tolist()],
    }


def run(config: ULIAConfig = ULIAConfig()) -> dict[str, Any]:
    start = time.perf_counter()
    set_seed(config.seed)
    device = get_device(config.device)
    out_dir = ensure_dir(Path(config.output_dir) / f"run-{now_stamp()}")

    train_ds, test_ds, info = load_dataset(
        config.dataset,
        root=config.data_root,
        train_size=config.train_size,
        test_size=config.test_size,
        seed=config.seed,
        download=config.download,
    )
    clients = partition_dataset(train_ds, config.num_clients, config.partition, config.dirichlet_alpha, config.seed)
    model = make_model(config.model, info).to(device)
    train_model(model, make_loader(train_ds, config.batch_size, True), device, config.pretrain_epochs, config.lr, config.momentum, config.weight_decay)
    clean_acc = evaluate_accuracy(model, make_loader(test_ds, config.batch_size, False), device)

    rng = random.Random(int(config.seed) + 991)
    rows: list[dict[str, Any]] = []
    for method in config.methods:
        for scenario in config.scenarios:
            for _ in range(int(config.attack_trials)):
                rows.append(_trial_for_scenario(scenario, method, model, clients, train_ds, test_ds, device, config, rng))

    summary: list[dict[str, Any]] = []
    for method in config.methods:
        for scenario in config.scenarios:
            vals = [r["asr_iou"] for r in rows if r["method"] == method and r["scenario"] == scenario]
            summary.append({
                "method": method,
                "scenario": scenario,
                "mean_asr_iou": float(np.mean(vals)) if vals else 0.0,
                "std_asr_iou": float(np.std(vals)) if vals else 0.0,
                "trials": len(vals),
            })

    save_csv(out_dir / "detailed_results.csv", rows)
    save_csv(out_dir / "summary.csv", summary)
    payload = {
        "paper": config.paper,
        "method": config.method,
        "config": asdict(config),
        "dataset_info": info,
        "clean_accuracy": clean_acc,
        "summary": summary,
        "detailed_results": rows,
        "metrics": {
            "overall_mean_asr_iou": float(np.mean([r["asr_iou"] for r in rows])) if rows else 0.0,
            "elapsed_sec": time.perf_counter() - start,
        },
        "output_dir": str(out_dir),
    }
    save_json(out_dir / "result.json", payload)
    save_json(Path(config.output_dir) / "result.json", payload)
    return payload
