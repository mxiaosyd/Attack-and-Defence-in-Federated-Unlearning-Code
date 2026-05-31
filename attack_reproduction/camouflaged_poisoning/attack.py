from __future__ import annotations

import copy
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import ConcatDataset, Dataset, Subset

from attack_reproduction.common_attack import (
    cosine_distance,
    ensure_dir,
    evaluate_accuracy,
    federated_train,
    get_device,
    gradient_vector,
    load_dataset,
    make_loader,
    make_model,
    now_stamp,
    partition_dataset,
    save_json,
    select_indices_by_label,
    set_seed,
)
from .config import CamouflagedPoisoningConfig


class CraftedDataset(Dataset):
    def __init__(self, xs: torch.Tensor, ys: torch.Tensor):
        self.xs = xs.detach().float().cpu()
        self.ys = ys.detach().long().cpu()

    def __len__(self) -> int:
        return int(self.ys.numel())

    def __getitem__(self, index: int):
        return self.xs[index], int(self.ys[index].item())


def _items_by_label(dataset, label: int, limit: int) -> tuple[torch.Tensor, torch.Tensor]:
    xs = []
    ys = []
    for i in range(len(dataset)):
        x, y = dataset[i]
        if int(y) == int(label):
            xs.append(x.float())
            ys.append(int(y))
            if len(xs) >= int(limit):
                break
    if not xs:
        raise ValueError(f"No samples found for label {label}")
    return torch.stack(xs), torch.tensor(ys, dtype=torch.long)


@torch.no_grad()
def _select_vulnerable_targets(model, dataset, target_label: int, adversarial_label: int, limit: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    candidates: list[tuple[float, torch.Tensor, int]] = []
    model.eval()
    for i in range(len(dataset)):
        x, y = dataset[i]
        if int(y) != int(target_label):
            continue
        logits = model(x.unsqueeze(0).to(device)).detach().cpu()[0]
        margin = float((logits[int(target_label)] - logits[int(adversarial_label)]).item())
        candidates.append((margin, x.float(), int(y)))
    if not candidates:
        raise ValueError(f"No target samples found for label {target_label}")
    candidates.sort(key=lambda row: row[0])
    chosen = candidates[: max(1, int(limit))]
    return torch.stack([x for _, x, _ in chosen]), torch.tensor([y for _, _, y in chosen], dtype=torch.long)


@torch.no_grad()
def _select_stable_vulnerable_targets(
    model,
    reference_model,
    dataset,
    target_label: int,
    adversarial_label: int,
    limit: int,
    device,
) -> tuple[torch.Tensor, torch.Tensor, list[float]]:
    candidates: list[tuple[float, torch.Tensor, int]] = []
    model.eval()
    reference_model.eval()
    for i in range(len(dataset)):
        x, y = dataset[i]
        if int(y) != int(target_label):
            continue
        x_dev = x.unsqueeze(0).to(device)
        logits = model(x_dev).detach().cpu()[0]
        ref_logits = reference_model(x_dev).detach().cpu()[0]
        if int(torch.argmax(logits).item()) != int(target_label):
            continue
        if int(torch.argmax(ref_logits).item()) != int(target_label):
            continue
        margin = float((logits[int(target_label)] - logits[int(adversarial_label)]).item())
        if margin <= 0:
            continue
        candidates.append((margin, x.float(), int(y)))
    if len(candidates) < int(limit):
        raise ValueError(f"Only found {len(candidates)} stable targets for {target_label}->{adversarial_label}")
    candidates.sort(key=lambda row: row[0])
    chosen = candidates[: max(1, int(limit))]
    return (
        torch.stack([x for margin, x, _ in chosen]),
        torch.tensor([y for _, _, y in chosen], dtype=torch.long),
        [float(margin) for margin, _, _ in chosen],
    )


def _batch_gradient(model, x: torch.Tensor, y: torch.Tensor, device, create_graph: bool) -> torch.Tensor:
    return gradient_vector(model, x.to(device), y.to(device), create_graph=create_graph, retain_graph=True)


def _project_delta(delta: torch.Tensor, epsilon: float) -> torch.Tensor:
    return delta.clamp(-float(epsilon), float(epsilon))


def _optimize_poison(
    model,
    base_x: torch.Tensor,
    base_y: torch.Tensor,
    target_x: torch.Tensor,
    target_adv_y: torch.Tensor,
    device,
    config: CamouflagedPoisoningConfig,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    target_grad = _batch_gradient(model, target_x, target_adv_y, device, create_graph=False).detach()
    best_x = base_x.clone()
    best_loss = float("inf")
    for restart in range(int(config.restarts)):
        delta = torch.empty_like(base_x, device=device).uniform_(-config.perturbation_epsilon, config.perturbation_epsilon)
        delta.requires_grad_(True)
        opt = torch.optim.Adam([delta], lr=float(config.optimize_lr))
        for _ in range(int(config.optimize_steps)):
            poisoned = (base_x.to(device) + delta).clamp(0.0, 1.0)
            grad = _batch_gradient(model, poisoned, base_y.to(device), device, create_graph=True)
            loss = cosine_distance(grad, target_grad)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            with torch.no_grad():
                delta.copy_(_project_delta(delta, config.perturbation_epsilon))
        poisoned_eval = (base_x.to(device) + delta.detach()).clamp(0.0, 1.0)
        loss_val = float(cosine_distance(_batch_gradient(model, poisoned_eval, base_y.to(device), device, False), target_grad).detach().cpu().item())
        if loss_val < best_loss:
            best_loss = loss_val
            best_x = poisoned_eval.detach().cpu()
    return best_x, base_y.detach().cpu(), best_loss


def _optimize_camouflage(
    model,
    base_x: torch.Tensor,
    base_y: torch.Tensor,
    poison_x: torch.Tensor,
    poison_y: torch.Tensor,
    device,
    config: CamouflagedPoisoningConfig,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    poison_grad = _batch_gradient(model, poison_x, poison_y, device, create_graph=False).detach()
    best_x = base_x.clone()
    best_loss = float("inf")
    for _ in range(int(config.restarts)):
        delta = torch.empty_like(base_x, device=device).uniform_(-config.perturbation_epsilon, config.perturbation_epsilon)
        delta.requires_grad_(True)
        opt = torch.optim.Adam([delta], lr=float(config.optimize_lr))
        for _ in range(int(config.optimize_steps)):
            camo = (base_x.to(device) + delta).clamp(0.0, 1.0)
            camo_grad = _batch_gradient(model, camo, base_y.to(device), device, create_graph=True)
            loss = cosine_distance(camo_grad, -poison_grad)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            with torch.no_grad():
                delta.copy_(_project_delta(delta, config.perturbation_epsilon))
        camo_eval = (base_x.to(device) + delta.detach()).clamp(0.0, 1.0)
        loss_val = float(cosine_distance(_batch_gradient(model, camo_eval, base_y.to(device), device, False), -poison_grad).detach().cpu().item())
        if loss_val < best_loss:
            best_loss = loss_val
            best_x = camo_eval.detach().cpu()
    return best_x, base_y.detach().cpu(), best_loss


def _optimize_camouflage_for_target_repair(
    model,
    base_x: torch.Tensor,
    base_y: torch.Tensor,
    target_x: torch.Tensor,
    target_y: torch.Tensor,
    device,
    config: CamouflagedPoisoningConfig,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    target_grad = _batch_gradient(model, target_x, target_y, device, create_graph=False).detach()
    best_x = base_x.clone()
    best_loss = float("inf")
    for _ in range(int(config.restarts)):
        delta = torch.empty_like(base_x, device=device).uniform_(-config.perturbation_epsilon, config.perturbation_epsilon)
        delta.requires_grad_(True)
        opt = torch.optim.Adam([delta], lr=float(config.optimize_lr))
        for _ in range(int(config.optimize_steps)):
            camo = (base_x.to(device) + delta).clamp(0.0, 1.0)
            camo_grad = _batch_gradient(model, camo, base_y.to(device), device, create_graph=True)
            loss = cosine_distance(camo_grad, target_grad)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            with torch.no_grad():
                delta.copy_(_project_delta(delta, config.perturbation_epsilon))
        camo_eval = (base_x.to(device) + delta.detach()).clamp(0.0, 1.0)
        loss_val = float(cosine_distance(_batch_gradient(model, camo_eval, base_y.to(device), device, False), target_grad).detach().cpu().item())
        if loss_val < best_loss:
            best_loss = loss_val
            best_x = camo_eval.detach().cpu()
    return best_x, base_y.detach().cpu(), best_loss


@torch.no_grad()
def _target_attack_success(model, targets: torch.Tensor, adversarial_label: int, device) -> float:
    model.eval()
    pred = model(targets.to(device)).argmax(dim=1).detach().cpu()
    return float((pred == int(adversarial_label)).float().mean().item())


def _repeat_dataset(dataset, repeats: int):
    return ConcatDataset([dataset for _ in range(max(1, int(repeats)))])


def _replace_malicious_clients(clients, poison_ds, camo_ds, malicious_count: int, include_camouflage: bool, poison_repeats: int, camouflage_repeats: int):
    out = list(clients)
    for cid in range(int(malicious_count)):
        parts = [out[cid], _repeat_dataset(poison_ds, poison_repeats)]
        if include_camouflage:
            parts.append(_repeat_dataset(camo_ds, camouflage_repeats))
        out[cid] = ConcatDataset(parts)
    return out


def _unlearn_camouflage_by_gradient_ascent(model, camouflage_ds, device, config: CamouflagedPoisoningConfig):
    after = copy.deepcopy(model).to(device)
    opt = torch.optim.SGD(
        after.parameters(),
        lr=float(config.unlearning_lr),
        momentum=float(config.momentum),
        weight_decay=float(config.weight_decay),
    )
    ce = nn.CrossEntropyLoss()
    for _ in range(int(config.unlearning_epochs)):
        for x, y in make_loader(camouflage_ds, config.batch_size, True):
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = -ce(after(x), y)
            loss.backward()
            opt.step()
    return after


def _train_from_clean_reference(clean_model, client_datasets, test_loader, target_x, adversarial_label, device, config: CamouflagedPoisoningConfig):
    model = copy.deepcopy(clean_model).to(device)
    federated_train(
        model,
        client_datasets,
        device,
        rounds=config.rounds,
        clients_per_round=config.clients_per_round,
        local_epochs=config.local_epochs,
        batch_size=config.batch_size,
        lr=config.lr,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
        aggregation=config.aggregation,
        seed=config.seed,
    )
    return (
        model,
        evaluate_accuracy(model, test_loader, device),
        _target_attack_success(model, target_x, adversarial_label, device),
    )


def run(config: CamouflagedPoisoningConfig = CamouflagedPoisoningConfig()) -> dict[str, Any]:
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
    clean_model = make_model(config.model, info).to(device)
    federated_train(
        clean_model,
        clients,
        device,
        rounds=config.clean_pretrain_rounds,
        clients_per_round=config.clients_per_round,
        local_epochs=config.local_epochs,
        batch_size=config.batch_size,
        lr=config.lr,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
        aggregation=config.aggregation,
        seed=config.seed,
    )

    test_loader = make_loader(test_ds, config.batch_size, False)
    clean_reference_model = copy.deepcopy(clean_model).to(device)
    federated_train(
        clean_reference_model,
        clients,
        device,
        rounds=config.rounds,
        clients_per_round=config.clients_per_round,
        local_epochs=config.local_epochs,
        batch_size=config.batch_size,
        lr=config.lr,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
        aggregation=config.aggregation,
        seed=config.seed,
    )
    if config.stable_target_reference:
        target_x, target_y, target_margins = _select_stable_vulnerable_targets(
            clean_model,
            clean_reference_model,
            train_ds,
            config.target_label,
            config.adversarial_label,
            config.target_eval_samples,
            device,
        )
    else:
        target_x, target_y = _select_vulnerable_targets(
            clean_model,
            train_ds,
            config.target_label,
            config.adversarial_label,
            config.target_eval_samples,
            device,
        )
        target_margins = []
    target_adv_y = torch.full((target_x.shape[0],), int(config.adversarial_label), dtype=torch.long)
    poison_base_x, poison_true_y = _items_by_label(train_ds, config.target_label, config.poison_budget)
    poison_base_y = torch.full_like(poison_true_y, int(config.adversarial_label))
    poison_x, poison_y, poison_loss = _optimize_poison(clean_model, poison_base_x, poison_base_y, target_x, target_adv_y, device, config)
    poison_ds = CraftedDataset(poison_x, poison_y)
    poison_generation_clients = _replace_malicious_clients(
        clients,
        poison_ds,
        poison_ds,
        config.malicious_clients,
        include_camouflage=False,
        poison_repeats=config.poison_repeats,
        camouflage_repeats=1,
    )
    poison_generation_model, _, _ = _train_from_clean_reference(
        clean_model,
        poison_generation_clients,
        test_loader,
        target_x,
        config.adversarial_label,
        device,
        config,
    )
    camo_base_x, camo_base_y = _items_by_label(train_ds, config.target_label, config.camouflage_budget)
    camo_x, camo_y, camo_loss = _optimize_camouflage_for_target_repair(
        poison_generation_model,
        camo_base_x,
        camo_base_y,
        target_x,
        target_y,
        device,
        config,
    )
    camo_ds = CraftedDataset(camo_x, camo_y)

    camouflaged_clients = _replace_malicious_clients(
        clients,
        poison_ds,
        camo_ds,
        config.malicious_clients,
        include_camouflage=True,
        poison_repeats=config.poison_repeats,
        camouflage_repeats=config.camouflage_repeats,
    )
    poison_only_clients = _replace_malicious_clients(
        clients,
        poison_ds,
        camo_ds,
        config.malicious_clients,
        include_camouflage=False,
        poison_repeats=config.poison_repeats,
        camouflage_repeats=config.camouflage_repeats,
    )

    camouflaged_model, before_clean_acc, before_target_asr = _train_from_clean_reference(
        clean_model,
        camouflaged_clients,
        test_loader,
        target_x,
        config.adversarial_label,
        device,
        config,
    )
    poison_reference_model, poison_only_clean_acc, poison_only_target_asr = _train_from_clean_reference(
        clean_model,
        poison_only_clients,
        test_loader,
        target_x,
        config.adversarial_label,
        device,
        config,
    )
    activated_model = poison_reference_model
    after_clean_acc = poison_only_clean_acc
    after_target_asr = poison_only_target_asr

    control_metrics: dict[str, float] = {}
    if config.include_no_poison_control:
        empty_poison_ds = CraftedDataset(poison_x[:0], poison_y[:0])
        camo_only_clients = _replace_malicious_clients(
            clients,
            empty_poison_ds,
            camo_ds,
            config.malicious_clients,
            include_camouflage=True,
            poison_repeats=1,
            camouflage_repeats=config.camouflage_repeats,
        )
        camo_only_model, camo_only_before_acc, camo_only_before_asr = _train_from_clean_reference(
            clean_model,
            camo_only_clients,
            test_loader,
            target_x,
            config.adversarial_label,
            device,
            config,
        )
        _, clean_only_after_acc, clean_only_after_asr = _train_from_clean_reference(
            clean_model,
            clients,
            test_loader,
            target_x,
            config.adversarial_label,
            device,
            config,
        )
        control_metrics = {
            "clean_reference_clean_accuracy": evaluate_accuracy(clean_reference_model, test_loader, device),
            "clean_reference_target_asr": _target_attack_success(clean_reference_model, target_x, config.adversarial_label, device),
            "no_poison_control_before_clean_accuracy": camo_only_before_acc,
            "no_poison_control_after_clean_accuracy": clean_only_after_acc,
            "no_poison_control_before_target_asr": camo_only_before_asr,
            "no_poison_control_after_target_asr": clean_only_after_asr,
        }

    torch.save(
        {
            "target": target_x,
            "target_labels": target_y,
            "target_margins": torch.tensor(target_margins),
            "poison": poison_x,
            "camouflage": camo_x,
            "poison_labels": poison_y,
            "camouflage_labels": camo_y,
        },
        out_dir / "crafted_sets.pt",
    )

    metrics = {
        "before_unlearning_clean_accuracy": before_clean_acc,
        "after_unlearning_clean_accuracy": after_clean_acc,
        "clean_accuracy_drop": before_clean_acc - after_clean_acc,
        "before_unlearning_target_asr": before_target_asr,
        "poison_only_clean_accuracy": poison_only_clean_acc,
        "poison_only_target_asr": poison_only_target_asr,
        "after_unlearning_target_asr": after_target_asr,
        "target_asr_increase": after_target_asr - before_target_asr,
        "poison_gradient_matching_loss": poison_loss,
        "camouflage_counter_gradient_loss": camo_loss,
        "unlearning_protocol": "exact_retrain_without_camouflage",
        "elapsed_sec": time.perf_counter() - start,
    }
    metrics.update(control_metrics)
    payload = {
        "paper": config.paper,
        "method": config.method,
        "config": asdict(config),
        "dataset_info": info,
        "metrics": metrics,
        "output_dir": str(out_dir),
    }
    save_json(out_dir / "result.json", payload)
    save_json(Path(config.output_dir) / "result.json", payload)
    return payload
