"""Unlearning Backdoor reproduction: UL-Subtract plus UL-Distill."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from common.metrics import evaluate_all
from common.recovery_base import RecoveryMethod, RecoveryResult, register
from common.utils import load_vector_into_model, model_to_vector


_EPS = 1e-12


@dataclass
class _KDStats:
    epochs: int
    batches: int
    mean_loss: float
    mean_kl: float
    mean_ce: float


def _get_arg(args, name: str, default):
    return getattr(args, name, default)


def _target_clients(context) -> Tuple[List[int], str]:
    detected = getattr(context, "detected_malicious_clients", None)
    if detected:
        return [int(cid) for cid in detected], "detected_malicious_clients"
    if bool(_get_arg(context.args, "unlearning_backdoor_use_oracle_malicious", False)):
        return [int(cid) for cid in context.malicious_clients], "explicit_oracle_malicious_clients"
    return [], "no_detector_output"


def _historical_attacker_contribution(context, target_clients: Sequence[int]) -> Tuple[torch.Tensor, Dict[int, torch.Tensor], Dict[str, float]]:
    """Compute historical attacker contribution."""
    removed = torch.zeros_like(context.poisoned_vector.detach().cpu())
    per_client: Dict[int, torch.Tensor] = {
        int(cid): torch.zeros_like(removed) for cid in target_clients
    }
    selected_rounds = 0
    removed_update_count = 0
    total_contribution_norm = 0.0

    for rec in context.history:
        weights: Mapping[int, float] = rec.get("client_weights", {})
        updates: Mapping[int, torch.Tensor] = rec.get("client_updates", {})
        selected_clients = list(rec.get("selected_clients", updates.keys()))
        if not selected_clients:
            continue

        total_weight = sum(float(weights.get(cid, 1.0)) for cid in selected_clients)
        if abs(total_weight) <= _EPS:
            continue

        any_removed_this_round = False
        for cid in target_clients:
            if cid not in updates:
                continue
            coeff = float(context.args.server_lr) * float(weights.get(cid, 1.0)) / total_weight
            contrib = coeff * updates[cid].detach().cpu()
            removed += contrib
            per_client[int(cid)] += contrib
            removed_update_count += 1
            any_removed_this_round = True
            total_contribution_norm += float(torch.norm(contrib.float()).item())
        if any_removed_this_round:
            selected_rounds += 1

    stats = {
        "removed_update_count": float(removed_update_count),
        "attacker_participating_rounds": float(selected_rounds),
        "sum_removed_update_norms": float(total_contribution_norm),
        "removed_norm": float(torch.norm(removed.float()).item()),
        "stored_client_summaries": float(len(per_client)),
    }
    return removed, per_client, stats


def _kd_remedy_clean_data(
    student: nn.Module,
    teacher: nn.Module,
    loader,
    device: torch.device,
    *,
    epochs: int,
    lr: float,
    temperature: float,
    alpha: float,
    weight_decay: float = 0.0,
    optimizer_name: str = "sgd",
) -> _KDStats:
    """Knowledge distillation on clean public data."""
    epochs = int(max(0, epochs))
    if epochs == 0:
        return _KDStats(epochs=0, batches=0, mean_loss=0.0, mean_kl=0.0, mean_ce=0.0)

    alpha = float(max(0.0, min(1.0, alpha)))
    T = float(max(_EPS, temperature))

    student.train()
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    if optimizer_name.lower() == "adam":
        opt = torch.optim.Adam(student.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        opt = torch.optim.SGD(student.parameters(), lr=lr, momentum=0.0, weight_decay=weight_decay)

    kl_loss = nn.KLDivLoss(reduction="batchmean")
    ce_loss = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_kl = 0.0
    total_ce = 0.0
    batches = 0

    for _ in range(epochs):
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            with torch.no_grad():
                teacher_logits = teacher(x)
                teacher_prob = torch.softmax(teacher_logits / T, dim=1)

            student_logits = student(x)
            kd = kl_loss(torch.log_softmax(student_logits / T, dim=1), teacher_prob) * (T ** 2)
            if alpha < 1.0:
                ce = ce_loss(student_logits, y)
                loss = alpha * kd + (1.0 - alpha) * ce
            else:
                ce = torch.zeros((), device=device)
                loss = kd

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            total_loss += float(loss.detach().cpu().item())
            total_kl += float(kd.detach().cpu().item())
            total_ce += float(ce.detach().cpu().item())
            batches += 1

    denom = max(1, batches)
    return _KDStats(
        epochs=epochs,
        batches=batches,
        mean_loss=total_loss / denom,
        mean_kl=total_kl / denom,
        mean_ce=total_ce / denom,
    )


@register
class UnlearningBackdoor(RecoveryMethod):
    """UL-Subtract plus clean KD."""

    name = "unlearning_backdoor"

    def recover(self, context):
        aggregation = str(context.args.aggregation).lower()
        aggregation_warning = ""
        if aggregation not in {"fedavg", "avg", "mean"}:
            aggregation_warning = (
                "paper_exact_subtraction_assumes_fedavg; "
                f"current_aggregation={context.args.aggregation}"
            )

        target_clients, target_source = _target_clients(context)
        removed, per_client_removed, subtract_stats = _historical_attacker_contribution(context, target_clients)
        subtracted_vec = context.poisoned_vector.detach().cpu().clone() - removed
        subtracted_model = context.new_model_from_vector(subtracted_vec)

        subtract_metrics = evaluate_all(
            subtracted_model,
            context.test_loader,
            context.device,
            target_label=context.target_label,
            trigger_size=context.trigger_size,
            trigger_pattern=getattr(context, "trigger_pattern", "square"),
            trigger_position=getattr(context, "trigger_position", "bottom_right"),
        )

        teacher = context.new_model_from_vector(context.poisoned_vector)
        alpha = float(_get_arg(context.args, "unlearning_backdoor_alpha", 1.0))
        epochs = int(_get_arg(context.args, "unlearning_backdoor_kd_epochs", context.args.repair_epochs))
        if epochs < 0:
            epochs = int(context.args.repair_epochs)
        lr = float(_get_arg(context.args, "unlearning_backdoor_kd_lr", context.args.repair_lr))
        if lr < 0:
            lr = float(context.args.repair_lr)
        temperature = float(_get_arg(context.args, "unlearning_backdoor_temperature", context.args.temperature))
        if temperature < 0:
            temperature = float(context.args.temperature)
        optimizer_name = str(_get_arg(context.args, "unlearning_backdoor_optimizer", "sgd"))
        kd_stats = _kd_remedy_clean_data(
            subtracted_model,
            teacher,
            context.public_loader,
            context.device,
            epochs=epochs,
            lr=lr,
            temperature=temperature,
            alpha=alpha,
            weight_decay=float(context.args.weight_decay),
            optimizer_name=optimizer_name,
        )

        per_client_norms = {
            f"removed_norm_client_{cid}": float(torch.norm(vec.float()).item())
            for cid, vec in per_client_removed.items()
        }

        extra = {
            "method_note": "paper_faithful_historical_update_subtraction_plus_clean_kd",
            "unlearning_backdoor_step1": "UL-Subtract",
            "unlearning_backdoor_step2": "UL-Distill",
            "unlearning_backdoor_alpha": alpha,
            "unlearning_backdoor_temperature": temperature,
            "unlearning_backdoor_kd_epochs": int(epochs),
            "unlearning_backdoor_kd_lr": lr,
            "unlearning_backdoor_optimizer": optimizer_name.lower(),
            "unlearning_backdoor_public_batches": int(kd_stats.batches),
            "unlearning_backdoor_kd_mean_loss": float(kd_stats.mean_loss),
            "unlearning_backdoor_kd_mean_kl": float(kd_stats.mean_kl),
            "unlearning_backdoor_kd_mean_ce": float(kd_stats.mean_ce),
            "ul_subtract_clean_acc": float(subtract_metrics["clean_acc"]),
            "ul_subtract_clean_loss": float(subtract_metrics["clean_loss"]),
            "ul_subtract_asr": float(subtract_metrics["asr"]),
            "removed_clients": ";".join(str(cid) for cid in target_clients),
            "removed_client_source": target_source,
            "aggregation_warning": aggregation_warning,
        }
        extra.update(subtract_stats)
        extra.update(per_client_norms)
        return RecoveryResult(self.name, subtracted_model, extra)
