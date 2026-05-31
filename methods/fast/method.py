"""FAST reproduction: subtract malicious history and repair on benchmark data."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader

from common.metrics import evaluate_accuracy, evaluate_loss
from common.recovery_base import RecoveryMethod, RecoveryResult, register
from common.training import finetune_clean


_EPS = 1e-12


@dataclass
class _FastStep:
    round_idx: int
    selected_clients: int
    malicious_selected: int
    contribution_norm: float
    prev_acc: float
    candidate_acc: float
    accepted: bool
    stopped: bool


def _get_arg(args, name: str, default):
    return getattr(args, name, default)


def _model_from_vec(context, vector: torch.Tensor):
    return context.new_model_from_vector(vector.detach().cpu())


def _target_clients(context) -> Tuple[List[int], str]:
    detected = getattr(context, "detected_malicious_clients", None)
    if detected:
        return [int(cid) for cid in detected], "detected_malicious_clients"
    if bool(_get_arg(context.args, "fast_use_oracle_malicious", False)):
        return [int(cid) for cid in context.malicious_clients], "explicit_oracle_malicious_clients"
    return [], "no_detector_output"


def _resolve_fast_extra_epochs(context) -> int:
    epochs = int(_get_arg(context.args, "fast_extra_epochs", context.args.repair_epochs))
    if epochs < 0:
        epochs = int(context.args.repair_epochs)
    return max(0, epochs)


def _resolve_fast_extra_lr(context) -> float:
    lr = float(_get_arg(context.args, "fast_extra_lr", context.args.repair_lr))
    if lr < 0:
        lr = float(context.args.repair_lr)
    return lr


@torch.no_grad()
def _benchmark_accuracy(context, vector: torch.Tensor) -> float:
    model = _model_from_vec(context, vector)
    return float(evaluate_accuracy(model, context.public_loader, context.device))


@torch.no_grad()
def _benchmark_loss(context, vector: torch.Tensor) -> float:
    model = _model_from_vec(context, vector)
    return float(evaluate_loss(model, context.public_loader, context.device))


def _train_extra_on_benchmark(model: nn.Module, context) -> None:
    """Run FAST benchmark repair."""
    epochs = _resolve_fast_extra_epochs(context)
    if epochs <= 0:
        return
    lr = _resolve_fast_extra_lr(context)
    weight_decay = float(_get_arg(context.args, "fast_extra_weight_decay", context.args.weight_decay))
    optimizer_name = str(_get_arg(context.args, "fast_extra_optimizer", "sgd")).lower()

    model.train()
    if optimizer_name == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.0, weight_decay=weight_decay)
    ce = nn.CrossEntropyLoss()
    for _ in range(epochs):
        for x, y in context.public_loader:
            x = x.to(context.device)
            y = y.to(context.device)
            opt.zero_grad(set_to_none=True)
            loss = ce(model(x), y)
            loss.backward()
            opt.step()


def _round_malicious_update_equal_clients(context, rec: Dict, target_clients: Sequence[int]) -> Tuple[torch.Tensor, int, int]:
    """Return FAST per-round malicious contribution."""
    selected = list(rec.get("selected_clients", []))
    n_selected = len(selected) if selected else len(rec.get("client_updates", {}))
    malicious_updates: List[torch.Tensor] = []
    for cid in target_clients:
        if cid in rec.get("client_updates", {}):
            malicious_updates.append(rec["client_updates"][cid].detach().cpu())
    j_selected = len(malicious_updates)
    if n_selected <= 0 or j_selected <= 0:
        return torch.zeros_like(context.poisoned_vector.detach().cpu()), j_selected, n_selected
    malicious_avg = torch.stack(malicious_updates, dim=0).mean(dim=0)
    contribution = float(context.args.server_lr) * (float(j_selected) / float(n_selected)) * malicious_avg
    return contribution.detach().cpu(), j_selected, n_selected


def _round_malicious_update_weighted(context, rec: Dict, target_clients: Sequence[int]) -> Tuple[torch.Tensor, int, int]:
    """Weighted FedAvg contribution variant."""
    updates = rec.get("client_updates", {})
    weights = rec.get("client_weights", {})
    total_weight = sum(float(w) for w in weights.values())
    contribution = torch.zeros_like(context.poisoned_vector.detach().cpu())
    j_selected = 0
    for cid in target_clients:
        if cid in updates:
            w = float(weights.get(cid, 1.0)) / max(total_weight, _EPS)
            contribution = contribution + float(context.args.server_lr) * w * updates[cid].detach().cpu()
            j_selected += 1
    return contribution.detach().cpu(), j_selected, len(updates)


def _format_steps(steps: Sequence[_FastStep]) -> str:
    chunks = []
    for s in steps:
        chunks.append(
            f"r={s.round_idx},N={s.selected_clients},J={s.malicious_selected},"
            f"norm={s.contribution_norm:.6g},prev_acc={s.prev_acc:.6f},"
            f"cur_acc={s.candidate_acc:.6f},accepted={int(s.accepted)},stopped={int(s.stopped)}"
        )
    return ";".join(chunks)


@register
class FAST(RecoveryMethod):
    """FAST recovery method."""

    name = "fast"

    def _sequential_unlearning(self, context) -> Tuple[torch.Tensor, Dict[str, object]]:
        current_vec = context.poisoned_vector.detach().cpu().clone()
        current_acc = _benchmark_accuracy(context, current_vec)
        target_clients, target_source = _target_clients(context)
        if not target_clients:
            return current_vec, {
                "fast_unlearning_order": str(_get_arg(context.args, "fast_unlearning_order", "forward")).lower(),
                "fast_weighted_contribution": 0,
                "fast_over_unlearning_tolerance": float(_get_arg(context.args, "fast_over_unlearning_tolerance", 0.0)),
                "fast_accepted_rounds": 0,
                "fast_considered_rounds": 0,
                "fast_stopped_early": 0,
                "fast_stop_round": -1,
                "fast_removed_norm": 0.0,
                "fast_benchmark_acc_before_extra": current_acc,
                "fast_benchmark_loss_before_extra": _benchmark_loss(context, current_vec),
                "fast_step_trace": "",
                "fast_target_clients": "",
                "fast_target_client_source": target_source,
            }

        direction = str(_get_arg(context.args, "fast_unlearning_order", "forward")).lower()
        records = list(context.history)
        if direction in {"reverse", "backward"}:
            records = list(reversed(records))

        use_weighted = bool(_get_arg(context.args, "fast_weighted_contribution", False))
        tolerance = float(_get_arg(context.args, "fast_over_unlearning_tolerance", 0.0))
        max_rounds = int(_get_arg(context.args, "fast_max_unlearning_rounds", -1))
        if max_rounds <= 0:
            max_rounds = len(records)

        accepted_rounds = 0
        stopped = False
        stop_round = -1
        total_removed = torch.zeros_like(current_vec)
        steps: List[_FastStep] = []
        patience = max(1, int(_get_arg(context.args, "fast_overunlearning_patience", 1)))
        bad_steps = 0
        disable_early_stop = bool(_get_arg(context.args, "fast_disable_early_stop", False))

        for idx, rec in enumerate(records[:max_rounds]):
            round_idx = int(rec.get("round", idx))
            if use_weighted:
                contribution, j_selected, n_selected = _round_malicious_update_weighted(context, rec, target_clients)
            else:
                contribution, j_selected, n_selected = _round_malicious_update_equal_clients(context, rec, target_clients)

            if j_selected <= 0 or float(torch.norm(contribution.float()).item()) <= _EPS:
                steps.append(_FastStep(
                    round_idx=round_idx,
                    selected_clients=n_selected,
                    malicious_selected=j_selected,
                    contribution_norm=0.0,
                    prev_acc=current_acc,
                    candidate_acc=current_acc,
                    accepted=False,
                    stopped=False,
                ))
                continue

            last_vec = current_vec.clone()
            last_acc = current_acc
            candidate_vec = current_vec - contribution
            candidate_acc = _benchmark_accuracy(context, candidate_vec)
            contribution_norm = float(torch.norm(contribution.float()).item())

            over_unlearned = (candidate_acc + tolerance) < last_acc
            if over_unlearned and not disable_early_stop:
                bad_steps += 1
                steps.append(_FastStep(
                    round_idx=round_idx,
                    selected_clients=n_selected,
                    malicious_selected=j_selected,
                    contribution_norm=contribution_norm,
                    prev_acc=last_acc,
                    candidate_acc=candidate_acc,
                    accepted=False,
                    stopped=bad_steps >= patience,
                ))
                if bad_steps >= patience:
                    current_vec = last_vec
                    stopped = True
                    stop_round = round_idx
                    break
                current_vec = candidate_vec.detach().cpu().clone()
                current_acc = candidate_acc
                total_removed = total_removed + contribution
                accepted_rounds += 1
                continue

            current_vec = candidate_vec.detach().cpu().clone()
            current_acc = candidate_acc
            bad_steps = 0
            total_removed = total_removed + contribution
            accepted_rounds += 1
            steps.append(_FastStep(
                round_idx=round_idx,
                selected_clients=n_selected,
                malicious_selected=j_selected,
                contribution_norm=contribution_norm,
                prev_acc=last_acc,
                candidate_acc=candidate_acc,
                accepted=True,
                stopped=False,
            ))

        before_extra_model = _model_from_vec(context, current_vec)
        before_extra_loss = float(evaluate_loss(before_extra_model, context.public_loader, context.device))
        before_extra_acc = float(evaluate_accuracy(before_extra_model, context.public_loader, context.device))

        extra = {
            "fast_unlearning_order": direction,
            "fast_weighted_contribution": int(use_weighted),
            "fast_over_unlearning_tolerance": tolerance,
            "fast_accepted_rounds": accepted_rounds,
            "fast_considered_rounds": len(steps),
            "fast_stopped_early": int(stopped),
            "fast_stop_round": stop_round,
            "fast_removed_norm": float(torch.norm(total_removed.float()).item()),
            "fast_benchmark_acc_before_extra": before_extra_acc,
            "fast_benchmark_loss_before_extra": before_extra_loss,
            "fast_step_trace": _format_steps(steps),
            "fast_target_clients": ";".join(str(cid) for cid in target_clients),
            "fast_target_client_source": target_source,
        }
        return current_vec, extra

    def recover(self, context):
        unlearn_vec, extra = self._sequential_unlearning(context)

        model = _model_from_vec(context, unlearn_vec)
        extra_epochs = _resolve_fast_extra_epochs(context)
        if extra_epochs > 0:
            _train_extra_on_benchmark(model, context)

        extra.update({
            "fast_extra_epochs": extra_epochs,
            "fast_extra_lr": _resolve_fast_extra_lr(context),
            "fast_extra_optimizer": str(_get_arg(context.args, "fast_extra_optimizer", "sgd")).lower(),
            "method_note": "fast_sequential_malicious_update_subtraction_over_unlearning_judge_benchmark_extra_training",
        })
        return RecoveryResult(self.name, model, extra)
