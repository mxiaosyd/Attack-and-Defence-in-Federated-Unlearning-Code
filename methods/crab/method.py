"""Crab reproduction with adaptive rollback and chained recovery."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.data import make_loader
from common.recovery_base import RecoveryMethod, RecoveryResult, register
from common.training import train_one_client
from common.utils import load_vector_into_model, model_to_vector, robust_aggregate, vector_to_state_dict


_EPS = 1e-12


@dataclass
class _CrabSelection:
    """Selected rounds and clients."""

    info_storage: Dict[int, List[int]]
    window_summaries: List[str]
    train_losses: List[float]


def _state_from_vector(context, vector: torch.Tensor) -> Dict[str, torch.Tensor]:
    return vector_to_state_dict(vector.detach().cpu(), context.template_state)


def _vector_from_state(context, state: Mapping[str, torch.Tensor]) -> torch.Tensor:
    model = context.model_fn().to(context.device)
    model.load_state_dict(dict(state), strict=True)
    return model_to_vector(model).detach().cpu()


def _model_parameters_from_vector(context, vector: torch.Tensor) -> List[torch.Tensor]:
    state = _state_from_vector(context, vector)
    return [p.detach().clone().cpu() for p in state.values()]


def _model_trajectory(context, vectors: Sequence[torch.Tensor]) -> List[List[torch.Tensor]]:
    return [_model_parameters_from_vector(context, vec) for vec in vectors]


def _official_select_round(context, start_epoch: int, gm_list: Sequence[torch.Tensor], p_rounds: float) -> List[int]:
    """Faithful adaptation of serverCrab.Crab.select_round."""
    k = int(len(gm_list) * float(p_rounds))
    gm_trajectory = _model_trajectory(context, gm_list)
    if len(gm_trajectory) < 2:
        return [int(start_epoch)]

    prior = gm_trajectory[0]
    kl_list: List[float] = []
    for now_traj in gm_trajectory[1:]:
        kl = torch.tensor(0.0)
        for module, prior_module in zip(now_traj, prior):
            module = module.detach().float().cpu()
            prior_module = prior_module.detach().float().cpu()
            log_x = F.log_softmax(module, dim=-1)
            y = F.softmax(prior_module, dim=-1)
            kl = kl + F.kl_div(log_x, y, reduction="sum")
        kl_list.append(float(kl.cpu().item()))
        prior = now_traj

    if k <= 0:
        return []
    sel_round = np.argsort(np.array(kl_list))[::-1]
    return (sel_round[:k] + int(start_epoch)).astype(int).tolist()


def _official_select_client_in_round(
    context,
    round_idx: int,
    gm_list: Sequence[torch.Tensor],
    start_epoch: int,
    x_clients: float,
) -> List[int]:
    """Faithful adaptation of serverCrab.Crab.select_client_in_round."""
    if round_idx < 0 or round_idx >= len(context.history):
        return []
    rec = context.history[round_idx]
    updates: Mapping[int, torch.Tensor] = rec.get("client_updates", {})
    if not updates:
        return []

    cm_ids = list(updates.keys())
    k = int(len(cm_ids) * float(x_clients))
    if k <= 0:
        return []

    weights = rec.get("client_weights", {})
    agg_update = robust_aggregate(
        [updates[cid].detach().cpu().float() for cid in cm_ids],
        [float(weights.get(cid, len(context.client_datasets[cid]))) for cid in cm_ids],
        rule=context.args.aggregation,
        trim_ratio=context.args.trim_ratio,
    ).detach().cpu().float()

    similarity: List[float] = []
    for cid in cm_ids:
        upd = updates[cid].detach().cpu().float().flatten()
        agg = agg_update.flatten()
        cos = torch.dot(upd, agg) / (torch.norm(upd) * torch.norm(agg) + _EPS)
        similarity.append(float(cos.cpu().item()))

    sel_client = np.argsort(np.array(similarity))[::-1]
    sel_client = sel_client[:k].tolist()
    return [int(cm_ids[sel]) for sel in sel_client]


def _client_train_loss(context, model_vec: torch.Tensor, client_ids: Sequence[int]) -> float:
    """Clean training loss over remaining clients, used for Crab loss windows."""
    if not client_ids:
        return 0.0
    model = context.model_fn().to(context.device)
    load_vector_into_model(model, model_vec.detach().cpu(), context.template_state)
    model.eval()
    ce = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    total = 0
    with torch.no_grad():
        for cid in client_ids:
            loader = make_loader(context.client_datasets[cid], batch_size=context.args.batch_size, shuffle=False)
            for x, y in loader:
                x = x.to(context.device)
                y = y.to(context.device)
                total_loss += float(ce(model(x), y).item())
                total += int(y.numel())
    return total_loss / max(1, total)


def _all_client_ids(context) -> List[int]:
    try:
        return [int(cid) for cid in range(len(context.client_datasets))]
    except Exception:
        return [int(cid) for cid in sorted(set(context.benign_clients) | set(context.malicious_clients))]


def _detected_clients_to_remove(context) -> Tuple[Set[int], str]:
    for name in (
        "detected_malicious_clients",
        "flagged_malicious_clients",
        "malicious_clients_detected",
        "clients_to_unlearn",
        "unlearn_clients",
        "forget_clients",
        "removed_clients",
    ):
        if hasattr(context, name):
            value = getattr(context, name)
            if value is not None:
                detected = {int(cid) for cid in value}
                if detected:
                    return detected, name
    if bool(getattr(context.args, "crab_use_oracle_malicious", False)):
        return {int(cid) for cid in context.malicious_clients}, "explicit_oracle_malicious_clients"
    return set(), "no_detector_output"


def _global_vector_at_epoch(context, epoch: int) -> torch.Tensor:
    if epoch <= 0:
        return context.history[0]["global_before"].detach().cpu()
    if epoch < len(context.history):
        return context.history[epoch]["global_before"].detach().cpu()
    return context.history[-1]["global_after"].detach().cpu()


def _official_train_with_select(context) -> _CrabSelection:
    """Select important rounds and clients."""
    alpha = float(getattr(context.args, "crab_loss_drop_alpha", 0.1))
    p_rounds = float(context.args.crab_select_round_ratio)
    x_clients = float(context.args.crab_select_client_ratio)

    gm_list: List[torch.Tensor] = []
    start_epoch = 0
    start_loss: float | None = None
    info_storage: Dict[int, List[int]] = {}
    window_summaries: List[str] = []
    train_losses: List[float] = []

    total_rounds = len(context.history)
    loss_clients = _all_client_ids(context)
    for i in range(total_rounds):
        current_gm = _global_vector_at_epoch(context, i)
        train_loss = _client_train_loss(context, current_gm, loss_clients)
        train_losses.append(train_loss)

        if i == 0:
            start_loss = float(train_loss)
        else:
            gm_list.append(current_gm.detach().cpu().clone())

        is_final = i == total_rounds - 1
        if start_loss is not None and (train_loss < start_loss * (1.0 - alpha) or is_final):
            rounds = _official_select_round(context, start_epoch, gm_list, p_rounds)
            valid_rounds: List[int] = []
            for selected_round in rounds:
                selected_round = int(selected_round)
                if selected_round < 0 or selected_round >= total_rounds:
                    continue
                clients_id = _official_select_client_in_round(
                    context,
                    selected_round,
                    gm_list,
                    start_epoch,
                    x_clients,
                )
                info_storage[selected_round] = clients_id
                valid_rounds.append(selected_round)

            window_summaries.append(
                f"start={start_epoch},end={i},start_loss={start_loss:.6f},"
                f"end_loss={train_loss:.6f},selected={','.join(map(str, valid_rounds))}"
            )
            start_loss = float(train_loss)
            gm_list = []
            start_epoch = i

    return _CrabSelection(info_storage=dict(sorted(info_storage.items())), window_summaries=window_summaries, train_losses=train_losses)


def _select_adaptive_rollback(
    context,
    info_storage: Mapping[int, Sequence[int]],
    removed_clients: Set[int],
) -> Tuple[torch.Tensor, int, Dict[str, object]]:
    """Crab adaptive rollback using cumulative sensitivity bound beta.

    For each selected round, compare the aggregate update before and after
    removing detected malicious clients. The latest round whose cumulative
    malicious sensitivity stays below beta times retained-client update norm is
    used as the rollback point.
    """
    selected_rounds = [int(r) for r in sorted(info_storage.keys()) if 0 <= int(r) < len(context.history)]
    beta = float(getattr(context.args, "crab_sensitivity_threshold", 0.3))
    if not selected_rounds:
        return context.poisoned_vector.detach().cpu().clone(), -1, {
            "crab_adaptive_rollback_round": -1,
            "crab_sensitivity_threshold": beta,
            "crab_cumulative_sensitivity": 0.0,
            "crab_cumulative_retained_norm": 0.0,
            "crab_rollback_policy": "no_selected_round_return_poisoned",
        }

    cumulative_sensitivity = 0.0
    cumulative_retained_norm = 0.0
    rollback_round = selected_rounds[0]
    considered = 0

    for round_idx in selected_rounds:
        rec = context.history[round_idx]
        updates: Mapping[int, torch.Tensor] = rec.get("client_updates", {})
        weights: Mapping[int, float] = rec.get("client_weights", {})
        stored_clients = [int(cid) for cid in info_storage.get(round_idx, []) if int(cid) in updates]
        retained_clients = [cid for cid in stored_clients if cid not in removed_clients]
        if not stored_clients or not retained_clients:
            continue

        full_updates = [updates[cid].detach().cpu().float() for cid in stored_clients]
        full_weights = [float(weights.get(cid, len(context.client_datasets[cid]))) for cid in stored_clients]
        retained_updates = [updates[cid].detach().cpu().float() for cid in retained_clients]
        retained_weights = [float(weights.get(cid, len(context.client_datasets[cid]))) for cid in retained_clients]

        full_agg = robust_aggregate(
            full_updates,
            full_weights,
            rule=context.args.aggregation,
            trim_ratio=context.args.trim_ratio,
        )
        retained_agg = robust_aggregate(
            retained_updates,
            retained_weights,
            rule=context.args.aggregation,
            trim_ratio=context.args.trim_ratio,
        )
        cumulative_sensitivity += float(torch.norm((full_agg - retained_agg).float()).item())
        cumulative_retained_norm += float(torch.norm(retained_agg.float()).item())
        considered += 1

        if cumulative_sensitivity <= beta * max(cumulative_retained_norm, _EPS):
            rollback_round = int(round_idx)

    rollback_vec = _global_vector_at_epoch(context, rollback_round).detach().cpu().clone()
    return rollback_vec, int(rollback_round), {
        "crab_adaptive_rollback_round": int(rollback_round),
        "crab_sensitivity_threshold": beta,
        "crab_cumulative_sensitivity": float(cumulative_sensitivity),
        "crab_cumulative_retained_norm": float(cumulative_retained_norm),
        "crab_sensitivity_considered_rounds": int(considered),
        "crab_rollback_policy": "latest_round_under_cumulative_beta_bound",
    }


def _train_one_step_from_vector(context, start_vec: torch.Tensor, client_id: int) -> torch.Tensor:
    """Train one selected client."""
    model = context.model_fn().to(context.device)
    load_vector_into_model(model, start_vec.detach().cpu(), context.template_state)
    loader = make_loader(context.client_datasets[client_id], batch_size=context.args.batch_size, shuffle=True)
    train_one_client(
        model,
        loader,
        context.device,
        epochs=1,
        lr=context.args.lr,
        weight_decay=context.args.weight_decay,
        optimizer_name=getattr(context.args, "optimizer", "sgd"),
        attack="none",
        target_label=context.args.target_label,
        poison_fraction=0.0,
        trigger_size=context.args.trigger_size,
        trigger_pattern=getattr(context.args, "trigger_pattern", "square"),
        trigger_position=getattr(context.args, "trigger_position", "bottom_right"),
        num_classes=context.num_classes,
    )
    return model_to_vector(model).detach().cpu()


def _aggregate_models_official(context, model_vecs: Sequence[torch.Tensor], client_ids: Sequence[int]) -> torch.Tensor:
    """Aggregate selected model vectors."""
    if not model_vecs:
        raise ValueError("Cannot aggregate an empty model list")

    rule = str(context.args.aggregation).lower()
    states = [_state_from_vector(context, v) for v in model_vecs]
    keys = list(context.template_state.keys())

    if rule in {"fedavg", "avg", "mean"}:
        weights = torch.tensor([float(len(context.client_datasets[cid])) for cid in client_ids], dtype=torch.float64)
        weights = weights / weights.sum().clamp_min(1e-12)
        out: Dict[str, torch.Tensor] = {}
        for k in keys:
            vals = [s[k].detach().cpu() for s in states]
            if not torch.is_floating_point(vals[0]):
                out[k] = vals[0].clone()
                continue
            acc = torch.zeros_like(vals[0], dtype=torch.float32)
            for w, val in zip(weights, vals):
                acc += val.float() * float(w.item())
            out[k] = acc.to(dtype=vals[0].dtype)
        return _vector_from_state(context, out)

    if rule in {"median"}:
        out = {}
        for k in keys:
            vals = [s[k].detach().cpu() for s in states]
            if not torch.is_floating_point(vals[0]):
                out[k] = vals[0].clone()
                continue
            stacked = torch.stack([v.float() for v in vals], dim=0)
            out[k] = torch.median(stacked, dim=0).values.to(dtype=vals[0].dtype)
        return _vector_from_state(context, out)

    if rule in {"trimmedmean", "trimmed_mean", "trim"}:
        n = len(states)
        trim_default = 2
        trim = int(getattr(context.args, "crab_trimmed_clients_num", trim_default))
        trim = max(0, min(trim, n - 1))
        keep = max(1, n - trim)
        out = {}
        for k in keys:
            vals = [s[k].detach().cpu() for s in states]
            if not torch.is_floating_point(vals[0]):
                out[k] = vals[0].clone()
                continue
            stacked = torch.stack([v.float() for v in vals], dim=0)
            med = torch.median(stacked, dim=0, keepdim=True).values
            distances = torch.mean(torch.abs(med - stacked).view(n, -1), dim=1)
            indices = torch.argsort(distances)[:keep]
            out[k] = torch.mean(stacked[indices], dim=0).to(dtype=vals[0].dtype)
        return _vector_from_state(context, out)

    if rule == "krum":
        flat = torch.stack([v.detach().cpu().float() for v in model_vecs], dim=0)
        n = flat.shape[0]
        f = int(max(0, min((n - 2) // 2, np.floor(float(context.args.trim_ratio) * n))))
        scores = []
        for i in range(n):
            dists = torch.sum((flat - flat[i]) ** 2, dim=1)
            nearest = torch.topk(dists, k=max(1, n - f - 2), largest=False).values
            scores.append(float(nearest.sum().item()))
        return flat[int(np.argmin(scores))].detach().cpu()

    raise ValueError(f"Unknown aggregation rule: {context.args.aggregation}")


def _official_unlearning_step_once(
    context,
    old_client_vecs: Sequence[torch.Tensor],
    new_client_vecs: Sequence[torch.Tensor],
    global_model_before_forget: torch.Tensor,
    global_model_after_forget: torch.Tensor,
) -> torch.Tensor:
    """FedEraser-style calibration step."""
    assert len(old_client_vecs) == len(new_client_vecs)
    if len(old_client_vecs) == 0:
        return global_model_after_forget.detach().cpu().clone()

    old_states = [_state_from_vector(context, v) for v in old_client_vecs]
    new_states = [_state_from_vector(context, v) for v in new_client_vecs]
    old_global_state = _state_from_vector(context, global_model_before_forget)
    new_global_state = _state_from_vector(context, global_model_after_forget)

    return_state: Dict[str, torch.Tensor] = {}
    for layer in old_global_state.keys():
        ng = new_global_state[layer].detach().cpu()
        if not torch.is_floating_point(ng):
            return_state[layer] = ng.clone()
            continue

        old_param_update = torch.zeros_like(old_global_state[layer].detach().float().cpu())
        new_param_update = torch.zeros_like(old_global_state[layer].detach().float().cpu())
        for ii in range(len(new_states)):
            old_param_update += old_states[ii][layer].detach().float().cpu()
            new_param_update += new_states[ii][layer].detach().float().cpu()
        old_param_update /= len(new_states)
        new_param_update /= len(new_states)

        old_param_update = old_param_update - old_global_state[layer].detach().float().cpu()
        new_param_update = new_param_update - new_global_state[layer].detach().float().cpu()

        step_length = torch.norm(old_param_update)
        denom = torch.norm(new_param_update)
        if float(denom.item()) <= _EPS:
            return_state[layer] = ng.clone()
        else:
            step_direction = new_param_update / denom
            return_state[layer] = (ng.float() + step_length * step_direction).to(dtype=ng.dtype)

    return _vector_from_state(context, return_state)


@register
class Crab(RecoveryMethod):
    """Crab recovery method."""

    name = "crab"

    def recover(self, context):
        selection = _official_train_with_select(context)
        info_storage = selection.info_storage
        removed_clients, removed_source = _detected_clients_to_remove(context)

        if not info_storage:
            model = context.new_model_from_vector(context.poisoned_vector)
            return RecoveryResult(self.name, model, {
                "selected_rounds": 0,
                "used_clients": 0,
                "skipped_rounds": 0,
                "crab_windows": ";".join(selection.window_summaries),
                "crab_removed_client_source": removed_source,
                "crab_removed_clients": ",".join(str(cid) for cid in sorted(removed_clients)),
                "method_note": "official_crab_no_round_selected_return_poisoned",
            })

        current_vec, rollback_round, rollback_meta = _select_adaptive_rollback(context, info_storage, removed_clients)
        used_clients_total = 0
        skipped_rounds = 0
        per_round_clients: Dict[str, str] = {}
        per_round_remaining_clients: Dict[str, str] = {}
        calibration_rounds = 0

        for global_round, selected_ids in sorted(info_storage.items()):
            if global_round < rollback_round:
                skipped_rounds += 1
                continue
            if global_round < 0 or global_round >= len(context.history):
                skipped_rounds += 1
                continue

            rec = context.history[global_round]
            selected_ids = [int(cid) for cid in selected_ids]
            remaining_selected_ids = [cid for cid in selected_ids if cid not in removed_clients]
            local_models: Mapping[int, torch.Tensor] = rec.get("local_model_vectors", {})
            actual_client_ids = [cid for cid in remaining_selected_ids if cid in local_models]

            per_round_clients[str(global_round)] = ",".join(map(str, selected_ids))
            per_round_remaining_clients[str(global_round)] = ",".join(map(str, actual_client_ids))

            if not actual_client_ids:
                skipped_rounds += 1
                continue

            old_global_vec = rec["global_before"].detach().cpu()
            updates: Mapping[int, torch.Tensor] = rec.get("client_updates", {})
            weights: Mapping[int, float] = rec.get("client_weights", {})

            calibrated_updates: List[torch.Tensor] = []
            calibrated_weights: List[float] = []
            for cid in actual_client_ids:
                renewed_model_vec = _train_one_step_from_vector(context, current_vec, cid)
                renewed_update = renewed_model_vec.detach().cpu().float() - current_vec.detach().cpu().float()
                old_update = updates[cid].detach().cpu().float()
                renewed_norm = torch.norm(renewed_update)
                old_norm = torch.norm(old_update)
                if float(renewed_norm.item()) <= _EPS:
                    calibrated = torch.zeros_like(renewed_update)
                else:
                    calibrated = float(old_norm.item()) * renewed_update / renewed_norm
                calibrated_updates.append(calibrated.detach().cpu())
                calibrated_weights.append(float(weights.get(cid, len(context.client_datasets[cid]))))

            calibrated_agg = robust_aggregate(
                calibrated_updates,
                calibrated_weights,
                rule=context.args.aggregation,
                trim_ratio=context.args.trim_ratio,
            )
            current_vec = (current_vec.detach().cpu().float() + float(context.args.server_lr) * calibrated_agg.detach().cpu().float()).detach().cpu()
            calibration_rounds += 1
            used_clients_total += len(actual_client_ids)

        model = context.new_model_from_vector(current_vec)
        return RecoveryResult(self.name, model, {
            **rollback_meta,
            "selected_rounds": len(info_storage),
            "selected_round_ids": ",".join(str(r) for r in sorted(info_storage.keys())),
            "used_clients": used_clients_total,
            "skipped_rounds": skipped_rounds,
            "calibration_rounds": calibration_rounds,
            "selected_clients_by_round": ";".join(f"{r}:{ids}" for r, ids in per_round_clients.items()),
            "remaining_selected_clients_by_round": ";".join(f"{r}:{ids}" for r, ids in per_round_remaining_clients.items()),
            "crab_windows": ";".join(selection.window_summaries),
            "crab_num_windows": len(selection.window_summaries),
            "crab_train_loss_first": selection.train_losses[0] if selection.train_losses else "",
            "crab_train_loss_last": selection.train_losses[-1] if selection.train_losses else "",
            "crab_select_round_ratio": float(context.args.crab_select_round_ratio),
            "crab_select_client_ratio": float(context.args.crab_select_client_ratio),
            "crab_loss_drop_alpha": float(getattr(context.args, "crab_loss_drop_alpha", 0.1)),
            "crab_removed_client_source": removed_source,
            "crab_removed_clients": ",".join(str(cid) for cid in sorted(removed_clients)),
            "crab_recovery_mode": "adaptive_rollback_chained_selective_recovery",
            "method_note": "official_crab_selection_adaptive_rollback_norm_calibrated_update_recovery",
        })
