"""FedUP reproduction: detected-client pruning and recovery."""
from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Set, Tuple

import torch
from torch import nn

from common.data import make_loader
from common.recovery_base import RecoveryMethod, RecoveryResult, register
from common.utils import load_vector_into_model, model_to_vector, robust_aggregate


_EPS = 1e-12


def _get_arg(args: Any, name: str, default: Any) -> Any:
    return getattr(args, name, default)


def _as_int_set(values: Any) -> Set[int]:
    if values is None:
        return set()
    if isinstance(values, Mapping):
        values = values.keys()
    out: Set[int] = set()
    try:
        iterator = list(values)
    except TypeError:
        iterator = [values]
    for x in iterator:
        try:
            out.add(int(x))
        except Exception:
            continue
    return out


def _first_present_mapping(*maps: Mapping[str, Any], names: Sequence[str]) -> Any:
    for mp in maps:
        for name in names:
            if isinstance(mp, Mapping) and name in mp and mp[name] is not None:
                return mp[name]
    return None


def _first_present_attr(obj: Any, names: Sequence[str]) -> Any:
    for name in names:
        if hasattr(obj, name):
            value = getattr(obj, name)
            if value is not None:
                return value
    return None


def _all_clients(context: Any) -> Set[int]:
    for name in ("all_clients", "client_ids", "clients"):
        value = _first_present_attr(context, (name,))
        if value is not None:
            s = _as_int_set(value)
            if s:
                return s
    if hasattr(context, "client_datasets"):
        try:
            return _as_int_set(context.client_datasets.keys())
        except AttributeError:
            return set(range(len(context.client_datasets)))
    return _as_int_set(_first_present_attr(context, ("benign_clients",))) | _as_int_set(
        _first_present_attr(context, ("malicious_clients",))
    )


def _detected_unlearn_clients(context: Any, rec: Mapping[str, Any]) -> Tuple[Set[int], str]:
    """Resolve clients flagged for unlearning."""
    names = (
        "detected_malicious_clients",
        "flagged_malicious_clients",
        "malicious_clients_detected",
        "clients_to_unlearn",
        "unlearn_clients",
        "forget_clients",
        "removed_clients",
    )
    value = _first_present_mapping(rec, names=names)
    source = "record_detector_output"
    if value is None:
        value = _first_present_attr(context, names)
        source = "context_detector_output"
    pending = _first_present_attr(context, ("pending_detected_malicious_clients", "pending_unlearn_clients"))
    unlearn = _as_int_set(value) | _as_int_set(pending)
    if unlearn:
        return unlearn, source

    if bool(_get_arg(context.args, "fedup_use_oracle_malicious", False)):
        return _as_int_set(_first_present_attr(context, ("malicious_clients",))), "explicit_oracle_malicious_clients"
    return set(), "no_detector_output"


def _remaining_clients(context: Any, rec: Mapping[str, Any], unlearn_clients: Set[int]) -> Tuple[Set[int], str]:
    names = (
        "remaining_clients",
        "recovery_clients",
        "detected_benign_clients",
        "clients_remaining_after_detection",
        "retained_clients",
    )
    value = _first_present_mapping(rec, names=names)
    source = "record_remaining_clients"
    if value is None:
        value = _first_present_attr(context, names)
        source = "context_remaining_clients"
    remaining = _as_int_set(value)
    if remaining:
        return remaining - unlearn_clients, source

    all_clients = _all_clients(context)
    if all_clients:
        return all_clients - unlearn_clients, "all_clients_minus_detected"

    benign = _as_int_set(_first_present_attr(context, ("benign_clients",)))
    return benign - unlearn_clients, "ground_truth_benign_fallback"


def _record_round(rec: Mapping[str, Any], default: int) -> int:
    try:
        return int(rec.get("round", default))
    except Exception:
        return int(default)


def _record_client_weights(context: Any, rec: Mapping[str, Any], cid: int) -> float:
    weights = rec.get("client_weights", {}) if isinstance(rec, Mapping) else {}
    if isinstance(weights, Mapping) and cid in weights:
        try:
            return float(weights[cid])
        except Exception:
            pass
    if hasattr(context, "client_datasets"):
        try:
            return float(len(context.client_datasets[cid]))
        except Exception:
            pass
    return 1.0


def _weighted_average(vectors: Sequence[torch.Tensor], weights: Sequence[float] | None = None) -> torch.Tensor:
    """FedAvg-style weighted average of model vectors."""
    if not vectors:
        raise ValueError("Cannot average an empty vector list")
    if weights is None:
        weights = [1.0] * len(vectors)
    return robust_aggregate([v.detach().cpu().float() for v in vectors], weights, rule="fedavg")


def _vector_slices(template_state: Mapping[str, torch.Tensor]) -> Dict[str, slice]:
    slices: Dict[str, slice] = {}
    offset = 0
    for name, tensor in template_state.items():
        n = int(tensor.numel())
        slices[name] = slice(offset, offset + n)
        offset += n
    return slices


def _prunable_weight_slices(template_state: Mapping[str, torch.Tensor]) -> List[Tuple[str, slice, torch.Size]]:
    """Return Conv/Linear-like weight slices."""
    slices = _vector_slices(template_state)
    out: List[Tuple[str, slice, torch.Size]] = []
    for name, tensor in template_state.items():
        if tensor.is_floating_point() and name.endswith(".weight") and tensor.ndim >= 2:
            out.append((name, slices[name], tensor.shape))
    return out


def _last_prunable_layer_slice(template_state: Mapping[str, torch.Tensor]) -> Tuple[str, slice, torch.Size] | None:
    layers = _prunable_weight_slices(template_state)
    return layers[-1] if layers else None


def _extract_model_mapping(rec: Mapping[str, Any]) -> Mapping[int, torch.Tensor] | None:
    for key in ("local_model_vectors", "client_model_vectors", "local_models", "client_models"):
        value = rec.get(key)
        if isinstance(value, Mapping) and value:
            return {int(cid): vec for cid, vec in value.items()}
    return None


def _global_before_vector(context: Any, rec: Mapping[str, Any]) -> Tuple[torch.Tensor, str]:
    if isinstance(rec, Mapping) and rec.get("global_before") is not None:
        return rec["global_before"].detach().cpu().float(), "record_global_before"
    if isinstance(rec, Mapping) and rec.get("global_model_before") is not None:
        return rec["global_model_before"].detach().cpu().float(), "record_global_model_before"
    if hasattr(context, "global_before_vector"):
        return context.global_before_vector.detach().cpu().float(), "context_global_before_vector"
    if hasattr(context, "poisoned_vector"):
        return context.poisoned_vector.detach().cpu().float(), "poisoned_vector_fallback"
    raise ValueError("FedUP requires a global_before vector or context.poisoned_vector")


def _local_models_from_record(context: Any, rec: Mapping[str, Any]) -> Tuple[Dict[int, torch.Tensor], str, str]:
    """Return or reconstruct local model vectors from a record."""
    direct = _extract_model_mapping(rec)
    if direct is not None:
        return {cid: v.detach().cpu().float() for cid, v in direct.items()}, "local_model_vectors", "model"

    updates = rec.get("client_updates", {}) if isinstance(rec, Mapping) else {}
    if not isinstance(updates, Mapping) or not updates:
        return {}, "missing_local_models", "none"

    global_before, _ = _global_before_vector(context, rec)
    convention = str(_get_arg(context.args, "fedup_update_convention", "delta")).lower()
    if convention in {"model", "local_model", "weights", "weight"}:
        return {int(cid): upd.detach().cpu().float() for cid, upd in updates.items()}, "client_updates_as_models", convention

    lr = float(_get_arg(context.args, "fedup_gradient_lr", _get_arg(context.args, "server_lr", _get_arg(context.args, "lr", 1.0))))
    out: Dict[int, torch.Tensor] = {}
    for cid, upd in updates.items():
        u = upd.detach().cpu().float()
        if convention in {"delta", "model_delta", "weight_delta", "update"}:
            out[int(cid)] = global_before + u
        elif convention in {"gradient", "grad", "sgd_gradient"}:
            out[int(cid)] = global_before - lr * u
        else:
            raise ValueError(
                "Unsupported fedup_update_convention. Use 'model', 'delta', or 'gradient'."
            )
    return out, "reconstructed_from_client_updates", convention


def _selected_clients_from_record(rec: Mapping[str, Any], local_models: Mapping[int, torch.Tensor]) -> Set[int]:
    selected = _as_int_set(rec.get("selected_clients"))
    if selected:
        return selected
    if local_models:
        return set(local_models.keys())
    weights = rec.get("client_weights", {}) if isinstance(rec, Mapping) else {}
    return _as_int_set(weights)


def _find_last_usable_record(
    context: Any,
    unlearn_clients: Set[int],
    remaining_clients: Set[int],
) -> Tuple[Mapping[str, Any] | None, Dict[int, torch.Tensor], Set[int], Dict[str, Any]]:
    """Find the latest record containing both client groups."""
    history = list(getattr(context, "history", []))
    meta: Dict[str, Any] = {}
    for idx in range(len(history) - 1, -1, -1):
        rec = history[idx]
        local_models, model_source, convention = _local_models_from_record(context, rec)
        if not local_models:
            continue
        selected = _selected_clients_from_record(rec, local_models)
        usable_unlearn = (unlearn_clients & selected) & set(local_models.keys())
        usable_remaining = (remaining_clients & selected) & set(local_models.keys())
        if usable_unlearn and usable_remaining:
            meta.update({
                "fedup_reference_history_index": int(idx),
                "fedup_reference_round": _record_round(rec, idx),
                "fedup_local_model_source": model_source,
                "fedup_update_convention": convention,
            })
            return rec, local_models, selected, meta
    return None, {}, set(), meta


def _mean_pairwise_cosine(vectors: Sequence[torch.Tensor], sl: slice) -> float:
    """Mean pairwise cosine similarity among clients on a selected layer."""
    if len(vectors) < 2:
        return 1.0
    flat = [v.detach().float().cpu()[sl].flatten() for v in vectors]
    vals: List[float] = []
    for i in range(len(flat)):
        for j in range(i + 1, len(flat)):
            denom = torch.norm(flat[i]) * torch.norm(flat[j])
            if float(denom.item()) <= _EPS:
                continue
            vals.append(float(torch.dot(flat[i], flat[j]).item() / float(denom.item())))
    if not vals:
        return 1.0
    return float(sum(vals) / len(vals))


def _similarity_vectors(
    benign_vectors: Sequence[torch.Tensor],
    global_before: torch.Tensor,
    source: str,
) -> List[torch.Tensor]:
    """Choose model or update vectors for the paper's similarity guideline."""
    if source.lower() in {"update", "delta", "model_delta"}:
        return [(v.detach().cpu().float() - global_before.detach().cpu().float()) for v in benign_vectors]
    return [v.detach().cpu().float() for v in benign_vectors]


def _paper_auto_prune_ratio(
    context: Any,
    benign_vectors: Sequence[torch.Tensor],
    global_before: torch.Tensor,
) -> Tuple[float, Dict[str, float | str]]:
    """FedUP similarity-based pruning ratio."""
    args = context.args
    p_min = float(_get_arg(args, "fedup_p_min", 0.01))
    p_max = float(_get_arg(args, "fedup_p_max", 0.15))
    gamma = float(_get_arg(args, "fedup_gamma", 5.0))
    sim_min = float(_get_arg(args, "fedup_similarity_min", 0.5))
    sim_max = float(_get_arg(args, "fedup_similarity_max", 1.0))
    sim_source = str(_get_arg(args, "fedup_similarity_source", _get_arg(args, "fedup_similarity_mode", "update")))

    last_layer = _last_prunable_layer_slice(context.template_state)
    if last_layer is None:
        similarity = 1.0
    else:
        _, sl, _ = last_layer
        sim_vecs = _similarity_vectors(benign_vectors, global_before, sim_source)
        similarity = _mean_pairwise_cosine(sim_vecs, sl)

    if abs(sim_max - sim_min) <= _EPS:
        z = 1.0
    else:
        z = (similarity - sim_min) / (sim_max - sim_min)
    z = max(0.0, min(1.0, float(z)))
    p = (p_max - p_min) * (z ** gamma) + p_min
    p = max(0.0, min(1.0, float(p)))
    return p, {
        "fedup_similarity": float(similarity),
        "fedup_similarity_z": float(z),
        "fedup_similarity_source": sim_source,
        "fedup_p_min": p_min,
        "fedup_p_max": p_max,
        "fedup_gamma": gamma,
    }


def _choose_prune_ratio(
    context: Any,
    benign_vectors: Sequence[torch.Tensor],
    global_before: torch.Tensor,
) -> Tuple[float, str, Dict[str, Any]]:
    """Choose FedUP pruning ratio."""
    policy = str(_get_arg(context.args, "fedup_prune_policy", "auto")).lower()
    if hasattr(context.args, "fedup_auto_prune_ratio") and not bool(context.args.fedup_auto_prune_ratio):
        policy = "manual"
    if policy in {"manual", "fixed"}:
        p = float(_get_arg(context.args, "fedup_prune_ratio", 0.05))
        return max(0.0, min(1.0, p)), "manual", {}
    p, meta = _paper_auto_prune_ratio(context, benign_vectors, global_before)
    return p, "auto_similarity_paper_formula", meta


def _layerwise_top_p_mask(
    scores: torch.Tensor,
    template_state: Mapping[str, torch.Tensor],
    ratio: float,
) -> Tuple[torch.Tensor, Dict[str, int]]:
    """Generate a layer-wise top-P mask."""
    ratio = float(max(0.0, min(1.0, ratio)))
    clean_scores = torch.nan_to_num(scores.detach().cpu().float(), nan=0.0, posinf=0.0, neginf=0.0)
    mask = torch.zeros_like(clean_scores, dtype=torch.bool)
    layer_counts: Dict[str, int] = {}
    if ratio <= 0.0:
        return mask, layer_counts

    for name, sl, _shape in _prunable_weight_slices(template_state):
        layer_scores = clean_scores[sl].abs().flatten()
        if layer_scores.numel() == 0:
            continue
        k = int(math.ceil(ratio * int(layer_scores.numel())))
        k = max(1, min(k, int(layer_scores.numel())))
        top_idx = torch.topk(layer_scores, k=k, largest=True).indices
        local_mask = torch.zeros_like(layer_scores, dtype=torch.bool)
        local_mask[top_idx] = True
        mask[sl] = local_mask
        layer_counts[name] = int(k)
    return mask, layer_counts


def _make_optimizer(params: Iterable[torch.nn.Parameter], optimizer_name: str, lr: float, weight_decay: float):
    opt = optimizer_name.lower()
    if opt == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if opt == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.0, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer for FedUP recovery: {optimizer_name}")


def _train_one_client_clean(model: nn.Module, loader: Any, context: Any, *, epochs: int, lr: float, optimizer_name: str) -> None:
    """Clean local training used during FedUP recovery rounds."""
    model.train()
    opt = _make_optimizer(model.parameters(), optimizer_name, lr=lr, weight_decay=float(context.args.weight_decay))
    ce = nn.CrossEntropyLoss()
    for _ in range(int(epochs)):
        for x, y in loader:
            x = x.to(context.device)
            y = y.to(context.device)
            opt.zero_grad(set_to_none=True)
            loss = ce(model(x), y)
            loss.backward()
            opt.step()


def _infer_recovery_rounds(context: Any, prune_ratio: float) -> Tuple[int, int, str]:
    """Return recovery rounds and bound metadata."""
    r_star = int(_get_arg(context.args, "fedup_retrain_rounds", _get_arg(context.args, "global_rounds", 0)))
    bound = max(1, int(math.ceil(r_star * float(prune_ratio)))) if prune_ratio > 0 else 0
    if bool(_get_arg(context.args, "fedup_auto_recovery_rounds", False)):
        return bound, bound, "paper_bound_ceil_Rstar_times_P"
    return int(max(0, _get_arg(context.args, "repair_epochs", bound))), bound, "manual_repair_epochs"


def _fedup_recovery_rounds(model: nn.Module, context: Any, *, rounds: int, recovery_clients: Sequence[int]) -> int:
    """Run FedUP recovery rounds."""
    rounds = int(max(0, rounds))
    clients = [int(cid) for cid in recovery_clients]
    if rounds == 0 or not clients:
        return 0

    current_vec = model_to_vector(model).detach().cpu().float()
    optimizer_name = str(_get_arg(context.args, "fedup_recovery_optimizer", "adam"))
    lr = float(_get_arg(context.args, "fedup_recovery_lr", 1e-3))
    local_epochs = int(_get_arg(context.args, "fedup_recovery_local_epochs", context.args.local_epochs))
    if local_epochs < 0:
        local_epochs = int(context.args.local_epochs)

    for _ in range(rounds):
        local_vectors: List[torch.Tensor] = []
        weights: List[float] = []
        for cid in clients:
            local_model = context.model_fn().to(context.device)
            load_vector_into_model(local_model, current_vec, context.template_state)
            loader = make_loader(context.client_datasets[cid], batch_size=context.args.batch_size, shuffle=True)
            _train_one_client_clean(
                local_model,
                loader,
                context,
                epochs=local_epochs,
                lr=lr,
                optimizer_name=optimizer_name,
            )
            local_vectors.append(model_to_vector(local_model).detach().cpu().float())
            weights.append(_record_client_weights(context, {}, cid))

        new_global = robust_aggregate(
            local_vectors,
            weights,
            rule=context.args.aggregation,
            trim_ratio=context.args.trim_ratio,
        )
        current_vec = new_global.detach().cpu().float()

    load_vector_into_model(model, current_vec, context.template_state)
    return rounds


def _rate_limit_allows_unlearning(context: Any, detected_round: int) -> Tuple[bool, Dict[str, int]]:
    """FedUP DoS-resilience rate-limit bookkeeping."""
    threshold = int(_get_arg(context.args, "fedup_rate_limit_rounds", 10))
    last_unlearning_round = int(_get_arg(context.args, "fedup_last_unlearning_round", -10**9))
    force = bool(_get_arg(context.args, "fedup_force_unlearning", False))
    allowed = force or (detected_round - last_unlearning_round >= threshold)
    return allowed, {
        "fedup_rate_limit_rounds": threshold,
        "fedup_last_unlearning_round": last_unlearning_round,
        "fedup_detection_round": int(detected_round),
        "fedup_rate_limit_allowed": int(bool(allowed)),
    }


def _storage_bytes_for_last_round(local_models: Mapping[int, torch.Tensor], global_before: torch.Tensor) -> int:
    if local_models:
        first = next(iter(local_models.values()))
        elem = int(first.element_size())
        return int((len(local_models) + 1) * first.numel() * elem)
    return int(global_before.numel() * global_before.element_size())


@register
class FedUP(RecoveryMethod):
    name = "fedup"

    def recover(self, context: Any):
        if not getattr(context, "history", None):
            model = context.new_model_from_vector(context.poisoned_vector)
            return RecoveryResult(self.name, model, {
                "method_note": "fedup_no_history_available",
                "pruned_params": 0,
            })

        last_rec = context.history[-1]
        unlearn_clients, unlearn_source = _detected_unlearn_clients(context, last_rec)
        remaining_clients, remaining_source = _remaining_clients(context, last_rec, unlearn_clients)

        if not unlearn_clients:
            model = context.new_model_from_vector(context.poisoned_vector)
            return RecoveryResult(self.name, model, {
                "method_note": "fedup_no_detected_clients_to_unlearn",
                "fedup_detection_source": unlearn_source,
                "pruned_params": 0,
            })

        rec, local_models, selected, ref_meta = _find_last_usable_record(context, unlearn_clients, remaining_clients)
        if rec is None:
            model = context.new_model_from_vector(context.poisoned_vector)
            return RecoveryResult(self.name, model, {
                "method_note": "fedup_no_last_round_with_both_unlearn_and_remaining_models",
                "fedup_detection_source": unlearn_source,
                "fedup_remaining_source": remaining_source,
                "fedup_detected_unlearn_clients": len(unlearn_clients),
                "fedup_remaining_clients": len(remaining_clients),
                "pruned_params": 0,
            })

        detected_round = _record_round(rec, len(context.history) - 1)
        allowed, rate_meta = _rate_limit_allows_unlearning(context, detected_round)
        if not allowed:
            model = context.new_model_from_vector(context.poisoned_vector)
            return RecoveryResult(self.name, model, {
                **rate_meta,
                **ref_meta,
                "method_note": "fedup_rate_limit_skipped_unlearning_pending_clients_accumulated",
                "fedup_pending_unlearn_clients": len(unlearn_clients),
                "pruned_params": 0,
            })

        usable_unlearn_ids = sorted((unlearn_clients & selected) & set(local_models.keys()))
        usable_remaining_ids = sorted((remaining_clients & selected) & set(local_models.keys()))

        if not usable_unlearn_ids or not usable_remaining_ids:
            model = context.new_model_from_vector(context.poisoned_vector)
            return RecoveryResult(self.name, model, {
                **rate_meta,
                **ref_meta,
                "method_note": "fedup_last_round_missing_unlearn_or_remaining_reference_group",
                "pruned_params": 0,
                "fedup_unlearn_last_round": len(usable_unlearn_ids),
                "fedup_remaining_last_round": len(usable_remaining_ids),
            })

        unlearn_vecs = [local_models[cid].detach().cpu().float() for cid in usable_unlearn_ids]
        remaining_vecs = [local_models[cid].detach().cpu().float() for cid in usable_remaining_ids]
        unlearn_weights = [_record_client_weights(context, rec, cid) for cid in usable_unlearn_ids]
        remaining_weights = [_record_client_weights(context, rec, cid) for cid in usable_remaining_ids]

        avg_unlearn_model = _weighted_average(unlearn_vecs, unlearn_weights)
        avg_remaining_model = _weighted_average(remaining_vecs, remaining_weights)

        global_before, global_source = _global_before_vector(context, rec)
        difference = (avg_unlearn_model - avg_remaining_model).pow(2)
        scores = difference * global_before.abs()

        prune_ratio, prune_policy, prune_meta = _choose_prune_ratio(context, remaining_vecs, global_before)
        mask, layer_counts = _layerwise_top_p_mask(scores, context.template_state, prune_ratio)

        pruned_vec = avg_remaining_model.detach().cpu().float().clone()
        pruned_vec[mask] = 0.0
        pruned_params = int(mask.sum().item())

        model = context.new_model_from_vector(pruned_vec)

        recovery_rounds, recovery_bound, recovery_policy = _infer_recovery_rounds(context, prune_ratio)
        actual_recovery_rounds = _fedup_recovery_rounds(
            model,
            context,
            rounds=recovery_rounds,
            recovery_clients=sorted(remaining_clients),
        )
        resolved_local_epochs = int(_get_arg(context.args, "fedup_recovery_local_epochs", context.args.local_epochs))
        if resolved_local_epochs < 0:
            resolved_local_epochs = int(context.args.local_epochs)

        prunable_params = sum(
            int(context.template_state[name].numel()) for name, _, _ in _prunable_weight_slices(context.template_state)
        )
        total_params = int(context.poisoned_vector.numel())
        all_clients = _all_clients(context)
        benign_majority_ok = int(len(unlearn_clients) <= max(0, (len(all_clients) - 1) // 2)) if all_clients else -1

        storage_bytes = _storage_bytes_for_last_round(local_models, global_before)

        return RecoveryResult(self.name, model, {
            **rate_meta,
            **ref_meta,
            **prune_meta,
            "fedup_detection_source": unlearn_source,
            "fedup_remaining_source": remaining_source,
            "fedup_global_before_source": global_source,
            "fedup_detected_unlearn_clients": len(unlearn_clients),
            "fedup_remaining_clients": len(remaining_clients),
            "fedup_unlearn_last_round": len(usable_unlearn_ids),
            "fedup_remaining_last_round": len(usable_remaining_ids),
            "fedup_benign_majority_assumption_ok": benign_majority_ok,
            "pruned_params": pruned_params,
            "prunable_params": int(prunable_params),
            "total_params": int(total_params),
            "prune_ratio": float(prune_ratio),
            "fedup_prune_policy": prune_policy,
            "fedup_layer_pruned": layer_counts,
            "fedup_recovery_rounds": int(actual_recovery_rounds),
            "fedup_recovery_round_bound": int(recovery_bound),
            "fedup_recovery_round_policy": recovery_policy,
            "fedup_recovery_optimizer": str(_get_arg(context.args, "fedup_recovery_optimizer", "adam")),
            "fedup_recovery_lr": float(_get_arg(context.args, "fedup_recovery_lr", 1e-3)),
            "fedup_recovery_local_epochs": int(resolved_local_epochs),
            "fedup_last_round_storage_bytes": int(storage_bytes),
            "fedup_last_round_storage_mb": float(storage_bytes) / (1024.0 ** 2),
            "method_note": "fedup_algorithm1_algorithm2_detected_clients_autoP_rate_limit_recovery",
        })
