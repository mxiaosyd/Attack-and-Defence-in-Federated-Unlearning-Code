from __future__ import annotations

from collections import Counter, defaultdict, deque
from typing import Any, Deque, Dict, List, Optional, Sequence, Set, Tuple

import torch

from common.recovery_base import RecoveryMethod, RecoveryResult, register
from common.training import train_one_client
from common.utils import load_vector_into_model, model_to_vector, robust_aggregate
from common.data import make_loader


_EPS = 1e-12


def _arg(args: Any, name: str, default: Any) -> Any:
    return getattr(args, name, default)


def _as_client_set(value: Any) -> Set[int]:
    if value is None:
        return set()
    return {int(x) for x in value}


def _context_client_count(context: Any) -> Optional[int]:
    if hasattr(context, "client_datasets"):
        try:
            return len(context.client_datasets)
        except TypeError:
            return None
    return None


def _remaining_client_set(context: Any, rec: Optional[Dict[str, Any]] = None) -> Set[int]:
    """Clients remaining after detection."""
    for name in (
        "remaining_clients",
        "recovery_clients",
        "surviving_clients",
        "detected_benign_clients",
        "clients_after_detection",
    ):
        if hasattr(context, name):
            clients = _as_client_set(getattr(context, name))
            if clients:
                return clients

    n = _context_client_count(context)
    if hasattr(context, "detected_malicious_clients") and n is not None:
        return set(range(n)) - _as_client_set(getattr(context, "detected_malicious_clients"))

    if hasattr(context, "benign_clients"):
        clients = _as_client_set(getattr(context, "benign_clients"))
        if clients:
            return clients

    if rec is not None:
        selected = rec.get("selected_clients") or list(rec.get("client_updates", {}).keys())
        return {int(cid) for cid in selected}
    return set()


def _known_malicious_clients(context: Any) -> Set[int]:
    """Ground-truth malicious clients, used only to replay missed attacks."""
    for name in ("malicious_clients", "true_malicious_clients", "poisoned_clients"):
        if hasattr(context, name):
            clients = _as_client_set(getattr(context, name))
            if clients:
                return clients
    return set()


def _round_clients(context: Any, rec: Dict[str, Any]) -> List[int]:
    """Post-detection participants that also have historical information."""
    selected = rec.get("selected_clients") or list(rec.get("client_updates", {}).keys())
    remaining = _remaining_client_set(context, rec)
    updates = rec.get("client_updates", {})
    out: List[int] = []
    for cid in selected:
        cid_int = int(cid)
        if cid_int in remaining and cid_int in updates:
            out.append(cid_int)
    return out


def _infer_update_convention(context: Any) -> str:
    """Return 'delta' or 'gradient'."""
    args = context.args
    convention = str(
        _arg(args, "fedrecover_update_convention", _arg(args, "update_convention", "delta"))
    ).lower()
    if convention in {"grad", "gradient", "gradients"}:
        return "gradient"
    if convention in {"delta", "model_delta", "model-delta", "update_delta"}:
        return "delta"
    raise ValueError(
        "fedrecover_update_convention must be either 'delta' or 'gradient', "
        f"got {convention!r}."
    )


def _apply_server_update(context: Any, vec: torch.Tensor, agg: torch.Tensor) -> torch.Tensor:
    server_lr = float(_arg(context.args, "server_lr", _arg(context.args, "lr", 1.0)))
    if _infer_update_convention(context) == "gradient":
        return vec - server_lr * agg.detach().cpu().float()
    return vec + server_lr * agg.detach().cpu().float()


def _make_recovery_loader(context: Any, cid: int, round_idx: int):
    """Create a deterministic recovery loader."""
    args = context.args
    batch_size = int(_arg(args, "batch_size", 32))
    shuffle = bool(_arg(args, "fedrecover_shuffle_exact_loader", True))

    seed = int(_arg(args, "seed", 0))
    seed = seed + 1_000_003 * int(round_idx) + 9_176 * int(cid)
    gen = torch.Generator()
    gen.manual_seed(seed)

    try:
        return make_loader(context.client_datasets[cid], batch_size=batch_size, shuffle=shuffle, generator=gen)
    except TypeError:
        return make_loader(context.client_datasets[cid], batch_size=batch_size, shuffle=shuffle)


def _call_optional_context_update(
    context: Any,
    *,
    cid: int,
    base_vec: torch.Tensor,
    round_idx: int,
    malicious: bool,
) -> Optional[torch.Tensor]:
    """Use project-specific update hooks when available."""
    hook = getattr(context, "compute_client_update", None)
    if hook is not None:
        for kwargs in (
            dict(cid=cid, base_vec=base_vec, round_idx=round_idx, malicious=malicious, recovery=True),
            dict(client_id=cid, base_vec=base_vec, round_idx=round_idx, malicious=malicious, recovery=True),
            dict(cid=cid, model_vector=base_vec, round_idx=round_idx),
        ):
            try:
                out = hook(**kwargs)
                return out.detach().cpu().float().flatten()
            except TypeError:
                continue

    malicious_hook = getattr(context, "malicious_update_fn", None)
    if malicious and malicious_hook is not None:
        for kwargs in (
            dict(cid=cid, base_vec=base_vec, round_idx=round_idx, recovery=True),
            dict(client_id=cid, model_vector=base_vec, round_idx=round_idx),
        ):
            try:
                out = malicious_hook(**kwargs)
                return out.detach().cpu().float().flatten()
            except TypeError:
                continue
    return None


def _exact_client_update(context: Any, base_vec: torch.Tensor, cid: int, round_idx: int) -> torch.Tensor:
    """Compute an exact update for one recovery client."""
    malicious = cid in _known_malicious_clients(context)
    replay_missed = bool(_arg(context.args, "fedrecover_replay_missed_malicious_attacks", False))
    hooked = _call_optional_context_update(
        context,
        cid=cid,
        base_vec=base_vec.detach().cpu().float(),
        round_idx=round_idx,
        malicious=malicious and replay_missed,
    )
    if hooked is not None:
        return hooked

    model = context.model_fn().to(context.device)
    load_vector_into_model(model, base_vec.detach().cpu(), context.template_state)
    loader = _make_recovery_loader(context, cid, round_idx)

    attack = "none"
    if malicious and replay_missed:
        attack = str(_arg(context.args, "fedrecover_missed_malicious_attack", _arg(context.args, "attack", "none")))

    train_one_client(
        model,
        loader,
        context.device,
        epochs=int(_arg(context.args, "local_epochs", 1)),
        lr=float(_arg(context.args, "lr", 1e-3)),
        weight_decay=float(_arg(context.args, "weight_decay", 0.0)),
        optimizer_name=str(_arg(context.args, "optimizer", "sgd")),
        attack=attack,
        target_label=int(_arg(context.args, "target_label", 0)),
        poison_fraction=float(0.0 if attack == "none" else _arg(context.args, "poison_fraction", 1.0)),
        trigger_size=int(_arg(context.args, "trigger_size", 4)),
        trigger_pattern=str(_arg(context.args, "trigger_pattern", "square")),
        trigger_position=str(_arg(context.args, "trigger_position", "bottom_right")),
        num_classes=int(context.num_classes),
    )

    after = model_to_vector(model).detach().cpu().float().flatten()
    before = base_vec.detach().cpu().float().flatten()

    if _infer_update_convention(context) == "gradient":
        lr = float(_arg(context.args, "lr", 1e-3))
        upd = (before - after) / max(lr, _EPS)
    else:
        upd = after - before

    if malicious and replay_missed and attack != "none":
        scale = float(
            _arg(
                context.args,
                "fedrecover_malicious_scale",
                _arg(context.args, "scale_factor", _arg(context.args, "attack_scale", 1.0)),
            )
        )
        upd = upd * scale
    return upd.float()


def _estimate_tau(context: Any, alpha: float, explicit_tau: Optional[float] = None) -> float:
    """Estimate abnormality threshold tau."""
    if explicit_tau is not None and explicit_tau >= 0:
        return float(explicit_tau)
    alpha = float(alpha)
    if alpha <= 0:
        return float("inf")

    q = max(0.0, min(1.0, 1.0 - alpha))
    tau = 0.0
    for rec in context.history:
        vals: List[torch.Tensor] = []
        for cid in _round_clients(context, rec):
            upd = rec.get("client_updates", {}).get(cid)
            if upd is not None:
                vals.append(upd.detach().abs().flatten().float().cpu())
        if not vals:
            continue
        flat = torch.cat(vals)
        if flat.numel() == 0:
            continue
        try:
            round_tau = float(torch.quantile(flat, q).item())
        except Exception:
            k = min(flat.numel() - 1, max(0, int(q * (flat.numel() - 1))))
            round_tau = float(torch.sort(flat).values[k].item())
        tau = max(tau, round_tau)
    return tau if tau > 0.0 else float("inf")


def _lbfgs_hvp(delta_w: Sequence[torch.Tensor], delta_g: Sequence[torch.Tensor], v: torch.Tensor) -> torch.Tensor:
    """Compact L-BFGS Hessian-vector product."""
    if not delta_w or not delta_g or len(delta_w) != len(delta_g):
        return torch.zeros_like(v.detach().cpu().float().flatten())

    v64 = v.detach().cpu().double().flatten()
    W = torch.stack([x.detach().cpu().double().flatten() for x in delta_w], dim=1)
    G = torch.stack([x.detach().cpu().double().flatten() for x in delta_g], dim=1)

    if W.numel() == 0 or torch.linalg.vector_norm(W) < _EPS or torch.linalg.vector_norm(G) < _EPS:
        return torch.zeros_like(v64).float()

    A = W.t().matmul(G)
    D = torch.diag(torch.diag(A))
    L = torch.tril(A, diagonal=-1)

    last_w = W[:, -1]
    last_g = G[:, -1]
    denom = torch.dot(last_w, last_w).clamp_min(_EPS)
    sigma = torch.dot(last_g, last_w) / denom
    if not torch.isfinite(sigma) or abs(float(sigma.item())) < _EPS:
        sigma = torch.tensor(1.0, dtype=W.dtype)

    WT_W = W.t().matmul(W)
    mat = torch.cat(
        [
            torch.cat([-D, L.t()], dim=1),
            torch.cat([L, sigma * WT_W], dim=1),
        ],
        dim=0,
    )
    rhs = torch.cat([G.t().matmul(v64), sigma * W.t().matmul(v64)], dim=0)

    eye = torch.eye(mat.shape[0], dtype=mat.dtype)
    try:
        p = torch.linalg.solve(mat + 1e-10 * eye, rhs)
    except RuntimeError:
        p = torch.linalg.pinv(mat, rcond=1e-10).matmul(rhs)

    hv = sigma * v64 - torch.cat([G, sigma * W], dim=1).matmul(p)
    if not torch.all(torch.isfinite(hv)):
        return torch.zeros_like(v64).float()
    return hv.float()


@register
class FedRecover(RecoveryMethod):
    name = "fedrecover"

    def recover(self, context: Any):
        args = context.args
        T = len(context.history)
        if T == 0:
            return RecoveryResult(
                self.name,
                context.new_model_from_vector(context.init_vector),
                {"method_note": "fedrecover_no_history_return_initial_model"},
            )

        s = max(1, int(_arg(args, "fedrecover_lbfgs_buffer_size", 2)))
        Tw_req = max(0, int(_arg(args, "fedrecover_warmup_rounds", 20)))
        Tc = max(1, int(_arg(args, "fedrecover_correction_period", 10)))
        Tf_req = max(0, int(_arg(args, "fedrecover_final_tuning_rounds", 5)))
        alpha = float(_arg(args, "fedrecover_tolerance_rate", 1e-6))
        explicit_tau = float(_arg(args, "fedrecover_tau", -1.0))
        disable_abnormality = bool(_arg(args, "fedrecover_disable_abnormality_fixing", False))

        Tf = min(T, Tf_req)
        Tw = min(T - Tf, Tw_req)
        warmup_adjusted = False
        if T - Tf > s and Tw <= s:
            Tw = min(T - Tf, s + 1)
            warmup_adjusted = True

        tau = float("inf") if disable_abnormality else _estimate_tau(context, alpha, explicit_tau)
        update_convention = _infer_update_convention(context)

        vec = context.init_vector.detach().cpu().float().clone().flatten()

        buffers: Dict[int, Deque[Tuple[torch.Tensor, torch.Tensor]]] = defaultdict(lambda: deque(maxlen=s))

        baseline_counts: Counter[int] = Counter()
        for rec in context.history:
            baseline_counts.update(_round_clients(context, rec))

        exact_counts: Counter[int] = Counter()
        exact_rounds = 0
        estimated_rounds = 0
        periodic_rounds = 0
        final_tuning_rounds = 0
        abnormality_fixes = 0
        fallback_exact_updates = 0
        estimated_client_updates = 0

        def _record_exact_pair(cid: int, wdiff: torch.Tensor, upd: torch.Tensor, orig: torch.Tensor) -> None:
            buffers[cid].append((wdiff.detach().cpu().float().clone(), (upd - orig).detach().cpu().float().clone()))
            exact_counts[cid] += 1

        def exact_training_round(t: int, rec: Dict[str, Any], reason: str) -> torch.Tensor:
            nonlocal exact_rounds, periodic_rounds, final_tuning_rounds
            clients = _round_clients(context, rec)
            if not clients:
                return vec

            updates: List[torch.Tensor] = []
            weights: List[float] = []
            global_before = rec["global_before"].detach().cpu().float().flatten()
            wdiff = vec - global_before
            for cid in clients:
                upd = _exact_client_update(context, vec, cid, t)
                orig = rec["client_updates"][cid].detach().cpu().float().flatten()
                _record_exact_pair(cid, wdiff, upd, orig)
                updates.append(upd)
                weights.append(float(rec.get("client_weights", {}).get(cid, len(context.client_datasets[cid]))))

            agg = robust_aggregate(updates, weights, rule=args.aggregation, trim_ratio=args.trim_ratio)
            if reason == "periodic":
                periodic_rounds += 1
            elif reason == "final":
                final_tuning_rounds += 1
            exact_rounds += 1
            return _apply_server_update(context, vec, agg)

        for t, rec in enumerate(context.history):
            if t < Tw:
                vec = exact_training_round(t, rec, reason="warmup")
                continue
            if t >= T - Tf:
                vec = exact_training_round(t, rec, reason="final")
                continue
            if ((t - Tw + 1) % Tc) == 0:
                vec = exact_training_round(t, rec, reason="periodic")
                continue

            clients = _round_clients(context, rec)
            if not clients:
                continue

            global_before = rec["global_before"].detach().cpu().float().flatten()
            wdiff = vec - global_before
            updates: List[torch.Tensor] = []
            weights: List[float] = []

            for cid in clients:
                orig = rec["client_updates"][cid].detach().cpu().float().flatten()
                buf = list(buffers[cid])
                if len(buf) < s:
                    upd = _exact_client_update(context, vec, cid, t)
                    _record_exact_pair(cid, wdiff, upd, orig)
                    fallback_exact_updates += 1
                else:
                    dw_buf = [pair[0] for pair in buf]
                    dg_buf = [pair[1] for pair in buf]
                    hvp = _lbfgs_hvp(dw_buf, dg_buf, wdiff)
                    upd = (orig + hvp).float()
                    estimated_client_updates += 1

                    if (not disable_abnormality) and torch.max(torch.abs(upd)).item() > tau:
                        upd = _exact_client_update(context, vec, cid, t)
                        _record_exact_pair(cid, wdiff, upd, orig)
                        abnormality_fixes += 1

                updates.append(upd.detach().cpu().float())
                weights.append(float(rec.get("client_weights", {}).get(cid, len(context.client_datasets[cid]))))

            agg = robust_aggregate(updates, weights, rule=args.aggregation, trim_ratio=args.trim_ratio)
            vec = _apply_server_update(context, vec, agg)
            estimated_rounds += 1

        model = context.new_model_from_vector(vec)

        cp_values: List[float] = []
        for cid, baseline in baseline_counts.items():
            if baseline <= 0:
                continue
            cp_values.append(100.0 * (1.0 - float(exact_counts[cid]) / float(baseline)))
        acp = float(sum(cp_values) / len(cp_values)) if cp_values else 0.0

        exact_client_updates = int(sum(exact_counts.values()))
        train_from_scratch_client_updates = int(sum(baseline_counts.values()))

        return RecoveryResult(
            self.name,
            model,
            {
                "fedrecover_warmup_rounds": Tw,
                "fedrecover_requested_warmup_rounds": Tw_req,
                "fedrecover_warmup_adjusted_to_satisfy_Tw_gt_s": warmup_adjusted,
                "fedrecover_correction_period": Tc,
                "fedrecover_final_tuning_rounds": Tf,
                "fedrecover_requested_final_tuning_rounds": Tf_req,
                "fedrecover_lbfgs_buffer_size": s,
                "fedrecover_tau": tau,
                "fedrecover_tolerance_rate": alpha,
                "fedrecover_update_convention": update_convention,
                "exact_rounds": exact_rounds,
                "estimated_rounds": estimated_rounds,
                "periodic_rounds": periodic_rounds,
                "final_tuning_rounds": final_tuning_rounds,
                "exact_client_updates": exact_client_updates,
                "estimated_client_updates": estimated_client_updates,
                "abnormality_fixes": abnormality_fixes,
                "fallback_exact_updates": fallback_exact_updates,
                "train_from_scratch_client_updates": train_from_scratch_client_updates,
                "average_cost_saving_percentage": acp,
                "method_note": "fedrecover_paper_faithful_lbfgs_warmup_periodic_abnormality_final_tuning",
            },
        )
