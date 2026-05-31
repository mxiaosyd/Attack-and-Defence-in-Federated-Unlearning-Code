"""MCC-Fed reproduction with distance detection and K-regularized unlearning."""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from common.data import make_loader
from common.recovery_base import RecoveryMethod, RecoveryResult, register
from common.utils import load_vector_into_model, model_to_vector, robust_aggregate


_EPS = 1e-12


def _weighted_average_model_vectors(
    vectors: Sequence[torch.Tensor],
    weights: Sequence[float] | None = None,
) -> torch.Tensor:
    if not vectors:
        raise ValueError("Cannot average an empty model-vector list.")
    if weights is None:
        weights = [1.0] * len(vectors)
    return robust_aggregate(
        [v.detach().cpu() for v in vectors],
        [float(w) for w in weights],
        rule="fedavg",
    ).detach().cpu()


def _model_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.norm(a.detach().float().cpu() - b.detach().float().cpu()).item())


def _parameter_count(vec: torch.Tensor) -> int:
    return int(vec.numel())


def _safe_float(x, default: float) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _safe_int(x, default: int) -> int:
    try:
        return int(x)
    except Exception:
        return int(default)


def _flat_parameters(model: torch.nn.Module) -> torch.Tensor:
    return torch.cat([p.reshape(-1) for p in model.parameters()])


def _high_score_kmeans(scores: Mapping[int, float]) -> Tuple[List[int], str]:
    """Split one-dimensional anomaly scores without using malicious count."""
    if len(scores) < 2:
        return [], "kmeans_insufficient_clients"

    client_ids = sorted(int(cid) for cid in scores.keys())
    values = np.array([float(scores[cid]) for cid in client_ids], dtype=np.float64)
    if float(values.max() - values.min()) <= _EPS:
        return [], "kmeans_degenerate_equal_scores"

    centers = np.array([values.min(), values.max()], dtype=np.float64)
    labels = np.zeros(values.shape[0], dtype=np.int64)
    for _ in range(100):
        old = labels.copy()
        d0 = np.abs(values - centers[0])
        d1 = np.abs(values - centers[1])
        labels = (d1 < d0).astype(np.int64)
        for k in (0, 1):
            if np.any(labels == k):
                centers[k] = float(values[labels == k].mean())
        if np.array_equal(labels, old):
            break

    high_label = int(np.argmax(centers))
    detected = [cid for cid, label in zip(client_ids, labels.tolist()) if int(label) == high_label]
    if len(detected) == len(client_ids):
        return [], "kmeans_degenerate_all_high_score"
    return detected, "kmeans_high_distance_ratio_cluster"


def _minority_distance_kmeans(scores: Mapping[int, float]) -> Tuple[List[int], str]:
    """MCC-Fed distance clustering under the malicious-minority assumption."""
    if len(scores) < 2:
        return [], "kmeans_insufficient_clients"

    client_ids = sorted(int(cid) for cid in scores.keys())
    values = np.array([float(scores[cid]) for cid in client_ids], dtype=np.float64)
    if float(values.max() - values.min()) <= _EPS:
        return [], "kmeans_degenerate_equal_scores"

    centers = np.array([values.min(), values.max()], dtype=np.float64)
    labels = np.zeros(values.shape[0], dtype=np.int64)
    for _ in range(100):
        old = labels.copy()
        dist = np.stack([np.abs(values - centers[0]), np.abs(values - centers[1])], axis=1)
        labels = np.argmin(dist, axis=1).astype(np.int64)
        for k in (0, 1):
            if np.any(labels == k):
                centers[k] = float(values[labels == k].mean())
        if np.array_equal(labels, old):
            break

    counts = {k: int(np.sum(labels == k)) for k in (0, 1)}
    if counts[0] == 0 or counts[1] == 0:
        return [], "kmeans_degenerate_single_cluster"
    if counts[0] == counts[1]:
        global_mean = float(values.mean())
        chosen = int(np.argmax(np.abs(centers - global_mean)))
        reason = "kmeans_equal_size_farthest_distance_cluster"
    else:
        chosen = 0 if counts[0] < counts[1] else 1
        reason = "kmeans_malicious_minority_distance_cluster"
    detected = [cid for cid, label in zip(client_ids, labels.tolist()) if int(label) == chosen]
    return detected, reason


def _train_client_with_contribution_regularization(
    context,
    start_vec: torch.Tensor,
    client_id: int,
    k_value: float,
    reference_vec: torch.Tensor,
) -> torch.Tensor:
    """Train a benign client with K regularization."""
    model = context.model_fn().to(context.device)
    load_vector_into_model(model, start_vec.detach().cpu(), context.template_state)
    model.train()

    loader = make_loader(
        context.client_datasets[client_id],
        batch_size=context.args.batch_size,
        shuffle=True,
    )

    lr = float(context.args.lr)
    weight_decay = float(context.args.weight_decay)

    optimizer_name = str(getattr(context.args, "mcc_optimizer", "sgd")).lower()
    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)

    alpha = float(getattr(context.args, "mcc_alpha", 1.0))
    reg_scale = float(getattr(context.args, "mcc_regularization_scale", 1e-4))
    reg_mode = str(getattr(context.args, "mcc_regularization_mode", "repel_malicious")).lower()

    k_strength = alpha * reg_scale * float(k_value)

    ref = reference_vec.detach().float().to(context.device)
    ref_norm = max(float(torch.norm(ref).item()), _EPS)
    param_scale = max(float(ref.numel()) ** 0.5, 1.0)

    for _ in range(int(context.args.local_epochs)):
        for x, y in loader:
            x = x.to(context.device)
            y = y.to(context.device)

            optimizer.zero_grad(set_to_none=True)

            logits = model(x)
            loss = F.cross_entropy(logits, y)

            current = _flat_parameters(model).float()
            if reg_mode == "proximal_current":
                reg = 0.5 * torch.mean((current - ref) ** 2) / ref_norm
            else:
                reg = -torch.norm(current - ref, p=2) / param_scale
            total_loss = loss + k_strength * reg

            total_loss.backward()
            optimizer.step()

    return model_to_vector(model).detach().cpu()


@register
class MCCFed(RecoveryMethod):
    """MCC-Fed recovery method."""

    name = "mcc_fed"

    def _detect_malicious_clients(self, context) -> Tuple[List[int], Dict[int, float], str]:
        """Detect malicious clients by distance ratio."""
        distance_ratio_scores: Dict[int, List[float]] = defaultdict(list)

        for rec in context.history:
            local_models: Mapping[int, torch.Tensor] = rec.get("local_model_vectors", {})
            if not local_models:
                continue

            global_vec = rec.get("global_after", rec.get("global_before")).detach().cpu()

            distances: Dict[int, float] = {}
            for cid, local_vec in local_models.items():
                distances[int(cid)] = _model_distance(global_vec, local_vec)

            if not distances:
                continue

            avg_distance = float(np.mean(list(distances.values())))
            for cid, dist in distances.items():
                distance_ratio_scores[int(cid)].append(float(dist) / max(avg_distance, _EPS))

        avg_scores = {
            cid: float(np.mean(vals))
            for cid, vals in distance_ratio_scores.items()
            if vals
        }

        ratio_threshold = float(getattr(context.args, "mcc_detection_ratio_threshold", 1.0))
        detection_policy = str(getattr(context.args, "mcc_detection_policy", "minority_kmeans")).lower()
        if detection_policy == "threshold_high":
            detected = [int(cid) for cid, score in avg_scores.items() if float(score) > ratio_threshold]
            policy = f"distance_ratio_gt_{ratio_threshold:g}"
            if not detected or len(detected) == len(avg_scores):
                detected, policy = _high_score_kmeans(avg_scores)
        else:
            detected, policy = _minority_distance_kmeans(avg_scores)
            if not detected:
                detected, policy = _high_score_kmeans(avg_scores)

        return [int(cid) for cid in sorted(set(detected))], avg_scores, policy

    def _malicious_aggregation_models(
        self,
        context,
        malicious_clients: Sequence[int],
    ) -> Dict[int, torch.Tensor]:
        """Build historical malicious aggregation models."""
        out: Dict[int, torch.Tensor] = {}

        for cid in malicious_clients:
            models: List[torch.Tensor] = []
            for rec in context.history:
                local_models: Mapping[int, torch.Tensor] = rec.get("local_model_vectors", {})
                if cid in local_models:
                    models.append(local_models[cid].detach().cpu())

            if models:
                out[int(cid)] = torch.stack([m.float() for m in models], dim=0).mean(dim=0).detach().cpu()
            else:
                out[int(cid)] = context.poisoned_vector.detach().cpu().clone()

        return out

    def _perturb_malicious_models(
        self,
        context,
        malicious_agg: Mapping[int, torch.Tensor],
    ) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
        """Add Gaussian perturbation to malicious models."""
        sigma = float(getattr(context.args, "mcc_noise_std", 0.01))
        seed = int(getattr(context.args, "seed", 42)) + 2025

        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)

        perturbed: Dict[int, torch.Tensor] = {}
        noises: Dict[int, torch.Tensor] = {}

        for cid, vec in malicious_agg.items():
            noise = torch.randn(
                vec.shape,
                generator=generator,
                dtype=vec.detach().cpu().dtype,
                device="cpu",
            ) * sigma
            noises[int(cid)] = noise.detach().cpu()
            perturbed[int(cid)] = (vec.detach().cpu() + noise).detach().cpu()

        return perturbed, noises

    def _contribution_metric(
        self,
        global_vec: torch.Tensor,
        perturbed_malicious: Mapping[int, torch.Tensor],
        noises: Mapping[int, torch.Tensor],
    ) -> Tuple[float, Dict[int, float]]:
        """Compute contribution metric K."""
        per_client_k: Dict[int, float] = {}

        g = global_vec.detach().float().cpu()
        for cid, wj in perturbed_malicious.items():
            noise = noises[int(cid)].detach().float().cpu()
            numerator = torch.norm(g - wj.detach().float().cpu())
            denominator = torch.norm(noise) + _EPS
            per_client_k[int(cid)] = float((numerator / denominator).item())

        if not per_client_k:
            return 0.0, {}

        K = float(np.mean(list(per_client_k.values())))
        return K, per_client_k

    def _aggregate_benign_models(
        self,
        context,
        model_vecs: Sequence[torch.Tensor],
        benign_clients: Sequence[int],
    ) -> torch.Tensor:
        if not model_vecs:
            return context.poisoned_vector.detach().cpu().clone()

        weights = [float(len(context.client_datasets[cid])) for cid in benign_clients]
        return robust_aggregate(
            [v.detach().cpu() for v in model_vecs],
            weights,
            rule=context.args.aggregation,
            trim_ratio=context.args.trim_ratio,
        ).detach().cpu()

    def _remove_detected_historical_contributions(
        self,
        context,
        detected_malicious: Sequence[int],
    ) -> Tuple[torch.Tensor, float, int]:
        """Initialize recovery by removing stored contributions of detected clients."""
        detected = set(int(cid) for cid in detected_malicious)
        current_vec = context.poisoned_vector.detach().cpu().float().clone()
        total_removed = torch.zeros_like(current_vec)
        removed_updates = 0

        for rec in context.history:
            updates: Mapping[int, torch.Tensor] = rec.get("client_updates", {})
            weights: Mapping[int, float] = rec.get("client_weights", {})
            if not updates:
                continue
            total_weight = sum(float(w) for w in weights.values())
            if total_weight <= _EPS:
                total_weight = float(len(updates))
            contribution = torch.zeros_like(current_vec)
            for cid in detected:
                if cid not in updates:
                    continue
                w = float(weights.get(cid, 1.0)) / max(total_weight, _EPS)
                contribution = contribution + float(context.args.server_lr) * w * updates[cid].detach().cpu().float()
                removed_updates += 1
            total_removed = total_removed + contribution

        current_vec = current_vec - total_removed
        return current_vec.detach().cpu(), float(torch.norm(total_removed.float()).item()), int(removed_updates)

    def recover(self, context):
        if bool(getattr(context.args, "mcc_use_oracle_malicious", False)):
            detected_malicious = [int(cid) for cid in context.malicious_clients]
            detection_scores = {int(cid): 0.0 for cid in detected_malicious}
            detection_policy = "explicit_oracle_malicious_clients"
        else:
            detected_malicious, detection_scores, detection_policy = self._detect_malicious_clients(context)

        true_malicious = set(int(cid) for cid in context.malicious_clients)
        detected_set = set(int(cid) for cid in detected_malicious)
        tp = len(true_malicious & detected_set)
        fp = len(detected_set - true_malicious)
        fn = len(true_malicious - detected_set)
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)

        benign_clients = [
            int(cid)
            for cid in range(int(context.args.num_clients))
            if int(cid) not in detected_set
        ]

        if not detected_malicious:
            model = context.new_model_from_vector(context.poisoned_vector)
            return RecoveryResult(self.name, model, {
                "detected_malicious": "",
                "true_malicious": ";".join(map(str, sorted(true_malicious))),
                "mcc_detection_precision": float(precision),
                "mcc_detection_recall": float(recall),
                "mcc_detection_tp": int(tp),
                "mcc_detection_fp": int(fp),
                "mcc_detection_fn": int(fn),
                "mcc_detection_policy": detection_policy,
                "mcc_detection_scores": ";".join(
                    f"{cid}:{score:.6f}" for cid, score in sorted(detection_scores.items())
                ),
                "method_note": "mccfed_no_detected_malicious_clients_no_oracle_fallback",
            })

        if not benign_clients:
            model = context.new_model_from_vector(context.poisoned_vector)
            return RecoveryResult(self.name, model, {
                "detected_malicious": ";".join(map(str, detected_malicious)),
                "true_malicious": ";".join(map(str, sorted(true_malicious))),
                "mcc_detection_precision": float(precision),
                "mcc_detection_recall": float(recall),
                "mcc_detection_tp": int(tp),
                "mcc_detection_fp": int(fp),
                "mcc_detection_fn": int(fn),
                "mcc_detection_policy": detection_policy,
                "mcc_detection_scores": ";".join(
                    f"{cid}:{score:.6f}" for cid, score in sorted(detection_scores.items())
                ),
                "method_note": "mccfed_detection_marked_all_clients_no_benign_recovery_set",
            })


        malicious_agg = self._malicious_aggregation_models(context, detected_malicious)
        perturbed_malicious, noises = self._perturb_malicious_models(context, malicious_agg)
        malicious_reference_vec = _weighted_average_model_vectors(
            list(perturbed_malicious.values()),
            [1.0 for _ in perturbed_malicious],
        )

        if bool(getattr(context.args, "mcc_contribution_init", False)):
            current_vec, contribution_removed_norm, contribution_removed_updates = self._remove_detected_historical_contributions(
                context,
                detected_malicious,
            )
            contribution_init = "detected_historical_contribution_removed"
        else:
            current_vec = context.poisoned_vector.detach().cpu().clone()
            contribution_removed_norm = 0.0
            contribution_removed_updates = 0
            contribution_init = "poisoned_model"
        initial_K, initial_per_client_k = self._contribution_metric(
            current_vec,
            perturbed_malicious,
            noises,
        )

        tau = _safe_float(getattr(context.args, "mcc_tau", -1.0), -1.0)
        if tau <= 0:
            tau = float(initial_K * float(getattr(context.args, "mcc_tau_multiplier", 1.05)))

        max_rounds = _safe_int(
            getattr(context.args, "mcc_unlearn_rounds", getattr(context.args, "repair_epochs", 5)),
            5,
        )
        max_rounds = max(1, max_rounds)

        trace: List[str] = []
        per_round_K: List[float] = [float(initial_K)]

        stopped = False
        stop_round = -1
        last_per_client_k = dict(initial_per_client_k)

        for round_idx in range(max_rounds):
            K, per_client_k = self._contribution_metric(
                current_vec,
                perturbed_malicious,
                noises,
            )

            local_models: List[torch.Tensor] = []
            for cid in benign_clients:
                local_vec = _train_client_with_contribution_regularization(
                    context=context,
                    start_vec=current_vec,
                    client_id=int(cid),
                    k_value=K,
                    reference_vec=malicious_reference_vec,
                )
                local_models.append(local_vec)

            next_vec = self._aggregate_benign_models(
                context,
                local_models,
                benign_clients,
            )

            next_K, next_per_client_k = self._contribution_metric(
                next_vec,
                perturbed_malicious,
                noises,
            )

            trace.append(
                f"round={round_idx},K_before={K:.6f},K_after={next_K:.6f},"
                f"benign_clients={len(benign_clients)}"
            )
            per_round_K.append(float(next_K))

            current_vec = next_vec.detach().cpu().clone()
            last_per_client_k = dict(next_per_client_k)

            if next_K > tau:
                stopped = True
                stop_round = int(round_idx)
                break

        model = context.new_model_from_vector(current_vec)

        return RecoveryResult(self.name, model, {
            "detected_malicious": ";".join(map(str, detected_malicious)),
            "true_malicious": ";".join(map(str, sorted(true_malicious))),
            "mcc_detection_precision": float(precision),
            "mcc_detection_recall": float(recall),
            "mcc_detection_tp": int(tp),
            "mcc_detection_fp": int(fp),
            "mcc_detection_fn": int(fn),
            "mcc_detection_policy": detection_policy,
            "mcc_detection_scores": ";".join(
                f"{cid}:{score:.6f}" for cid, score in sorted(detection_scores.items())
            ),
            "mcc_initial_K": float(initial_K),
            "mcc_final_K": float(per_round_K[-1]),
            "mcc_tau": float(tau),
            "mcc_per_client_K": ";".join(
                f"{cid}:{val:.6f}" for cid, val in sorted(last_per_client_k.items())
            ),
            "mcc_noise_std": float(getattr(context.args, "mcc_noise_std", 0.01)),
            "mcc_alpha": float(getattr(context.args, "mcc_alpha", 1.0)),
            "mcc_regularization_scale": float(getattr(context.args, "mcc_regularization_scale", 1e-4)),
            "mcc_regularization_mode": str(getattr(context.args, "mcc_regularization_mode", "repel_malicious")),
            "mcc_unlearn_rounds_run": int(len(per_round_K) - 1),
            "mcc_unlearn_rounds_max": int(max_rounds),
            "mcc_stopped_by_tau": bool(stopped),
            "mcc_stop_round": int(stop_round),
            "mcc_benign_clients_used": int(len(benign_clients)),
            "mcc_contribution_init": contribution_init,
            "mcc_contribution_removed_norm": float(contribution_removed_norm),
            "mcc_contribution_removed_updates": int(contribution_removed_updates),
            "mcc_K_trace": ";".join(f"{x:.6f}" for x in per_round_K),
            "mcc_round_trace": "|".join(trace),
            "method_note": "mccfed_detection_contribution_metric_K_regularized_federated_unlearning",
        })
