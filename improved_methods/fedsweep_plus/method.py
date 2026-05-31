"""FedSweep+ improvement: robust client scoring and generated-trigger repair."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from common.data import make_loader
from common.recovery_base import RecoveryMethod, RecoveryResult, register
from common.training import train_one_client
from common.utils import load_vector_into_model, model_to_vector, robust_aggregate


_EPS = 1e-12


def _arg(args, name: str, default):
    return getattr(args, name, default)


def _trigger_l1_norm(delta: torch.Tensor) -> torch.Tensor:
    """Per-sample L1 trigger norm used by FedSweep Eq. 2."""
    if delta.ndim < 2:
        return torch.mean(torch.abs(delta))
    return delta.abs().flatten(1).sum(dim=1).mean()


def _total_variation(x: torch.Tensor) -> torch.Tensor:
    """Optional TV penalty. Paper Eq. 2 does not require this; default should be 0."""
    if x.ndim < 4:
        return torch.tensor(0.0, device=x.device)
    tv_h = torch.mean(torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]))
    tv_w = torch.mean(torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]))
    return tv_h + tv_w


class _PatchTriggerGenerator(nn.Module):
    """Constrained local trigger generator for square-pattern attacks."""

    def __init__(
        self,
        channels: int,
        trigger_size: int,
        strength: float = 1.0,
        init_logit: float = -3.0,
        position: str = "bottom_right",
    ):
        super().__init__()
        self.channels = int(channels)
        self.trigger_size = int(trigger_size)
        self.strength = float(strength)
        self.position = str(position)

        self.patch_param = nn.Parameter(
            torch.full(
                (1, self.channels, self.trigger_size, self.trigger_size),
                float(init_logit),
            )
        )
        self.last_active_pixels = self.channels * self.trigger_size * self.trigger_size

    def _place_patch(self, delta: torch.Tensor, patch: torch.Tensor) -> torch.Tensor:
        h, w = delta.shape[-2], delta.shape[-1]
        s_h = min(self.trigger_size, h)
        s_w = min(self.trigger_size, w)
        patch = patch[:, :, :s_h, :s_w]

        if self.position in {"bottom_right", "right_bottom", "br"}:
            delta[:, :, h - s_h:h, w - s_w:w] = patch
        elif self.position in {"top_left", "left_top", "tl"}:
            delta[:, :, :s_h, :s_w] = patch
        elif self.position in {"top_right", "right_top", "tr"}:
            delta[:, :, :s_h, w - s_w:w] = patch
        elif self.position in {"bottom_left", "left_bottom", "bl"}:
            delta[:, :, h - s_h:h, :s_w] = patch
        elif self.position in {"center", "middle"}:
            top = max(0, (h - s_h) // 2)
            left = max(0, (w - s_w) // 2)
            delta[:, :, top:top + s_h, left:left + s_w] = patch
        else:
            raise ValueError(f"Unsupported trigger patch position: {self.position}")

        self.last_active_pixels = int(self.channels * s_h * s_w)
        return delta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patch = torch.sigmoid(self.patch_param).to(x.device) * self.strength
        patch = patch.expand(x.shape[0], -1, -1, -1)
        delta = torch.zeros_like(x)
        return self._place_patch(delta, patch)


class _DoubleConv(nn.Module):
    """U-Net block."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _UNetTriggerGenerator(nn.Module):
    """U-Net trigger generator."""

    def __init__(self, channels: int, base: int = 16, strength: float = 1.0, bias_init: float = -4.0):
        super().__init__()
        self.strength = float(strength)
        self.last_active_pixels = 0

        self.down1 = _DoubleConv(channels, base)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = _DoubleConv(base, base * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.bottleneck = _DoubleConv(base * 2, base * 4)

        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = _DoubleConv(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = _DoubleConv(base * 2, base)
        self.out = nn.Conv2d(base, channels, 1)

        nn.init.zeros_(self.out.weight)
        nn.init.constant_(self.out.bias, float(bias_init))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2], x.shape[-1]

        d1 = self.down1(x)
        d2 = self.down2(self.pool1(d1))
        b = self.bottleneck(self.pool2(d2))

        u2 = self.up2(b)
        if u2.shape[-2:] != d2.shape[-2:]:
            u2 = F.interpolate(u2, size=d2.shape[-2:], mode="bilinear", align_corners=False)
        u2 = torch.cat([u2, d2], dim=1)
        u2 = self.dec2(u2)

        u1 = self.up1(u2)
        if u1.shape[-2:] != d1.shape[-2:]:
            u1 = F.interpolate(u1, size=d1.shape[-2:], mode="bilinear", align_corners=False)
        u1 = torch.cat([u1, d1], dim=1)
        u1 = self.dec1(u1)

        delta = torch.sigmoid(self.out(u1)) * self.strength
        if delta.shape[-2:] != (h, w):
            delta = F.interpolate(delta, size=(h, w), mode="bilinear", align_corners=False)

        self.last_active_pixels = int((delta.detach() > 1e-3).float().flatten(1).sum(dim=1).mean().item())
        return delta


@dataclass
class _TriggerInversionResult:
    generator: nn.Module
    final_loss: float
    ce_loss: float
    reg_loss: float
    l1_norm: float
    tv_loss: float
    delta_mean: float
    delta_max: float
    active_pixels: int
    generator_type: str
    success: bool


def _apply_generated_trigger(x: torch.Tensor, generator: nn.Module) -> torch.Tensor:
    """Apply a generated trigger: D_b <- Poi(D, Delta), Delta = G(x)."""
    delta = generator(x)
    delta = _sparsify_delta(delta, float(getattr(generator, "sparse_ratio", 0.0)))
    return torch.clamp(x + delta, 0.0, 1.0)


def _sparsify_delta(delta: torch.Tensor, sparse_ratio: float, *, straight_through: bool = False) -> torch.Tensor:
    """Keep only the largest trigger pixels to enforce FedSweep's concise trigger."""
    ratio = float(max(0.0, min(1.0, sparse_ratio)))
    if ratio <= 0.0 or ratio >= 1.0:
        return delta
    flat = delta.detach().abs().flatten(1)
    k = max(1, int(round(ratio * flat.shape[1])))
    top_idx = torch.topk(flat, k=k, dim=1, largest=True).indices
    mask = torch.zeros_like(flat, dtype=delta.dtype)
    mask.scatter_(1, top_idx, 1.0)
    sparse = delta * mask.view_as(delta)
    if straight_through:
        return (sparse - delta).detach() + delta
    return sparse


def _make_trigger_generator(
    *,
    generator_type: str,
    channels: int,
    trigger_size: int,
    base_channels: int,
    strength: float,
    bias_init: float,
    patch_init_logit: float,
    patch_position: str,
) -> nn.Module:
    generator_type = str(generator_type).lower()
    if generator_type in {"patch", "square_patch", "local_patch"}:
        return _PatchTriggerGenerator(
            channels=channels,
            trigger_size=trigger_size,
            strength=strength,
            init_logit=patch_init_logit,
            position=patch_position,
        )
    if generator_type in {"unet", "u-net"}:
        return _UNetTriggerGenerator(
            channels=channels,
            base=base_channels,
            strength=strength,
            bias_init=bias_init,
        )
    raise ValueError(f"Unsupported FedSweep trigger generator: {generator_type}")


def _train_trigger_generator(
    model: nn.Module,
    loader,
    device: torch.device,
    target_label: int,
    trigger_size: int,
    steps: int,
    lr: float,
    lambda_l1: float,
    lambda_tv: float,
    base_channels: int,
    strength: float,
    generator_type: str,
    bias_init: float,
    patch_init_logit: float,
    patch_position: str,
    sparse_ratio: float,
) -> _TriggerInversionResult:
    """Train the reverse-trigger generator."""
    if str(generator_type).lower() in {"patch", "square_patch", "local_patch"} and str(patch_position).lower() == "search":
        candidates = ["bottom_right", "top_left", "top_right", "bottom_left", "center"]
        results = [
            _train_trigger_generator(
                model=model,
                loader=loader,
                device=device,
                target_label=target_label,
                trigger_size=trigger_size,
                steps=steps,
                lr=lr,
                lambda_l1=lambda_l1,
                lambda_tv=lambda_tv,
                base_channels=base_channels,
                strength=strength,
                generator_type=generator_type,
                bias_init=bias_init,
                patch_init_logit=patch_init_logit,
                patch_position=pos,
                sparse_ratio=sparse_ratio,
            )
            for pos in candidates
        ]
        return min(results, key=lambda item: (item.ce_loss + item.reg_loss, item.l1_norm))

    model.eval()
    try:
        first_x, _ = next(iter(loader))
    except StopIteration:
        raise ValueError("FedSweep requires a non-empty clean/public loader for trigger inversion")

    channels = int(first_x.shape[1])
    generator = _make_trigger_generator(
        generator_type=generator_type,
        channels=channels,
        trigger_size=trigger_size,
        base_channels=base_channels,
        strength=strength,
        bias_init=bias_init,
        patch_init_logit=patch_init_logit,
        patch_position=patch_position,
    ).to(device)
    setattr(generator, "sparse_ratio", float(sparse_ratio))

    opt = torch.optim.Adam(generator.parameters(), lr=float(lr))
    batches = list(loader)
    ce_fn = nn.CrossEntropyLoss()

    final_loss = ce_val = reg_val = l1_val = tv_val = delta_mean = delta_max = 0.0
    active_pixels = 0

    for step in range(max(1, int(steps))):
        x, _ = batches[step % len(batches)]
        x = x.to(device)
        y_t = torch.full((x.shape[0],), int(target_label), dtype=torch.long, device=device)

        opt.zero_grad(set_to_none=True)

        delta = _sparsify_delta(generator(x), float(sparse_ratio), straight_through=True)
        poisoned_x = torch.clamp(x + delta, 0.0, 1.0)
        logits = model(poisoned_x)

        ce_loss = ce_fn(logits, y_t)
        l1_norm = _trigger_l1_norm(delta)
        tv_loss = _total_variation(delta)
        reg_loss = float(lambda_l1) * l1_norm + float(lambda_tv) * tv_loss
        loss = ce_loss + reg_loss

        loss.backward()
        opt.step()

        final_loss = float(loss.detach().cpu().item())
        ce_val = float(ce_loss.detach().cpu().item())
        reg_val = float(reg_loss.detach().cpu().item())
        l1_val = float(l1_norm.detach().cpu().item())
        tv_val = float(tv_loss.detach().cpu().item())
        delta_mean = float(delta.detach().mean().cpu().item())
        delta_max = float(delta.detach().max().cpu().item())
        active_pixels = int((delta.detach().abs() > 1e-6).float().flatten(1).sum(dim=1).mean().item())

    success = bool(ce_val < 1.0 and l1_val > 0.0)

    return _TriggerInversionResult(
        generator=generator,
        final_loss=final_loss,
        ce_loss=ce_val,
        reg_loss=reg_val,
        l1_norm=l1_val,
        tv_loss=tv_val,
        delta_mean=delta_mean,
        delta_max=delta_max,
        active_pixels=active_pixels,
        generator_type=str(generator_type),
        success=success,
    )


@torch.no_grad()
def _client_entropy(model: nn.Module, loader, device: torch.device, generator: nn.Module, max_batches: int) -> float:
    """Average entropy on generated-trigger data."""
    model.eval()
    entropies: List[torch.Tensor] = []

    for batch_idx, (x, _) in enumerate(loader):
        if batch_idx >= int(max_batches):
            break

        x = x.to(device)
        x_b = _apply_generated_trigger(x, generator)
        prob = torch.softmax(model(x_b), dim=1).clamp_min(1e-12)
        entropy = -torch.sum(prob * torch.log(prob), dim=1)
        entropies.append(entropy.detach().cpu())

    if not entropies:
        return float("inf")
    return float(torch.cat(entropies).mean().item())


def _weighted_kmeans_two_clusters(features: torch.Tensor, sample_weights: torch.Tensor, max_iter: int = 100) -> torch.Tensor:
    """Weighted K-Means with k=2."""
    if features.ndim != 2 or features.shape[0] == 0:
        raise ValueError("features must have shape [n, d] with n > 0")

    n = int(features.shape[0])
    if n == 1:
        return torch.zeros(1, dtype=torch.long)

    low_idx = int(torch.argmin(features[:, 0]).item())
    high_idx = int(torch.argmax(features[:, 0]).item())
    if low_idx == high_idx:
        high_idx = 1 if low_idx == 0 else 0

    centers = torch.stack([features[low_idx], features[high_idx]], dim=0).clone()
    weights = sample_weights.detach().float().clamp_min(1e-6)
    labels = torch.full((n,), -1, dtype=torch.long)

    for _ in range(int(max_iter)):
        old_labels = labels.clone()
        dist = torch.cdist(features, centers, p=2)
        labels = torch.argmin(dist, dim=1)

        for k in range(2):
            mask = labels == k
            if torch.any(mask):
                w = weights[mask].view(-1, 1)
                centers[k] = torch.sum(features[mask] * w, dim=0) / torch.sum(w)

        if torch.equal(labels, old_labels):
            break

    return labels


def _normalize(values: torch.Tensor) -> torch.Tensor:
    lo = torch.min(values)
    hi = torch.max(values)
    if float((hi - lo).abs().item()) <= _EPS:
        return torch.zeros_like(values)
    return (values - lo) / (hi - lo)


def _historical_update_risk(context, client_ids: Sequence[int]) -> Dict[int, float]:
    """Compute non-oracle historical update-risk scores for clients."""
    ratios: Dict[int, List[float]] = {int(cid): [] for cid in client_ids}
    for rec in context.history:
        updates: Mapping[int, torch.Tensor] = rec.get("client_updates", {})
        if not updates:
            continue
        norms = {int(cid): float(torch.norm(update.detach().float().cpu()).item()) for cid, update in updates.items()}
        mean_norm = sum(norms.values()) / max(1, len(norms))
        for cid in client_ids:
            cid = int(cid)
            if cid in norms:
                ratios[cid].append(norms[cid] / max(mean_norm, _EPS))

    raw = torch.tensor(
        [sum(ratios[int(cid)]) / max(1, len(ratios[int(cid)])) for cid in client_ids],
        dtype=torch.float32,
    )
    normed = _normalize(raw)
    return {int(cid): float(score) for cid, score in zip(client_ids, normed.tolist())}


def _cluster_clients_by_combined_risk(
    entropies: Dict[int, float],
    credits: Dict[int, float],
    historical_risk: Dict[int, float],
    credit_weight: float,
    update_weight: float,
) -> Tuple[List[int], Dict[int, int], Dict[int, float], Dict[int, float]]:
    """Cluster clients by low triggered entropy plus historical anomaly risk."""
    client_ids = sorted(entropies.keys())
    if not client_ids:
        return [], {}, {}, {}

    entropy_tensor = torch.tensor([float(entropies[c]) for c in client_ids], dtype=torch.float32)
    entropy_norm = _normalize(entropy_tensor)
    low_entropy_risk = 1.0 - entropy_norm
    credit_risk = 1.0 - torch.tensor([float(credits.get(c, 1.0)) for c in client_ids], dtype=torch.float32).clamp(0.0, 1.0)
    update_risk = torch.tensor([float(historical_risk.get(c, 0.0)) for c in client_ids], dtype=torch.float32).clamp(0.0, 1.0)
    combined = low_entropy_risk + float(credit_weight) * credit_risk + float(update_weight) * update_risk

    if len(client_ids) == 1 or float((combined.max() - combined.min()).abs().item()) <= _EPS:
        labels = torch.zeros(len(client_ids), dtype=torch.long)
        detected = [client_ids[int(torch.argmax(combined).item())]] if client_ids else []
    else:
        features = combined.reshape(-1, 1)
        sample_weights = 1.0 + update_risk + credit_risk
        labels = _weighted_kmeans_two_clusters(features, sample_weights)
        cluster_scores = {}
        for k in [0, 1]:
            mask = labels == k
            cluster_scores[k] = float(combined[mask].mean().item()) if torch.any(mask) else float("-inf")
        malicious_cluster = max(cluster_scores, key=cluster_scores.get)
        detected = [cid for cid, label in zip(client_ids, labels.tolist()) if int(label) == int(malicious_cluster)]

    if len(detected) >= len(client_ids) / 2:
        ordered = sorted(client_ids, key=lambda cid: (combined[client_ids.index(cid)].item(), cid), reverse=True)
        ordered_vals = [float(combined[client_ids.index(cid)].item()) for cid in ordered]
        max_malicious = max(1, (len(client_ids) - 1) // 2)
        best_cut = 1
        best_gap = -1.0
        for idx in range(1, min(max_malicious, len(ordered) - 1) + 1):
            gap = ordered_vals[idx - 1] - ordered_vals[idx]
            if gap > best_gap:
                best_gap = float(gap)
                best_cut = idx
        min_detected = max(1, min(max_malicious, int(round(0.20 * len(client_ids)))))
        best_cut = max(best_cut, min_detected)
        detected = ordered[:best_cut]

    label_by_client = {cid: int(label) for cid, label in zip(client_ids, labels.tolist())}
    normalized_entropy = {cid: float(val) for cid, val in zip(client_ids, entropy_norm.tolist())}
    combined_scores = {cid: float(val) for cid, val in zip(client_ids, combined.tolist())}
    return detected, label_by_client, normalized_entropy, combined_scores


def _cluster_clients_by_entropy_and_credit(
    entropies: Dict[int, float],
    credits: Dict[int, float],
    credit_weight: float,
) -> Tuple[List[int], Dict[int, int], Dict[int, float]]:
    """Cluster clients by entropy and credit."""
    client_ids = sorted(entropies.keys())

    entropy_tensor = torch.tensor([float(entropies[c]) for c in client_ids], dtype=torch.float32)
    entropy_norm = _normalize(entropy_tensor)

    credit_tensor = torch.tensor(
        [float(credits.get(c, 1.0)) for c in client_ids],
        dtype=torch.float32,
    ).clamp(0.0, 1.0)

    credit_risk = 1.0 - credit_tensor

    features = torch.stack([entropy_norm, float(credit_weight) * credit_risk], dim=1)
    sample_weights = 1.0 + float(credit_weight) * credit_risk
    labels = _weighted_kmeans_two_clusters(features, sample_weights)

    cluster_scores: Dict[int, float] = {}
    for k in [0, 1]:
        mask = labels == k
        if not torch.any(mask):
            cluster_scores[k] = float("inf")
            continue
        mean_entropy = float(entropy_norm[mask].mean().item())
        mean_credit_risk = float(credit_risk[mask].mean().item())
        cluster_scores[k] = mean_entropy - float(credit_weight) * mean_credit_risk

    malicious_cluster = min(cluster_scores, key=cluster_scores.get)
    detected = [cid for cid, label in zip(client_ids, labels.tolist()) if int(label) == int(malicious_cluster)]

    if len(detected) >= len(client_ids) / 2:
        ordered = sorted(client_ids, key=lambda cid: (entropy_norm[client_ids.index(cid)].item(), cid))
        ordered_vals = [float(entropy_norm[client_ids.index(cid)].item()) for cid in ordered]
        max_malicious = max(1, (len(client_ids) - 1) // 2)
        best_cut = 1
        best_gap = -1.0
        for idx in range(1, min(max_malicious, len(ordered) - 1) + 1):
            gap = ordered_vals[idx] - ordered_vals[idx - 1]
            if gap > best_gap:
                best_gap = float(gap)
                best_cut = idx
        detected = ordered[:best_cut]

    label_by_client = {cid: int(label) for cid, label in zip(client_ids, labels.tolist())}
    normalized_entropy = {cid: float(val) for cid, val in zip(client_ids, entropy_norm.tolist())}
    return detected, label_by_client, normalized_entropy



def _find_last_linear(model: nn.Module) -> nn.Linear | None:
    last = None
    for module in model.modules():
        if isinstance(module, nn.Linear):
            last = module
    return last


def _logits_and_features(model: nn.Module, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return logits and penultimate features f_i."""
    last_linear = _find_last_linear(model)
    captured: Dict[str, torch.Tensor] = {}
    handle = None

    if last_linear is not None:
        def _hook(_module, inputs):
            captured["feat"] = inputs[0]
        handle = last_linear.register_forward_pre_hook(_hook)

    logits = model(x)

    if handle is not None:
        handle.remove()

    feat = captured.get("feat", logits)
    return logits, feat.flatten(1)


def _unlearn_single_client_model(
    model: nn.Module,
    loader,
    device: torch.device,
    generator: nn.Module,
    epochs: int,
    lr: float,
    weight_decay: float,
    feature_weight: float,
    clean_weight: float,
) -> None:
    """Feature-level unlearning for one detected model."""
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=float(lr), momentum=0.9, weight_decay=float(weight_decay))
    ce_fn = nn.CrossEntropyLoss()

    for _ in range(max(1, int(epochs))):
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            x_b = _apply_generated_trigger(x, generator)

            opt.zero_grad(set_to_none=True)

            clean_logits, clean_feat = _logits_and_features(model, x)
            poison_logits, poison_feat = _logits_and_features(model, x_b)

            poison_ce = ce_fn(poison_logits, y)
            cos = F.cosine_similarity(poison_feat, clean_feat, dim=1, eps=1e-8).mean()
            loss = poison_ce - float(feature_weight) * cos

            if float(clean_weight) > 0.0:
                clean_ce = ce_fn(clean_logits, y)
                loss = loss + float(clean_weight) * clean_ce

            loss.backward()
            opt.step()


def _repair_recovered_model_with_generated_trigger(
    model: nn.Module,
    context,
    generator: nn.Module,
    epochs: int,
    lr: float,
) -> Dict[str, object]:
    """Post-aggregation repair using FedSweep's learned trigger generator."""
    epochs = max(0, int(epochs))
    if epochs <= 0:
        return {
            "fedsweep_plus_post_repair_epochs": 0,
            "fedsweep_plus_post_repair_lr": float(lr),
            "fedsweep_plus_post_repair_batches": 0,
            "fedsweep_plus_post_repair_loss": 0.0,
        }

    teacher = context.new_model_from_vector(model_to_vector(model).detach().cpu())
    teacher.eval()
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=float(lr), momentum=0.0, weight_decay=float(context.args.weight_decay))
    ce_fn = nn.CrossEntropyLoss()
    kl_fn = nn.KLDivLoss(reduction="batchmean")
    temperature = float(getattr(context.args, "temperature", 2.0))
    kd_weight = float(getattr(context.args, "fedsweep_plus_kd_weight", 0.30))
    trigger_weight = float(getattr(context.args, "fedsweep_plus_trigger_weight", 0.70))

    total_loss = 0.0
    batches = 0
    for _ in range(epochs):
        for x, y in context.public_loader:
            x = x.to(context.device)
            y = y.to(context.device)
            x_b = _apply_generated_trigger(x, generator)

            with torch.no_grad():
                teacher_logits = teacher(x)

            clean_logits = model(x)
            poison_logits = model(x_b)
            clean_loss = ce_fn(clean_logits, y)
            trigger_loss = ce_fn(poison_logits, y)
            kd_loss = kl_fn(
                F.log_softmax(clean_logits / temperature, dim=1),
                F.softmax(teacher_logits / temperature, dim=1),
            ) * (temperature ** 2)
            loss = clean_loss + trigger_weight * trigger_loss + kd_weight * kd_loss

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            total_loss += float(loss.detach().cpu().item())
            batches += 1

    return {
        "fedsweep_plus_post_repair_epochs": int(epochs),
        "fedsweep_plus_post_repair_lr": float(lr),
        "fedsweep_plus_post_repair_batches": int(batches),
        "fedsweep_plus_post_repair_loss": total_loss / max(1, batches),
    }


def _aggregate_model_vectors(vectors: Sequence[torch.Tensor], weights: Sequence[float], rule: str, trim_ratio: float) -> torch.Tensor:
    if not vectors:
        raise ValueError("Cannot aggregate empty local model list")
    return robust_aggregate(
        [v.detach().cpu() for v in vectors],
        [float(w) for w in weights],
        rule=rule,
        trim_ratio=trim_ratio,
    ).detach().cpu()


def _select_reference_record(context, args):
    history = context.history
    if not history:
        raise ValueError("FedSweep requires non-empty FL history")

    defense_round = int(_arg(args, "fedsweep_defense_round", -1))
    if defense_round < 0:
        reference_index = len(history) - 1
    else:
        reference_index = min(max(defense_round - 1, 0), len(history) - 1)

    return reference_index, history[reference_index], defense_round


def _simulate_defense_round_local_models(
    context,
    start_global_vec: torch.Tensor,
    selected_clients: Sequence[int],
) -> Tuple[Dict[int, torch.Tensor], Dict[int, float]]:
    args = context.args
    template_state = context.template_state
    local_models: Dict[int, torch.Tensor] = {}
    client_weights: Dict[int, float] = {}

    for cid in selected_clients:
        cid = int(cid)
        local_model = context.model_fn().to(context.device)
        load_vector_into_model(local_model, start_global_vec.detach().cpu(), template_state)

        loader = make_loader(context.client_datasets[cid], batch_size=args.batch_size, shuffle=True)
        attack = args.attack if cid in context.malicious_clients else "none"

        train_one_client(
            local_model,
            loader,
            context.device,
            epochs=args.local_epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            optimizer_name=getattr(args, "optimizer", "sgd"),
            attack=attack,
            target_label=args.target_label,
            poison_fraction=args.poison_fraction,
            trigger_size=args.trigger_size,
            trigger_pattern=getattr(args, "trigger_pattern", "square"),
            trigger_position=getattr(args, "trigger_position", "bottom_right"),
            num_classes=context.num_classes,
        )

        local_vec = model_to_vector(local_model).detach().cpu()
        attack_scale = float(getattr(args, "attack_scale", 1.0))
        if attack != "none" and abs(attack_scale - 1.0) > 1e-12:
            local_vec = start_global_vec.detach().cpu() + attack_scale * (local_vec - start_global_vec.detach().cpu())
        local_models[cid] = local_vec
        client_weights[cid] = float(len(context.client_datasets[cid]))

    return local_models, client_weights


@register
class FedSweepPlus(RecoveryMethod):
    name = "fedsweep_plus"

    def recover(self, context):
        args = context.args
        reference_index, reference_rec, defense_round = _select_reference_record(context, args)

        if defense_round >= 0:
            start_global_vec = reference_rec["global_after"].detach().cpu()
            selected_clients = reference_rec.get("selected_clients", None)
            if not selected_clients:
                selected_clients = list(range(int(args.num_clients)))
            selected_clients = [int(cid) for cid in selected_clients]
            local_models, client_weights = _simulate_defense_round_local_models(context, start_global_vec, selected_clients)
            local_model_source = "simulated_defense_round_from_reference_global_after"
        else:
            local_models: Mapping[int, torch.Tensor] = reference_rec.get("local_model_vectors", {})
            client_weights: Mapping[int, float] = reference_rec.get("client_weights", {})
            selected_clients = sorted(local_models.keys())
            local_model_source = "stored_history_record"

        if not local_models:
            model = context.new_model_from_vector(context.poisoned_vector)
            return RecoveryResult(self.name, model, {
                "method_note": "fedsweep_no_local_models_available",
                "detected_malicious": "",
                "fedsweep_precision": 0.0,
                "fedsweep_recall": 0.0,
                "fedsweep_reference_history_index": int(reference_index),
                "fedsweep_defense_round": int(defense_round),
                "fedsweep_local_model_source": local_model_source,
            })

        selected_clients = sorted(local_models.keys())
        weights = [float(client_weights.get(cid, len(context.client_datasets[cid]))) for cid in selected_clients]

        dummy_global_vec = _aggregate_model_vectors(
            [local_models[cid] for cid in selected_clients],
            weights,
            rule=args.aggregation,
            trim_ratio=args.trim_ratio,
        )
        dummy_global_model = context.new_model_from_vector(dummy_global_vec)

        inversion = _train_trigger_generator(
            model=dummy_global_model,
            loader=context.public_loader,
            device=context.device,
            target_label=context.target_label,
            trigger_size=int(args.trigger_size),
            steps=int(_arg(args, "fedsweep_trigger_steps", 30)),
            lr=float(_arg(args, "fedsweep_generator_lr", 0.01)),
            lambda_l1=float(_arg(args, "fedsweep_lambda_l1", 1e-3)),
            lambda_tv=float(_arg(args, "fedsweep_lambda_tv", 0.0)),
            base_channels=int(_arg(args, "fedsweep_unet_base_channels", 16)),
            strength=float(_arg(args, "fedsweep_trigger_strength", 1.0)),
            generator_type=str(_arg(args, "fedsweep_trigger_generator", "unet")),
            bias_init=float(_arg(args, "fedsweep_trigger_bias_init", -4.0)),
            patch_init_logit=float(_arg(args, "fedsweep_patch_init_logit", -3.0)),
            patch_position=str(_arg(args, "fedsweep_patch_position", "bottom_right")),
            sparse_ratio=float(_arg(args, "fedsweep_trigger_sparse_ratio", 0.02)),
        )
        generator = inversion.generator

        credit_history = getattr(context, "fedsweep_credit_history", None)
        if credit_history is None:
            credit_history = {cid: 1.0 for cid in selected_clients}
        credits = {cid: float(credit_history.get(cid, 1.0)) for cid in selected_clients}

        entropies: Dict[int, float] = {}
        max_batches = int(_arg(args, "fedsweep_entropy_batches", 3))
        for cid in selected_clients:
            local_model = context.new_model_from_vector(local_models[cid])
            entropies[cid] = _client_entropy(local_model, context.public_loader, context.device, generator, max_batches=max_batches)

        entropy_detected, entropy_cluster_labels, entropy_norm_base = _cluster_clients_by_entropy_and_credit(
            entropies,
            credits,
            credit_weight=float(_arg(args, "fedsweep_credit_weight", 1.0)),
        )
        historical_risk = _historical_update_risk(context, selected_clients)
        combined_detected, cluster_labels, entropy_norm, combined_risk = _cluster_clients_by_combined_risk(
            entropies,
            credits,
            credit_weight=float(_arg(args, "fedsweep_credit_weight", 1.0)),
            historical_risk=historical_risk,
            update_weight=float(_arg(args, "fedsweep_plus_update_weight", 0.65)),
        )
        detected = sorted(set(int(cid) for cid in entropy_detected) | set(int(cid) for cid in combined_detected))
        max_detected = max(1, (len(selected_clients) - 1) // 2)
        if len(detected) > max_detected:
            entropy_set = set(int(cid) for cid in entropy_detected)
            detected = [
                cid
                for cid, _ in sorted(
                    (
                        (
                            cid,
                            float(combined_risk.get(cid, 0.0)) + (0.50 if cid in entropy_set else 0.0),
                        )
                        for cid in detected
                    ),
                    key=lambda item: (item[1], item[0]),
                    reverse=True,
                )[:max_detected]
            ]
        for cid, label in entropy_cluster_labels.items():
            cluster_labels.setdefault(cid, label)
        for cid, val in entropy_norm_base.items():
            entropy_norm.setdefault(cid, val)

        credit_decay = float(_arg(args, "fedsweep_credit_decay", 0.5))
        credit_reward = float(_arg(args, "fedsweep_credit_reward", 0.05))
        updated_credits = dict(credits)
        for cid in selected_clients:
            if cid in detected:
                updated_credits[cid] = max(0.0, credits[cid] * credit_decay)
            else:
                updated_credits[cid] = min(1.0, credits[cid] + credit_reward)

        cleaned_vectors: Dict[int, torch.Tensor] = {}
        unlearn_epochs = int(_arg(args, "fedsweep_unlearn_epochs", max(1, int(args.repair_epochs))))
        unlearn_lr = float(_arg(args, "fedsweep_unlearn_lr", args.repair_lr))
        feature_weight = float(_arg(args, "fedsweep_feature_weight", 1.0))
        clean_weight = float(_arg(args, "fedsweep_clean_weight", 0.0))

        for cid in detected:
            if cid not in local_models:
                continue

            client_model = context.new_model_from_vector(local_models[cid])
            _unlearn_single_client_model(
                model=client_model,
                loader=context.public_loader,
                device=context.device,
                generator=generator,
                epochs=unlearn_epochs,
                lr=unlearn_lr,
                weight_decay=float(args.weight_decay),
                feature_weight=feature_weight,
                clean_weight=clean_weight,
            )
            cleaned_vectors[cid] = model_to_vector(client_model).detach().cpu()

        formal_vectors: List[torch.Tensor] = []
        formal_weights: List[float] = []
        for cid in selected_clients:
            formal_vectors.append(cleaned_vectors.get(cid, local_models[cid]).detach().cpu())
            formal_weights.append(float(client_weights.get(cid, len(context.client_datasets[cid]))))

        recovered_vec = _aggregate_model_vectors(
            formal_vectors,
            formal_weights,
            rule=args.aggregation,
            trim_ratio=args.trim_ratio,
        )
        recovered_model = context.new_model_from_vector(recovered_vec)
        post_epochs = int(_arg(args, "fedsweep_plus_post_repair_epochs", max(1, unlearn_epochs // 2)))
        post_lr = float(_arg(args, "fedsweep_plus_post_repair_lr", max(unlearn_lr * 0.5, 1e-4)))
        plus_repair = _repair_recovered_model_with_generated_trigger(
            recovered_model,
            context,
            generator,
            epochs=post_epochs,
            lr=post_lr,
        )

        true_malicious = set(int(cid) for cid in context.malicious_clients)
        detected_set = set(int(cid) for cid in detected)
        tp = len(detected_set & true_malicious)
        fp = len(detected_set - true_malicious)
        fn = len(true_malicious - detected_set)
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)

        entropy_string = ";".join(f"{cid}:{entropies[cid]:.6f}" for cid in selected_clients)
        entropy_norm_string = ";".join(f"{cid}:{entropy_norm[cid]:.3f}" for cid in selected_clients)
        historical_risk_string = ";".join(f"{cid}:{historical_risk[cid]:.3f}" for cid in selected_clients)
        combined_risk_string = ";".join(f"{cid}:{combined_risk[cid]:.3f}" for cid in selected_clients)
        credit_string = ";".join(f"{cid}:{updated_credits[cid]:.3f}" for cid in selected_clients)
        cluster_string = ";".join(f"{cid}:{cluster_labels[cid]}" for cid in selected_clients)

        return RecoveryResult(self.name, recovered_model, {
            "detected_malicious": ";".join(map(str, detected)),
            "fedsweep_true_malicious": ";".join(map(str, sorted(true_malicious))),
            "fedsweep_precision": float(precision),
            "fedsweep_recall": float(recall),
            "fedsweep_true_positive": int(tp),
            "fedsweep_false_positive": int(fp),
            "fedsweep_false_negative": int(fn),
            "fedsweep_entropies": entropy_string,
            "fedsweep_entropy_norm": entropy_norm_string,
            "fedsweep_plus_entropy_only_detected": ";".join(map(str, entropy_detected)),
            "fedsweep_plus_combined_detected": ";".join(map(str, combined_detected)),
            "fedsweep_plus_historical_update_risk": historical_risk_string,
            "fedsweep_plus_combined_risk": combined_risk_string,
            "fedsweep_clusters": cluster_string,
            "fedsweep_updated_credit": credit_string,
            "fedsweep_trigger_loss": float(inversion.final_loss),
            "fedsweep_trigger_ce": float(inversion.ce_loss),
            "fedsweep_trigger_reg": float(inversion.reg_loss),
            "fedsweep_trigger_l1_norm": float(inversion.l1_norm),
            "fedsweep_trigger_tv_loss": float(inversion.tv_loss),
            "fedsweep_trigger_delta_mean": float(inversion.delta_mean),
            "fedsweep_trigger_delta_max": float(inversion.delta_max),
            "fedsweep_trigger_active_pixels": int(inversion.active_pixels),
            "fedsweep_trigger_generator": str(inversion.generator_type),
            "fedsweep_trigger_position": str(getattr(generator, "position", "")),
            "fedsweep_credit_usage": "feature_and_sample_weight",
            "fedsweep_trigger_success": int(inversion.success),
            "fedsweep_unlearned_clients": int(len(cleaned_vectors)),
            "fedsweep_unlearn_epochs": int(unlearn_epochs),
            "fedsweep_unlearn_lr": float(unlearn_lr),
            "fedsweep_feature_weight": float(feature_weight),
            "fedsweep_clean_weight": float(clean_weight),
            "fedsweep_reference_history_index": int(reference_index),
            "fedsweep_defense_round": int(defense_round),
            "fedsweep_reference_round": int(reference_rec.get("round", reference_index)),
            "fedsweep_reference_clients": int(len(selected_clients)),
            "fedsweep_local_model_source": local_model_source,
            "fedsweep_paper_faithful_step4_loader": "public_clean_loader",
            **plus_repair,
            "method_note": "fedsweep_plus_trigger_inversion_historical_risk_post_repair",
        })
