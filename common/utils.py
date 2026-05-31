"""Utility functions for the FU recovery benchmark.

The benchmark intentionally keeps the model architecture simple and avoids
framework-specific side effects so that all recovery algorithms can be compared
under exactly the same poisoned FL history.
"""
from __future__ import annotations

import csv
import json
import math
import os
import random
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn


StateDict = Dict[str, torch.Tensor]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def get_device(device: str = "auto") -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def ensure_dir(path: os.PathLike | str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def timestamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def clone_state_dict(state: Mapping[str, torch.Tensor], device: Optional[torch.device] = None) -> StateDict:
    out: StateDict = {}
    for k, v in state.items():
        t = v.detach().clone()
        if device is not None:
            t = t.to(device)
        out[k] = t
    return out


def state_dict_to_vector(state: Mapping[str, torch.Tensor], keys: Optional[Sequence[str]] = None) -> torch.Tensor:
    if keys is None:
        keys = list(state.keys())
    return torch.cat([state[k].detach().reshape(-1).cpu() for k in keys])


def vector_to_state_dict(vector: torch.Tensor, template: Mapping[str, torch.Tensor], keys: Optional[Sequence[str]] = None) -> StateDict:
    if keys is None:
        keys = list(template.keys())
    vector = vector.detach().cpu()
    out: StateDict = {}
    offset = 0
    for k in keys:
        ref = template[k]
        n = int(ref.numel())
        out[k] = vector[offset:offset + n].reshape(ref.shape).to(dtype=ref.dtype)
        offset += n
    if offset != int(vector.numel()):
        raise ValueError(f"Vector has {vector.numel()} values, but template consumed {offset} values")
    return out


def load_vector_into_model(model: nn.Module, vector: torch.Tensor, template: Optional[Mapping[str, torch.Tensor]] = None) -> None:
    if template is None:
        template = model.state_dict()
    state = vector_to_state_dict(vector, template)
    model.load_state_dict(state, strict=True)


def model_to_vector(model: nn.Module) -> torch.Tensor:
    return state_dict_to_vector(model.state_dict())


def add_vectors_weighted(vectors: Sequence[torch.Tensor], weights: Optional[Sequence[float]] = None) -> torch.Tensor:
    if not vectors:
        raise ValueError("No vectors were provided")
    if weights is None:
        weights = [1.0 / len(vectors)] * len(vectors)
    total = torch.zeros_like(vectors[0].detach().cpu())
    wsum = float(sum(weights))
    if abs(wsum) < 1e-12:
        raise ValueError("Sum of weights is zero")
    for v, w in zip(vectors, weights):
        total += v.detach().cpu() * float(w / wsum)
    return total


def l2_norm(vec: torch.Tensor) -> float:
    return float(torch.norm(vec.detach().float().cpu(), p=2).item())


def cosine_similarity(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> float:
    a = a.detach().float().cpu().flatten()
    b = b.detach().float().cpu().flatten()
    return float(torch.dot(a, b).item() / (torch.norm(a).item() * torch.norm(b).item() + eps))


def topk_mask(scores: torch.Tensor, ratio: float) -> torch.Tensor:
    ratio = float(max(0.0, min(1.0, ratio)))
    flat = scores.detach().abs().flatten().cpu()
    if ratio <= 0.0:
        return torch.zeros_like(flat, dtype=torch.bool)
    k = max(1, int(math.ceil(ratio * flat.numel())))
    if k >= flat.numel():
        return torch.ones_like(flat, dtype=torch.bool)
    idx = torch.topk(flat, k=k, largest=True).indices
    mask = torch.zeros_like(flat, dtype=torch.bool)
    mask[idx] = True
    return mask


def robust_aggregate(updates: Sequence[torch.Tensor], weights: Optional[Sequence[float]] = None,
                     rule: str = "fedavg", trim_ratio: float = 0.1) -> torch.Tensor:
    """Aggregate client updates.

    FedAvg is the main setting for the unified experiments. Median and trimmed
    mean are implemented because Crab/FedRecover papers also evaluate recovery
    under robust aggregation rules.
    """
    if not updates:
        raise ValueError("Cannot aggregate an empty update list")
    rule = rule.lower()
    stacked = torch.stack([u.detach().cpu().float() for u in updates], dim=0)
    if rule in {"fedavg", "avg", "mean"}:
        if weights is None:
            return stacked.mean(dim=0)
        w = torch.tensor(weights, dtype=stacked.dtype).reshape(-1, 1)
        w = w / w.sum().clamp_min(1e-12)
        return (stacked * w).sum(dim=0)
    if rule == "median":
        return stacked.median(dim=0).values
    if rule in {"trimmedmean", "trimmed_mean", "trim"}:
        n = stacked.shape[0]
        trim = int(max(0, min(n // 2 - 1 if n > 2 else 0, math.floor(trim_ratio * n))))
        sorted_vals, _ = torch.sort(stacked, dim=0)
        if trim > 0:
            sorted_vals = sorted_vals[trim:-trim]
        return sorted_vals.mean(dim=0)
    if rule == "krum":
        n = stacked.shape[0]
        f = int(max(0, min((n - 2) // 2, math.floor(trim_ratio * n))))
        scores = []
        for i in range(n):
            dists = torch.sum((stacked - stacked[i]) ** 2, dim=1)
            nearest = torch.topk(dists, k=max(1, n - f - 2), largest=False).values
            scores.append(float(nearest.sum().item()))
        return stacked[int(np.argmin(scores))]
    raise ValueError(f"Unknown aggregation rule: {rule}")


def write_csv(path: os.PathLike | str, rows: List[Mapping[str, Any]]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    if not rows:
        return
    fieldnames = sorted({k for row in rows for k in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def append_csv(path: os.PathLike | str, row: Mapping[str, Any]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    exists = path.exists()
    fieldnames = list(row.keys())
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(dict(row))


def json_default(obj: Any) -> Any:
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, torch.device):
        return str(obj)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    raise TypeError(type(obj).__name__)


def save_json(path: os.PathLike | str, payload: Any) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=json_default)


def parameter_count_from_vector(vec: torch.Tensor) -> int:
    return int(vec.numel())


def estimate_history_storage_mb(history: Sequence[Mapping[str, Any]], selected_only: bool = False) -> float:
    total_params = 0
    for rec in history:
        if selected_only and not rec.get("selected_for_storage", True):
            continue
        for upd in rec.get("client_updates", {}).values():
            total_params += int(upd.numel())
    return total_params * 4 / (1024 ** 2)


class Timer:
    def __enter__(self) -> "Timer":
        self.start = time.perf_counter()
        self.elapsed = 0.0
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.elapsed = time.perf_counter() - self.start
