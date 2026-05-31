"""Poisoning and backdoor helpers."""
from __future__ import annotations

from typing import Tuple

import torch


def _trigger_bounds(h: int, w: int, trigger_size: int, position: str) -> Tuple[slice, slice]:
    s = min(int(trigger_size), h, w)
    if position == "bottom_right":
        return slice(h - s, h), slice(w - s, w)
    if position == "top_left":
        return slice(0, s), slice(0, s)
    if position == "center":
        y0 = (h - s) // 2
        x0 = (w - s) // 2
        return slice(y0, y0 + s), slice(x0, x0 + s)
    raise ValueError(f"Unknown trigger position: {position}")


def add_pattern_trigger(
    x: torch.Tensor,
    trigger_size: int = 4,
    value: float = 1.0,
    position: str = "bottom_right",
    pattern: str = "square",
) -> torch.Tensor:
    """Add a BadNets-style trigger pattern to a batch or single image tensor.

    Supported patterns are square, plus, and a deterministic trojan-like sparse
    patch. ``x`` shape: [N,C,H,W] or [C,H,W]. Returns a cloned tensor.
    """
    single = x.dim() == 3
    if single:
        x = x.unsqueeze(0)
    out = x.detach().clone()
    _, _, h, w = out.shape
    ys, xs = _trigger_bounds(h, w, trigger_size, position)
    patch = out[:, :, ys, xs]
    pattern = str(pattern).lower()
    if pattern in {"square", "badnets_square", "badnet_square"}:
        patch[:] = value
    elif pattern in {"plus", "badnets_plus", "badnet_plus"}:
        patch.zero_()
        cy = patch.shape[-2] // 2
        cx = patch.shape[-1] // 2
        patch[:, :, cy, :] = value
        patch[:, :, :, cx] = value
    elif pattern in {"trojan", "trnn", "trojan_sparse"}:
        patch.zero_()
        hh, ww = patch.shape[-2], patch.shape[-1]
        coords = {(0, 0), (0, ww - 1), (hh - 1, 0), (hh - 1, ww - 1), (hh // 2, ww // 2)}
        for yy, xx in coords:
            patch[:, :, yy, xx] = value
    else:
        raise ValueError(f"Unknown trigger pattern: {pattern}")
    out[:, :, ys, xs] = patch
    return out.squeeze(0) if single else out


def add_square_trigger(x: torch.Tensor, trigger_size: int = 4, value: float = 1.0,
                       position: str = "bottom_right") -> torch.Tensor:
    """Backward-compatible square-trigger helper."""
    return add_pattern_trigger(
        x,
        trigger_size=trigger_size,
        value=value,
        position=position,
        pattern="square",
    )


def make_backdoor_batch(x: torch.Tensor, y: torch.Tensor, target_label: int,
                        poison_fraction: float = 1.0, trigger_size: int = 4,
                        trigger_value: float = 1.0,
                        trigger_pattern: str = "square",
                        trigger_position: str = "bottom_right") -> Tuple[torch.Tensor, torch.Tensor]:
    """Poison a fraction of a mini-batch with a fixed-label backdoor."""
    x_bd = x.detach().clone()
    y_bd = y.detach().clone()
    n = x.shape[0]
    k = int(round(max(0.0, min(1.0, poison_fraction)) * n))
    if k <= 0:
        return x_bd, y_bd
    idx = torch.randperm(n, device=x.device)[:k]
    x_bd[idx] = add_pattern_trigger(
        x_bd[idx],
        trigger_size=trigger_size,
        value=trigger_value,
        pattern=trigger_pattern,
        position=trigger_position,
    )
    y_bd[idx] = int(target_label)
    return x_bd, y_bd


def make_label_flip_batch(y: torch.Tensor, num_classes: int, source_label: int | None = None,
                          target_label: int | None = None) -> torch.Tensor:
    y2 = y.detach().clone()
    if source_label is None:
        return (y2 + 1) % int(num_classes)
    if target_label is None:
        target_label = (int(source_label) + 1) % int(num_classes)
    mask = y2 == int(source_label)
    y2[mask] = int(target_label)
    return y2
