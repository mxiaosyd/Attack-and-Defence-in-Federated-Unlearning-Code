"""Evaluation metrics."""
from __future__ import annotations

from typing import Dict

import torch
from torch import nn
from torch.utils.data import DataLoader

from .attacks import add_pattern_trigger


@torch.no_grad()
def evaluate_accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        pred = model(x).argmax(dim=1)
        correct += int((pred == y).sum().item())
        total += int(y.numel())
    return correct / max(1, total)


@torch.no_grad()
def evaluate_loss(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    ce = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    total = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        total_loss += float(ce(logits, y).item())
        total += int(y.numel())
    return total_loss / max(1, total)


@torch.no_grad()
def evaluate_backdoor_asr(model: nn.Module, loader: DataLoader, device: torch.device,
                          target_label: int = 0, trigger_size: int = 4,
                          trigger_pattern: str = "square",
                          trigger_position: str = "bottom_right",
                          exclude_target: bool = True) -> float:
    model.eval()
    success = 0
    total = 0
    for x, y in loader:
        if exclude_target:
            mask = y != int(target_label)
            if int(mask.sum().item()) == 0:
                continue
            x = x[mask]
            y = y[mask]
        x = add_pattern_trigger(
            x,
            trigger_size=trigger_size,
            pattern=trigger_pattern,
            position=trigger_position,
        ).to(device)
        pred = model(x).argmax(dim=1).detach().cpu()
        success += int((pred == int(target_label)).sum().item())
        total += int(pred.numel())
    return success / max(1, total)


@torch.no_grad()
def evaluate_all(model: nn.Module, clean_loader: DataLoader, device: torch.device,
                 target_label: int, trigger_size: int,
                 trigger_pattern: str = "square",
                 trigger_position: str = "bottom_right") -> Dict[str, float]:
    return {
        "clean_acc": evaluate_accuracy(model, clean_loader, device),
        "clean_loss": evaluate_loss(model, clean_loader, device),
        "asr": evaluate_backdoor_asr(model, clean_loader, device, target_label=target_label,
                                     trigger_size=trigger_size,
                                     trigger_pattern=trigger_pattern,
                                     trigger_position=trigger_position),
    }
