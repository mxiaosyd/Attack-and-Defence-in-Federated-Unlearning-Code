"""Training and repair routines shared by all methods."""
from __future__ import annotations

from typing import Optional

import torch
from torch import nn
from torch.utils.data import DataLoader

from .attacks import add_pattern_trigger, make_backdoor_batch, make_label_flip_batch


def make_optimizer(params, optimizer_name: str = "sgd", lr: float = 0.05, weight_decay: float = 0.0):
    """Create optimizer used by local FL training and repair routines.

    FedUP paper experiments use Adam with lr=1e-3. Most previous benchmark
    commands used SGD, so SGD remains the default for backward compatibility.
    """
    name = str(optimizer_name).lower()
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.0, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def train_one_client(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    epochs: int = 1,
    lr: float = 0.05,
    weight_decay: float = 0.0,
    optimizer_name: str = "sgd",
    attack: str = "none",
    target_label: int = 0,
    poison_fraction: float = 0.5,
    trigger_size: int = 4,
    trigger_pattern: str = "square",
    trigger_position: str = "bottom_right",
    num_classes: int = 10,
) -> None:
    model.train()
    opt = make_optimizer(
        model.parameters(),
        optimizer_name=optimizer_name,
        lr=lr,
        weight_decay=weight_decay,
    )
    ce = nn.CrossEntropyLoss()

    for _ in range(int(epochs)):
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            if attack in {"backdoor", "backdoor_scale", "model_replacement"}:
                x, y = make_backdoor_batch(
                    x,
                    y,
                    target_label=target_label,
                    poison_fraction=poison_fraction,
                    trigger_size=trigger_size,
                    trigger_pattern=trigger_pattern,
                    trigger_position=trigger_position,
                )
            elif attack == "label_flip":
                y = make_label_flip_batch(
                    y,
                    num_classes=num_classes,
                    target_label=target_label,
                )

            opt.zero_grad(set_to_none=True)
            loss = ce(model(x), y)
            loss.backward()
            opt.step()


def finetune_clean(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    epochs: int = 1,
    lr: float = 0.01,
    weight_decay: float = 0.0,
    optimizer_name: str = "sgd",
    proximal_vector: Optional[torch.Tensor] = None,
    proximal_mu: float = 0.0,
) -> None:
    model.train()
    opt = make_optimizer(
        model.parameters(),
        optimizer_name=optimizer_name,
        lr=lr,
        weight_decay=weight_decay,
    )
    ce = nn.CrossEntropyLoss()
    prox = proximal_vector.detach().to(device) if proximal_vector is not None else None

    for _ in range(int(epochs)):
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            opt.zero_grad(set_to_none=True)

            loss = ce(model(x), y)
            if prox is not None and proximal_mu > 0:
                cur = torch.cat([p.reshape(-1) for p in model.parameters()])
                loss = loss + proximal_mu * torch.mean((cur - prox) ** 2)

            loss.backward()
            opt.step()


def distill_and_unlearn_backdoor(
    student: nn.Module,
    teacher: nn.Module,
    loader: DataLoader,
    device: torch.device,
    epochs: int = 1,
    lr: float = 0.01,
    weight_decay: float = 0.0,
    optimizer_name: str = "sgd",
    target_label: int = 0,
    trigger_size: int = 4,
    trigger_pattern: str = "square",
    trigger_position: str = "bottom_right",
    kd_weight: float = 0.5,
    temperature: float = 2.0,
) -> None:
    """Clean KD + triggered true-label training.

    This captures the practical common component of several backdoor-unlearning
    methods: preserve clean behaviour via distillation while explicitly mapping
    triggered inputs back to their original labels.
    """
    student.train()
    teacher.eval()

    opt = make_optimizer(
        student.parameters(),
        optimizer_name=optimizer_name,
        lr=lr,
        weight_decay=weight_decay,
    )
    ce = nn.CrossEntropyLoss()
    kl = nn.KLDivLoss(reduction="batchmean")
    T = float(temperature)

    for _ in range(int(epochs)):
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            x_trig = add_pattern_trigger(
                x,
                trigger_size=trigger_size,
                pattern=trigger_pattern,
                position=trigger_position,
            )

            with torch.no_grad():
                teacher_logits = teacher(x)

            student_logits = student(x)
            trig_logits = student(x_trig)

            loss_ce = ce(student_logits, y) + ce(trig_logits, y)
            loss_kd = kl(
                torch.log_softmax(student_logits / T, dim=1),
                torch.softmax(teacher_logits / T, dim=1),
            ) * (T ** 2)

            loss = (1.0 - kd_weight) * loss_ce + kd_weight * loss_kd

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
