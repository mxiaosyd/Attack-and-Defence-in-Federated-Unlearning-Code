"""Context objects passed to recovery algorithms."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


@dataclass
class FLContext:
    args: Any
    device: torch.device
    model_fn: Callable[[], nn.Module]
    template_state: Mapping[str, torch.Tensor]
    init_vector: torch.Tensor
    poisoned_vector: torch.Tensor
    history: List[Dict[str, Any]]
    malicious_clients: List[int]
    benign_clients: List[int]
    client_datasets: Sequence[Dataset]
    public_loader: DataLoader
    test_loader: DataLoader
    num_classes: int
    target_label: int
    trigger_size: int
    trigger_pattern: str = "square"
    trigger_position: str = "bottom_right"
    detected_malicious_clients: List[int] | None = None

    def new_model_from_vector(self, vector: torch.Tensor) -> nn.Module:
        from .utils import load_vector_into_model
        model = self.model_fn().to(self.device)
        load_vector_into_model(model, vector, self.template_state)
        return model
