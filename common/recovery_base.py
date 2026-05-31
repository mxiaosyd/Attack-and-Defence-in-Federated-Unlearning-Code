"""Base class and registry for recovery methods."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Type

from torch import nn

from .context import FLContext


@dataclass
class RecoveryResult:
    method: str
    model: nn.Module
    extra: Dict[str, float | int | str]


class RecoveryMethod:
    name = "base"

    def recover(self, context: FLContext) -> RecoveryResult:
        raise NotImplementedError


REGISTRY: Dict[str, Type[RecoveryMethod]] = {}


def register(cls: Type[RecoveryMethod]) -> Type[RecoveryMethod]:
    REGISTRY[cls.name.lower()] = cls
    return cls


def get_method(name: str) -> RecoveryMethod:
    key = name.lower().replace("-", "_")
    aliases = {
        "fedrecover": "fedrecover",
        "crab": "crab",
        "fedup": "fedup",
        "fedsweep": "fedsweep",
        "unlearning_backdoor": "unlearning_backdoor",
        "uba": "unlearning_backdoor",
        "unlearningbackdoor": "unlearning_backdoor",
        "unlearning_backdoor_attacks": "unlearning_backdoor",
        "fast": "fast",
        "fast+": "fast_plus",
        "fast_plus": "fast_plus",
        "fastplus": "fast_plus",
        "mccfed": "mcc_fed",
        "mcc_fed": "mcc_fed",
        "fedsweep+": "fedsweep_plus",
        "fedsweep_plus": "fedsweep_plus",
        "fedsweepplus": "fedsweep_plus",
    }
    key = aliases.get(key, key)
    if key not in REGISTRY:
        raise KeyError(f"Unknown method {name}. Registered methods: {sorted(REGISTRY)}")
    return REGISTRY[key]()
