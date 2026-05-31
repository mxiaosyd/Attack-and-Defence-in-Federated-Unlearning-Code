"""Federated training simulator and unified experiment driver."""
from __future__ import annotations

import copy
import time
from argparse import Namespace
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .context import FLContext
from .data import get_datasets, make_loader, make_public_subset, partition_dataset
from .metrics import evaluate_all
from .models import get_model
from .training import train_one_client
from .utils import (Timer, clone_state_dict, estimate_history_storage_mb, get_device,
                    load_vector_into_model, model_to_vector, robust_aggregate, set_seed)


def _parse_client_list(raw: Any) -> List[int]:
    if raw is None:
        return []
    if isinstance(raw, (list, tuple, set)):
        return [int(x) for x in raw]
    text = str(raw).strip()
    if not text:
        return []
    out: List[int] = []
    for chunk in text.replace(";", ",").split(","):
        chunk = chunk.strip()
        if chunk:
            out.append(int(chunk))
    return out


class FLSimulator:
    def __init__(self, args: Namespace):
        self.args = args
        set_seed(int(args.seed))
        try:
            torch.set_num_threads(int(args.num_threads))
        except Exception:
            pass
        self.device = get_device(args.device)
        self.train_dataset, self.test_dataset, self.info = get_datasets(
            args.dataset, root=args.data_root, train_size=args.train_size, test_size=args.test_size,
            download=args.download, seed=args.seed,
        )
        self.client_datasets = partition_dataset(
            self.train_dataset, num_clients=args.num_clients, partition=args.partition,
            alpha=args.dirichlet_alpha, seed=args.seed,
        )
        self.malicious_clients = list(range(int(args.malicious_clients)))
        self.benign_clients = [i for i in range(args.num_clients) if i not in self.malicious_clients]
        if not self.benign_clients:
            raise ValueError("At least one benign client is required")
        public_source = str(getattr(args, "public_source", "benign_clients")).lower()
        if public_source == "test_split":
            n_test = len(self.test_dataset)
            if n_test <= 1:
                public_ds = self.test_dataset
                eval_test_ds = self.test_dataset
            else:
                frac = max(0.0, min(1.0, float(getattr(args, "public_fraction", 0.5))))
                n_public = max(1, min(n_test - 1, int(round(frac * n_test))))
                rng = np.random.default_rng(int(args.seed) + 321)
                perm = rng.permutation(n_test).tolist()
                public_ds = Subset(self.test_dataset, perm[:n_public])
                eval_test_ds = Subset(self.test_dataset, perm[n_public:])
            self.public_source = "test_split"
        else:
            public_ds = make_public_subset([self.client_datasets[i] for i in self.benign_clients],
                                           per_client=args.public_per_client, seed=args.seed + 123)
            eval_test_ds = self.test_dataset
            self.public_source = "benign_clients"
        self.public_loader = make_loader(public_ds, batch_size=args.batch_size, shuffle=True)
        self.test_loader = make_loader(eval_test_ds, batch_size=args.test_batch_size, shuffle=False)

    def model_fn(self):
        return get_model(self.args.model, self.info.in_channels, self.info.num_classes, self.info.image_size)

    def build_context(self) -> FLContext:
        global_model = self.model_fn().to(self.device)
        init_vector = model_to_vector(global_model)
        template_state = clone_state_dict(global_model.state_dict())
        history: List[Dict[str, Any]] = []
        rng = np.random.default_rng(int(self.args.seed))

        for rnd in range(int(self.args.global_rounds)):
            before_vector = model_to_vector(global_model)
            if self.args.client_fraction >= 1.0:
                selected_clients = list(range(self.args.num_clients))
            else:
                k = max(1, int(round(self.args.client_fraction * self.args.num_clients)))
                selected_clients = rng.choice(self.args.num_clients, size=k, replace=False).tolist()
            updates: Dict[int, torch.Tensor] = {}
            weights: Dict[int, float] = {}
            local_model_vectors: Dict[int, torch.Tensor] = {}
            for cid in selected_clients:
                local_model = self.model_fn().to(self.device)
                load_vector_into_model(local_model, before_vector, template_state)
                loader = make_loader(self.client_datasets[cid], batch_size=self.args.batch_size, shuffle=True)
                is_mal = cid in self.malicious_clients and rnd >= int(self.args.attack_start_round)
                attack = self.args.attack if is_mal else "none"
                train_one_client(
                    local_model,
                    loader,
                    self.device,
                    epochs=self.args.local_epochs,
                    lr=self.args.lr,
                    weight_decay=self.args.weight_decay,
                    optimizer_name=getattr(self.args, "optimizer", "sgd"),
                    attack=attack,
                    target_label=self.args.target_label,
                    poison_fraction=self.args.poison_fraction,
                    trigger_size=self.args.trigger_size,
                    trigger_pattern=getattr(self.args, "trigger_pattern", "square"),
                    trigger_position=getattr(self.args, "trigger_position", "bottom_right"),
                    num_classes=self.info.num_classes,
                )
                local_vec = model_to_vector(local_model)
                update_vec = local_vec - before_vector
                attack_scale = float(getattr(self.args, "attack_scale", 1.0))
                if is_mal and attack != "none" and abs(attack_scale - 1.0) > 1e-12:
                    update_vec = attack_scale * update_vec
                    local_vec = before_vector + update_vec
                updates[cid] = update_vec
                weights[cid] = float(len(self.client_datasets[cid]))
                local_model_vectors[cid] = local_vec.detach().cpu()

            selected_updates = [updates[cid] for cid in selected_clients]
            selected_weights = [weights[cid] for cid in selected_clients]
            agg = robust_aggregate(selected_updates, selected_weights, rule=self.args.aggregation,
                                   trim_ratio=self.args.trim_ratio)
            after_vector = before_vector + self.args.server_lr * agg
            load_vector_into_model(global_model, after_vector, template_state)

            mal_updates = [updates[cid] for cid in selected_clients if cid in self.malicious_clients]
            ben_updates = [updates[cid] for cid in selected_clients if cid in self.benign_clients]
            selected_for_storage = (rnd % max(1, int(self.args.history_stride)) == 0) or rnd == int(self.args.global_rounds) - 1
            history.append({
                "round": rnd,
                "global_before": before_vector.detach().cpu(),
                "global_after": after_vector.detach().cpu(),
                "client_updates": {cid: u.detach().cpu() for cid, u in updates.items()},
                "client_weights": dict(weights),
                "selected_clients": list(selected_clients),
                "local_model_vectors": {cid: v.detach().cpu() for cid, v in local_model_vectors.items()},
                "benign_avg_update": robust_aggregate(ben_updates, None, rule="fedavg") if ben_updates else torch.zeros_like(before_vector),
                "malicious_avg_update": robust_aggregate(mal_updates, None, rule="fedavg") if mal_updates else torch.zeros_like(before_vector),
                "selected_for_storage": selected_for_storage,
            })

        poisoned_vector = model_to_vector(global_model)
        return FLContext(
            args=self.args,
            device=self.device,
            model_fn=self.model_fn,
            template_state=template_state,
            init_vector=init_vector.detach().cpu(),
            poisoned_vector=poisoned_vector.detach().cpu(),
            history=history,
            malicious_clients=self.malicious_clients,
            benign_clients=self.benign_clients,
            client_datasets=self.client_datasets,
            public_loader=self.public_loader,
            test_loader=self.test_loader,
            num_classes=self.info.num_classes,
            target_label=self.args.target_label,
            trigger_size=self.args.trigger_size,
            trigger_pattern=getattr(self.args, "trigger_pattern", "square"),
            trigger_position=getattr(self.args, "trigger_position", "bottom_right"),
            detected_malicious_clients=_parse_client_list(getattr(self.args, "detected_malicious_clients", "")),
        )

    def train_retrain_baseline(self) -> torch.nn.Module:
        """Train from scratch with malicious clients excluded.

        This is used as an approximate gold recovery target. It is intentionally
        trained using the same FL settings and benign client partitions.
        """
        model = self.model_fn().to(self.device)
        template_state = clone_state_dict(model.state_dict())
        rng = np.random.default_rng(int(self.args.seed) + 999)
        for rnd in range(int(self.args.global_rounds)):
            before = model_to_vector(model)
            if self.args.client_fraction >= 1.0:
                selected = list(self.benign_clients)
            else:
                k = max(1, int(round(self.args.client_fraction * len(self.benign_clients))))
                selected = rng.choice(self.benign_clients, size=k, replace=False).tolist()
            updates = []
            weights = []
            for cid in selected:
                local = self.model_fn().to(self.device)
                load_vector_into_model(local, before, template_state)
                loader = make_loader(self.client_datasets[cid], batch_size=self.args.batch_size, shuffle=True)
                train_one_client(
                    local,
                    loader,
                    self.device,
                    epochs=self.args.local_epochs,
                    lr=self.args.lr,
                    weight_decay=self.args.weight_decay,
                    optimizer_name=getattr(self.args, "optimizer", "sgd"),
                    attack="none",
                    target_label=self.args.target_label,
                    poison_fraction=0.0,
                    trigger_size=self.args.trigger_size,
                    trigger_pattern=getattr(self.args, "trigger_pattern", "square"),
                    trigger_position=getattr(self.args, "trigger_position", "bottom_right"),
                    num_classes=self.info.num_classes,
                )
                updates.append(model_to_vector(local) - before)
                weights.append(float(len(self.client_datasets[cid])))
            agg = robust_aggregate(updates, weights, rule=self.args.aggregation, trim_ratio=self.args.trim_ratio)
            load_vector_into_model(model, before + self.args.server_lr * agg, template_state)
        return model


def evaluate_recovery_result(context: FLContext, method: str, model: torch.nn.Module,
                             extra: Dict[str, Any] | None = None) -> Dict[str, Any]:
    metrics = evaluate_all(model, context.test_loader, context.device,
                           target_label=context.target_label, trigger_size=context.trigger_size,
                           trigger_pattern=context.trigger_pattern,
                           trigger_position=context.trigger_position)
    row: Dict[str, Any] = {
        "method": method,
        "clean_acc": metrics["clean_acc"],
        "clean_loss": metrics["clean_loss"],
        "asr": metrics["asr"],
        "history_storage_mb": estimate_history_storage_mb(context.history),
    }
    if extra:
        row.update(extra)
    return row
