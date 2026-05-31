"""Shared infrastructure for independent attack reproductions."""
from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset, TensorDataset

from common.models import get_model
from common.utils import ensure_dir, load_vector_into_model, model_to_vector, robust_aggregate


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def now_stamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def json_default(obj: Any) -> Any:
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, torch.device):
        return str(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    if torch.is_tensor(obj):
        return obj.detach().cpu().tolist()
    raise TypeError(type(obj).__name__)


def save_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, default=json_default), encoding="utf-8")


def save_csv(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    import csv

    path = Path(path)
    ensure_dir(path.parent)
    fields = sorted({k for row in rows for k in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


class RemappedSubset(Dataset):
    def __init__(self, dataset: Dataset, indices: Sequence[int], label_map: Mapping[int, int] | None = None):
        self.dataset = dataset
        self.indices = list(map(int, indices))
        self.label_map = dict(label_map or {})

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        x, y = self.dataset[self.indices[index]]
        y = int(y)
        if self.label_map:
            y = self.label_map[y]
        return x.float(), y


def _targets(dataset: Dataset) -> np.ndarray:
    if isinstance(dataset, Subset):
        base = _targets(dataset.dataset)
        return base[np.asarray(dataset.indices)]
    if isinstance(dataset, RemappedSubset):
        return np.asarray([int(dataset[i][1]) for i in range(len(dataset))])
    if hasattr(dataset, "targets"):
        targets = getattr(dataset, "targets")
        if torch.is_tensor(targets):
            return targets.detach().cpu().numpy()
        return np.asarray(targets)
    if hasattr(dataset, "labels"):
        return np.asarray(getattr(dataset, "labels"))
    return np.asarray([int(dataset[i][1]) for i in range(len(dataset))])


def load_dataset(
    name: str,
    root: str | Path,
    train_size: int | None,
    test_size: int | None,
    seed: int,
    download: bool,
    classes: Sequence[int] | None = None,
    remap_labels: bool = True,
) -> tuple[Dataset, Dataset, dict[str, Any]]:
    import torchvision.datasets as dsets
    import torchvision.transforms as transforms

    lname = name.lower()
    root = str(root)
    if lname in {"mnist", "fmnist", "fashionmnist", "fashion-mnist"}:
        transform = transforms.ToTensor()
        ds_cls = dsets.MNIST if lname == "mnist" else dsets.FashionMNIST
        train = ds_cls(root=root, train=True, download=download, transform=transform)
        test = ds_cls(root=root, train=False, download=download, transform=transform)
        info = {"name": "mnist" if lname == "mnist" else "fmnist", "in_channels": 1, "image_size": 28, "num_classes": 10}
    elif lname in {"cifar10", "cifar-10"}:
        transform = transforms.ToTensor()
        train = dsets.CIFAR10(root=root, train=True, download=download, transform=transform)
        test = dsets.CIFAR10(root=root, train=False, download=download, transform=transform)
        info = {"name": "cifar10", "in_channels": 3, "image_size": 32, "num_classes": 10}
    elif lname in {"cifar100", "cifar-100"}:
        transform = transforms.ToTensor()
        train = dsets.CIFAR100(root=root, train=True, download=download, transform=transform)
        test = dsets.CIFAR100(root=root, train=False, download=download, transform=transform)
        info = {"name": "cifar100", "in_channels": 3, "image_size": 32, "num_classes": 100}
    else:
        raise ValueError(f"Unsupported attack dataset: {name}")

    rng = np.random.default_rng(int(seed))

    def filter_and_subset(ds: Dataset, n: int | None, offset: int) -> Dataset:
        targets = _targets(ds)
        candidates = np.arange(len(ds))
        label_map = None
        if classes is not None:
            class_list = [int(c) for c in classes]
            mask = np.isin(targets, class_list)
            candidates = candidates[mask]
            if remap_labels:
                label_map = {c: i for i, c in enumerate(class_list)}
        if n is not None and n < len(candidates):
            local_rng = np.random.default_rng(int(seed) + offset)
            candidates = local_rng.choice(candidates, size=int(n), replace=False)
        if label_map is None:
            return Subset(ds, candidates.tolist())
        return RemappedSubset(ds, candidates.tolist(), label_map)

    train = filter_and_subset(train, train_size, 17)
    test = filter_and_subset(test, test_size, 31)
    if classes is not None and remap_labels:
        info = dict(info)
        info["num_classes"] = len(classes)
        info["classes"] = list(map(int, classes))
    return train, test, info


def partition_dataset(dataset: Dataset, num_clients: int, partition: str, alpha: float, seed: int) -> list[Dataset]:
    targets = _targets(dataset)
    rng = np.random.default_rng(int(seed))
    indices = np.arange(len(dataset))
    if partition.lower() in {"iid", "uniform"}:
        rng.shuffle(indices)
        return [Subset(dataset, split.tolist()) for split in np.array_split(indices, int(num_clients))]

    if partition.lower() not in {"dirichlet", "noniid", "non-iid"}:
        raise ValueError(f"Unknown partition: {partition}")
    client_indices: list[list[int]] = [[] for _ in range(int(num_clients))]
    for c in np.unique(targets):
        idx = indices[targets == c]
        rng.shuffle(idx)
        proportions = rng.dirichlet(np.repeat(float(alpha), int(num_clients)))
        cuts = (np.cumsum(proportions) * len(idx)).astype(int)[:-1]
        for cid, split in enumerate(np.split(idx, cuts)):
            client_indices[cid].extend(split.tolist())
    for cid, idxs in enumerate(client_indices):
        if not idxs:
            donor = max(range(len(client_indices)), key=lambda i: len(client_indices[i]))
            idxs.append(client_indices[donor].pop())
        rng.shuffle(idxs)
    return [Subset(dataset, idxs) for idxs in client_indices]


def make_loader(dataset: Dataset, batch_size: int, shuffle: bool, drop_last: bool = False) -> DataLoader:
    return DataLoader(dataset, batch_size=int(batch_size), shuffle=shuffle, num_workers=0, drop_last=drop_last)


def make_model(model_name: str, info: Mapping[str, Any]) -> nn.Module:
    return get_model(
        model_name,
        in_channels=int(info["in_channels"]),
        num_classes=int(info["num_classes"]),
        image_size=int(info["image_size"]),
    )


def train_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    momentum: float = 0.0,
    weight_decay: float = 0.0,
) -> None:
    model.to(device)
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=float(lr), momentum=float(momentum), weight_decay=float(weight_decay))
    ce = nn.CrossEntropyLoss()
    for _ in range(int(epochs)):
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = ce(model(x), y)
            loss.backward()
            opt.step()


@torch.no_grad()
def evaluate_accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        pred = model(x).argmax(dim=1)
        correct += int((pred == y).sum().item())
        total += int(y.numel())
    return float(correct / max(1, total))


def gradient_vector(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    create_graph: bool = False,
    retain_graph: bool = True,
) -> torch.Tensor:
    ce = nn.CrossEntropyLoss()
    logits = model(x)
    loss = ce(logits, y)
    params = [p for p in model.parameters() if p.requires_grad]
    grads = torch.autograd.grad(loss, params, create_graph=create_graph, retain_graph=retain_graph)
    return torch.cat([g.reshape(-1) for g in grads])


def batched_gradient_vector(model: nn.Module, loader: DataLoader, device: torch.device, max_batches: int | None = None) -> torch.Tensor:
    grads: list[torch.Tensor] = []
    n = 0
    for b, (x, y) in enumerate(loader):
        if max_batches is not None and b >= int(max_batches):
            break
        x = x.to(device)
        y = y.to(device)
        model.zero_grad(set_to_none=True)
        g = gradient_vector(model, x, y, create_graph=False, retain_graph=False).detach().cpu()
        grads.append(g * int(y.numel()))
        n += int(y.numel())
    if not grads:
        raise ValueError("No batches available for gradient computation")
    return torch.stack(grads, dim=0).sum(dim=0) / max(1, n)


def cosine_distance(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    a = a.float().flatten()
    b = b.float().flatten().to(a.device)
    return 1.0 - torch.dot(a, b) / (torch.norm(a) * torch.norm(b) + eps)


def total_variation(x: torch.Tensor) -> torch.Tensor:
    tv_h = torch.mean(torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]))
    tv_w = torch.mean(torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]))
    return tv_h + tv_w


def mse_value(x: torch.Tensor, y: torch.Tensor) -> float:
    return float(torch.mean((x.detach().cpu() - y.detach().cpu()) ** 2).item())


def psnr_value(mse: float, max_value: float = 1.0) -> float:
    return float(20.0 * math.log10(max_value) - 10.0 * math.log10(max(float(mse), 1e-12)))


def ssim_value(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.detach().float().cpu()
    y = y.detach().float().cpu()
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    mux = x.mean()
    muy = y.mean()
    vx = x.var(unbiased=False)
    vy = y.var(unbiased=False)
    cov = ((x - mux) * (y - muy)).mean()
    score = ((2 * mux * muy + c1) * (2 * cov + c2)) / ((mux ** 2 + muy ** 2 + c1) * (vx + vy + c2))
    return float(score.clamp(-1.0, 1.0).item())


def infer_label_from_last_layer_gradient(model: nn.Module, grad_vec: torch.Tensor) -> int:
    linear = last_linear(model)
    params = [p for p in model.parameters() if p.requires_grad]
    offset = 0
    best = None
    for p in params:
        n = p.numel()
        if p is linear.weight:
            g = grad_vec[offset:offset + n].reshape_as(linear.weight)
            row_scores = g.mean(dim=1)
            best = int(torch.argmin(row_scores).item())
            break
        offset += n
    if best is None:
        raise ValueError("Could not locate final linear weight in gradient vector")
    return best


def last_linear(model: nn.Module) -> nn.Linear:
    found: nn.Linear | None = None
    for module in model.modules():
        if isinstance(module, nn.Linear):
            found = module
    if found is None:
        raise ValueError("Model has no Linear layer")
    return found


def classifier_change_scores(before: nn.Module, after: nn.Module, beta: float = 0.5) -> torch.Tensor:
    lo = last_linear(before)
    lu = last_linear(after)
    wdiff = torch.sum(torch.abs(lo.weight.detach().cpu() - lu.weight.detach().cpu()), dim=1)
    if lo.bias is not None and lu.bias is not None:
        bdiff = torch.abs(lo.bias.detach().cpu() - lu.bias.detach().cpu())
    else:
        bdiff = torch.zeros_like(wdiff)
    wnorm = wdiff / wdiff.sum().clamp_min(1e-12)
    bnorm = bdiff / bdiff.sum().clamp_min(1e-12)
    return float(beta) * wnorm + (1.0 - float(beta)) * bnorm


def federated_train(
    model: nn.Module,
    client_datasets: Sequence[Dataset],
    device: torch.device,
    rounds: int,
    clients_per_round: int,
    local_epochs: int,
    batch_size: int,
    lr: float,
    momentum: float,
    weight_decay: float,
    aggregation: str = "fedavg",
    seed: int = 0,
    record_updates: bool = False,
) -> list[dict[str, Any]]:
    rng = random.Random(int(seed))
    history: list[dict[str, Any]] = []
    model.to(device)
    global_vec = model_to_vector(model).detach().cpu()
    template = model.state_dict()
    all_clients = list(range(len(client_datasets)))
    for rnd in range(int(rounds)):
        selected = all_clients[:]
        rng.shuffle(selected)
        selected = selected[: min(int(clients_per_round), len(selected))]
        updates: list[torch.Tensor] = []
        weights: list[float] = []
        rec: dict[str, Any] = {"round": rnd, "selected_clients": selected, "client_updates": {}}
        for cid in selected:
            local = make_model_from_template(model)
            local.load_state_dict(template, strict=True)
            load_vector_into_model(local, global_vec, template)
            loader = make_loader(client_datasets[cid], batch_size=batch_size, shuffle=True)
            train_model(local, loader, device, local_epochs, lr, momentum=momentum, weight_decay=weight_decay)
            local_vec = model_to_vector(local).detach().cpu()
            update = local_vec - global_vec
            updates.append(update)
            weights.append(float(len(client_datasets[cid])))
            if record_updates:
                rec["client_updates"][cid] = update.clone()
        agg = robust_aggregate(updates, weights=weights, rule=aggregation)
        global_vec = global_vec + agg.detach().cpu()
        load_vector_into_model(model, global_vec, template)
        template = model.state_dict()
        rec["global_vector"] = global_vec.clone() if record_updates else None
        history.append(rec)
    return history


def make_model_from_template(model: nn.Module) -> nn.Module:
    import copy

    return copy.deepcopy(model)


def concat_without_indices(dataset: Dataset, remove: set[int]) -> Dataset:
    keep = [i for i in range(len(dataset)) if i not in remove]
    return Subset(dataset, keep)


def select_indices_by_label(dataset: Dataset, labels: Iterable[int], limit: int | None = None) -> list[int]:
    wanted = set(map(int, labels))
    out: list[int] = []
    for i in range(len(dataset)):
        _, y = dataset[i]
        if int(y) in wanted:
            out.append(i)
            if limit is not None and len(out) >= int(limit):
                break
    return out


def tensor_dataset_from_items(items: Sequence[tuple[torch.Tensor, int]]) -> TensorDataset:
    xs = torch.stack([x.detach().float().cpu() for x, _ in items])
    ys = torch.tensor([int(y) for _, y in items], dtype=torch.long)
    return TensorDataset(xs, ys)
