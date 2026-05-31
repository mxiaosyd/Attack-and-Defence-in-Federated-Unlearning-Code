"""Dataset creation and client partitioning.

The default synthetic dataset is deliberately included so that the whole project
can be quick-tested without internet access or torchvision. Real datasets are
loaded through torchvision if it is available in the user's environment.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset


@dataclass
class DatasetInfo:
    name: str
    in_channels: int
    image_size: int
    num_classes: int


class SyntheticImageDataset(Dataset):
    """Small image classification dataset with class-specific prototypes."""

    def __init__(self, n_samples: int, num_classes: int = 10, in_channels: int = 1,
                 image_size: int = 28, seed: int = 0, noise: float = 0.15):
        g = torch.Generator().manual_seed(seed)
        labels = torch.arange(n_samples) % num_classes
        labels = labels[torch.randperm(n_samples, generator=g)]
        prototypes = torch.zeros(num_classes, in_channels, image_size, image_size)
        grid = int(math.ceil(math.sqrt(num_classes)))
        cell = max(2, image_size // (grid + 1))
        for c in range(num_classes):
            r = c // grid
            col = c % grid
            y0 = min(image_size - cell, (r + 1) * cell - cell // 2)
            x0 = min(image_size - cell, (col + 1) * cell - cell // 2)
            prototypes[c, :, y0:y0 + cell, x0:x0 + cell] = 0.8
            prototypes[c, :, :, (x0 + c) % image_size] += 0.15
        data = prototypes[labels].clone()
        data += noise * torch.randn(data.shape, generator=g)
        data = data.clamp(0.0, 1.0)
        self.data = data.float()
        self.targets = labels.long()

    def __len__(self) -> int:
        return int(self.targets.numel())

    def __getitem__(self, index: int):
        return self.data[index], self.targets[index]


def _load_torchvision_dataset(name: str, root: str, train: bool, download: bool):
    try:
        import torchvision.transforms as T
        import torchvision.datasets as dsets
    except Exception as e:
        raise RuntimeError(
            "torchvision is not available or is incompatible in this environment. "
            "Use --dataset synthetic for quick checks, or install a matching torch/torchvision build."
        ) from e

    lname = name.lower()
    if lname in {"mnist", "fmnist", "fashionmnist", "fashion-mnist"}:
        transform = T.ToTensor()
        if lname == "mnist":
            return dsets.MNIST(root=root, train=train, transform=transform, download=download)
        return dsets.FashionMNIST(root=root, train=train, transform=transform, download=download)
    if lname in {"cifar10", "cifar-10"}:
        transform = T.ToTensor()
        return dsets.CIFAR10(root=root, train=train, transform=transform, download=download)
    raise ValueError(f"Unsupported real dataset: {name}")


def get_datasets(name: str, root: str = "./data", train_size: int | None = None,
                 test_size: int | None = None, download: bool = False,
                 seed: int = 0) -> Tuple[Dataset, Dataset, DatasetInfo]:
    lname = name.lower()
    if lname == "synthetic":
        n_train = int(train_size or 2000)
        n_test = int(test_size or 500)
        info = DatasetInfo("synthetic", in_channels=1, image_size=28, num_classes=10)
        return (
            SyntheticImageDataset(n_train, info.num_classes, info.in_channels, info.image_size, seed=seed),
            SyntheticImageDataset(n_test, info.num_classes, info.in_channels, info.image_size, seed=seed + 10000),
            info,
        )

    train = _load_torchvision_dataset(lname, root=root, train=True, download=download)
    test = _load_torchvision_dataset(lname, root=root, train=False, download=download)
    if lname == "mnist":
        info = DatasetInfo("mnist", 1, 28, 10)
    elif lname in {"fmnist", "fashionmnist", "fashion-mnist"}:
        info = DatasetInfo("fmnist", 1, 28, 10)
    elif lname in {"cifar10", "cifar-10"}:
        info = DatasetInfo("cifar10", 3, 32, 10)
    else:
        raise ValueError(lname)

    def maybe_subset(ds: Dataset, n: int | None, offset_seed: int) -> Dataset:
        if n is None or n >= len(ds):
            return ds
        rng = np.random.default_rng(seed + offset_seed)
        idx = rng.choice(len(ds), size=int(n), replace=False).tolist()
        return Subset(ds, idx)

    return maybe_subset(train, train_size, 1), maybe_subset(test, test_size, 2), info


def get_targets(dataset: Dataset) -> np.ndarray:
    if isinstance(dataset, Subset):
        base_targets = get_targets(dataset.dataset)
        return base_targets[np.array(dataset.indices)]
    if hasattr(dataset, "targets"):
        targets = getattr(dataset, "targets")
        if isinstance(targets, list):
            return np.array(targets)
        if torch.is_tensor(targets):
            return targets.detach().cpu().numpy()
        return np.asarray(targets)
    return np.array([int(dataset[i][1]) for i in range(len(dataset))])


def partition_dataset(dataset: Dataset, num_clients: int, partition: str = "dirichlet",
                      alpha: float = 0.5, seed: int = 0) -> List[Subset]:
    targets = get_targets(dataset)
    rng = np.random.default_rng(seed)
    num_clients = int(num_clients)
    if partition.lower() in {"iid", "uniform"}:
        indices = np.arange(len(dataset))
        rng.shuffle(indices)
        splits = np.array_split(indices, num_clients)
        return [Subset(dataset, split.tolist()) for split in splits]

    if partition.lower() in {"dirichlet", "noniid", "non-iid"}:
        client_indices: List[List[int]] = [[] for _ in range(num_clients)]
        classes = np.unique(targets)
        for c in classes:
            idx_c = np.where(targets == c)[0]
            rng.shuffle(idx_c)
            proportions = rng.dirichlet(np.repeat(alpha, num_clients))
            proportions = (np.cumsum(proportions) * len(idx_c)).astype(int)[:-1]
            splits = np.split(idx_c, proportions)
            for cid, split in enumerate(splits):
                client_indices[cid].extend(split.tolist())
        for cid in range(num_clients):
            if len(client_indices[cid]) == 0:
                donor = int(np.argmax([len(x) for x in client_indices]))
                client_indices[cid].append(client_indices[donor].pop())
        for idx in client_indices:
            rng.shuffle(idx)
        return [Subset(dataset, idx) for idx in client_indices]

    raise ValueError(f"Unknown partition: {partition}")


def make_loader(dataset: Dataset, batch_size: int, shuffle: bool = False,
                num_workers: int = 0, drop_last: bool = False) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, drop_last=drop_last)


def make_public_subset(client_datasets: Sequence[Dataset], per_client: int = 8,
                       seed: int = 0) -> Dataset:
    """Build a small server benchmark set from benign clients.

    This is used by FAST/FedSweep/UBA as the clean benchmark or public repair
    data, matching the common assumption that a small clean validation set is
    available at the server.
    """
    xs: List[torch.Tensor] = []
    ys: List[torch.Tensor] = []
    rng = random.Random(seed)
    for ds in client_datasets:
        n = len(ds)
        if n <= 0:
            continue
        chosen = list(range(n))
        rng.shuffle(chosen)
        for i in chosen[:max(1, per_client)]:
            x, y = ds[i]
            xs.append(x.detach().clone() if torch.is_tensor(x) else torch.tensor(x))
            ys.append(torch.tensor(int(y), dtype=torch.long))
    if not xs:
        raise ValueError("Cannot build public subset from empty client datasets")
    return TensorDataset(torch.stack(xs).float(), torch.stack(ys).long())
