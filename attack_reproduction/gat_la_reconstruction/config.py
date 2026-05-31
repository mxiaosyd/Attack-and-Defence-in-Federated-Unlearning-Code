from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from attack_reproduction.common_attack import repo_root


@dataclass
class GATLAConfig:
    paper: str = "Reconstruction Attacks on Forgotten Data in Federated Unlearning"
    method: str = "GAT-LA"
    dataset: str = "mnist"
    data_root: str = str(repo_root() / "data")
    download: bool = False
    train_size: int = 5000
    test_size: int = 1000
    num_clients: int = 10
    partition: str = "iid"
    dirichlet_alpha: float = 0.5
    client_id: int = 0
    forgotten_samples: int = 1
    model: str = "mlp"
    pretrain_epochs: int = 5
    batch_size: int = 64
    lr: float = 0.05
    momentum: float = 0.9
    weight_decay: float = 5e-4
    aux_samples: int = 256
    generator_epochs: int = 8
    generator_hidden: int = 512
    gradient_feature_dim: int = 512
    autoencoder_epochs: int = 5
    inversion_steps: int = 900
    inversion_restarts: int = 3
    inversion_lr: float = 0.06
    tv_weight: float = 1e-5
    as_weight: float = 1e-3
    gradient_magnitude_weight: float = 1e-2
    infer_label: bool = True
    seed: int = 2026
    device: str = "auto"
    output_dir: str = str(repo_root() / "outputs" / "attack_reproduction" / "gat_la_reconstruction")

    def to_dict(self) -> dict:
        return asdict(self)


DEFAULT_CONFIG = GATLAConfig()
