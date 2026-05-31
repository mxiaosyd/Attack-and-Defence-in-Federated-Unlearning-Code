from __future__ import annotations

from dataclasses import asdict, dataclass

from attack_reproduction.common_attack import repo_root


@dataclass
class FUIAConfig:
    paper: str = "Model Inversion Attack Against Federated Unlearning"
    method: str = "FUIA"
    dataset: str = "mnist"
    classes: tuple[int, ...] = (0, 1)
    data_root: str = str(repo_root() / "data")
    download: bool = False
    train_size: int = 6000
    test_size: int = 1000
    num_clients: int = 10
    partition: str = "iid"
    dirichlet_alpha: float = 0.5
    client_id: int = 0
    forgotten_samples: int = 1
    forgotten_classes: tuple[int, ...] = (1,)
    model: str = "smallcnn"
    pretrain_epochs: int = 20
    unlearning_epochs: int = 3
    local_epochs: int = 2
    batch_size: int = 32
    lr: float = 0.08
    momentum: float = 0.9
    weight_decay: float = 5e-4
    client_unlearning_samples: int = 1
    inversion_steps: int = 1500
    inversion_restarts: int = 5
    inversion_lr: float = 0.05
    tv_weight: float = 5e-5
    gradient_magnitude_weight: float = 1e-3
    client_gamma: float = 0.1
    class_beta: float = 0.5
    seed: int = 2026
    device: str = "auto"
    output_dir: str = str(repo_root() / "outputs" / "attack_reproduction" / "fuia_model_inversion")

    def to_dict(self) -> dict:
        return asdict(self)


DEFAULT_CONFIG = FUIAConfig()
