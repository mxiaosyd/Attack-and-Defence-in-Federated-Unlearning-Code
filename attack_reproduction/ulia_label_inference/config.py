from __future__ import annotations

from dataclasses import asdict, dataclass

from attack_reproduction.common_attack import repo_root


@dataclass
class ULIAConfig:
    paper: str = "Label Inference Attacks Against Federated Unlearning"
    method: str = "ULIA"
    dataset: str = "mnist"
    data_root: str = str(repo_root() / "data")
    download: bool = False
    train_size: int = 8000
    test_size: int = 1000
    num_clients: int = 10
    partition: str = "iid"
    dirichlet_alpha: float = 0.5
    model: str = "smallcnn"
    pretrain_epochs: int = 8
    local_epochs: int = 1
    unlearning_epochs: int = 3
    batch_size: int = 64
    lr: float = 0.05
    momentum: float = 0.9
    weight_decay: float = 0.0
    forgotten_fraction: float = 0.10
    forgotten_label_count: int = 1
    tau: float = 2.0
    attacker_knows_label_count: bool = False
    attack_trials: int = 30
    methods: tuple[str, ...] = ("federaser", "rapid_retrain", "sga_ewc")
    scenarios: tuple[str, ...] = ("sample", "class", "client")
    seed: int = 2026
    device: str = "auto"
    output_dir: str = str(repo_root() / "outputs" / "attack_reproduction" / "ulia_label_inference")

    def to_dict(self) -> dict:
        return asdict(self)


DEFAULT_CONFIG = ULIAConfig()
