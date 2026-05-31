from __future__ import annotations

from dataclasses import asdict, dataclass

from attack_reproduction.common_attack import repo_root


@dataclass
class CamouflagedPoisoningConfig:
    paper: str = "Hidden Threats in Federated Unlearning: Camouflaged Poisoning Attacks and Their Unlearning Consequences"
    method: str = "Camouflaged Poisoning"
    dataset: str = "mnist"
    data_root: str = str(repo_root() / "data")
    download: bool = False
    train_size: int = 12000
    test_size: int = 2000
    num_clients: int = 10
    clients_per_round: int = 10
    malicious_clients: int = 2
    partition: str = "iid"
    dirichlet_alpha: float = 0.5
    model: str = "smallcnn"
    clean_pretrain_rounds: int = 12
    rounds: int = 8
    unlearning_epochs: int = 5
    unlearning_lr: float = 0.05
    local_epochs: int = 1
    batch_size: int = 32
    lr: float = 0.05
    momentum: float = 0.9
    weight_decay: float = 0.0
    aggregation: str = "fedavg"
    target_label: int = 1
    adversarial_label: int = 0
    target_eval_samples: int = 1
    poison_budget: int = 96
    camouflage_budget: int = 96
    poison_repeats: int = 25
    camouflage_repeats: int = 80
    perturbation_epsilon: float = 0.25
    restarts: int = 3
    optimize_steps: int = 120
    optimize_lr: float = 0.05
    include_no_poison_control: bool = True
    stable_target_reference: bool = True
    seed: int = 2026
    device: str = "auto"
    output_dir: str = str(repo_root() / "outputs" / "attack_reproduction" / "camouflaged_poisoning")

    def to_dict(self) -> dict:
        return asdict(self)


DEFAULT_CONFIG = CamouflagedPoisoningConfig()
