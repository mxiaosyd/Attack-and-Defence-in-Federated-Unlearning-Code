"""Direct configuration for comprehensive experiments."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Mapping


METHODS = [
    "FedRecover",
    "Crab",
    "FedUP",
    "FedSweep",
    "UnlearningBackdoor",
    "FAST",
    "MCCFed",
]

IMPROVED_VALIDATION_METHODS = [
    "FedSweep+",
    "FAST+",
]


BASE_ARGS = {
    "data_root": "./data",
    "download": True,
    "seed": 42,
    "model": "smallcnn",
    "train_size": 15000,
    "test_size": 3000,
    "num_clients": 20,
    "malicious_clients": 4,
    "client_fraction": 1.0,
    "dirichlet_alpha": 0.5,
    "global_rounds": 60,
    "local_epochs": 2,
    "batch_size": 64,
    "test_batch_size": 256,
    "lr": 0.08,
    "server_lr": 1.0,
    "aggregation": "fedavg",
    "attack": "backdoor",
    "attack_start_round": 0,
    "target_label": 0,
    "poison_fraction": 0.7,
    "trigger_size": 4,
    "trigger_pattern": "square",
    "trigger_position": "bottom_right",
    "public_source": "test_split",
    "public_fraction": 0.5,
    "repair_epochs": 4,
    "repair_lr": 0.02,
    "num_threads": 1,
}


METHOD_ARGS = [
    "--include_retrain_baseline",
    "--save_models",
    "--fedrecover_warmup_rounds", "20",
    "--fedrecover_correction_period", "10",
    "--fedrecover_final_tuning_rounds", "6",
    "--fedrecover_lbfgs_buffer_size", "2",
    "--fedrecover_tolerance_rate", "1e-6",
    "--crab_select_round_ratio", "0.6",
    "--crab_select_client_ratio", "0.7",
    "--crab_sensitivity_threshold", "0.05",
    "--fedup_auto_prune_ratio",
    "--fedup_auto_recovery_rounds",
    "--fedup_similarity_mode", "model",
    "--fedup_recovery_optimizer", "adam",
    "--fedup_recovery_lr", "0.001",
    "--fedsweep_trigger_generator", "patch",
    "--fedsweep_patch_position", "search",
    "--fedsweep_trigger_steps", "150",
    "--fedsweep_generator_lr", "0.1",
    "--fedsweep_lambda_l1", "0.001",
    "--fedsweep_unet_base_channels", "16",
    "--fedsweep_trigger_sparse_ratio", "0",
    "--fedsweep_entropy_batches", "5",
    "--fedsweep_credit_weight", "0.5",
    "--fedsweep_credit_decay", "0.9",
    "--fedsweep_credit_reward", "0.1",
    "--fedsweep_unlearn_epochs", "2",
    "--fedsweep_unlearn_lr", "0.003",
    "--fedsweep_feature_weight", "0.3",
    "--fedsweep_clean_weight", "2.0",
    "--public_per_client", "50",
    "--unlearning_backdoor_alpha", "1.0",
    "--unlearning_backdoor_kd_epochs", "4",
    "--unlearning_backdoor_kd_lr", "0.02",
    "--unlearning_backdoor_temperature", "2.0",
    "--fast_extra_epochs", "4",
    "--fast_extra_lr", "0.02",
    "--fast_over_unlearning_tolerance", "0.01",
    "--fast_overunlearning_patience", "3",
    "--fast_unlearning_order", "forward",
    "--mcc_detection_ratio_threshold", "1.0",
    "--mcc_detection_policy", "minority_kmeans",
    "--mcc_contribution_init",
    "--mcc_unlearn_rounds", "10",
    "--mcc_alpha", "1.0",
    "--mcc_noise_std", "0.01",
    "--mcc_regularization_scale", "1.0",
    "--mcc_regularization_mode", "repel_malicious",
    "--mcc_optimizer", "sgd",
]


SCENARIOS = [
    {
        "name": "mnist_iid_backdoor_square",
        "dataset": "mnist",
        "partition": "iid",
        "description": "MNIST, IID, square-trigger backdoor",
    },
    {
        "name": "mnist_dirichlet_alpha1_backdoor_square",
        "dataset": "mnist",
        "partition": "dirichlet",
        "dirichlet_alpha": 1.0,
        "description": "MNIST, mildly non-IID Dirichlet alpha=1.0, square-trigger backdoor",
    },
    {
        "name": "mnist_dirichlet_alpha05_backdoor_square",
        "dataset": "mnist",
        "partition": "dirichlet",
        "dirichlet_alpha": 0.5,
        "description": "MNIST, non-IID Dirichlet alpha=0.5, square-trigger backdoor",
    },
    {
        "name": "mnist_dirichlet_alpha01_backdoor_square",
        "dataset": "mnist",
        "partition": "dirichlet",
        "dirichlet_alpha": 0.1,
        "description": "MNIST, strongly non-IID Dirichlet alpha=0.1, square-trigger backdoor",
    },
    {
        "name": "fmnist_iid_backdoor_square",
        "dataset": "fmnist",
        "partition": "iid",
        "description": "Fashion-MNIST, IID, square-trigger backdoor",
    },
    {
        "name": "fmnist_dirichlet_alpha05_backdoor_square",
        "dataset": "fmnist",
        "partition": "dirichlet",
        "dirichlet_alpha": 0.5,
        "description": "Fashion-MNIST, non-IID Dirichlet alpha=0.5, square-trigger backdoor",
    },
    {
        "name": "fmnist_dirichlet_alpha05_backdoor_plus",
        "dataset": "fmnist",
        "partition": "dirichlet",
        "dirichlet_alpha": 0.5,
        "trigger_pattern": "plus",
        "description": "Fashion-MNIST, non-IID, plus-trigger backdoor",
    },
    {
        "name": "fmnist_dirichlet_alpha05_backdoor_center",
        "dataset": "fmnist",
        "partition": "dirichlet",
        "dirichlet_alpha": 0.5,
        "trigger_position": "center",
        "description": "Fashion-MNIST, non-IID, center-position square-trigger backdoor",
    },
    {
        "name": "cifar10_iid_backdoor_square",
        "dataset": "cifar10",
        "partition": "iid",
        "train_size": 12000,
        "test_size": 3000,
        "global_rounds": 40,
        "description": "CIFAR-10, IID, square-trigger backdoor",
    },
    {
        "name": "cifar10_dirichlet_alpha05_backdoor_square",
        "dataset": "cifar10",
        "partition": "dirichlet",
        "dirichlet_alpha": 0.5,
        "train_size": 12000,
        "test_size": 3000,
        "global_rounds": 40,
        "description": "CIFAR-10, non-IID Dirichlet alpha=0.5, square-trigger backdoor",
    },
    {
        "name": "cifar10_dirichlet_alpha05_backdoor_plus",
        "dataset": "cifar10",
        "partition": "dirichlet",
        "dirichlet_alpha": 0.5,
        "train_size": 12000,
        "test_size": 3000,
        "global_rounds": 40,
        "trigger_pattern": "plus",
        "description": "CIFAR-10, non-IID, plus-trigger backdoor",
    },
    {
        "name": "cifar10_dirichlet_alpha05_backdoor_center",
        "dataset": "cifar10",
        "partition": "dirichlet",
        "dirichlet_alpha": 0.5,
        "train_size": 12000,
        "test_size": 3000,
        "global_rounds": 40,
        "trigger_position": "center",
        "description": "CIFAR-10, non-IID, center-position square-trigger backdoor",
    },
    {
        "name": "mnist_dirichlet_alpha05_backdoor_plus",
        "dataset": "mnist",
        "partition": "dirichlet",
        "dirichlet_alpha": 0.5,
        "trigger_pattern": "plus",
        "description": "MNIST, non-IID, plus-trigger backdoor",
    },
    {
        "name": "mnist_dirichlet_alpha05_backdoor_center",
        "dataset": "mnist",
        "partition": "dirichlet",
        "dirichlet_alpha": 0.5,
        "trigger_position": "center",
        "description": "MNIST, non-IID, center-position square-trigger backdoor",
    },
    {
        "name": "mnist_dirichlet_alpha05_label_flip",
        "dataset": "mnist",
        "partition": "dirichlet",
        "dirichlet_alpha": 0.5,
        "attack": "label_flip",
        "description": "MNIST, non-IID, label-flipping poisoning",
    },
    {
        "name": "fmnist_dirichlet_alpha05_label_flip",
        "dataset": "fmnist",
        "partition": "dirichlet",
        "dirichlet_alpha": 0.5,
        "attack": "label_flip",
        "description": "Fashion-MNIST, non-IID, label-flipping poisoning",
    },
    {
        "name": "cifar10_dirichlet_alpha05_label_flip",
        "dataset": "cifar10",
        "partition": "dirichlet",
        "dirichlet_alpha": 0.5,
        "train_size": 12000,
        "test_size": 3000,
        "global_rounds": 40,
        "attack": "label_flip",
        "description": "CIFAR-10, non-IID, label-flipping poisoning",
    },
    {
        "name": "mnist_dirichlet_alpha05_model_replacement",
        "dataset": "mnist",
        "partition": "dirichlet",
        "dirichlet_alpha": 0.5,
        "attack": "model_replacement",
        "attack_scale": 5.0,
        "description": "MNIST, non-IID, scaled model-replacement backdoor",
    },
    {
        "name": "fmnist_dirichlet_alpha05_model_replacement",
        "dataset": "fmnist",
        "partition": "dirichlet",
        "dirichlet_alpha": 0.5,
        "attack": "model_replacement",
        "attack_scale": 5.0,
        "description": "Fashion-MNIST, non-IID, scaled model-replacement backdoor",
    },
    {
        "name": "cifar10_dirichlet_alpha05_model_replacement",
        "dataset": "cifar10",
        "partition": "dirichlet",
        "dirichlet_alpha": 0.5,
        "train_size": 12000,
        "test_size": 3000,
        "global_rounds": 40,
        "attack": "model_replacement",
        "attack_scale": 5.0,
        "description": "CIFAR-10, non-IID, scaled model-replacement backdoor",
    },
    {
        "name": "mnist_dirichlet_alpha05_backdoor_10p_malicious",
        "dataset": "mnist",
        "partition": "dirichlet",
        "dirichlet_alpha": 0.5,
        "malicious_clients": 2,
        "description": "MNIST, non-IID, square-trigger backdoor with 10% malicious clients",
    },
    {
        "name": "mnist_dirichlet_alpha05_backdoor_30p_malicious",
        "dataset": "mnist",
        "partition": "dirichlet",
        "dirichlet_alpha": 0.5,
        "malicious_clients": 6,
        "description": "MNIST, non-IID, square-trigger backdoor with 30% malicious clients",
    },
    {
        "name": "mnist_dirichlet_alpha05_backdoor_40p_malicious",
        "dataset": "mnist",
        "partition": "dirichlet",
        "dirichlet_alpha": 0.5,
        "malicious_clients": 8,
        "description": "MNIST, non-IID, square-trigger backdoor with 40% malicious clients",
    },
    {
        "name": "fmnist_dirichlet_alpha05_backdoor_10p_malicious",
        "dataset": "fmnist",
        "partition": "dirichlet",
        "dirichlet_alpha": 0.5,
        "malicious_clients": 2,
        "description": "Fashion-MNIST, non-IID, square-trigger backdoor with 10% malicious clients",
    },
    {
        "name": "fmnist_dirichlet_alpha05_backdoor_40p_malicious",
        "dataset": "fmnist",
        "partition": "dirichlet",
        "dirichlet_alpha": 0.5,
        "malicious_clients": 8,
        "description": "Fashion-MNIST, non-IID, square-trigger backdoor with 40% malicious clients",
    },
    {
        "name": "cifar10_dirichlet_alpha05_backdoor_10p_malicious",
        "dataset": "cifar10",
        "partition": "dirichlet",
        "dirichlet_alpha": 0.5,
        "train_size": 12000,
        "test_size": 3000,
        "global_rounds": 40,
        "malicious_clients": 2,
        "description": "CIFAR-10, non-IID, square-trigger backdoor with 10% malicious clients",
    },
    {
        "name": "cifar10_dirichlet_alpha05_backdoor_40p_malicious",
        "dataset": "cifar10",
        "partition": "dirichlet",
        "dirichlet_alpha": 0.5,
        "train_size": 12000,
        "test_size": 3000,
        "global_rounds": 40,
        "malicious_clients": 8,
        "description": "CIFAR-10, non-IID, square-trigger backdoor with 40% malicious clients",
    },
    {
        "name": "synthetic_dirichlet_alpha05_backdoor_square",
        "dataset": "synthetic",
        "partition": "dirichlet",
        "dirichlet_alpha": 0.5,
        "description": "Synthetic image data, non-IID Dirichlet alpha=0.5, square-trigger backdoor",
    },
]


DEFAULT_SCENARIOS = [str(scenario["name"]) for scenario in SCENARIOS]


EXPERIMENTS = [
    {
        "name": "data_distribution",
        "description": "Dataset and IID/non-IID distribution comparison.",
        "focus_metrics": [
            "clean_acc",
            "clean_acc_gap_to_retrain",
            "asr",
            "asr_gap_to_retrain",
        ],
        "scenarios": [
            "mnist_iid_backdoor_square",
            "mnist_dirichlet_alpha1_backdoor_square",
            "mnist_dirichlet_alpha05_backdoor_square",
            "mnist_dirichlet_alpha01_backdoor_square",
            "fmnist_iid_backdoor_square",
            "fmnist_dirichlet_alpha05_backdoor_square",
            "cifar10_iid_backdoor_square",
            "cifar10_dirichlet_alpha05_backdoor_square",
        ],
    },
    {
        "name": "attack_robustness",
        "description": "Attack-type and trigger-variant comparison.",
        "focus_metrics": [
            "asr",
            "asr_drop_vs_poisoned",
            "clean_acc",
            "method-specific detection fields",
        ],
        "scenarios": [
            "mnist_dirichlet_alpha05_backdoor_square",
            "mnist_dirichlet_alpha05_backdoor_plus",
            "mnist_dirichlet_alpha05_backdoor_center",
            "mnist_dirichlet_alpha05_label_flip",
            "mnist_dirichlet_alpha05_model_replacement",
            "fmnist_dirichlet_alpha05_backdoor_square",
            "fmnist_dirichlet_alpha05_label_flip",
            "fmnist_dirichlet_alpha05_model_replacement",
            "cifar10_dirichlet_alpha05_backdoor_square",
            "cifar10_dirichlet_alpha05_label_flip",
            "cifar10_dirichlet_alpha05_model_replacement",
        ],
    },
    {
        "name": "malicious_ratio",
        "description": "Malicious-client ratio stress test.",
        "focus_metrics": [
            "clean_acc",
            "asr",
            "asr_drop_vs_poisoned",
            "clean_acc_gap_to_retrain",
        ],
        "scenarios": [
            "mnist_dirichlet_alpha05_backdoor_10p_malicious",
            "mnist_dirichlet_alpha05_backdoor_square",
            "mnist_dirichlet_alpha05_backdoor_30p_malicious",
            "mnist_dirichlet_alpha05_backdoor_40p_malicious",
            "fmnist_dirichlet_alpha05_backdoor_10p_malicious",
            "fmnist_dirichlet_alpha05_backdoor_40p_malicious",
            "cifar10_dirichlet_alpha05_backdoor_10p_malicious",
            "cifar10_dirichlet_alpha05_backdoor_40p_malicious",
        ],
    },
    {
        "name": "efficiency",
        "description": "Runtime and storage-cost comparison on representative settings.",
        "focus_metrics": [
            "runtime_sec",
            "history_storage_mb",
            "method-specific storage fields",
        ],
        "scenarios": [
            "synthetic_dirichlet_alpha05_backdoor_square",
            "mnist_dirichlet_alpha05_backdoor_square",
            "fmnist_dirichlet_alpha05_backdoor_square",
            "cifar10_dirichlet_alpha05_backdoor_square",
        ],
    },
]


DEFAULT_EXPERIMENTS = [str(experiment["name"]) for experiment in EXPERIMENTS]


QUICK_SCENARIO = {
    "name": "synthetic_quick",
    "dataset": "synthetic",
    "partition": "iid",
    "model": "tinycnn",
    "train_size": 240,
    "test_size": 100,
    "num_clients": 5,
    "malicious_clients": 1,
    "global_rounds": 2,
    "local_epochs": 1,
    "repair_epochs": 1,
    "public_source": "benign_clients",
    "public_per_client": 5,
    "description": "Small synthetic quick check",
    "method_args": [
        "--public_per_client", "5",
        "--fedrecover_warmup_rounds", "1",
        "--fedrecover_correction_period", "1",
        "--fedrecover_final_tuning_rounds", "1",
        "--fedsweep_trigger_steps", "4",
        "--fedsweep_entropy_batches", "1",
        "--fedsweep_unlearn_epochs", "1",
        "--unlearning_backdoor_kd_epochs", "1",
        "--fast_extra_epochs", "1",
        "--mcc_unlearn_rounds", "1",
    ],
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def runtime_env(root: Path) -> Dict[str, str]:
    env = dict(os.environ)
    site_packages = root / ".venv" / "Lib" / "site-packages"
    parts = [str(root)]
    if site_packages.exists():
        parts.append(str(site_packages))
    if env.get("PYTHONPATH"):
        parts.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(parts)
    return env


def list_scenario_names() -> List[str]:
    return [str(scenario["name"]) for scenario in SCENARIOS]


def list_experiment_names(include_quick: bool = True) -> List[str]:
    names = [str(experiment["name"]) for experiment in EXPERIMENTS]
    if include_quick:
        names.append("quick")
    return names


def get_scenario(name: str) -> Dict[str, object]:
    if name == str(QUICK_SCENARIO["name"]):
        return dict(QUICK_SCENARIO)
    for scenario in SCENARIOS:
        if scenario["name"] == name:
            return dict(scenario)
    available = ", ".join([*list_scenario_names(), str(QUICK_SCENARIO["name"])])
    raise KeyError(f"Unknown scenario {name!r}. Available: {available}")


def get_experiment(name: str) -> Dict[str, object]:
    if name == "quick":
        return {
            "name": "quick",
            "description": "Small synthetic quick check.",
            "focus_metrics": ["clean_acc", "asr", "runtime_sec"],
            "scenarios": [str(QUICK_SCENARIO["name"])],
        }
    for experiment in EXPERIMENTS:
        if experiment["name"] == name:
            return dict(experiment)
    available = ", ".join(list_experiment_names())
    raise KeyError(f"Unknown experiment {name!r}. Available: {available}")


def scenario_args(scenario: Mapping[str, object]) -> List[str]:
    config = dict(BASE_ARGS)
    config.update({key: value for key, value in scenario.items() if key not in {"name", "description", "method_args"}})

    malicious_clients = int(config["malicious_clients"])
    detected = ",".join(str(i) for i in range(malicious_clients))
    args: List[str] = []
    for key, value in config.items():
        flag = f"--{key}"
        if isinstance(value, bool):
            if value:
                args.append(flag)
        else:
            args.extend([flag, str(value)])
    args.extend(["--detected_malicious_clients", detected])
    args.extend(METHOD_ARGS)
    args.extend(str(item) for item in scenario.get("method_args", []))
    return args
