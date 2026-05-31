"""Command line parser shared by root and per-method entry points."""
from __future__ import annotations

import argparse


DEFAULT_METHODS = ["FedRecover", "Crab", "FedUP", "FedSweep", "UnlearningBackdoor", "FAST", "MCCFed"]


def build_parser(single_method: str | None = None) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Federated unlearning recovery benchmark")
    if single_method is None:
        p.add_argument(
            "--methods",
            nargs="+",
            default=DEFAULT_METHODS,
            help="Recovery methods to run",
        )
    else:
        p.set_defaults(methods=[single_method])

    p.add_argument("--dataset", default="synthetic", choices=["synthetic", "mnist", "fmnist", "fashionmnist", "cifar10"])
    p.add_argument("--data_root", default="./data")
    p.add_argument("--download", action="store_true", help="Download torchvision datasets if needed")
    p.add_argument("--train_size", type=int, default=1200)
    p.add_argument("--test_size", type=int, default=300)
    p.add_argument("--partition", default="dirichlet", choices=["iid", "dirichlet", "noniid"])
    p.add_argument("--dirichlet_alpha", type=float, default=0.5)

    p.add_argument(
        "--model",
        default="tinycnn",
        choices=[
            "tinycnn",
            "smallcnn",
            "mlp",
            "resnet18",
            "resnet",
            "cifar_resnet18",
            "cifar-resnet18",
        ],
    )

    p.add_argument("--num_clients", type=int, default=10)
    p.add_argument("--malicious_clients", type=int, default=2)
    p.add_argument("--client_fraction", type=float, default=1.0)
    p.add_argument("--global_rounds", type=int, default=8)
    p.add_argument("--local_epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--test_batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.08)
    p.add_argument("--optimizer", default="sgd", choices=["sgd", "adam"])
    p.add_argument("--server_lr", type=float, default=1.0)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--aggregation", default="fedavg", choices=["fedavg", "median", "trimmedmean", "krum"])
    p.add_argument("--trim_ratio", type=float, default=0.1)
    p.add_argument("--history_stride", type=int, default=1)

    p.add_argument("--attack", default="backdoor", choices=["none", "backdoor", "backdoor_scale", "model_replacement", "label_flip"])
    p.add_argument("--attack_start_round", type=int, default=0)
    p.add_argument("--target_label", type=int, default=0)
    p.add_argument("--poison_fraction", type=float, default=0.7)
    p.add_argument("--attack_scale", type=float, default=1.0, help="Scale malicious submitted model updates for model-replacement style attacks")
    p.add_argument("--trigger_size", type=int, default=4)
    p.add_argument("--trigger_pattern", default="square", choices=["square", "plus", "trojan"], help="Backdoor trigger pattern for train/eval")
    p.add_argument("--trigger_position", default="bottom_right", choices=["bottom_right", "top_left", "center"], help="Backdoor trigger location")
    p.add_argument("--detected_malicious_clients", default="", help="Comma/semicolon separated external detector output for methods that require identified clients")

    p.add_argument("--public_per_client", type=int, default=10)
    p.add_argument(
        "--public_source",
        default="benign_clients",
        choices=["benign_clients", "test_split"],
        help="Source for server public/benchmark data. FAST paper uses a held-out test split.",
    )
    p.add_argument("--public_fraction", type=float, default=0.5, help="Fraction of test data used as server benchmark when --public_source=test_split")
    p.add_argument("--repair_epochs", type=int, default=2)
    p.add_argument("--repair_lr", type=float, default=0.02)
    p.add_argument("--kd_weight", type=float, default=0.5)
    p.add_argument("--temperature", type=float, default=2.0)

    p.add_argument("--crab_select_round_ratio", type=float, default=0.6)
    p.add_argument("--crab_select_client_ratio", type=float, default=0.7)
    p.add_argument("--crab_contamination_threshold", type=float, default=0.05)
    p.add_argument("--crab_loss_drop_alpha", type=float, default=0.1)
    p.add_argument("--crab_sensitivity_threshold", type=float, default=0.3, help="Crab beta threshold for adaptive rollback")
    p.add_argument("--crab_use_oracle_malicious", action="store_true", help="Use ground-truth malicious clients for Crab removal; for controlled ablations only")

    p.add_argument("--fedup_prune_ratio", type=float, default=0.05)
    p.add_argument("--fedup_mode", default="subtract", choices=["subtract", "zero"])
    p.add_argument(
        "--fedup_auto_prune_ratio",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use FedUP's similarity-based pruning-rate rule instead of fixed --fedup_prune_ratio",
    )
    p.add_argument("--fedup_prune_policy", default="auto", choices=["auto", "manual", "fixed"])
    p.add_argument("--fedup_p_min", type=float, default=0.01)
    p.add_argument("--fedup_p_max", type=float, default=0.15)
    p.add_argument("--fedup_gamma", type=float, default=5.0)
    p.add_argument("--fedup_similarity_min", type=float, default=0.5)
    p.add_argument("--fedup_similarity_max", type=float, default=1.0)
    p.add_argument("--fedup_similarity_mode", default="update", choices=["update", "model"])
    p.add_argument("--fedup_rate_limit_rounds", type=int, default=10)
    p.add_argument("--fedup_last_unlearning_round", type=int, default=-10**9)
    p.add_argument("--fedup_detection_round", type=int, default=-1)
    p.add_argument("--fedup_force_unlearning", action="store_true")
    p.add_argument("--fedup_use_oracle_malicious", action="store_true", help="Use ground-truth malicious clients for FedUP; for controlled ablations only")
    p.add_argument("--fedup_recovery_optimizer", default="adam", choices=["adam", "sgd"])
    p.add_argument("--fedup_recovery_lr", type=float, default=0.001)
    p.add_argument("--fedup_recovery_local_epochs", type=int, default=-1, help="Negative means use --local_epochs")
    p.add_argument(
        "--fedup_auto_recovery_rounds",
        action="store_true",
        help="Use FedUP's R_P <= ceil(R* x P) recovery-round bound instead of fixed --repair_epochs",
    )

    p.add_argument("--fast_alpha_grid", default="0.5,1.0,1.5")
    p.add_argument("--fast_benchmark_metric", default="accuracy", choices=["accuracy", "loss"])
    p.add_argument("--fast_extra_epochs", type=int, default=-1, help="Extra benchmark training epochs; negative uses --repair_epochs")
    p.add_argument("--fast_extra_lr", type=float, default=-1.0, help="Extra benchmark training lr; negative uses --repair_lr")
    p.add_argument("--fast_overunlearning_patience", type=int, default=1)
    p.add_argument("--fast_use_lazy_scaling", action="store_true", help="Use FAST J/N malicious contribution scaling")
    p.add_argument("--fast_disable_early_stop", action="store_true")
    p.add_argument("--fast_over_unlearning_tolerance", type=float, default=0.0, help="Benchmark accuracy slack before FAST declares over-unlearning")
    p.add_argument("--fast_unlearning_order", default="forward", choices=["forward", "reverse", "backward"])
    p.add_argument("--fast_max_unlearning_rounds", type=int, default=-1)
    p.add_argument("--fast_weighted_contribution", action="store_true", help="Use data-size weighted malicious contribution instead of FAST equal-client J/N scaling")

    p.add_argument("--mcc_alpha", type=float, default=1.0)
    p.add_argument("--mcc_detect", action="store_true")
    p.add_argument("--mcc_use_oracle_malicious", action="store_true")
    p.add_argument("--mcc_tau", type=float, default=-1.0)
    p.add_argument("--mcc_tau_multiplier", type=float, default=1.05)
    p.add_argument("--mcc_detection_ratio_threshold", type=float, default=1.0, help="Distance-ratio threshold above the round average for MCC-Fed detection")
    p.add_argument("--mcc_detection_policy", default="minority_kmeans", choices=["threshold_high", "minority_kmeans"], help="MCC-Fed Euclidean-distance detection policy")
    p.add_argument("--mcc_unlearn_rounds", type=int, default=5)
    p.add_argument("--mcc_noise_std", type=float, default=0.01)
    p.add_argument("--mcc_regularization_scale", type=float, default=1e-4)
    p.add_argument("--mcc_regularization_mode", default="repel_malicious", choices=["repel_malicious", "proximal_current"])
    p.add_argument("--mcc_contribution_init", action="store_true", help="Initialize MCC-Fed recovery by removing detected clients' stored historical contributions")
    p.add_argument("--mcc_optimizer", default="sgd", choices=["sgd", "adam"])

    p.add_argument("--fedsweep_defense_round", type=int, default=-1)
    p.add_argument("--fedsweep_trigger_generator", default="unet", choices=["patch", "square_patch", "local_patch", "unet", "u-net"])
    p.add_argument("--fedsweep_trigger_steps", type=int, default=30)
    p.add_argument("--fedsweep_generator_lr", type=float, default=0.1)
    p.add_argument("--fedsweep_lambda_l1", type=float, default=1e-3)
    p.add_argument("--fedsweep_lambda_tv", type=float, default=0.0)
    p.add_argument("--fedsweep_unet_base_channels", type=int, default=16)
    p.add_argument("--fedsweep_trigger_strength", type=float, default=1.0)
    p.add_argument("--fedsweep_trigger_bias_init", type=float, default=-4.0)
    p.add_argument("--fedsweep_patch_init_logit", type=float, default=-3.0)
    p.add_argument("--fedsweep_patch_position", default="bottom_right", choices=["bottom_right", "top_left", "top_right", "bottom_left", "center", "search"])
    p.add_argument("--fedsweep_entropy_batches", type=int, default=3)
    p.add_argument("--fedsweep_credit_weight", type=float, default=0.5)
    p.add_argument("--fedsweep_credit_decay", type=float, default=0.9)
    p.add_argument("--fedsweep_credit_reward", type=float, default=0.1)
    p.add_argument("--fedsweep_unlearn_epochs", type=int, default=1)
    p.add_argument("--fedsweep_unlearn_lr", type=float, default=0.01)
    p.add_argument("--fedsweep_feature_weight", type=float, default=1.0)
    p.add_argument("--fedsweep_clean_weight", type=float, default=0.0)
    p.add_argument("--fedsweep_trigger_sparse_ratio", type=float, default=0.02)

    p.add_argument("--unlearning_backdoor_alpha", type=float, default=1.0)
    p.add_argument("--unlearning_backdoor_kd_epochs", type=int, default=-1, help="Negative uses --repair_epochs")
    p.add_argument("--unlearning_backdoor_kd_lr", type=float, default=-1.0, help="Negative uses --repair_lr")
    p.add_argument("--unlearning_backdoor_temperature", type=float, default=-1.0, help="Negative uses --temperature")
    p.add_argument("--unlearning_backdoor_optimizer", default="sgd", choices=["sgd", "adam"])
    p.add_argument("--unlearning_backdoor_use_labels", action="store_true")
    p.add_argument("--unlearning_backdoor_use_oracle_malicious", action="store_true", help="Controlled ablation only: use simulator malicious-client ids when detector output is absent")

    p.add_argument("--fedrecover_warmup_rounds", type=int, default=20, help="FedRecover Tw: exact warm-up rounds")
    p.add_argument("--fedrecover_correction_period", type=int, default=10, help="FedRecover Tc: periodic exact-correction period")
    p.add_argument("--fedrecover_final_tuning_rounds", type=int, default=5, help="FedRecover Tf: exact final tuning rounds")
    p.add_argument("--fedrecover_lbfgs_buffer_size", type=int, default=2, help="FedRecover L-BFGS buffer size s")
    p.add_argument("--fedrecover_tolerance_rate", type=float, default=1e-6, help="FedRecover alpha for abnormality threshold selection")
    p.add_argument("--fedrecover_tau", type=float, default=-1.0, help="Explicit FedRecover tau; negative means estimate from history")
    p.add_argument("--fedrecover_disable_abnormality_fixing", action="store_true", help="Disable FedRecover abnormality fixing")
    p.add_argument("--fedrecover_replay_missed_malicious_attacks", action="store_true", help="Controlled ablation only: replay attacks for malicious clients missed by the detector")

    p.add_argument("--include_retrain_baseline", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    p.add_argument("--num_threads", type=int, default=1, help="CPU torch thread count; 1 is often fastest for small FL simulations")
    p.add_argument("--output_dir", default="./outputs")
    p.add_argument("--save_models", action="store_true")
    return p
