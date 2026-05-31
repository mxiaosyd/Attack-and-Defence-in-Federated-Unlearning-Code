from __future__ import annotations

import copy
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import ConcatDataset, Subset

from common.utils import model_to_vector
from attack_reproduction.common_attack import (
    batched_gradient_vector,
    classifier_change_scores,
    cosine_distance,
    ensure_dir,
    evaluate_accuracy,
    get_device,
    gradient_vector,
    infer_label_from_last_layer_gradient,
    load_dataset,
    make_loader,
    make_model,
    mse_value,
    now_stamp,
    partition_dataset,
    psnr_value,
    save_json,
    set_seed,
    ssim_value,
    total_variation,
    train_model,
)
from .config import FUIAConfig


def _client_forgotten_gradient(model, client_dataset, forgotten_positions: list[int], device, config: FUIAConfig) -> torch.Tensor:
    n_total = len(client_dataset)
    remove_set = set(forgotten_positions)
    retained = [i for i in range(n_total) if i not in remove_set]
    full = batched_gradient_vector(model, make_loader(client_dataset, config.batch_size, False), device)
    if retained:
        retained_grad = batched_gradient_vector(model, make_loader(Subset(client_dataset, retained), config.batch_size, False), device)
    else:
        retained_grad = torch.zeros_like(full)
    m = max(1, len(forgotten_positions))
    return (full * n_total - retained_grad * max(0, n_total - m)) / m


def _invert_single_image(
    model,
    target_grad: torch.Tensor,
    label: int,
    image_shape: tuple[int, int, int],
    device,
    steps: int,
    lr: float,
    tv_weight: float,
    auxiliary_grad: torch.Tensor | None = None,
    auxiliary_weight: float = 0.0,
    magnitude_weight: float = 0.0,
    restarts: int = 1,
) -> tuple[torch.Tensor, list[dict[str, float]]]:
    c, h, w = image_shape
    y = torch.tensor([int(label)], dtype=torch.long, device=device)
    target_grad = target_grad.to(device)
    aux = auxiliary_grad.to(device) if auxiliary_grad is not None else None
    best_recon: torch.Tensor | None = None
    best_trace: list[dict[str, float]] = []
    best_objective = float("inf")

    def grad_objective(g: torch.Tensor) -> torch.Tensor:
        loss = cosine_distance(g, target_grad)
        if magnitude_weight > 0:
            loss = loss + float(magnitude_weight) * (torch.log(torch.norm(g).clamp_min(1e-12)) - torch.log(torch.norm(target_grad).clamp_min(1e-12))) ** 2
        if aux is not None and auxiliary_weight > 0:
            aux_loss = cosine_distance(g, aux)
            loss = (1.0 - float(auxiliary_weight)) * loss + float(auxiliary_weight) * aux_loss
        return loss

    for restart in range(max(1, int(restarts))):
        dummy_logits = torch.randn(1, c, h, w, device=device).requires_grad_(True)
        opt = torch.optim.Adam([dummy_logits], lr=float(lr))
        trace: list[dict[str, float]] = []
        for step in range(int(steps)):
            x = torch.sigmoid(dummy_logits)
            model.zero_grad(set_to_none=True)
            g = gradient_vector(model, x, y, create_graph=True, retain_graph=True)
            grad_loss = grad_objective(g)
            tv_loss = total_variation(x)
            loss = grad_loss + float(tv_weight) * tv_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if step % max(1, int(steps) // 10) == 0 or step == int(steps) - 1:
                trace.append({
                    "restart": int(restart),
                    "step": int(step),
                    "gradient_loss": float(grad_loss.detach().cpu().item()),
                    "tv_loss": float(tv_loss.detach().cpu().item()),
                    "total_loss": float(loss.detach().cpu().item()),
                })
        recon = torch.sigmoid(dummy_logits).detach()
        final_grad = gradient_vector(model, recon, y, create_graph=False, retain_graph=False)
        final_obj = float(grad_objective(final_grad).detach().cpu().item())
        if final_obj < best_objective:
            best_objective = final_obj
            best_recon = recon.detach().cpu()
            best_trace = trace
    if best_recon is None:
        raise RuntimeError("FUIA inversion failed to produce a reconstruction")
    return best_recon, best_trace


def _sample_unlearning_attack(model, client, info: dict[str, Any], device, config: FUIAConfig) -> dict[str, Any]:
    forgotten_positions = list(range(min(int(config.forgotten_samples), len(client))))
    target_x, target_y = client[forgotten_positions[0]]
    forgotten_grad = _client_forgotten_gradient(model, client, forgotten_positions, device, config)
    inferred_label = infer_label_from_last_layer_gradient(model, forgotten_grad)
    recon, trace = _invert_single_image(
        model,
        forgotten_grad,
        inferred_label,
        (int(info["in_channels"]), int(info["image_size"]), int(info["image_size"])),
        device,
        config.inversion_steps,
        config.inversion_lr,
        config.tv_weight,
        magnitude_weight=config.gradient_magnitude_weight,
        restarts=config.inversion_restarts,
    )
    target = target_x.unsqueeze(0).detach().cpu()
    mse = mse_value(recon, target)
    return {
        "scenario": "sample_unlearning",
        "true_label": int(target_y),
        "inferred_label": int(inferred_label),
        "label_correct": float(int(inferred_label == int(target_y))),
        "mse": mse,
        "psnr": psnr_value(mse),
        "ssim": ssim_value(recon, target),
        "trace": trace,
        "reconstruction": recon,
        "target": target,
    }


def _client_unlearning_attack(model, clients, info: dict[str, Any], device, config: FUIAConfig) -> dict[str, Any]:
    client = clients[int(config.client_id)]
    target_count = max(1, min(int(config.client_unlearning_samples), len(client)))
    attack_client = Subset(client, list(range(target_count)))
    target_x, target_y = attack_client[0]
    clean_grad = batched_gradient_vector(model, make_loader(attack_client, config.batch_size, False), device)

    retained_tail = Subset(client, list(range(target_count, len(client)))) if target_count < len(client) else None
    retained_clients = [ds for cid, ds in enumerate(clients) if cid != int(config.client_id)]
    if retained_tail is not None:
        retained_clients.append(retained_tail)
    retained_data = ConcatDataset(retained_clients)
    unlearned = copy.deepcopy(model)
    train_model(unlearned, make_loader(retained_data, config.batch_size, True), device, config.unlearning_epochs, config.lr, config.momentum, config.weight_decay)
    global_diff = model_to_vector(model).detach().cpu() - model_to_vector(unlearned).detach().cpu()
    inferred_label = infer_label_from_last_layer_gradient(model, clean_grad)
    recon, trace = _invert_single_image(
        model,
        clean_grad,
        inferred_label,
        (int(info["in_channels"]), int(info["image_size"]), int(info["image_size"])),
        device,
        config.inversion_steps,
        config.inversion_lr,
        config.tv_weight,
        auxiliary_grad=global_diff,
        auxiliary_weight=config.client_gamma,
        magnitude_weight=config.gradient_magnitude_weight,
        restarts=config.inversion_restarts,
    )
    target = target_x.unsqueeze(0).detach().cpu()
    mse = mse_value(recon, target)
    return {
        "scenario": "client_unlearning",
        "true_label": int(target_y),
        "inferred_label": int(inferred_label),
        "label_correct": float(int(inferred_label == int(target_y))),
        "mse": mse,
        "psnr": psnr_value(mse),
        "ssim": ssim_value(recon, target),
        "trace": trace,
        "reconstruction": recon,
        "target": target,
    }


def _class_unlearning_attack(model, train_ds, test_ds, info: dict[str, Any], device, config: FUIAConfig) -> dict[str, Any]:
    forgotten = set(map(int, config.forgotten_classes))
    retained_indices = [i for i in range(len(train_ds)) if int(train_ds[i][1]) not in forgotten]
    retained = Subset(train_ds, retained_indices)
    unlearned = copy.deepcopy(model)
    train_model(unlearned, make_loader(retained, config.batch_size, True), device, config.unlearning_epochs, config.lr, config.momentum, config.weight_decay)
    scores = classifier_change_scores(model, unlearned, beta=config.class_beta)
    k = len(forgotten)
    predicted = set(torch.topk(scores, k=k).indices.cpu().numpy().astype(int).tolist())
    true = forgotten
    class_acc = float(len(predicted & true) / max(1, len(true)))
    return {
        "scenario": "class_unlearning",
        "true_labels": sorted(true),
        "predicted_labels": sorted(predicted),
        "accuracy": class_acc,
        "scores": [float(x) for x in scores.tolist()],
        "post_unlearning_clean_accuracy": evaluate_accuracy(unlearned, make_loader(test_ds, config.batch_size, False), device),
    }


def run(config: FUIAConfig = FUIAConfig()) -> dict[str, Any]:
    start = time.perf_counter()
    set_seed(config.seed)
    device = get_device(config.device)
    out_dir = ensure_dir(Path(config.output_dir) / f"run-{now_stamp()}")

    train_ds, test_ds, info = load_dataset(
        config.dataset,
        root=config.data_root,
        train_size=config.train_size,
        test_size=config.test_size,
        seed=config.seed,
        download=config.download,
        classes=config.classes,
        remap_labels=True,
    )
    clients = partition_dataset(train_ds, config.num_clients, config.partition, config.dirichlet_alpha, config.seed)
    model = make_model(config.model, info).to(device)
    train_model(model, make_loader(train_ds, config.batch_size, True), device, config.pretrain_epochs, config.lr, config.momentum, config.weight_decay)
    clean_acc = evaluate_accuracy(model, make_loader(test_ds, config.batch_size, False), device)

    sample = _sample_unlearning_attack(model, clients[int(config.client_id)], info, device, config)
    client = _client_unlearning_attack(model, clients, info, device, config)
    class_result = _class_unlearning_attack(model, train_ds, test_ds, info, device, config)

    torch.save(
        {
            "sample_reconstruction": sample.pop("reconstruction"),
            "sample_target": sample.pop("target"),
            "client_reconstruction": client.pop("reconstruction"),
            "client_target": client.pop("target"),
        },
        out_dir / "reconstructions.pt",
    )
    payload = {
        "paper": config.paper,
        "method": config.method,
        "config": asdict(config),
        "dataset_info": info,
        "clean_accuracy": clean_acc,
        "results": [sample, client, class_result],
        "metrics": {
            "sample_mse": sample["mse"],
            "sample_psnr": sample["psnr"],
            "client_mse": client["mse"],
            "client_psnr": client["psnr"],
            "class_accuracy": class_result["accuracy"],
            "elapsed_sec": time.perf_counter() - start,
        },
        "output_dir": str(out_dir),
    }
    save_json(out_dir / "result.json", payload)
    save_json(Path(config.output_dir) / "result.json", payload)
    return payload
