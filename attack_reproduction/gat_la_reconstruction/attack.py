from __future__ import annotations

import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import Subset

from attack_reproduction.common_attack import (
    batched_gradient_vector,
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
from .config import GATLAConfig


class GradientImageGenerator(nn.Module):
    def __init__(self, feature_dim: int, image_shape: tuple[int, int, int], hidden: int):
        super().__init__()
        c, h, w = image_shape
        self.image_shape = image_shape
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, c * h * w),
            nn.Sigmoid(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        out = self.net(features)
        return out.reshape(features.shape[0], *self.image_shape)


class ConvAutoEncoder(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        width = 16 if in_channels == 1 else 32
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, width, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(width, width * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(width * 2, width, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(width, in_channels, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


def _gradient_features(grad: torch.Tensor, feature_idx: torch.Tensor) -> torch.Tensor:
    g = grad.detach().float().cpu()
    return g[feature_idx].clone()


def _feature_indices(target_grad: torch.Tensor, feature_dim: int) -> torch.Tensor:
    flat = target_grad.detach().abs().cpu().flatten()
    k = min(int(feature_dim), int(flat.numel()))
    if k <= 0:
        raise ValueError("gradient_feature_dim must be positive")
    return torch.topk(flat, k=k, largest=True).indices.sort().values


def _train_generator(
    model: nn.Module,
    aux_dataset,
    device: torch.device,
    feature_idx: torch.Tensor,
    info: dict[str, Any],
    config: GATLAConfig,
) -> GradientImageGenerator:
    image_shape = (int(info["in_channels"]), int(info["image_size"]), int(info["image_size"]))
    generator = GradientImageGenerator(len(feature_idx), image_shape, int(config.generator_hidden)).to(device)
    opt = torch.optim.Adam(generator.parameters(), lr=1e-3)
    mse = nn.MSELoss()
    loader = make_loader(aux_dataset, batch_size=1, shuffle=True)
    model.eval()
    for _ in range(int(config.generator_epochs)):
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            model.zero_grad(set_to_none=True)
            grad = gradient_vector(model, x, y, create_graph=False, retain_graph=False)
            feat = _gradient_features(grad, feature_idx).to(device).unsqueeze(0)
            pred = generator(feat)
            loss = mse(pred, x)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    return generator


def _train_autoencoder(aux_dataset, device: torch.device, info: dict[str, Any], config: GATLAConfig) -> ConvAutoEncoder:
    ae = ConvAutoEncoder(int(info["in_channels"])).to(device)
    opt = torch.optim.Adam(ae.parameters(), lr=1e-3)
    mse = nn.MSELoss()
    loader = make_loader(aux_dataset, batch_size=int(config.batch_size), shuffle=True)
    for _ in range(int(config.autoencoder_epochs)):
        ae.train()
        for x, _ in loader:
            x = x.to(device)
            recon = ae(x)
            loss = mse(recon, x)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    for p in ae.parameters():
        p.requires_grad_(False)
    ae.eval()
    return ae


def _forgotten_gradient_from_client(model: nn.Module, client_dataset, target_positions: list[int], device: torch.device, config: GATLAConfig) -> torch.Tensor:
    n_total = len(client_dataset)
    remove_set = set(target_positions)
    retain_positions = [i for i in range(n_total) if i not in remove_set]
    full_loader = make_loader(client_dataset, batch_size=int(config.batch_size), shuffle=False)
    retain_loader = make_loader(Subset(client_dataset, retain_positions), batch_size=int(config.batch_size), shuffle=False)
    full_grad = batched_gradient_vector(model, full_loader, device)
    retain_grad = batched_gradient_vector(model, retain_loader, device) if retain_positions else torch.zeros_like(full_grad)
    m = max(1, len(target_positions))
    return (full_grad * n_total - retain_grad * max(0, n_total - m)) / m


def _gradient_matching_loss(dummy_grad: torch.Tensor, target_grad: torch.Tensor, magnitude_weight: float) -> torch.Tensor:
    direction = cosine_distance(dummy_grad, target_grad)
    if magnitude_weight <= 0:
        return direction
    dummy_norm = torch.norm(dummy_grad.float()).clamp_min(1e-12)
    target_norm = torch.norm(target_grad.float()).clamp_min(1e-12)
    magnitude = (torch.log(dummy_norm) - torch.log(target_norm)) ** 2
    return direction + float(magnitude_weight) * magnitude


def _invert_with_restarts(
    model: nn.Module,
    label: torch.Tensor,
    target_grad: torch.Tensor,
    init: torch.Tensor,
    autoencoder: nn.Module,
    config: GATLAConfig,
) -> tuple[torch.Tensor, list[dict[str, float]]]:
    device = target_grad.device
    best_recon: torch.Tensor | None = None
    best_trace: list[dict[str, float]] = []
    best_objective = float("inf")
    restarts = max(1, int(config.inversion_restarts))

    for restart in range(restarts):
        if restart == 0:
            start = init.detach().clone().clamp(1e-4, 1 - 1e-4)
            dummy_logits = torch.logit(start).detach().clone().requires_grad_(True)
        else:
            dummy_logits = torch.randn_like(init, device=device).requires_grad_(True)
        opt = torch.optim.Adam([dummy_logits], lr=float(config.inversion_lr))
        trace: list[dict[str, float]] = []

        for step in range(int(config.inversion_steps)):
            dummy = torch.sigmoid(dummy_logits)
            model.zero_grad(set_to_none=True)
            dummy_grad = gradient_vector(model, dummy, label, create_graph=True, retain_graph=True)
            grad_loss = _gradient_matching_loss(dummy_grad, target_grad, config.gradient_magnitude_weight)
            tv_loss = total_variation(dummy)
            as_loss = torch.mean((autoencoder(dummy) - dummy) ** 2)
            loss = grad_loss + float(config.tv_weight) * tv_loss + float(config.as_weight) * as_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            if step % max(1, int(config.inversion_steps) // 10) == 0 or step == int(config.inversion_steps) - 1:
                trace.append({
                    "restart": int(restart),
                    "step": int(step),
                    "gradient_loss": float(grad_loss.detach().cpu().item()),
                    "tv_loss": float(tv_loss.detach().cpu().item()),
                    "as_loss": float(as_loss.detach().cpu().item()),
                    "total_loss": float(loss.detach().cpu().item()),
                })

        recon = torch.sigmoid(dummy_logits).detach()
        final_grad = gradient_vector(model, recon, label, create_graph=False, retain_graph=False)
        final_obj = float(_gradient_matching_loss(final_grad, target_grad, config.gradient_magnitude_weight).detach().cpu().item())
        if final_obj < best_objective:
            best_objective = final_obj
            best_recon = recon.detach().cpu()
            best_trace = trace

    if best_recon is None:
        raise RuntimeError("GAT-LA inversion failed to produce a reconstruction")
    return best_recon, best_trace


def run(config: GATLAConfig = GATLAConfig()) -> dict[str, Any]:
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
    )
    clients = partition_dataset(train_ds, config.num_clients, config.partition, config.dirichlet_alpha, config.seed)
    client = clients[int(config.client_id)]

    model = make_model(config.model, info).to(device)
    pre_loader = make_loader(train_ds, batch_size=config.batch_size, shuffle=True)
    train_model(model, pre_loader, device, config.pretrain_epochs, config.lr, config.momentum, config.weight_decay)
    clean_acc = evaluate_accuracy(model, make_loader(test_ds, config.batch_size, shuffle=False), device)

    target_positions = list(range(min(int(config.forgotten_samples), len(client))))
    target_xs = []
    target_ys = []
    for pos in target_positions:
        x, y = client[pos]
        target_xs.append(x)
        target_ys.append(int(y))
    target_x = torch.stack(target_xs).to(device)
    target_y = torch.tensor(target_ys, dtype=torch.long, device=device)

    forgotten_grad = _forgotten_gradient_from_client(model, client, target_positions, device, config)
    feature_idx = _feature_indices(forgotten_grad, config.gradient_feature_dim)

    aux_count = min(int(config.aux_samples), len(train_ds))
    aux_dataset = Subset(train_ds, list(range(aux_count)))
    generator = _train_generator(model, aux_dataset, device, feature_idx, info, config)
    autoencoder = _train_autoencoder(aux_dataset, device, info, config)

    inferred_label = infer_label_from_last_layer_gradient(model, forgotten_grad) if config.infer_label else int(target_y[0].item())
    label = torch.tensor([inferred_label], dtype=torch.long, device=device)
    feat = _gradient_features(forgotten_grad, feature_idx).to(device).unsqueeze(0)
    with torch.no_grad():
        init = generator(feat).clamp(1e-4, 1 - 1e-4)
    target_grad_dev = forgotten_grad.to(device)
    recon, loss_trace = _invert_with_restarts(model, label, target_grad_dev, init, autoencoder, config)
    target_cpu = target_x[:1].detach().cpu()
    mse = mse_value(recon, target_cpu)
    metrics = {
        "clean_accuracy": clean_acc,
        "mse": mse,
        "psnr": psnr_value(mse),
        "ssim": ssim_value(recon, target_cpu),
        "label_inference_correct": float(int(inferred_label == int(target_y[0].item()))),
        "inferred_label": int(inferred_label),
        "true_label": int(target_y[0].item()),
        "elapsed_sec": time.perf_counter() - start,
    }

    torch.save({"reconstruction": recon, "target": target_cpu, "init": init.detach().cpu()}, out_dir / "reconstruction.pt")
    try:
        from torchvision.utils import save_image

        save_image(torch.cat([target_cpu, init.detach().cpu(), recon], dim=0), out_dir / "target_init_reconstruction.png", nrow=3)
    except Exception:
        pass

    payload = {
        "paper": config.paper,
        "method": config.method,
        "config": asdict(config),
        "dataset_info": info,
        "metrics": metrics,
        "loss_trace": loss_trace,
        "output_dir": str(out_dir),
    }
    save_json(out_dir / "result.json", payload)
    save_json(Path(config.output_dir) / "result.json", payload)
    return payload
