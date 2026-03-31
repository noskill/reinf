#!/usr/bin/env python3
"""Shared utility functions for world-model training and evaluation."""

import random
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def sg(x: torch.Tensor) -> torch.Tensor:
    return x.detach()
def make_probe_head(in_dim: int, out_dim: int, hidden_dim: int, num_layers: int) -> nn.Module:
    """Build probe head with configurable depth (number of Linear layers)."""
    layers = int(num_layers)
    if layers <= 1:
        return nn.Linear(in_dim, out_dim)
    h = int(hidden_dim)
    if h <= 0:
        raise ValueError("probe_hidden_dim must be > 0 when probe_layers > 1")
    blocks: List[nn.Module] = [nn.Linear(in_dim, h), nn.ReLU()]
    for _ in range(layers - 2):
        blocks.extend([nn.Linear(h, h), nn.ReLU()])
    blocks.append(nn.Linear(h, out_dim))
    return nn.Sequential(*blocks)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # mask: True for padded -> ignore
    valid = ~mask
    if valid.sum() == 0:
        return torch.tensor(0.0, device=pred.device)
    diff = pred[valid] - target[valid]
    return (diff * diff).mean()


def masked_rmse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid = ~mask
    if valid.sum() == 0:
        return torch.tensor(0.0, device=pred.device)
    diff = pred[valid] - target[valid]
    return torch.sqrt((diff * diff).mean())


def soft_cross_entropy(
    logits: torch.Tensor, target_probs: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    valid = ~mask
    if valid.sum() == 0:
        return torch.tensor(0.0, device=logits.device)
    logp = F.log_softmax(logits, dim=-1)
    loss = -(target_probs * logp).sum(dim=-1)
    return loss[valid].mean()


def make_soft_table(size: int, sigma: float, device: torch.device) -> torch.Tensor:
    # Build a [size, size] lookup where each row is a soft target distribution
    # centered on the true index (Gaussian kernel over class distance).
    if sigma is None or sigma <= 0:
        return torch.eye(size, device=device)
    coords = torch.arange(size, device=device, dtype=torch.float32)
    diff = coords[None, :] - coords[:, None]
    probs = torch.exp(-0.5 * (diff / sigma) ** 2)
    probs = probs / probs.sum(dim=-1, keepdim=True)
    return probs


def make_label_smoothing_table(size: int, epsilon: float, device: torch.device) -> torch.Tensor:
    if epsilon is None or epsilon <= 0:
        return torch.eye(size, device=device)
    off = epsilon / max(1, size - 1)
    table = torch.full((size, size), off, device=device)
    diag = 1.0 - epsilon
    table.fill_diagonal_(diag)
    return table


def expected_from_logits(logits: torch.Tensor, min_val: float) -> torch.Tensor:
    probs = F.softmax(logits, dim=-1)
    vals = torch.arange(logits.shape[-1], device=logits.device, dtype=probs.dtype) + min_val
    return (probs * vals).sum(dim=-1)


def masked_lr_metrics_logits(
    pred_left: torch.Tensor,
    pred_right: torch.Tensor,
    target_sensor_idx: torch.Tensor,
    mask: torch.Tensor,
    sensor_min: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    valid = ~mask
    if valid.sum() == 0:
        zero = torch.tensor(0.0, device=pred_left.device)
        return zero, zero
    min_l = float(sensor_min[0].item())
    min_r = float(sensor_min[2].item())
    pred_l = expected_from_logits(pred_left, min_l)
    pred_r = expected_from_logits(pred_right, min_r)
    tgt_l = target_sensor_idx[..., 0].to(pred_l.dtype)
    tgt_r = target_sensor_idx[..., 2].to(pred_r.dtype)
    diff = torch.stack([pred_l - tgt_l, pred_r - tgt_r], dim=-1)[valid]
    lr_rmse = torch.sqrt((diff * diff).mean())

    # Report exact classification accuracy from categorical winners.
    pred_l_idx = torch.argmax(pred_left, dim=-1).to(torch.long) + int(min_l)
    pred_r_idx = torch.argmax(pred_right, dim=-1).to(torch.long) + int(min_r)
    tgt_l_idx = torch.round(target_sensor_idx[..., 0]).to(torch.long)
    tgt_r_idx = torch.round(target_sensor_idx[..., 2]).to(torch.long)
    both_correct = (pred_l_idx == tgt_l_idx) & (pred_r_idx == tgt_r_idx)
    lr_acc = both_correct[valid].float().mean()
    return lr_rmse, lr_acc


def masked_lr_metrics(
    pred_sensor: torch.Tensor,
    target_sensor: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    valid = ~mask
    if valid.sum() == 0:
        zero = torch.tensor(0.0, device=pred_sensor.device)
        return zero, zero
    # select left/right indices 0 and 2 in sensor slice
    pred_lr = pred_sensor[..., [0, 2]]
    tgt_lr = target_sensor[..., [0, 2]]

    # RMSE in raw units
    diff = (pred_lr - tgt_lr)[valid]
    lr_rmse = torch.sqrt((diff * diff).mean())

    # Accuracy: both left and right match after rounding
    pred_round = torch.round(pred_lr)
    tgt_round = torch.round(tgt_lr)
    both_correct = (pred_round == tgt_round).all(dim=-1)
    lr_acc = both_correct[valid].float().mean()
    return lr_rmse, lr_acc


def compute_baseline_lr(
    loader,
    device: torch.device,
) -> tuple[float, float]:
    total_lr_rmse = 0.0
    total_lr_acc = 0.0
    total_batches = 0
    for obs, y_sensor, _y_sensor_idx, _y_loc, _y_head, _y_turn, _y_step, _kpm in loader:
        obs = {k: v.to(device) for k, v in obs.items()}
        y_sensor = y_sensor.to(device)
        kpm = obs["key_padding_mask"]
        # Build a pseudo prediction that repeats current sensors.
        pred_sensor = obs["sensor"]
        lr_rmse, lr_acc = masked_lr_metrics(pred_sensor, y_sensor, kpm)
        total_lr_rmse += float(lr_rmse.detach().cpu())
        total_lr_acc += float(lr_acc.detach().cpu())
        total_batches += 1
    if total_batches == 0:
        return 0.0, 0.0
    return total_lr_rmse / total_batches, total_lr_acc / total_batches


def masked_coord_rmse(
    logits: torch.Tensor,
    target_idx: torch.Tensor,
    mask: torch.Tensor,
    loc_min: torch.Tensor,
) -> torch.Tensor:
    valid = ~mask
    if valid.sum() == 0:
        return torch.tensor(0.0, device=logits.device)
    probs = F.softmax(logits, dim=-1)
    vals = torch.arange(logits.shape[-1], device=logits.device, dtype=probs.dtype) + loc_min
    expected = (probs * vals).sum(dim=-1)
    target = target_idx.to(expected.dtype) + loc_min
    diff = (expected - target)[valid]
    return torch.sqrt((diff * diff).mean())


def masked_coord_acc(logits: torch.Tensor, target_idx: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid = ~mask
    if valid.sum() == 0:
        return torch.tensor(0.0, device=logits.device)
    pred = logits.argmax(dim=-1)
    correct = (pred == target_idx)[valid].float().mean()
    return correct


def masked_action_acc(logits: torch.Tensor, target_idx: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid = (~mask) & (target_idx >= 0)
    if valid.sum() == 0:
        return torch.tensor(0.0, device=logits.device)
    pred = logits.argmax(dim=-1)
    correct = (pred == target_idx)[valid].float().mean()
    return correct


def batch_row_next_step_contrastive_pair_loss(
    pred_emb: Optional[torch.Tensor],
    target_emb: Optional[torch.Tensor],
    key_padding_mask: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """InfoNCE for world models: prior-branch prediction vs posterior-z projection.

    Positive pair: (pred_t, target_{t+1}) from the same row.
    Negative pool: all valid target_{t+1} from other rows.
    Gradients flow through both branches.
    """
    if pred_emb is None or target_emb is None:
        return torch.tensor(0.0, device=key_padding_mask.device)
    if pred_emb.ndim != 3 or target_emb.ndim != 3:
        raise ValueError(
            f"Expected pred_emb/target_emb [B, T, D], got {tuple(pred_emb.shape)} and {tuple(target_emb.shape)}"
        )
    if pred_emb.shape != target_emb.shape:
        raise ValueError(f"pred_emb and target_emb must have same shape, got {pred_emb.shape} vs {target_emb.shape}")
    if temperature <= 0:
        raise ValueError("contrastive temperature must be > 0")

    B, T, _ = pred_emb.shape
    if B < 2 or T < 2:
        return torch.tensor(0.0, device=pred_emb.device, dtype=pred_emb.dtype)
    anchor = pred_emb[:, :-1, :]    # [B, T-1, D]
    key = target_emb[:, 1:, :]      # [B, T-1, D]
    valid = (~key_padding_mask[:, :-1]) & (~key_padding_mask[:, 1:])  # [B, T-1]
    return _batch_row_infonce(anchor, key, valid, temperature)


def _batch_row_infonce(
    anchor: torch.Tensor,
    key: torch.Tensor,
    valid: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    if valid.sum() == 0:
        return torch.tensor(0.0, device=anchor.device, dtype=anchor.dtype)

    B, Tm1, D = anchor.shape
    K = B * Tm1
    anchor_flat = anchor.reshape(K, D)
    key_flat = key.reshape(K, D)
    valid_flat = valid.reshape(K)
    row_ids = torch.arange(B, device=anchor.device).view(B, 1).expand(B, Tm1).reshape(K)

    sel = torch.nonzero(valid_flat, as_tuple=False).squeeze(1)
    anchor_sel = F.normalize(anchor_flat[sel], dim=-1)
    key_norm = F.normalize(key_flat, dim=-1)
    logits = torch.matmul(anchor_sel, key_norm.transpose(0, 1)) / temperature  # [N, K]

    pos_idx = sel
    anchor_rows = row_ids[sel]
    # Keep only valid keys from other rows; always include own positive key.
    allowed = valid_flat.unsqueeze(0) & (row_ids.unsqueeze(0) != anchor_rows.unsqueeze(1))
    allowed.scatter_(1, pos_idx.unsqueeze(1), True)
    logits = logits.masked_fill(~allowed, -1e9)

    # Future improvement: add negatives from virtual rollouts (model-generated embeddings).
    # Future improvement: add hard negatives from the same episode but temporally distant steps.
    return F.cross_entropy(logits, pos_idx)
