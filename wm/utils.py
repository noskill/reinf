#!/usr/bin/env python3
"""Shared utility functions for world-model training and evaluation."""

import random
from typing import List, Optional, Tuple

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


def batch_row_k_step_contrastive_pair_stats(
    pred_steps: List[torch.Tensor],
    target_emb: Optional[torch.Tensor],
    key_padding_mask: torch.Tensor,
    temperature: float,
    horizon_discount: float = 1.0,
    max_negatives: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return `(loss, acc)` for multi-horizon row-wise InfoNCE."""
    if target_emb is None:
        zero = torch.tensor(0.0, device=key_padding_mask.device)
        return zero, zero
    if target_emb.ndim != 3:
        raise ValueError(f"Expected target_emb [B, T, D], got {tuple(target_emb.shape)}")
    if temperature <= 0:
        raise ValueError("contrastive temperature must be > 0")
    if horizon_discount <= 0:
        raise ValueError("horizon_discount must be > 0")
    if max_negatives < 0:
        raise ValueError("max_negatives must be >= 0")
    if not pred_steps:
        zero = torch.tensor(0.0, device=target_emb.device, dtype=target_emb.dtype)
        return zero, zero

    B, T, _ = target_emb.shape
    losses: List[torch.Tensor] = []
    accs: List[torch.Tensor] = []
    weights: List[torch.Tensor] = []
    K = len(pred_steps)
    if K == 0:
        zero = torch.tensor(0.0, device=target_emb.device, dtype=target_emb.dtype)
        return zero, zero
    norm_denom = sum(float(horizon_discount) ** t for t in range(K))
    for t, pred_h in enumerate(pred_steps):
        h = t + 1
        if pred_h is None or pred_h.ndim != 3:
            continue
        if pred_h.shape[0] != B:
            raise ValueError(f"pred_steps[{h-1}] batch mismatch: {pred_h.shape[0]} vs {B}")
        Th = int(pred_h.shape[1])
        if Th <= 0 or h >= T:
            continue
        max_th = T - h
        if Th > max_th:
            pred_h = pred_h[:, :max_th, :]
            Th = max_th
        key_h = target_emb[:, h : h + Th, :]
        if key_h.shape[-1] != pred_h.shape[-1]:
            raise ValueError(
                f"pred/target dim mismatch at horizon {h}: {pred_h.shape[-1]} vs {key_h.shape[-1]}"
            )
        valid = (~key_padding_mask[:, :Th]) & (~key_padding_mask[:, h : h + Th])
        loss_h, acc_h = _batch_row_infonce(
            pred_h,
            key_h,
            valid,
            temperature,
            max_negatives=max_negatives,
        )
        losses.append(loss_h)
        accs.append(acc_h)
        weights.append(pred_h.new_tensor((float(horizon_discount) ** t) / norm_denom))

    if not losses:
        zero = torch.tensor(0.0, device=target_emb.device, dtype=target_emb.dtype)
        return zero, zero
    loss_t = torch.stack(losses)
    acc_t = torch.stack(accs)
    weight_t = torch.stack(weights)
    weighted_loss = (loss_t * weight_t).sum()
    weighted_acc = (acc_t * weight_t).sum() / weight_t.sum()
    return weighted_loss, weighted_acc


def _batch_row_infonce(
    anchor: torch.Tensor,
    key: torch.Tensor,
    valid: torch.Tensor,
    temperature: float,
    max_negatives: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if valid.sum() == 0:
        zero = torch.tensor(0.0, device=anchor.device, dtype=anchor.dtype)
        return zero, zero

    if max_negatives < 0:
        raise ValueError("max_negatives must be >= 0")

    B, Tm1, D = anchor.shape
    K = B * Tm1
    anchor_flat = anchor.reshape(K, D)
    key_flat = key.reshape(K, D)
    valid_flat = valid.reshape(K)

    sel = torch.nonzero(valid_flat, as_tuple=False).squeeze(1)
    anchor_sel = F.normalize(anchor_flat[sel], dim=-1)
    key_norm = F.normalize(key_flat, dim=-1)
    pos_idx = sel

    # Positive index for each selected anchor is the aligned flattened timestep.
    # This is equivalent to taking diagonal similarities in Twister-style code
    # after flattening anchor/key matrices to [B', D] and computing [B', B'].
    if max_negatives <= 0:
        logits = torch.matmul(anchor_sel, key_norm.transpose(0, 1)) / temperature  # [N, K]
        # Full-matrix negatives: all valid keys (including same-row keys),
        # while keeping the aligned positive at pos_idx.
        allowed = valid_flat.unsqueeze(0).expand(pos_idx.shape[0], -1).clone()
        allowed.scatter_(1, pos_idx.unsqueeze(1), True)
        logits = logits.masked_fill(~allowed, -1e9)
        loss = F.cross_entropy(logits, pos_idx)
        acc = (torch.argmax(logits, dim=-1) == pos_idx).to(logits.dtype).mean()
        return loss, acc

    # Sampled-negatives path: keep positive + up to max_negatives valid keys.
    # This reduces O(N*K) similarity compute to O(N*max_negatives).
    valid_idx = torch.nonzero(valid_flat, as_tuple=False).squeeze(1)  # [V]
    V = int(valid_idx.numel())
    N = int(pos_idx.numel())
    if V <= 1:
        zero = torch.tensor(0.0, device=anchor.device, dtype=anchor.dtype)
        return zero, zero
    M = min(int(max_negatives), V - 1)
    if M <= 0:
        zero = torch.tensor(0.0, device=anchor.device, dtype=anchor.dtype)
        return zero, zero

    rank_map = torch.full((K,), -1, dtype=torch.long, device=anchor.device)
    rank_map[valid_idx] = torch.arange(V, device=anchor.device)
    pos_rank = rank_map[pos_idx]  # [N]
    # Sample from [0, V-2], then skip each anchor's positive rank.
    rnd = torch.randint(0, V - 1, (N, M), device=anchor.device)
    neg_rank = rnd + (rnd >= pos_rank.unsqueeze(1)).to(torch.long)
    neg_idx = valid_idx[neg_rank]  # [N, M]

    pos_key = key_norm[pos_idx]  # [N, D]
    neg_key = key_norm[neg_idx]  # [N, M, D]
    logits_pos = (anchor_sel * pos_key).sum(dim=-1, keepdim=True) / temperature
    logits_neg = torch.einsum("nd,nmd->nm", anchor_sel, neg_key) / temperature
    logits = torch.cat([logits_pos, logits_neg], dim=1)  # [N, 1+M]
    labels = torch.zeros((N,), dtype=torch.long, device=anchor.device)

    # Future improvement: add negatives from virtual rollouts (model-generated embeddings).
    # Future improvement: add hard negatives from the same episode but temporally distant steps.
    loss = F.cross_entropy(logits, labels)
    # Twister-style contrastive accuracy: matched positive is argmax in row.
    acc = (torch.argmax(logits, dim=-1) == labels).to(logits.dtype).mean()
    return loss, acc
