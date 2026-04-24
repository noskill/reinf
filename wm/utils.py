#!/usr/bin/env python3
"""Shared utility functions for world-model training and evaluation."""

import random
from typing import Dict, List, Optional, Tuple

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
) -> Dict[str, torch.Tensor]:
    """Return multi-horizon row-wise InfoNCE stats (scalar and per-step)."""
    B, T = key_padding_mask.shape
    K = len(pred_steps)
    device = key_padding_mask.device
    default_dtype = target_emb.dtype if target_emb is not None else torch.float32

    def _zero_stats(*, num_horizons: int) -> Dict[str, torch.Tensor]:
        return {
            "loss": torch.tensor(0.0, device=device, dtype=default_dtype),
            "acc": torch.tensor(0.0, device=device, dtype=default_dtype),
            "per_step_loss": torch.zeros((B, T), device=device, dtype=default_dtype),
            "per_step_acc": torch.zeros((B, T), device=device, dtype=default_dtype),
            "per_step_weight": torch.zeros((B, T), device=device, dtype=default_dtype),
            "per_step_valid": torch.zeros((B, T), device=device, dtype=torch.bool),
            "per_horizon_loss": torch.zeros((num_horizons,), device=device, dtype=default_dtype),
            "per_horizon_acc": torch.zeros((num_horizons,), device=device, dtype=default_dtype),
            "per_horizon_valid": torch.zeros((num_horizons,), device=device, dtype=torch.bool),
        }

    if target_emb is None:
        return _zero_stats(num_horizons=K)
    if target_emb.ndim != 3:
        raise ValueError(f"Expected target_emb [B, T, D], got {tuple(target_emb.shape)}")
    if temperature <= 0:
        raise ValueError("contrastive temperature must be > 0")
    if horizon_discount <= 0:
        raise ValueError("horizon_discount must be > 0")
    if not pred_steps:
        return _zero_stats(num_horizons=0)

    B, T, _ = target_emb.shape
    losses: List[torch.Tensor] = []
    accs: List[torch.Tensor] = []
    weights: List[torch.Tensor] = []
    stats = _zero_stats(num_horizons=K)
    per_step_loss_weighted = torch.zeros((B, T), device=target_emb.device, dtype=target_emb.dtype)
    per_step_acc_weighted = torch.zeros((B, T), device=target_emb.device, dtype=target_emb.dtype)
    per_step_weight = torch.zeros((B, T), device=target_emb.device, dtype=target_emb.dtype)
    norm_denom = sum(float(horizon_discount) ** t for t in range(K))
    
    for t, pred_h in enumerate(pred_steps):
        assert pred_h.shape[1] == T - 1 - t
        offset = t + 1
        if pred_h is None or pred_h.ndim != 3:
            continue
        if pred_h.shape[0] != B:
            raise ValueError(f"pred_steps[{offset-1}] batch mismatch: {pred_h.shape[0]} vs {B}")
        L_pred_h = pred_h.shape[1]
        key_h = target_emb[:, offset:, :]
        if key_h.shape[-1] != pred_h.shape[-1]:
            raise ValueError(
                f"pred/target dim mismatch at horizon {offset}: {pred_h.shape[-1]} vs {key_h.shape[-1]}"
            )
        
        # valid[t] = not pad[t] and not pad[t + offset]
        valid = (~key_padding_mask[:, :L_pred_h]) & (~key_padding_mask[:, offset:])
        window_size = 128
        if L_pred_h <= window_size:
            stats_h = _batch_row_infonce(
                pred_h,
                key_h,
                valid,
                temperature,
            )
        else:
            row_loss_h = torch.zeros((B, L_pred_h), device=target_emb.device, dtype=target_emb.dtype)
            row_acc_h = torch.zeros((B, L_pred_h), device=target_emb.device, dtype=target_emb.dtype)
            row_valid_h = torch.zeros((B, L_pred_h), device=target_emb.device, dtype=torch.bool)
            loss_sum_h = target_emb.new_tensor(0.0)
            acc_sum_h = target_emb.new_tensor(0.0)
            num_rows_h = 0
            for w_start in range(0, L_pred_h, window_size):
                w_end = min(w_start + window_size, L_pred_h)
                stats_w = _batch_row_infonce(
                    pred_h[:, w_start:w_end, :],
                    key_h[:, w_start:w_end, :],
                    valid[:, w_start:w_end],
                    temperature,
                )
                row_loss_h[:, w_start:w_end] = stats_w["row_loss"]
                row_acc_h[:, w_start:w_end] = stats_w["row_acc"]
                row_valid_h[:, w_start:w_end] = stats_w["row_valid"]
                rows_w = int(stats_w["num_rows"].item())
                if rows_w > 0:
                    loss_sum_h = loss_sum_h + stats_w["loss"] * rows_w
                    acc_sum_h = acc_sum_h + stats_w["acc"] * rows_w
                    num_rows_h += rows_w
            if num_rows_h > 0:
                stats_h = {
                    "loss": loss_sum_h / num_rows_h,
                    "acc": acc_sum_h / num_rows_h,
                    "row_loss": row_loss_h,
                    "row_acc": row_acc_h,
                    "row_valid": row_valid_h,
                    "num_rows": torch.tensor(num_rows_h, device=target_emb.device, dtype=torch.long),
                }
            else:
                stats_h = {
                    "loss": target_emb.new_tensor(0.0),
                    "acc": target_emb.new_tensor(0.0),
                    "row_loss": row_loss_h,
                    "row_acc": row_acc_h,
                    "row_valid": row_valid_h,
                    "num_rows": torch.tensor(0, device=target_emb.device, dtype=torch.long),
                }
        weight_t = pred_h.new_tensor((float(horizon_discount) ** t) / norm_denom)
        stats["per_horizon_loss"][t] = stats_h["loss"]
        stats["per_horizon_acc"][t] = stats_h["acc"]
        if int(stats_h["num_rows"].item()) > 0:
            stats["per_horizon_valid"][t] = True
            losses.append(stats_h["loss"])
            accs.append(stats_h["acc"])
            weights.append(weight_t)

            row_valid = stats_h["row_valid"]
            row_loss = stats_h["row_loss"]
            row_acc = stats_h["row_acc"]
            valid_weight = row_valid.to(row_loss.dtype)
            per_step_loss_weighted[:, :L_pred_h] = per_step_loss_weighted[:, :L_pred_h] + weight_t * row_loss
            per_step_acc_weighted[:, :L_pred_h] = per_step_acc_weighted[:, :L_pred_h] + weight_t * row_acc
            per_step_weight[:, :L_pred_h] = per_step_weight[:, :L_pred_h] + weight_t * valid_weight

    stats["per_step_weight"] = per_step_weight
    stats["per_step_valid"] = per_step_weight > 0
    valid_steps = stats["per_step_valid"]
    if valid_steps.any():
        stats["per_step_loss"][valid_steps] = per_step_loss_weighted[valid_steps] / per_step_weight[valid_steps]
        stats["per_step_acc"][valid_steps] = per_step_acc_weighted[valid_steps] / per_step_weight[valid_steps]

    if not losses:
        return stats
    loss_t = torch.stack(losses)
    acc_t = torch.stack(accs)
    weight_t = torch.stack(weights)
    weighted_loss = (loss_t * weight_t).sum()
    weighted_acc = (acc_t * weight_t).sum() / weight_t.sum()
    stats["loss"] = weighted_loss
    stats["acc"] = weighted_acc
    return stats


def _batch_row_infonce(
    anchor: torch.Tensor,
    key: torch.Tensor,
    valid: torch.Tensor,
    temperature: float,
) -> Dict[str, torch.Tensor]:
    B, Tm1, _ = anchor.shape
    row_loss = torch.zeros((B, Tm1), device=anchor.device, dtype=anchor.dtype)
    row_acc = torch.zeros((B, Tm1), device=anchor.device, dtype=anchor.dtype)
    row_valid = valid.to(torch.bool)

    if row_valid.sum() == 0:
        return {
            "loss": torch.tensor(0.0, device=anchor.device, dtype=anchor.dtype),
            "acc": torch.tensor(0.0, device=anchor.device, dtype=anchor.dtype),
            "row_loss": row_loss,
            "row_acc": row_acc,
            "row_valid": torch.zeros_like(row_valid),
            "num_rows": torch.tensor(0, device=anchor.device, dtype=torch.long),
        }

    B, Tm1, D = anchor.shape
    K = B * Tm1
    anchor_flat = anchor.reshape(K, D)
    key_flat = key.reshape(K, D)
    valid_flat = row_valid.reshape(K)

    sel = torch.nonzero(valid_flat, as_tuple=False).squeeze(1)
    anchor_sel = F.normalize(anchor_flat[sel], dim=-1)
    key_norm = F.normalize(key_flat, dim=-1)
    pos_idx = sel
    # Positive index for each selected anchor is the aligned flattened timestep.
    # This is equivalent to taking diagonal similarities in Twister-style code
    # after flattening anchor/key matrices to [B', D] and computing [B', B'].
    logits = torch.matmul(anchor_sel, key_norm.transpose(0, 1)) / temperature  # [N, K]
    # Full-matrix negatives: all valid keys (including same-row keys),
    # while keeping the aligned positive at pos_idx.
    allowed = valid_flat.unsqueeze(0).expand(pos_idx.shape[0], -1).clone()
    allowed.scatter_(1, pos_idx.unsqueeze(1), True)
    logits = logits.masked_fill(~allowed, -1e9)
    per_row_loss = F.cross_entropy(logits, pos_idx, reduction="none")
    per_row_acc = (torch.argmax(logits, dim=-1) == pos_idx).to(logits.dtype)
    row_loss.view(-1)[sel] = per_row_loss
    row_acc.view(-1)[sel] = per_row_acc
    return {
        "loss": per_row_loss.mean(),
        "acc": per_row_acc.mean(),
        "row_loss": row_loss,
        "row_acc": row_acc,
        "row_valid": row_valid,
        "num_rows": torch.tensor(int(sel.numel()), device=anchor.device, dtype=torch.long),
    }
