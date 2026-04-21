#!/usr/bin/env python3
"""Train a causal transformer to predict next observations from history of observations/actions.

Data format: data.txt contains episodes prefixed by lines starting with "random ".
Each episode line contains semicolon-separated dicts with keys:
  location: [x, y]
  heading: 'u'|'d'|'l'|'r'|'up'
  sensor: [s0, s1, s2]
  action: (turn_deg, steps)
"""

import argparse
import os
import random
import sys
from dataclasses import asdict
from typing import List

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from base import DiscreteLatentPredictorBase, LossConfig
from baselines import TransformerBaseline
from data import HEADING_CANON, WindowedDataset, build_sequences, collate_batch, compute_stats, load_episodes
from log import Logger
from trainer import run_epoch_joint
from utils import (
    compute_baseline_lr,
    make_label_smoothing_table,
    make_soft_table,
    set_seed,
)
from agent_utils_wm import add_create_model_args, create_model, extract_create_model_args


def configure_recon_head_only_training(model: nn.Module) -> List[str]:
    """Freeze model and enable training only of reconstruction heads ([z,h], z, h)."""
    if not isinstance(model, DiscreteLatentPredictorBase):
        raise ValueError("--train-recon-heads-only requires model-type rssm-discrete or tssm")
    if model.sensor_mode != "categorical":
        raise ValueError("--train-recon-heads-only currently supports --sensor-mode categorical only")
    if model.obs_loss_mode != "soft":
        raise ValueError("--train-recon-heads-only requires --obs-loss-mode soft")

    for p in model.parameters():
        p.requires_grad = False

    enabled: List[str] = []
    for module_name in ("obs_head", "z_obs_head", "h_obs_head"):
        module = getattr(model, module_name, None)
        if module is None:
            continue
        for pname, p in module.named_parameters():
            p.requires_grad = True
            enabled.append(f"{module_name}.{pname}")

    if not enabled:
        raise RuntimeError("No reconstruction heads were found to train.")
    return enabled


def configure_state_probe_only_training(model: nn.Module) -> List[str]:
    """Freeze model and enable only location/heading probe heads."""
    if not isinstance(model, (DiscreteLatentPredictorBase, TransformerBaseline)):
        raise ValueError("--train-state-probes-only requires model-type transformer/rnn/rssm-discrete/tssm")

    for p in model.parameters():
        p.requires_grad = False

    enabled: List[str] = []
    if isinstance(model, DiscreteLatentPredictorBase):
        module_names = ("loc_probe_x", "loc_probe_y", "head_probe")
    else:
        module_names = ("loc_x_head", "loc_y_head", "heading_head")
    for module_name in module_names:
        module = getattr(model, module_name, None)
        if module is None:
            continue
        for pname, p in module.named_parameters():
            p.requires_grad = True
            enabled.append(f"{module_name}.{pname}")

    if not enabled:
        raise RuntimeError("No state probes were found to train.")
    return enabled
def main():
    parser = argparse.ArgumentParser(description="Train transformer to predict next observations.")
    parser.add_argument("--data", type=str, default="data.txt", help="Path to data.txt")
    parser.add_argument("--load-path", type=str, default=None, help="Optional checkpoint to warm start from.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01, help="AdamW weight decay.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    add_create_model_args(parser, include_load_path=False)
    parser.add_argument(
        "--contrastive-weight",
        type=float,
        default=0.0,
        help="Weight for next-step embedding contrastive (InfoNCE) loss.",
    )
    parser.add_argument(
        "--contrastive-temp",
        type=float,
        default=0.1,
        help="Temperature for next-step contrastive loss.",
    )
    parser.add_argument(
        "--contrastive-horizon-discount",
        type=float,
        default=0.75,
        help="Exponential horizon weighting for CPC (horizon h uses weight discount**h).",
    )
    parser.add_argument(
        "--contrastive-negatives",
        type=int,
        default=0,
        help="Number of sampled negatives per anchor for CPC (0 uses all valid negatives).",
    )
    parser.add_argument(
        "--train-recon-heads-only",
        action="store_true",
        help="Freeze backbone/dynamics and train only sensor reconstruction heads from [z,h], z-only, and h-only.",
    )
    parser.add_argument(
        "--train-state-probes-only",
        action="store_true",
        help="Freeze backbone/dynamics and train only location/heading probe heads.",
    )
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument(
        "--context-len",
        type=int,
        default=0,
        help="Input/target sequence length per sample; 0 uses full episode (t->t+1 over full trajectory).",
    )
    parser.add_argument("--save-path", type=str, default="outputs/wm_next_obs.pt")
    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="TensorBoard log directory (default: logs/wm_train/<save_path_stem>).",
    )
    parser.add_argument("--sensor-weight", type=float, default=1.0)
    parser.add_argument("--loc-weight", type=float, default=1.0)
    parser.add_argument("--head-weight", type=float, default=1.0)
    parser.add_argument("--turn-weight", type=float, default=1.0)
    parser.add_argument("--step-weight", type=float, default=1.0)
    parser.add_argument("--sensor-sigma", type=float, default=1.0, help="Soft target sigma for sensor bins")
    parser.add_argument("--pos-sigma", type=float, default=1.0, help="Soft target sigma for location bins")
    parser.add_argument("--heading-smoothing", type=float, default=0.0, help="Label smoothing for heading targets")
    parser.add_argument(
        "--baseline",
        type=str,
        choices=["none", "repeat"],
        default="none",
        help="Baseline for left/right sensors. 'repeat' predicts next sensors as current sensors.",
    )
    args = parser.parse_args()
    if args.context_len < 0:
        raise ValueError("--context-len must be >= 0")
    if args.probe_hidden_dim < 0:
        raise ValueError("--probe-hidden-dim must be >= 0")
    if args.probe_layers < 1:
        raise ValueError("--probe-layers must be >= 1")
    if args.contrastive_dim < 0:
        raise ValueError("--contrastive-dim must be >= 0")
    if args.contrastive_temp <= 0:
        raise ValueError("--contrastive-temp must be > 0")
    if args.contrastive_horizon_discount <= 0:
        raise ValueError("--contrastive-horizon-discount must be > 0")
    if args.contrastive_weight > 0 and args.contrastive_dim <= 0:
        raise ValueError("--contrastive-dim must be > 0 when --contrastive-weight > 0")
    if args.contrastive_steps < 1:
        raise ValueError("--contrastive-steps must be >= 1")
    if args.contrastive_negatives < 0:
        raise ValueError("--contrastive-negatives must be >= 0")
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    set_seed(args.seed)
    save_stem = os.path.splitext(os.path.basename(args.save_path))[0]
    log_dir = args.log_dir or os.path.join("logs", "wm_train", save_stem)
    logger = Logger(log_dir=log_dir)

    episodes = load_episodes(args.data, max_episodes=args.max_episodes)
    if not episodes:
        raise RuntimeError(f"No episodes found in {args.data}")

    stats = compute_stats(episodes)
    sequences = build_sequences(episodes, stats, args.sensor_mode)

    heading_dim = len(stats.heading_to_idx)
    if heading_dim < 2:
        raise RuntimeError("Not enough heading categories to train classifier.")

    dataset = WindowedDataset(
        sequences,
        heading_dim=heading_dim,
        context_len=args.context_len,
    )
    if len(dataset) == 0:
        raise RuntimeError("Dataset is empty.")

    # train/val split by full episodes
    episode_ids = list(range(len(sequences)))
    random.shuffle(episode_ids)
    val_ep_count = max(1, int(len(episode_ids) * args.val_split))
    val_eps = set(episode_ids[:val_ep_count])
    train_indices = []
    val_indices = []
    for i in range(len(dataset)):
        if dataset.episode_id(i) in val_eps:
            val_indices.append(i)
        else:
            train_indices.append(i)

    train_subset = torch.utils.data.Subset(dataset, train_indices)
    val_subset = torch.utils.data.Subset(dataset, val_indices)

    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)
    print(
        f"run config | model={args.model_type} train_batches={len(train_loader)} val_batches={len(val_loader)} "
        f"batch_size={args.batch_size} epochs={args.epochs} attention_window={args.attention_window} "
        f"context_len={args.context_len} probe_hidden_dim={args.probe_hidden_dim} probe_layers={args.probe_layers} "
        f"rnn_state_norm={args.rnn_state_norm} "
        f"rssm_transition={args.rssm_transition} rssm_residual_scale={args.rssm_residual_scale} "
        f"rssm_state_norm={args.rssm_state_norm} "
        f"contrastive_w={args.contrastive_weight} contrastive_dim={args.contrastive_dim} "
        f"contrastive_temp={args.contrastive_temp} contrastive_steps={args.contrastive_steps} "
        f"contrastive_discount={args.contrastive_horizon_discount} contrastive_negs={args.contrastive_negatives} "
        f"attn_dropout={args.attention_dropout} weight_decay={args.weight_decay} "
        f"prior_roll_w={args.prior_rollout_weight} z_only_w={args.z_only_weight} "
        f"h_only_w={args.h_only_weight} "
        f"bptt_horizon={args.bptt_horizon} prior_roll_steps={args.prior_rollout_steps} "
        f"recon_heads_only={args.train_recon_heads_only}",
        flush=True,
    )
    logger.log_scalar("config/train_batches", float(len(train_loader)), step=0)
    logger.log_scalar("config/val_batches", float(len(val_loader)), step=0)
    logger.log_scalar("config/batch_size", float(args.batch_size), step=0)
    logger.log_scalar("config/epochs", float(args.epochs), step=0)
    logger.log_scalar("config/lr", float(args.lr), step=0)
    logger.log_scalar("config/weight_decay", float(args.weight_decay), step=0)

    input_obs_dim = sequences[0].obs_cont.shape[-1]
    input_dim = input_obs_dim + sequences[0].action_cont.shape[-1]
    obs_dim = input_obs_dim
    sensor_dim = 3
    sensor_min_idx = torch.tensor(stats.sensor_min, dtype=torch.long, device=args.device)
    loc_min = torch.tensor(stats.loc_min, dtype=torch.float32, device=args.device)
    loc_max = torch.tensor(stats.loc_max, dtype=torch.float32, device=args.device)
    loc_x_bins = int(stats.loc_max[0] - stats.loc_min[0] + 1)
    loc_y_bins = int(stats.loc_max[1] - stats.loc_min[1] + 1)
    sensor_bins = (stats.sensor_max - stats.sensor_min + 1).astype(np.int64)
    turn_bins = len(stats.turn_to_idx)
    step_bins = len(stats.step_to_idx)
    action_dim = int(sequences[0].action_cont.shape[-1])
    model_args = extract_create_model_args(args, device=args.device)
    active_attention_window = (
        model_args.attention_window
        if (model_args.attention_window is not None and model_args.attention_window > 0)
        else None
    )
    model_config_extra = {}
    model = create_model(
        model_args,
        input_dim=input_dim,
        sensor_dim=sensor_dim,
        sensor_bins=sensor_bins,
        loc_x_bins=loc_x_bins,
        loc_y_bins=loc_y_bins,
        heading_dim=heading_dim,
        turn_bins=turn_bins,
        step_bins=step_bins,
        obs_dim=obs_dim,
        action_dim=action_dim,
        active_attention_window=active_attention_window,
        model_config_extra=model_config_extra,
    )
    if args.train_recon_heads_only and args.train_state_probes_only:
        raise ValueError("--train-recon-heads-only and --train-state-probes-only are mutually exclusive")
    if args.train_recon_heads_only:
        enabled = configure_recon_head_only_training(model)
        print(f"train-recon-heads-only enabled: {enabled}", flush=True)
    if args.train_state_probes_only:
        enabled = configure_state_probe_only_training(model)
        print(f"train-state-probes-only enabled: {enabled}", flush=True)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters selected.")
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    if args.baseline == "repeat":
        train_bl_rmse, train_bl_acc = compute_baseline_lr(train_loader, args.device)
        val_bl_rmse, val_bl_acc = compute_baseline_lr(val_loader, args.device)
        print(
            f"baseline repeat | train lr_rmse {train_bl_rmse:.3f} lr_acc {train_bl_acc:.3f} | "
            f"val lr_rmse {val_bl_rmse:.3f} lr_acc {val_bl_acc:.3f}"
            ,
            flush=True,
        )
        logger.log_scalar("baseline/train_lr_rmse", float(train_bl_rmse), step=0)
        logger.log_scalar("baseline/train_lr_acc", float(train_bl_acc), step=0)
        logger.log_scalar("baseline/val_lr_rmse", float(val_bl_rmse), step=0)
        logger.log_scalar("baseline/val_lr_acc", float(val_bl_acc), step=0)

    loc_x_table = make_soft_table(loc_x_bins, args.pos_sigma, args.device)
    loc_y_table = make_soft_table(loc_y_bins, args.pos_sigma, args.device)
    heading_table = make_label_smoothing_table(heading_dim, args.heading_smoothing, args.device)
    sensor_tables = None
    if args.sensor_mode == "categorical":
        sensor_tables = [
            make_soft_table(int(sensor_bins[0]), args.sensor_sigma, args.device),
            make_soft_table(int(sensor_bins[1]), args.sensor_sigma, args.device),
            make_soft_table(int(sensor_bins[2]), args.sensor_sigma, args.device),
        ]

    model.config = LossConfig(
        sensor_tables=sensor_tables,
        sensor_min_idx=sensor_min_idx,
        loc_x_table=loc_x_table,
        loc_y_table=loc_y_table,
        heading_table=heading_table,
        sensor_weight=args.sensor_weight,
        loc_weight=args.loc_weight,
        head_weight=args.head_weight,
        turn_weight=args.turn_weight,
        step_weight=args.step_weight,
        contrastive_weight=args.contrastive_weight,
        contrastive_temp=args.contrastive_temp,
        contrastive_horizon_discount=args.contrastive_horizon_discount,
        contrastive_negatives=args.contrastive_negatives,
        loc_min=loc_min,
    )
    for epoch in range(1, args.epochs + 1):
        print(f"starting epoch {epoch:03d}", flush=True)

        run_epoch_joint(
            model,
            val_loader,
            optimizer=None,
            device=args.device,
            logger=logger,
            metric_prefix="val",
            step=epoch,
        )

        run_epoch_joint(
            model,
            train_loader,
            optimizer,
            device=args.device,
            logger=logger,
            metric_prefix="train",
            step=epoch,
        )
        logger.log_scalar("train/lr", float(optimizer.param_groups[0]["lr"]), step=epoch)
        print(f"finished epoch {epoch:03d}", flush=True)

    save_dir = os.path.dirname(args.save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    torch.save(
        {
            "model_state": model.state_dict(),
            "stats": asdict(stats),
            "config": {
                "input_dim": input_dim,
                "sensor_dim": sensor_dim,
                "loc_x_bins": loc_x_bins,
                "loc_y_bins": loc_y_bins,
                "heading_dim": heading_dim,
                "obs_dim": obs_dim,
                "obs_latent_dim": args.obs_latent_dim,
                "turn_bins": turn_bins,
                "step_bins": step_bins,
                "model_type": args.model_type,
                "model_config": model_config_extra,
                "context_len": args.context_len,
                "probe_hidden_dim": args.probe_hidden_dim,
                "probe_layers": args.probe_layers,
                "rnn_state_norm": args.rnn_state_norm,
                "rssm_transition": args.rssm_transition,
                "rssm_residual_scale": args.rssm_residual_scale,
                "rssm_state_norm": args.rssm_state_norm,
                "contrastive_weight": args.contrastive_weight,
                "contrastive_dim": args.contrastive_dim,
                "contrastive_temp": args.contrastive_temp,
                "contrastive_steps": args.contrastive_steps,
                "contrastive_horizon_discount": args.contrastive_horizon_discount,
                "contrastive_negatives": args.contrastive_negatives,
                "pos_sigma": args.pos_sigma,
                "heading_smoothing": args.heading_smoothing,
            },
        },
        args.save_path,
    )
    print(f"saved checkpoint to {args.save_path}", flush=True)
    logger.log_scalar("run/saved", 1.0, step=args.epochs)


if __name__ == "__main__":
    main()
