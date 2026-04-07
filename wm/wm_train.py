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
from baselines import RNNPredictor, UnifiedPredictor
from data import HEADING_CANON, WindowedDataset, build_sequences, collate_batch, compute_stats, load_episodes
from rssm import RSSMDiscretePredictor
from trainer import run_epoch_joint
from transformer import LlamaConfig
from tssm import TSSMDiscretePredictor
from utils import (
    compute_baseline_lr,
    make_label_smoothing_table,
    make_soft_table,
    set_seed,
)
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
    if not isinstance(model, (DiscreteLatentPredictorBase, UnifiedPredictor)):
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
    parser.add_argument("--hidden-size", type=int, default=176)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=44)
    parser.add_argument("--intermediate", type=int, default=704)
    parser.add_argument("--attention-window", type=int, default=None)
    parser.add_argument(
        "--attention-dropout",
        type=float,
        default=0.0,
        help="Attention dropout probability for transformer model.",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["transformer", "rnn", "rssm-discrete", "tssm"],
        default="transformer",
        help="Model architecture to train.",
    )
    parser.add_argument(
        "--rnn-transition",
        type=str,
        choices=["gru", "residual", "residual_mlp"],
        default="gru",
        help="RNN deterministic transition: GRU, residual add, or residual add + LLaMA MLP block.",
    )
    parser.add_argument(
        "--rnn-residual-scale",
        type=float,
        default=1.0,
        help="Scale a for residual RNN update when --rnn-transition is residual or residual_mlp: h_t=h_{t-1}+a*g_t.",
    )
    parser.add_argument(
        "--rnn-state-norm",
        type=str,
        choices=["none", "layernorm", "rmsnorm"],
        default="none",
        help="Normalization on RNN hidden sequence before prediction heads.",
    )
    parser.add_argument(
        "--rssm-transition",
        type=str,
        choices=["gru", "residual"],
        default="gru",
        help="RSSM deterministic transition: GRUCell or residual update h_t=h_{t-1}+g_t.",
    )
    parser.add_argument(
        "--rssm-residual-scale",
        type=float,
        default=1.0,
        help="Scale for RSSM residual update when --rssm-transition=residual.",
    )
    parser.add_argument(
        "--rssm-state-norm",
        type=str,
        choices=["none", "layernorm", "rmsnorm"],
        default="none",
        help="Pre-normalization on RSSM h_{t-1} before transition step.",
    )
    parser.add_argument("--stoch-size", type=int, default=32, help="Number of categorical latent groups.")
    parser.add_argument("--stoch-classes", type=int, default=32, help="Number of classes per latent group.")
    parser.add_argument("--stoch-temp", type=float, default=1.0, help="Gumbel-softmax temperature.")
    parser.add_argument("--kl-dyn-beta", type=float, default=1.0, help="Weight for dynamics KL term.")
    parser.add_argument("--kl-rep-beta", type=float, default=0.1, help="Weight for representation KL term.")
    parser.add_argument("--kl-free-nats", type=float, default=1.0, help="Free nats clamp for both KL terms.")
    parser.add_argument(
        "--prior-rollout-weight",
        type=float,
        default=0.0,
        help="Auxiliary weight for full-sequence open-loop prior rollout sensor prediction.",
    )
    parser.add_argument(
        "--z-only-weight",
        type=float,
        default=0.0,
        help="Auxiliary weight for z-only sensor prediction.",
    )
    parser.add_argument(
        "--h-only-weight",
        type=float,
        default=0.0,
        help="Auxiliary weight for h-only sensor prediction.",
    )
    parser.add_argument(
        "--contrastive-weight",
        type=float,
        default=0.0,
        help="Weight for next-step embedding contrastive (InfoNCE) loss.",
    )
    parser.add_argument(
        "--contrastive-dim",
        type=int,
        default=0,
        help="Embedding dimension for contrastive head (0 disables contrastive head).",
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
        "--contrastive-steps",
        type=int,
        default=1,
        help="Number of Twister-style action-conditioned contrastive horizons (K).",
    )
    parser.add_argument(
        "--contrastive-negatives",
        type=int,
        default=0,
        help="Number of sampled negatives per anchor for CPC (0 uses all valid negatives).",
    )
    parser.add_argument(
        "--bptt-horizon",
        type=int,
        default=0,
        help="Truncated BPTT horizon for RSSM/TSSM recurrent state (0 disables truncation).",
    )
    parser.add_argument(
        "--prior-rollout-steps",
        type=int,
        default=0,
        help="Open-loop prior rollout length for prior_roll loss (0 uses full sequence).",
    )
    parser.add_argument("--recon-beta", type=float, default=1.0, help="Weight for observation reconstruction term.")
    parser.add_argument(
        "--obs-loss-mode",
        type=str,
        choices=["soft", "recon"],
        default="soft",
        help="Observation loss for RSSM/TSSM: soft-target heads or L2 reconstruction from [h_t, z_t].",
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
    parser.add_argument(
        "--probe-hidden-dim",
        type=int,
        default=256,
        help="Hidden size for probe MLPs (set 0 to keep linear probes).",
    )
    parser.add_argument(
        "--probe-layers",
        type=int,
        default=2,
        help="Number of linear layers in probe heads (1=linear probe).",
    )
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument(
        "--context-len",
        type=int,
        default=0,
        help="Input/target sequence length per sample; 0 uses full episode (t->t+1 over full trajectory).",
    )
    parser.add_argument("--save-path", type=str, default="outputs/wm_next_obs.pt")
    parser.add_argument("--sensor-weight", type=float, default=1.0)
    parser.add_argument("--loc-weight", type=float, default=1.0)
    parser.add_argument("--head-weight", type=float, default=1.0)
    parser.add_argument("--turn-weight", type=float, default=1.0)
    parser.add_argument("--step-weight", type=float, default=1.0)
    parser.add_argument("--obs-latent-dim", type=int, default=64)
    parser.add_argument(
        "--sensor-mode",
        type=str,
        choices=["raw", "categorical"],
        default="raw",
        help="Sensor target mode: raw regression or categorical",
    )
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
        f"rnn_transition={args.rnn_transition} rnn_residual_scale={args.rnn_residual_scale} "
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
    active_attention_window = args.attention_window if (args.attention_window is not None and args.attention_window > 0) else None
    cfg = None
    model_config_extra = {}
    if args.model_type == "transformer":
        if args.sensor_mode != "categorical":
            raise ValueError("model-type=transformer currently supports --sensor-mode categorical only")
        if args.hidden_size != args.heads * args.head_dim:
            raise ValueError("hidden_size must equal heads * head_dim")
        cfg = LlamaConfig(
            input_size=input_dim,
            hidden_size=args.hidden_size,
            intermediate_size=args.intermediate,
            num_hidden_layers=args.layers,
            num_attention_heads=args.heads,
            num_key_value_heads=args.heads,
            head_dim=args.head_dim,
            attention_dropout=args.attention_dropout,
            attention_window=active_attention_window,
        )
        model = UnifiedPredictor(
            cfg,
            sensor_mode=args.sensor_mode,
            sensor_dim=sensor_dim,
            sensor_bins=sensor_bins if args.sensor_mode == "categorical" else None,
            loc_x_bins=loc_x_bins,
            loc_y_bins=loc_y_bins,
            heading_dim=heading_dim,
            turn_bins=turn_bins,
            step_bins=step_bins,
            obs_dim=obs_dim,
            obs_latent_dim=args.obs_latent_dim,
            probe_hidden_dim=args.probe_hidden_dim,
            probe_layers=args.probe_layers,
            contrastive_dim=args.contrastive_dim,
            contrastive_steps=args.contrastive_steps,
        ).to(args.device)
        model_config_extra["llama"] = asdict(cfg)
        model_config_extra["probe_hidden_dim"] = args.probe_hidden_dim
        model_config_extra["probe_layers"] = args.probe_layers
        model_config_extra["contrastive_dim"] = args.contrastive_dim
        model_config_extra["contrastive_steps"] = args.contrastive_steps
    elif args.model_type == "rnn":
        if args.sensor_mode != "categorical":
            raise ValueError("model-type=rnn currently supports --sensor-mode categorical only")
        cfg = LlamaConfig(
            input_size=input_dim,
            hidden_size=args.hidden_size,
            intermediate_size=args.intermediate,
            num_hidden_layers=args.layers,
            num_attention_heads=args.heads,
            num_key_value_heads=args.heads,
            head_dim=args.head_dim,
            attention_dropout=args.attention_dropout,
            attention_window=active_attention_window,
        )
        model = RNNPredictor(
            cfg,
            sensor_mode=args.sensor_mode,
            sensor_dim=sensor_dim,
            sensor_bins=sensor_bins if args.sensor_mode == "categorical" else None,
            loc_x_bins=loc_x_bins,
            loc_y_bins=loc_y_bins,
            heading_dim=heading_dim,
            turn_bins=turn_bins,
            step_bins=step_bins,
            obs_dim=obs_dim,
            obs_latent_dim=args.obs_latent_dim,
            probe_hidden_dim=args.probe_hidden_dim,
            probe_layers=args.probe_layers,
            transition=args.rnn_transition,
            residual_scale=args.rnn_residual_scale,
            state_norm=args.rnn_state_norm,
            contrastive_dim=args.contrastive_dim,
            contrastive_steps=args.contrastive_steps,
        ).to(args.device)
        model_config_extra["rnn"] = {
            "input_size": input_dim,
            "hidden_size": args.hidden_size,
            "layers": args.layers,
            "probe_hidden_dim": args.probe_hidden_dim,
            "probe_layers": args.probe_layers,
            "transition": args.rnn_transition,
            "residual_scale": args.rnn_residual_scale,
            "state_norm": args.rnn_state_norm,
            "contrastive_dim": args.contrastive_dim,
            "contrastive_steps": args.contrastive_steps,
        }
    elif args.model_type == "rssm-discrete":
        model = RSSMDiscretePredictor(
            hidden_size=args.hidden_size,
            sensor_mode=args.sensor_mode,
            sensor_dim=sensor_dim,
            sensor_bins=sensor_bins if args.sensor_mode == "categorical" else None,
            loc_x_bins=loc_x_bins,
            loc_y_bins=loc_y_bins,
            heading_dim=heading_dim,
            turn_bins=turn_bins,
            step_bins=step_bins,
            obs_dim=obs_dim,
            obs_latent_dim=args.obs_latent_dim,
            action_dim=action_dim,
            stoch_size=args.stoch_size,
            stoch_classes=args.stoch_classes,
            stoch_temp=args.stoch_temp,
            kl_dyn_beta=args.kl_dyn_beta,
            kl_rep_beta=args.kl_rep_beta,
            kl_free_nats=args.kl_free_nats,
            prior_rollout_weight=args.prior_rollout_weight,
            bptt_horizon=args.bptt_horizon,
            z_only_weight=args.z_only_weight,
            h_only_weight=args.h_only_weight,
            prior_rollout_steps=args.prior_rollout_steps,
            probe_hidden_dim=args.probe_hidden_dim,
            probe_layers=args.probe_layers,
            contrastive_dim=args.contrastive_dim,
            contrastive_steps=args.contrastive_steps,
            transition=args.rssm_transition,
            residual_scale=args.rssm_residual_scale,
            state_norm=args.rssm_state_norm,
            recon_beta=args.recon_beta,
            obs_loss_mode=args.obs_loss_mode,
        ).to(args.device)
        model_config_extra["rssm"] = {
            "hidden_size": args.hidden_size,
            "stoch_size": args.stoch_size,
            "stoch_classes": args.stoch_classes,
            "stoch_temp": args.stoch_temp,
            "kl_dyn_beta": args.kl_dyn_beta,
            "kl_rep_beta": args.kl_rep_beta,
            "kl_free_nats": args.kl_free_nats,
            "prior_rollout_weight": args.prior_rollout_weight,
            "bptt_horizon": args.bptt_horizon,
            "prior_rollout_steps": args.prior_rollout_steps,
            "z_only_weight": args.z_only_weight,
            "h_only_weight": args.h_only_weight,
            "probe_hidden_dim": args.probe_hidden_dim,
            "probe_layers": args.probe_layers,
            "contrastive_dim": args.contrastive_dim,
            "contrastive_steps": args.contrastive_steps,
            "transition": args.rssm_transition,
            "residual_scale": args.rssm_residual_scale,
            "state_norm": args.rssm_state_norm,
            "recon_beta": args.recon_beta,
            "action_dim": action_dim,
            "obs_loss_mode": args.obs_loss_mode,
        }
    elif args.model_type == "tssm":
        model = TSSMDiscretePredictor(
            hidden_size=args.hidden_size,
            layers=args.layers,
            heads=args.heads,
            head_dim=args.head_dim,
            intermediate=args.intermediate,
            attention_window=active_attention_window,
            sensor_mode=args.sensor_mode,
            sensor_dim=sensor_dim,
            sensor_bins=sensor_bins if args.sensor_mode == "categorical" else None,
            loc_x_bins=loc_x_bins,
            loc_y_bins=loc_y_bins,
            heading_dim=heading_dim,
            turn_bins=turn_bins,
            step_bins=step_bins,
            obs_dim=obs_dim,
            obs_latent_dim=args.obs_latent_dim,
            action_dim=action_dim,
            stoch_size=args.stoch_size,
            stoch_classes=args.stoch_classes,
            stoch_temp=args.stoch_temp,
            kl_dyn_beta=args.kl_dyn_beta,
            kl_rep_beta=args.kl_rep_beta,
            kl_free_nats=args.kl_free_nats,
            prior_rollout_weight=args.prior_rollout_weight,
            bptt_horizon=args.bptt_horizon,
            z_only_weight=args.z_only_weight,
            h_only_weight=args.h_only_weight,
            prior_rollout_steps=args.prior_rollout_steps,
            probe_hidden_dim=args.probe_hidden_dim,
            probe_layers=args.probe_layers,
            contrastive_dim=args.contrastive_dim,
            contrastive_steps=args.contrastive_steps,
            recon_beta=args.recon_beta,
            obs_loss_mode=args.obs_loss_mode,
        ).to(args.device)
        model_config_extra["tssm"] = {
            "hidden_size": args.hidden_size,
            "layers": args.layers,
            "heads": args.heads,
            "head_dim": args.head_dim,
            "intermediate": args.intermediate,
            "attention_window": active_attention_window,
            "stoch_size": args.stoch_size,
            "stoch_classes": args.stoch_classes,
            "stoch_temp": args.stoch_temp,
            "kl_dyn_beta": args.kl_dyn_beta,
            "kl_rep_beta": args.kl_rep_beta,
            "kl_free_nats": args.kl_free_nats,
            "prior_rollout_weight": args.prior_rollout_weight,
            "bptt_horizon": args.bptt_horizon,
            "prior_rollout_steps": args.prior_rollout_steps,
            "z_only_weight": args.z_only_weight,
            "h_only_weight": args.h_only_weight,
            "probe_hidden_dim": args.probe_hidden_dim,
            "probe_layers": args.probe_layers,
            "contrastive_dim": args.contrastive_dim,
            "contrastive_steps": args.contrastive_steps,
            "recon_beta": args.recon_beta,
            "action_dim": action_dim,
            "obs_loss_mode": args.obs_loss_mode,
        }
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
    if args.load_path:
        ckpt = torch.load(args.load_path, map_location=args.device, weights_only=False)
        state = ckpt.get("model_state", ckpt)
        if isinstance(ckpt, dict) and "config" in ckpt:
            ckpt_type = ckpt["config"].get("model_type")
            if ckpt_type and ckpt_type != args.model_type:
                print(
                    f"warning: checkpoint model_type={ckpt_type} != args.model_type={args.model_type}",
                    flush=True,
                )
        model_state = model.state_dict()
        filtered_state = {}
        skipped_mismatch = []
        for k, v in state.items():
            if k not in model_state:
                continue
            if model_state[k].shape != v.shape:
                skipped_mismatch.append((k, tuple(v.shape), tuple(model_state[k].shape)))
                continue
            filtered_state[k] = v
        missing, unexpected = model.load_state_dict(filtered_state, strict=False)
        if skipped_mismatch:
            print(
                f"load warning: skipped {len(skipped_mismatch)} mismatched tensors (e.g. {skipped_mismatch[:3]})",
                flush=True,
            )
        if missing:
            print(f"load warning: missing keys: {missing}", flush=True)
        if unexpected:
            print(f"load warning: unexpected keys: {unexpected}", flush=True)
        print(f"loaded checkpoint from {args.load_path}", flush=True)

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

        val_metrics = run_epoch_joint(
            model,
            val_loader,
            optimizer=None,
            device=args.device,
            attention_window=active_attention_window,
        )

        train_metrics = run_epoch_joint(
            model,
            train_loader,
            optimizer,
            device=args.device,
            attention_window=active_attention_window,
        )

        print(
            f"epoch {epoch:03d} | "
            f"train loss {train_metrics['loss']:.4f} mse {train_metrics['mse']:.4f} rmse {train_metrics['rmse']:.4f} "
            f"lr_rmse {train_metrics['lr_rmse']:.3f} lr_acc {train_metrics['lr_acc']:.3f} "
            f"loc_x_rmse {train_metrics['loc_x_rmse']:.3f} loc_y_rmse {train_metrics['loc_y_rmse']:.3f} "
            f"loc_x_acc {train_metrics['loc_x_acc']:.3f} loc_y_acc {train_metrics['loc_y_acc']:.3f} "
            f"turn_acc {train_metrics['turn_acc']:.3f} step_acc {train_metrics['step_acc']:.3f} "
            f"kl_dyn {train_metrics['kl_dyn']:.4f} kl_rep {train_metrics['kl_rep']:.4f} prior_roll {train_metrics['prior_roll']:.4f} z_only {train_metrics['z_only']:.4f} h_only {train_metrics['h_only']:.4f} recon {train_metrics['recon']:.4f} cpc {train_metrics['contrastive']:.4f} cpc_acc {train_metrics['contrastive_acc']:.3f} | "
            f"val loss {val_metrics['loss']:.4f} mse {val_metrics['mse']:.4f} rmse {val_metrics['rmse']:.4f} "
            f"lr_rmse {val_metrics['lr_rmse']:.3f} lr_acc {val_metrics['lr_acc']:.3f} "
            f"loc_x_rmse {val_metrics['loc_x_rmse']:.3f} loc_y_rmse {val_metrics['loc_y_rmse']:.3f} "
            f"loc_x_acc {val_metrics['loc_x_acc']:.3f} loc_y_acc {val_metrics['loc_y_acc']:.3f} "
            f"turn_acc {val_metrics['turn_acc']:.3f} step_acc {val_metrics['step_acc']:.3f} "
            f"kl_dyn {val_metrics['kl_dyn']:.4f} kl_rep {val_metrics['kl_rep']:.4f} prior_roll {val_metrics['prior_roll']:.4f} z_only {val_metrics['z_only']:.4f} h_only {val_metrics['h_only']:.4f} recon {val_metrics['recon']:.4f} cpc {val_metrics['contrastive']:.4f} cpc_acc {val_metrics['contrastive_acc']:.3f}"
            ,
            flush=True,
        )

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
                "rnn_transition": args.rnn_transition,
                "rnn_residual_scale": args.rnn_residual_scale,
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


if __name__ == "__main__":
    main()
