#!/usr/bin/env python3
"""Model construction utilities for world-model training."""

import argparse
from dataclasses import asdict
from types import SimpleNamespace
from typing import Dict

import torch

from baselines import RNNPredictor, TransformerBaseline
from rssm import RSSMDiscretePredictor
from transformer import LlamaConfig
from tssm import TSSMDiscretePredictor


_MODEL_ARG_DEFAULTS = {
    "model_type": "transformer",
    "sensor_mode": "raw",
    "hidden_size": 176,
    "layers": 3,
    "heads": 4,
    "head_dim": 44,
    "intermediate": 704,
    "attention_window": None,
    "attention_dropout": 0.0,
    "rnn_state_norm": "none",
    "rssm_transition": "gru",
    "rssm_residual_scale": 1.0,
    "rssm_state_norm": "none",
    "stoch_size": 32,
    "stoch_classes": 32,
    "stoch_temp": 1.0,
    "kl_dyn_beta": 1.0,
    "kl_rep_beta": 0.1,
    "kl_free_nats": 1.0,
    "prior_rollout_weight": 0.0,
    "z_only_weight": 0.0,
    "h_only_weight": 0.0,
    "contrastive_dim": 0,
    "contrastive_steps": 1,
    "bptt_horizon": 0,
    "prior_rollout_steps": 0,
    "recon_beta": 1.0,
    "obs_loss_mode": "soft",
    "probe_hidden_dim": 256,
    "probe_layers": 2,
    "obs_latent_dim": 64,
}

MAZE_WM_MODEL_DEFAULTS = {
    "model_type": "rnn",
    "sensor_mode": "categorical",
    "hidden_size": 176,
    "layers": 3,
    "heads": 4,
    "head_dim": 44,
    "intermediate": 704,
    "attention_dropout": 0.0,
    "obs_latent_dim": 64,
    "probe_hidden_dim": 128,
    "probe_layers": 2,
    "contrastive_dim": 64,
    "contrastive_steps": 1,
    "rnn_state_norm": "none",
}

_MODEL_ARG_FIELDS = (
    "model_type",
    "sensor_mode",
    "hidden_size",
    "layers",
    "heads",
    "head_dim",
    "intermediate",
    "attention_window",
    "attention_dropout",
    "rnn_state_norm",
    "rssm_transition",
    "rssm_residual_scale",
    "rssm_state_norm",
    "stoch_size",
    "stoch_classes",
    "stoch_temp",
    "kl_dyn_beta",
    "kl_rep_beta",
    "kl_free_nats",
    "prior_rollout_weight",
    "z_only_weight",
    "h_only_weight",
    "contrastive_dim",
    "contrastive_steps",
    "bptt_horizon",
    "prior_rollout_steps",
    "recon_beta",
    "obs_loss_mode",
    "probe_hidden_dim",
    "probe_layers",
    "obs_latent_dim",
    "load_path",
)


def _arg_dest(arg_prefix: str, name: str) -> str:
    if not arg_prefix:
        return name
    return f"{arg_prefix}_{name}"


def _arg_option(arg_prefix: str, name: str) -> str:
    option_name = name.replace("_", "-")
    if not arg_prefix:
        return f"--{option_name}"
    return f"--{arg_prefix}-{option_name}"


def add_create_model_args(
    parser: argparse.ArgumentParser,
    *,
    arg_prefix: str = "",
    defaults: Dict | None = None,
    include_load_path: bool = True,
) -> None:
    resolved_defaults = dict(_MODEL_ARG_DEFAULTS)
    if defaults:
        resolved_defaults.update(defaults)

    parser.add_argument(
        _arg_option(arg_prefix, "hidden_size"),
        dest=_arg_dest(arg_prefix, "hidden_size"),
        type=int,
        default=resolved_defaults["hidden_size"],
    )
    parser.add_argument(
        _arg_option(arg_prefix, "layers"),
        dest=_arg_dest(arg_prefix, "layers"),
        type=int,
        default=resolved_defaults["layers"],
    )
    parser.add_argument(
        _arg_option(arg_prefix, "heads"),
        dest=_arg_dest(arg_prefix, "heads"),
        type=int,
        default=resolved_defaults["heads"],
    )
    parser.add_argument(
        _arg_option(arg_prefix, "head_dim"),
        dest=_arg_dest(arg_prefix, "head_dim"),
        type=int,
        default=resolved_defaults["head_dim"],
    )
    parser.add_argument(
        _arg_option(arg_prefix, "intermediate"),
        dest=_arg_dest(arg_prefix, "intermediate"),
        type=int,
        default=resolved_defaults["intermediate"],
    )
    parser.add_argument(
        _arg_option(arg_prefix, "attention_window"),
        dest=_arg_dest(arg_prefix, "attention_window"),
        type=int,
        default=resolved_defaults["attention_window"],
    )
    parser.add_argument(
        _arg_option(arg_prefix, "attention_dropout"),
        dest=_arg_dest(arg_prefix, "attention_dropout"),
        type=float,
        default=resolved_defaults["attention_dropout"],
        help="Attention dropout probability for transformer model.",
    )
    parser.add_argument(
        _arg_option(arg_prefix, "model_type"),
        dest=_arg_dest(arg_prefix, "model_type"),
        type=str,
        choices=["transformer", "rnn", "rssm-discrete", "tssm"],
        default=resolved_defaults["model_type"],
        help="Model architecture to train.",
    )
    parser.add_argument(
        _arg_option(arg_prefix, "rnn_state_norm"),
        dest=_arg_dest(arg_prefix, "rnn_state_norm"),
        type=str,
        choices=["none", "layernorm", "rmsnorm"],
        default=resolved_defaults["rnn_state_norm"],
        help="Normalization on RNN hidden sequence before prediction heads.",
    )
    parser.add_argument(
        _arg_option(arg_prefix, "rssm_transition"),
        dest=_arg_dest(arg_prefix, "rssm_transition"),
        type=str,
        choices=["gru", "residual"],
        default=resolved_defaults["rssm_transition"],
        help="RSSM deterministic transition: GRUCell or residual update h_t=h_{t-1}+g_t.",
    )
    parser.add_argument(
        _arg_option(arg_prefix, "rssm_residual_scale"),
        dest=_arg_dest(arg_prefix, "rssm_residual_scale"),
        type=float,
        default=resolved_defaults["rssm_residual_scale"],
        help="Scale for RSSM residual update when --rssm-transition=residual.",
    )
    parser.add_argument(
        _arg_option(arg_prefix, "rssm_state_norm"),
        dest=_arg_dest(arg_prefix, "rssm_state_norm"),
        type=str,
        choices=["none", "layernorm", "rmsnorm"],
        default=resolved_defaults["rssm_state_norm"],
        help="Pre-normalization on RSSM h_{t-1} before transition step.",
    )
    parser.add_argument(
        _arg_option(arg_prefix, "stoch_size"),
        dest=_arg_dest(arg_prefix, "stoch_size"),
        type=int,
        default=resolved_defaults["stoch_size"],
        help="Number of categorical latent groups.",
    )
    parser.add_argument(
        _arg_option(arg_prefix, "stoch_classes"),
        dest=_arg_dest(arg_prefix, "stoch_classes"),
        type=int,
        default=resolved_defaults["stoch_classes"],
        help="Number of classes per latent group.",
    )
    parser.add_argument(
        _arg_option(arg_prefix, "stoch_temp"),
        dest=_arg_dest(arg_prefix, "stoch_temp"),
        type=float,
        default=resolved_defaults["stoch_temp"],
        help="Gumbel-softmax temperature.",
    )
    parser.add_argument(
        _arg_option(arg_prefix, "kl_dyn_beta"),
        dest=_arg_dest(arg_prefix, "kl_dyn_beta"),
        type=float,
        default=resolved_defaults["kl_dyn_beta"],
        help="Weight for dynamics KL term.",
    )
    parser.add_argument(
        _arg_option(arg_prefix, "kl_rep_beta"),
        dest=_arg_dest(arg_prefix, "kl_rep_beta"),
        type=float,
        default=resolved_defaults["kl_rep_beta"],
        help="Weight for representation KL term.",
    )
    parser.add_argument(
        _arg_option(arg_prefix, "kl_free_nats"),
        dest=_arg_dest(arg_prefix, "kl_free_nats"),
        type=float,
        default=resolved_defaults["kl_free_nats"],
        help="Free nats clamp for both KL terms.",
    )
    parser.add_argument(
        _arg_option(arg_prefix, "prior_rollout_weight"),
        dest=_arg_dest(arg_prefix, "prior_rollout_weight"),
        type=float,
        default=resolved_defaults["prior_rollout_weight"],
        help="Auxiliary weight for full-sequence open-loop prior rollout sensor prediction.",
    )
    parser.add_argument(
        _arg_option(arg_prefix, "z_only_weight"),
        dest=_arg_dest(arg_prefix, "z_only_weight"),
        type=float,
        default=resolved_defaults["z_only_weight"],
        help="Auxiliary weight for z-only sensor prediction.",
    )
    parser.add_argument(
        _arg_option(arg_prefix, "h_only_weight"),
        dest=_arg_dest(arg_prefix, "h_only_weight"),
        type=float,
        default=resolved_defaults["h_only_weight"],
        help="Auxiliary weight for h-only sensor prediction.",
    )
    parser.add_argument(
        _arg_option(arg_prefix, "contrastive_dim"),
        dest=_arg_dest(arg_prefix, "contrastive_dim"),
        type=int,
        default=resolved_defaults["contrastive_dim"],
        help="Embedding dimension for contrastive head (0 disables contrastive head).",
    )
    parser.add_argument(
        _arg_option(arg_prefix, "contrastive_steps"),
        dest=_arg_dest(arg_prefix, "contrastive_steps"),
        type=int,
        default=resolved_defaults["contrastive_steps"],
        help="Number of Twister-style action-conditioned contrastive horizons (K).",
    )
    parser.add_argument(
        _arg_option(arg_prefix, "bptt_horizon"),
        dest=_arg_dest(arg_prefix, "bptt_horizon"),
        type=int,
        default=resolved_defaults["bptt_horizon"],
        help="Truncated BPTT horizon for RSSM/TSSM recurrent state (0 disables truncation).",
    )
    parser.add_argument(
        _arg_option(arg_prefix, "prior_rollout_steps"),
        dest=_arg_dest(arg_prefix, "prior_rollout_steps"),
        type=int,
        default=resolved_defaults["prior_rollout_steps"],
        help="Open-loop prior rollout length for prior_roll loss (0 uses full sequence).",
    )
    parser.add_argument(
        _arg_option(arg_prefix, "recon_beta"),
        dest=_arg_dest(arg_prefix, "recon_beta"),
        type=float,
        default=resolved_defaults["recon_beta"],
        help="Weight for observation reconstruction term.",
    )
    parser.add_argument(
        _arg_option(arg_prefix, "obs_loss_mode"),
        dest=_arg_dest(arg_prefix, "obs_loss_mode"),
        type=str,
        choices=["soft", "recon"],
        default=resolved_defaults["obs_loss_mode"],
        help="Observation loss for RSSM/TSSM: soft-target heads or L2 reconstruction from [h_t, z_t].",
    )
    parser.add_argument(
        _arg_option(arg_prefix, "probe_hidden_dim"),
        dest=_arg_dest(arg_prefix, "probe_hidden_dim"),
        type=int,
        default=resolved_defaults["probe_hidden_dim"],
        help="Hidden size for probe MLPs (set 0 to keep linear probes).",
    )
    parser.add_argument(
        _arg_option(arg_prefix, "probe_layers"),
        dest=_arg_dest(arg_prefix, "probe_layers"),
        type=int,
        default=resolved_defaults["probe_layers"],
        help="Number of linear layers in probe heads (1=linear probe).",
    )
    parser.add_argument(
        _arg_option(arg_prefix, "obs_latent_dim"),
        dest=_arg_dest(arg_prefix, "obs_latent_dim"),
        type=int,
        default=resolved_defaults["obs_latent_dim"],
    )
    parser.add_argument(
        _arg_option(arg_prefix, "sensor_mode"),
        dest=_arg_dest(arg_prefix, "sensor_mode"),
        type=str,
        choices=["raw", "categorical"],
        default=resolved_defaults["sensor_mode"],
        help="Sensor target mode: raw regression or categorical",
    )
    if include_load_path:
        parser.add_argument(
            _arg_option(arg_prefix, "load_path"),
            dest=_arg_dest(arg_prefix, "load_path"),
            type=str,
            default=None,
            help="Optional checkpoint to warm start from.",
        )


def extract_create_model_args(
    args: argparse.Namespace,
    *,
    arg_prefix: str = "",
    device=None,
) -> SimpleNamespace:
    values = {}
    for field in _MODEL_ARG_FIELDS:
        dest = _arg_dest(arg_prefix, field)
        if hasattr(args, dest):
            values[field] = getattr(args, dest)
            continue
        if arg_prefix and hasattr(args, field):
            values[field] = getattr(args, field)
            continue
        raise AttributeError(f"Missing create_model argument '{dest}'")

    if device is None:
        if arg_prefix and hasattr(args, _arg_dest(arg_prefix, "device")):
            device = getattr(args, _arg_dest(arg_prefix, "device"))
        elif hasattr(args, "device"):
            device = getattr(args, "device")
        else:
            raise AttributeError("Missing device argument for create_model")
    values["device"] = device

    return SimpleNamespace(**values)


def create_model(
    args,
    *,
    input_dim: int,
    sensor_dim: int,
    sensor_bins,
    loc_x_bins: int,
    loc_y_bins: int,
    heading_dim: int,
    turn_bins: int,
    step_bins: int,
    obs_dim: int,
    action_dim: int,
    active_attention_window,
    model_config_extra: Dict,
):
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
        model = TransformerBaseline(
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
        for key, value in state.items():
            if key not in model_state:
                continue
            if model_state[key].shape != value.shape:
                skipped_mismatch.append((key, tuple(value.shape), tuple(model_state[key].shape)))
                continue
            filtered_state[key] = value
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

    return model
