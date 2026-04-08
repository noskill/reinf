#!/usr/bin/env python3
"""Model construction utilities for world-model training."""

from dataclasses import asdict
from typing import Dict

import torch

from baselines import RNNPredictor, UnifiedPredictor
from rssm import RSSMDiscretePredictor
from transformer import LlamaConfig
from tssm import TSSMDiscretePredictor


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
