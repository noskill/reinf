#!/usr/bin/env python3
"""Training/evaluation epoch utilities."""

import torch

def run_epoch_joint(
    model,
    loader,
    optimizer,
    device,
    logger,
    metric_prefix=None,
    step=None,
):
    is_train = optimizer is not None
    model.train(is_train)

    totals = {
        "loss": 0.0,
        "mse": 0.0,
        "rmse": 0.0,
        "lr_rmse": 0.0,
        "lr_acc": 0.0,
        "loc_x_rmse": 0.0,
        "loc_y_rmse": 0.0,
        "loc_x_acc": 0.0,
        "loc_y_acc": 0.0,
        "turn_acc": 0.0,
        "step_acc": 0.0,
        "kl_dyn": 0.0,
        "kl_rep": 0.0,
        "prior_roll": 0.0,
        "z_only": 0.0,
        "h_only": 0.0,
        "recon": 0.0,
        "contrastive": 0.0,
        "contrastive_acc": 0.0,
    }
    total_batches = 0
    cfg = model.config

    for item in loader:
        obs, y_sensor, y_sensor_idx, y_loc_xy, y_head, y_turn, y_step, _ = item

        obs = {k: v.to(device) for k, v in obs.items()}
        y_sensor = y_sensor.to(device)
        y_sensor_idx = y_sensor_idx.to(device)
        y_loc_xy = y_loc_xy.to(device)
        y_head = y_head.to(device)
        y_turn = y_turn.to(device)
        y_step = y_step.to(device)
        kpm = obs["key_padding_mask"]

        targets = {
            "y_sensor": y_sensor,
            "y_sensor_idx": y_sensor_idx,
            "y_loc_xy": y_loc_xy,
            "y_head": y_head,
            "y_turn": y_turn,
            "y_step": y_step,
            "key_padding_mask": kpm,
        }

        forward_out = model(
            obs,
        )
        if not isinstance(forward_out, dict):
            raise ValueError("forward(...) must return dict with preds/aux/state")
        preds = forward_out.get("preds")
        aux_inputs = forward_out.get("aux")
        if preds is None or not isinstance(preds, tuple) or len(preds) != 6:
            raise ValueError("forward(...)[\"preds\"] must be a 6-tuple")
        loss_dict = model.compute_prediction_losses(
            preds=preds,
            targets=targets,
            aux_inputs=aux_inputs,
        )
        metrics = model.compute_prediction_metrics(
            preds=preds,
            targets=targets,
            aux_inputs=aux_inputs,
        )

        obs_total = loss_dict.get("obs_total")
        if obs_total is None:
            obs_total = (
                cfg.sensor_weight * loss_dict["sensor"]
                + cfg.loc_weight * (loss_dict["loc_x"] + loss_dict["loc_y"])
                + cfg.head_weight * loss_dict["head"]
            )
        loss = (
            obs_total
            + cfg.turn_weight * loss_dict["turn"]
            + cfg.step_weight * loss_dict["step"]
            + loss_dict.get("aux_total", torch.tensor(0.0, device=device))
        )
        if cfg.contrastive_weight > 0:
            loss = loss + cfg.contrastive_weight * loss_dict.get("contrastive", torch.tensor(0.0, device=device))

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        totals["loss"] += float(loss.detach().cpu())
        for key in (
            "mse",
            "rmse",
            "lr_rmse",
            "lr_acc",
            "loc_x_rmse",
            "loc_y_rmse",
            "loc_x_acc",
            "loc_y_acc",
            "turn_acc",
            "step_acc",
            "contrastive_acc",
        ):
            val = metrics.get(key, torch.tensor(0.0, device=device))
            totals[key] += float(val.detach().cpu())
        for loss_key, total_key in (
            ("kl_dyn", "kl_dyn"),
            ("kl_rep", "kl_rep"),
            ("prior_rollout_sensor", "prior_roll"),
            ("z_only_sensor", "z_only"),
            ("h_only_sensor", "h_only"),
            ("recon", "recon"),
            ("contrastive", "contrastive"),
        ):
            totals[total_key] += float(
                loss_dict.get(loss_key, torch.tensor(0.0, device=device)).detach().cpu()
            )
        total_batches += 1

    if total_batches == 0:
        return
    epoch_metrics = {k: v / total_batches for k, v in totals.items()}
    for key, value in epoch_metrics.items():
        name = f"{metric_prefix}/{key}" if metric_prefix else key
        logger.log_scalar(name, value, step=step)
