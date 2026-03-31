#!/usr/bin/env python3
"""Training/evaluation epoch utilities."""

from typing import Optional

import torch

from base import DiscreteLatentPredictorBase
from utils import batch_row_next_step_contrastive_pair_loss
def run_epoch_joint(
    model,
    loader,
    optimizer,
    device,
    attention_window=None,
    sensor_weight=1.0,
    loc_weight=1.0,
    head_weight=1.0,
    turn_weight=1.0,
    step_weight=1.0,
    loc_min: Optional[torch.Tensor] = None,
    sensor_mode="raw",
    sensor_min_idx=None,
    sensor_tables=None,
    loc_x_table=None,
    loc_y_table=None,
    heading_table=None,
    contrastive_weight=0.0,
    contrastive_temp=0.1,
):
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_mse = 0.0
    total_rmse = 0.0
    total_lr_rmse = 0.0
    total_lr_acc = 0.0
    total_loc_x_rmse = 0.0
    total_loc_y_rmse = 0.0
    total_loc_x_acc = 0.0
    total_loc_y_acc = 0.0
    total_batches = 0

    total_turn_acc = 0.0
    total_step_acc = 0.0
    total_kl_dyn = 0.0
    total_kl_rep = 0.0
    total_prior_roll = 0.0
    total_z_only = 0.0
    total_h_only = 0.0
    total_recon = 0.0
    total_contrastive = 0.0

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

        need_aux = bool(contrastive_weight > 0)
        if isinstance(model, DiscreteLatentPredictorBase):
            pred_sensor, pred_loc_x, pred_loc_y, pred_head, pred_turn, pred_step, aux_inputs = model(
                obs, attention_window=attention_window
            )
        else:
            if need_aux:
                pred_sensor, pred_loc_x, pred_loc_y, pred_head, pred_turn, pred_step, aux_inputs = model(
                    obs, attention_window=attention_window, return_aux=True
                )
            else:
                pred_sensor, pred_loc_x, pred_loc_y, pred_head, pred_turn, pred_step = model(
                    obs, attention_window=attention_window
                )
                aux_inputs = None
        loss_dict = model.compute_all_losses(
            pred_sensor=pred_sensor,
            pred_loc_x=pred_loc_x,
            pred_loc_y=pred_loc_y,
            pred_head=pred_head,
            pred_turn=pred_turn,
            pred_step=pred_step,
            y_sensor=y_sensor,
            y_sensor_idx=y_sensor_idx,
            y_loc_xy=y_loc_xy,
            y_head=y_head,
            y_turn=y_turn,
            y_step=y_step,
            key_padding_mask=kpm,
            sensor_tables=sensor_tables,
            sensor_min_idx=sensor_min_idx,
            loc_x_table=loc_x_table,
            loc_y_table=loc_y_table,
            heading_table=heading_table,
            sensor_weight=sensor_weight,
            loc_weight=loc_weight,
            head_weight=head_weight,
            turn_weight=turn_weight,
            step_weight=step_weight,
            aux_inputs=aux_inputs,
        )

        metrics = model.compute_metrics(
            pred_sensor=pred_sensor,
            pred_loc_x=pred_loc_x,
            pred_loc_y=pred_loc_y,
            pred_turn=pred_turn,
            pred_step=pred_step,
            y_sensor=y_sensor,
            y_sensor_idx=y_sensor_idx,
            y_loc_xy=y_loc_xy,
            y_turn=y_turn,
            y_step=y_step,
            key_padding_mask=kpm,
            sensor_min_idx=sensor_min_idx,
            loc_min=loc_min,
            aux_inputs=aux_inputs,
        )
        loss = loss_dict["total"]
        contrastive_loss = torch.tensor(0.0, device=device)
        if contrastive_weight > 0:
            pred_emb = aux_inputs.get("contrastive_pred_emb") if aux_inputs is not None else None
            tgt_emb = aux_inputs.get("contrastive_tgt_emb") if aux_inputs is not None else None
            contrastive_loss = batch_row_next_step_contrastive_pair_loss(
                pred_emb=pred_emb,
                target_emb=tgt_emb,
                key_padding_mask=kpm,
                temperature=contrastive_temp,
            )
            loss = loss + contrastive_weight * contrastive_loss

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        total_loss += float(loss.detach().cpu())
        total_mse += float(metrics["mse"].detach().cpu())
        total_rmse += float(metrics["rmse"].detach().cpu())
        total_lr_rmse += float(metrics["lr_rmse"].detach().cpu())
        total_lr_acc += float(metrics["lr_acc"].detach().cpu())
        total_turn_acc += float(metrics["turn_acc"].detach().cpu())
        total_step_acc += float(metrics["step_acc"].detach().cpu())
        total_kl_dyn += float(loss_dict.get("kl_dyn", torch.tensor(0.0, device=device)).detach().cpu())
        total_kl_rep += float(loss_dict.get("kl_rep", torch.tensor(0.0, device=device)).detach().cpu())
        total_prior_roll += float(
            loss_dict.get("prior_rollout_sensor", torch.tensor(0.0, device=device)).detach().cpu()
        )
        total_z_only += float(
            loss_dict.get("z_only_sensor", torch.tensor(0.0, device=device)).detach().cpu()
        )
        total_h_only += float(
            loss_dict.get("h_only_sensor", torch.tensor(0.0, device=device)).detach().cpu()
        )
        total_recon += float(loss_dict.get("recon", torch.tensor(0.0, device=device)).detach().cpu())
        total_contrastive += float(contrastive_loss.detach().cpu())
        total_loc_x_rmse += float(metrics["loc_x_rmse"].detach().cpu())
        total_loc_y_rmse += float(metrics["loc_y_rmse"].detach().cpu())
        total_loc_x_acc += float(metrics["loc_x_acc"].detach().cpu())
        total_loc_y_acc += float(metrics["loc_y_acc"].detach().cpu())
        total_batches += 1

    if total_batches == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    return (
        total_loss / total_batches,
        total_mse / total_batches,
        total_rmse / total_batches,
        total_lr_rmse / total_batches,
        total_lr_acc / total_batches,
        total_loc_x_rmse / total_batches,
        total_loc_y_rmse / total_batches,
        total_loc_x_acc / total_batches,
        total_loc_y_acc / total_batches,
        total_turn_acc / total_batches,
        total_step_acc / total_batches,
        total_kl_dyn / total_batches,
        total_kl_rep / total_batches,
        total_prior_roll / total_batches,
        total_z_only / total_batches,
        total_h_only / total_batches,
        total_recon / total_batches,
        total_contrastive / total_batches,
    )
