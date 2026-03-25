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
import ast
import os
import random
import sys
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader

from transformer import LlamaConfig, LlamaModel
from tr_cache import PositionBasedDynamicCache, WindowedPositionBasedDynamicCache


sg = lambda x: x.detach()

HEADING_CANON = {"up": "u"}


def make_probe_head(in_dim: int, out_dim: int, hidden_dim: int) -> nn.Module:
    """Build a linear probe or a 2-layer MLP probe."""
    if int(hidden_dim) <= 0:
        return nn.Linear(in_dim, out_dim)
    h = int(hidden_dim)
    return nn.Sequential(
        nn.Linear(in_dim, h),
        nn.ReLU(),
        nn.Linear(h, out_dim),
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_episodes(path: str, max_episodes: Optional[int] = None) -> List[List[dict]]:
    episodes: List[List[dict]] = []
    buf: List[str] = []

    def flush_buf():
        if not buf:
            return
        joined = "".join(buf)
        steps = []
        for item in joined.split(";"):
            item = item.strip()
            if not item:
                continue
            d = ast.literal_eval(item)
            h = d.get("heading")
            if h in HEADING_CANON:
                d["heading"] = HEADING_CANON[h]
            steps.append(d)
        if steps:
            episodes.append(steps)
        buf.clear()

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("random "):
                flush_buf()
                if max_episodes is not None and len(episodes) >= max_episodes:
                    break
                continue
            buf.append(line)
        flush_buf()

    if max_episodes is not None:
        episodes = episodes[:max_episodes]
    return episodes


@dataclass
class FeatureStats:
    loc_min: np.ndarray
    loc_max: np.ndarray
    sensor_min: np.ndarray
    sensor_max: np.ndarray
    action_turn_max: float
    action_step_max: float
    heading_to_idx: dict
    turn_to_idx: dict
    step_to_idx: dict


def compute_stats(episodes: List[List[dict]]) -> FeatureStats:
    loc_min = np.array([np.inf, np.inf], dtype=np.float32)
    loc_max = np.array([-np.inf, -np.inf], dtype=np.float32)
    sensor_min = np.array([np.inf, np.inf, np.inf], dtype=np.float32)
    sensor_max = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float32)
    action_turn_max = 1.0
    action_step_max = 1.0
    headings = set()
    turns = set()
    steps = set()

    for ep in episodes:
        for step in ep:
            loc = np.array(step["location"], dtype=np.float32)
            sensor = np.array(step["sensor"], dtype=np.float32)[:3]
            action = step["action"]
            heading = step["heading"]
            headings.add(heading)

            loc_min = np.minimum(loc_min, loc)
            loc_max = np.maximum(loc_max, loc)
            sensor_min = np.minimum(sensor_min, sensor)
            sensor_max = np.maximum(sensor_max, sensor)
            action_turn_max = max(action_turn_max, abs(float(action[0])))
            action_step_max = max(action_step_max, abs(float(action[1])))
            turns.add(int(action[0]))
            steps.add(int(action[1]))

    heading_list = sorted(headings)
    heading_to_idx = {h: i for i, h in enumerate(heading_list)}
    # Map discrete action values to class indices for categorical action prediction.
    turn_list = sorted(turns)
    step_list = sorted(steps)
    turn_to_idx = {t: i for i, t in enumerate(turn_list)}
    step_to_idx = {s: i for i, s in enumerate(step_list)}

    return FeatureStats(
        loc_min=loc_min,
        loc_max=loc_max,
        sensor_min=sensor_min,
        sensor_max=sensor_max,
        action_turn_max=action_turn_max,
        action_step_max=action_step_max,
        heading_to_idx=heading_to_idx,
        turn_to_idx=turn_to_idx,
        step_to_idx=step_to_idx,
    )


def normalize_range(x: np.ndarray, min_v: np.ndarray, max_v: np.ndarray) -> np.ndarray:
    denom = np.where(max_v > min_v, max_v - min_v, 1.0)
    return (x - min_v) / denom


def normalize_action(turn: float, step: float, stats: FeatureStats) -> np.ndarray:
    turn_n = float(turn) / stats.action_turn_max if stats.action_turn_max != 0 else 0.0
    step_n = float(step) / stats.action_step_max if stats.action_step_max != 0 else 0.0
    return np.array([turn_n, step_n], dtype=np.float32)


class EpisodeSequence:
    def __init__(
        self,
        obs_cont: np.ndarray,
        heading_idx: np.ndarray,
        action_cont: np.ndarray,
        loc_idx: np.ndarray,
        sensor_idx: np.ndarray,
        turn_idx: np.ndarray,
        step_idx: np.ndarray,
    ):
        self.obs_cont = obs_cont
        self.heading_idx = heading_idx
        self.action_cont = action_cont
        self.loc_idx = loc_idx
        self.sensor_idx = sensor_idx
        self.turn_idx = turn_idx
        self.step_idx = step_idx

    @property
    def length(self) -> int:
        return self.obs_cont.shape[0]


def build_sequences(
    episodes: List[List[dict]],
    stats: FeatureStats,
    sensor_mode: str,
) -> List[EpisodeSequence]:
    sequences: List[EpisodeSequence] = []
    for ep in episodes:
        obs_cont = []
        heading_idx = []
        action_cont = []
        loc_idx = []
        sensor_idx = []
        turn_idx = []
        step_idx = []
        for step in ep:
            loc = np.array(step["location"], dtype=np.float32)
            sensor = np.array(step["sensor"], dtype=np.float32)[:3]
            heading = step["heading"]
            action = step["action"]

            sensor_in = sensor
            # Only sensors are used as model inputs; location/heading are targets/probes.
            obs_cont.append(sensor_in)
            heading_idx.append(stats.heading_to_idx[heading])
            action_cont.append(normalize_action(action[0], action[1], stats))
            loc_idx.append(loc.astype(np.int64))
            sensor_idx.append(sensor.astype(np.int64))
            turn_idx.append(stats.turn_to_idx[int(action[0])])
            step_idx.append(stats.step_to_idx[int(action[1])])

        obs_cont = np.asarray(obs_cont, dtype=np.float32)
        heading_idx = np.asarray(heading_idx, dtype=np.int64)
        action_cont = np.asarray(action_cont, dtype=np.float32)
        loc_idx = np.asarray(loc_idx, dtype=np.int64)
        sensor_idx = np.asarray(sensor_idx, dtype=np.int64)
        turn_idx = np.asarray(turn_idx, dtype=np.int64)
        step_idx = np.asarray(step_idx, dtype=np.int64)

        if obs_cont.shape[0] >= 2:
            sequences.append(
                EpisodeSequence(obs_cont, heading_idx, action_cont, loc_idx, sensor_idx, turn_idx, step_idx)
            )
    return sequences


class WindowedDataset(Dataset):
    def __init__(
        self,
        sequences: List[EpisodeSequence],
        heading_dim: int = 4,
        context_len: int = 0,
    ):
        self.samples: List[Tuple[int, int, int]] = []
        self.sequences = sequences
        self.heading_dim = heading_dim
        self.context_len = int(context_len)
        if self.context_len < 0:
            raise ValueError("context_len must be >= 0")
        # Full-episode samples only (need t+1 targets) when context_len == 0.
        # Otherwise, use sliding windows of fixed length context_len.
        for si, seq in enumerate(sequences):
            max_start = seq.length - 1
            if max_start <= 0:
                continue
            if self.context_len == 0:
                self.samples.append((si, 0, max_start))
                continue
            if self.context_len > max_start:
                continue
            for start in range(0, max_start - self.context_len + 1):
                end = start + self.context_len
                self.samples.append((si, start, end))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        si, start, end = self.samples[idx]
        seq = self.sequences[si]

        obs_cont = seq.obs_cont[start:end]
        heading = seq.heading_idx[start:end]
        action = seq.action_cont[start:end]

        obs_sensor = obs_cont
        # Location is a target/probe only; keep current loc separately.
        obs_loc = seq.loc_idx[start:end].astype(np.float32)

        # targets: next observation
        y_sensor = seq.obs_cont[start + 1 : end + 1]  # next-step sensors (left/front/right)
        y_sensor_idx = seq.sensor_idx[start + 1 : end + 1]
        y_heading = seq.heading_idx[start + 1 : end + 1]
        y_loc_xy = seq.loc_idx[start + 1 : end + 1]
        # action targets aligned to current obs (t), first step will be masked
        y_turn = seq.turn_idx[start:end]
        y_step = seq.step_idx[start:end]

        return (
            torch.from_numpy(obs_sensor),
            torch.from_numpy(obs_loc),
            torch.from_numpy(heading),
            torch.from_numpy(action),
            torch.from_numpy(y_sensor),
            torch.from_numpy(y_sensor_idx),
            torch.from_numpy(y_loc_xy),
            torch.from_numpy(y_heading),
            torch.from_numpy(y_turn),
            torch.from_numpy(y_step),
        )

    def episode_id(self, idx: int) -> int:
        return self.samples[idx][0]


def collate_batch(batch):
    obs_sensors, obs_locs, obs_heads, obs_actions, ys_sensor, ys_sensor_idx, ys_loc_xy, ys_head, ys_turn, ys_step = zip(*batch)
    lengths = [x.shape[0] for x in obs_sensors]
    max_len = max(lengths)

    sensor_dim = ys_sensor[0].shape[-1]
    obs_sensor_pad = torch.zeros((len(obs_sensors), max_len, sensor_dim), dtype=torch.float32)
    obs_loc_pad = torch.zeros((len(obs_sensors), max_len, 2), dtype=torch.float32)
    obs_head_pad = torch.full((len(obs_sensors), max_len), -100, dtype=torch.long)
    obs_action_pad = torch.zeros((len(obs_sensors), max_len, 2), dtype=torch.float32)

    batch_size = len(obs_sensors)
    y_sensor_pad = torch.zeros((batch_size, max_len, sensor_dim), dtype=torch.float32)
    y_sensor_idx_pad = torch.full((batch_size, max_len, sensor_dim), -1, dtype=torch.long)
    loc_xy_pad = torch.full((batch_size, max_len, 2), -1, dtype=torch.long)
    y_head_pad = torch.full((batch_size, max_len), -100, dtype=torch.long)
    y_turn_pad = torch.full((batch_size, max_len), -100, dtype=torch.long)
    y_step_pad = torch.full((batch_size, max_len), -100, dtype=torch.long)
    key_padding_mask = torch.ones((batch_size, max_len), dtype=torch.bool)

    for i, (o_s, o_l, o_h, o_a, y_s, y_si, y_xy, y_h, y_t, y_st) in enumerate(
        zip(obs_sensors, obs_locs, obs_heads, obs_actions, ys_sensor, ys_sensor_idx, ys_loc_xy, ys_head, ys_turn, ys_step)
    ):
        L = o_s.shape[0]
        obs_sensor_pad[i, :L] = o_s
        obs_loc_pad[i, :L] = o_l
        obs_head_pad[i, :L] = o_h
        obs_action_pad[i, :L] = o_a
        y_sensor_pad[i, :L] = y_s
        y_sensor_idx_pad[i, :L] = y_si
        loc_xy_pad[i, :L] = y_xy
        y_head_pad[i, :L] = y_h
        y_turn_pad[i, :L] = y_t
        y_step_pad[i, :L] = y_st
        key_padding_mask[i, :L] = False
        if L > 0:
            y_turn_pad[i, 0] = -100
            y_step_pad[i, 0] = -100

    obs = {
        "sensor": obs_sensor_pad,
        "loc": obs_loc_pad,
        "heading": obs_head_pad,
        "actions": obs_action_pad,
        "key_padding_mask": key_padding_mask,
    }

    return (
        obs,
        y_sensor_pad,
        y_sensor_idx_pad,
        loc_xy_pad,
        y_head_pad,
        y_turn_pad,
        y_step_pad,
        key_padding_mask,
    )


class PredictionLossMixin:
    def compute_soft_target_loss(
        self,
        *,
        pred_sensor,
        pred_loc_x: torch.Tensor,
        pred_loc_y: torch.Tensor,
        pred_head: torch.Tensor,
        y_sensor: torch.Tensor,
        y_sensor_idx: torch.Tensor,
        y_loc_xy: torch.Tensor,
        y_head: torch.Tensor,
        key_padding_mask: torch.Tensor,
        sensor_tables,
        sensor_min_idx,
        loc_x_table: torch.Tensor,
        loc_y_table: torch.Tensor,
        heading_table: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        # Soft-target losses treat discrete targets as smoothed distributions (label smoothing / Gaussian kernels).
        pred_l, pred_f, pred_r = pred_sensor
        idx = y_sensor_idx.clamp(min=0)
        if sensor_min_idx is not None:
            idx = (idx - sensor_min_idx.view(1, 1, -1)).clamp(min=0)
        # sensor_tables maps each index to a softened target distribution; we use it for soft CE.
        loss_sensor = (
            soft_cross_entropy(pred_l, sensor_tables[0][idx[..., 0]], key_padding_mask)
            + soft_cross_entropy(pred_f, sensor_tables[1][idx[..., 1]], key_padding_mask)
            + soft_cross_entropy(pred_r, sensor_tables[2][idx[..., 2]], key_padding_mask)
        )

        # Convert target coords to valid bin indices (padding uses -1, so clamp to 0 for table lookup).
        loc_idx = y_loc_xy.clamp(min=0)
        loss_loc_x = soft_cross_entropy(pred_loc_x, loc_x_table[loc_idx[..., 0]], key_padding_mask)
        loss_loc_y = soft_cross_entropy(pred_loc_y, loc_y_table[loc_idx[..., 1]], key_padding_mask)

        head_idx = y_head.clamp(min=0)
        loss_head = soft_cross_entropy(pred_head, heading_table[head_idx], key_padding_mask)
        return {
            "sensor": loss_sensor,
            "loc_x": loss_loc_x,
            "loc_y": loss_loc_y,
            "head": loss_head,
        }

    def compute_observation_losses(
        self,
        *,
        pred_sensor,
        pred_loc_x: torch.Tensor,
        pred_loc_y: torch.Tensor,
        pred_head: torch.Tensor,
        y_sensor: torch.Tensor,
        y_sensor_idx: torch.Tensor,
        y_loc_xy: torch.Tensor,
        y_head: torch.Tensor,
        key_padding_mask: torch.Tensor,
        sensor_tables,
        sensor_min_idx,
        loc_x_table: torch.Tensor,
        loc_y_table: torch.Tensor,
        heading_table: torch.Tensor,
        sensor_weight: float,
        loc_weight: float,
        head_weight: float,
        aux_inputs: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        del aux_inputs
        losses = self.compute_soft_target_loss(
            pred_sensor=pred_sensor,
            pred_loc_x=pred_loc_x,
            pred_loc_y=pred_loc_y,
            pred_head=pred_head,
            y_sensor=y_sensor,
            y_sensor_idx=y_sensor_idx,
            y_loc_xy=y_loc_xy,
            y_head=y_head,
            key_padding_mask=key_padding_mask,
            sensor_tables=sensor_tables,
            sensor_min_idx=sensor_min_idx,
            loc_x_table=loc_x_table,
            loc_y_table=loc_y_table,
            heading_table=heading_table,
        )
        obs_total = (
            sensor_weight * losses["sensor"]
            + loc_weight * (losses["loc_x"] + losses["loc_y"])
            + head_weight * losses["head"]
        )
        return {"obs_total": obs_total}

    def compute_action_losses(
        self,
        *,
        pred_turn: torch.Tensor,
        pred_step: torch.Tensor,
        y_turn: torch.Tensor,
        y_step: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        loss_turn = F.cross_entropy(
            pred_turn.view(-1, pred_turn.shape[-1]),
            y_turn.view(-1),
            ignore_index=-100,
        )
        loss_step = F.cross_entropy(
            pred_step.view(-1, pred_step.shape[-1]),
            y_step.view(-1),
            ignore_index=-100,
        )
        return {
            "turn": loss_turn,
            "step": loss_step,
        }

    def compute_metrics(
        self,
        *,
        pred_sensor,
        pred_loc_x: torch.Tensor,
        pred_loc_y: torch.Tensor,
        pred_turn: torch.Tensor,
        pred_step: torch.Tensor,
        y_sensor: torch.Tensor,
        y_sensor_idx: torch.Tensor,
        y_loc_xy: torch.Tensor,
        y_turn: torch.Tensor,
        y_step: torch.Tensor,
        key_padding_mask: torch.Tensor,
        sensor_min_idx,
        loc_min: Optional[torch.Tensor],
        loc_max: Optional[torch.Tensor] = None,
        aux_inputs: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        del aux_inputs, loc_max
        pred_l, pred_f, pred_r = pred_sensor
        exp_l = expected_from_logits(pred_l, float(sensor_min_idx[0].item()))
        exp_f = expected_from_logits(pred_f, float(sensor_min_idx[1].item()))
        exp_r = expected_from_logits(pred_r, float(sensor_min_idx[2].item()))
        pred_sensor_exp = torch.stack([exp_l, exp_f, exp_r], dim=-1)
        target_sensor_cont = y_sensor_idx.to(pred_sensor_exp.dtype)
        mse = masked_mse(pred_sensor_exp, target_sensor_cont, key_padding_mask)
        rmse = masked_rmse(pred_sensor_exp, target_sensor_cont, key_padding_mask)
        if sensor_min_idx is not None:
            lr_rmse, lr_acc = masked_lr_metrics_logits(
                pred_l, pred_r, y_sensor_idx, key_padding_mask, sensor_min_idx.to(torch.float32)
            )
        else:
            lr_rmse, lr_acc = masked_lr_metrics_logits(
                pred_l, pred_r, y_sensor_idx, key_padding_mask, torch.zeros(3, device=pred_l.device)
            )

        if loc_min is not None:
            loc_x_rmse = masked_coord_rmse(pred_loc_x, y_loc_xy[..., 0], key_padding_mask, loc_min[0])
            loc_y_rmse = masked_coord_rmse(pred_loc_y, y_loc_xy[..., 1], key_padding_mask, loc_min[1])
            loc_x_acc = masked_coord_acc(pred_loc_x, y_loc_xy[..., 0], key_padding_mask)
            loc_y_acc = masked_coord_acc(pred_loc_y, y_loc_xy[..., 1], key_padding_mask)
        else:
            zero = torch.tensor(0.0, device=pred_loc_x.device)
            loc_x_rmse = zero
            loc_y_rmse = zero
            loc_x_acc = zero
            loc_y_acc = zero

        turn_acc = masked_action_acc(pred_turn, y_turn, key_padding_mask)
        step_acc = masked_action_acc(pred_step, y_step, key_padding_mask)

        return {
            "mse": mse,
            "rmse": rmse,
            "lr_rmse": lr_rmse,
            "lr_acc": lr_acc,
            "loc_x_rmse": loc_x_rmse,
            "loc_y_rmse": loc_y_rmse,
            "loc_x_acc": loc_x_acc,
            "loc_y_acc": loc_y_acc,
            "turn_acc": turn_acc,
            "step_acc": step_acc,
        }

    def compute_all_losses(
        self,
        *,
        pred_sensor,
        pred_loc_x: torch.Tensor,
        pred_loc_y: torch.Tensor,
        pred_head: torch.Tensor,
        pred_turn: torch.Tensor,
        pred_step: torch.Tensor,
        y_sensor: torch.Tensor,
        y_sensor_idx: torch.Tensor,
        y_loc_xy: torch.Tensor,
        y_head: torch.Tensor,
        y_turn: torch.Tensor,
        y_step: torch.Tensor,
        key_padding_mask: torch.Tensor,
        sensor_tables,
        sensor_min_idx,
        loc_x_table: torch.Tensor,
        loc_y_table: torch.Tensor,
        heading_table: torch.Tensor,
        sensor_weight: float,
        loc_weight: float,
        head_weight: float,
        turn_weight: float,
        step_weight: float,
        aux_inputs: Optional[Dict[str, torch.Tensor]] = None,
        aux_total: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        obs_losses = self.compute_observation_losses(
            pred_sensor=pred_sensor,
            pred_loc_x=pred_loc_x,
            pred_loc_y=pred_loc_y,
            pred_head=pred_head,
            y_sensor=y_sensor,
            y_sensor_idx=y_sensor_idx,
            y_loc_xy=y_loc_xy,
            y_head=y_head,
            key_padding_mask=key_padding_mask,
            sensor_tables=sensor_tables,
            sensor_min_idx=sensor_min_idx,
            loc_x_table=loc_x_table,
            loc_y_table=loc_y_table,
            heading_table=heading_table,
            sensor_weight=sensor_weight,
            loc_weight=loc_weight,
            head_weight=head_weight,
            aux_inputs=aux_inputs,
        )
        action_losses = self.compute_action_losses(
            pred_turn=pred_turn,
            pred_step=pred_step,
            y_turn=y_turn,
            y_step=y_step,
        )
        if aux_total is None:
            aux_total = torch.tensor(0.0, device=pred_loc_x.device)
        obs_total = obs_losses["obs_total"]
        total = (
            obs_total
            + turn_weight * action_losses["turn"]
            + step_weight * action_losses["step"]
            + aux_total
        )
        return {
            "turn": action_losses["turn"],
            "step": action_losses["step"],
            "aux_total": aux_total,
            "obs_total": obs_total,
            "total": total,
        }


class UnifiedPredictor(PredictionLossMixin, nn.Module):
    def __init__(
        self,
        config: LlamaConfig,
        *,
        sensor_mode: str,
        sensor_dim: int,
        sensor_bins: Optional[np.ndarray],
        loc_x_bins: int,
        loc_y_bins: int,
        heading_dim: int,
        turn_bins: int,
        step_bins: int,
        obs_dim: int,
        obs_latent_dim: int,
        probe_hidden_dim: int = 0,
    ):
        super().__init__()
        self.backbone = LlamaModel(config)
        self.sensor_mode = sensor_mode
        self.heading_dim = heading_dim
        self.turn_bins = turn_bins
        self.step_bins = step_bins
        self.input_size = config.input_size
        self.probe_hidden_dim = int(probe_hidden_dim)
        if self.probe_hidden_dim < 0:
            raise ValueError("probe_hidden_dim must be >= 0")
        if sensor_mode == "categorical":
            assert sensor_bins is not None and len(sensor_bins) == 3
            self.sensor_head_l = nn.Linear(config.hidden_size, int(sensor_bins[0]))
            self.sensor_head_f = nn.Linear(config.hidden_size, int(sensor_bins[1]))
            self.sensor_head_r = nn.Linear(config.hidden_size, int(sensor_bins[2]))
        else:
            self.sensor_head = nn.Linear(config.hidden_size, sensor_dim)
        self.loc_x_head = make_probe_head(config.hidden_size, loc_x_bins, self.probe_hidden_dim)
        self.loc_y_head = make_probe_head(config.hidden_size, loc_y_bins, self.probe_hidden_dim)
        self.heading_head = make_probe_head(config.hidden_size, heading_dim, self.probe_hidden_dim)
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, obs_latent_dim),
            nn.ReLU(),
            nn.Linear(obs_latent_dim, obs_latent_dim),
        )
        self.fuse = nn.Linear(config.hidden_size + obs_latent_dim, config.hidden_size)
        self.turn_head = nn.Linear(config.hidden_size, turn_bins)
        self.step_head = nn.Linear(config.hidden_size, step_bins)

    def forward(
        self,
        obs,
        attention_window=None,
        past_key_values=None,
        cache_position=None,
        prev_hidden=None,
        return_state=False,
    ):
        sensor = obs["sensor"]
        actions = obs["actions"]
        key_padding_mask = obs.get("key_padding_mask")

        x = torch.cat([sensor, actions], dim=-1)
        if x.shape[-1] != self.input_size:
            raise ValueError(f"Expected input_size {self.input_size}, got {x.shape[-1]}")

        h = self.backbone(
            x,
            past_key_values=past_key_values,
            cache_position=cache_position,
            key_padding_mask=key_padding_mask,
            attention_window=attention_window,
        )
        if self.sensor_mode != "categorical":
            raise ValueError("UnifiedPredictor currently supports sensor_mode='categorical' only.")
        pred_sensor = (
            self.sensor_head_l(h),
            self.sensor_head_f(h),
            self.sensor_head_r(h),
        )
        # Probe heads should not shape backbone dynamics.
        h_probe = h.detach()
        loc_x = self.loc_x_head(h_probe)
        loc_y = self.loc_y_head(h_probe)
        heading = self.heading_head(h_probe)
        z = self.obs_encoder(sensor)
        # For chunked training, the first action in a chunk uses previous chunk state.
        if prev_hidden is None:
            prev_hidden = torch.zeros((h.shape[0], 1, h.shape[-1]), device=h.device, dtype=h.dtype)
        elif prev_hidden.dim() == 2:
            prev_hidden = prev_hidden.unsqueeze(1)
        h_prev = torch.cat([prev_hidden, h[:, :-1, :]], dim=1)
        fused = torch.tanh(self.fuse(torch.cat([h_prev, z], dim=-1)))
        turn = self.turn_head(fused)
        step = self.step_head(fused)
        if return_state:
            return pred_sensor, loc_x, loc_y, heading, turn, step, h[:, -1, :]
        return pred_sensor, loc_x, loc_y, heading, turn, step

    def compute_all_losses(self, **kwargs) -> Dict[str, torch.Tensor]:
        return super().compute_all_losses(aux_total=None, **kwargs)


class RNNPredictor(UnifiedPredictor):
    def __init__(
        self,
        config: LlamaConfig,
        *,
        sensor_mode: str,
        sensor_dim: int,
        sensor_bins: Optional[np.ndarray],
        loc_x_bins: int,
        loc_y_bins: int,
        heading_dim: int,
        turn_bins: int,
        step_bins: int,
        obs_dim: int,
        obs_latent_dim: int,
        probe_hidden_dim: int = 0,
    ):
        super().__init__(
            config,
            sensor_mode=sensor_mode,
            sensor_dim=sensor_dim,
            sensor_bins=sensor_bins,
            loc_x_bins=loc_x_bins,
            loc_y_bins=loc_y_bins,
            heading_dim=heading_dim,
            turn_bins=turn_bins,
            step_bins=step_bins,
            obs_dim=obs_dim,
            obs_latent_dim=obs_latent_dim,
            probe_hidden_dim=probe_hidden_dim,
        )
        self.backbone = nn.GRU(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_hidden_layers,
            batch_first=True,
        )

    def forward(
        self,
        obs,
        attention_window=None,
        past_key_values=None,
        cache_position=None,
        prev_hidden=None,
        return_state=False,
    ):
        del attention_window, past_key_values, cache_position
        sensor = obs["sensor"]
        actions = obs["actions"]

        x = torch.cat([sensor, actions], dim=-1)
        if x.shape[-1] != self.input_size:
            raise ValueError(f"Expected input_size {self.input_size}, got {x.shape[-1]}")

        h, _ = self.backbone(x)
        if self.sensor_mode != "categorical":
            raise ValueError("RNNPredictor currently supports sensor_mode='categorical' only.")
        pred_sensor = (
            self.sensor_head_l(h),
            self.sensor_head_f(h),
            self.sensor_head_r(h),
        )
        # Probe heads should not shape backbone dynamics.
        h_probe = h.detach()
        loc_x = self.loc_x_head(h_probe)
        loc_y = self.loc_y_head(h_probe)
        heading = self.heading_head(h_probe)
        z = self.obs_encoder(sensor)
        if prev_hidden is None:
            prev_hidden = torch.zeros((h.shape[0], 1, h.shape[-1]), device=h.device, dtype=h.dtype)
        elif prev_hidden.dim() == 2:
            prev_hidden = prev_hidden.unsqueeze(1)
        h_prev = torch.cat([prev_hidden, h[:, :-1, :]], dim=1)
        fused = torch.tanh(self.fuse(torch.cat([h_prev, z], dim=-1)))
        turn = self.turn_head(fused)
        step = self.step_head(fused)
        if return_state:
            return pred_sensor, loc_x, loc_y, heading, turn, step, h[:, -1, :]
        return pred_sensor, loc_x, loc_y, heading, turn, step


class DiscreteLatentPredictorBase(PredictionLossMixin, nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        sensor_mode: str,
        sensor_dim: int,
        sensor_bins: Optional[np.ndarray],
        loc_x_bins: int,
        loc_y_bins: int,
        heading_dim: int,
        turn_bins: int,
        step_bins: int,
        obs_dim: int,
        obs_latent_dim: int,
        action_dim: int,
        stoch_size: int,
        stoch_classes: int,
        stoch_temp: float,
        kl_dyn_beta: float,
        kl_rep_beta: float,
        kl_free_nats: float,
        recon_beta: float,
        obs_loss_mode: str,
        prior_rollout_weight: float,
        bptt_horizon: int,
        z_only_weight: float,
        h_only_weight: float,
        prior_rollout_steps: int = 0,
        probe_hidden_dim: int = 0,
    ):
        super().__init__()
        self.sensor_mode = sensor_mode
        self.sensor_dim = int(sensor_dim)
        self.sensor_bins = sensor_bins
        self.heading_dim = heading_dim
        self.hidden_size = int(hidden_size)
        self.action_dim = int(action_dim)
        self.stoch_size = int(stoch_size)
        self.stoch_classes = int(stoch_classes)
        self.stoch_flat = self.stoch_size * self.stoch_classes
        self.stoch_temp = float(stoch_temp)
        self.kl_dyn_beta = float(kl_dyn_beta)
        self.kl_rep_beta = float(kl_rep_beta)
        self.kl_free_nats = float(kl_free_nats)
        self.recon_beta = float(recon_beta)
        self.prior_rollout_weight = float(prior_rollout_weight)
        self.bptt_horizon = int(bptt_horizon)
        self.z_only_weight = float(z_only_weight)
        self.h_only_weight = float(h_only_weight)
        self.prior_rollout_steps = int(prior_rollout_steps)
        self.probe_hidden_dim = int(probe_hidden_dim)
        if self.prior_rollout_steps < 0:
            raise ValueError("prior_rollout_steps must be >= 0")
        if self.probe_hidden_dim < 0:
            raise ValueError("probe_hidden_dim must be >= 0")
        if obs_loss_mode not in {"soft", "recon"}:
            raise ValueError(f"obs_loss_mode must be 'soft' or 'recon', got {obs_loss_mode}")
        self.obs_loss_mode = obs_loss_mode
        self.loc_x_bins = int(loc_x_bins)
        self.loc_y_bins = int(loc_y_bins)

        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, obs_latent_dim),
            nn.ReLU(),
            nn.Linear(obs_latent_dim, obs_latent_dim),
            nn.ReLU(),
        )
        self.prior_head = nn.Linear(self.hidden_size, self.stoch_flat)
        self.post_head = nn.Linear(self.hidden_size + obs_latent_dim, self.stoch_flat)

        feat_dim = self.hidden_size + self.stoch_flat
        self.z_obs_head: Optional[nn.Module] = None
        self.h_obs_head: Optional[nn.Module] = None
        if self.obs_loss_mode == "soft":
            if sensor_mode == "categorical":
                assert sensor_bins is not None and len(sensor_bins) == 3
                sensor_out_dim = int(sensor_bins[0] + sensor_bins[1] + sensor_bins[2])
            else:
                sensor_out_dim = self.sensor_dim
            # Sensor prediction head only; location/heading are probe heads.
            self.obs_head = nn.Linear(feat_dim, sensor_out_dim)
            # Optional z-only sensor head for auxiliary loss.
            self.z_obs_head = nn.Linear(self.stoch_flat, sensor_out_dim)
            # Optional h-only sensor head for auxiliary loss.
            self.h_obs_head = nn.Linear(self.hidden_size, sensor_out_dim)
        else:
            # Recon mode decoder outputs continuous sensor observation.
            self.obs_head = nn.Sequential(
                nn.Linear(feat_dim, obs_latent_dim),
                nn.ReLU(),
                nn.Linear(obs_latent_dim, obs_dim),
            )
        # Probe heads for location/heading: trained from detached state so they don't shape dynamics.
        self.loc_probe_x = make_probe_head(feat_dim, self.loc_x_bins, self.probe_hidden_dim)
        self.loc_probe_y = make_probe_head(feat_dim, self.loc_y_bins, self.probe_hidden_dim)
        self.head_probe = make_probe_head(feat_dim, self.heading_dim, self.probe_hidden_dim)
        self.turn_head = nn.Linear(feat_dim, turn_bins)
        self.step_head = nn.Linear(feat_dim, step_bins)

    def _sample_stoch(self, logits: torch.Tensor, training: bool) -> torch.Tensor:
        # logits: [B, T, S, C]
        if training:
            flat = logits.reshape(-1, self.stoch_classes)
            z = F.gumbel_softmax(flat, tau=self.stoch_temp, hard=True, dim=-1)
            return z.reshape(*logits.shape)
        probs = F.softmax(logits, dim=-1)
        idx = probs.argmax(dim=-1)
        return F.one_hot(idx, num_classes=self.stoch_classes).to(dtype=probs.dtype)

    def _masked_mean(self, x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        if mask is None:
            return x.mean()
        valid = ~mask
        if valid.sum() == 0:
            return torch.tensor(0.0, device=x.device, dtype=x.dtype)
        return x[valid].mean()

    def _compute_kl_losses(
        self,
        prior_logits: torch.Tensor,  # [B, T, S, C]
        post_logits: torch.Tensor,   # [B, T, S, C]
        key_padding_mask: Optional[torch.Tensor],  # [B, T], True for PAD
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute L_dyn - dynamic prediction loss
        eq 4 in twister paper
        """
        # prior - hat z distribution
        log_p = F.log_softmax(prior_logits, dim=-1)
        # posterior
        log_q = F.log_softmax(post_logits, dim=-1)

        q = log_q.exp()

        # make prior match posterior
        kl_dyn = (sg(q) * (sg(log_q) - log_p)).sum(dim=-1).sum(dim=-1)  # [B, T]

        # make obs -> q representation more predictable by matching prior
        kl_rep = (q * (log_q - sg(log_p))).sum(dim=-1).sum(dim=-1)  # [B, T]

        if self.kl_free_nats > 0:
            # same as max(KL, 1)
            kl_dyn = torch.clamp(kl_dyn - self.kl_free_nats, min=0)
            kl_rep = torch.clamp(kl_rep - self.kl_free_nats, min=0)

        dyn_loss = self.kl_dyn_beta * self._masked_mean(kl_dyn, key_padding_mask)
        rep_loss = self.kl_rep_beta * self._masked_mean(kl_rep, key_padding_mask)
        return dyn_loss, rep_loss

    def compute_observation_losses(
        self,
        *,
        pred_sensor=None,
        pred_loc_x: Optional[torch.Tensor] = None,
        pred_loc_y: Optional[torch.Tensor] = None,
        pred_head: Optional[torch.Tensor] = None,
        y_sensor: Optional[torch.Tensor] = None,
        y_sensor_idx: Optional[torch.Tensor] = None,
        y_loc_xy: Optional[torch.Tensor] = None,
        y_head: Optional[torch.Tensor] = None,
        key_padding_mask: torch.Tensor,
        sensor_tables=None,
        sensor_min_idx=None,
        loc_x_table: Optional[torch.Tensor] = None,
        loc_y_table: Optional[torch.Tensor] = None,
        heading_table: Optional[torch.Tensor] = None,
        sensor_weight: float = 1.0,
        loc_weight: float = 1.0,
        head_weight: float = 1.0,
        aux_inputs: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        if self.obs_loss_mode == "recon":
            # Recon mode: single decoder head outputs a continuous obs vector.
            # We use MSE against a mixed target (continuous loc/sensor + one-hot heading).
            if aux_inputs is None or "obs_hat" not in aux_inputs or "obs_target" not in aux_inputs:
                raise ValueError("aux_inputs['obs_hat'] and aux_inputs['obs_target'] are required for reconstruction loss")
            obs_hat = aux_inputs["obs_hat"]
            obs_target = aux_inputs["obs_target"].to(obs_hat.dtype)
            # Note: unlike soft-target losses, this treats heading as a continuous one-hot vector.
            recon = masked_mse(obs_hat, obs_target, key_padding_mask)
            return {"obs_total": self.recon_beta * recon}

        if aux_inputs is None or "sensor_target" not in aux_inputs or "loc_target" not in aux_inputs or "head_target" not in aux_inputs:
            raise ValueError("aux_inputs targets are required for soft-mode observation loss")
        sensor_target = aux_inputs["sensor_target"]
        loc_target = aux_inputs["loc_target"]
        head_target = aux_inputs["head_target"]

        if self.sensor_mode == "categorical":
            pred_l, pred_f, pred_r = pred_sensor
            idx = sensor_target.round().to(torch.long).clamp(min=0)
            if sensor_min_idx is not None:
                idx = (idx - sensor_min_idx.view(1, 1, -1)).clamp(min=0)
            loss_sensor = (
                soft_cross_entropy(pred_l, sensor_tables[0][idx[..., 0]], key_padding_mask)
                + soft_cross_entropy(pred_f, sensor_tables[1][idx[..., 1]], key_padding_mask)
                + soft_cross_entropy(pred_r, sensor_tables[2][idx[..., 2]], key_padding_mask)
            )
        else:
            loss_sensor = masked_mse(pred_sensor, sensor_target, key_padding_mask)

        loc_idx = loc_target.round().to(torch.long).clamp(min=0)
        loss_loc_x = soft_cross_entropy(pred_loc_x, loc_x_table[loc_idx[..., 0]], key_padding_mask)
        loss_loc_y = soft_cross_entropy(pred_loc_y, loc_y_table[loc_idx[..., 1]], key_padding_mask)
        head_idx = head_target.clamp(min=0)
        loss_head = soft_cross_entropy(pred_head, heading_table[head_idx], key_padding_mask)

        obs_total = (
            sensor_weight * loss_sensor
            + loc_weight * (loss_loc_x + loss_loc_y)
            + head_weight * loss_head
        )
        return {"obs_total": obs_total}

    def compute_metrics(
        self,
        *,
        pred_sensor,
        pred_loc_x: torch.Tensor,
        pred_loc_y: torch.Tensor,
        pred_turn: torch.Tensor,
        pred_step: torch.Tensor,
        y_sensor: torch.Tensor,
        y_sensor_idx: torch.Tensor,
        y_loc_xy: torch.Tensor,
        y_turn: torch.Tensor,
        y_step: torch.Tensor,
        key_padding_mask: torch.Tensor,
        sensor_min_idx,
        loc_min: Optional[torch.Tensor],
        aux_inputs: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        if aux_inputs is None or "sensor_target" not in aux_inputs or "loc_target" not in aux_inputs or "head_target" not in aux_inputs:
            return super().compute_metrics(
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
                key_padding_mask=key_padding_mask,
                sensor_min_idx=sensor_min_idx,
                loc_min=loc_min,
                aux_inputs=aux_inputs,
            )

        sensor_target = aux_inputs["sensor_target"]
        loc_target = aux_inputs["loc_target"]

        sensor_for_lr = aux_inputs.get("prior_sensor_pred", pred_sensor)
        if self.sensor_mode == "categorical":
            pred_l, pred_f, pred_r = pred_sensor
            lr_l, _lr_f, lr_r = sensor_for_lr
            sensor_target_abs = sensor_target.round().to(torch.long).clamp(min=0)
            idx = sensor_target_abs
            if sensor_min_idx is not None:
                idx = (idx - sensor_min_idx.view(1, 1, -1)).clamp(min=0)
                sensor_min_metric = sensor_min_idx.to(torch.float32)
            else:
                sensor_min_metric = torch.zeros(3, device=pred_l.device, dtype=torch.float32)
            exp_l = expected_from_logits(pred_l, float(sensor_min_metric[0].item()))
            exp_f = expected_from_logits(pred_f, float(sensor_min_metric[1].item()))
            exp_r = expected_from_logits(pred_r, float(sensor_min_metric[2].item()))
            pred_sensor_exp = torch.stack([exp_l, exp_f, exp_r], dim=-1)
            target_sensor_cont = sensor_target_abs.to(pred_sensor_exp.dtype)
            mse = masked_mse(pred_sensor_exp, target_sensor_cont, key_padding_mask)
            rmse = masked_rmse(pred_sensor_exp, target_sensor_cont, key_padding_mask)
            lr_rmse, lr_acc = masked_lr_metrics_logits(
                lr_l, lr_r, sensor_target_abs, key_padding_mask, sensor_min_metric
            )
        else:
            mse = masked_mse(pred_sensor, sensor_target, key_padding_mask)
            rmse = masked_rmse(pred_sensor, sensor_target, key_padding_mask)
            lr_rmse, lr_acc = masked_lr_metrics(sensor_for_lr, sensor_target, key_padding_mask)

        if loc_min is not None:
            loc_x_idx = loc_target[..., 0].to(torch.long)
            loc_y_idx = loc_target[..., 1].to(torch.long)
            loc_x_rmse = masked_coord_rmse(pred_loc_x, loc_x_idx, key_padding_mask, loc_min[0])
            loc_y_rmse = masked_coord_rmse(pred_loc_y, loc_y_idx, key_padding_mask, loc_min[1])
            loc_x_acc = masked_coord_acc(pred_loc_x, loc_x_idx, key_padding_mask)
            loc_y_acc = masked_coord_acc(pred_loc_y, loc_y_idx, key_padding_mask)
        else:
            zero = torch.tensor(0.0, device=pred_loc_x.device)
            loc_x_rmse = zero
            loc_y_rmse = zero
            loc_x_acc = zero
            loc_y_acc = zero

        turn_acc = masked_action_acc(pred_turn, y_turn, key_padding_mask)
        step_acc = masked_action_acc(pred_step, y_step, key_padding_mask)

        return {
            "mse": mse,
            "rmse": rmse,
            "lr_rmse": lr_rmse,
            "lr_acc": lr_acc,
            "loc_x_rmse": loc_x_rmse,
            "loc_y_rmse": loc_y_rmse,
            "loc_x_acc": loc_x_acc,
            "loc_y_acc": loc_y_acc,
            "turn_acc": turn_acc,
            "step_acc": step_acc,
        }

    def compute_all_losses(
        self,
        *,
        pred_sensor,
        pred_loc_x: torch.Tensor,
        pred_loc_y: torch.Tensor,
        pred_head: torch.Tensor,
        pred_turn: torch.Tensor,
        pred_step: torch.Tensor,
        y_sensor: torch.Tensor,
        y_sensor_idx: torch.Tensor,
        y_loc_xy: torch.Tensor,
        y_head: torch.Tensor,
        y_turn: torch.Tensor,
        y_step: torch.Tensor,
        key_padding_mask: torch.Tensor,
        sensor_tables,
        sensor_min_idx,
        loc_x_table: torch.Tensor,
        loc_y_table: torch.Tensor,
        heading_table: torch.Tensor,
        sensor_weight: float,
        loc_weight: float,
        head_weight: float,
        turn_weight: float,
        step_weight: float,
        aux_inputs: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        def compute_sensor_loss(pred_sensor, target_idx, target_raw, mask):
            """Compute sensor loss for either categorical or regression heads."""
            if pred_sensor is None:
                return torch.tensor(0.0, device=pred_loc_x.device)
            if self.sensor_mode == "categorical":
                pred_l, pred_f, pred_r = pred_sensor
                idx = target_idx.clamp(min=0)
                if sensor_min_idx is not None:
                    idx = (idx - sensor_min_idx.view(1, 1, -1)).clamp(min=0)
                return (
                    soft_cross_entropy(pred_l, sensor_tables[0][idx[..., 0]], mask)
                    + soft_cross_entropy(pred_f, sensor_tables[1][idx[..., 1]], mask)
                    + soft_cross_entropy(pred_r, sensor_tables[2][idx[..., 2]], mask)
                )
            return masked_mse(pred_sensor, target_raw, mask)

        def compute_prior_rollout_sensor_loss(prior_roll_sensor_pred, sensor_target):
            if prior_roll_sensor_pred is None:
                return torch.tensor(0.0, device=pred_loc_x.device)
            if isinstance(prior_roll_sensor_pred, tuple):
                pred_len = int(prior_roll_sensor_pred[0].shape[1])
            else:
                pred_len = int(prior_roll_sensor_pred.shape[1])
            target_raw = sensor_target[:, :pred_len]
            target_idx = target_raw.round().to(torch.long).clamp(min=0)
            mask = key_padding_mask[:, :pred_len]
            return compute_sensor_loss(prior_roll_sensor_pred, target_idx, target_raw, mask)

        def compute_z_only_sensor_loss(z_only_pred, target_raw, target_idx):
            return compute_sensor_loss(z_only_pred, target_idx, target_raw, key_padding_mask)

        def compute_h_only_sensor_loss(h_only_pred, target_raw, target_idx):
            return compute_sensor_loss(h_only_pred, target_idx, target_raw, key_padding_mask)

        aux_losses: Dict[str, torch.Tensor] = {}
        aux_total = torch.tensor(0.0, device=pred_loc_x.device)
        if aux_inputs is not None:
            prior_logits = aux_inputs.get("prior_logits")
            post_logits = aux_inputs.get("post_logits")
            if prior_logits is not None and post_logits is not None:
                kl_dyn, kl_rep = self._compute_kl_losses(prior_logits, post_logits, key_padding_mask)
                aux_losses["kl_dyn"] = kl_dyn
                aux_losses["kl_rep"] = kl_rep
                aux_total = aux_total + kl_dyn + kl_rep
            prior_roll_sensor_pred = aux_inputs.get("prior_roll_sensor_pred")
            sensor_target = aux_inputs.get("sensor_target")
            if self.prior_rollout_weight > 0 and prior_roll_sensor_pred is not None and sensor_target is not None:
                prior_rollout_sensor = compute_prior_rollout_sensor_loss(prior_roll_sensor_pred, sensor_target)
                aux_losses["prior_rollout_sensor"] = prior_rollout_sensor
                aux_total = aux_total + self.prior_rollout_weight * prior_rollout_sensor
            sensor_recon_target = aux_inputs.get("sensor_target", y_sensor)
            sensor_recon_target_idx = sensor_recon_target.round().to(torch.long).clamp(min=0)
            z_only_pred = aux_inputs.get("z_only_pred")
            if self.z_only_weight > 0 and z_only_pred is not None:
                z_only_sensor = compute_z_only_sensor_loss(
                    z_only_pred,
                    sensor_recon_target,
                    sensor_recon_target_idx,
                )
                aux_losses["z_only_sensor"] = z_only_sensor
                aux_total = aux_total + self.z_only_weight * z_only_sensor
            h_only_pred = aux_inputs.get("h_only_pred")
            if self.h_only_weight > 0 and h_only_pred is not None:
                h_only_sensor = compute_h_only_sensor_loss(
                    h_only_pred,
                    sensor_recon_target,
                    sensor_recon_target_idx,
                )
                aux_losses["h_only_sensor"] = h_only_sensor
                aux_total = aux_total + self.h_only_weight * h_only_sensor
        base_losses = super().compute_all_losses(
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
            key_padding_mask=key_padding_mask,
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
            aux_total=aux_total,
            aux_inputs=aux_inputs,
        )
        if self.obs_loss_mode == "recon":
            base_losses["recon"] = base_losses["obs_total"]
        return {**base_losses, **aux_losses}

    def _encode_obs(self, obs):
        sensor = obs["sensor"][..., :3]      # use only left/front/right sensors
        loc = obs["loc"]                     # [B, T, 2]
        heading = obs["heading"]             # [B, T]
        actions = obs["actions"]             # [B, T, A]
        key_padding_mask = obs.get("key_padding_mask")
        if actions.ndim != 3:
            raise ValueError(f"{self.__class__.__name__} expects batched sequences [B, T, A].")
        B, T, _ = actions.shape
        obs_features = sensor
        obs_embed = self.obs_encoder(obs_features)
        return obs_embed, obs_features, actions, key_padding_mask, B, T

    @staticmethod
    def _build_prev_actions(actions: torch.Tensor) -> torch.Tensor:
        a_prev = torch.zeros_like(actions)
        if actions.shape[1] > 1:
            a_prev[:, 1:, :] = actions[:, :-1, :]
        return a_prev

    def _split_sensor_logits(self, obs_out: torch.Tensor):
        """Map flattened sensor logits into (left, front, right) heads."""
        if self.sensor_mode == "categorical":
            assert self.sensor_bins is not None
            b0 = int(self.sensor_bins[0])
            b1 = int(self.sensor_bins[1])
            b2 = int(self.sensor_bins[2])
            pred_l = obs_out[..., :b0]
            pred_f = obs_out[..., b0 : b0 + b1]
            pred_r = obs_out[..., b0 + b1 : b0 + b1 + b2]
            return (pred_l, pred_f, pred_r)
        return obs_out

    def _decode_sensor_from_feat(self, feat: torch.Tensor):
        """Decode sensors from full feature [h, z] (main prediction path)."""
        if self.obs_loss_mode == "soft":
            return self._split_sensor_logits(self.obs_head(feat))
        # Recon mode decoder outputs a continuous sensor observation.
        return self.obs_head(feat)

    def _decode_sensor_from_z(self, z_flat: torch.Tensor):
        """Decode sensors from z only (auxiliary path to encourage latent usage)."""
        if self.obs_loss_mode != "soft" or self.z_obs_head is None:
            return None
        return self._split_sensor_logits(self.z_obs_head(z_flat.detach()))

    def _decode_sensor_from_h(self, h_flat: torch.Tensor):
        """Decode sensors from h only (auxiliary probe of deterministic state)."""
        if self.obs_loss_mode != "soft" or self.h_obs_head is None:
            return None
        return self._split_sensor_logits(self.h_obs_head(h_flat.detach()))

    def _decode_feat(self, feat: torch.Tensor):
        """Decode sensors/actions from full feature; probes use detached features."""
        turn = self.turn_head(feat)
        step = self.step_head(feat)
        feat_detached = feat.detach()
        loc_x = self.loc_probe_x(feat_detached)
        loc_y = self.loc_probe_y(feat_detached)
        heading_out = self.head_probe(feat_detached)
        pred_sensor = self._decode_sensor_from_feat(feat)
        if self.obs_loss_mode == "soft":
            return (pred_sensor, loc_x, loc_y, heading_out, turn, step), None
        return (pred_sensor, loc_x, loc_y, heading_out, turn, step), pred_sensor


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


class RSSMDiscretePredictor(DiscreteLatentPredictorBase):
    def __init__(
        self,
        *,
        hidden_size: int,
        sensor_mode: str,
        sensor_dim: int,
        sensor_bins: Optional[np.ndarray],
        loc_x_bins: int,
        loc_y_bins: int,
        heading_dim: int,
        turn_bins: int,
        step_bins: int,
        obs_dim: int,
        obs_latent_dim: int,
        action_dim: int,
        stoch_size: int,
        stoch_classes: int,
        stoch_temp: float,
        kl_dyn_beta: float,
        kl_rep_beta: float,
        kl_free_nats: float,
        recon_beta: float,
        obs_loss_mode: str,
        prior_rollout_weight: float,
        bptt_horizon: int,
        z_only_weight: float,
        h_only_weight: float,
        prior_rollout_steps: int = 0,
        probe_hidden_dim: int = 0,
    ):
        super().__init__(
            hidden_size=hidden_size,
            sensor_mode=sensor_mode,
            sensor_dim=sensor_dim,
            sensor_bins=sensor_bins,
            loc_x_bins=loc_x_bins,
            loc_y_bins=loc_y_bins,
            heading_dim=heading_dim,
            turn_bins=turn_bins,
            step_bins=step_bins,
            obs_dim=obs_dim,
            obs_latent_dim=obs_latent_dim,
            action_dim=action_dim,
            stoch_size=stoch_size,
            stoch_classes=stoch_classes,
            stoch_temp=stoch_temp,
            kl_dyn_beta=kl_dyn_beta,
            kl_rep_beta=kl_rep_beta,
            kl_free_nats=kl_free_nats,
            recon_beta=recon_beta,
            obs_loss_mode=obs_loss_mode,
            prior_rollout_weight=prior_rollout_weight,
            bptt_horizon=bptt_horizon,
            z_only_weight=z_only_weight,
            h_only_weight=h_only_weight,
            prior_rollout_steps=prior_rollout_steps,
            probe_hidden_dim=probe_hidden_dim,
        )
        self.rnn = nn.GRUCell(self.stoch_flat + self.action_dim, self.hidden_size)

    def forward(
        self,
        obs,
        attention_window=None,
        return_state=False,
    ):
        del attention_window
        obs_embed, obs_features, actions, key_padding_mask, B, T = self._encode_obs(obs)
        a_prev = self._build_prev_actions(actions)
        h_prev = actions.new_zeros((B, self.hidden_size))
        z_prev_flat = actions.new_zeros((B, self.stoch_flat))
        h_init = h_prev
        z_init_flat = z_prev_flat

        prior_logits_steps = []
        post_logits_steps = []
        feat_steps = []
        feat_prior_steps = []
        z_only_steps = []
        h_only_steps = []
        for t in range(T):
            if self.bptt_horizon > 0 and t > 0 and (t % self.bptt_horizon) == 0:
                h_prev = h_prev.detach()
                z_prev_flat = z_prev_flat.detach()
            h_t = self.rnn(torch.cat([z_prev_flat, a_prev[:, t, :]], dim=-1), h_prev)
            prior_logits_t = self.prior_head(h_t).view(B, self.stoch_size, self.stoch_classes)
            z_prior_t = self._sample_stoch(prior_logits_t.unsqueeze(1), self.training).squeeze(1)
            z_prior_flat = z_prior_t.reshape(B, self.stoch_flat)
            # Detach h_t so posterior gradients don't flow into the deterministic state.
            post_logits_t = self.post_head(torch.cat([h_t.detach(), obs_embed[:, t, :]], dim=-1)).view(
                B, self.stoch_size, self.stoch_classes
            )
            z_t = self._sample_stoch(post_logits_t.unsqueeze(1), self.training).squeeze(1)
            z_t_flat = z_t.reshape(B, self.stoch_flat)

            feat_steps.append(torch.cat([h_t, z_t_flat], dim=-1))
            feat_prior_steps.append(torch.cat([h_t, z_prior_flat], dim=-1))
            z_only_steps.append(z_t_flat)
            h_only_steps.append(h_t)
            prior_logits_steps.append(prior_logits_t)
            post_logits_steps.append(post_logits_t)

            h_prev = h_t
            z_prev_flat = z_t_flat

        feat = torch.stack(feat_steps, dim=1)                 # [B, T, H + S*C]
        feat_prior = torch.stack(feat_prior_steps, dim=1)     # [B, T, H + S*C]
        prior_logits = torch.stack(prior_logits_steps, dim=1) # [B, T, S, C]
        post_logits = torch.stack(post_logits_steps, dim=1)   # [B, T, S, C]

        outputs, obs_hat = self._decode_feat(feat)
        prior_sensor_pred = self._decode_sensor_from_feat(feat_prior)
        z_only_pred = None
        if z_only_steps:
            z_only_pred = self._decode_sensor_from_z(torch.stack(z_only_steps, dim=1))
        h_only_pred = None
        if h_only_steps:
            h_only_pred = self._decode_sensor_from_h(torch.stack(h_only_steps, dim=1))
        prior_roll_sensor_pred = None
        if self.prior_rollout_weight > 0:
            h_roll = h_init
            z_roll_flat = z_init_flat
            feat_roll_steps = []
            roll_T = T if self.prior_rollout_steps <= 0 else min(T, self.prior_rollout_steps)
            for t in range(roll_T):
                if self.bptt_horizon > 0 and t > 0 and (t % self.bptt_horizon) == 0:
                    h_roll = h_roll.detach()
                    z_roll_flat = z_roll_flat.detach()
                h_roll = self.rnn(torch.cat([z_roll_flat, a_prev[:, t, :]], dim=-1), h_roll)
                prior_logits_roll_t = self.prior_head(h_roll).view(B, self.stoch_size, self.stoch_classes)
                z_roll_t = self._sample_stoch(prior_logits_roll_t.unsqueeze(1), self.training).squeeze(1)
                z_roll_flat = z_roll_t.reshape(B, self.stoch_flat)
                feat_roll_steps.append(torch.cat([h_roll, z_roll_flat], dim=-1))
            if feat_roll_steps:
                feat_roll = torch.stack(feat_roll_steps, dim=1)
                prior_roll_sensor_pred = self._decode_sensor_from_feat(feat_roll)
        aux_inputs = {
            "prior_logits": prior_logits,
            "post_logits": post_logits,
            "feat": feat,
            "obs_target": obs_features,
            "sensor_target": obs["sensor"],
            "loc_target": obs["loc"],
            "head_target": obs["heading"],
            "prior_sensor_pred": prior_sensor_pred,
            "prior_roll_sensor_pred": prior_roll_sensor_pred,
            "z_only_pred": z_only_pred,
            "h_only_pred": h_only_pred,
        }
        if obs_hat is not None:
            aux_inputs["obs_hat"] = obs_hat
        if return_state:
            last_state = torch.cat([h_prev, z_prev_flat], dim=-1)
            return (*outputs, aux_inputs, last_state)
        return (*outputs, aux_inputs)


class TSSMDiscretePredictor(DiscreteLatentPredictorBase):
    """Transformer state-space model with discrete stochastic latents.

    Deterministic transition is autoregressive in latent-action space:
      h_t = f_theta(h_{<t}, z_{<t}, a_{<t})
    while stochastic state uses stacked categoricals with two-sided KL.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        layers: int,
        heads: int,
        head_dim: int,
        intermediate: int,
        attention_window: Optional[int],
        sensor_mode: str,
        sensor_dim: int,
        sensor_bins: Optional[np.ndarray],
        loc_x_bins: int,
        loc_y_bins: int,
        heading_dim: int,
        turn_bins: int,
        step_bins: int,
        obs_dim: int,
        obs_latent_dim: int,
        action_dim: int,
        stoch_size: int,
        stoch_classes: int,
        stoch_temp: float,
        kl_dyn_beta: float,
        kl_rep_beta: float,
        kl_free_nats: float,
        recon_beta: float,
        obs_loss_mode: str,
        prior_rollout_weight: float,
        bptt_horizon: int,
        z_only_weight: float,
        h_only_weight: float,
        prior_rollout_steps: int = 0,
        probe_hidden_dim: int = 0,
    ):
        if int(hidden_size) != int(heads) * int(head_dim):
            raise ValueError("For tssm, hidden_size must equal heads * head_dim")
        super().__init__(
            hidden_size=hidden_size,
            sensor_mode=sensor_mode,
            sensor_dim=sensor_dim,
            sensor_bins=sensor_bins,
            loc_x_bins=loc_x_bins,
            loc_y_bins=loc_y_bins,
            heading_dim=heading_dim,
            turn_bins=turn_bins,
            step_bins=step_bins,
            obs_dim=obs_dim,
            obs_latent_dim=obs_latent_dim,
            action_dim=action_dim,
            stoch_size=stoch_size,
            stoch_classes=stoch_classes,
            stoch_temp=stoch_temp,
            kl_dyn_beta=kl_dyn_beta,
            kl_rep_beta=kl_rep_beta,
            kl_free_nats=kl_free_nats,
            recon_beta=recon_beta,
            obs_loss_mode=obs_loss_mode,
            prior_rollout_weight=prior_rollout_weight,
            bptt_horizon=bptt_horizon,
            z_only_weight=z_only_weight,
            h_only_weight=h_only_weight,
            prior_rollout_steps=prior_rollout_steps,
            probe_hidden_dim=probe_hidden_dim,
        )
        self.attention_window = attention_window

        trans_cfg = LlamaConfig(
            input_size=self.stoch_flat + self.action_dim,
            hidden_size=self.hidden_size,
            intermediate_size=int(intermediate),
            num_hidden_layers=int(layers),
            num_attention_heads=int(heads),
            num_key_value_heads=int(heads),
            head_dim=int(head_dim),
            attention_window=attention_window,
            use_rope=True,
        )
        self.transition = LlamaModel(trans_cfg)

    def forward(
        self,
        obs,
        attention_window=None,
        return_state=False,
    ):
        obs_embed, obs_features, actions, key_padding_mask, B, T = self._encode_obs(obs)
        device = actions.device

        a_prev = self._build_prev_actions(actions)
        h_prev = actions.new_zeros((B, self.hidden_size))
        z_prev_flat = actions.new_zeros((B, self.stoch_flat))
        h_init = h_prev
        z_init_flat = z_prev_flat

        if attention_window is None:
            attention_window = self.attention_window
        detach_on_pop = bool(self.training and self.bptt_horizon > 0)
        if attention_window is not None and attention_window > 0:
            cache = WindowedPositionBasedDynamicCache(
                int(attention_window),
                detach_on_pop=detach_on_pop,
            ).to(device=device)
        else:
            cache = PositionBasedDynamicCache(detach_on_pop=detach_on_pop).to(device=device)

        pos = torch.arange(T, device=device, dtype=torch.long).unsqueeze(0).expand(B, T)

        prior_logits_steps = []
        post_logits_steps = []
        feat_steps = []
        feat_prior_steps = []
        z_only_steps = []
        h_only_steps = []
        for t in range(T):
            if self.bptt_horizon > 0 and t > 0 and (t % self.bptt_horizon) == 0:
                z_prev_flat = z_prev_flat.detach()
            trans_in_t = torch.cat([z_prev_flat, a_prev[:, t, :]], dim=-1).unsqueeze(1)
            h_step = self.transition(
                trans_in_t,
                past_key_values=cache,
                cache_position=pos[:, t : t + 1],
                attention_window=attention_window,
            )
            h_t = h_step[:, -1, :]
            prior_logits_t = self.prior_head(h_t).view(B, self.stoch_size, self.stoch_classes)
            z_prior_t = self._sample_stoch(prior_logits_t.unsqueeze(1), self.training).squeeze(1)
            z_prior_flat = z_prior_t.reshape(B, self.stoch_flat)
            post_logits_t = self.post_head(torch.cat([h_t.detach(), obs_embed[:, t, :]], dim=-1)).view(
                B, self.stoch_size, self.stoch_classes
            )
            z_t = self._sample_stoch(post_logits_t.unsqueeze(1), self.training).squeeze(1)
            z_t_flat = z_t.reshape(B, self.stoch_flat)

            feat_steps.append(torch.cat([h_t, z_t_flat], dim=-1))
            feat_prior_steps.append(torch.cat([h_t, z_prior_flat], dim=-1))
            z_only_steps.append(z_t_flat)
            h_only_steps.append(h_t)
            prior_logits_steps.append(prior_logits_t)
            post_logits_steps.append(post_logits_t)

            h_prev = h_t
            z_prev_flat = z_t_flat

        feat = torch.stack(feat_steps, dim=1)                 # [B, T, H + S*C]
        feat_prior = torch.stack(feat_prior_steps, dim=1)     # [B, T, H + S*C]
        prior_logits = torch.stack(prior_logits_steps, dim=1) # [B, T, S, C]
        post_logits = torch.stack(post_logits_steps, dim=1)   # [B, T, S, C]

        outputs, obs_hat = self._decode_feat(feat)
        prior_sensor_pred = self._decode_sensor_from_feat(feat_prior)
        z_only_pred = None
        if z_only_steps:
            z_only_pred = self._decode_sensor_from_z(torch.stack(z_only_steps, dim=1))
        h_only_pred = None
        if h_only_steps:
            h_only_pred = self._decode_sensor_from_h(torch.stack(h_only_steps, dim=1))
        prior_roll_sensor_pred = None
        if self.prior_rollout_weight > 0:
            if attention_window is not None and attention_window > 0:
                cache_roll = WindowedPositionBasedDynamicCache(
                    int(attention_window),
                    detach_on_pop=detach_on_pop,
                ).to(device=device)
            else:
                cache_roll = PositionBasedDynamicCache(detach_on_pop=detach_on_pop).to(device=device)
            h_roll = h_init
            z_roll_flat = z_init_flat
            feat_roll_steps = []
            roll_T = T if self.prior_rollout_steps <= 0 else min(T, self.prior_rollout_steps)
            for t in range(roll_T):
                if self.bptt_horizon > 0 and t > 0 and (t % self.bptt_horizon) == 0:
                    z_roll_flat = z_roll_flat.detach()
                trans_in_roll_t = torch.cat([z_roll_flat, a_prev[:, t, :]], dim=-1).unsqueeze(1)
                h_step_roll = self.transition(
                    trans_in_roll_t,
                    past_key_values=cache_roll,
                    cache_position=pos[:, t : t + 1],
                    attention_window=attention_window,
                )
                h_roll = h_step_roll[:, -1, :]
                prior_logits_roll_t = self.prior_head(h_roll).view(B, self.stoch_size, self.stoch_classes)
                z_roll_t = self._sample_stoch(prior_logits_roll_t.unsqueeze(1), self.training).squeeze(1)
                z_roll_flat = z_roll_t.reshape(B, self.stoch_flat)
                feat_roll_steps.append(torch.cat([h_roll, z_roll_flat], dim=-1))
            if feat_roll_steps:
                feat_roll = torch.stack(feat_roll_steps, dim=1)
                prior_roll_sensor_pred = self._decode_sensor_from_feat(feat_roll)

        aux_inputs = {
            "prior_logits": prior_logits,
            "post_logits": post_logits,
            "feat": feat,
            "obs_target": obs_features,
            "sensor_target": obs["sensor"],
            "loc_target": obs["loc"],
            "head_target": obs["heading"],
            "prior_sensor_pred": prior_sensor_pred,
            "prior_roll_sensor_pred": prior_roll_sensor_pred,
            "z_only_pred": z_only_pred,
            "h_only_pred": h_only_pred,
        }
        if obs_hat is not None:
            aux_inputs["obs_hat"] = obs_hat
        if return_state:
            last_state = torch.cat([h_prev, z_prev_flat], dim=-1)
            return (*outputs, aux_inputs, last_state)
        return (*outputs, aux_inputs)


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

        if isinstance(model, DiscreteLatentPredictorBase):
            pred_sensor, pred_loc_x, pred_loc_y, pred_head, pred_turn, pred_step, aux_inputs = model(
                obs, attention_window=attention_window
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
        total_loc_x_rmse += float(metrics["loc_x_rmse"].detach().cpu())
        total_loc_y_rmse += float(metrics["loc_y_rmse"].detach().cpu())
        total_loc_x_acc += float(metrics["loc_x_acc"].detach().cpu())
        total_loc_y_acc += float(metrics["loc_y_acc"].detach().cpu())
        total_batches += 1

    if total_batches == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
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
    )

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
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=32)
    parser.add_argument("--intermediate", type=int, default=512)
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
        default=0,
        help="Hidden size for probe MLPs (0 keeps linear probes).",
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
        f"context_len={args.context_len} probe_hidden_dim={args.probe_hidden_dim} "
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
        ).to(args.device)
        model_config_extra["llama"] = asdict(cfg)
        model_config_extra["probe_hidden_dim"] = args.probe_hidden_dim
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
        ).to(args.device)
        model_config_extra["rnn"] = {
            "input_size": input_dim,
            "hidden_size": args.hidden_size,
            "layers": args.layers,
            "probe_hidden_dim": args.probe_hidden_dim,
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
    for epoch in range(1, args.epochs + 1):
        print(f"starting epoch {epoch:03d}", flush=True)

        val_loss, val_mse, val_rmse, val_lr_rmse, val_lr_acc, val_loc_x_rmse, val_loc_y_rmse, val_loc_x_acc, val_loc_y_acc, val_turn_acc, val_step_acc, val_kl_dyn, val_kl_rep, val_prior_roll, val_z_only, val_h_only, val_recon = run_epoch_joint(
            model,
            val_loader,
            optimizer=None,
            device=args.device,
            attention_window=active_attention_window,
            sensor_weight=args.sensor_weight,
            loc_weight=args.loc_weight,
            head_weight=args.head_weight,
            turn_weight=args.turn_weight,
            step_weight=args.step_weight,
            loc_min=loc_min,
            sensor_mode=args.sensor_mode,
            sensor_min_idx=sensor_min_idx,
            sensor_tables=sensor_tables,
            loc_x_table=loc_x_table,
            loc_y_table=loc_y_table,
            heading_table=heading_table,
        )

        train_loss, train_mse, train_rmse, train_lr_rmse, train_lr_acc, train_loc_x_rmse, train_loc_y_rmse, train_loc_x_acc, train_loc_y_acc, train_turn_acc, train_step_acc, train_kl_dyn, train_kl_rep, train_prior_roll, train_z_only, train_h_only, train_recon = run_epoch_joint(
            model,
            train_loader,
            optimizer,
            device=args.device,
            attention_window=active_attention_window,
            sensor_weight=args.sensor_weight,
            loc_weight=args.loc_weight,
            head_weight=args.head_weight,
            turn_weight=args.turn_weight,
            step_weight=args.step_weight,
            loc_min=loc_min,
            sensor_mode=args.sensor_mode,
            sensor_min_idx=sensor_min_idx,
            sensor_tables=sensor_tables,
            loc_x_table=loc_x_table,
            loc_y_table=loc_y_table,
            heading_table=heading_table,
        )

        print(
            f"epoch {epoch:03d} | "
            f"train loss {train_loss:.4f} mse {train_mse:.4f} rmse {train_rmse:.4f} "
            f"lr_rmse {train_lr_rmse:.3f} lr_acc {train_lr_acc:.3f} "
            f"loc_x_rmse {train_loc_x_rmse:.3f} loc_y_rmse {train_loc_y_rmse:.3f} "
            f"loc_x_acc {train_loc_x_acc:.3f} loc_y_acc {train_loc_y_acc:.3f} "
            f"turn_acc {train_turn_acc:.3f} step_acc {train_step_acc:.3f} "
            f"kl_dyn {train_kl_dyn:.4f} kl_rep {train_kl_rep:.4f} prior_roll {train_prior_roll:.4f} z_only {train_z_only:.4f} h_only {train_h_only:.4f} recon {train_recon:.4f} | "
            f"val loss {val_loss:.4f} mse {val_mse:.4f} rmse {val_rmse:.4f} "
            f"lr_rmse {val_lr_rmse:.3f} lr_acc {val_lr_acc:.3f} "
            f"loc_x_rmse {val_loc_x_rmse:.3f} loc_y_rmse {val_loc_y_rmse:.3f} "
            f"loc_x_acc {val_loc_x_acc:.3f} loc_y_acc {val_loc_y_acc:.3f} "
            f"turn_acc {val_turn_acc:.3f} step_acc {val_step_acc:.3f} "
            f"kl_dyn {val_kl_dyn:.4f} kl_rep {val_kl_rep:.4f} prior_roll {val_prior_roll:.4f} z_only {val_z_only:.4f} h_only {val_h_only:.4f} recon {val_recon:.4f}"
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
                "pos_sigma": args.pos_sigma,
                "heading_smoothing": args.heading_smoothing,
            },
        },
        args.save_path,
    )
    print(f"saved checkpoint to {args.save_path}", flush=True)


if __name__ == "__main__":
    main()
