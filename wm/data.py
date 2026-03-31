#!/usr/bin/env python3
"""Data loading and batching utilities for world-model training."""

import ast
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

HEADING_CANON = {"up": "u"}
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
    (
        obs_sensors,
        obs_locs,
        obs_heads,
        obs_actions,
        ys_sensor,
        ys_sensor_idx,
        ys_loc_xy,
        ys_head,
        ys_turn,
        ys_step,
    ) = zip(*batch)
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
