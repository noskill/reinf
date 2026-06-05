#!/usr/bin/env python3
"""Analyze whether world-model embeddings are spatially organized in MazeVecEnv."""

from __future__ import annotations

import argparse
import ast
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

THIS_DIR = Path(__file__).resolve().parent
PARENT_DIR = THIS_DIR.parent
MAZE_DIR = THIS_DIR / "maze"
for path in (PARENT_DIR, MAZE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from agent_utils_wm import MAZE_WM_MODEL_DEFAULTS, add_create_model_args, extract_create_model_args  # noqa: E402
from rl_env import DEFAULT_ACTIONS, MazeVecEnv  # noqa: E402
from wm_joint_agent import create_maze_world_model  # noqa: E402

try:  # noqa: E402
    from image import DummyImage  # type: ignore
    from robot import Robot  # type: ignore
except Exception:  # pragma: no cover - only needed for --policy algernon
    DummyImage = None
    Robot = None


HEADING_TO_IDX = {
    "u": 0,
    "up": 0,
    "r": 1,
    "right": 1,
    "d": 2,
    "down": 2,
    "l": 3,
    "left": 3,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wm-load-path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=256)
    parser.add_argument("--maze-path", type=str, default="", help="Fixed maze path for rollout collection.")
    parser.add_argument("--random-dim", type=int, default=10)
    parser.add_argument("--random-extra-openings", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--policy", choices=("random", "sweep", "algernon"), default="random")
    parser.add_argument(
        "--data-path",
        type=str,
        default="",
        help="Replay saved Algernon/tester episodes instead of collecting a rollout.",
    )
    parser.add_argument("--max-episodes", type=int, default=64, help="Maximum saved episodes to load with --data-path.")
    parser.add_argument("--sample-size", type=int, default=4096)
    parser.add_argument("--knn-k", type=int, default=10)
    parser.add_argument("--cluster-count", type=int, default=100)
    parser.add_argument("--embedding", choices=("contrastive", "state"), default="contrastive")
    parser.add_argument("--no-cosine", action="store_true", help="Skip cosine-distance metrics.")
    parser.add_argument("--wm-contrastive-temp", type=float, default=0.1)
    parser.add_argument("--wm-contrastive-discount", type=float, default=0.75)
    parser.add_argument("--wm-sensor-max-bin", type=int, default=64)
    parser.add_argument("--wm-sensor-weight", type=float, default=1.0)
    parser.add_argument("--wm-loc-weight", type=float, default=0.1)
    parser.add_argument("--wm-head-weight", type=float, default=0.1)
    parser.add_argument("--wm-turn-weight", type=float, default=0.1)
    parser.add_argument("--wm-step-weight", type=float, default=0.1)
    parser.add_argument("--wm-sensor-sigma", type=float, default=0.0)
    parser.add_argument("--wm-pos-sigma", type=float, default=1.0)
    parser.add_argument("--wm-heading-smoothing", type=float, default=0.0)
    add_create_model_args(
        parser,
        arg_prefix="wm",
        defaults=MAZE_WM_MODEL_DEFAULTS,
        include_load_path=False,
    )
    return parser.parse_args()


def load_checkpoint_config(args: argparse.Namespace) -> None:
    ckpt = torch.load(args.wm_load_path, map_location=args.device, weights_only=False)
    cfg = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
    model_cfg = cfg.get("model_config", {}) if isinstance(cfg, dict) else {}

    def set_if_present(arg_name: str, value):
        if value is not None and hasattr(args, arg_name):
            setattr(args, arg_name, value)

    set_if_present("wm_model_type", cfg.get("model_type"))
    set_if_present("wm_obs_latent_dim", cfg.get("obs_latent_dim"))
    set_if_present("wm_probe_hidden_dim", cfg.get("probe_hidden_dim"))
    set_if_present("wm_probe_layers", cfg.get("probe_layers"))
    set_if_present("wm_rnn_state_norm", cfg.get("rnn_state_norm"))
    set_if_present("wm_rssm_transition", cfg.get("rssm_transition"))
    set_if_present("wm_rssm_residual_scale", cfg.get("rssm_residual_scale"))
    set_if_present("wm_rssm_state_norm", cfg.get("rssm_state_norm"))
    set_if_present("wm_contrastive_dim", cfg.get("contrastive_dim"))
    set_if_present("wm_contrastive_steps", cfg.get("contrastive_steps"))
    args._checkpoint_loc_x_bins = int(cfg.get("loc_x_bins", 0) or 0)
    args._checkpoint_loc_y_bins = int(cfg.get("loc_y_bins", 0) or 0)

    specific = model_cfg.get(getattr(args, "wm_model_type", ""), {}) if isinstance(model_cfg, dict) else {}
    for key, value in specific.items():
        set_if_present(f"wm_{key}", value)

    args.wm_load_path = args.wm_load_path


def build_model(
    args: argparse.Namespace,
    *,
    env: Optional[MazeVecEnv] = None,
    maze_dim: Optional[int] = None,
    action_table: Optional[List[Tuple[int, int]]] = None,
):
    load_checkpoint_config(args)
    model_args = extract_create_model_args(args, arg_prefix="wm", device=args.device)
    checkpoint_dim = max(int(getattr(args, "_checkpoint_loc_x_bins", 0)), int(getattr(args, "_checkpoint_loc_y_bins", 0)))
    if checkpoint_dim > 0:
        maze_dim = checkpoint_dim
    elif maze_dim is None:
        if env is not None:
            maze_dim = int(env._mazes[0].dim)
        else:
            maze_dim = int(args.random_dim)
    if action_table is None:
        action_table = list(env.action_table if env is not None else DEFAULT_ACTIONS)
    turn_bins = len({int(turn) for turn, _ in action_table})
    step_bins = len({int(step) for _, step in action_table})
    args._model_maze_dim = int(maze_dim)
    model = create_maze_world_model(
        model_args=model_args,
        device=torch.device(args.device),
        maze_dim=maze_dim,
        turn_bins=turn_bins,
        step_bins=step_bins,
        contrastive_temp=getattr(args, "wm_contrastive_temp", 0.1),
        contrastive_horizon_discount=getattr(args, "wm_contrastive_discount", 0.75),
        sensor_weight=getattr(args, "wm_sensor_weight", 1.0),
        loc_weight=getattr(args, "wm_loc_weight", 0.1),
        head_weight=getattr(args, "wm_head_weight", 0.1),
        turn_weight=getattr(args, "wm_turn_weight", 0.1),
        step_weight=getattr(args, "wm_step_weight", 0.1),
        sensor_sigma=getattr(args, "wm_sensor_sigma", 0.0),
        pos_sigma=getattr(args, "wm_pos_sigma", 1.0),
        heading_smoothing=getattr(args, "wm_heading_smoothing", 0.0),
        sensor_max_bin=getattr(args, "wm_sensor_max_bin", 64),
    )
    model.eval()
    return model


def collect_rollout(args: argparse.Namespace):
    env = MazeVecEnv(
        num_envs=args.num_envs,
        maze_path=args.maze_path or None,
        random_dim=args.random_dim,
        random_extra_openings=args.random_extra_openings,
        max_steps=args.max_steps,
        seed=args.seed,
        auto_reset=False,
        return_torch=True,
        device=args.device,
    )
    obs_seq = []
    loc_seq = []
    heading_seq = []
    sensor_seq = []
    action_seq = []
    obs = env.reset()
    rng = np.random.default_rng(args.seed + 17)
    robots = None
    action_to_idx = {tuple(map(int, action)): idx for idx, action in enumerate(env.action_table)}
    if args.policy == "algernon":
        if Robot is None or DummyImage is None:
            raise RuntimeError("--policy algernon requires robot.py and image.py in wm/maze or Algernon maze deps.")
        robots = [Robot(env._mazes[i].dim, DummyImage()) for i in range(env.num_envs)]
    for step in range(args.max_steps):
        obs_seq.append({key: value.detach().clone() for key, value in obs.items()})
        loc_seq.append(obs["location"].detach().clone())
        heading_seq.append(obs["heading_idx"].detach().clone())
        sensor_seq.append(obs["sensor"].detach().clone())
        if args.policy == "random":
            actions_np = rng.integers(0, env.action_space_n, size=env.num_envs, dtype=np.int64)
        elif args.policy == "sweep":
            actions_np = np.full(env.num_envs, 3 + (step % 3), dtype=np.int64)
            actions_np = np.clip(actions_np, 0, env.action_space_n - 1)
        else:
            assert robots is not None
            actions_np = np.zeros(env.num_envs, dtype=np.int64)
            reset_envs = []
            sensors_np = obs["sensor"].detach().cpu().numpy()
            for env_idx, robot in enumerate(robots):
                action = robot.next_move([int(x) for x in sensors_np[env_idx].tolist()])
                if action == ("Reset", "Reset"):
                    robot.reset()
                    reset_envs.append(env_idx)
                    action = (0, 0)
                action_key = tuple(map(int, action))
                if action_key not in action_to_idx:
                    raise ValueError(f"Algernon policy produced unsupported action {action_key}; action_table={env.action_table}")
                actions_np[env_idx] = action_to_idx[action_key]
        actions = torch.as_tensor(actions_np, dtype=torch.long, device=env.device)
        action_seq.append(actions.detach().clone())
        obs, _rew, _done, _info = env.step(actions)
        if args.policy == "algernon" and reset_envs:
            for env_idx in reset_envs:
                env._single_reset(env_idx)
            obs = env._build_obs()
    action_table = list(env.action_table)
    maze_dim = int(env._mazes[0].dim)
    env.close()
    return (
        obs_seq,
        torch.stack(loc_seq, dim=1),
        torch.stack(heading_seq, dim=1),
        torch.stack(sensor_seq, dim=1),
        torch.stack(action_seq, dim=1),
        action_table,
        maze_dim,
    )


def load_saved_episodes(path: str, max_episodes: Optional[int], max_steps: int):
    episodes = []
    current = []

    def flush_current():
        nonlocal current
        if current:
            episodes.append(current[:max_steps])
            current = []

    with open(os.path.expanduser(path), "r") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("random "):
                flush_current()
                if max_episodes is not None and len(episodes) >= max_episodes:
                    break
                continue
            for item in line.split(";"):
                item = item.strip()
                if not item:
                    continue
                step = ast.literal_eval(item)
                current.append(step)
                if len(current) >= max_steps:
                    flush_current()
                    if max_episodes is not None and len(episodes) >= max_episodes:
                        break
            if max_episodes is not None and len(episodes) >= max_episodes:
                break
        flush_current()

    if max_episodes is not None:
        episodes = episodes[:max_episodes]
    episodes = [episode for episode in episodes if episode]
    if not episodes:
        raise ValueError(f"No episodes loaded from {path}")
    return episodes


def build_saved_episode_tensors(args: argparse.Namespace):
    episodes = load_saved_episodes(args.data_path, args.max_episodes, args.max_steps)
    batch_size = len(episodes)
    seq_len = max(len(episode) for episode in episodes)
    sensors = torch.zeros((batch_size, seq_len, 3), dtype=torch.float32)
    heading = torch.zeros((batch_size, seq_len), dtype=torch.long)
    location = torch.zeros((batch_size, seq_len, 2), dtype=torch.long)
    step_count = torch.zeros((batch_size, seq_len), dtype=torch.long)
    actions = torch.zeros((batch_size, seq_len, 2), dtype=torch.float32)
    key_padding_mask = torch.ones((batch_size, seq_len), dtype=torch.bool)

    max_coord = 0
    for episode_idx, episode in enumerate(episodes):
        for step_idx, step in enumerate(episode):
            loc = step["location"]
            act = step["action"]
            sensors[episode_idx, step_idx] = torch.tensor(step["sensor"], dtype=torch.float32)
            heading[episode_idx, step_idx] = HEADING_TO_IDX[str(step["heading"]).lower()]
            location[episode_idx, step_idx] = torch.tensor(loc, dtype=torch.long)
            step_count[episode_idx, step_idx] = step_idx
            actions[episode_idx, step_idx] = torch.tensor(act, dtype=torch.float32)
            key_padding_mask[episode_idx, step_idx] = False
            max_coord = max(max_coord, int(loc[0]), int(loc[1]))

    device = torch.device(args.device)
    wm_obs = {
        "sensor": sensors.to(device),
        "heading_idx": heading.to(device),
        "location": location.to(device),
        "step_count": step_count.to(device),
        "action": actions.to(device),
        "actions": actions.to(device),
        "key_padding_mask": key_padding_mask.to(device),
    }
    return wm_obs, location, heading, sensors, key_padding_mask, max_coord + 1, list(DEFAULT_ACTIONS)


def build_wm_obs(obs_seq, actions, action_table, device):
    sensors = torch.stack([obs["sensor"] for obs in obs_seq], dim=1).to(device)
    heading = torch.stack([obs["heading_idx"] for obs in obs_seq], dim=1).to(device)
    location = torch.stack([obs["location"] for obs in obs_seq], dim=1).to(device)
    step_count = torch.stack([obs["step_count"] for obs in obs_seq], dim=1).to(device)
    action_table_t = torch.tensor(action_table, dtype=torch.float32, device=device)
    action_vals = action_table_t[actions.long().to(device)]
    return {
        "sensor": sensors,
        "heading_idx": heading,
        "location": location,
        "step_count": step_count,
        "action": action_vals,
        "actions": action_vals,
        "key_padding_mask": torch.zeros(sensors.shape[:2], dtype=torch.bool, device=device),
    }


def corrcoef(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.float().flatten()
    y = y.float().flatten()
    keep = torch.isfinite(x) & torch.isfinite(y)
    x = x[keep]
    y = y[keep]
    if x.numel() < 2:
        return float("nan")
    x = x - x.mean()
    y = y - y.mean()
    denom = x.norm() * y.norm()
    if denom <= 0:
        return float("nan")
    return float((x * y).sum().detach().cpu() / denom.detach().cpu())


def sample_rows(
    emb: torch.Tensor,
    loc: torch.Tensor,
    heading: torch.Tensor,
    sensor: torch.Tensor,
    sample_size: int,
    seed: int,
    key_padding_mask: Optional[torch.Tensor] = None,
):
    flat_emb = emb.reshape(-1, emb.shape[-1]).float().cpu()
    flat_loc = loc.reshape(-1, 2).float().cpu()
    flat_heading = heading.reshape(-1).long().cpu()
    flat_sensor = sensor.reshape(-1, sensor.shape[-1]).float().cpu()
    if key_padding_mask is not None:
        valid = (~key_padding_mask).reshape(-1).cpu()
        flat_emb = flat_emb[valid]
        flat_loc = flat_loc[valid]
        flat_heading = flat_heading[valid]
        flat_sensor = flat_sensor[valid]
    total = flat_emb.shape[0]
    gen = torch.Generator().manual_seed(seed)
    if total > sample_size:
        idx = torch.randperm(total, generator=gen)[:sample_size]
        flat_emb = flat_emb[idx]
        flat_loc = flat_loc[idx]
        flat_heading = flat_heading[idx]
        flat_sensor = flat_sensor[idx]
    return flat_emb, flat_loc, flat_heading, flat_sensor


def sample_valid_rows(
    emb: torch.Tensor,
    loc: torch.Tensor,
    heading: torch.Tensor,
    sensor: torch.Tensor,
    sample_size: int,
    seed: int,
):
    total = emb.shape[0]
    gen = torch.Generator().manual_seed(seed)
    if total > sample_size:
        idx = torch.randperm(total, generator=gen)[:sample_size]
        emb = emb[idx]
        loc = loc[idx]
        heading = heading[idx]
        sensor = sensor[idx]
    return emb.float().cpu(), loc.float().cpu(), heading.long().cpu(), sensor.float().cpu()


def pairwise_metric_values(emb: torch.Tensor, loc: torch.Tensor, heading: torch.Tensor, sensor: torch.Tensor, cosine: bool) -> Dict[str, float]:
    if emb.shape[0] < 2:
        return {
            "pair_corr_spatial_l2": float("nan"),
            "pair_corr_spatial_l1": float("nan"),
            "pair_corr_sensor": float("nan"),
            "pair_corr_heading_diff": float("nan"),
            "emb_dist_mean": float("nan"),
            "emb_dist_std": float("nan"),
        }
    if cosine:
        emb_dist = 1.0 - F.normalize(emb, dim=-1).matmul(F.normalize(emb, dim=-1).T)
    else:
        emb_dist = torch.cdist(emb, emb)
    spatial_l2 = torch.cdist(loc, loc)
    spatial_l1 = torch.cdist(loc, loc, p=1)
    sensor_dist = torch.cdist(sensor, sensor)
    heading_diff = (heading[:, None] != heading[None, :]).float()
    tri = torch.triu_indices(emb.shape[0], emb.shape[0], offset=1)
    ed = emb_dist[tri[0], tri[1]]
    return {
        "pair_corr_spatial_l2": corrcoef(ed, spatial_l2[tri[0], tri[1]]),
        "pair_corr_spatial_l1": corrcoef(ed, spatial_l1[tri[0], tri[1]]),
        "pair_corr_sensor": corrcoef(ed, sensor_dist[tri[0], tri[1]]),
        "pair_corr_heading_diff": corrcoef(ed, heading_diff[tri[0], tri[1]]),
        "emb_dist_mean": ed.mean().item(),
        "emb_dist_std": ed.std(unbiased=False).item(),
    }


def knn_metric_values(emb: torch.Tensor, loc: torch.Tensor, heading: torch.Tensor, sensor: torch.Tensor, k: int, cosine: bool) -> Dict[str, float]:
    if emb.shape[0] < 2:
        return {
            "knn_loc_l2_mean": float("nan"),
            "knn_loc_l1_mean": float("nan"),
            "knn_same_cell_rate": float("nan"),
            "knn_same_heading_rate": float("nan"),
            "knn_sensor_l2_mean": float("nan"),
        }
    if cosine:
        dist = 1.0 - F.normalize(emb, dim=-1).matmul(F.normalize(emb, dim=-1).T)
    else:
        dist = torch.cdist(emb, emb)
    dist.fill_diagonal_(float("inf"))
    nn_idx = dist.topk(k=min(k, emb.shape[0] - 1), largest=False).indices
    nn_loc = loc[nn_idx]
    nn_heading = heading[nn_idx]
    nn_sensor = sensor[nn_idx]
    loc_l2 = (nn_loc - loc[:, None, :]).norm(dim=-1)
    loc_l1 = (nn_loc - loc[:, None, :]).abs().sum(dim=-1)
    same_cell = (loc_l1 == 0).float()
    same_heading = (nn_heading == heading[:, None]).float()
    sensor_l2 = (nn_sensor - sensor[:, None, :]).norm(dim=-1)
    return {
        "knn_loc_l2_mean": loc_l2.mean().item(),
        "knn_loc_l1_mean": loc_l1.mean().item(),
        "knn_same_cell_rate": same_cell.mean().item(),
        "knn_same_heading_rate": same_heading.mean().item(),
        "knn_sensor_l2_mean": sensor_l2.mean().item(),
    }


def cluster_metric_values(emb: torch.Tensor, loc: torch.Tensor, cluster_count: int, seed: int) -> Dict[str, float]:
    try:
        from sklearn.cluster import MiniBatchKMeans
    except Exception as exc:
        print(f"cluster/skipped {exc}")
        return {}
    if emb.shape[0] < 2:
        return {
            "n_clusters": float("nan"),
            "used_clusters": float("nan"),
            "unique_cells": float("nan"),
            "cells_per_cluster_mean": float("nan"),
            "cells_per_cluster_p90": float("nan"),
            "cells_per_cluster_max": float("nan"),
            "clusters_per_cell_mean": float("nan"),
            "clusters_per_cell_p90": float("nan"),
            "clusters_per_cell_max": float("nan"),
        }
    k = min(cluster_count, emb.shape[0])
    model = MiniBatchKMeans(n_clusters=k, batch_size=min(10000, emb.shape[0]), random_state=seed, n_init="auto")
    labels = model.fit_predict(emb.numpy())
    cell_ids = [tuple(map(int, xy)) for xy in loc.numpy()]
    cluster_to_cells = defaultdict(set)
    cell_to_clusters = defaultdict(set)
    for label, cell in zip(labels, cell_ids):
        cluster_to_cells[int(label)].add(cell)
        cell_to_clusters[cell].add(int(label))
    cells_per_cluster = np.array([len(v) for v in cluster_to_cells.values()], dtype=np.float32)
    clusters_per_cell = np.array([len(v) for v in cell_to_clusters.values()], dtype=np.float32)
    return {
        "n_clusters": float(k),
        "used_clusters": float(len(cluster_to_cells)),
        "unique_cells": float(len(cell_to_clusters)),
        "cells_per_cluster_mean": float(cells_per_cluster.mean()),
        "cells_per_cluster_p90": float(np.quantile(cells_per_cluster, 0.90)),
        "cells_per_cluster_max": float(cells_per_cluster.max()),
        "clusters_per_cell_mean": float(clusters_per_cell.mean()),
        "clusters_per_cell_p90": float(np.quantile(clusters_per_cell, 0.90)),
        "clusters_per_cell_max": float(clusters_per_cell.max()),
    }


def print_metrics(prefix: str, metrics: Dict[str, float]) -> None:
    for key, value in metrics.items():
        print(f"{prefix}/{key} {value:.4f}")


def mean_metric_dicts(items: List[Dict[str, float]]) -> Dict[str, float]:
    keys = sorted({key for item in items for key in item})
    result = {}
    for key in keys:
        values = np.asarray([item[key] for item in items if key in item and np.isfinite(item[key])], dtype=np.float64)
        result[key] = float(values.mean()) if values.size else float("nan")
    return result


def full_metric_values(
    emb: torch.Tensor,
    loc: torch.Tensor,
    heading: torch.Tensor,
    sensor: torch.Tensor,
    *,
    knn_k: int,
    cluster_count: int,
    seed: int,
    include_cosine: bool,
) -> Dict[str, Dict[str, float]]:
    result = {
        "euclidean": {
            **pairwise_metric_values(emb, loc, heading, sensor, cosine=False),
            **knn_metric_values(emb, loc, heading, sensor, knn_k, cosine=False),
        },
        "cluster": cluster_metric_values(emb, loc, cluster_count, seed),
    }
    if include_cosine:
        result["cosine"] = {
            **pairwise_metric_values(emb, loc, heading, sensor, cosine=True),
            **knn_metric_values(emb, loc, heading, sensor, knn_k, cosine=True),
        }
    return result


def print_full_metrics(prefix: str, metrics: Dict[str, Dict[str, float]]) -> None:
    for family, family_metrics in metrics.items():
        print_metrics(f"{prefix}/{family}" if prefix else family, family_metrics)


def per_sequence_average_metrics(
    emb: torch.Tensor,
    loc: torch.Tensor,
    heading: torch.Tensor,
    sensor: torch.Tensor,
    key_padding_mask: torch.Tensor,
    args: argparse.Namespace,
) -> Tuple[Dict[str, Dict[str, float]], int]:
    per_family: Dict[str, List[Dict[str, float]]] = defaultdict(list)
    rows_used = 0
    for seq_idx in range(emb.shape[0]):
        valid = (~key_padding_mask[seq_idx]).cpu()
        if int(valid.sum().item()) < 2:
            continue
        emb_i, loc_i, heading_i, sensor_i = sample_valid_rows(
            emb[seq_idx].detach().cpu()[valid],
            loc[seq_idx].detach().cpu()[valid],
            heading[seq_idx].detach().cpu()[valid],
            sensor[seq_idx].detach().cpu()[valid],
            args.sample_size,
            args.seed + seq_idx,
        )
        rows_used += int(emb_i.shape[0])
        metrics_i = full_metric_values(
            emb_i,
            loc_i,
            heading_i,
            sensor_i,
            knn_k=args.knn_k,
            cluster_count=args.cluster_count,
            seed=args.seed + seq_idx,
            include_cosine=not args.no_cosine,
        )
        for family, family_metrics in metrics_i.items():
            per_family[family].append(family_metrics)
    return {family: mean_metric_dicts(items) for family, items in per_family.items()}, rows_used


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.data_path:
        wm_obs, loc, heading, sensor, key_padding_mask, maze_dim, action_table = build_saved_episode_tensors(args)
        model = build_model(args, maze_dim=maze_dim, action_table=action_table)
    else:
        obs_seq, loc, heading, sensor, actions, action_table, maze_dim = collect_rollout(args)
        model = build_model(args, maze_dim=maze_dim, action_table=action_table)
        wm_obs = build_wm_obs(obs_seq, actions, action_table, torch.device(args.device))
        key_padding_mask = wm_obs["key_padding_mask"].detach().cpu()
    with torch.no_grad():
        out = model(wm_obs)
    if args.embedding == "contrastive":
        emb = out["aux"]["contrastive_tgt_emb"].detach()
    else:
        emb = out["state_seq"].detach()
    emb_s, loc_s, heading_s, sensor_s = sample_rows(emb, loc, heading, sensor, args.sample_size, args.seed, key_padding_mask)
    per_seq_metrics, per_seq_rows = per_sequence_average_metrics(emb, loc, heading, sensor, key_padding_mask, args)
    pooled_metrics = full_metric_values(
        emb_s,
        loc_s,
        heading_s,
        sensor_s,
        knn_k=args.knn_k,
        cluster_count=args.cluster_count,
        seed=args.seed,
        include_cosine=not args.no_cosine,
    )
    print(f"samples {emb_s.shape[0]}")
    print(f"sequences {emb.shape[0]}")
    print(f"per_sequence_samples_total {per_seq_rows}")
    print(f"source {'data_path' if args.data_path else args.policy}")
    print(f"data_maze_dim {maze_dim}")
    print(f"model_maze_dim {getattr(args, '_model_maze_dim', maze_dim)}")
    print(f"embedding_dim {emb_s.shape[1]}")
    print(f"emb/std_dim_mean {emb_s.std(dim=0, unbiased=False).mean().item():.4f}")
    print(f"emb/std_dim_max {emb_s.std(dim=0, unbiased=False).max().item():.4f}")
    print(f"emb/norm_mean {emb_s.norm(dim=1).mean().item():.4f}")
    print(f"emb/norm_std {emb_s.norm(dim=1).std(unbiased=False).item():.4f}")
    print_full_metrics("pooled", pooled_metrics)
    print_full_metrics("per_sequence_mean", per_seq_metrics)


if __name__ == "__main__":
    main()
