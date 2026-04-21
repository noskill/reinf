#!/usr/bin/env python3
"""Reverse-path integration test on validation episodes.

Test protocol:
1) Warm up world model on the first n+1 observations of a validation trajectory.
2) Turn 180 degrees (two +90 turns with zero movement).
3) Replay reverse control by traversing warmup actions backward as (-turn, +step).
4) Roll out model in open-loop prior mode (no posterior observations on reverse leg).
5) Compare predicted left/right sensors against maze-ground-truth sensors.
"""

import argparse
import ast
import os
import random
import re
import sys
import tempfile
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from transformer import LlamaConfig
from tr_cache import PositionBasedDynamicCache, WindowedPositionBasedDynamicCache
from wm_train import (
    HEADING_CANON,
    RNNPredictor,
    RSSMDiscretePredictor,
    TSSMDiscretePredictor,
    TransformerBaseline,
    compute_stats,
    set_seed,
)


# Reuse maze generation/sensing from local ./maze symlink.
MAZE_DEPS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "maze"))
if not os.path.exists(os.path.join(MAZE_DEPS_DIR, "genmaze.py")):
    raise ModuleNotFoundError(
        f"Missing maze dependency path: {MAZE_DEPS_DIR}. "
        "Expected genmaze.py in ./maze."
    )
if MAZE_DEPS_DIR not in sys.path:
    sys.path.insert(0, MAZE_DEPS_DIR)
from genmaze import generate_maze, write_maze  # type: ignore  # noqa: E402
from maze import Maze  # type: ignore  # noqa: E402


DIR_SENSORS = {
    "u": ["l", "u", "r"],
    "r": ["u", "r", "d"],
    "d": ["r", "d", "l"],
    "l": ["d", "l", "u"],
}
DIR_MOVE = {
    "u": (0, 1),
    "r": (1, 0),
    "d": (0, -1),
    "l": (-1, 0),
}
DIR_REVERSE = {
    "u": "d",
    "r": "l",
    "d": "u",
    "l": "r",
}


def parse_map_meta(map_id: str) -> Optional[Tuple[int, int, int]]:
    m = re.match(r"random dim=(\d+) extra_openings=(\d+) seed=(\d+)", map_id.strip())
    if not m:
        return None
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def load_episodes_with_maps(path: str, max_episodes: Optional[int] = None):
    episodes: List[List[dict]] = []
    maps: List[str] = []
    buf: List[str] = []
    current_map: Optional[str] = None

    def flush_buf():
        nonlocal current_map
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
            maps.append(current_map or "")
        buf.clear()

    with open(path, "r") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("random "):
                flush_buf()
                current_map = line
                if max_episodes is not None and len(episodes) >= max_episodes:
                    break
                continue
            buf.append(line)
        flush_buf()

    if max_episodes is not None:
        episodes = episodes[:max_episodes]
        maps = maps[:max_episodes]
    return episodes, maps


def turn_heading(heading: str, rotation: int) -> str:
    order = ["u", "r", "d", "l"]
    idx = order.index(heading)
    if rotation == 90:
        return order[(idx + 1) % 4]
    if rotation == -90:
        return order[(idx - 1) % 4]
    if rotation == 0:
        return heading
    raise ValueError(f"Unsupported rotation: {rotation}")


def apply_action(maze: Maze, loc: Tuple[int, int], heading: str, rotation: int, movement: int):
    heading = turn_heading(heading, int(rotation))
    movement = int(max(min(movement, 3), -3))
    x, y = int(loc[0]), int(loc[1])
    while movement != 0:
        if movement > 0:
            if maze.is_permissible([x, y], heading):
                dx, dy = DIR_MOVE[heading]
                x += dx
                y += dy
                movement -= 1
            else:
                break
        else:
            rev = DIR_REVERSE[heading]
            if maze.is_permissible([x, y], rev):
                dx, dy = DIR_MOVE[rev]
                x += dx
                y += dy
                movement += 1
            else:
                break
    return (x, y), heading


def sense_lfr(maze: Maze, loc: Tuple[int, int], heading: str) -> Tuple[int, int, int]:
    dirs = DIR_SENSORS[heading]
    return (
        int(maze.dist_to_wall(list(loc), dirs[0])),
        int(maze.dist_to_wall(list(loc), dirs[1])),
        int(maze.dist_to_wall(list(loc), dirs[2])),
    )


def build_rollout_actions(
    ep: List[dict],
    n_step: int,
    max_rollout_steps: int,
) -> Optional[List[Tuple[int, int]]]:
    actions: List[Tuple[int, int]] = [(90, 0), (90, 0)]
    for t in range(n_step - 1, -1, -1):
        turn_fwd = int(ep[t]["action"][0])
        step_fwd = int(ep[t]["action"][1])
        if step_fwd < 0:
            return None
        actions.append((-turn_fwd, step_fwd))
    return actions[:max_rollout_steps]


def normalize_action(turn: int, step: int, turn_max: float, step_max: float) -> torch.Tensor:
    t = float(turn) / turn_max if turn_max != 0 else 0.0
    s = float(step) / step_max if step_max != 0 else 0.0
    return torch.tensor([[t, s]], dtype=torch.float32)


def build_model_from_checkpoint(ckpt: dict, sensor_mode: str, device: str):
    cfg = ckpt["config"]
    stats = ckpt["stats"]
    model_type = cfg["model_type"]
    model_cfg = cfg["model_config"]
    sensor_dim = int(cfg["sensor_dim"])
    obs_dim = int(cfg["obs_dim"])
    obs_latent_dim = int(cfg["obs_latent_dim"])
    loc_x_bins = int(cfg["loc_x_bins"])
    loc_y_bins = int(cfg["loc_y_bins"])
    heading_dim = int(cfg["heading_dim"])
    turn_bins = int(cfg["turn_bins"])
    step_bins = int(cfg["step_bins"])
    sensor_min = np.asarray(stats["sensor_min"], dtype=np.int64)
    sensor_max = np.asarray(stats["sensor_max"], dtype=np.int64)
    sensor_bins = (sensor_max - sensor_min + 1).astype(np.int64)
    contrastive_dim_cfg = int(cfg.get("contrastive_dim", 0))
    contrastive_steps_cfg = int(cfg.get("contrastive_steps", 1))

    def resolve_probe_layers(model_cfg_section: dict, probe_hidden_dim: int) -> int:
        raw = model_cfg_section.get("probe_layers", cfg.get("probe_layers", None))
        if raw is not None:
            return int(raw)
        # Backward compatibility for old checkpoints:
        # hidden_dim <= 0 used linear probes; hidden_dim > 0 used MLP probes.
        return 1 if int(probe_hidden_dim) <= 0 else 2

    if model_type == "tssm":
        mcfg = model_cfg["tssm"]
        probe_hidden_dim = int(mcfg.get("probe_hidden_dim", cfg.get("probe_hidden_dim", 0)))
        probe_layers = resolve_probe_layers(mcfg, probe_hidden_dim)
        contrastive_dim = int(mcfg.get("contrastive_dim", contrastive_dim_cfg))
        contrastive_steps = int(mcfg.get("contrastive_steps", contrastive_steps_cfg))
        model = TSSMDiscretePredictor(
            hidden_size=int(mcfg["hidden_size"]),
            layers=int(mcfg["layers"]),
            heads=int(mcfg["heads"]),
            head_dim=int(mcfg["head_dim"]),
            intermediate=int(mcfg["intermediate"]),
            attention_window=mcfg["attention_window"],
            sensor_mode=sensor_mode,
            sensor_dim=sensor_dim,
            sensor_bins=sensor_bins if sensor_mode == "categorical" else None,
            loc_x_bins=loc_x_bins,
            loc_y_bins=loc_y_bins,
            heading_dim=heading_dim,
            turn_bins=turn_bins,
            step_bins=step_bins,
            obs_dim=obs_dim,
            obs_latent_dim=obs_latent_dim,
            action_dim=int(mcfg["action_dim"]),
            stoch_size=int(mcfg["stoch_size"]),
            stoch_classes=int(mcfg["stoch_classes"]),
            stoch_temp=float(mcfg["stoch_temp"]),
            kl_dyn_beta=float(mcfg["kl_dyn_beta"]),
            kl_rep_beta=float(mcfg["kl_rep_beta"]),
            kl_free_nats=float(mcfg["kl_free_nats"]),
            prior_rollout_weight=float(mcfg.get("prior_rollout_weight", 0.0)),
            bptt_horizon=int(mcfg.get("bptt_horizon", 0)),
            z_only_weight=float(mcfg.get("z_only_weight", 0.0)),
            h_only_weight=float(mcfg.get("h_only_weight", 0.0)),
            probe_hidden_dim=probe_hidden_dim,
            probe_layers=probe_layers,
            contrastive_dim=contrastive_dim,
            contrastive_steps=contrastive_steps,
            recon_beta=float(mcfg["recon_beta"]),
            obs_loss_mode=str(mcfg["obs_loss_mode"]),
        ).to(device)
    elif model_type == "rssm-discrete":
        mcfg = model_cfg["rssm"]
        probe_hidden_dim = int(mcfg.get("probe_hidden_dim", cfg.get("probe_hidden_dim", 0)))
        probe_layers = resolve_probe_layers(mcfg, probe_hidden_dim)
        contrastive_dim = int(mcfg.get("contrastive_dim", contrastive_dim_cfg))
        contrastive_steps = int(mcfg.get("contrastive_steps", contrastive_steps_cfg))
        transition = str(mcfg.get("transition", cfg.get("rssm_transition", "gru")))
        residual_scale = float(mcfg.get("residual_scale", cfg.get("rssm_residual_scale", 1.0)))
        state_norm = str(mcfg.get("state_norm", cfg.get("rssm_state_norm", "none")))
        model = RSSMDiscretePredictor(
            hidden_size=int(mcfg["hidden_size"]),
            sensor_mode=sensor_mode,
            sensor_dim=sensor_dim,
            sensor_bins=sensor_bins if sensor_mode == "categorical" else None,
            loc_x_bins=loc_x_bins,
            loc_y_bins=loc_y_bins,
            heading_dim=heading_dim,
            turn_bins=turn_bins,
            step_bins=step_bins,
            obs_dim=obs_dim,
            obs_latent_dim=obs_latent_dim,
            action_dim=int(mcfg["action_dim"]),
            stoch_size=int(mcfg["stoch_size"]),
            stoch_classes=int(mcfg["stoch_classes"]),
            stoch_temp=float(mcfg["stoch_temp"]),
            kl_dyn_beta=float(mcfg["kl_dyn_beta"]),
            kl_rep_beta=float(mcfg["kl_rep_beta"]),
            kl_free_nats=float(mcfg["kl_free_nats"]),
            prior_rollout_weight=float(mcfg.get("prior_rollout_weight", 0.0)),
            bptt_horizon=int(mcfg.get("bptt_horizon", 0)),
            z_only_weight=float(mcfg.get("z_only_weight", 0.0)),
            h_only_weight=float(mcfg.get("h_only_weight", 0.0)),
            probe_hidden_dim=probe_hidden_dim,
            probe_layers=probe_layers,
            contrastive_dim=contrastive_dim,
            contrastive_steps=contrastive_steps,
            transition=transition,
            residual_scale=residual_scale,
            state_norm=state_norm,
            recon_beta=float(mcfg["recon_beta"]),
            obs_loss_mode=str(mcfg["obs_loss_mode"]),
        ).to(device)
    elif model_type == "transformer":
        llama_cfg = LlamaConfig(**model_cfg["llama"])
        probe_hidden_dim = int(model_cfg.get("probe_hidden_dim", cfg.get("probe_hidden_dim", 0)))
        probe_layers = resolve_probe_layers(model_cfg, probe_hidden_dim)
        contrastive_dim = int(model_cfg.get("contrastive_dim", contrastive_dim_cfg))
        contrastive_steps = int(model_cfg.get("contrastive_steps", contrastive_steps_cfg))
        model = TransformerBaseline(
            llama_cfg,
            sensor_mode=sensor_mode,
            sensor_dim=sensor_dim,
            sensor_bins=sensor_bins if sensor_mode == "categorical" else None,
            loc_x_bins=loc_x_bins,
            loc_y_bins=loc_y_bins,
            heading_dim=heading_dim,
            turn_bins=turn_bins,
            step_bins=step_bins,
            obs_dim=obs_dim,
            obs_latent_dim=obs_latent_dim,
            probe_hidden_dim=probe_hidden_dim,
            probe_layers=probe_layers,
            contrastive_dim=contrastive_dim,
            contrastive_steps=contrastive_steps,
        ).to(device)
    elif model_type == "rnn":
        rcfg = model_cfg.get("rnn", {})
        probe_hidden_dim = int(rcfg.get("probe_hidden_dim", model_cfg.get("probe_hidden_dim", cfg.get("probe_hidden_dim", 0))))
        probe_layers = resolve_probe_layers(rcfg, probe_hidden_dim)
        contrastive_dim = int(rcfg.get("contrastive_dim", model_cfg.get("contrastive_dim", contrastive_dim_cfg)))
        contrastive_steps = int(rcfg.get("contrastive_steps", model_cfg.get("contrastive_steps", contrastive_steps_cfg)))
        state_norm = str(rcfg.get("state_norm", cfg.get("rnn_state_norm", "none")))
        default_input_size = sensor_dim + 2
        rnn_cfg = LlamaConfig(
            input_size=int(rcfg.get("input_size", default_input_size)),
            hidden_size=int(rcfg.get("hidden_size", 128)),
            num_hidden_layers=int(rcfg.get("layers", 2)),
        )
        model = RNNPredictor(
            rnn_cfg,
            sensor_mode=sensor_mode,
            sensor_dim=sensor_dim,
            sensor_bins=sensor_bins if sensor_mode == "categorical" else None,
            loc_x_bins=loc_x_bins,
            loc_y_bins=loc_y_bins,
            heading_dim=heading_dim,
            turn_bins=turn_bins,
            step_bins=step_bins,
            obs_dim=obs_dim,
            obs_latent_dim=obs_latent_dim,
            probe_hidden_dim=probe_hidden_dim,
            probe_layers=probe_layers,
            state_norm=state_norm,
            contrastive_dim=contrastive_dim,
            contrastive_steps=contrastive_steps,
        ).to(device)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    missing, unexpected = model.load_state_dict(ckpt["model_state"], strict=False)
    allowed_missing = {
        "z_obs_head.weight",
        "z_obs_head.bias",
        "h_obs_head.weight",
        "h_obs_head.bias",
        "contrastive_head.weight",
        "contrastive_head.bias",
    }
    bad_missing = [k for k in missing if k not in allowed_missing]
    if bad_missing:
        raise RuntimeError(f"Checkpoint is missing required keys: {bad_missing}")
    if unexpected:
        raise RuntimeError(f"Checkpoint has unexpected keys: {unexpected}")
    if missing:
        print(f"load warning: missing optional keys: {missing}", flush=True)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description="Reverse path integration test.")
    parser.add_argument("--data", type=str, default="data.txt")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--n-step", type=int, default=20, help="Warmup to observation index n before reversing.")
    parser.add_argument("--sensor-mode", type=str, choices=["raw", "categorical"], default="categorical")
    parser.add_argument(
        "--sensor-head",
        type=str,
        choices=["zh", "z", "h"],
        default="zh",
        help="Sensor decoder to evaluate during reverse rollout: zh=[h,z], z=z-only probe head, h=h-only probe head.",
    )
    parser.add_argument(
        "--max-rollout-steps",
        type=int,
        default=None,
        help="Maximum open-loop rollout steps after warmup. Defaults to --n-step.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--max-val-episodes", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    max_rollout_steps = args.max_rollout_steps if args.max_rollout_steps is not None else args.n_step
    if max_rollout_steps <= 0:
        raise ValueError("--max-rollout-steps must be > 0")

    set_seed(args.seed)
    episodes, map_ids = load_episodes_with_maps(args.data, max_episodes=args.max_episodes)
    if not episodes:
        raise RuntimeError(f"No episodes found in {args.data}")

    stats = compute_stats(episodes)
    episode_ids = list(range(len(episodes)))
    random.shuffle(episode_ids)
    val_ep_count = max(1, int(len(episode_ids) * args.val_split))
    val_eps = sorted(episode_ids[:val_ep_count])
    if args.max_val_episodes is not None:
        val_eps = val_eps[: args.max_val_episodes]

    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    model = build_model_from_checkpoint(ckpt, args.sensor_mode, args.device)
    if args.sensor_head == "z" and getattr(model, "z_obs_head", None) is None:
        raise RuntimeError("Selected --sensor-head z, but checkpoint/model has no z_obs_head.")
    if args.sensor_head == "h" and getattr(model, "h_obs_head", None) is None:
        raise RuntimeError("Selected --sensor-head h, but checkpoint/model has no h_obs_head.")
    sensor_min_idx = torch.tensor(stats.sensor_min, dtype=torch.float32, device=args.device)
    loc_min_xy = torch.tensor(stats.loc_min, dtype=torch.long, device=args.device)
    turn_max = float(ckpt["stats"]["action_turn_max"])
    step_max = float(ckpt["stats"]["action_step_max"])

    maze_cache: Dict[Tuple[int, int, int], Maze] = {}
    temp_files: List[str] = []

    tested_eps = 0
    skipped_eps = 0
    total_steps = 0
    total_lr_both = 0
    total_loc_both = 0
    final_steps = 0
    final_lr_both = 0
    final_loc_both = 0
    after_turn_steps = 0
    after_turn_lr_both = 0
    after_turn_loc_both = 0

    with torch.no_grad():
        for ep_idx in val_eps:
            ep = episodes[ep_idx]
            if len(ep) <= args.n_step + 1:
                skipped_eps += 1
                continue
            meta = parse_map_meta(map_ids[ep_idx])
            if meta is None:
                skipped_eps += 1
                continue
            if meta not in maze_cache:
                dim, extra_openings, seed = meta
                walls = generate_maze(dim, seed=seed, extra_openings=extra_openings)
                fd, path = tempfile.mkstemp(prefix="maze_eval_", suffix=".txt")
                os.close(fd)
                write_maze(path, dim, walls)
                temp_files.append(path)
                maze_cache[meta] = Maze(path)
            maze = maze_cache[meta]

            warm = ep[: args.n_step + 1]
            obs_sensor_np = np.asarray(
                [[np.asarray(s["sensor"], dtype=np.float32)[:3] for s in warm]],
                dtype=np.float32,
            )
            obs_sensor = torch.from_numpy(obs_sensor_np).to(args.device)
            obs_loc_np = np.asarray(
                [[np.asarray(s["location"], dtype=np.float32) for s in warm]],
                dtype=np.float32,
            )
            obs_loc = torch.from_numpy(obs_loc_np).to(args.device)
            obs_head = torch.tensor(
                [[stats.heading_to_idx[str(s["heading"])] for s in warm]],
                dtype=torch.long,
                device=args.device,
            )
            obs_actions = torch.tensor(
                [
                    [
                        [
                            float(s["action"][0]) / turn_max if turn_max != 0 else 0.0,
                            float(s["action"][1]) / step_max if step_max != 0 else 0.0,
                        ]
                        for s in warm
                    ]
                ],
                dtype=torch.float32,
                device=args.device,
            )
            kpm = torch.zeros((1, len(warm)), dtype=torch.bool, device=args.device)
            obs_batch = {
                "sensor": obs_sensor,
                "loc": obs_loc,
                "heading": obs_head,
                "actions": obs_actions,
                "key_padding_mask": kpm,
            }

            out = model(obs_batch, return_state=True)
            h_prev = None
            z_prev_flat = None
            if isinstance(model, (RSSMDiscretePredictor, TSSMDiscretePredictor)):
                if not isinstance(out, dict):
                    skipped_eps += 1
                    continue
                last_state = out.get("state")
                if last_state is None:
                    skipped_eps += 1
                    continue
                state_prev = last_state[:, 0, :] if last_state.dim() == 3 else last_state
                h_prev = state_prev[:, : model.hidden_size]
                z_prev_flat = state_prev[:, model.hidden_size :]

            # Build rollout action replay from the warmup boundary state.
            loc = tuple(int(x) for x in warm[-1]["location"])
            heading = str(warm[-1]["heading"])
            rollout_actions = build_rollout_actions(ep, args.n_step, max_rollout_steps)
            if rollout_actions is None:
                skipped_eps += 1
                continue
            if not rollout_actions:
                continue

            # Open-loop prior rollout on selected action sequence.
            roll_loc = loc
            roll_head = heading

            if isinstance(model, TSSMDiscretePredictor):
                detach_on_pop = bool(getattr(model, "bptt_horizon", 0) > 0)
                if model.attention_window is not None and model.attention_window > 0:
                    cache = WindowedPositionBasedDynamicCache(
                        int(model.attention_window),
                        detach_on_pop=detach_on_pop,
                    ).to(device=args.device)
                else:
                    cache = PositionBasedDynamicCache(
                        detach_on_pop=detach_on_pop
                    ).to(device=args.device)
                pos_idx = 0

            ep_total = 0
            ep_correct = 0
            ep_loc_correct = 0
            ep_final_correct = 0
            ep_final_loc_correct = 0
            ep_after_turn_correct: Optional[int] = None
            ep_after_turn_loc_correct: Optional[int] = None

            for ridx, (rot, step) in enumerate(rollout_actions):
                a_norm = normalize_action(rot, step, turn_max, step_max).to(args.device)

                if isinstance(model, RSSMDiscretePredictor):
                    assert h_prev is not None and z_prev_flat is not None
                    h_t = model.rnn(torch.cat([z_prev_flat, a_norm], dim=-1), h_prev)
                elif isinstance(model, TSSMDiscretePredictor):
                    assert z_prev_flat is not None
                    trans_in = torch.cat([z_prev_flat, a_norm], dim=-1).unsqueeze(1)
                    pos = torch.tensor([[pos_idx]], dtype=torch.long, device=args.device)
                    h_step = model.transition(
                        trans_in,
                        past_key_values=cache,
                        cache_position=pos,
                        attention_window=model.attention_window,
                    )
                    h_t = h_step[:, -1, :]
                    pos_idx += 1
                elif isinstance(model, TransformerBaseline):
                    # Deterministic non-latent rollout: feed predicted sensor as next input.
                    if ridx == 0:
                        roll_sensor_in = obs_sensor[:, -1:, :]
                    roll_obs = {
                        "sensor": roll_sensor_in,
                        "loc": torch.zeros((1, 1, 2), dtype=torch.float32, device=args.device),
                        "heading": torch.zeros((1, 1), dtype=torch.long, device=args.device),
                        "actions": a_norm.unsqueeze(1),
                        "key_padding_mask": torch.zeros((1, 1), dtype=torch.bool, device=args.device),
                    }
                    pred_sensor, pred_loc_x_logits, pred_loc_y_logits, _pred_head, _pred_turn, _pred_step = model(roll_obs)
                    if args.sensor_mode == "categorical":
                        pred_l_logits, pred_f_logits, pred_r_logits = pred_sensor
                        pred_l = torch.argmax(pred_l_logits, dim=-1).to(torch.float32) + sensor_min_idx[0]
                        pred_f = torch.argmax(pred_f_logits, dim=-1).to(torch.float32) + sensor_min_idx[1]
                        pred_r = torch.argmax(pred_r_logits, dim=-1).to(torch.float32) + sensor_min_idx[2]
                        roll_sensor_in = torch.stack([pred_l, pred_f, pred_r], dim=-1)
                    else:
                        roll_sensor_in = pred_sensor
                    pred_sensor = pred_sensor
                else:
                    skipped_eps += 1
                    ep_total = 0
                    break

                if isinstance(model, (RSSMDiscretePredictor, TSSMDiscretePredictor)):
                    prior_logits = model.prior_head(h_t).view(1, model.stoch_size, model.stoch_classes)
                    z_prior = model._sample_stoch(prior_logits.unsqueeze(1), False).squeeze(1)
                    z_prior_flat = z_prior.reshape(1, model.stoch_flat)
                    feat = torch.cat([h_t, z_prior_flat], dim=-1).unsqueeze(1)
                    if args.sensor_head == "zh":
                        pred_sensor = model._decode_sensor_from_feat(feat)
                    elif args.sensor_head == "z":
                        pred_sensor = model._decode_sensor_from_z(z_prior_flat.unsqueeze(1))
                    else:
                        pred_sensor = model._decode_sensor_from_h(h_t.unsqueeze(1))
                    if pred_sensor is None:
                        raise RuntimeError(f"Decoder head '{args.sensor_head}' is unavailable for this model/checkpoint.")
                    pred_loc_x_logits = model.loc_probe_x(feat)
                    pred_loc_y_logits = model.loc_probe_y(feat)

                roll_loc, roll_head = apply_action(maze, roll_loc, roll_head, rot, step)
                true_l, _true_f, true_r = sense_lfr(maze, roll_loc, roll_head)

                if args.sensor_mode == "categorical":
                    pred_l_logits, _pred_f_logits, pred_r_logits = pred_sensor
                    l_idx = int(torch.argmax(pred_l_logits, dim=-1).item())
                    r_idx = int(torch.argmax(pred_r_logits, dim=-1).item())
                    pred_l = l_idx + int(sensor_min_idx[0].item())
                    pred_r = r_idx + int(sensor_min_idx[2].item())
                    l_ok = pred_l == true_l
                    r_ok = pred_r == true_r
                else:
                    pred_cont = pred_sensor
                    l_ok = int(torch.round(pred_cont[0, 0, 0]).item()) == true_l
                    r_ok = int(torch.round(pred_cont[0, 0, 2]).item()) == true_r

                both_ok = int(l_ok and r_ok)
                pred_x = int(torch.argmax(pred_loc_x_logits, dim=-1).item()) + int(loc_min_xy[0].item())
                pred_y = int(torch.argmax(pred_loc_y_logits, dim=-1).item()) + int(loc_min_xy[1].item())
                loc_ok = int(pred_x == int(roll_loc[0]) and pred_y == int(roll_loc[1]))
                ep_total += 1
                ep_correct += both_ok
                ep_loc_correct += loc_ok
                if ridx == 2:
                    ep_after_turn_correct = both_ok
                    ep_after_turn_loc_correct = loc_ok
                if ridx == len(rollout_actions) - 1:
                    ep_final_correct = both_ok
                    ep_final_loc_correct = loc_ok

                if isinstance(model, (RSSMDiscretePredictor, TSSMDiscretePredictor)):
                    h_prev = h_t
                    z_prev_flat = z_prior_flat

            if ep_total == 0:
                continue

            tested_eps += 1
            total_steps += ep_total
            total_lr_both += ep_correct
            total_loc_both += ep_loc_correct
            final_steps += 1
            final_lr_both += ep_final_correct
            final_loc_both += ep_final_loc_correct
            if ep_after_turn_correct is not None:
                after_turn_steps += 1
                after_turn_lr_both += ep_after_turn_correct
            if ep_after_turn_loc_correct is not None:
                after_turn_loc_both += ep_after_turn_loc_correct

    for p in temp_files:
        try:
            os.remove(p)
        except OSError:
            pass

    if tested_eps == 0 or total_steps == 0:
        print("No episodes were tested. Check --n-step, --val-split, and map metadata.", flush=True)
        return

    step_acc = total_lr_both / total_steps
    step_loc_acc = total_loc_both / total_steps
    after_turn_acc = after_turn_lr_both / after_turn_steps if after_turn_steps > 0 else 0.0
    after_turn_loc_acc = after_turn_loc_both / after_turn_steps if after_turn_steps > 0 else 0.0
    final_acc = final_lr_both / final_steps if final_steps > 0 else 0.0
    final_loc_acc = final_loc_both / final_steps if final_steps > 0 else 0.0
    print(
        f"reverse-path test | max_rollout_steps={max_rollout_steps} "
        f"episodes_tested={tested_eps} skipped={skipped_eps} "
        f"steps={total_steps} sensor_head={args.sensor_head} lr_acc={step_acc:.4f} "
        f"loc_acc={step_loc_acc:.4f} "
        f"after_turn_step_lr_acc={after_turn_acc:.4f} "
        f"after_turn_step_loc_acc={after_turn_loc_acc:.4f} "
        f"final_start_lr_acc={final_acc:.4f} "
        f"final_start_loc_acc={final_loc_acc:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
