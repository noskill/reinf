#!/usr/bin/env python3
"""Visualize/evaluate a trained maze WM policy checkpoint (no training updates)."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.distributions import Categorical

from agent_utils_wm import MAZE_WM_MODEL_DEFAULTS, add_create_model_args, extract_create_model_args
from wm_joint_agent import WMActionHeadPolicy, create_maze_world_model


THIS_DIR = Path(__file__).resolve().parent
PARENT_DIR = THIS_DIR.parent
MAZE_DIR = THIS_DIR / "maze"

if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))
if str(MAZE_DIR) not in sys.path:
    sys.path.insert(0, str(MAZE_DIR))

from rl_env import MazeVecEnv  # noqa: E402


def _find_args_json(checkpoint: Path, explicit: Optional[str]) -> Optional[Path]:
    if explicit:
        path = Path(explicit)
        return path if path.exists() else None
    candidates = [
        checkpoint.parent.parent / "args.json",  # logs/runs/<run>/checkpoints/ckpt.pt
        checkpoint.parent / "args.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def _load_saved_args(checkpoint: Path, args_json: Optional[str]) -> Dict[str, Any]:
    path = _find_args_json(checkpoint, args_json)
    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as file_obj:
        data = json.load(file_obj)
    if not isinstance(data, dict):
        return {}
    return data


def _saved(saved: Dict[str, Any], key: str, default: Any) -> Any:
    return saved.get(key, default)


def _model_defaults_from_saved(saved: Dict[str, Any]) -> Dict[str, Any]:
    defaults = dict(MAZE_WM_MODEL_DEFAULTS)
    for key in list(defaults.keys()):
        saved_key = f"wm_{key}"
        if saved_key in saved:
            defaults[key] = saved[saved_key]
    return defaults


def _build_parser(saved: Dict[str, Any], inferred_args_json: Optional[Path]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Play/visualize maze policy from joint WM checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--args-json",
        type=str,
        default=str(inferred_args_json) if inferred_args_json is not None else None,
        help="Path to run args.json (auto-inferred from checkpoint if present).",
    )

    parser.add_argument("--episodes", type=int, default=20, help="Stop after this many completed env episodes.")
    parser.add_argument("--max-total-steps", type=int, default=0, help="Optional hard cap (0 disables).")
    parser.add_argument("--num-envs", type=int, default=int(_saved(saved, "num_envs", 1)))
    parser.add_argument("--seed", type=int, default=int(_saved(saved, "seed", 42)))
    parser.add_argument("--device", type=str, default=str(_saved(saved, "device", "cpu")))
    parser.add_argument("--sleep", type=float, default=0.0, help="Sleep between env steps.")
    parser.add_argument("--print-every", type=int, default=20, help="Status print interval in env steps.")
    parser.set_defaults(print_actions=True)
    parser.add_argument(
        "--print-actions",
        dest="print_actions",
        action="store_true",
        help="Print timestep and selected action each step.",
    )
    parser.add_argument(
        "--no-print-actions",
        dest="print_actions",
        action="store_false",
        help="Disable per-step action printing.",
    )
    parser.add_argument("--deterministic", action="store_true", help="Use argmax instead of sampling.")

    parser.set_defaults(render=True)
    parser.add_argument("--render", dest="render", action="store_true")
    parser.add_argument("--no-render", dest="render", action="store_false")

    parser.add_argument("--maze-path", type=str, default=_saved(saved, "maze_path", None))
    parser.add_argument("--random-dim", type=int, default=int(_saved(saved, "random_dim", 10)))
    parser.add_argument(
        "--random-extra-openings",
        type=int,
        default=int(_saved(saved, "random_extra_openings", 0)),
    )
    parser.add_argument(
        "--randomize-each-reset",
        action="store_true",
        default=bool(_saved(saved, "randomize_each_reset", False)),
    )
    parser.add_argument("--max-steps", type=int, default=int(_saved(saved, "max_steps", 256)))

    add_create_model_args(
        parser,
        arg_prefix="wm",
        defaults=_model_defaults_from_saved(saved),
        include_load_path=True,
    )
    parser.add_argument("--wm-contrastive-temp", type=float, default=float(_saved(saved, "wm_contrastive_temp", 0.1)))
    parser.add_argument(
        "--wm-contrastive-discount",
        type=float,
        default=float(_saved(saved, "wm_contrastive_discount", 0.75)),
    )
    parser.add_argument("--wm-sensor-max-bin", type=int, default=int(_saved(saved, "wm_sensor_max_bin", 64)))
    parser.add_argument("--wm-sensor-weight", type=float, default=float(_saved(saved, "wm_sensor_weight", 1.0)))
    parser.add_argument("--wm-loc-weight", type=float, default=float(_saved(saved, "wm_loc_weight", 0.0)))
    parser.add_argument("--wm-head-weight", type=float, default=float(_saved(saved, "wm_head_weight", 0.0)))
    parser.add_argument("--wm-turn-weight", type=float, default=float(_saved(saved, "wm_turn_weight", 0.0)))
    parser.add_argument("--wm-step-weight", type=float, default=float(_saved(saved, "wm_step_weight", 0.0)))
    parser.add_argument("--wm-sensor-sigma", type=float, default=float(_saved(saved, "wm_sensor_sigma", 1.0)))
    parser.add_argument("--wm-pos-sigma", type=float, default=float(_saved(saved, "wm_pos_sigma", 1.0)))
    parser.add_argument(
        "--wm-heading-smoothing",
        type=float,
        default=float(_saved(saved, "wm_heading_smoothing", 0.0)),
    )
    return parser


def parse_args() -> argparse.Namespace:
    if any(flag in sys.argv for flag in ("-h", "--help")):
        parser = _build_parser(saved={}, inferred_args_json=None)
        return parser.parse_args()

    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--checkpoint", type=str, required=True)
    pre.add_argument("--args-json", type=str, default=None)
    pre_args, _ = pre.parse_known_args()

    checkpoint = Path(pre_args.checkpoint)
    inferred_args_json = _find_args_json(checkpoint, pre_args.args_json)
    saved = _load_saved_args(checkpoint, pre_args.args_json)
    parser = _build_parser(saved, inferred_args_json)
    return parser.parse_args()


def _extract_action_table_from_checkpoint(checkpoint: Dict[str, Any]) -> Optional[Sequence[Tuple[int, int]]]:
    if not isinstance(checkpoint, dict):
        return None
    agent_state = checkpoint.get("agent_state", {})
    if not isinstance(agent_state, dict):
        return None
    policy_state = agent_state.get("policy_state", {})
    if not isinstance(policy_state, dict):
        return None

    table_tensor = None
    policy_blob = policy_state.get("policy", {})
    if isinstance(policy_blob, dict):
        table_tensor = policy_blob.get("action_cont_table", None)
    if table_tensor is None:
        table_tensor = policy_state.get("action_cont_table", None)
    if table_tensor is None:
        return None

    table_t = torch.as_tensor(table_tensor).detach().cpu()
    if table_t.ndim != 2 or table_t.shape[1] != 2:
        return None
    return [(int(x), int(y)) for x, y in table_t.tolist()]


def _load_wm_weights_from_checkpoint(wm_model: torch.nn.Module, checkpoint: Dict[str, Any]) -> None:
    state = checkpoint.get("agent_state", checkpoint)
    if isinstance(state, dict) and "wm_model" in state:
        wm_model.load_state_dict(state["wm_model"])
        return
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        wm_model.load_state_dict(checkpoint["model_state"])
        return
    if isinstance(checkpoint, dict):
        wm_model.load_state_dict(checkpoint)
        return
    raise RuntimeError("Unsupported checkpoint format.")


def main() -> None:
    args = parse_args()

    if args.render and args.num_envs != 1:
        raise ValueError("--render requires --num-envs 1")
    if args.wm_contrastive_dim <= 0:
        raise ValueError("--wm-contrastive-dim must be > 0")
    if args.wm_sensor_max_bin < 1:
        raise ValueError("--wm-sensor-max-bin must be >= 1")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    action_table = _extract_action_table_from_checkpoint(checkpoint)
    env_kwargs = dict(
        num_envs=args.num_envs,
        maze_path=args.maze_path,
        random_dim=args.random_dim,
        random_extra_openings=args.random_extra_openings,
        randomize_each_reset=args.randomize_each_reset,
        max_steps=args.max_steps,
        render=args.render,
        seed=args.seed,
        auto_reset=True,
        return_torch=True,
        device=args.device,
    )
    if action_table is not None:
        env_kwargs["action_table"] = action_table
    env = MazeVecEnv(**env_kwargs)

    maze_dim = int(env._mazes[0].dim)
    turn_bins = len({int(turn) for turn, _ in env.action_table})
    step_bins = len({int(step) for _, step in env.action_table})
    wm_model_args = extract_create_model_args(args, arg_prefix="wm", device=args.device)
    wm_model = create_maze_world_model(
        model_args=wm_model_args,
        device=device,
        maze_dim=maze_dim,
        turn_bins=turn_bins,
        step_bins=step_bins,
        contrastive_temp=args.wm_contrastive_temp,
        contrastive_horizon_discount=args.wm_contrastive_discount,
        sensor_weight=args.wm_sensor_weight,
        loc_weight=args.wm_loc_weight,
        head_weight=args.wm_head_weight,
        turn_weight=args.wm_turn_weight,
        step_weight=args.wm_step_weight,
        sensor_sigma=args.wm_sensor_sigma,
        pos_sigma=args.wm_pos_sigma,
        heading_smoothing=args.wm_heading_smoothing,
        sensor_max_bin=args.wm_sensor_max_bin,
    )
    _load_wm_weights_from_checkpoint(wm_model, checkpoint)
    wm_model.eval()

    policy = WMActionHeadPolicy(
        wm_model=wm_model,
        action_table=env.action_table,
        device=device,
        backprop_through_wm=False,
    )
    policy.eval()
    policy.reset_policy_state()

    obs = env.reset(seed=args.seed)
    episode_start = torch.ones(env.num_envs, dtype=torch.bool, device=device)
    completed = 0
    total_steps = 0
    hit_goal_events = 0
    done_lengths = []

    print(
        f"Loaded checkpoint: {args.checkpoint}\n"
        f"Run: num_envs={env.num_envs} max_steps={env.max_steps} "
        f"render={args.render} deterministic={args.deterministic}",
        flush=True,
    )

    try:
        while completed < args.episodes:
            with torch.no_grad():
                logits = policy(obs, episode_start=episode_start)
                if args.deterministic:
                    action = torch.argmax(logits, dim=-1).to(torch.int32)
                else:
                    action = Categorical(logits=logits).sample().to(torch.int32)
            policy.record_sampled_actions(action)

            obs, _reward, done, info = env.step(action)
            done_t = done.to(torch.bool)
            episode_start = done_t
            total_steps += 1
            if args.print_actions:
                action_vals = [int(x) for x in action.detach().cpu().view(-1).tolist()]
                if env.num_envs == 1:
                    idx = action_vals[0]
                    rot, mov = env.action_table[idx]
                    print(f"t={total_steps} action_idx={idx} rotation={int(rot)} movement={int(mov)}", flush=True)
                else:
                    print(f"t={total_steps} action_idx={action_vals}", flush=True)

            per_env = info.get("per_env", None)
            if per_env is not None:
                hit_goal_events += int(sum(1 for item in per_env if item.get("hit_goal", False)))

            new_done = int(done_t.sum().item())
            if new_done > 0:
                done_cpu = done_t.detach().cpu().tolist()
                if per_env is not None:
                    lengths = [int(per_env[i]["step_count"]) for i, d in enumerate(done_cpu) if d]
                    done_lengths.extend(lengths)
                completed += new_done
                print(f"completed={completed}/{args.episodes} new_done={new_done}", flush=True)

            if args.print_every > 0 and (total_steps % args.print_every == 0):
                sensor0 = obs["sensor"][0].detach().cpu().tolist()
                heading0 = int(obs["heading_idx"][0].detach().cpu())
                loc0 = obs["location"][0].detach().cpu().tolist()
                print(
                    f"step={total_steps} completed={completed}/{args.episodes} "
                    f"hit_goal_events={hit_goal_events} sensor0={sensor0} heading0={heading0} loc0={loc0}",
                    flush=True,
                )

            if args.max_total_steps > 0 and total_steps >= args.max_total_steps:
                print(f"Stopping at max_total_steps={args.max_total_steps}", flush=True)
                break

            if args.sleep > 0:
                time.sleep(args.sleep)
    finally:
        env.close()

    mean_len = float(np.mean(done_lengths)) if done_lengths else 0.0
    print(
        f"Finished: completed={completed} total_steps={total_steps} "
        f"mean_done_length={mean_len:.2f} hit_goal_events={hit_goal_events}",
        flush=True,
    )


if __name__ == "__main__":
    main()
