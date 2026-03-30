#!/usr/bin/env python3
"""Benchmark RSSM vs TSSM step speed with matched dimensions.

This script measures wall-clock iteration time for both models on the same
synthetic batch shape and reports relative slowdown.
"""

import argparse
import time
from statistics import mean, pstdev

import numpy as np
import torch

from wm_train import (
    RSSMDiscretePredictor,
    TSSMDiscretePredictor,
    build_sequences,
    compute_stats,
    load_episodes,
    set_seed,
)


def build_bench_obs(
    *,
    batch_size: int,
    seq_len: int,
    stats,
    device: torch.device,
) -> dict:
    # Sensor values are represented in absolute bin coordinates in this codebase.
    smin = torch.tensor(stats.sensor_min, dtype=torch.long, device=device)
    smax = torch.tensor(stats.sensor_max, dtype=torch.long, device=device)
    sensor_channels = []
    for c in range(3):
        sensor_channels.append(
            torch.randint(
                low=int(smin[c].item()),
                high=int(smax[c].item()) + 1,
                size=(batch_size, seq_len),
                device=device,
            )
        )
    sensor = torch.stack(sensor_channels, dim=-1).to(torch.float32)

    lmin = torch.tensor(stats.loc_min, dtype=torch.long, device=device)
    lmax = torch.tensor(stats.loc_max, dtype=torch.long, device=device)
    loc_x = torch.randint(
        low=int(lmin[0].item()),
        high=int(lmax[0].item()) + 1,
        size=(batch_size, seq_len),
        device=device,
    )
    loc_y = torch.randint(
        low=int(lmin[1].item()),
        high=int(lmax[1].item()) + 1,
        size=(batch_size, seq_len),
        device=device,
    )
    loc = torch.stack([loc_x, loc_y], dim=-1).to(torch.float32)

    heading_dim = len(stats.heading_to_idx)
    heading = torch.randint(0, heading_dim, (batch_size, seq_len), device=device, dtype=torch.long)

    # Actions are normalized in training data to roughly [-1, 1].
    actions = torch.empty((batch_size, seq_len, 2), device=device, dtype=torch.float32).uniform_(-1.0, 1.0)
    kpm = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=device)

    return {
        "sensor": sensor,
        "loc": loc,
        "heading": heading,
        "actions": actions,
        "key_padding_mask": kpm,
    }


def reduce_outputs_to_loss(outputs) -> torch.Tensor:
    pred_sensor, loc_x, loc_y, heading, turn, step, *_ = outputs
    loss = loc_x.mean() + loc_y.mean() + heading.mean() + turn.mean() + step.mean()
    if isinstance(pred_sensor, tuple):
        loss = loss + pred_sensor[0].mean() + pred_sensor[1].mean() + pred_sensor[2].mean()
    else:
        loss = loss + pred_sensor.mean()
    return loss


def bench_model(
    *,
    name: str,
    model: torch.nn.Module,
    obs: dict,
    device: torch.device,
    mode: str,
    warmup: int,
    iters: int,
    lr: float,
) -> dict:
    if mode not in {"eval", "train"}:
        raise ValueError(f"Unsupported mode: {mode}")

    use_cuda = device.type == "cuda"
    if mode == "train":
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    else:
        model.eval()
        optimizer = None

    times_ms = []
    for i in range(warmup + iters):
        if use_cuda:
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()

        if mode == "train":
            assert optimizer is not None
            optimizer.zero_grad(set_to_none=True)
            outputs = model(obs)
            loss = reduce_outputs_to_loss(outputs)
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                _ = model(obs)

        if use_cuda:
            torch.cuda.synchronize(device)
        t1 = time.perf_counter()
        if i >= warmup:
            times_ms.append((t1 - t0) * 1000.0)

    avg = mean(times_ms)
    std = pstdev(times_ms) if len(times_ms) > 1 else 0.0
    return {
        "name": name,
        "mode": mode,
        "avg_ms": avg,
        "std_ms": std,
        "iter_per_s": 1000.0 / avg if avg > 0 else 0.0,
    }


def print_result(r: dict):
    print(
        f"{r['name']:>5} | {r['mode']:>5} | "
        f"{r['avg_ms']:8.3f} ms/iter | std {r['std_ms']:7.3f} | {r['iter_per_s']:8.2f} iter/s",
        flush=True,
    )


def main():
    p = argparse.ArgumentParser(description="Benchmark RSSM vs TSSM speed.")
    p.add_argument("--data", type=str, default="data.txt")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=64)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--mode", type=str, choices=["eval", "train", "both"], default="both")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-episodes", type=int, default=1000)
    p.add_argument("--hidden-size", type=int, default=128)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--head-dim", type=int, default=32)
    p.add_argument("--intermediate", type=int, default=512)
    p.add_argument("--obs-latent-dim", type=int, default=64)
    p.add_argument("--stoch-size", type=int, default=8)
    p.add_argument("--stoch-classes", type=int, default=8)
    p.add_argument("--stoch-temp", type=float, default=0.8)
    p.add_argument("--kl-dyn-beta", type=float, default=1.5)
    p.add_argument("--kl-rep-beta", type=float, default=0.05)
    p.add_argument("--kl-free-nats", type=float, default=0.5)
    p.add_argument("--lr", type=float, default=2e-4)
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    if args.hidden_size != args.heads * args.head_dim:
        raise ValueError("hidden-size must equal heads * head-dim")

    episodes = load_episodes(args.data, max_episodes=args.max_episodes)
    if not episodes:
        raise RuntimeError(f"No episodes found in {args.data}")
    stats = compute_stats(episodes)
    sequences = build_sequences(episodes, stats, sensor_mode="categorical")
    if not sequences:
        raise RuntimeError("No usable sequences built from data.")

    sensor_bins = (stats.sensor_max - stats.sensor_min + 1).astype(np.int64)
    loc_x_bins = int(stats.loc_max[0] - stats.loc_min[0] + 1)
    loc_y_bins = int(stats.loc_max[1] - stats.loc_min[1] + 1)
    heading_dim = len(stats.heading_to_idx)
    turn_bins = len(stats.turn_to_idx)
    step_bins = len(stats.step_to_idx)
    obs_dim = int(sequences[0].obs_cont.shape[-1])
    action_dim = int(sequences[0].action_cont.shape[-1])

    obs = build_bench_obs(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        stats=stats,
        device=device,
    )

    rssm = RSSMDiscretePredictor(
        hidden_size=args.hidden_size,
        sensor_mode="categorical",
        sensor_dim=3,
        sensor_bins=sensor_bins,
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
        recon_beta=1.0,
        obs_loss_mode="soft",
        prior_rollout_weight=0.0,
        bptt_horizon=0,
        z_only_weight=0.0,
        h_only_weight=0.0,
        prior_rollout_steps=0,
        probe_hidden_dim=256,
        probe_layers=2,
        transition="gru",
        residual_scale=1.0,
        state_norm="none",
    ).to(device)

    tssm = TSSMDiscretePredictor(
        hidden_size=args.hidden_size,
        layers=args.layers,
        heads=args.heads,
        head_dim=args.head_dim,
        intermediate=args.intermediate,
        attention_window=None,
        sensor_mode="categorical",
        sensor_dim=3,
        sensor_bins=sensor_bins,
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
        recon_beta=1.0,
        obs_loss_mode="soft",
        prior_rollout_weight=0.0,
        bptt_horizon=0,
        z_only_weight=0.0,
        h_only_weight=0.0,
        prior_rollout_steps=0,
        probe_hidden_dim=256,
        probe_layers=2,
    ).to(device)

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    modes = ["eval", "train"] if args.mode == "both" else [args.mode]
    print(
        f"bench config | device={device} batch={args.batch_size} seq_len={args.seq_len} "
        f"warmup={args.warmup} iters={args.iters} modes={modes}",
        flush=True,
    )
    print(
        f"models | rssm_params={sum(p.numel() for p in rssm.parameters()):,} "
        f"tssm_params={sum(p.numel() for p in tssm.parameters()):,}",
        flush=True,
    )

    for mode in modes:
        r_rssm = bench_model(
            name="RSSM",
            model=rssm,
            obs=obs,
            device=device,
            mode=mode,
            warmup=args.warmup,
            iters=args.iters,
            lr=args.lr,
        )
        r_tssm = bench_model(
            name="TSSM",
            model=tssm,
            obs=obs,
            device=device,
            mode=mode,
            warmup=args.warmup,
            iters=args.iters,
            lr=args.lr,
        )
        print_result(r_rssm)
        print_result(r_tssm)
        slowdown = r_tssm["avg_ms"] / r_rssm["avg_ms"] if r_rssm["avg_ms"] > 0 else float("inf")
        print(f"ratio | mode={mode} tssm_vs_rssm={slowdown:.2f}x slower", flush=True)

    if device.type == "cuda":
        peak = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        print(f"cuda peak allocated: {peak:.1f} MiB", flush=True)


if __name__ == "__main__":
    main()
