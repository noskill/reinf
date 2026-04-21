#!/usr/bin/env python3
"""Lightweight TensorBoard scalar tail for maze WM runs."""

from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable, List

from tensorboard.backend.event_processing import event_accumulator


DEFAULT_METRICS: List[str] = [
    "joint/loss_total",
    "wm/loss/total",
    "wm/loss/contrastive",
    "wm/metric/contrastive_acc",
    "policy/loss",
    "policy/entropy_mean",
    "reward/intrinsic_mean",
    "reward/total_mean",
    "Average episode length",
]


def _find_latest_run(root: Path) -> Path:
    if not root.exists():
        raise FileNotFoundError(f"Run root not found: {root}")
    run_dirs = [d for d in root.iterdir() if d.is_dir()]
    run_dirs = [d for d in run_dirs if list(d.glob("events.out.tfevents*"))]
    if not run_dirs:
        raise FileNotFoundError(f"No TensorBoard event runs found under: {root}")
    return max(run_dirs, key=lambda d: d.stat().st_mtime)


def _resolve_event_file(run: str | None) -> Path:
    if run is None:
        run_dir = _find_latest_run(Path("logs/runs"))
    else:
        run_path = Path(run)
        if run_path.is_file():
            return run_path
        run_dir = run_path
    events = sorted(run_dir.glob("events.out.tfevents*"), key=lambda p: p.stat().st_mtime)
    if not events:
        raise FileNotFoundError(f"No event files in run: {run_dir}")
    return events[-1]


def _format_metric_line(values: Iterable[float]) -> str:
    step, last, mean = values
    return f"{last:.6g}@{int(step)} (avg={mean:.6g})"


def _snapshot(event_file: Path, metrics: List[str], last_n: int) -> str:
    ea = event_accumulator.EventAccumulator(
        str(event_file),
        size_guidance={event_accumulator.SCALARS: 0},
    )
    ea.Reload()
    tags = set(ea.Tags().get("scalars", []))
    parts: List[str] = []
    for tag in metrics:
        if tag not in tags:
            continue
        series = ea.Scalars(tag)
        if not series:
            continue
        last = series[-1]
        recent = series[-last_n:]
        mean_recent = sum(x.value for x in recent) / len(recent)
        parts.append(f"{tag}: {_format_metric_line((last.step, last.value, mean_recent))}")
    if not parts:
        return "No matching scalar data yet."
    return " | ".join(parts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tail TensorBoard scalar metrics.")
    parser.add_argument(
        "--run",
        type=str,
        default=None,
        help="Run dir or event file path. Default: latest run in logs/runs.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=5.0,
        help="Polling interval in seconds.",
    )
    parser.add_argument(
        "--last-n",
        type=int,
        default=20,
        help="Window size for recent average.",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default=",".join(DEFAULT_METRICS),
        help="Comma-separated scalar tags.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Print one snapshot and exit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]

    while True:
        event_file = _resolve_event_file(args.run)
        timestamp = datetime.now().strftime("%H:%M:%S")
        line = _snapshot(event_file, metrics, max(1, int(args.last_n)))
        print(f"[{timestamp}] run={event_file.parent.name} | {line}", flush=True)
        if args.once:
            return
        time.sleep(max(0.5, float(args.interval)))


if __name__ == "__main__":
    main()
