#!/usr/bin/env python3
"""Plot Tinker DivPO training metrics for a single run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def rolling_mean(values: list[float], window: int) -> list[float]:
    out: list[float] = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        chunk = values[start : i + 1]
        out.append(sum(chunk) / len(chunk))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Tinker DivPO training metrics")
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--title", default="DivPO Training Run")
    parser.add_argument("--window", type=int, default=10)
    args = parser.parse_args()

    metrics_doc = json.load(args.metrics.open())
    ckpt_doc = json.load(args.checkpoints.open())

    metrics = metrics_doc["metrics"]
    steps = [m["step"] for m in metrics]
    loss = [m["loss"] for m in metrics]
    reward_margin = [m["reward_margin"] for m in metrics]
    loss_smooth = rolling_mean(loss, args.window)
    reward_smooth = rolling_mean(reward_margin, args.window)
    checkpoint_steps = [c["step"] for c in ckpt_doc["checkpoints"]]

    args.out.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    axes[0].plot(steps, loss, color="#c26d38", alpha=0.25, linewidth=1.2, label="Loss")
    axes[0].plot(steps, loss_smooth, color="#9c3f12", linewidth=2.2, label=f"Loss ({args.window}-step avg)")
    for step in checkpoint_steps:
        axes[0].axvline(step, color="#d6d6d6", linewidth=0.8, alpha=0.5)
    axes[0].set_ylabel("DPO Loss")
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.2)

    axes[1].plot(steps, reward_margin, color="#4d7fb8", alpha=0.25, linewidth=1.2, label="Reward Margin")
    axes[1].plot(
        steps,
        reward_smooth,
        color="#165a9e",
        linewidth=2.2,
        label=f"Reward Margin ({args.window}-step avg)",
    )
    axes[1].axhline(0.0, color="#666666", linewidth=1.0, alpha=0.8)
    for step in checkpoint_steps:
        axes[1].axvline(step, color="#d6d6d6", linewidth=0.8, alpha=0.5)
    axes[1].set_xlabel("Training Step")
    axes[1].set_ylabel("Reward Margin")
    axes[1].legend(frameon=False)
    axes[1].grid(alpha=0.2)

    fig.suptitle(args.title)
    fig.tight_layout()
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
