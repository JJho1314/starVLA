#!/usr/bin/env python3
"""Parse a WanFastWAM olabots/local training launch log into CSV + PNG.

The trainer logs every `logging_frequency` steps a Python dict containing
{action_dit_loss, video_loss, mse_score, data_time, model_time, lr, epoch,
step}. Rich console wraps the dict across many lines; we flatten the
whitespace and then split on the dict opener so each segment contains at
most one record (this avoids catastrophic backtracking from a long `.*?`
chain against a multi-MB log).

Usage:
    parse_olabots_training_metrics.py --log path/to/launch.log \
        --csv out/metrics.csv --png out/metrics.png

Outputs:
    CSV with columns: step, epoch, lr, action_loss, video_loss, mse_score,
                      data_time, model_time
    PNG with 4 panels: losses / mse / lr / per-step timing
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path


def parse_log(log_path: Path) -> list[dict]:
    text = log_path.read_text(errors="ignore")
    # Strip tqdm carriage returns AND the rich-console multi-line wrapping by
    # collapsing every whitespace run to a single space. Once flat, each
    # metric dict lives on one synthetic line.
    flat = re.sub(r"\s+", " ", text)
    # Split on the dict opener so each segment contains at most one record.
    # The leading element (before the first opener) is non-metric preamble.
    segments = re.split(r"\{'action_dit_loss': ", flat)[1:]

    rows: list[dict] = []
    # The metrics dict carries action_dit_loss, video_loss, mse_score, data_time,
    # model_time, learning_rate, epoch (in that order). The current step is NOT
    # in the dict — it lives in the tqdm progress bar `N/<MAX>` that flushes
    # immediately after the dict closes, e.g. `…'epoch': 0.02})  0%|… 100/21700`.
    field_re = re.compile(
        r"^([0-9.eE+-]+),\s*'video_loss':\s*([0-9.eE+-]+)"
        r".*?'mse_score':\s*([0-9.eE+-]+)"
        r".*?'data_time':\s*([0-9.eE+-]+)"
        r".*?'model_time':\s*([0-9.eE+-]+)"
        r".*?'learning_rate':\s*([0-9.eE+-]+)"
        r".*?'epoch':\s*([0-9.eE+-]+)"
        r"\}\).*?([0-9]+)/[0-9]+\s*\["
    )
    for seg in segments:
        head = seg[:1200]   # dict + first tqdm flush is ~400-800 chars
        m = field_re.search(head)
        if not m:
            continue
        a, v, mse, dt, mt, lr, ep, st = m.groups()
        rows.append({
            "step": int(st),
            "epoch": float(ep),
            "lr": float(lr),
            "action_loss": float(a),
            "video_loss": float(v),
            "mse_score": float(mse),
            "data_time": float(dt),
            "model_time": float(mt),
        })
    # Dedup on step (the trainer can log the same step twice during resume);
    # keep the later one and sort.
    by_step = {r["step"]: r for r in rows}
    return sorted(by_step.values(), key=lambda r: r["step"])


def write_csv(rows: list[dict], path: Path) -> None:
    fieldnames = ["step", "epoch", "lr", "action_loss", "video_loss",
                  "mse_score", "data_time", "model_time"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def write_png(rows: list[dict], path: Path, title: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plot] matplotlib not installed; skipping PNG", file=sys.stderr)
        return

    steps = [r["step"] for r in rows]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(steps, [r["action_loss"] for r in rows], label="action_dit_loss")
    axes[0, 0].plot(steps, [r["video_loss"] for r in rows], label="video_loss")
    axes[0, 0].set_xlabel("step"); axes[0, 0].set_ylabel("loss")
    axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3); axes[0, 0].set_title("training losses")

    axes[0, 1].plot(steps, [r["mse_score"] for r in rows], color="C2")
    axes[0, 1].set_xlabel("step"); axes[0, 1].set_ylabel("mse_score")
    axes[0, 1].grid(alpha=0.3); axes[0, 1].set_title("action MSE (denorm space)")

    axes[1, 0].plot(steps, [r["lr"] for r in rows], color="C3")
    axes[1, 0].set_xlabel("step"); axes[1, 0].set_ylabel("lr")
    axes[1, 0].grid(alpha=0.3); axes[1, 0].set_title("learning rate")

    axes[1, 1].plot(steps, [r["model_time"] for r in rows], label="model_time", alpha=0.7)
    axes[1, 1].plot(steps, [r["data_time"] for r in rows], label="data_time", alpha=0.7)
    axes[1, 1].set_xlabel("step"); axes[1, 1].set_ylabel("seconds")
    axes[1, 1].legend(); axes[1, 1].grid(alpha=0.3); axes[1, 1].set_title("per-step timing")

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=100, bbox_inches="tight")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log", required=True, type=Path)
    ap.add_argument("--csv", required=True, type=Path)
    ap.add_argument("--png", required=True, type=Path)
    ap.add_argument("--title", default=None, help="plot title (default: log file's parent dir name)")
    args = ap.parse_args()

    if not args.log.exists():
        print(f"[fatal] log not found: {args.log}", file=sys.stderr)
        return 2

    rows = parse_log(args.log)
    if not rows:
        print(f"[parse] no metrics found in {args.log} — has the trainer logged any yet?", file=sys.stderr)
        return 3

    write_csv(rows, args.csv)
    print(f"[csv] {len(rows)} rows, last step={rows[-1]['step']} → {args.csv}")

    title = args.title or args.log.stem
    write_png(rows, args.png, title)
    print(f"[png] → {args.png}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
