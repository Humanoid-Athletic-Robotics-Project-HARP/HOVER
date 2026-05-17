"""Print scalar summaries from a TensorBoard log directory (no UI)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tensorboard.backend.event_processing import event_accumulator as ea


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "logdir",
        type=Path,
        nargs="?",
        default=None,
        help="Directory containing events.out.tfevents.*",
    )
    p.add_argument(
        "--glob",
        dest="glob_pattern",
        default="",
        help="If set, pick newest matching child dir under parent (e.g. teacher_h1/*/events*)",
    )
    args = p.parse_args()

    logdir = args.logdir
    if logdir is None:
        # default: newest run under scripts/rsl_rl/logs/teacher_h1
        root = Path(__file__).resolve().parent / "logs" / "teacher_h1"
        runs = sorted(root.glob("*"), key=lambda x: x.stat().st_mtime, reverse=True)
        runs = [r for r in runs if r.is_dir() and any(r.glob("events.out.tfevents*"))]
        if not runs:
            print("No runs with event files under", root, file=sys.stderr)
            return 1
        logdir = runs[0]
        print("Using newest run:", logdir)

    logdir = logdir.resolve()
    if not any(logdir.glob("events.out.tfevents*")):
        print("No events.out.tfevents* in", logdir, file=sys.stderr)
        return 1

    acc = ea.EventAccumulator(str(logdir), size_guidance={ea.SCALARS: 0})
    acc.Reload()
    tags = sorted(acc.Tags().get("scalars", []))
    print("scalar_tags", len(tags))
    for t in tags:
        s = acc.Scalars(t)
        if not s:
            continue
        vals = [x.value for x in s]
        steps = [x.step for x in s]
        print(
            f"{t}\tsteps:{steps[0]}..{steps[-1]} n={len(s)}\t"
            f"first:{vals[0]:.6g}\tlast:{vals[-1]:.6g}\t"
            f"min:{min(vals):.6g}\tmax:{max(vals):.6g}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
