#!/usr/bin/env python3
"""Run a matrix of main.py settings.

Default matrix:
  datasets: noise4a, noise5a
  models: baseline_aet, full_xyz
  durations: 30, 180, 365 days
"""

from __future__ import annotations

import argparse
import itertools
import subprocess
import sys
from pathlib import Path


DEFAULT_DATASETS = ("noise4a", "noise5a")
DEFAULT_MODELS = ("baseline_aet", "full_xyz")
DEFAULT_DURATIONS = (30.0, 180.0, 365.0)


def _duration_label(days: float) -> str:
    return str(int(days)) if days.is_integer() else f"{days:g}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a dataset/model/duration sweep via main.py.")
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--duration-days", nargs="+", type=float, default=DEFAULT_DURATIONS)
    parser.add_argument("--python", default=sys.executable, help="Python executable to use for child runs.")
    parser.add_argument("--main", type=Path, default=Path(__file__).with_name("main.py"))
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    parser.add_argument(
        "--keep-going",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Continue after a failed child run.",
    )
    return parser.parse_args()


def build_command(args: argparse.Namespace, dataset: str, model: str, duration_days: float) -> list[str]:
    cmd = [
        args.python,
        str(args.main),
        "--dataset",
        dataset,
        "--model",
        model,
        "--duration-days",
        _duration_label(duration_days),
    ]
    return cmd


def main() -> int:
    args = parse_args()
    failures: list[tuple[str, str, float, int]] = []
    settings = list(itertools.product(args.datasets, args.models, args.duration_days))

    for idx, (dataset, model, duration_days) in enumerate(settings, start=1):
        cmd = build_command(args, dataset, model, duration_days)
        print(f"\n[{idx}/{len(settings)}] {' '.join(cmd)}", flush=True)
        if args.dry_run:
            continue
        proc = subprocess.run(cmd, check=False)
        if proc.returncode != 0:
            failures.append((dataset, model, duration_days, proc.returncode))
            if not args.keep_going:
                return proc.returncode

    if failures:
        print("\nFailed runs:", file=sys.stderr)
        for dataset, model, duration_days, returncode in failures:
            print(
                f"  dataset={dataset} model={model} duration_days={duration_days:g} "
                f"returncode={returncode}",
                file=sys.stderr,
            )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
