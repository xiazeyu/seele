#!/usr/bin/env python
"""M2 — run the scaling sweeps: train FM models and measure generated widths.

Trains the flow-matching runs of the requested preset and measures the edge
width of their generated samples over the (T, NFE) measurement grid.  The
run matrix factorizes (see :mod:`seele.sweeps`): T is read off checkpoints,
NFE is chosen at sampling time, so only (N, sigma_min) combinations cost a
training run.

Presets
-------
``isolated``  baseline + N sweep + sigma_min sweep         (the H2 laws)
``joint``     extra N x sigma crossed runs                 (the H1 test)
``heldout``   held-out-target runs                         (M4 evaluation)
``all``       isolated + joint + heldout                   (default)

Everything is resumable: finished checkpoints and measured (T, NFE, edge)
rows are skipped, so re-running after an interruption only does new work.

Run::

    python scripts/m2_sweep.py --smoke            # minutes-scale pipeline check
    python scripts/m2_sweep.py                    # the real study (hours)
    python scripts/m2_sweep.py --preset heldout   # only the M4 eval runs

Outputs (under ``--outdir``, default ``results/m2``): ``runs/<run_id>/`` with
checkpoints, ``grids.json``, and ``measurements.csv`` (one row per edge per
(run, T, NFE) measurement).
"""

from __future__ import annotations

import argparse
import logging
from collections import Counter
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from seele.fm import FMConfig, pick_device  # noqa: E402
from seele.sweeps import (  # noqa: E402
    DEFAULT_GRIDS, SMOKE_FM, SMOKE_GRIDS, execute, measurement_pairs, plan_runs,
)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", default="all",
                    choices=["isolated", "joint", "heldout", "all"])
    ap.add_argument("--targets", nargs="*", default=None,
                    help="restrict targets (default: all of the preset's)")
    ap.add_argument("--seeds", type=int, default=3,
                    help="training seeds per config (heldout uses min(2, seeds))")
    ap.add_argument("--outdir", type=Path, default=None,
                    help="default results/m2 (results/m2_smoke with --smoke)")
    ap.add_argument("--measure-n", type=int, default=None,
                    help="samples per width measurement (default 500k; 40k smoke)")
    ap.add_argument("--solver", default="euler",
                    choices=["euler", "midpoint", "rk4"])
    ap.add_argument("--device", default=None, help="cuda | mps | cpu (default: auto)")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny grids + tiny net: end-to-end pipeline check")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan and exit without training")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")

    grids = SMOKE_GRIDS if args.smoke else DEFAULT_GRIDS
    cfg = SMOKE_FM if args.smoke else FMConfig()
    outdir = args.outdir or Path("results/m2_smoke" if args.smoke else "results/m2")
    measure_n = args.measure_n or (40_000 if args.smoke else 500_000)
    seeds = tuple(range(args.seeds))
    heldout_seeds = tuple(range(min(2, args.seeds)))

    specs = plan_runs(args.preset, grids, targets=args.targets,
                      seeds=seeds, heldout_seeds=heldout_seeds)
    n_meas = sum(len(measurement_pairs(s, grids)) for s in specs)
    by_role = Counter(s.role for s in specs)

    print(f"preset={args.preset}  smoke={args.smoke}  device={pick_device(args.device)}")
    print(f"training runs: {len(specs)}  ({dict(by_role)})")
    print(f"measurements (T, NFE) pairs: {n_meas}  x {measure_n:,} samples each")
    print(f"outdir: {outdir}\n")
    if args.dry_run:
        for s in specs:
            print(f"  {s.role:<9s} {s.run_id:<40s} steps={s.steps} "
                  f"pairs={len(measurement_pairs(s, grids))}")
        return 0

    csv_path = execute(specs, grids, cfg, outdir, measure_n=measure_n,
                       solver=args.solver, device=args.device)
    print(f"\nmeasurements written to {csv_path}")
    print("next: python scripts/m2_analyze.py"
          + (" --smoke" if args.smoke else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
