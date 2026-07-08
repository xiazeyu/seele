#!/usr/bin/env python
"""M3 — the floor experiment: extrapolate w as T and NFE keep increasing.

Continues the **same** M2 baseline runs (no new workstream): training resumes
from each baseline's ``last.pt`` up to ``floor_steps``, and sampling extends
the NFE axis beyond the M2 grid.  Then fits two asymptote models to ``w²``
along each axis,

    plateau:  w² = c + a·x^(−b)      (c = w∞², the candidate floor)
    decay:    w² = a·x^(−b)          (no floor)

and reports which is preferred (ΔAIC) — **conservatively**: an empirical
plateau over the tested budgets is evidence, not proof of irreducibility,
and if the two fits are indistinguishable the verdict says exactly that.

Run::

    python scripts/m3_floor.py --smoke     # after m2_sweep.py --smoke
    python scripts/m3_floor.py             # after the real m2_sweep.py
    python scripts/m3_floor.py --skip-run  # analyze existing rows only

Training/measurement artifacts go to the *sweep* dir (shared with M2, since
the runs are shared); analysis artifacts go to ``--outdir`` (default
``results/m3``): ``floor.json`` + ``fig_m3_floor``.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from seele.analysis import aggregate_configs, fit_floor_models, select  # noqa: E402
from seele.fm import FMConfig  # noqa: E402
from seele.sweeps import (  # noqa: E402
    SMOKE_FM, execute, load_grids, load_measurements, plan_runs,
)
from seele.targets import TRAIN_TARGETS  # noqa: E402

plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 150, "font.size": 11,
    "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
})


def _axis_points(stats, grids, target: str, axis: str):
    """(x, w², w²_err) along the T axis (at NFE*) or the NFE axis (at T_max)."""
    if axis == "T":
        hits = select(stats, target, N=grids.n_star, sigma=grids.sig_star,
                      nfe=grids.nfe_star)
        xs = sorted({s.T for s in hits})
        key = lambda s: s.T  # noqa: E731
    else:
        hits = select(stats, target, N=grids.n_star, sigma=grids.sig_star,
                      T=grids.floor_steps)
        xs = sorted({s.nfe for s in hits})
        key = lambda s: s.nfe  # noqa: E731
    by_x = {key(s): s for s in hits}
    pts = [by_x[x] for x in xs]
    return (np.array([key(s) for s in pts], float),
            np.array([s.w2_mean for s in pts]),
            np.array([s.w2_err for s in pts]))


def _plot_axis(ax, x, w2, w2e, fits: dict, xlabel: str) -> None:
    ax.errorbar(x, w2, yerr=np.where(np.isfinite(w2e), w2e, 0), fmt="o", ms=5,
                color="tab:blue", label="measured")
    if x.size:
        xs = np.geomspace(x.min(), x.max() * 8, 200)  # extrapolate one octave+
        if fits.get("decay"):
            p = fits["decay"]["params"]
            ax.plot(xs, p["a"] * xs ** (-p["b"]), "--", color="tab:orange",
                    label=f"decay: b={p['b']:.2f} (AIC {fits['decay']['aic']:.1f})")
        if fits.get("plateau"):
            p = fits["plateau"]["params"]
            ax.plot(xs, p["c"] + p["a"] * xs ** (-p["b"]), "-", color="tab:green",
                    label=f"plateau: w∞={p['w_inf']:.4g} "
                          f"(AIC {fits['plateau']['aic']:.1f})")
            if p["c"] > 0:
                ax.axhline(p["c"], color="tab:green", ls=":", lw=1)
        ax.axvline(x.max(), color="k", ls=":", lw=1, alpha=0.5)
        ax.text(x.max(), ax.get_ylim()[0], " tested range ends", fontsize=7,
                rotation=90, va="bottom")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(xlabel); ax.set_ylabel(r"$w^2$")
    ax.legend(fontsize=8)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sweep-dir", type=Path, default=None,
                    help="M2 sweep dir holding runs/ and measurements.csv")
    ap.add_argument("--outdir", type=Path, default=None,
                    help="analysis output (default results/m3)")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--measure-n", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--skip-run", action="store_true",
                    help="skip training/measurement; analyze existing rows")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")

    sweep_dir = args.sweep_dir or Path("results/m2_smoke" if args.smoke
                                       else "results/m2")
    outdir = args.outdir or Path("results/m3_smoke" if args.smoke
                                 else "results/m3")
    outdir.mkdir(parents=True, exist_ok=True)
    grids = load_grids(sweep_dir)
    cfg = SMOKE_FM if args.smoke else FMConfig()
    measure_n = args.measure_n or (40_000 if args.smoke else 500_000)

    if not args.skip_run:
        specs = plan_runs("floor", grids, seeds=tuple(range(args.seeds)))
        print(f"floor continuation: {len(specs)} runs -> T={grids.floor_steps}, "
              f"NFE up to {max(grids.floor_nfe)}")
        execute(specs, grids, cfg, sweep_dir, measure_n=measure_n,
                device=args.device)

    rows = load_measurements(sweep_dir / "measurements.csv")
    if not rows:
        print(f"no measurements in {sweep_dir}")
        return 1
    stats = aggregate_configs(rows)
    targets = [t for t in TRAIN_TARGETS if any(s.target == t for s in stats)]

    results: dict = {}
    fig, axes = plt.subplots(len(targets), 2,
                             figsize=(11, 4.4 * len(targets)), squeeze=False)
    print("\n" + "=" * 72)
    print("M3 — floor experiment: plateau vs continued decay")
    print("=" * 72)
    for i, target in enumerate(targets):
        results[target] = {}
        for j, axis in enumerate(("T", "nfe")):
            x, w2, w2e = _axis_points(stats, grids, target, axis)
            fits = fit_floor_models(x, w2, w2e)
            results[target][axis] = fits
            label = ("training budget T [steps]" if axis == "T"
                     else f"NFE (at T={grids.floor_steps})")
            _plot_axis(axes[i][j], x, w2, w2e, fits, label)
            axes[i][j].set_title(f"{target}: w² vs {axis}")
            print(f"\n{target} / {axis} axis ({fits.get('n_points', 0)} points):")
            if "verdict" in fits:
                print(f"  ΔAIC(decay−plateau) = "
                      f"{fits['delta_aic_decay_minus_plateau']:+.1f}")
                print(f"  verdict: {fits['verdict']}")
            else:
                print(f"  {fits.get('error', 'fits failed')}")

    (outdir / "floor.json").write_text(json.dumps(results, indent=2))
    fig.suptitle("M3 — extrapolating the edge width (conservative floor test)")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(outdir / f"fig_m3_floor.{ext}")
    plt.close(fig)
    print(f"\nartifacts written to {outdir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
