#!/usr/bin/env python
"""M4 — real-field sanity check: apply the width tools to external samples.

Loads generated samples from an ``.npz`` (e.g. a ScatterPrism unfolding
output), measures the edge width at a user-specified reference interface,
and — if a fitted budget model plus knob settings are given — compares the
measurement against the a-priori prediction.  The success criterion is
*qualitative* (proposal M4): the predicted-width / trust score should
highlight the expected problematic boundary regions (kinematic cutoffs).

Examples::

    # 1-D kinematic cutoff (density edge at t = -0.4) in column 0
    python scripts/m4_realfield.py --samples unfolded.npz --key points \\
        --edge1d -0.4 --column 0 --width-guess 0.005

    # 2-D radial edge + comparison against the fitted budget model
    python scripts/m4_realfield.py --samples gen.npz --key points \\
        --radial 0 0 1.0 \\
        --budget results/m4/budget_model.json --knobs 600000 16000 0.0 256

The ``.npz`` must hold an ``[N]`` or ``[N, d]`` float array under ``--key``.
Output: measured w (fit + 10-90 cross-check), overshoot diagnostics, the
optional predicted-vs-measured comparison, and a profile figure.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from seele.edgewidth import (  # noqa: E402
    estimate_edge_width, gaussian_convolved_step, profile_hist_1d,
    profile_normal_density, profile_radial_density,
)
from seele.predict import WidthBudgetModel  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--samples", type=Path, required=True, help=".npz file")
    ap.add_argument("--key", default="points", help="array key in the npz")
    edge = ap.add_mutually_exclusive_group(required=True)
    edge.add_argument("--edge1d", type=float, metavar="LOC",
                      help="1-D density edge at LOC (use with --column)")
    edge.add_argument("--radial", type=float, nargs=3,
                      metavar=("CX", "CY", "R"), help="radial edge (2-D)")
    edge.add_argument("--normal", type=float, nargs=3,
                      metavar=("NX", "NY", "OFFSET"),
                      help="planar interface with normal (NX, NY) at OFFSET")
    ap.add_argument("--column", type=int, default=0,
                    help="column for --edge1d on [N, d] arrays")
    ap.add_argument("--width-guess", type=float, default=0.01)
    ap.add_argument("--n-bins", type=int, default=40)
    ap.add_argument("--span", type=float, default=4.0)
    ap.add_argument("--budget", type=Path, default=None,
                    help="budget_model.json from m4_predict.py")
    ap.add_argument("--knobs", type=float, nargs=4, default=None,
                    metavar=("N", "T", "SIGMA", "NFE"),
                    help="knob settings of the run that produced the samples")
    ap.add_argument("--out", type=Path, default=Path("results/m4/realfield"))
    args = ap.parse_args()

    arr = np.load(args.samples)[args.key]
    wg, span, nb = args.width_guess, args.span, args.n_bins
    if args.edge1d is not None:
        x = arr if arr.ndim == 1 else arr[:, args.column]
        window = (args.edge1d - span * wg, args.edge1d + span * wg)
        s, g, sig = profile_hist_1d(x, window, n_bins=nb)
        desc = f"1-D edge at {args.edge1d} (column {args.column})"
    elif args.radial is not None:
        cx, cy, R = args.radial
        window = (max(R - span * wg, 1e-9), R + span * wg)
        s, g, sig = profile_radial_density(arr, np.array([cx, cy]), window,
                                           n_bins=nb)
        desc = f"radial edge R={R} @ ({cx}, {cy})"
    else:
        nx, ny, off = args.normal
        window = (-span * wg, span * wg)
        s, g, sig = profile_normal_density(arr, np.array([nx, ny]), off,
                                           window, n_bins=nb)
        desc = f"interface n=({nx}, {ny}) offset={off}"

    res = estimate_edge_width(s, g, sigma=sig)
    print(f"samples: {args.samples} [{args.key}]  n={len(arr):,}")
    print(f"edge:    {desc}")
    print(f"w_fit        = {res.w_fit:.5g} ± {res.w_fit_err:.2g}")
    print(f"w_from_rise  = {res.w_from_rise:.5g} "
          f"(rel. disagreement {res.rel_disagreement:.1%})")
    print(f"overshoot    = {res.overshoot:+.3f} of step height "
          f"(z = {res.overshoot_z:.1f}, at {res.overshoot_loc_w:+.2f} w)")
    if not res.fit_success:
        print("WARNING: erf fit did not converge — treat numbers as indicative")

    w_pred = None
    if args.budget and args.knobs:
        model = WidthBudgetModel.from_json(args.budget)
        N, T, sigma, nfe = args.knobs
        w_pred = model.predict_w(N=N, T=T, sigma=sigma, nfe=nfe)
        print(f"\na-priori prediction (N={N:g}, T={T:g}, σ={sigma:g}, "
              f"NFE={nfe:g}):")
        print(f"w_pred       = {w_pred:.5g}   "
              f"measured/predicted = {res.w_fit / w_pred:.2f}")

    args.out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.errorbar(s, g, yerr=sig, fmt="o", ms=3.5, alpha=0.7, label="profile")
    ss = np.linspace(s[0], s[-1], 400)
    ax.plot(ss, gaussian_convolved_step(ss, res.level_low, res.level_high,
                                        res.center, res.w_fit),
            "r-", lw=2, label=f"erf fit: w={res.w_fit:.4g}")
    if w_pred is not None:
        ax.axvspan(res.center - w_pred, res.center + w_pred, color="orange",
                   alpha=0.2, label=f"predicted ±ŵ={w_pred:.4g}")
    ax.set_xlabel("interface-normal coordinate [physical units]")
    ax.set_ylabel("density")
    ax.set_title(f"real-field check — {desc}", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    stem = args.out / f"realfield_{args.samples.stem}"
    for ext in ("pdf", "png"):
        fig.savefig(f"{stem}.{ext}")
    print(f"\nfigure written to {stem}.pdf/.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
