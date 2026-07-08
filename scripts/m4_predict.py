#!/usr/bin/env python
"""M4 — fit the a-priori predicted-width model and evaluate on held-out targets.

1. Fits the :class:`seele.predict.WidthBudgetModel` (floor + four knob laws)
   on the **training targets'** isolated sweeps from M2.
2. Predicts the edge width of every **held-out** configuration from its knob
   settings alone — no access to the held-out geometry or samples — and
   compares against the measured widths (R², relative error): the H3 core
   criterion.
3. Renders the prototype trust maps: per held-out target, the trust field
   ``1 − exp(−d²/2ŵ²)`` at its widest- and narrowest-predicted-width eval
   configs.

Requires the held-out runs to exist::

    python scripts/m2_sweep.py --preset heldout   # (or --preset all)
    python scripts/m4_predict.py

Outputs (under ``--outdir``, default ``results/m4``): ``budget_model.json``
(the fitted predictor — reusable via ``WidthBudgetModel.from_json``),
``heldout_eval.json``, ``fig_m4_pred_vs_meas``, ``fig_m4_trustmap``.
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

from seele.analysis import aggregate_configs  # noqa: E402
from seele.predict import (  # noqa: E402
    WidthBudgetModel, evaluate_heldout, fit_width_budget, trust_grid,
)
from seele.sweeps import load_grids, load_measurements  # noqa: E402
from seele.targets import HELDOUT_TARGETS, TARGETS, TRAIN_TARGETS  # noqa: E402

plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 150, "font.size": 11,
    "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
})
COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def fig_pred_vs_meas(outdir: Path, ev: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))

    ax = axes[0]
    targets = sorted({e["target"] for e in ev["entries"]})
    for i, tgt in enumerate(targets):
        es = [e for e in ev["entries"] if e["target"] == tgt]
        ax.errorbar([e["w_pred"] for e in es], [e["w_meas"] for e in es],
                    yerr=[e["w_meas_err"] if np.isfinite(e["w_meas_err"]) else 0
                          for e in es],
                    fmt="o", ms=6, color=COLORS[i], alpha=0.85, label=tgt)
    vals = [v for e in ev["entries"] for v in (e["w_pred"], e["w_meas"])
            if np.isfinite(v) and v > 0]
    if vals:
        lo, hi = min(vals) * 0.8, max(vals) * 1.25
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="ideal")
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xscale("log"); ax.set_yscale("log")
    s = ev["summary"]
    ax.set_xlabel("predicted width ŵ  (a-priori, before sampling)")
    ax.set_ylabel("measured width w")
    ax.set_title(f"held-out targets: R²={s['r2']:.3f}, "
                 f"median |rel err|={s['median_abs_rel_error']:.1%}")
    ax.legend(fontsize=8)

    ax = axes[1]
    rel = [e["rel_error"] for e in ev["entries"] if np.isfinite(e["rel_error"])]
    ax.hist(rel, bins=16, color="tab:blue", alpha=0.8)
    ax.axvline(0, color="k", lw=1)
    if rel:
        ax.axvline(float(np.median(rel)), color="r", ls=":",
                   label=f"median={np.median(rel):+.1%}")
        ax.legend(fontsize=8)
    ax.set_xlabel("(ŵ − w) / w")
    ax.set_ylabel("held-out configs")
    ax.set_title("prediction residuals")

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(outdir / f"fig_m4_pred_vs_meas.{ext}")
    plt.close(fig)


def fig_trustmap(outdir: Path, ev: dict, n_grid: int = 350) -> None:
    """Per held-out target: sharp target vs trust map at min/max predicted ŵ."""
    targets = sorted({e["target"] for e in ev["entries"]})
    if not targets:
        return
    fig, axes = plt.subplots(3, len(targets),
                             figsize=(3.6 * len(targets), 10.4), squeeze=False)
    rng = np.random.default_rng(0)
    for j, tgt_name in enumerate(targets):
        tgt = TARGETS[tgt_name]
        es = sorted((e for e in ev["entries"] if e["target"] == tgt_name),
                    key=lambda e: e["w_pred"])
        pts = tgt.sample(rng, 150_000)
        x0, x1, y0, y1 = tgt.domain
        axes[0][j].hist2d(pts[:, 0], pts[:, 1], bins=180,
                          range=[[x0, x1], [y0, y1]], cmap="viridis")
        axes[0][j].set_title(f"{tgt_name} (sharp target)", fontsize=9)
        for row, e, tag in ((1, es[0], "best"), (2, es[-1], "worst")):
            X, Y, trust = trust_grid(tgt, e["w_pred"], n=n_grid)
            im = axes[row][j].pcolormesh(X, Y, trust, cmap="RdYlGn",
                                         vmin=0, vmax=1, shading="auto")
            axes[row][j].set_title(
                f"trust @ {tag} config  ŵ={e['w_pred']:.3g}\n"
                f"(T={e['T']}, N={e['N']}, k={e['nfe']}, σ={e['sigma']:g})",
                fontsize=8)
        for i in range(3):
            axes[i][j].set_aspect("equal")
            axes[i][j].set_xticks([]); axes[i][j].set_yticks([])
    fig.colorbar(im, ax=axes[1:, :].ravel().tolist(), shrink=0.5,
                 label="trust score (1 = trustworthy)")
    fig.suptitle("M4 — prototype a-priori trust maps "
                 "(red band = within predicted width of a sharp feature)")
    for ext in ("pdf", "png"):
        fig.savefig(outdir / f"fig_m4_trustmap.{ext}", bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sweep-dir", type=Path, default=None,
                    help="M2 sweep dir (default results/m2)")
    ap.add_argument("--outdir", type=Path, default=None,
                    help="analysis output (default results/m4)")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)

    sweep_dir = args.sweep_dir or Path("results/m2_smoke" if args.smoke
                                       else "results/m2")
    outdir = args.outdir or Path("results/m4_smoke" if args.smoke
                                 else "results/m4")
    outdir.mkdir(parents=True, exist_ok=True)

    rows = load_measurements(sweep_dir / "measurements.csv")
    if not rows:
        print(f"no measurements in {sweep_dir} — run m2_sweep.py first")
        return 1
    grids = load_grids(sweep_dir)
    stats = aggregate_configs(rows)

    fit_targets = tuple(t for t in TRAIN_TARGETS
                        if any(s.target == t for s in stats))
    model = fit_width_budget(stats, grids, fit_targets)
    model.to_json(outdir / "budget_model.json")

    print("=" * 72)
    print("M4 — a-priori width-budget model (fitted on training targets)")
    print("=" * 72)
    print(f"fit targets: {fit_targets}")
    print(f"floor w² = {model.floor_w2:.3e} ± {model.floor_w2_spread:.1e} "
          f"(w_floor = {np.sqrt(max(model.floor_w2, 0)):.4g})")
    for k, kl in model.laws.items():
        if kl.law:
            print(f"  {k:<6s} Δw² = {kl.law.amp:.3e} · x^{kl.law.exponent:+.3f} "
                  f"(± {kl.law.exponent_err:.3f}, n={kl.law.n_used})")
        else:
            print(f"  {k:<6s} — no usable law (Δw² below noise everywhere?)")

    if not any(s.heldout for s in stats):
        print("\nno held-out measurements found — "
              "run: python scripts/m2_sweep.py --preset heldout"
              + (" --smoke" if args.smoke else ""))
        return 1
    ev = evaluate_heldout(stats, model)
    (outdir / "heldout_eval.json").write_text(json.dumps(ev, indent=2))

    s = ev["summary"]
    print("\n" + "=" * 72)
    print("M4 — held-out evaluation (H3): predicted BEFORE sampling")
    print("=" * 72)
    print(f"configs: {s['n_configs']}   R² = {s['r2']:.3f} "
          f"(log-space R² = {s['r2_log']:.3f})")
    print(f"|rel err|: median = {s['median_abs_rel_error']:.1%}, "
          f"p90 = {s['p90_abs_rel_error']:.1%}; "
          f"signed mean = {s['mean_rel_error']:+.1%}")
    for tgt_name in sorted({e['target'] for e in ev['entries']}):
        es = [e for e in ev["entries"] if e["target"] == tgt_name]
        rel = np.array([e["rel_error"] for e in es if np.isfinite(e["rel_error"])])
        print(f"  {tgt_name:<10s} n={len(es)}  "
              f"median|rel err|={np.median(np.abs(rel)):.1%}  "
              f"mean={rel.mean():+.1%}")

    print("\nRendering figures ...")
    fig_pred_vs_meas(outdir, ev)
    fig_trustmap(outdir, ev)
    print(f"artifacts written to {outdir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
