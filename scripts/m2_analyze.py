#!/usr/bin/env python
"""M2 — analyze the sweep measurements: width laws (H2) + additivity (H1).

Reads ``measurements.csv`` produced by ``m2_sweep.py`` and produces

* per-knob scaling exponents of the floor-subtracted excess ``Δw²``
  (log-log power-law fits with seed-level errors) — the H2 deliverable;
* the additivity test of the width budget at joint configurations
  (interaction residuals; see ``docs/questions-remaining.md`` Q3) — the H1
  deliverable;
* the overshoot diagnostics vs T and NFE (Q1).

Run::

    python scripts/m2_analyze.py            # reads results/m2
    python scripts/m2_analyze.py --smoke    # reads results/m2_smoke

Outputs (under ``<outdir>/analysis``): ``laws.json``, ``additivity.json``,
and the figure gallery (``fig_m2_*``).
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

from seele.analysis import (  # noqa: E402
    KNOB_FIELD, additivity_report, aggregate_configs, fit_all_laws,
    isolated_sweep, knob_star,
)
from seele.sweeps import load_grids, load_measurements  # noqa: E402
from seele.targets import TRAIN_TARGETS  # noqa: E402

plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 150, "font.size": 11,
    "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
})

#: Reference slopes implied by the H2 theory inputs (annotation only):
#: sigma: w ∝ σ ⇒ Δw² slope 2;   NFE: Euler ⇒ w ∝ 1/NFE ⇒ slope −2;
#: N: 2-D KDE bandwidth ⇒ w ∝ N^{−1/6} ⇒ slope −1/3;
#: T: last-converged mode of the |k|⁻² tail ⇒ w ∝ T^{−1/2} ⇒ slope −1.
H2_REFERENCE_SLOPE = {"sigma": 2.0, "nfe": -2.0, "N": -1.0 / 3.0, "T": -1.0}

KNOB_LABEL = {"T": "training budget T [steps]", "N": "training-set size N",
              "nfe": "NFE", "sigma": r"$\sigma_{\min}$"}
COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def _save(fig, outdir: Path, name: str) -> None:
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(outdir / f"{name}.{ext}")
    plt.close(fig)


# ── Figures ──────────────────────────────────────────────────────────────────

def fig_laws(outdir: Path, all_laws: dict) -> None:
    """2x2 log-log panels: Δw² vs knob with fitted power laws per target."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    for ax, knob in zip(axes.ravel(), KNOB_FIELD):
        for i, (target, laws_d) in enumerate(all_laws.items()):
            kl = laws_d["laws"][knob]
            x = np.asarray(kl.x); d = np.asarray(kl.dw2); e = np.asarray(kl.dw2_err)
            pos = d > 0
            ax.errorbar(x[pos], d[pos], yerr=np.where(e[pos] > 0, e[pos], np.nan),
                        fmt="o", ms=5, color=COLORS[i], alpha=0.85,
                        label=f"{target}")
            if np.any(~pos):
                ax.plot(x[~pos], np.abs(d[~pos]), "v", ms=5, mfc="none",
                        color=COLORS[i], alpha=0.5)
            if kl.law is not None:
                xs = np.geomspace(x[x > 0].min(), x[x > 0].max(), 100)
                ax.plot(xs, kl.law(xs), "-", color=COLORS[i], lw=1.5,
                        label=f"fit: slope {kl.law.exponent:+.2f}"
                              f"$\\pm${kl.law.exponent_err:.2f}")
        # H2 reference slope, anchored at the last fitted point of target 0
        kl0 = next(iter(all_laws.values()))["laws"][knob]
        if kl0.law is not None and len(kl0.x) >= 2:
            xs = np.geomspace(min(kl0.x), max(kl0.x), 50)
            ref = H2_REFERENCE_SLOPE[knob]
            anchor_x, anchor_y = max(kl0.x), float(kl0.law(max(kl0.x)))
            ax.plot(xs, anchor_y * (xs / anchor_x) ** ref, "k:", lw=1,
                    label=f"H2 ref slope {ref:+.2g}")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(KNOB_LABEL[knob])
        ax.set_ylabel(r"$\Delta w^2$ (floor-subtracted)")
        ax.set_title(f"isolated {knob} sweep")
        ax.legend(fontsize=7)
    fig.suptitle("M2 — per-knob width laws (H2)", y=1.0)
    _save(fig, outdir, "fig_m2_laws")


def fig_additivity(outdir: Path, reports: dict) -> None:
    """Predicted vs measured w² at joint configs + per-pair residuals."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))

    ax = axes[0]
    for i, (target, rep) in enumerate(reports.items()):
        m = np.array([e["w2_meas"] for e in rep["entries"]])
        p = np.array([e["w2_pred"] for e in rep["entries"]])
        ax.plot(p, m, "o", ms=5, color=COLORS[i], alpha=0.8, label=target)
    lims = ax.get_xlim() + ax.get_ylim()
    lo, hi = min(v for v in lims if v > 0), max(lims)
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"additive prediction $\hat w^2$")
    ax.set_ylabel(r"measured $w^2$")
    ax.set_title("H1: joint configs vs additive budget")
    ax.legend(fontsize=8)

    ax = axes[1]
    rel_all = []
    for i, (target, rep) in enumerate(reports.items()):
        rel = [e["rel_residual"] for e in rep["entries"]
               if np.isfinite(e["rel_residual"])]
        rel_all.extend(rel)
        ax.hist(rel, bins=20, alpha=0.6, color=COLORS[i], label=target)
    ax.axvline(0, color="k", lw=1)
    if rel_all:
        ax.axvline(float(np.median(rel_all)), color="r", ls=":",
                   label=f"median={np.median(rel_all):+.2%}")
    ax.set_xlabel(r"interaction residual $(w^2-\hat w^2)/w^2$")
    ax.set_ylabel("configs")
    ax.set_title("residual distribution")
    ax.legend(fontsize=8)

    ax = axes[2]
    pairs, meds = [], []
    for target, rep in reports.items():
        for pair, st in rep["summary"]["by_pair"].items():
            pairs.append(f"{target}\n{pair}")
            meds.append(st["median_abs"])
    if pairs:
        ax.bar(range(len(pairs)), meds, color="tab:blue", alpha=0.8)
        ax.set_xticks(range(len(pairs)))
        ax.set_xticklabels(pairs, fontsize=7)
        ax.set_ylabel("median |rel. residual|")
    ax.set_title("interaction size by knob pair")
    _save(fig, outdir, "fig_m2_additivity")


def fig_budget(outdir: Path, all_laws: dict, reports: dict) -> None:
    """Stacked predicted budget vs measured w² per joint config (the Q3 visual)."""
    n_t = len(reports)
    fig, axes = plt.subplots(1, max(n_t, 1), figsize=(7 * max(n_t, 1), 4.6),
                             squeeze=False)
    part_colors = {"floor": "0.6", "T": COLORS[0], "N": COLORS[1],
                   "nfe": COLORS[2], "sigma": COLORS[3]}
    for ax, (target, rep) in zip(axes[0], reports.items()):
        laws_d = all_laws[target]
        entries = sorted(rep["entries"], key=lambda e: e["w2_meas"])
        xs = np.arange(len(entries))
        bottom = np.zeros(len(entries))
        vals = {"floor": np.full(len(entries), laws_d["floor_w2"])}
        for k in KNOB_FIELD:
            vals[k] = np.array([
                laws_d["laws"][k].delta_w2(
                    {"T": e["T"], "N": e["N"], "nfe": e["nfe"],
                     "sigma": e["sigma"]}[k])
                for e in entries])
        for part, v in vals.items():
            ax.bar(xs, v, bottom=bottom, color=part_colors[part],
                   label=part, width=0.7)
            bottom += v
        ax.errorbar(xs, [e["w2_meas"] for e in entries],
                    yerr=[e["w2_err"] if np.isfinite(e["w2_err"]) else 0
                          for e in entries],
                    fmt="k_", ms=14, lw=1.4, capsize=3, label="measured $w^2$")
        ax.set_xticks(xs)
        ax.set_xticklabels(
            [f"T={e['T']}\nN={e['N']}\nk={e['nfe']}\nσ={e['sigma']:g}"
             for e in entries], fontsize=6)
        ax.set_ylabel(r"$w^2$")
        ax.set_title(f"{target}: stacked budget vs measurement")
        ax.legend(fontsize=7)
    _save(fig, outdir, "fig_m2_budget")


def fig_interaction(outdir: Path, reports: dict) -> None:
    """Interaction-residual heatmaps over each 2-knob grid (the Q3 visual:
    structure in these maps — not noise — is the falsification signature)."""
    KNOB_SHORT = {"T": "T", "N": "N", "nfe": "NFE", "sigma": "σ"}
    for target, rep in reports.items():
        pairs: dict[tuple, list[dict]] = {}
        for e in rep["entries"]:
            if len(e["knobs"]) == 2 and np.isfinite(e["rel_residual"]):
                pairs.setdefault(tuple(e["knobs"]), []).append(e)
        pairs = {p: es for p, es in pairs.items() if len(es) >= 2}
        if not pairs:
            continue
        fig, axes = plt.subplots(1, len(pairs),
                                 figsize=(3.6 * len(pairs), 3.6), squeeze=False)
        field = {"T": "T", "N": "N", "nfe": "nfe", "sigma": "sigma"}
        for ax, (pair, es) in zip(axes[0], pairs.items()):
            ka, kb = pair
            avals = sorted({e[field[ka]] for e in es})
            bvals = sorted({e[field[kb]] for e in es})
            M = np.full((len(bvals), len(avals)), np.nan)
            for e in es:
                M[bvals.index(e[field[kb]]), avals.index(e[field[ka]])] = \
                    e["rel_residual"]
            vmax = max(np.nanmax(np.abs(M)), 1e-3)
            im = ax.imshow(M, origin="lower", cmap="coolwarm",
                           vmin=-vmax, vmax=vmax, aspect="auto")
            ax.set_xticks(range(len(avals)))
            ax.set_xticklabels([f"{v:g}" for v in avals], fontsize=7)
            ax.set_yticks(range(len(bvals)))
            ax.set_yticklabels([f"{v:g}" for v in bvals], fontsize=7)
            ax.set_xlabel(KNOB_SHORT[ka]); ax.set_ylabel(KNOB_SHORT[kb])
            ax.set_title(f"{KNOB_SHORT[ka]} × {KNOB_SHORT[kb]}", fontsize=9)
            fig.colorbar(im, ax=ax, shrink=0.85,
                         label=r"$(w^2-\hat w^2)/w^2$")
        fig.suptitle(f"{target} — interaction residuals of the additive budget")
        _save(fig, outdir, f"fig_m2_interaction_{target}")


def fig_overshoot(outdir: Path, stats, grids) -> None:
    """Overshoot amplitude vs NFE and vs T (Q1 diagnostics)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, knob in zip(axes, ("nfe", "T")):
        for i, target in enumerate(t for t in TRAIN_TARGETS):
            sweep = isolated_sweep(stats, grids, target, knob)
            x = [s.knob(knob) for s in sweep]
            y = [s.overshoot_mean for s in sweep]
            if not x:
                continue
            ax.plot(x, y, "o-", color=COLORS[i], label=target)
        ax.set_xscale("log")
        ax.axhline(0, color="k", lw=1)
        ax.set_xlabel(KNOB_LABEL[knob])
        ax.set_ylabel("overshoot / step height")
        ax.set_title(f"edge overshoot vs {knob}")
        ax.legend(fontsize=8)
    fig.suptitle("Q1 — overshoot diagnostics (see docs/questions-remaining.md)")
    _save(fig, outdir, "fig_m2_overshoot")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", type=Path, default=None,
                    help="sweep output dir (default results/m2)")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)

    outdir = args.outdir or Path("results/m2_smoke" if args.smoke else "results/m2")
    adir = outdir / "analysis"
    adir.mkdir(parents=True, exist_ok=True)

    rows = load_measurements(outdir / "measurements.csv")
    if not rows:
        print(f"no measurements found in {outdir} — run m2_sweep.py first")
        return 1
    grids = load_grids(outdir)
    stats = aggregate_configs(rows)
    targets = [t for t in TRAIN_TARGETS
               if any(s.target == t for s in stats)]

    all_laws, reports = {}, {}
    for target in targets:
        all_laws[target] = fit_all_laws(stats, grids, target)
        reports[target] = additivity_report(stats, grids, target,
                                            all_laws[target])

    # ── console summary ──
    print("=" * 72)
    print("M2 — width laws (H2): Δw² = amp · knob^slope   (floor-subtracted)")
    print("=" * 72)
    for target, laws_d in all_laws.items():
        print(f"\n{target}:  floor w² = {laws_d['floor_w2']:.3e} "
              f"(w = {np.sqrt(max(laws_d['floor_w2'], 0)):.4g})")
        for k, kl in laws_d["laws"].items():
            if kl.law is None:
                print(f"  {k:<6s} —  no usable points")
                continue
            print(f"  {k:<6s} slope {kl.law.exponent:+.3f} ± {kl.law.exponent_err:.3f}"
                  f"   (H2 ref {H2_REFERENCE_SLOPE[k]:+.2g}; "
                  f"n={kl.law.n_used}, R²_log={kl.law.r2_log:.3f})")

    print("\n" + "=" * 72)
    print("M2 — additivity of the budget (H1) at joint configurations")
    print("=" * 72)
    any_joint = False
    for target, rep in reports.items():
        s = rep["summary"]
        if s["n_configs"] == 0:
            print(f"\n{target}: no joint configs measured "
                  f"(run m2_sweep.py --preset joint)")
            continue
        any_joint = True
        print(f"\n{target}: {s['n_configs']} joint configs — "
              f"median |rel residual| = {s['median_abs_rel_residual']:.2%}, "
              f"p90 = {s['p90_abs_rel_residual']:.2%}, "
              f"mean (signed) = {s['mean_rel_residual']:+.2%}")
        for pair, st in s["by_pair"].items():
            print(f"    {pair:<12s} n={st['n']:2d}  "
                  f"median|res|={st['median_abs']:.2%}  mean={st['mean']:+.2%}")

    # ── artifacts ──
    (adir / "laws.json").write_text(json.dumps(
        {t: {"floor_w2": d["floor_w2"], "floor_w2_err": d["floor_w2_err"],
             "laws": {k: kl.as_dict() for k, kl in d["laws"].items()}}
         for t, d in all_laws.items()}, indent=2))
    (adir / "additivity.json").write_text(json.dumps(reports, indent=2))

    print("\nRendering figures ...")
    fig_laws(adir, all_laws)
    if any_joint:
        fig_additivity(adir, reports)
        fig_budget(adir, all_laws, reports)
        fig_interaction(adir, reports)
    fig_overshoot(adir, stats, grids)
    print(f"artifacts written to {adir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
