#!/usr/bin/env python
"""M1 — validate the resolution-independent edge-width estimator (the GATE).

Everything downstream of the proposal depends on the edge-width observable being
trustworthy, so before touching a trained flow-matching model we validate the
*measurement tool* against analytic ground truth (see :mod:`seele.synthetic`).

The estimator must satisfy the two M1 done-criteria:

    (1) STABLE across resolutions and reference interfaces
        - invariant to histogram bin width (in the sensible regime),
        - convergent and unbiased as sample size grows,
        - invariant to disc radius, interface orientation, window/span, and — the
          crux of resolution-independence — an overall rescaling of the units;
    (2) fitted ``w`` AGREES with the model-free 10-90 % rise distance.

Run::

    python scripts/m1_validate.py                 # defaults
    python scripts/m1_validate.py --n 2000000 --seeds 6

Outputs (under ``--outdir``, default ``results/m1/validation``): ``results.csv``
(every measurement), ``gate_summary.json`` (metrics + pass/fail), and diagnostic
PDFs.
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

from seele.edgewidth import (  # noqa: E402
    K_1090, estimate_edge_width, gaussian_convolved_step,
    profile_hist_1d, profile_radial_density, profile_normal_density,
    edge_width_box_1d, edge_width_disc, edge_width_interface,
)
from seele import synthetic as syn  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(message)s")
log = logging.getLogger("m1")


# ── Experiments ─────────────────────────────────────────────────────────────

def exp_recovery_vs_wtrue(rng_master, n, seeds, rows):
    widths = [0.01, 0.02, 0.04, 0.08, 0.16]
    out = []
    for w in widths:
        wf, wr = [], []
        for sd in range(seeds):
            rng = np.random.default_rng(rng_master.integers(2**32))
            x = syn.box_1d(rng, n, w)
            res = edge_width_box_1d(x, edge_loc=1.0, width_guess=w, n_bins=40)
            wf.append(res.w_fit); wr.append(res.w_from_rise)
            rows.append(dict(exp="recovery", target="box1d", w_true=w, seed=sd,
                             w_fit=res.w_fit, w_from_rise=res.w_from_rise,
                             agreement=res.agreement_ratio,
                             rel_disagree=res.rel_disagreement, n=n, nbins=40))
        wf, wr = np.array(wf), np.array(wr)
        out.append(dict(w_true=w, w_fit_mean=wf.mean(), w_fit_std=wf.std(),
                        rel_err=abs(wf.mean() - w) / w, w_rise_mean=wr.mean()))
    return out


def exp_resolution_bins(rng_master, n, seeds, rows, w=0.05):
    nbins_list = [8, 12, 16, 24, 32, 48, 64, 96, 128]
    window = 8.0 * w
    out = []
    for nb in nbins_list:
        binw_over_w = window / nb / w
        wf = []
        for sd in range(seeds):
            rng = np.random.default_rng(rng_master.integers(2**32))
            x = syn.box_1d(rng, n, w)
            res = edge_width_box_1d(x, edge_loc=1.0, width_guess=w, n_bins=nb)
            wf.append(res.w_fit)
            rows.append(dict(exp="resolution_bins", target="box1d", w_true=w,
                             seed=sd, nbins=nb, binw_over_w=binw_over_w,
                             w_fit=res.w_fit, w_from_rise=res.w_from_rise,
                             agreement=res.agreement_ratio,
                             rel_disagree=res.rel_disagreement, n=n))
        wf = np.array(wf)
        out.append(dict(nbins=nb, binw_over_w=binw_over_w, w_fit_mean=wf.mean(),
                        w_fit_std=wf.std(), rel_err=abs(wf.mean() - w) / w))
    return out


def exp_sample_size(rng_master, n_list, seeds, rows, w=0.05):
    out = []
    for nn in n_list:
        wf = []
        for sd in range(seeds):
            rng = np.random.default_rng(rng_master.integers(2**32))
            x = syn.box_1d(rng, int(nn), w)
            res = edge_width_box_1d(x, edge_loc=1.0, width_guess=w, n_bins=40)
            wf.append(res.w_fit)
            rows.append(dict(exp="sample_size", target="box1d", w_true=w, seed=sd,
                             n=int(nn), w_fit=res.w_fit, w_from_rise=res.w_from_rise,
                             agreement=res.agreement_ratio,
                             rel_disagree=res.rel_disagreement, nbins=40))
        wf = np.array(wf)
        out.append(dict(n=int(nn), w_fit_mean=wf.mean(), w_fit_std=wf.std(),
                        rel_err=abs(wf.mean() - w) / w))
    return out


def exp_disc_radius(rng_master, n, seeds, rows, w=0.03):
    radii = [0.5, 1.0, 2.0, 4.0]
    out = []
    for R in radii:
        wf = []
        for sd in range(seeds):
            rng = np.random.default_rng(rng_master.integers(2**32))
            pts = syn.disc(rng, n, w, radius=R)
            res = edge_width_disc(pts, center=np.zeros(2), radius=R,
                                  width_guess=w, n_bins=40)
            wf.append(res.w_fit)
            rows.append(dict(exp="disc_radius", target="disc2d", w_true=w, seed=sd,
                             radius=R, w_over_R=w / R, w_fit=res.w_fit,
                             w_from_rise=res.w_from_rise, agreement=res.agreement_ratio,
                             rel_disagree=res.rel_disagreement, n=n, nbins=40))
        wf = np.array(wf)
        out.append(dict(radius=R, w_over_R=w / R, w_fit_mean=wf.mean(),
                        w_fit_std=wf.std(), rel_err=abs(wf.mean() - w) / w))
    return out


def exp_interface_orientation(rng_master, n, seeds, rows, w=0.03):
    thetas = [0.0, 15.0, 30.0, 45.0, 60.0, 75.0]
    out = []
    for th in thetas:
        wf = []
        for sd in range(seeds):
            rng = np.random.default_rng(rng_master.integers(2**32))
            pts, nrm = syn.interface(rng, n, w, theta=np.deg2rad(th))
            res = edge_width_interface(pts, normal=nrm, offset=0.0,
                                       width_guess=w, n_bins=40)
            wf.append(res.w_fit)
            rows.append(dict(exp="interface_orientation", target="interface2d",
                             w_true=w, seed=sd, theta_deg=th, w_fit=res.w_fit,
                             w_from_rise=res.w_from_rise, agreement=res.agreement_ratio,
                             rel_disagree=res.rel_disagreement, n=n, nbins=40))
        wf = np.array(wf)
        out.append(dict(theta_deg=th, w_fit_mean=wf.mean(), w_fit_std=wf.std(),
                        rel_err=abs(wf.mean() - w) / w))
    return out


def exp_rescale_units(rng_master, n, seeds, rows, w=0.05):
    """Resolution-independence: scaling coordinates by lambda scales w by lambda."""
    lambdas = [0.1, 1.0, 10.0, 100.0]
    out = []
    for lam in lambdas:
        norm = []
        for sd in range(seeds):
            rng = np.random.default_rng(rng_master.integers(2**32))
            x = syn.box_1d(rng, n, w) * lam
            res = edge_width_box_1d(x, edge_loc=1.0 * lam, width_guess=w * lam,
                                    n_bins=40)
            norm.append(res.w_fit / lam)
            rows.append(dict(exp="rescale_units", target="box1d", w_true=w, seed=sd,
                             lam=lam, w_fit=res.w_fit, w_fit_over_lam=res.w_fit / lam,
                             w_from_rise=res.w_from_rise, agreement=res.agreement_ratio,
                             rel_disagree=res.rel_disagreement, n=n, nbins=40))
        norm = np.array(norm)
        out.append(dict(lam=lam, w_fit_over_lam_mean=norm.mean(),
                        w_fit_over_lam_std=norm.std(), rel_err=abs(norm.mean() - w) / w))
    return out


def exp_window_span(rng_master, n, seeds, rows, w=0.05):
    spans = [3.0, 4.0, 5.0, 6.0, 8.0]
    out = []
    for sp in spans:
        wf = []
        for sd in range(seeds):
            rng = np.random.default_rng(rng_master.integers(2**32))
            x = syn.box_1d(rng, n, w)
            nb = int(round(10 * sp))
            res = edge_width_box_1d(x, edge_loc=1.0, width_guess=w, n_bins=nb, span=sp)
            wf.append(res.w_fit)
            rows.append(dict(exp="window_span", target="box1d", w_true=w, seed=sd,
                             span=sp, nbins=nb, w_fit=res.w_fit,
                             w_from_rise=res.w_from_rise, agreement=res.agreement_ratio,
                             rel_disagree=res.rel_disagreement, n=n))
        wf = np.array(wf)
        out.append(dict(span=sp, w_fit_mean=wf.mean(), w_fit_std=wf.std(),
                        rel_err=abs(wf.mean() - w) / w))
    return out


# ── Figures (validation evidence) ────────────────────────────────────────────

def fig_example_profiles(outdir, n=1_500_000, seed=7):
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))

    w = 0.05
    x = syn.box_1d(rng, n, w)
    s, g, sig = profile_hist_1d(x, (1.0 - 4 * w, 1.0 + 4 * w), 40)
    res = estimate_edge_width(s, g, sig)
    _panel(axes[0], s, g, res, f"1-D box edge (w_true={w})", "x  [physical units]")

    w = 0.04; R = 1.0
    pts = syn.disc(rng, n, w, radius=R)
    s, g, sig = profile_radial_density(pts, np.zeros(2), (R - 4 * w, R + 4 * w), 40)
    res = estimate_edge_width(s, g, sig)
    _panel(axes[1], s, g, res, f"2-D disc boundary (w_true={w}, R={R})", "r  [physical units]")

    w = 0.03; th = 40.0
    pts, nrm = syn.interface(rng, n, w, theta=np.deg2rad(th))
    s, g, sig = profile_normal_density(pts, nrm, 0.0, (-4 * w, 4 * w), 40)
    res = estimate_edge_width(s, g, sig)
    _panel(axes[2], s, g, res, f"2-D interface @ {th}deg (w_true={w})", "s (normal)  [physical units]")

    fig.tight_layout()
    fig.savefig(outdir / "example_profiles.pdf")
    plt.close(fig)


def _panel(ax, s, g, res, title, xlabel):
    ax.plot(s, g, "o", ms=3, alpha=0.6, label="profile")
    ss = np.linspace(s[0], s[-1], 400)
    ax.plot(ss, gaussian_convolved_step(ss, res.level_low, res.level_high,
            res.center, res.w_fit), "-", lw=2, label=f"fit w={res.w_fit:.4f}")
    ax.axvline(res.center, color="k", ls=":", lw=1)
    ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel("density")
    ax.legend(fontsize=8)


def fig_summary(outdir, rec, binsw, ssz, disc, orient, resc, span):
    fig, ax = plt.subplots(2, 3, figsize=(14, 8))

    a = ax[0, 0]
    wt = [d["w_true"] for d in rec]; wf = [d["w_fit_mean"] for d in rec]
    a.plot(wt, wt, "k--", lw=1, label="ideal"); a.plot(wt, wf, "o-", label="fitted")
    a.set_xscale("log"); a.set_yscale("log")
    a.set_xlabel("w_true"); a.set_ylabel("w_fit"); a.set_title("Recovery vs w_true"); a.legend(fontsize=8)

    a = ax[0, 1]
    r = [d["binw_over_w"] for d in binsw]; m = [d["w_fit_mean"] for d in binsw]
    sdv = [d["w_fit_std"] for d in binsw]
    a.errorbar(r, m, yerr=sdv, fmt="o-"); a.axhline(0.05, color="k", ls="--", lw=1, label="w_true")
    a.axvline(1 / 3, color="r", ls=":", lw=1, label="binw=w/3")
    a.set_xscale("log"); a.set_xlabel("bin width / w"); a.set_ylabel("w_fit")
    a.set_title("Resolution (bin width)"); a.legend(fontsize=8)

    a = ax[0, 2]
    nn = [d["n"] for d in ssz]; m = [d["w_fit_mean"] for d in ssz]; sdv = [d["w_fit_std"] for d in ssz]
    a.errorbar(nn, m, yerr=sdv, fmt="o-"); a.axhline(0.05, color="k", ls="--", lw=1, label="w_true")
    a.set_xscale("log"); a.set_xlabel("N samples"); a.set_ylabel("w_fit")
    a.set_title("Sample-size convergence"); a.legend(fontsize=8)

    a = ax[1, 0]
    R = [d["radius"] for d in disc]; m = [d["w_fit_mean"] for d in disc]; sdv = [d["w_fit_std"] for d in disc]
    a.errorbar(R, m, yerr=sdv, fmt="o-"); a.axhline(0.03, color="k", ls="--", lw=1, label="w_true")
    a.set_xlabel("disc radius R"); a.set_ylabel("w_fit"); a.set_title("Disc radius invariance"); a.legend(fontsize=8)

    a = ax[1, 1]
    th = [d["theta_deg"] for d in orient]; m = [d["w_fit_mean"] for d in orient]; sdv = [d["w_fit_std"] for d in orient]
    a.errorbar(th, m, yerr=sdv, fmt="o-"); a.axhline(0.03, color="k", ls="--", lw=1, label="w_true")
    a.set_xlabel("interface angle [deg]"); a.set_ylabel("w_fit"); a.set_title("Orientation invariance"); a.legend(fontsize=8)

    a = ax[1, 2]
    lam = [d["lam"] for d in resc]; m = [d["w_fit_over_lam_mean"] for d in resc]; sdv = [d["w_fit_over_lam_std"] for d in resc]
    a.errorbar(lam, m, yerr=sdv, fmt="o-"); a.axhline(0.05, color="k", ls="--", lw=1, label="w_true (base units)")
    a.set_xscale("log"); a.set_xlabel("unit scale lambda"); a.set_ylabel("w_fit / lambda")
    a.set_title("Resolution independence (rescale)"); a.legend(fontsize=8)

    fig.tight_layout(); fig.savefig(outdir / "validation_summary.pdf"); plt.close(fig)


def fig_agreement(outdir, rows):
    wf = np.array([r["w_fit"] for r in rows if np.isfinite(r.get("w_fit", np.nan))])
    wr = np.array([r["w_from_rise"] for r in rows if np.isfinite(r.get("w_from_rise", np.nan))])
    m = min(len(wf), len(wr)); wf, wr = wf[:m], wr[:m]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    lim = [min(wf.min(), wr.min()) * 0.9, max(wf.max(), wr.max()) * 1.1]
    ax[0].plot(lim, lim, "k--", lw=1); ax[0].plot(wr, wf, ".", alpha=0.4)
    ax[0].set_xscale("log"); ax[0].set_yscale("log")
    ax[0].set_xlabel("w from 10-90 rise"); ax[0].set_ylabel("w fitted"); ax[0].set_title("Fit vs model-free rise")
    reld = np.abs(wf - wr) / wf
    ax[1].hist(reld, bins=40); ax[1].axvline(np.median(reld), color="r", label=f"median={np.median(reld):.3%}")
    ax[1].set_xlabel("|w_fit - w_rise| / w_fit"); ax[1].set_ylabel("count")
    ax[1].set_title("Fit/rise relative disagreement"); ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(outdir / "fit_vs_rise_agreement.pdf"); plt.close(fig)


# ── GATE evaluation ──────────────────────────────────────────────────────────

def cv(values):
    values = np.asarray(values, dtype=float)
    return float(np.std(values) / np.mean(values))


def _resolved(row, n_min=200_000, binw_max=1.0 / 3.0):
    if row.get("n", 0) < n_min:
        return False
    bw = row.get("binw_over_w", None)
    if bw is not None and bw > binw_max + 1e-9:
        return False
    return True


def evaluate_gate(rec, binsw, ssz, disc, orient, resc, span, rows):
    checks = []
    max_rel = max(d["rel_err"] for d in rec)
    checks.append(("recovery_bias_max<0.06", max_rel, max_rel < 0.06))

    good = [d["w_fit_mean"] for d in binsw if d["binw_over_w"] <= 1 / 3 + 1e-9]
    checks.append(("resolution_bins_CV<0.03", cv(good), cv(good) < 0.03))

    big = max(ssz, key=lambda d: d["n"]); small = min(ssz, key=lambda d: d["n"])
    checks.append(("largeN_bias<0.03", big["rel_err"], big["rel_err"] < 0.03))
    checks.append(("scatter_shrinks_with_N",
                   small["w_fit_std"] / max(big["w_fit_std"], 1e-12),
                   big["w_fit_std"] <= small["w_fit_std"]))

    disc_good = [d["w_fit_mean"] for d in disc if d["radius"] >= 1.0]
    checks.append(("disc_radius_CV<0.04", cv(disc_good), cv(disc_good) < 0.04))
    checks.append(("orientation_CV<0.02", cv([d["w_fit_mean"] for d in orient]),
                   cv([d["w_fit_mean"] for d in orient]) < 0.02))
    checks.append(("rescale_units_CV<0.01", cv([d["w_fit_over_lam_mean"] for d in resc]),
                   cv([d["w_fit_over_lam_mean"] for d in resc]) < 0.01))
    checks.append(("window_span_CV<0.03", cv([d["w_fit_mean"] for d in span]),
                   cv([d["w_fit_mean"] for d in span]) < 0.03))

    def _reld(subset):
        return np.array([r["rel_disagree"] for r in subset
                         if np.isfinite(r.get("rel_disagree", np.nan))])
    reld_all = _reld(rows); reld_reg = _reld([r for r in rows if _resolved(r)])
    med = float(np.median(reld_reg)); p95 = float(np.percentile(reld_reg, 95))
    checks.append(("fit_vs_rise_median<0.02", med, med < 0.02))
    checks.append(("fit_vs_rise_p95<0.04", p95, p95 < 0.04))

    extra = {
        "agreement_regime_median": med, "agreement_regime_p95": p95,
        "agreement_regime_n": int(reld_reg.size),
        "agreement_allconfigs_median": float(np.median(reld_all)),
        "agreement_allconfigs_p95": float(np.percentile(reld_all, 95)),
        "agreement_allconfigs_n": int(reld_all.size),
    }
    return all(c[2] for c in checks), checks, extra


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=1_500_000)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--outdir", type=Path, default=Path("results/m1/validation"))
    ap.add_argument("--master-seed", type=int, default=20260701)
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    rng_master = np.random.default_rng(args.master_seed)
    rows: list[dict] = []

    print(f"K_1090 (rise = K_1090 * w) = {K_1090:.6f}")
    print(f"N={args.n:,}  seeds={args.seeds}  outdir={args.outdir}\n")

    print("[1/7] recovery vs w_true ...");            rec = exp_recovery_vs_wtrue(rng_master, args.n, args.seeds, rows)
    print("[2/7] resolution (bin width) ...");        binsw = exp_resolution_bins(rng_master, args.n, args.seeds, rows)
    print("[3/7] sample-size convergence ...");       ssz = exp_sample_size(rng_master, [30_000, 100_000, 300_000, 1_000_000, 3_000_000], args.seeds, rows)
    print("[4/7] disc radius invariance ...");        disc = exp_disc_radius(rng_master, args.n, args.seeds, rows)
    print("[5/7] interface orientation invariance ..."); orient = exp_interface_orientation(rng_master, args.n, args.seeds, rows)
    print("[6/7] resolution independence (rescale) ..."); resc = exp_rescale_units(rng_master, args.n, args.seeds, rows)
    print("[7/7] window/span robustness ...");        span = exp_window_span(rng_master, args.n, args.seeds, rows)

    _write_csv(args.outdir / "results.csv", rows)
    print("\nRendering figures ...")
    fig_example_profiles(args.outdir)
    fig_summary(args.outdir, rec, binsw, ssz, disc, orient, resc, span)
    fig_agreement(args.outdir, rows)

    passed, checks, extra = evaluate_gate(rec, binsw, ssz, disc, orient, resc, span, rows)

    summary = {
        "n_per_measurement": args.n, "seeds": args.seeds, "K_1090": K_1090,
        "gate_passed": passed,
        "checks": [{"name": c[0], "value": float(c[1]), "pass": bool(c[2])} for c in checks],
        "agreement_detail": extra,
        "tables": {"recovery": rec, "resolution_bins": binsw, "sample_size": ssz,
                   "disc_radius": disc, "orientation": orient, "rescale_units": resc,
                   "window_span": span},
    }
    with open(args.outdir / "gate_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 68)
    print("M1 GATE — resolution-independent edge-width estimator")
    print("=" * 68)
    for name, val, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}]  {name:<34} = {val:.4g}")
    print("-" * 68)
    print(f"  fit/rise agreement — regime (n={extra['agreement_regime_n']}): "
          f"median={extra['agreement_regime_median']:.3%}, p95={extra['agreement_regime_p95']:.3%}")
    print(f"  fit/rise agreement — all configs (n={extra['agreement_allconfigs_n']}): "
          f"median={extra['agreement_allconfigs_median']:.3%}, p95={extra['agreement_allconfigs_p95']:.3%}")
    print("-" * 68)
    print(f"  OVERALL: {'PASS  ✓  gate open' if passed else 'FAIL  ✗  gate closed'}")
    print("=" * 68)
    print(f"\nArtifacts written to {args.outdir}/")
    return 0 if passed else 1


def _write_csv(path, rows):
    if not rows:
        return
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    import csv
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for r in rows:
            w.writerow(r)


if __name__ == "__main__":
    raise SystemExit(main())
