#!/usr/bin/env python
"""M1 — communicative visualizations of the edge-width observable.

Complements ``m1_validate.py`` (which produces the quantitative GATE evidence)
with a small gallery that *explains* the observable and shows why it is
resolution-independent.  All figures use synthetic targets with known width.

Figures (saved as both PNG and PDF under ``--outdir``):

  fig1_phenomenon              — the edge-width phenomenon: a 2-D disc at
                                 increasing blur, with its radial profile + erf fit.
  fig2_anatomy                 — anatomy of the estimator: erf fit, fitted ``w``
                                 band, 10-90 % rise; and the standardized "master
                                 curve" onto which all target types collapse.
  fig3_resolution_independence — same physical edge is measured identically across
                                 histogram resolutions and unit scales.
  fig4_recovery_agreement      — recovered ``w`` vs truth, and fitted ``w`` vs the
                                 model-free 10-90 rise.

Run::

    python scripts/m1_visualize.py                 # defaults
    python scripts/m1_visualize.py --n 2000000 --outdir results/m1/figures
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.special import ndtr  # standard-normal CDF, for the master curve

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from seele.edgewidth import (  # noqa: E402
    K_1090, estimate_edge_width, gaussian_convolved_step,
    profile_hist_1d, profile_radial_density, profile_normal_density,
)
from seele import synthetic as syn  # noqa: E402

plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 150, "font.size": 11,
    "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
})
CMAP = "viridis"
C_DATA, C_FIT, C_ACCENT = "#1f77b4", "#d62728", "#2ca02c"


def _save(fig, outdir, name):
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(outdir / f"{name}.{ext}")
    plt.close(fig)
    print(f"  wrote {name}.png / .pdf")


# ── fig1: the phenomenon ─────────────────────────────────────────────────────

def fig1_phenomenon(outdir, n, seed=1):
    rng = np.random.default_rng(seed)
    R = 1.0
    ws = [0.004, 0.02, 0.06]
    titles = ["sharp target (w→0)", f"blurred  w={ws[1]}", f"blurred  w={ws[2]}"]

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.2),
                             gridspec_kw={"height_ratios": [1.25, 1.0]})
    lim = 1.4
    # Common radial window + y-range for the bottom row so the three edges are
    # directly comparable; bins adapt per-w so each edge stays resolved.
    r_lo, r_hi = 0.6, 1.4
    for j, w in enumerate(ws):
        pts = syn.disc(rng, n, w, radius=R)
        # top: 2-D density heatmap
        H, xe, ye = np.histogram2d(pts[:, 0], pts[:, 1], bins=260,
                                   range=[[-lim, lim], [-lim, lim]])
        ax = axes[0, j]
        ax.imshow(H.T, origin="lower", extent=[-lim, lim, -lim, lim],
                  cmap=CMAP, aspect="equal")
        ax.set_title(titles[j]); ax.set_xticks([-1, 0, 1]); ax.set_yticks([-1, 0, 1])
        ax.grid(False)
        # draw the true boundary
        thc = np.linspace(0, 2 * np.pi, 200)
        ax.plot(R * np.cos(thc), R * np.sin(thc), color="w", lw=0.8, ls="--", alpha=0.7)

        # bottom: radial profile + erf fit, on the shared window
        nb = min(700, max(44, int(round((r_hi - r_lo) / (w / 6)))))
        s, g, sig = profile_radial_density(pts, np.zeros(2), (r_lo, r_hi), nb)
        res = estimate_edge_width(s, g, sig)
        ax = axes[1, j]
        ms = 2.0 if nb > 150 else 3.5
        ax.plot(s, g, "o", ms=ms, color=C_DATA, alpha=0.45, label="radial density")
        ss = np.linspace(r_lo, r_hi, 600)
        ax.plot(ss, gaussian_convolved_step(ss, res.level_low, res.level_high,
                res.center, res.w_fit), "-", lw=2.2, color=C_FIT,
                label=f"erf fit: w={res.w_fit:.4f}")
        ax.axvline(R, color="k", ls="--", lw=1, alpha=0.6)
        ax.set_xlim(r_lo, r_hi); ax.set_ylim(-0.03, 0.37)
        ax.set_xlabel("radius r  [physical units]")
        if j == 0:
            ax.set_ylabel("density")
        ax.legend(fontsize=8, loc="upper right")
        ax.set_title(f"recovered w = {res.w_fit:.4f}  (truth {w})", fontsize=10)

    fig.suptitle("The edge-width phenomenon — a sharp interface is rendered as a "
                 "smoothed transition of finite width w", fontsize=13)
    _save(fig, outdir, "fig1_phenomenon")


# ── fig2: anatomy + master curve ─────────────────────────────────────────────

def _measure(kind, rng, n, w, **kw):
    if kind == "box":
        x = syn.box_1d(rng, n, w, edge=kw.get("edge", 1.0))
        c = kw.get("edge", 1.0)
        s, g, sig = profile_hist_1d(x, (c - 4.5 * w, c + 4.5 * w), 44)
    elif kind == "disc":
        R = kw.get("radius", 1.0)
        pts = syn.disc(rng, n, w, radius=R)
        s, g, sig = profile_radial_density(pts, np.zeros(2), (R - 4.5 * w, R + 4.5 * w), 44)
    elif kind == "interface":
        pts, nrm = syn.interface(rng, n, w, theta=kw.get("theta", 0.0))
        s, g, sig = profile_normal_density(pts, nrm, 0.0, (-4.5 * w, 4.5 * w), 44)
    else:
        raise ValueError(kind)
    return s, g, sig, estimate_edge_width(s, g, sig)


def fig2_anatomy(outdir, n, seed=2):
    rng = np.random.default_rng(seed)
    fig, ax = plt.subplots(1, 2, figsize=(13.5, 5.4))

    # -- left: anatomy on a single interface edge --
    w = 0.03
    s, g, sig, res = _measure("interface", rng, n, w, theta=np.deg2rad(30))
    A, B, s0, wf = res.level_low, res.level_high, res.center, res.w_fit
    ss = np.linspace(s[0], s[-1], 500)
    a = ax[0]
    a.errorbar(s, g, yerr=sig, fmt="o", ms=3.5, color=C_DATA, alpha=0.75,
               elinewidth=0.7, label="generated profile")
    a.plot(ss, gaussian_convolved_step(ss, A, B, s0, wf), "-", lw=2.4, color=C_FIT,
           label="erf fit  g=A+(B−A)·Φ((s−s₀)/w)")
    # fitted-w band around s0
    a.axvspan(s0 - wf, s0 + wf, color=C_FIT, alpha=0.10)
    a.axvline(s0, color="k", ls=":", lw=1)
    a.annotate("s₀", (s0, A + 0.02 * (B - A)), fontsize=10, ha="center")
    # 10 / 90 levels and rise
    lo10 = A + 0.10 * (B - A); hi90 = A + 0.90 * (B - A)
    # crossings of the fitted curve (monotone) for a clean annotation
    from scipy.special import ndtri
    s10 = s0 + wf * ndtri(0.10); s90 = s0 + wf * ndtri(0.90)
    for lev in (lo10, hi90):
        a.axhline(lev, color="grey", ls="--", lw=0.8, alpha=0.7)
    ymid = A + 0.5 * (B - A)
    a.annotate("", xy=(s90, ymid), xytext=(s10, ymid),
               arrowprops=dict(arrowstyle="<->", color=C_ACCENT, lw=2))
    a.text(0.5 * (s10 + s90), ymid + 0.05 * (B - A), "10–90% rise",
           color=C_ACCENT, ha="center", fontsize=10, fontweight="bold")
    txt = (f"w_fit        = {wf:.4f}\n"
           f"10–90 rise = {res.rise_10_90:.4f}\n"
           f"w_from_rise = rise / K₁₀₉₀ = {res.w_from_rise:.4f}\n"
           f"K₁₀₉₀        = {K_1090:.4f}\n"
           f"agreement  = {res.agreement_ratio:.3f}")
    a.text(0.03, 0.97, txt, transform=a.transAxes, va="top", ha="left", fontsize=9,
           family="monospace", bbox=dict(boxstyle="round", fc="w", ec="0.7", alpha=0.9))
    a.set_xlabel("s along interface normal  [physical units]"); a.set_ylabel("density")
    a.set_title("Anatomy of the estimator"); a.legend(fontsize=8, loc="lower left")

    # -- right: standardized master curve (all target types collapse to Φ) --
    a = ax[1]
    z = np.linspace(-4, 4, 300)
    a.plot(z, ndtr(z), "k-", lw=2.5, label="Φ(z)  (universal)")
    cases = [("box edge", "box", 0.05, {}),
             ("disc R=1", "disc", 0.04, {}),
             ("interface 30°", "interface", 0.03, {"theta": np.deg2rad(30)})]
    marks = ["o", "s", "^"]
    for (lab, kind, w, kw), mk in zip(cases, marks):
        s, g, sig, res = _measure(kind, rng, n, w, **kw)
        A, B, s0, wf = res.level_low, res.level_high, res.center, res.w_fit
        zc = (s - s0) / wf
        gc = (g - A) / (B - A)
        sel = (zc > -4) & (zc < 4)
        a.plot(zc[sel], gc[sel], mk, ms=5, alpha=0.8, label=f"{lab} (w={wf:.3f})")
    a.set_xlabel("standardized normal  z = (s − s₀) / w")
    a.set_ylabel("normalized profile  (g − A)/(B − A)")
    a.set_title("All interfaces collapse onto one Gaussian-step curve")
    a.legend(fontsize=8, loc="upper left")

    _save(fig, outdir, "fig2_anatomy")


# ── fig3: resolution independence ────────────────────────────────────────────

def fig3_resolution_independence(outdir, n, seed=3):
    rng = np.random.default_rng(seed)
    fig, ax = plt.subplots(1, 2, figsize=(13.5, 5.2))

    # -- left: same physical edge at 3 histogram resolutions -> same w --
    w = 0.05
    x = syn.box_1d(rng, n, w, edge=1.0)
    a = ax[0]
    for nb, col in zip([10, 24, 80], [C_DATA, C_ACCENT, C_FIT]):
        s, g, sig = profile_hist_1d(x, (1 - 4 * w, 1 + 4 * w), nb)
        res = estimate_edge_width(s, g, sig)
        binw_over_w = (8 * w / nb) / w
        a.plot(s, g, "o", ms=3.5, color=col, alpha=0.55)
        ss = np.linspace(s[0], s[-1], 400)
        a.plot(ss, gaussian_convolved_step(ss, res.level_low, res.level_high,
               res.center, res.w_fit), "-", lw=2, color=col,
               label=f"{nb} bins (binw={binw_over_w:.2f}·w): w={res.w_fit:.4f}")
    a.axvline(1.0, color="k", ls="--", lw=1, alpha=0.6)
    a.set_xlabel("x  [physical units]"); a.set_ylabel("density")
    a.set_title("Invariant to histogram resolution")
    a.legend(fontsize=8, loc="upper right", title="w_true = 0.05")

    # -- right: same physical edge in 4 unit systems -> profiles collapse --
    a = ax[1]
    from scipy.special import ndtr as _ndtr
    z = np.linspace(-4, 4, 300)
    a.plot(z, _ndtr(z), "k-", lw=2.5, label="Φ(z)")
    wfits = []
    for lam, mk in zip([0.1, 1.0, 10.0, 100.0], ["o", "s", "^", "D"]):
        xx = syn.box_1d(np.random.default_rng(seed + int(lam)), n, w, edge=1.0) * lam
        s, g, sig = profile_hist_1d(xx, (lam * (1 - 4 * w), lam * (1 + 4 * w)), 44)
        res = estimate_edge_width(s, g, sig)
        wfits.append(res.w_fit / lam)
        A, B, s0, wf = res.level_low, res.level_high, res.center, res.w_fit
        zc = (s - s0) / wf; gc = (g - A) / (B - A)
        sel = (zc > -4) & (zc < 4)
        a.plot(zc[sel], gc[sel], mk, ms=5, alpha=0.75,
               label=f"λ={lam:g}: w/λ={res.w_fit/lam:.4f}")
    a.set_xlabel("standardized normal  z = (s − s₀) / w")
    a.set_ylabel("normalized profile")
    spread = float(np.std(wfits) / np.mean(wfits))
    a.set_title(f"Invariant to unit scale (×1000 range; CV of w/λ = {spread:.2%})")
    a.legend(fontsize=8, loc="upper left")

    _save(fig, outdir, "fig3_resolution_independence")


# ── fig4: recovery + agreement ───────────────────────────────────────────────

def fig4_recovery_agreement(outdir, n, seed=4, seeds=6):
    rng = np.random.default_rng(seed)
    widths = [0.01, 0.02, 0.04, 0.08, 0.16]

    fig, ax = plt.subplots(1, 2, figsize=(13.5, 5.4))

    # -- left: recovery vs truth, with per-seed scatter --
    a = ax[0]
    means, stds = [], []
    for w in widths:
        vals = []
        for k in range(seeds):
            x = syn.box_1d(np.random.default_rng(seed * 100 + k + int(1e4 * w)), n, w)
            r = estimate_edge_width(*profile_hist_1d(x, (1 - 4 * w, 1 + 4 * w), 40))
            vals.append(r.w_fit)
        means.append(np.mean(vals)); stds.append(np.std(vals))
    lim = [0.008, 0.2]
    a.plot(lim, lim, "k--", lw=1, label="ideal  w_fit = w_true")
    a.errorbar(widths, means, yerr=stds, fmt="o", ms=7, color=C_DATA,
               capsize=3, label="fitted (mean ± sd)")
    a.set_xscale("log"); a.set_yscale("log")
    a.set_xlabel("true width  w_true"); a.set_ylabel("fitted width  w_fit")
    a.set_title("Unbiased recovery across a decade of widths")
    a.legend(fontsize=9)
    # relative-error annotations
    for w, m in zip(widths, means):
        a.annotate(f"{(m-w)/w:+.1%}", (w, m), textcoords="offset points",
                   xytext=(6, -10), fontsize=7, color="0.35")

    # -- right: fit vs model-free rise, colored by target type --
    a = ax[1]
    rng2 = np.random.default_rng(seed + 77)
    colors = {"box": C_DATA, "disc": C_ACCENT, "interface": C_FIT}
    allw = []
    for kind, ws in [("box", [0.02, 0.04, 0.08]),
                     ("disc", [0.02, 0.04, 0.06]),
                     ("interface", [0.02, 0.03, 0.05])]:
        xs, ys = [], []
        for w in ws:
            for k in range(3):
                _, _, _, r = _measure(kind, np.random.default_rng(int(1e5 * w) + k + hash(kind) % 1000),
                                      n, w, theta=np.deg2rad(35))
                xs.append(r.w_from_rise); ys.append(r.w_fit); allw += [r.w_fit, r.w_from_rise]
        a.plot(xs, ys, "o", ms=6, alpha=0.8, color=colors[kind], label=kind)
    lim = [0.9 * min(allw), 1.1 * max(allw)]
    a.plot(lim, lim, "k--", lw=1, label="y = x")
    a.set_xscale("log"); a.set_yscale("log")
    a.set_xlabel("w from model-free 10–90 rise"); a.set_ylabel("w from erf fit")
    a.set_title("Fit agrees with the independent 10–90 rise")
    a.legend(fontsize=9)

    _save(fig, outdir, "fig4_recovery_agreement")


# ── fig5: the three synthetic target types ───────────────────────────────────

def fig5_targets(outdir, n, seed=5):
    """Gallery of the three synthetic targets, each with its extracted profile + fit.

    box1d      — 1-D top-hat; the right density edge is measured.
    disc2d     — uniform disc; the radial boundary is measured.
    interface2d — rotated half-plane; the profile is taken along the interface normal.
    """
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.4),
                             gridspec_kw={"height_ratios": [1.15, 1.0]})

    # ---------- box1d ----------
    w = 0.05; edge = 1.0
    x = syn.box_1d(rng, n, w, edge=edge)
    a = axes[0, 0]
    xs_lo, xs_hi = -0.2, 1.2
    hb = np.linspace(xs_lo, xs_hi, 220)
    dens, _ = np.histogram(x, bins=hb, density=True)
    ctr = 0.5 * (hb[:-1] + hb[1:])
    a.fill_between(ctr, dens, step="mid", color=C_DATA, alpha=0.35)
    a.plot(ctr, dens, drawstyle="steps-mid", color=C_DATA, lw=1.2)
    a.axvline(edge, color="k", ls="--", lw=1, alpha=0.6)
    a.annotate("measured\nedge", (edge, 0.55), xytext=(edge - 0.42, 0.7),
               fontsize=8, color="k",
               arrowprops=dict(arrowstyle="->", color="k", lw=1))
    a.set_title("box1d  —  1-D top-hat"); a.set_xlabel("x  [physical units]")
    a.set_ylabel("density"); a.set_xlim(xs_lo, xs_hi)

    s, g, sig, res = _measure("box", rng, n, w, edge=edge)
    _profile_panel(axes[1, 0], s, g, sig, res, "x near edge  [physical units]", w)

    # ---------- disc2d ----------
    w = 0.04; R = 1.0
    pts = syn.disc(rng, n, w, radius=R)
    a = axes[0, 1]
    lim = 1.4
    H, _, _ = np.histogram2d(pts[:, 0], pts[:, 1], bins=260,
                             range=[[-lim, lim], [-lim, lim]])
    a.imshow(H.T, origin="lower", extent=[-lim, lim, -lim, lim], cmap=CMAP, aspect="equal")
    a.grid(False); a.set_xticks([-1, 0, 1]); a.set_yticks([-1, 0, 1])
    thc = np.linspace(0, 2 * np.pi, 200)
    a.plot(R * np.cos(thc), R * np.sin(thc), color="w", lw=0.8, ls="--", alpha=0.7)
    a.annotate("", xy=(R * np.cos(0.6), R * np.sin(0.6)), xytext=(0, 0),
               arrowprops=dict(arrowstyle="->", color="w", lw=1.4))
    a.text(0.30, 0.42, "r", color="w", fontsize=11)
    a.set_title("disc2d  —  uniform disc"); a.set_xlabel("x  [physical units]")

    s, g, sig, res = _measure("disc", rng, n, w, radius=R)
    _profile_panel(axes[1, 1], s, g, sig, res, "radius r  [physical units]", w)

    # ---------- interface2d ----------
    w = 0.03; th = np.deg2rad(30.0)
    pts, nrm = syn.interface(rng, n, w, theta=th)
    a = axes[0, 2]
    lim = 1.3
    H, _, _ = np.histogram2d(pts[:, 0], pts[:, 1], bins=260,
                             range=[[-lim, lim], [-lim, lim]])
    a.imshow(H.T, origin="lower", extent=[-lim, lim, -lim, lim], cmap=CMAP, aspect="equal")
    a.grid(False); a.set_xticks([-1, 0, 1]); a.set_yticks([-1, 0, 1])
    that = np.array([-np.sin(th), np.cos(th)])          # along the interface
    p0 = -1.2 * that; p1 = 1.2 * that
    a.plot([p0[0], p1[0]], [p0[1], p1[1]], color="w", ls="--", lw=0.9, alpha=0.8)
    a.annotate("", xy=tuple(0.45 * nrm), xytext=(0, 0),
               arrowprops=dict(arrowstyle="->", color="w", lw=1.6))
    a.text(0.45 * nrm[0] + 0.05, 0.45 * nrm[1], "n̂", color="w", fontsize=11)
    a.set_title("interface2d  —  half-plane @ 30°"); a.set_xlabel("x  [physical units]")

    s, g, sig, res = _measure("interface", rng, n, w, theta=th)
    _profile_panel(axes[1, 2], s, g, sig, res, "s along normal  [physical units]", w)

    fig.suptitle("The three synthetic target types — same estimator, one profile "
                 "along the interface normal", fontsize=13)
    _save(fig, outdir, "fig5_targets")


def _profile_panel(ax, s, g, sig, res, xlabel, w_true):
    ax.errorbar(s, g, yerr=sig, fmt="o", ms=3.2, color=C_DATA, alpha=0.7,
                elinewidth=0.7, label="profile")
    ss = np.linspace(s[0], s[-1], 400)
    ax.plot(ss, gaussian_convolved_step(ss, res.level_low, res.level_high,
            res.center, res.w_fit), "-", lw=2.2, color=C_FIT,
            label=f"erf fit: w={res.w_fit:.4f}")
    ax.axvline(res.center, color="k", ls=":", lw=1, alpha=0.7)
    ax.set_xlabel(xlabel); ax.set_ylabel("density")
    ax.set_title(f"w_fit={res.w_fit:.4f}  ·  w_rise={res.w_from_rise:.4f}  "
                 f"(truth {w_true})", fontsize=9)
    ax.legend(fontsize=8, loc="best")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=1_500_000)
    ap.add_argument("--outdir", type=Path, default=Path("results/m1/figures"))
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    print(f"Rendering M1 visualizations (N={args.n:,}) -> {args.outdir}/")
    fig1_phenomenon(args.outdir, args.n)
    fig2_anatomy(args.outdir, args.n)
    fig3_resolution_independence(args.outdir, args.n)
    fig4_recovery_agreement(args.outdir, args.n)
    fig5_targets(args.outdir, args.n)
    print("done.")


if __name__ == "__main__":
    main()
