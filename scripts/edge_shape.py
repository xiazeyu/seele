#!/usr/bin/env python3
"""Edge shape of flow-matching generated distributions: smoothing vs overshoot.

Question 1 — train simple CFM models on 2-D targets whose density has a sharp
cut-off (true edge width ``w_true = 0``) and steep-but-continuous (Gaussian-
blurred, ``w_true > 0``) cut-offs, then sample and inspect the generated edge.

Question 2 — is it *common* for the generated distribution to simultaneously
(a) smooth the edge (measured width > true width) and (b) overshoot — both in
the density sense (a bump above the fitted step near the edge, the
``overshoot`` metric of :mod:`seele.edgewidth`) and in the support sense
(sample mass spilling beyond the true support boundary)?

Conditions: {disc, iface30} x w_true {0, 0.02, 0.05} x seeds x samplers
(euler nfe 8/32/128, rk4 nfe 128).  Everything is resumable: checkpoints and
sample ``.npy`` files are reused, metric rows already in the CSV are skipped.

Outputs (under ``results/edge_shape[_smoke]/``):
    ckpt/{key}.pt                       trained velocity fields
    data/{geom}_w{w}_train.npy          training data (seed 0 draw)
    data/{geom}_w{w}_ref.npy            large true reference draw
    samples/{key}_{solver}{nfe}.npy     generated samples  [n, 2] float32
    metrics.csv                         one row per (run, sampler)
    fig_{geom}_hist2d.png               2-D histograms, true vs generated
    fig_{geom}_profiles.png             edge density profiles vs truth
    fig_summary.png                     width / overshoot / spill summary

Usage:
    python scripts/edge_shape.py --smoke     # minutes-scale sanity pass
    python scripts/edge_shape.py             # full study
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from seele import synthetic as syn
from seele.edgewidth import profile_normal_density, profile_radial_density
from seele.fm import FMConfig, load_checkpoint, sample_fm, save_checkpoint, train_fm
from seele.targets import EdgeSpec, measure_edge_adaptive

# ── Experiment grid ─────────────────────────────────────────────────────────

W_TRUE = (0.0, 0.02, 0.05)          # sharp, steep-but-continuous, slightly blurred
SAMPLERS = (("euler", 8), ("euler", 32), ("euler", 128), ("rk4", 128))

DISC_R = 1.0
IFACE_THETA = np.deg2rad(30.0)
IFACE_NORMAL = np.array([np.cos(IFACE_THETA), np.sin(IFACE_THETA)])
IFACE_THW = 0.8                     # transverse clip: stay away from strip ends


@dataclass(frozen=True)
class Geometry:
    name: str
    spec: EdgeSpec                   # measurement interface (the cut-off edge)
    domain: tuple[float, float, float, float]

    def sample(self, rng: np.random.Generator, n: int, w: float) -> np.ndarray:
        if self.name == "disc":
            return syn.disc(rng, n, w, radius=DISC_R)
        pts, _ = syn.interface(rng, n, w, theta=IFACE_THETA)
        return pts

    def signed_excess(self, pts: np.ndarray) -> np.ndarray:
        """Signed distance beyond the cut-off (> 0 = outside the support edge).

        For the interface only points within the transverse clip are returned,
        so the strip's *other* boundaries never contaminate the spill metric.
        """
        pts = np.asarray(pts, float)
        if self.name == "disc":
            return np.linalg.norm(pts, axis=1) - DISC_R
        u = pts @ IFACE_NORMAL
        v = pts @ np.array([-IFACE_NORMAL[1], IFACE_NORMAL[0]])
        return u[np.abs(v) <= IFACE_THW]

    def profile(self, pts: np.ndarray, half: float, n_bins: int):
        """Density profile across the cut-off, window ``edge +/- half``."""
        if self.name == "disc":
            return profile_radial_density(pts, np.zeros(2),
                                          (DISC_R - half, DISC_R + half), n_bins)
        return profile_normal_density(pts, IFACE_NORMAL, 0.0, (-half, half),
                                      n_bins, transverse_halfwidth=IFACE_THW)


GEOMETRIES = (
    Geometry("disc",
             EdgeSpec("boundary", "radial", {"center": (0.0, 0.0), "radius": DISC_R}),
             (-1.4, 1.4, -1.4, 1.4)),
    Geometry("iface30",
             EdgeSpec("interface", "normal",
                      {"normal": tuple(IFACE_NORMAL), "offset": 0.0,
                       "transverse_halfwidth": IFACE_THW}),
             (-1.6, 1.6, -1.6, 1.6)),
)


# ── Helpers ─────────────────────────────────────────────────────────────────

def wtag(w: float) -> str:
    return f"w{w:g}".replace(".", "p")


def run_key(geom: str, w: float, seed: int) -> str:
    return f"{geom}_{wtag(w)}_seed{seed}"


def spill_stats(excess: np.ndarray) -> dict:
    """Fraction and depth of mass beyond the support edge."""
    frac = float(np.mean(excess > 0.0))
    out = excess[excess > 0.0]
    return {
        "spill_frac": frac,
        "spill_mean_exceed": float(out.mean()) if out.size else 0.0,
        "spill_q99_exceed": float(np.quantile(out, 0.99)) if out.size else 0.0,
    }


# ── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--smoke", action="store_true",
                    help="tiny net / few steps / few samples, minutes-scale")
    ap.add_argument("--seeds", type=int, default=None,
                    help="number of seeds (default: 3 full, 1 smoke)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    smoke = args.smoke
    n_seeds = args.seeds if args.seeds is not None else (1 if smoke else 3)
    n_train = 50_000 if smoke else 500_000
    n_gen = 20_000 if smoke else 1_000_000
    n_ref = 100_000 if smoke else 2_000_000
    steps = 400 if smoke else 8000
    cfg_kw = dict(hidden_dim=64, n_blocks=2) if smoke else {}

    root = Path(args.outdir) if args.outdir else (
        Path(__file__).resolve().parent.parent / "results"
        / ("edge_shape_smoke" if smoke else "edge_shape"))
    for sub in ("ckpt", "data", "samples"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    csv_path = root / "metrics.csv"

    fields = ["geom", "w_true", "seed", "solver", "nfe",
              "w_fit", "w_fit_err", "w_gen_quad", "rise_10_90",
              "overshoot", "overshoot_z", "overshoot_loc_w",
              "spill_frac", "spill_mean_exceed", "spill_q99_exceed",
              "spill_frac_true", "excess_spill",
              "fit_success", "train_wallclock_s"]
    done_rows: set[tuple] = set()
    if csv_path.exists():
        with open(csv_path) as f:
            done_rows = {(r["geom"], r["w_true"], r["seed"], r["solver"], r["nfe"])
                         for r in csv.DictReader(f)}
    else:
        with open(csv_path, "w", newline="") as f:
            csv.DictWriter(f, fields).writeheader()

    t_start = time.perf_counter()
    for geom in GEOMETRIES:
        for w_true in W_TRUE:
            # True reference draw: profile overlay + spill baseline.
            ref_path = root / "data" / f"{geom.name}_{wtag(w_true)}_ref.npy"
            if ref_path.exists():
                ref = np.load(ref_path)
            else:
                ref = geom.sample(np.random.default_rng(10_000), n_ref,
                                  w_true).astype(np.float32)
                np.save(ref_path, ref)
            ref_spill = spill_stats(geom.signed_excess(ref))["spill_frac"]

            for seed in range(n_seeds):
                key = run_key(geom.name, w_true, seed)
                ckpt_path = root / "ckpt" / f"{key}.pt"
                data_path = root / "data" / f"{geom.name}_{wtag(w_true)}_train.npy"

                if seed == 0 and not data_path.exists():
                    np.save(data_path,
                            geom.sample(np.random.default_rng(0), n_train,
                                        w_true).astype(np.float32))

                # Train (or reload) --------------------------------------------------
                cfg = FMConfig(**cfg_kw)
                if ckpt_path.exists():
                    net, payload = load_checkpoint(ckpt_path, device=args.device)
                    wallclock = payload["meta"].get("wallclock_s", float("nan"))
                    print(f"[{key}] checkpoint reused", flush=True)
                else:
                    data = geom.sample(np.random.default_rng(seed), n_train, w_true)
                    t0 = time.perf_counter()
                    net, res = train_fm(cfg, data, steps=steps, seed=seed,
                                        device=args.device, log_every=0)
                    wallclock = time.perf_counter() - t0
                    save_checkpoint(ckpt_path, res.checkpoints[steps], cfg, steps,
                                    meta={"geom": geom.name, "w_true": w_true,
                                          "seed": seed, "n_train": n_train,
                                          "wallclock_s": wallclock})
                    print(f"[{key}] trained {steps} steps in {wallclock:.0f}s",
                          flush=True)

                # Sample + measure ---------------------------------------------------
                for solver, nfe in SAMPLERS:
                    row_id = (geom.name, f"{w_true:g}", str(seed), solver, str(nfe))
                    smp_path = root / "samples" / f"{key}_{solver}{nfe}.npy"
                    if smp_path.exists():
                        gen = np.load(smp_path)
                    else:
                        gen = sample_fm(net, n_gen, nfe=nfe, solver=solver,
                                        device=args.device,
                                        seed=1000 + seed).astype(np.float32)
                        np.save(smp_path, gen)
                    if row_id in done_rows:
                        continue

                    res_w, _ = measure_edge_adaptive(
                        gen.astype(float), geom.spec,
                        width_guess=max(0.03, 2.0 * w_true))
                    sp = spill_stats(geom.signed_excess(gen))
                    w_fit = res_w.w_fit
                    w_gen_quad = (float(np.sqrt(max(w_fit ** 2 - w_true ** 2, 0.0)))
                                  if np.isfinite(w_fit) else float("nan"))
                    row = {
                        "geom": geom.name, "w_true": f"{w_true:g}",
                        "seed": seed, "solver": solver, "nfe": nfe,
                        "w_fit": w_fit, "w_fit_err": res_w.w_fit_err,
                        "w_gen_quad": w_gen_quad, "rise_10_90": res_w.rise_10_90,
                        "overshoot": res_w.overshoot,
                        "overshoot_z": res_w.overshoot_z,
                        "overshoot_loc_w": res_w.overshoot_loc_w,
                        **sp,
                        "spill_frac_true": ref_spill,
                        "excess_spill": sp["spill_frac"] - ref_spill,
                        "fit_success": res_w.fit_success,
                        "train_wallclock_s": f"{wallclock:.1f}",
                    }
                    with open(csv_path, "a", newline="") as f:
                        csv.DictWriter(f, fields).writerow(row)
                    done_rows.add(row_id)
                    print(f"  {solver}{nfe}: w_fit={w_fit:.4f} "
                          f"overshoot={res_w.overshoot:+.3f} "
                          f"(z={res_w.overshoot_z:.1f}) "
                          f"spill={sp['spill_frac']:.4f} (true {ref_spill:.4f})",
                          flush=True)

    make_figures(root, n_seeds)
    print(f"done in {time.perf_counter() - t_start:.0f}s -> {root}", flush=True)


# ── Figures ─────────────────────────────────────────────────────────────────

def make_figures(root: Path, n_seeds: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    with open(root / "metrics.csv") as f:
        rows = list(csv.DictReader(f))

    for geom in GEOMETRIES:
        # 2-D histograms: rows = w_true, cols = true + samplers (seed 0) -----
        cols = ["true"] + [f"{s}{n}" for s, n in SAMPLERS]
        fig, axes = plt.subplots(len(W_TRUE), len(cols),
                                 figsize=(3.0 * len(cols), 3.0 * len(W_TRUE)),
                                 sharex=True, sharey=True, layout="constrained")
        xmin, xmax, ymin, ymax = geom.domain
        bins = [np.linspace(xmin, xmax, 200), np.linspace(ymin, ymax, 200)]
        for i, w in enumerate(W_TRUE):
            for j, col in enumerate(cols):
                ax = axes[i, j]
                if col == "true":
                    pts = np.load(root / "data" / f"{geom.name}_{wtag(w)}_ref.npy")
                else:
                    pts = np.load(root / "samples"
                                  / f"{run_key(geom.name, w, 0)}_{col}.npy")
                h, xe, ye = np.histogram2d(pts[:, 0], pts[:, 1], bins=bins,
                                           density=True)
                ax.pcolormesh(xe, ye, h.T, norm=LogNorm(vmin=1e-3),
                              cmap="viridis", rasterized=True)
                ax.set_aspect("equal")
                if i == 0:
                    ax.set_title(col)
                if j == 0:
                    ax.set_ylabel(f"w_true = {w:g}")
        fig.suptitle(f"{geom.name}: true vs FM-generated (seed 0, log density)")
        fig.savefig(root / f"fig_{geom.name}_hist2d.png", dpi=150)
        plt.close(fig)

        # Edge profiles ------------------------------------------------------
        fig, axes = plt.subplots(len(W_TRUE), len(SAMPLERS),
                                 figsize=(3.4 * len(SAMPLERS), 2.8 * len(W_TRUE)),
                                 sharex="row", layout="constrained")
        for i, w in enumerate(W_TRUE):
            half = max(0.15, 6.0 * w)
            ref = np.load(root / "data" / f"{geom.name}_{wtag(w)}_ref.npy")
            s_t, g_t, _ = geom.profile(ref, half, 80)
            for j, (solver, nfe) in enumerate(SAMPLERS):
                ax = axes[i, j]
                ax.plot(s_t, g_t, "k-", lw=1.5, label="true")
                for seed in range(n_seeds):
                    gen = np.load(root / "samples"
                                  / f"{run_key(geom.name, w, seed)}_{solver}{nfe}.npy")
                    s_g, g_g, sig = geom.profile(gen, half, 80)
                    ax.errorbar(s_g, g_g, yerr=sig, fmt=".", ms=3, lw=0.8,
                                alpha=0.7, label=f"gen s{seed}" if i == j == 0
                                else None)
                edge = DISC_R if geom.name == "disc" else 0.0
                ax.axvline(edge, color="r", ls=":", lw=0.8)
                if i == 0:
                    ax.set_title(f"{solver} nfe={nfe}")
                if j == 0:
                    ax.set_ylabel(f"w_true={w:g}\ndensity")
                if i == len(W_TRUE) - 1:
                    ax.set_xlabel("r" if geom.name == "disc" else "s (normal)")
        axes[0, 0].legend(fontsize=7)
        fig.suptitle(f"{geom.name}: edge density profile, true vs generated")
        fig.savefig(root / f"fig_{geom.name}_profiles.png", dpi=150)
        plt.close(fig)

    # Summary: width / overshoot / spill vs sampler, colored by w_true -------
    fig, axes = plt.subplots(len(GEOMETRIES), 3,
                             figsize=(12, 3.2 * len(GEOMETRIES)),
                             layout="constrained")
    xpos = {f"{s}{n}": k for k, (s, n) in enumerate(SAMPLERS)}
    colors = {f"{w:g}": c for w, c in zip(W_TRUE, ("C0", "C1", "C2"))}
    for gi, geom in enumerate(GEOMETRIES):
        sub = [r for r in rows if r["geom"] == geom.name]
        for r in sub:
            x = xpos[f"{r['solver']}{r['nfe']}"] \
                + 0.12 * (int(r["seed"]) - (n_seeds - 1) / 2)
            c = colors[r["w_true"]]
            axes[gi, 0].plot(x, float(r["w_fit"]), "o", color=c, ms=5)
            axes[gi, 1].plot(x, float(r["overshoot"]), "o", color=c, ms=5)
            axes[gi, 2].plot(x, float(r["excess_spill"]), "o", color=c, ms=5)
        for w in W_TRUE:
            axes[gi, 0].axhline(w, color=colors[f"{w:g}"], ls="--", lw=0.8,
                                label=f"w_true={w:g}")
        axes[gi, 1].axhline(0.0, color="k", lw=0.8)
        axes[gi, 2].axhline(0.0, color="k", lw=0.8)
        titles = ("measured edge width w_fit", "density overshoot (frac of step)",
                  "excess spill beyond support")
        for ax, ttl in zip(axes[gi], titles):
            ax.set_xticks(range(len(SAMPLERS)),
                          [f"{s}\n{n}" for s, n in SAMPLERS])
            ax.set_title(f"{geom.name}: {ttl}", fontsize=10)
        axes[gi, 0].legend(fontsize=7)
        axes[gi, 0].set_yscale("log")
    fig.savefig(root / "fig_summary.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
