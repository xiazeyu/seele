#!/usr/bin/env python3
"""Are the edge defects architecture-specific?  Train alternative velocity-field
architectures on the same sharp targets and measure the *extended* defect set.

Architectures (roughly matched scale, same training protocol as edge_shape):
    resmlp — baseline ResNet-MLP from seele.fm (reuses edge_shape checkpoints)
    plain  — plain MLP, no residual connections
    dit    — mini DiT: per-coordinate tokens + adaLN-zero time conditioning
    dit_s  — same DiT at d=88 (~440k params), param-matched to resmlp

Extended metrics (prototypes for the future metrics.csv columns):
    center bias      — erf-fit center + model-free half-max crossing
    plateau tilt     — relative interior density slope, edge-excluded band
    tail outlier     — jacobian-corrected exceedance ratio at z > 3 vs erf tail
    anisotropy (disc)— CV of per-sector edge width over 8 angular sectors

Sampling: rk4 nfe=128 only (converged regime).  Resumable like edge_shape.

Usage: python scripts/arch_defects.py [--smoke] [--seeds N]
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from seele.fm import (FMConfig, FourierEmbedding, load_checkpoint, pick_device,
                      sample_fm, train_fm)
from seele.targets import measure_edge_adaptive
from edge_shape import GEOMETRIES, run_key, spill_stats

GEOM = {g.name: g for g in GEOMETRIES}
DISC_R = 1.0
IFACE_N = np.array([np.cos(np.deg2rad(30)), np.sin(np.deg2rad(30))])
IFACE_T = np.array([-IFACE_N[1], IFACE_N[0]])


# ── Alternative velocity fields ─────────────────────────────────────────────

class PlainMLP(nn.Module):
    """Plain SiLU MLP on concat([x, time_embed(t)]) — no residuals."""

    def __init__(self, data_dim=2, hidden=256, depth=3, temb=64):
        super().__init__()
        self.data_dim = data_dim
        self.time_embed = FourierEmbedding(temb)
        dims = [data_dim + temb] + [hidden] * depth
        layers = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(a, b), nn.SiLU()]
        self.net = nn.Sequential(*layers, nn.Linear(hidden, data_dim))

    def forward(self, x, t):
        return self.net(torch.cat([x, self.time_embed(t)], dim=-1))


class DiTBlock(nn.Module):
    def __init__(self, d, heads=4):
        super().__init__()
        self.ln1 = nn.LayerNorm(d, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d, elementwise_affine=False)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.SiLU(),
                                 nn.Linear(4 * d, d))
        self.ada = nn.Linear(d, 6 * d)
        nn.init.zeros_(self.ada.weight)
        nn.init.zeros_(self.ada.bias)

    def forward(self, h, cond):
        s1, b1, g1, s2, b2, g2 = self.ada(cond).unsqueeze(1).chunk(6, dim=-1)
        u = self.ln1(h) * (1 + s1) + b1
        h = h + g1 * self.attn(u, u, u, need_weights=False)[0]
        u = self.ln2(h) * (1 + s2) + b2
        return h + g2 * self.mlp(u)


class DiTVelocity(nn.Module):
    """Mini DiT: one token per coordinate, adaLN-zero time conditioning."""

    def __init__(self, data_dim=2, d=128, n_blocks=3, temb_dim=64):
        super().__init__()
        self.data_dim = data_dim
        self.time_embed = FourierEmbedding(temb_dim)
        self.cond_proj = nn.Sequential(nn.Linear(temb_dim, d), nn.SiLU(),
                                       nn.Linear(d, d))
        self.tok = nn.Linear(1, d)
        self.pos = nn.Parameter(torch.randn(1, data_dim, d) * 0.02)
        self.blocks = nn.ModuleList(DiTBlock(d) for _ in range(n_blocks))
        self.ln_f = nn.LayerNorm(d, elementwise_affine=False)
        self.head = nn.Linear(d, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x, t):
        h = self.tok(x.unsqueeze(-1)) + self.pos
        cond = self.cond_proj(self.time_embed(t))
        for blk in self.blocks:
            h = blk(h, cond)
        return self.head(self.ln_f(h)).squeeze(-1)


def build_arch(name: str) -> nn.Module | None:
    if name == "resmlp":
        return None                       # reuse edge_shape checkpoint
    if name == "plain":
        return PlainMLP()
    if name == "dit":
        return DiTVelocity()
    if name == "dit_s":
        return DiTVelocity(d=88)          # ~430k params, matched to resmlp
    raise ValueError(name)


ARCHS = ("resmlp", "plain", "dit", "dit_s")


# ── Extended metrics ────────────────────────────────────────────────────────

def halfmax_center(s: np.ndarray, g: np.ndarray, lo: float, hi: float) -> float:
    """Model-free edge location: crossing of the (lo+hi)/2 level, linear interp."""
    mid = 0.5 * (lo + hi)
    above = g > mid
    for i in range(len(g) - 1):
        if above[i] != above[i + 1] and g[i + 1] != g[i]:
            return float(s[i] + (mid - g[i]) * (s[i + 1] - s[i])
                         / (g[i + 1] - g[i]))
    return float("nan")


def extended_metrics(pts: np.ndarray, gname: str) -> dict:
    geom = GEOM[gname]
    res, wg = measure_edge_adaptive(pts, geom.spec, width_guess=0.03)
    w, c = res.w_fit, res.center
    out = {"w_fit": w, "w_fit_err": res.w_fit_err,
           "overshoot": res.overshoot, "overshoot_z": res.overshoot_z,
           "center": c, "center_err": float("nan"),
           **spill_stats(geom.signed_excess(pts))}

    # model-free half-max center on a fine profile
    half = max(0.12, 6.0 * w)
    s_p, g_p, _ = geom.profile(pts, half, 80)
    out["center_halfmax"] = halfmax_center(s_p, g_p, res.level_high,
                                           res.level_low)

    if gname == "disc":
        r = np.linalg.norm(pts, axis=1)
        s_coord, jac_w = r, np.clip(c / np.maximum(r, 1e-9), None, 10.0)
        hi_b = c - 5.0 * w if np.isfinite(c - 5.0 * w) else 0.8
        band = (0.2, float(np.clip(hi_b, 0.35, 0.8)))
        edges_ = np.linspace(*band, 25)
        cnt, _ = np.histogram(r, bins=edges_)
        rc = 0.5 * (edges_[:-1] + edges_[1:])
        dens = cnt / (2 * np.pi * rc)
        slope, _ = np.polyfit(rc, dens / dens.mean(), 1)
        out["plateau_tilt"] = float(slope * (band[1] - band[0]))

        # anisotropy: per-sector width, 8 sectors
        phi = np.arctan2(pts[:, 1], pts[:, 0])
        ws = []
        for k in range(8):
            lo_a = -np.pi + k * np.pi / 4
            sel = (phi >= lo_a) & (phi < lo_a + np.pi / 4)
            r_k, _res = measure_edge_adaptive(pts[sel], geom.spec,
                                              width_guess=w)
            if r_k.fit_success and np.isfinite(r_k.w_fit):
                ws.append(r_k.w_fit)
        ws = np.array(ws)
        out["aniso_w_cv"] = (float(ws.std() / ws.mean())
                             if len(ws) >= 6 else float("nan"))
    else:
        u = pts @ IFACE_N
        v = pts @ IFACE_T
        keep = np.abs(v) <= 0.8
        s_coord, jac_w = u[keep], np.ones(keep.sum())
        hi_b = c - 5.0 * w if np.isfinite(c - 5.0 * w) else -0.2
        band = (-0.8, float(np.clip(hi_b, -0.65, -0.2)))
        edges_ = np.linspace(*band, 25)
        cnt, _ = np.histogram(s_coord, bins=edges_)
        uc = 0.5 * (edges_[:-1] + edges_[1:])
        slope, _ = np.polyfit(uc, cnt / cnt.mean(), 1)
        out["plateau_tilt"] = float(slope * (band[1] - band[0]))
        out["aniso_w_cv"] = float("nan")

    # tail outlier ratio, jacobian-corrected, vs half-normal expectation
    z = (s_coord - c) / w
    sel = z > 0
    zw = jac_w[sel] if gname == "disc" else jac_w[: sel.sum()]
    z = z[sel]
    p3 = float(np.sum(zw * (z > 3.0)) / np.sum(zw))
    out["tail_ratio_z3"] = p3 / 0.0026998                 # 2*Phi(-3)
    out["z_q999"] = (float(np.quantile(z, 0.999)) if z.size > 2000
                     else float("nan"))
    return out


# ── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seeds", type=int, default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    smoke = args.smoke
    n_seeds = args.seeds if args.seeds is not None else (1 if smoke else 3)
    n_train = 50_000 if smoke else 500_000
    n_gen = 20_000 if smoke else 1_000_000
    steps = 400 if smoke else 8000

    repo = Path(__file__).resolve().parent.parent
    root = repo / "results" / ("arch_defects_smoke" if smoke else "arch_defects")
    es_root = repo / "results" / ("edge_shape_smoke" if smoke else "edge_shape")
    for sub in ("ckpt", "samples"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    csv_path = root / "metrics.csv"

    fields = ["arch", "geom", "seed", "n_params",
              "w_fit", "w_fit_err", "overshoot", "overshoot_z",
              "center", "center_halfmax", "plateau_tilt",
              "spill_frac", "spill_mean_exceed", "spill_q99_exceed",
              "tail_ratio_z3", "z_q999", "aniso_w_cv",
              "final_loss_ema", "train_wallclock_s"]
    done: set[tuple] = set()
    if csv_path.exists():
        with open(csv_path) as f:
            done = {(r["arch"], r["geom"], r["seed"])
                    for r in csv.DictReader(f)}
    else:
        with open(csv_path, "w", newline="") as f:
            csv.DictWriter(f, fields).writeheader()

    cfg = FMConfig()
    for arch in ARCHS:
        for gname in ("disc", "iface30"):
            geom = GEOM[gname]
            for seed in range(n_seeds):
                if (arch, gname, str(seed)) in done:
                    continue
                key = f"{arch}_{gname}_seed{seed}"
                smp_path = root / "samples" / f"{key}_rk4128.npy"
                n_params, wallclock, loss_ema = -1, float("nan"), float("nan")

                if not smp_path.exists():
                    if arch == "resmlp":
                        # reuse the edge_shape run (w_true = 0)
                        src = (es_root / "samples"
                               / f"{run_key(gname, 0.0, seed)}_rk4128.npy")
                        ck = es_root / "ckpt" / f"{run_key(gname, 0.0, seed)}.pt"
                        if src.exists():
                            gen = np.load(src)
                            np.save(smp_path, gen)
                            net, _ = load_checkpoint(ck)
                            n_params = sum(p.numel() for p in net.parameters())
                        else:
                            raise FileNotFoundError(src)
                    else:
                        torch.manual_seed(seed)
                        net = build_arch(arch).to(pick_device(args.device))
                        n_params = sum(p.numel() for p in net.parameters())
                        data = geom.sample(np.random.default_rng(seed),
                                           n_train, 0.0)
                        t0 = time.perf_counter()
                        net, tres = train_fm(cfg, data, steps=steps, seed=seed,
                                             device=args.device, net=net,
                                             log_every=steps)
                        wallclock = time.perf_counter() - t0
                        loss_ema = (tres.loss_history[-1][1]
                                    if tres.loss_history else float("nan"))
                        torch.save({"arch": arch, "model": net.state_dict()},
                                   root / "ckpt" / f"{key}.pt")
                        gen = sample_fm(net, n_gen, nfe=128, solver="rk4",
                                        device=args.device,
                                        seed=1000 + seed).astype(np.float32)
                        np.save(smp_path, gen)
                else:
                    gen = np.load(smp_path)

                gen = np.load(smp_path).astype(float)
                m = extended_metrics(gen, gname)
                row = {"arch": arch, "geom": gname, "seed": seed,
                       "n_params": n_params,
                       "final_loss_ema": f"{loss_ema:.5f}",
                       "train_wallclock_s": f"{wallclock:.1f}", **{
                           k: m[k] for k in fields
                           if k in m}}
                with open(csv_path, "a", newline="") as f:
                    csv.DictWriter(f, fields).writerow(row)
                done.add((arch, gname, str(seed)))
                print(f"[{key}] params={n_params} w_fit={m['w_fit']:.4f} "
                      f"center={m['center']:.4f} tilt={m['plateau_tilt']:+.3f} "
                      f"oshoot={m['overshoot']:+.3f} "
                      f"tail_z3={m['tail_ratio_z3']:.2f} "
                      f"spill={m['spill_frac']:.4f} "
                      f"aniso={m['aniso_w_cv']:.3f}", flush=True)

    make_figure(root, n_seeds)
    print("done ->", root, flush=True)


def make_figure(root: Path, n_seeds: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    repo = root.parent.parent
    es_root = repo / "results" / "edge_shape"
    fig, axes = plt.subplots(len(ARCHS), 4, figsize=(15, 2.9 * len(ARCHS)),
                             layout="constrained")
    for ai, arch in enumerate(ARCHS):
        for gi, gname in enumerate(("disc", "iface30")):
            geom = GEOM[gname]
            ref = np.load(es_root / "data" / f"{gname}_w0_ref.npy")
            for zoom, col in ((0.15, 2 * gi), (0.5, 2 * gi + 1)):
                ax = axes[ai, col]
                s_t, g_t, _ = geom.profile(ref.astype(float), zoom, 90)
                ax.plot(s_t, g_t, "k-", lw=1.2, label="true")
                for seed in range(n_seeds):
                    p = root / "samples" / f"{arch}_{gname}_seed{seed}_rk4128.npy"
                    if not p.exists():
                        continue
                    s_g, g_g, _ = geom.profile(np.load(p).astype(float),
                                               zoom, 90)
                    ax.plot(s_g, g_g, ".", ms=2.5, alpha=0.7)
                edge = DISC_R if gname == "disc" else 0.0
                ax.axvline(edge, color="r", ls=":", lw=0.8)
                if ai == 0:
                    ax.set_title(f"{gname} " + ("(edge zoom)" if zoom == 0.15
                                                else "(wide)"))
                if col == 0:
                    ax.set_ylabel(arch)
    fig.suptitle("architecture comparison: edge profiles, rk4-128, w_true=0")
    fig.savefig(root / "fig_arch_profiles.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
