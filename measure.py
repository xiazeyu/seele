from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import curve_fit, isotonic_regression
from scipy.spatial import cKDTree
from scipy.special import erf, ndtri

R = 1.0
IFACE_NORMAL = np.array([np.cos(np.pi / 6), np.sin(np.pi / 6)])
IFACE_TANGENT = np.array([-IFACE_NORMAL[1], IFACE_NORMAL[0]])
IFACE_HALFWIDTH = 0.8
EDGES = {"disc": R, "iface30": 0.0}
DENSITIES = {"disc": 1.0 / np.pi, "iface30": 0.5}
K_1090 = float(ndtri(0.9) - ndtri(0.1))
FIT_DEFAULTS = dict(w_guess=0.03, n_bins=40, span=4.0, max_iter=5)
PROTOCOL = dict(version=1, sampler="rk4", param_denominator_floor="max(t_eps,dt)",
                radius=R, iface_angle=30, iface_halfwidth=IFACE_HALFWIDTH,
                width="erf_sigma", min_contrast_fraction=0.5, max_bin_over_w=1 / 3,
                min_snr=3.0, plateau_fraction=0.15, window_width_min=5e-4,
                window_width_max=0.5, nn_max_points=50_000, nn_min_points=100,
                reference_seed_offset=30_000, **FIT_DEFAULTS)


def sample_disc(rng: np.random.Generator, n: int) -> np.ndarray:
    u = rng.normal(size=(n, 2))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    r = R * rng.uniform(0.0, 1.0, n) ** 0.5
    return u * r[:, None]


def sample_target(rng: np.random.Generator, n: int, dataset: str = "disc") -> np.ndarray:
    if dataset == "disc":
        return sample_disc(rng, n)
    if dataset == "iface30":
        uv = np.column_stack([rng.uniform(-1.0, 0.0, n), rng.uniform(-1.0, 1.0, n)])
        return uv @ np.stack([IFACE_NORMAL, IFACE_TANGENT])
    raise ValueError(f"unknown dataset: {dataset}")


def edge_coordinates(pts: np.ndarray, dataset: str) -> tuple[np.ndarray, np.ndarray]:
    if dataset == "disc":
        return np.linalg.norm(pts, axis=1), np.ones(len(pts), dtype=bool)
    if dataset == "iface30":
        finite = np.isfinite(pts).all(axis=1)
        return pts @ IFACE_NORMAL, (np.abs(pts @ IFACE_TANGENT) <= IFACE_HALFWIDTH) | ~finite
    raise ValueError(f"unknown dataset: {dataset}")


def step(s: np.ndarray, lo: float, hi: float, c: float, w: float) -> np.ndarray:
    return lo + 0.5 * (hi - lo) * (1.0 + erf((s - c) / (w * np.sqrt(2.0))))


def density_profile(pts: np.ndarray, window: tuple[float, float], n_bins: int = 40,
                    dataset: str = "disc") -> dict[str, Any]:
    u, selected = edge_coordinates(pts, dataset)
    edges = np.linspace(*window, n_bins + 1)
    s = 0.5 * (edges[:-1] + edges[1:])
    counts, _ = np.histogram(u[selected], bins=edges)
    area = np.pi * np.diff(edges ** 2) if dataset == "disc" else 2 * IFACE_HALFWIDTH * np.diff(edges)
    denom = max(len(pts), 1) * area
    return dict(edges=edges, s=s, counts=counts, density=counts / denom,
                sigma=np.sqrt(np.maximum(counts, 1.0)) / denom, n_total=len(pts))


def _crossing(s: np.ndarray, g_inc: np.ndarray, target: float) -> float:
    if target <= g_inc[0]:
        return float(s[0])
    if target >= g_inc[-1]:
        return float(s[-1])
    j = int(np.searchsorted(g_inc, target))
    g0, g1 = g_inc[j - 1], g_inc[j]
    return float(0.5 * (s[j - 1] + s[j])) if g1 == g0 else float(s[j - 1] + (s[j] - s[j - 1]) * (target - g0) / (g1 - g0))


def width_1090(s: np.ndarray, g: np.ndarray) -> float:
    k = max(1, round(PROTOCOL["plateau_fraction"] * len(s)))
    lo, hi = float(np.median(g[:k])), float(np.median(g[-k:]))
    mono = isotonic_regression(-g, increasing=True).x
    a, b = -lo + 0.1 * (lo - hi), -lo + 0.9 * (lo - hi)
    if not mono[0] <= a < b <= mono[-1]:
        return float("nan")
    return _crossing(s, mono, b) - _crossing(s, mono, a)


def fit_step(s: np.ndarray, g: np.ndarray, sigma: np.ndarray) -> tuple[float, float, float, float] | None:
    k = max(1, round(PROTOCOL["plateau_fraction"] * len(s)))
    lo0, hi0 = float(np.median(g[:k])), float(np.median(g[-k:]))
    noise = float(np.sqrt(np.sum(sigma[:k] ** 2) + np.sum(sigma[-k:] ** 2)) / k)
    if lo0 - hi0 <= PROTOCOL["min_snr"] * noise:
        return None
    g_mono = isotonic_regression(-g, increasing=True).x
    c0 = _crossing(s, g_mono, -0.5 * (lo0 + hi0))
    span_s = float(s[-1] - s[0])
    span_g = lo0 - hi0
    a, b = -lo0 + 0.1 * span_g, -lo0 + 0.9 * span_g
    w0 = abs(_crossing(s, g_mono, b) - _crossing(s, g_mono, a)) / K_1090
    w0 = float(np.clip(w0 if np.isfinite(w0) and w0 > 0 else 0.1 * span_s, span_s / 1e4, span_s))
    bounds = ([0.0, 0.0, s[0], span_s / 1e6], [np.inf, np.inf, s[-1], span_s * 10.0])
    try:
        p, _ = curve_fit(step, s, g, p0=[lo0, hi0, c0, w0], sigma=sigma, bounds=bounds, maxfev=20000)
    except (RuntimeError, ValueError):
        return None
    return tuple(map(float, p)) if np.isfinite(p).all() and p[0] > p[1] else None


def overshoot(s: np.ndarray, g: np.ndarray, fit: tuple[float, float, float, float],
              reach: float = 5.0) -> float:
    lo, hi, c, w = fit
    mask = np.abs(s - c) <= reach * w
    if not mask.any() or hi == lo:
        return float("nan")
    return float(np.max(g[mask] - step(s[mask], lo, hi, c, w)) / abs(hi - lo))


def measure(pts: np.ndarray, w_guess: float = 0.03, n_bins: int = 40, span: float = 4.0,
            max_iter: int = 5, dataset: str = "disc", include_profile: bool = False) -> dict[str, Any]:
    pts = np.asarray(pts, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("measure expects an (n, 2) array")
    center, wg, valid = EDGES[dataset], w_guess, None
    min_contrast = PROTOCOL["min_contrast_fraction"] * DENSITIES[dataset]
    for _ in range(max_iter + 1):
        left = max(center - span * wg, 1e-9) if dataset == "disc" else center - span * wg
        prof = density_profile(pts, (left, center + span * wg), n_bins, dataset)
        s, g, sig = prof["s"], prof["density"], prof["sigma"]
        fit = fit_step(s, g, sig)
        if fit is None:
            break
        lo, hi, c, w = fit
        k = max(1, round(PROTOCOL["plateau_fraction"] * len(s)))
        supported = (lo - hi > min_contrast
                     and np.median(g[:k]) - np.median(g[-k:]) > min_contrast
                     and s[k - 1] <= c - 0.5 * K_1090 * w
                     and c + 0.5 * K_1090 * w <= s[-k])
        if supported:
            valid = (prof, fit)
        elif valid is not None:
            break
        if 0.6 * wg <= w <= 1.25 * wg and abs(c - center) <= wg * 0.25:
            break
        next_wg = float(np.clip(max(w, np.diff(prof["edges"]).max()),
                                PROTOCOL["window_width_min"], PROTOCOL["window_width_max"]))
        if next_wg == wg and abs(c - center) < (s[1] - s[0]):
            break
        center, wg = c, next_wg

    fit = None
    w = c = w1090 = bin_over_w = float("nan")
    if valid is not None:
        prof, fit = valid
        w, c = fit[3], fit[2]
        w1090 = width_1090(prof["s"], prof["density"])
        bin_over_w = float(np.diff(prof["edges"]).max() / w)
    u, selected = edge_coordinates(pts, dataset)
    result = dict(w_fit=w, c_fit=c, c_shift=c - EDGES[dataset],
                  nonfinite_frac=float(np.mean(~np.isfinite(pts).all(axis=1))),
                  fit_valid=bool(valid is not None and bin_over_w <= PROTOCOL["max_bin_over_w"]
                                 and np.isfinite(w1090)),
                  w_1090=w1090, agreement_ratio=w1090 / (K_1090 * w), bin_over_w=bin_over_w,
                  overshoot=overshoot(prof["s"], prof["density"], fit) if fit else float("nan"),
                  spill_frac=float(np.mean(~(np.isfinite(pts[selected]).all(axis=1)
                                             & (u[selected] <= EDGES[dataset])))) if selected.any() else float("nan"))
    if include_profile:
        result["profile"] = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in prof.items()}
        result["profile"]["fit"] = fit
    return result


def nearest_neighbor(pts: np.ndarray, data: np.ndarray, dataset: str,
                     band: float, seed: int) -> dict[str, float | int]:
    ref = sample_target(np.random.default_rng(seed), len(pts), dataset)
    tree = cKDTree(np.asarray(data, dtype=np.float32))
    distances, counts = [], []
    for cloud in (pts, ref):
        u, selected = edge_coordinates(cloud, dataset)
        selected &= np.isfinite(cloud).all(axis=1) & (np.abs(u - EDGES[dataset]) <= band)
        points = cloud[selected][:PROTOCOL["nn_max_points"]]
        counts.append(len(points))
        distances.append(float(np.median(tree.query(points)[0])) if len(points) >= PROTOCOL["nn_min_points"] else float("nan"))
    ratio = distances[0] / distances[1] if distances[1] > 0 else float("nan")
    return dict(nn_ratio=ratio, nn_gen_n=counts[0], nn_ref_n=counts[1])
