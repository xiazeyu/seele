"""Analysis of the sweep measurements: width laws, additivity, floor fits.

Everything here works in **w² space**, because the budget (proposal Eq. 1) is
additive in quadrature:

    w^2  ≈  w^2_floor + Δw^2_T(T) + Δw^2_N(N) + Δw^2_NFE(NFE) + Δw^2_sigma(σ)

* ``w^2_floor`` — the fully saturated configuration (T*, N*, NFE*, σ*); it
  contains ``w_arch`` plus whatever residual the saturated knobs leave.
* ``Δw^2_knob(x)`` — the *excess* over the floor measured in the isolated
  sweep of that knob, fitted as a power law (H2).

The additive prediction for an arbitrary configuration is then
``ŵ² = w²_floor + Σ_knob Δw²_knob``, and H1 is tested by comparing ``ŵ²``
against measurements at *joint* configurations (≥ 2 knobs desaturated) that
were never used in the fits.  See ``docs/questions-remaining.md`` (Q3) for
why the joint grid — not the isolated ablations — is the actual test of H1.

Dependency note: numpy + scipy only (no pandas), consistent with the package.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict

import numpy as np
from scipy.optimize import curve_fit

from .sweeps import SweepGrids

#: The four swept knobs and the row/config field each one reads.
KNOB_FIELD = {"T": "T", "N": "n_train", "nfe": "nfe", "sigma": "sigma_min"}


def knob_star(grids: SweepGrids, knob: str) -> float:
    return {"T": grids.t_star, "N": grids.n_star,
            "nfe": grids.nfe_star, "sigma": grids.sig_star}[knob]


# ── Aggregation: rows -> per-config statistics ──────────────────────────────

@dataclass
class ConfigStat:
    """One knob configuration of one target, aggregated over seeds.

    ``w2_mean`` / ``w2_err`` are the seed mean and standard error of the
    squared target-level width (RMS over the target's edges).
    """

    target: str
    heldout: bool
    n_train: int
    sigma_min: float
    T: int
    nfe: int
    w2_mean: float
    w2_err: float
    n_seeds: int
    w_mean: float
    overshoot_mean: float
    roles: tuple[str, ...] = ()

    def knob(self, name: str) -> float:
        return getattr(self, KNOB_FIELD[name])


def measurement_ok(r: dict, max_rel_disagreement: float = 0.5,
                   max_rel_fit_err: float = 0.5) -> bool:
    """Quality gate for one edge measurement.

    Severely under-trained models produce profiles with no step at all; the
    erf fit then converges onto noise with a spurious (often tiny) width.
    Such fits betray themselves through fit/rise disagreement and a large
    relative width error — both ~1e-2 in the M1-validated regime, so the 0.5
    thresholds only remove genuinely meaningless measurements.
    """
    return (bool(r["fit_success"])
            and np.isfinite(r["w_fit"]) and r["w_fit"] > 0
            and np.isfinite(r["rel_disagreement"])
            and r["rel_disagreement"] <= max_rel_disagreement
            and np.isfinite(r["w_fit_err"])
            and r["w_fit_err"] <= max_rel_fit_err * r["w_fit"])


def aggregate_configs(rows: list[dict]) -> list[ConfigStat]:
    """Per-edge rows -> per-config stats (edges RMS-combined, then seeds).

    Rows failing :func:`measurement_ok` are dropped up front.
    """
    # 1) combine edges within one (run, T, nfe) measurement
    per_meas: dict[tuple, dict] = {}
    for r in rows:
        if not measurement_ok(r):
            continue
        key = (r["run_id"], r["T"], r["nfe"])
        m = per_meas.setdefault(key, {"w2": [], "overshoot": [], "row": r})
        m["w2"].append(r["w_fit"] ** 2)
        if np.isfinite(r["overshoot"]):
            m["overshoot"].append(r["overshoot"])

    # 2) combine seeds within one configuration
    per_cfg: dict[tuple, dict] = {}
    for m in per_meas.values():
        r = m["row"]
        key = (r["target"], r["n_train"], r["sigma_min"], r["T"], r["nfe"])
        c = per_cfg.setdefault(key, {"w2": [], "overshoot": [], "roles": set(),
                                     "heldout": r["heldout"]})
        c["w2"].append(float(np.mean(m["w2"])))          # RMS^2 over edges
        if m["overshoot"]:
            c["overshoot"].append(float(np.mean(m["overshoot"])))
        c["roles"].add(r["role"])

    stats = []
    for (target, n_train, sigma_min, T, nfe), c in per_cfg.items():
        w2 = np.asarray(c["w2"])
        err = float(w2.std(ddof=1) / math.sqrt(len(w2))) if len(w2) > 1 else float("nan")
        stats.append(ConfigStat(
            target=target, heldout=c["heldout"], n_train=n_train,
            sigma_min=sigma_min, T=T, nfe=nfe,
            w2_mean=float(w2.mean()), w2_err=err, n_seeds=len(w2),
            w_mean=float(np.sqrt(w2.mean())),
            overshoot_mean=float(np.mean(c["overshoot"])) if c["overshoot"] else float("nan"),
            roles=tuple(sorted(c["roles"])),
        ))
    return stats


def select(stats: list[ConfigStat], target: str | None = None,
           **knob_values) -> list[ConfigStat]:
    """Filter stats by target and exact knob values (knob names of KNOB_FIELD)."""
    out = []
    for s in stats:
        if target is not None and s.target != target:
            continue
        if any(not np.isclose(s.knob(k), v) for k, v in knob_values.items()):
            continue
        out.append(s)
    return out


def floor_stat(stats: list[ConfigStat], grids: SweepGrids,
               target: str) -> ConfigStat | None:
    """The fully saturated configuration of one target."""
    hits = select(stats, target, T=grids.t_star, N=grids.n_star,
                  nfe=grids.nfe_star, sigma=grids.sig_star)
    return hits[0] if hits else None


def isolated_sweep(stats: list[ConfigStat], grids: SweepGrids,
                   target: str, knob: str) -> list[ConfigStat]:
    """Configs where ``knob`` varies and every *other* knob is saturated.

    For the T sweep this also excludes M3 floor points beyond T* so that the
    law is fitted on the M2 budget range only.
    """
    others = {k: knob_star(grids, k) for k in KNOB_FIELD if k != knob}
    hits = select(stats, target, **others)
    if knob == "T":
        hits = [s for s in hits if s.T <= grids.t_star]
    return sorted(hits, key=lambda s: s.knob(knob))


# ── Power-law fits (H2) ─────────────────────────────────────────────────────

@dataclass
class PowerLaw:
    """``y = amp * x^exponent`` fitted in log-log with weights."""

    amp: float
    exponent: float
    amp_err: float
    exponent_err: float
    n_used: int
    r2_log: float

    def __call__(self, x) -> np.ndarray:
        return self.amp * np.asarray(x, dtype=float) ** self.exponent

    def as_dict(self) -> dict:
        return asdict(self)


def fit_power_law(x: np.ndarray, y: np.ndarray,
                  yerr: np.ndarray | None = None) -> PowerLaw | None:
    """Weighted least squares of ``log y`` on ``log x``.

    Points with ``y <= 0`` (floor-subtracted noise) or ``y <= yerr`` are
    excluded: below its own uncertainty a Δw² carries no slope information
    and would only bias the log fit.
    """
    x = np.asarray(x, float); y = np.asarray(y, float)
    yerr = np.full_like(y, np.nan) if yerr is None else np.asarray(yerr, float)
    keep = (x > 0) & (y > 0) & (~np.isfinite(yerr) | (y > yerr))
    x, y, yerr = x[keep], y[keep], yerr[keep]
    if x.size < 2:
        return None

    lx, ly = np.log(x), np.log(y)
    # sigma_log ~ yerr / y; fall back to unweighted if errors are missing
    w = np.where(np.isfinite(yerr) & (yerr > 0), (y / yerr) ** 2, 1.0)
    W = np.sum(w)
    mx, my = np.sum(w * lx) / W, np.sum(w * ly) / W
    sxx = np.sum(w * (lx - mx) ** 2)
    if sxx == 0:
        return None
    b = np.sum(w * (lx - mx) * (ly - my)) / sxx
    a = my - b * mx
    resid = ly - (a + b * lx)
    dof = max(x.size - 2, 1)
    s2 = np.sum(w * resid ** 2) / dof
    b_err = math.sqrt(s2 / sxx)
    a_err = math.sqrt(s2 * (1.0 / W + mx ** 2 / sxx))
    ss_tot = np.sum(w * (ly - my) ** 2)
    r2 = 1.0 - np.sum(w * resid ** 2) / ss_tot if ss_tot > 0 else float("nan")
    return PowerLaw(amp=float(np.exp(a)), exponent=float(b),
                    amp_err=float(np.exp(a) * a_err), exponent_err=float(b_err),
                    n_used=int(x.size), r2_log=float(r2))


@dataclass
class KnobLaw:
    """Isolated width law of one knob: ``Δw²(x) = law(x) - law(x_star)``.

    Anchoring at ``x_star`` makes the excess *exactly* zero at the saturated
    setting (a raw power law never reaches zero), so additive predictions
    reduce to the floor when all knobs are saturated.
    """

    knob: str
    star: float
    law: PowerLaw | None
    x: list[float] = field(default_factory=list)       # sweep points
    dw2: list[float] = field(default_factory=list)     # floor-subtracted
    dw2_err: list[float] = field(default_factory=list)

    def delta_w2(self, value: float) -> float:
        if self.law is None:
            return 0.0
        if np.isclose(value, self.star):
            return 0.0
        anchor = float(self.law(self.star)) if self.star > 0 else 0.0
        return max(float(self.law(value)) - anchor, 0.0)

    def as_dict(self) -> dict:
        return {"knob": self.knob, "star": self.star,
                "law": self.law.as_dict() if self.law else None,
                "points": {"x": self.x, "dw2": self.dw2,
                           "dw2_err": self.dw2_err}}


def fit_knob_law(stats: list[ConfigStat], grids: SweepGrids,
                 target: str, knob: str) -> KnobLaw:
    """Fit the isolated Δw² power law of one knob (floor-subtracted)."""
    floor = floor_stat(stats, grids, target)
    sweep = isolated_sweep(stats, grids, target, knob)
    star = knob_star(grids, knob)
    if floor is None or not sweep:
        return KnobLaw(knob, star, None)

    xs, d, derr = [], [], []
    for s in sweep:
        if np.isclose(s.knob(knob), star):
            continue
        xs.append(float(s.knob(knob)))
        d.append(s.w2_mean - floor.w2_mean)
        e1 = s.w2_err if np.isfinite(s.w2_err) else 0.0
        e2 = floor.w2_err if np.isfinite(floor.w2_err) else 0.0
        derr.append(math.hypot(e1, e2))
    law = fit_power_law(np.array(xs), np.array(d), np.array(derr))
    return KnobLaw(knob, star, law, x=xs, dw2=d, dw2_err=derr)


def fit_all_laws(stats: list[ConfigStat], grids: SweepGrids,
                 target: str) -> dict:
    """Floor + the four knob laws of one target."""
    floor = floor_stat(stats, grids, target)
    return {
        "target": target,
        "floor_w2": floor.w2_mean if floor else float("nan"),
        "floor_w2_err": (floor.w2_err if floor else float("nan")),
        "laws": {k: fit_knob_law(stats, grids, target, k) for k in KNOB_FIELD},
    }


# ── Additive prediction + the H1 interaction test ───────────────────────────

def predict_w2_additive(laws: dict, *, T: float, N: float,
                        nfe: float, sigma: float) -> float:
    """``ŵ² = w²_floor + Σ Δw²_knob`` from one target's fitted laws."""
    values = {"T": T, "N": N, "nfe": nfe, "sigma": sigma}
    return laws["floor_w2"] + sum(
        laws["laws"][k].delta_w2(values[k]) for k in KNOB_FIELD)


def _desaturated_knobs(s: ConfigStat, grids: SweepGrids) -> tuple[str, ...]:
    out = []
    for k in KNOB_FIELD:
        star = knob_star(grids, k)
        v = s.knob(k)
        if k == "T" and v > grids.t_star:      # M3 extension, not a knob value
            continue
        if not np.isclose(v, star):
            out.append(k)
    return tuple(sorted(out))


def additivity_report(stats: list[ConfigStat], grids: SweepGrids,
                      target: str, laws: dict) -> dict:
    """Test H1: additive predictions vs measurements at joint configs.

    A config participates if ≥ 2 knobs are desaturated (isolated-sweep
    configs are the fit inputs, so they cannot test additivity).  The report
    lists per-config relative residuals ``(w² - ŵ²) / w²`` and a per-pair
    breakdown for the interaction heatmaps.
    """
    entries = []
    for s in select(stats, target):
        if s.heldout or s.T > grids.t_star:
            continue
        knobs = _desaturated_knobs(s, grids)
        if len(knobs) < 2:
            continue
        pred = predict_w2_additive(laws, T=s.T, N=s.n_train,
                                   nfe=s.nfe, sigma=s.sigma_min)
        entries.append({
            "knobs": knobs, "T": s.T, "N": s.n_train, "nfe": s.nfe,
            "sigma": s.sigma_min, "w2_meas": s.w2_mean, "w2_err": s.w2_err,
            "w2_pred": pred,
            "rel_residual": (s.w2_mean - pred) / s.w2_mean if s.w2_mean > 0 else float("nan"),
            "n_seeds": s.n_seeds,
        })
    rel = np.array([e["rel_residual"] for e in entries if np.isfinite(e["rel_residual"])])
    summary = {
        "n_configs": len(entries),
        "median_abs_rel_residual": float(np.median(np.abs(rel))) if rel.size else float("nan"),
        "p90_abs_rel_residual": float(np.percentile(np.abs(rel), 90)) if rel.size else float("nan"),
        "mean_rel_residual": float(rel.mean()) if rel.size else float("nan"),
    }
    by_pair: dict[str, list[float]] = {}
    for e in entries:
        if len(e["knobs"]) == 2 and np.isfinite(e["rel_residual"]):
            by_pair.setdefault("x".join(e["knobs"]), []).append(e["rel_residual"])
    summary["by_pair"] = {
        p: {"n": len(v), "median_abs": float(np.median(np.abs(v))),
            "mean": float(np.mean(v))} for p, v in by_pair.items()}
    return {"target": target, "entries": entries, "summary": summary}


# ── M3: plateau vs pure-decay model comparison ──────────────────────────────

@dataclass
class FloorFit:
    """One asymptote model fitted to ``w²(x)``; AIC uses the weighted chi²."""

    model: str                  # "plateau" | "decay"
    params: dict
    chi2: float
    aic: float
    n_points: int

    def as_dict(self) -> dict:
        return asdict(self)


def _aic(chi2: float, k: int) -> float:
    return chi2 + 2 * k


def fit_floor_models(x: np.ndarray, w2: np.ndarray,
                     w2_err: np.ndarray) -> dict:
    """Fit ``w² = c + a x^-b`` (plateau) and ``w² = a x^-b`` (pure decay).

    Returns both fits plus a conservative verdict keyed on ΔAIC:
    ``|ΔAIC| < 4`` — indistinguishable over the tested range (say so);
    otherwise evidence for the lower-AIC model.  The floor claim is an
    *empirical extrapolation over the tested budgets*, never a proof.
    """
    x = np.asarray(x, float); w2 = np.asarray(w2, float)
    err = np.asarray(w2_err, float)
    keep = np.isfinite(x) & np.isfinite(w2) & (x > 0) & (w2 > 0)
    x, w2, err = x[keep], w2[keep], err[keep]
    med_err = np.nanmedian(err[err > 0]) if np.any(err > 0) else 0.05 * np.median(w2)
    err = np.where(np.isfinite(err) & (err > 0), err, med_err)
    if x.size < 4:
        return {"error": f"only {x.size} usable points", "n_points": int(x.size)}

    def plateau(xx, c, a, b):
        return c + a * xx ** (-b)

    def decay(xx, a, b):
        return a * xx ** (-b)

    fits = {}
    # pure decay first — its parameters seed the plateau fit
    try:
        p_d, _ = curve_fit(decay, x, w2, p0=[w2[0] * x[0] ** 0.5, 0.5],
                           sigma=err, absolute_sigma=True,
                           bounds=([0, 0], [np.inf, 5.0]), maxfev=20000)
        chi2_d = float(np.sum(((w2 - decay(x, *p_d)) / err) ** 2))
        fits["decay"] = FloorFit("decay", {"a": float(p_d[0]), "b": float(p_d[1])},
                                 chi2_d, _aic(chi2_d, 2), int(x.size))
    except (RuntimeError, ValueError):
        fits["decay"] = None
    try:
        a0, b0 = (fits["decay"].params["a"], fits["decay"].params["b"]) \
            if fits["decay"] else (w2[0] * x[0] ** 0.5, 0.5)
        p_p, cov_p = curve_fit(plateau, x, w2,
                               p0=[0.5 * float(w2.min()), a0, b0],
                               sigma=err, absolute_sigma=True,
                               bounds=([0, 0, 0], [np.inf, np.inf, 5.0]),
                               maxfev=20000)
        chi2_p = float(np.sum(((w2 - plateau(x, *p_p)) / err) ** 2))
        c_err = float(np.sqrt(cov_p[0, 0])) if np.all(np.isfinite(cov_p)) else float("nan")
        fits["plateau"] = FloorFit(
            "plateau",
            {"c": float(p_p[0]), "a": float(p_p[1]), "b": float(p_p[2]),
             "c_err": c_err, "w_inf": float(np.sqrt(p_p[0])),
             "w_inf_err": float(0.5 * c_err / np.sqrt(p_p[0])) if p_p[0] > 0 else float("nan")},
            chi2_p, _aic(chi2_p, 3), int(x.size))
    except (RuntimeError, ValueError):
        fits["plateau"] = None

    out = {"n_points": int(x.size),
           "decay": fits["decay"].as_dict() if fits["decay"] else None,
           "plateau": fits["plateau"].as_dict() if fits["plateau"] else None}
    if fits["decay"] and fits["plateau"]:
        d_aic = fits["decay"].aic - fits["plateau"].aic
        out["delta_aic_decay_minus_plateau"] = float(d_aic)
        if d_aic > 4:
            verdict = ("evidence for a plateau over the tested range "
                       f"(w_inf ~ {fits['plateau'].params['w_inf']:.4g}); "
                       "reported as an empirical extrapolation, not a proof of "
                       "irreducibility")
        elif d_aic < -4:
            verdict = ("the apparent floor still decays over the tested range "
                       "(pure power law preferred)")
        else:
            verdict = ("plateau and continued decay are statistically "
                       "indistinguishable over the tested budget range")
        out["verdict"] = verdict
    return out


# ── Shared metric helpers (used by M4) ──────────────────────────────────────

def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, float); y_pred = np.asarray(y_pred, float)
    keep = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[keep], y_pred[keep]
    if y_true.size < 2:
        return float("nan")
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
