"""M4: a-priori predicted-width model and prototype trust map.

The :class:`WidthBudgetModel` packages the M2 outputs — the saturated floor
and the four fitted knob laws — into a single callable that predicts the
edge width of a *future* run **before sampling**, from nothing but its knob
settings ``(N, T, sigma_min, NFE)``:

    ŵ²(N, T, σ, NFE) = w²_floor + Δw²_N + Δw²_T + Δw²_σ + Δw²_NFE

Laws are fitted on the **training targets only** (``disc``, ``iface30``) and
evaluated on held-out geometries the model has never seen (small disc,
annulus, square, steep interface).  The implicit universality assumption —
that the law coefficients transfer across edge geometries — is exactly what
the held-out evaluation tests; all edges share the same ``|k|^-2`` Fourier
tail, which is why transfer is plausible (H2/H3).  The per-target floor
spread is carried as a systematic on the prediction.

The prototype trust map is deliberately simple: a region is untrustworthy in
proportion to how much of it lies within the predicted width of a sharp
feature,

    trust(x) = 1 - exp(-d(x)² / (2 ŵ²)),

with ``d`` the distance to the nearest reference edge.  Larger predicted
width ⇒ a wider distrusted band around every edge.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .analysis import (
    KNOB_FIELD,
    ConfigStat,
    KnobLaw,
    PowerLaw,
    fit_power_law,
    floor_stat,
    isolated_sweep,
    knob_star,
    r2_score,
    select,
)
from .sweeps import SweepGrids
from .targets import TARGETS, Target


# ── The predicted-width model ───────────────────────────────────────────────

@dataclass
class WidthBudgetModel:
    """Additive-in-quadrature width predictor fitted on training targets."""

    floor_w2: float
    floor_w2_spread: float              # per-target floor spread (systematic)
    laws: dict[str, KnobLaw]
    fit_targets: tuple[str, ...]
    grids: SweepGrids

    def predict_w2(self, *, N: float, T: float, sigma: float, nfe: float) -> float:
        values = {"T": T, "N": N, "nfe": nfe, "sigma": sigma}
        return self.floor_w2 + sum(
            self.laws[k].delta_w2(values[k]) for k in KNOB_FIELD)

    def predict_w(self, *, N: float, T: float, sigma: float, nfe: float) -> float:
        return math.sqrt(max(self.predict_w2(N=N, T=T, sigma=sigma, nfe=nfe), 0.0))

    # — serialization —

    def to_json(self, path: Path) -> None:
        from dataclasses import asdict
        payload = {
            "floor_w2": self.floor_w2,
            "floor_w2_spread": self.floor_w2_spread,
            "fit_targets": list(self.fit_targets),
            "grids": asdict(self.grids),
            "laws": {k: v.as_dict() for k, v in self.laws.items()},
        }
        Path(path).write_text(json.dumps(payload, indent=2))

    @classmethod
    def from_json(cls, path: Path) -> "WidthBudgetModel":
        payload = json.loads(Path(path).read_text())
        laws = {}
        for k, d in payload["laws"].items():
            law = PowerLaw(**d["law"]) if d["law"] else None
            laws[k] = KnobLaw(knob=d["knob"], star=d["star"], law=law,
                              x=d["points"]["x"], dw2=d["points"]["dw2"],
                              dw2_err=d["points"]["dw2_err"])
        return cls(
            floor_w2=payload["floor_w2"],
            floor_w2_spread=payload["floor_w2_spread"],
            laws=laws,
            fit_targets=tuple(payload["fit_targets"]),
            grids=SweepGrids(**{k: tuple(v) if isinstance(v, list) else v
                                for k, v in payload["grids"].items()}),
        )


def fit_width_budget(stats: list[ConfigStat], grids: SweepGrids,
                     targets: tuple[str, ...]) -> WidthBudgetModel:
    """Fit the budget model by pooling isolated sweeps of the fit targets.

    Each target's Δw² points are computed against *its own* floor (so
    geometry-specific floor differences do not contaminate the slopes), then
    pooled into one power law per knob.  The pooled floor is the mean of the
    per-target floors; their spread is kept as a systematic.
    """
    floors = {}
    for tgt in targets:
        fs = floor_stat(stats, grids, tgt)
        if fs is not None:
            floors[tgt] = fs
    if not floors:
        raise ValueError(f"no saturated floor configs found for {targets}")
    floor_vals = np.array([f.w2_mean for f in floors.values()])
    floor_w2 = float(floor_vals.mean())
    spread = float(floor_vals.std()) if len(floor_vals) > 1 else 0.0

    laws: dict[str, KnobLaw] = {}
    for knob in KNOB_FIELD:
        star = knob_star(grids, knob)
        xs, d, derr = [], [], []
        for tgt, fs in floors.items():
            for s in isolated_sweep(stats, grids, tgt, knob):
                if np.isclose(s.knob(knob), star):
                    continue
                xs.append(float(s.knob(knob)))
                d.append(s.w2_mean - fs.w2_mean)
                e1 = s.w2_err if np.isfinite(s.w2_err) else 0.0
                e2 = fs.w2_err if np.isfinite(fs.w2_err) else 0.0
                derr.append(math.hypot(e1, e2))
        law = fit_power_law(np.array(xs), np.array(d), np.array(derr))
        laws[knob] = KnobLaw(knob, star, law, x=xs, dw2=d, dw2_err=derr)

    return WidthBudgetModel(floor_w2=floor_w2, floor_w2_spread=spread,
                            laws=laws, fit_targets=tuple(targets), grids=grids)


# ── Held-out evaluation (H3 core criterion) ─────────────────────────────────

def evaluate_heldout(stats: list[ConfigStat], model: WidthBudgetModel,
                     targets: tuple[str, ...] | None = None) -> dict:
    """Predicted vs measured width on held-out configs (R², relative error).

    Predictions use knob settings only — nothing about the target geometry —
    so this is the a-priori test: the numbers exist before the held-out runs
    are sampled.
    """
    entries = []
    for s in stats:
        if not s.heldout:
            continue
        if targets is not None and s.target not in targets:
            continue
        w_pred = model.predict_w(N=s.n_train, T=s.T, sigma=s.sigma_min, nfe=s.nfe)
        w_meas = s.w_mean
        entries.append({
            "target": s.target, "N": s.n_train, "T": s.T,
            "sigma": s.sigma_min, "nfe": s.nfe, "n_seeds": s.n_seeds,
            "w_meas": w_meas,
            "w_meas_err": (0.5 * s.w2_err / w_meas
                           if (np.isfinite(s.w2_err) and w_meas > 0) else float("nan")),
            "w_pred": w_pred,
            "rel_error": (w_pred - w_meas) / w_meas if w_meas > 0 else float("nan"),
        })
    meas = np.array([e["w_meas"] for e in entries])
    pred = np.array([e["w_pred"] for e in entries])
    rel = np.array([e["rel_error"] for e in entries if np.isfinite(e["rel_error"])])
    return {
        "entries": entries,
        "summary": {
            "n_configs": len(entries),
            "r2": r2_score(meas, pred),
            "r2_log": r2_score(np.log(meas[meas > 0]), np.log(pred[meas > 0])),
            "median_abs_rel_error": float(np.median(np.abs(rel))) if rel.size else float("nan"),
            "p90_abs_rel_error": float(np.percentile(np.abs(rel), 90)) if rel.size else float("nan"),
            "mean_rel_error": float(rel.mean()) if rel.size else float("nan"),
        },
    }


# ── Prototype trust map ─────────────────────────────────────────────────────

def trust_scores(target: Target | str, xy: np.ndarray, w_pred: float) -> np.ndarray:
    """Trust in ``[0, 1]`` at locations ``xy``: 0 on a sharp feature, ->1 far
    away, with the distrusted band scaled by the predicted width."""
    tgt = TARGETS[target] if isinstance(target, str) else target
    d = tgt.edge_distance(np.asarray(xy, float))
    if not (np.isfinite(w_pred) and w_pred > 0):
        return np.ones_like(d)
    return 1.0 - np.exp(-d ** 2 / (2.0 * w_pred ** 2))


def trust_grid(target: Target | str, w_pred: float,
               n: int = 400) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Trust map on the target's plotting domain: ``(X, Y, trust)`` grids."""
    tgt = TARGETS[target] if isinstance(target, str) else target
    x0, x1, y0, y1 = tgt.domain
    xs = np.linspace(x0, x1, n)
    ys = np.linspace(y0, y1, n)
    X, Y = np.meshgrid(xs, ys)
    pts = np.stack([X.ravel(), Y.ravel()], axis=1)
    trust = trust_scores(tgt, pts, w_pred).reshape(X.shape)
    return X, Y, trust
