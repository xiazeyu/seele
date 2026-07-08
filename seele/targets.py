"""Sharp 2-D training / held-out targets for the M2-M4 experiments.

Every target here is **sharp** (true edge width = 0): the width measured on a
trained model's samples is therefore *entirely* generation error — the
quantity the sweeps decompose.  Contrast with :mod:`seele.synthetic`, whose
generators add a known blur ``w`` to validate the estimator itself (M1).

A :class:`Target` bundles

* a sampler for the sharp target density (the training data source),
* the list of reference interfaces (:class:`EdgeSpec`) at which generated
  edge width is measured, reusing the M1 profile builders, and
* a distance-to-nearest-measured-edge function (for the M4 trust map).

Training targets (M2 sweeps + M4 fit):   ``disc``, ``iface30``.
Held-out targets (M4 evaluation only):   ``disc_r06``, ``annulus``,
``square``, ``iface65``.

Held-out geometries differ in curvature sign (annulus inner edge), curvature
magnitude (small disc), edge multiplicity/corners (square), and orientation
(steep interface) — the axes along which a width law fitted on the training
targets could fail to transfer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .edgewidth import (
    EdgeWidthResult,
    estimate_edge_width,
    profile_normal_density,
    profile_radial_density,
)
from . import synthetic as syn


# ── Specs ───────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class EdgeSpec:
    """One reference interface of a target.

    ``kind='radial'`` uses :func:`profile_radial_density` with ``params``
    ``{center, radius}``; ``kind='normal'`` uses
    :func:`profile_normal_density` with ``params``
    ``{normal, offset, transverse_halfwidth?}``.
    """

    name: str
    kind: str                 # "radial" | "normal"
    params: dict


@dataclass(frozen=True)
class Target:
    """A sharp synthetic target with designated measurement interfaces."""

    name: str
    sample: Callable[[np.random.Generator, int], np.ndarray]  # -> [n, 2]
    edges: tuple[EdgeSpec, ...]
    edge_distance: Callable[[np.ndarray], np.ndarray]  # [m,2] -> [m] dist
    domain: tuple[float, float, float, float]          # (xmin, xmax, ymin, ymax)
    heldout: bool = False


# ── Samplers for geometries not in seele.synthetic ──────────────────────────

def _sample_annulus(rng: np.random.Generator, n: int,
                    r_in: float, r_out: float) -> np.ndarray:
    """Uniform density on the annulus ``r_in <= r <= r_out``."""
    r = np.sqrt(rng.uniform(r_in ** 2, r_out ** 2, n))
    phi = rng.uniform(0.0, 2.0 * np.pi, n)
    return np.stack([r * np.cos(phi), r * np.sin(phi)], axis=1)


def _sample_square(rng: np.random.Generator, n: int, half: float) -> np.ndarray:
    """Uniform density on the axis-aligned square ``[-half, half]^2``."""
    return rng.uniform(-half, half, size=(n, 2))


# ── Measurement ─────────────────────────────────────────────────────────────

def measure_edge(
    points: np.ndarray,
    spec: EdgeSpec,
    width_guess: float,
    n_bins: int = 40,
    span: float = 4.0,
) -> EdgeWidthResult:
    """Measure one interface of a generated point cloud (M1 machinery)."""
    p = spec.params
    if spec.kind == "radial":
        r0 = p["radius"]
        window = (max(r0 - span * width_guess, 1e-9), r0 + span * width_guess)
        s, g, sig = profile_radial_density(points, np.asarray(p["center"], float),
                                           window, n_bins=n_bins)
    elif spec.kind == "normal":
        window = (-span * width_guess, span * width_guess)
        s, g, sig = profile_normal_density(
            points, np.asarray(p["normal"], float), p["offset"], window,
            n_bins=n_bins, transverse_halfwidth=p.get("transverse_halfwidth"),
        )
    else:  # pragma: no cover - defensive
        raise ValueError(f"unknown edge kind {spec.kind!r}")
    return estimate_edge_width(s, g, sigma=sig)


def measure_edge_adaptive(
    points: np.ndarray,
    spec: EdgeSpec,
    width_guess: float = 0.03,
    n_bins: int = 40,
    span: float = 4.0,
    max_iter: int = 5,
    w_min: float = 5e-4,
    w_max: float = 0.5,
) -> tuple[EdgeWidthResult, float]:
    """Measure an edge with a self-consistent profiling window.

    The window is sized by ``width_guess`` (``+/- span * guess``, bin width
    ``2 * span * guess / n_bins``).  Because generated widths span decades
    across the sweeps, a fixed guess would either under-resolve small widths
    (bin width > w/3, the M1 validity bound) or clip the plateaus of large
    ones.  We iterate ``guess <- w_fit`` until the fitted width is consistent
    with its own window (``0.6 * guess <= w_fit <= 1.25 * guess``, i.e.
    bin width <= w/3 at the defaults).

    Returns:
        ``(result, width_guess_used)`` from the final iteration.
    """
    wg = float(width_guess)
    res = measure_edge(points, spec, wg, n_bins=n_bins, span=span)
    for _ in range(max_iter):
        if not (res.fit_success and np.isfinite(res.w_fit) and res.w_fit > 0):
            break
        if 0.6 * wg <= res.w_fit <= 1.25 * wg:
            break
        wg = float(np.clip(res.w_fit, w_min, w_max))
        res = measure_edge(points, spec, wg, n_bins=n_bins, span=span)
    return res, wg


def measure_target(
    points: np.ndarray,
    target: Target,
    width_guess: float = 0.03,
    **kwargs,
) -> list[tuple[EdgeSpec, EdgeWidthResult, float]]:
    """Adaptively measure every designated edge of ``target``."""
    return [
        (spec, *measure_edge_adaptive(points, spec, width_guess, **kwargs))
        for spec in target.edges
    ]


def combine_edge_widths(results: list[EdgeWidthResult]) -> float:
    """Target-level width: RMS of the per-edge fitted widths.

    RMS (rather than mean) so the combination is consistent with the
    quadrature bookkeeping of the width budget.
    """
    ws = np.array([r.w_fit for r in results
                   if r.fit_success and np.isfinite(r.w_fit)])
    return float(np.sqrt(np.mean(ws ** 2))) if ws.size else float("nan")


# ── Target definitions ──────────────────────────────────────────────────────

def _disc_target(name: str, radius: float, heldout: bool) -> Target:
    return Target(
        name=name,
        sample=lambda rng, n, R=radius: syn.disc(rng, n, 0.0, radius=R),
        edges=(EdgeSpec("boundary", "radial",
                        {"center": (0.0, 0.0), "radius": radius}),),
        edge_distance=lambda xy, R=radius: np.abs(
            np.linalg.norm(np.asarray(xy, float), axis=-1) - R),
        domain=(-1.4 * radius, 1.4 * radius, -1.4 * radius, 1.4 * radius),
        heldout=heldout,
    )


def _iface_target(name: str, theta_deg: float, heldout: bool) -> Target:
    th = np.deg2rad(theta_deg)
    normal = np.array([np.cos(th), np.sin(th)])

    def sample(rng: np.random.Generator, n: int) -> np.ndarray:
        pts, _ = syn.interface(rng, n, 0.0, theta=th)
        return pts

    return Target(
        name=name,
        sample=sample,
        edges=(EdgeSpec("interface", "normal",
                        {"normal": tuple(normal), "offset": 0.0}),),
        edge_distance=lambda xy, nh=normal: np.abs(
            np.asarray(xy, float) @ nh),
        domain=(-1.6, 1.6, -1.6, 1.6),
        heldout=heldout,
    )


def _annulus_target(name: str, r_in: float, r_out: float) -> Target:
    def dist(xy):
        r = np.linalg.norm(np.asarray(xy, float), axis=-1)
        return np.minimum(np.abs(r - r_in), np.abs(r - r_out))

    return Target(
        name=name,
        sample=lambda rng, n: _sample_annulus(rng, n, r_in, r_out),
        edges=(
            EdgeSpec("inner", "radial", {"center": (0.0, 0.0), "radius": r_in}),
            EdgeSpec("outer", "radial", {"center": (0.0, 0.0), "radius": r_out}),
        ),
        edge_distance=dist,
        domain=(-1.4 * r_out, 1.4 * r_out, -1.4 * r_out, 1.4 * r_out),
        heldout=True,
    )


def _square_target(name: str, half: float) -> Target:
    # Transverse clip keeps the normal profile away from the corners, where
    # two edges meet and the 1-D step model stops being valid.
    thw = 0.5 * half
    edges = tuple(
        EdgeSpec(f"face_{label}", "normal",
                 {"normal": tuple(nrm), "offset": half,
                  "transverse_halfwidth": thw})
        for label, nrm in (("+x", (1.0, 0.0)), ("-x", (-1.0, 0.0)),
                           ("+y", (0.0, 1.0)), ("-y", (0.0, -1.0)))
    )

    def dist(xy):
        xy = np.asarray(xy, float)
        return np.abs(np.max(np.abs(xy), axis=-1) - half)

    return Target(
        name=name,
        sample=lambda rng, n: _sample_square(rng, n, half),
        edges=edges,
        edge_distance=dist,
        domain=(-1.4 * half, 1.4 * half, -1.4 * half, 1.4 * half),
        heldout=True,
    )


#: All targets by name.  Training targets first, then held-out geometries.
TARGETS: dict[str, Target] = {
    t.name: t for t in (
        _disc_target("disc", 1.0, heldout=False),
        _iface_target("iface30", 30.0, heldout=False),
        _disc_target("disc_r06", 0.6, heldout=True),
        _annulus_target("annulus", 0.55, 1.0),
        _square_target("square", 0.8),
        _iface_target("iface65", 65.0, heldout=True),
    )
}

TRAIN_TARGETS: tuple[str, ...] = tuple(n for n, t in TARGETS.items() if not t.heldout)
HELDOUT_TARGETS: tuple[str, ...] = tuple(n for n, t in TARGETS.items() if t.heldout)
