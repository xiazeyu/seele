"""Controlled synthetic targets with a *known* edge width.

For M1 we validate the edge-width estimator against analytic ground truth.  The
key fact we exploit:

    convolving a target density with an isotropic Gaussian of standard deviation
    ``w`` is exactly the same as adding ``N(0, w^2 I)`` noise to its samples.

So every generator here draws samples from a sharp target (a step / disc / half
plane) and adds Gaussian noise of a chosen ``w``; the resulting point cloud has a
smoothed interface whose true width is ``w`` by construction.  This lets us ask
whether the estimator recovers ``w`` and how stable that recovery is.

All generators take a ``numpy.random.Generator`` so runs are reproducible from a
seed.  Widths and coordinates are in arbitrary but *consistent* physical units.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


# ── Ground-truth generators ─────────────────────────────────────────────────

def box_1d(rng: np.random.Generator, n: int, w: float,
           edge: float = 1.0, plateau: float = 1.0) -> np.ndarray:
    """1-D box edge: ``Uniform(edge - L, edge)`` blurred by ``N(0, w^2)``.

    The right edge (a downward density step) sits at ``edge``; ``L = max(plateau,
    8 w)`` keeps the interior plateau flat over the profiling window.

    Returns:
        ``[n]`` sample array.
    """
    L = max(plateau, 8.0 * w)
    return rng.uniform(edge - L, edge, n) + rng.normal(0.0, w, n)


def disc(rng: np.random.Generator, n: int, w: float, radius: float = 1.0,
         center=(0.0, 0.0), dim: int = 2) -> np.ndarray:
    """Uniform disc (``dim=2``) or ball (``dim=3``) blurred by isotropic ``N(0, w^2 I)``.

    Returns:
        ``[n, dim]`` sample array.  The boundary (radial density step) is at
        ``radius`` from ``center``.
    """
    center = np.asarray(center, dtype=float)
    if center.size != dim:
        center = np.zeros(dim)
    # Uniform in the ball: radius ~ U^{1/dim}, direction uniform on the sphere.
    u = rng.normal(size=(n, dim))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    r = radius * rng.uniform(0.0, 1.0, n) ** (1.0 / dim)
    pts = u * r[:, None] + rng.normal(0.0, w, size=(n, dim))
    return pts + center


def interface(rng: np.random.Generator, n: int, w: float, theta: float = 0.0,
              offset: float = 0.0, extent: float = 1.0):
    """Straight 2-D interface: a filled half-strip rotated by ``theta`` (radians).

    The filled region is ``{u in [-L, 0], v in [-extent, extent]}`` in a frame
    rotated by ``theta``; the density steps down across ``u = 0``.  The interface
    normal in the lab frame is the rotated x-axis.

    Returns:
        ``(points [n, 2], normal [2])`` — pass ``normal`` and ``offset`` straight
        to :func:`seele.edgewidth.profile_normal_density`.
    """
    L = max(1.0, 8.0 * w)
    u = rng.uniform(-L, 0.0, n) + offset
    v = rng.uniform(-extent, extent, n)
    base = np.stack([u, v], axis=1)
    c, s = np.cos(theta), np.sin(theta)
    rot = np.array([[c, -s], [s, c]])
    pts = base @ rot.T + rng.normal(0.0, w, size=(n, 2))
    normal = np.array([c, s])
    return pts, normal


# ── Reproducible named datasets (small, for the record / visualisations) ─────

@dataclass
class SyntheticSpec:
    """A reproducible synthetic dataset specification."""
    name: str
    kind: str          # "box_1d" | "disc" | "interface"
    w_true: float
    n: int
    seed: int
    params: dict

    def generate(self):
        rng = np.random.default_rng(self.seed)
        if self.kind == "box_1d":
            return box_1d(rng, self.n, self.w_true, **self.params), None
        if self.kind == "disc":
            return disc(rng, self.n, self.w_true, **self.params), None
        if self.kind == "interface":
            return interface(rng, self.n, self.w_true, **self.params)
        raise ValueError(f"unknown kind {self.kind!r}")


#: A small representative suite used for the M1 record and figures.  Sizes are
#: kept modest so the cached ``.npz`` snapshot stays light; the manifest fully
#: specifies regeneration at any ``n`` from ``(kind, w_true, seed, params)``.
M1_SUITE = [
    SyntheticSpec("box_w0.05",       "box_1d",    0.05, 120_000, 101, {"edge": 1.0}),
    SyntheticSpec("disc_R1_w0.04",   "disc",      0.04, 180_000, 102, {"radius": 1.0}),
    SyntheticSpec("ball_R1_w0.04",   "disc",      0.04, 220_000, 103, {"radius": 1.0, "dim": 3}),
    SyntheticSpec("iface_40deg_w0.03", "interface", 0.03, 180_000, 104, {"theta": float(np.deg2rad(40.0))}),
]


def save_suite(outdir: str | Path, suite=M1_SUITE) -> Path:
    """Generate and cache the representative suite as ``.npz`` files + a manifest.

    Synthetic data is fully determined by ``(kind, w_true, n, seed, params)``, so
    the manifest is the real ground truth; the ``.npz`` files are a convenience
    snapshot.  Returns the manifest path.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for spec in suite:
        pts, normal = spec.generate()
        npz = outdir / f"{spec.name}.npz"
        payload = {"points": pts, "w_true": spec.w_true}
        if normal is not None:
            payload["normal"] = normal
        np.savez_compressed(npz, **payload)
        manifest.append({
            "name": spec.name, "kind": spec.kind, "w_true": spec.w_true,
            "n": spec.n, "seed": spec.seed, "params": spec.params,
            "file": npz.name,
        })
    mpath = outdir / "manifest.json"
    with open(mpath, "w") as f:
        json.dump(manifest, f, indent=2)
    return mpath
