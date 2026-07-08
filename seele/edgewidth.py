"""Resolution-independent edge-width estimator for generated sharp features.

This module implements milestone **M1** of the edge-error proposal: a local,
dimensionful, resolution-independent measurement of how wide a *generated* sharp
feature (edge / interface / discontinuity) is.

Physical model
--------------
A sharp interface in a target field is a step.  A generative model reproduces it
as the step convolved with a smoothing kernel of scale ``w``.  Along the local
interface normal ``s`` (in the *physical* units of the data domain), the
transition profile is therefore a **Gaussian-convolved step**::

    g(s) = A + (B - A) * Phi((s - s0) / w)

where ``Phi`` is the standard-normal CDF, ``A`` / ``B`` are the plateau levels on
either side, ``s0`` the interface location, and ``w`` the edge width we want.
``Phi`` follows from convolving a Heaviside step with a zero-mean Gaussian of
standard deviation ``w``; ``w`` is thus reported directly in physical length
units, which is what makes the observable resolution-independent.

Two estimates are produced for every interface and are meant to agree:

* **fitted w** — least-squares fit of the Gaussian-convolved step above;
* **model-free 10-90 % rise distance** — the ``s`` distance over which the
  (monotonised) profile climbs from 10 % to 90 % of its total change.  For an
  exact Gaussian-convolved step this equals ``w * K_1090`` with
  ``K_1090 = Phi^{-1}(0.9) - Phi^{-1}(0.1) ~= 2.5631``, so
  ``w_from_rise = rise / K_1090`` is an assumption-light cross-check of the fit.

Profile builders turn raw samples (point clouds) into a ``(s, g)`` profile:

* :func:`profile_hist_1d`      — 1-D density across an edge (histogram);
* :func:`profile_ecdf_1d`      — 1-D empirical CDF (grid-free, for CDF jumps);
* :func:`profile_radial_density` — radial density of a 2-D/3-D disc/ball;
* :func:`profile_normal_density` — density along the normal of a straight
  interface (projection onto ``n_hat``).

The top-level entry point is :func:`estimate_edge_width`, which takes a profile
and returns an :class:`EdgeWidthResult`.  Thin wrappers
(:func:`edge_width_box_1d`, :func:`edge_width_disc`,
:func:`edge_width_interface`) cover the common synthetic targets end-to-end.

The module depends only on ``numpy`` and ``scipy`` so it can be imported and
validated without the training stack.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
import logging

import numpy as np
from scipy.optimize import curve_fit, isotonic_regression
from scipy.special import erf, ndtri  # ndtri == inverse standard-normal CDF

log = logging.getLogger(__name__)

_SQRT2 = np.sqrt(2.0)

#: 10-90 % rise distance of a unit-width Gaussian-convolved step,
#: ``Phi^{-1}(0.9) - Phi^{-1}(0.1)``.  ``rise = K_1090 * w`` exactly.
K_1090: float = float(ndtri(0.9) - ndtri(0.1))  # ~= 2.5631031311


# ── The parametric edge model ───────────────────────────────────────────────

def gaussian_convolved_step(
    s: np.ndarray,
    level_low: float,
    level_high: float,
    center: float,
    width: float,
) -> np.ndarray:
    """Gaussian-convolved step ``A + (B-A)*Phi((s-s0)/w)``.

    Written with :func:`scipy.special.erf` for speed:
    ``Phi(z) = 0.5 * (1 + erf(z / sqrt(2)))``.

    Args:
        s:          Normal coordinate(s), physical units.
        level_low:  Plateau level as ``s -> -inf`` (``A``).
        level_high: Plateau level as ``s -> +inf`` (``B``).
        center:     Interface location ``s0``.
        width:      Edge width ``w`` (standard deviation of the smoothing kernel).

    Returns:
        Model profile evaluated at ``s``.
    """
    z = (np.asarray(s, dtype=float) - center) / (width * _SQRT2)
    return level_low + 0.5 * (level_high - level_low) * (1.0 + erf(z))


# ── Result container ────────────────────────────────────────────────────────

@dataclass
class EdgeWidthResult:
    """Outcome of an edge-width measurement on a single interface.

    All width-like quantities (``w_fit``, ``rise_10_90``, ``w_from_rise``,
    ``center``) are in the physical units of the input coordinate ``s``.

    Attributes:
        w_fit:        Fitted edge width ``w`` (kernel std-dev).  ``nan`` on failure.
        w_fit_err:    1-sigma std-error of ``w_fit`` from the fit covariance.
        center:       Fitted interface location ``s0``.
        level_low:    Fitted low-side plateau ``A``.
        level_high:   Fitted high-side plateau ``B``.
        rmse:         RMS residual of the fit (in ``g`` units).
        fit_success:  Whether the non-linear fit converged.
        rise_10_90:   Model-free 10-90 % rise distance.
        w_from_rise:  ``rise_10_90 / K_1090`` — width implied by the rise distance.
        agreement_ratio:  ``w_fit / w_from_rise`` (should be ~1).
        rel_disagreement: ``|w_fit - w_from_rise| / w_fit`` (should be ~0).
        n_points:     Number of profile points used.
        overshoot:    Peak residual above the fitted step near the edge, in
                      units of the step height (see :func:`overshoot_metric`).
        overshoot_z:  Same peak in units of the per-point uncertainty (a
                      significance; ``nan`` when no ``sigma`` was supplied).
        overshoot_loc_w: Signed location of the peak relative to the fitted
                      center, in units of ``w_fit``.
    """

    w_fit: float
    w_fit_err: float
    center: float
    level_low: float
    level_high: float
    rmse: float
    fit_success: bool
    rise_10_90: float
    w_from_rise: float
    agreement_ratio: float
    rel_disagreement: float
    n_points: int
    overshoot: float = float("nan")
    overshoot_z: float = float("nan")
    overshoot_loc_w: float = float("nan")

    def as_dict(self) -> dict:
        return asdict(self)


# ── Model-free 10-90 % rise distance ────────────────────────────────────────

def _robust_levels(s: np.ndarray, g: np.ndarray, tail_frac: float = 0.15) -> tuple[float, float]:
    """Estimate the two plateau levels from the outer ``tail_frac`` of the profile.

    Uses medians for robustness against edge overshoot / noise.  ``s`` is assumed
    sorted ascending; returns ``(left_level, right_level)``.
    """
    n = len(s)
    k = max(1, int(round(tail_frac * n)))
    left = float(np.median(g[:k]))
    right = float(np.median(g[-k:]))
    return left, right


def _interp_crossing(s: np.ndarray, g_inc: np.ndarray, target: float) -> float:
    """First ``s`` at which a monotone-increasing ``g_inc`` reaches ``target``.

    Local linear interpolation between the bracketing samples — near-unbiased on a
    fine grid because it imposes no shape across the (curved) transition
    shoulders.  Clamps to the profile endpoints when ``target`` lies outside the
    observed range.
    """
    if target <= g_inc[0]:
        return float(s[0])
    if target >= g_inc[-1]:
        return float(s[-1])
    j = int(np.searchsorted(g_inc, target))
    g0, g1 = g_inc[j - 1], g_inc[j]
    s0, s1 = s[j - 1], s[j]
    if g1 == g0:
        return float(0.5 * (s0 + s1))
    return float(s0 + (s1 - s0) * (target - g0) / (g1 - g0))


def rise_distance_10_90(
    s: np.ndarray,
    g: np.ndarray,
    level_low: float | None = None,
    level_high: float | None = None,
) -> tuple[float, float]:
    """Model-free 10-90 % rise distance of a transition profile.

    Makes **no** parametric assumption about the edge shape.  The profile is
    oriented to be increasing, monotonised by isotonic regression (so noisy
    plateaus do not create spurious crossings), and the 10 % / 90 % levels — set
    relative to the two robust plateau levels — are located by interpolation.

    Args:
        s:          Normal coordinate, ascending, physical units.
        g:          Profile values at ``s``.
        level_low:  Optional override for the low plateau level.
        level_high: Optional override for the high plateau level.

    Returns:
        ``(rise, w_from_rise)`` where ``rise`` is the 10-90 % distance and
        ``w_from_rise = rise / K_1090``.
    """
    s = np.asarray(s, dtype=float)
    g = np.asarray(g, dtype=float)
    order = np.argsort(s)
    s, g = s[order], g[order]

    lo, hi = _robust_levels(s, g)
    if level_low is not None:
        lo = level_low
    if level_high is not None:
        hi = level_high

    ascending = hi >= lo
    g_work = g if ascending else -g
    lo_w, hi_w = (lo, hi) if ascending else (-lo, -hi)

    # Monotone (increasing) projection — keeps the estimate model-free while
    # taming statistical noise on the plateaus, then interpolate the crossings.
    g_mono = isotonic_regression(g_work, increasing=True).x

    span = hi_w - lo_w
    if span <= 0:
        return float("nan"), float("nan")
    t_lo = lo_w + 0.10 * span
    t_hi = lo_w + 0.90 * span
    s_lo = _interp_crossing(s, g_mono, t_lo)
    s_hi = _interp_crossing(s, g_mono, t_hi)
    rise = abs(s_hi - s_lo)
    return rise, rise / K_1090


# ── Parametric fit ──────────────────────────────────────────────────────────

def fit_gaussian_convolved_step(
    s: np.ndarray,
    g: np.ndarray,
    sigma: np.ndarray | None = None,
) -> tuple[dict, bool]:
    """Least-squares fit of :func:`gaussian_convolved_step` to ``(s, g)``.

    Robustly initialised from the data (plateau medians, midpoint crossing, and
    the raw 10-90 rise for the width) so it converges for both ascending and
    descending edges without user hints.

    Args:
        s:      Normal coordinate, physical units (any order; sorted internally).
        g:      Profile values.
        sigma:  Optional per-point 1-sigma uncertainties (e.g. Poisson errors of a
                histogram) used to weight the fit.

    Returns:
        ``(params, success)``.  ``params`` has keys ``level_low``, ``level_high``,
        ``center``, ``width``, ``width_err``, ``rmse``; values are ``nan`` when the
        fit fails.
    """
    s = np.asarray(s, dtype=float)
    g = np.asarray(g, dtype=float)
    order = np.argsort(s)
    s, g = s[order], g[order]
    if sigma is not None:
        sigma = np.asarray(sigma, dtype=float)[order]

    span_s = float(s[-1] - s[0]) if len(s) > 1 else 1.0
    lo0, hi0 = _robust_levels(s, g)

    # Initial centre: midpoint crossing of the monotonised profile.
    mid = 0.5 * (lo0 + hi0)
    g_work = g if hi0 >= lo0 else -g
    g_mono = isotonic_regression(g_work, increasing=True).x
    center0 = _interp_crossing(s, g_mono, mid if hi0 >= lo0 else -mid)

    # Initial width from the raw model-free rise (falls back to a fraction of span).
    rise0, w_rise0 = rise_distance_10_90(s, g, lo0, hi0)
    width0 = w_rise0 if np.isfinite(w_rise0) and w_rise0 > 0 else 0.1 * span_s
    width0 = float(np.clip(width0, span_s / 1e4, span_s))

    p0 = [lo0, hi0, center0, width0]
    # width strictly positive; everything else free.
    bounds = (
        [-np.inf, -np.inf, s[0] - span_s, span_s / 1e6],
        [np.inf, np.inf, s[-1] + span_s, span_s * 10.0],
    )

    failed = {
        "level_low": np.nan, "level_high": np.nan, "center": np.nan,
        "width": np.nan, "width_err": np.nan, "rmse": np.nan,
    }
    try:
        popt, pcov = curve_fit(
            gaussian_convolved_step, s, g, p0=p0, sigma=sigma,
            absolute_sigma=False, bounds=bounds, maxfev=20000,
        )
    except (RuntimeError, ValueError) as exc:  # pragma: no cover - defensive
        log.warning("edge-width fit failed: %s", exc)
        return failed, False

    perr = np.sqrt(np.diag(pcov)) if np.all(np.isfinite(pcov)) else np.full(4, np.nan)
    resid = g - gaussian_convolved_step(s, *popt)
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    params = {
        "level_low": float(popt[0]),
        "level_high": float(popt[1]),
        "center": float(popt[2]),
        "width": float(abs(popt[3])),
        "width_err": float(perr[3]),
        "rmse": rmse,
    }
    return params, True


# ── Overshoot diagnostic ────────────────────────────────────────────────────

def overshoot_metric(
    s: np.ndarray,
    g: np.ndarray,
    params: dict,
    sigma: np.ndarray | None = None,
    reach: float = 5.0,
) -> tuple[float, float, float]:
    """Quantify overshoot: excess density above the fitted step near the edge.

    A Gaussian-convolved step is monotone; a positive residual *bump* just
    inside the edge (mass pile-up / Gibbs-like ringing — see
    ``docs/questions-remaining.md`` Q1) is information the erf fit cannot
    represent.  We report the largest positive residual against the fitted
    profile within ``|s - center| <= reach * w``:

    * ``overshoot``     — peak residual / |step height| (shape-relative);
    * ``overshoot_z``   — peak residual / its per-point uncertainty
      (a significance, guards against reading noise as ringing);
    * ``loc_over_w``    — signed peak location, ``(s_peak - center) / w``
      (pile-up sits just inside the high plateau, sampler overshoot can land
      elsewhere).

    Args:
        s:      Profile coordinate (physical units).
        g:      Profile values.
        params: Fit parameters from :func:`fit_gaussian_convolved_step`.
        sigma:  Optional per-point 1-sigma uncertainties (for the z-score).
        reach:  Half-window around the fitted center, in units of ``w``.

    Returns:
        ``(overshoot, overshoot_z, loc_over_w)`` — all ``nan`` if the fit
        failed or no point lies in the window.
    """
    w = params.get("width", float("nan"))
    c = params.get("center", float("nan"))
    step = abs(params.get("level_high", np.nan) - params.get("level_low", np.nan))
    if not (np.isfinite(w) and w > 0 and np.isfinite(c) and step > 0):
        return float("nan"), float("nan"), float("nan")

    s = np.asarray(s, dtype=float)
    g = np.asarray(g, dtype=float)
    mask = np.abs(s - c) <= reach * w
    if not np.any(mask):
        return float("nan"), float("nan"), float("nan")

    resid = g[mask] - gaussian_convolved_step(
        s[mask], params["level_low"], params["level_high"], c, w)
    j = int(np.argmax(resid))
    peak = float(resid[j])
    amp = peak / step
    loc = float((s[mask][j] - c) / w)
    if sigma is not None:
        sig = np.asarray(sigma, dtype=float)[mask]
        z = float(peak / sig[j]) if sig[j] > 0 else float("nan")
    else:
        z = float("nan")
    return amp, z, loc


# ── Top-level estimate ──────────────────────────────────────────────────────

def estimate_edge_width(
    s: np.ndarray,
    g: np.ndarray,
    sigma: np.ndarray | None = None,
) -> EdgeWidthResult:
    """Measure edge width from a transition profile ``(s, g)``.

    Combines the parametric fit (:func:`fit_gaussian_convolved_step`) with the
    model-free rise distance (:func:`rise_distance_10_90`) and packages both,
    plus their agreement, into an :class:`EdgeWidthResult`.

    Args:
        s:      Normal coordinate in physical units.
        g:      Transition profile values (density, CDF, ...).
        sigma:  Optional per-point uncertainties for the weighted fit.

    Returns:
        :class:`EdgeWidthResult`.
    """
    s = np.asarray(s, dtype=float)
    g = np.asarray(g, dtype=float)
    params, ok = fit_gaussian_convolved_step(s, g, sigma=sigma)
    rise, w_rise = rise_distance_10_90(s, g)

    w_fit = params["width"]
    if np.isfinite(w_fit) and np.isfinite(w_rise) and w_fit > 0:
        ratio = w_fit / w_rise
        rel = abs(w_fit - w_rise) / w_fit
    else:
        ratio = rel = float("nan")

    over, over_z, over_loc = overshoot_metric(s, g, params, sigma=sigma)

    return EdgeWidthResult(
        w_fit=w_fit,
        w_fit_err=params["width_err"],
        center=params["center"],
        level_low=params["level_low"],
        level_high=params["level_high"],
        rmse=params["rmse"],
        fit_success=ok,
        rise_10_90=rise,
        w_from_rise=w_rise,
        agreement_ratio=ratio,
        rel_disagreement=rel,
        n_points=int(len(s)),
        overshoot=over,
        overshoot_z=over_z,
        overshoot_loc_w=over_loc,
    )


# ── Profile builders: raw samples -> (s, g) ─────────────────────────────────

def profile_hist_1d(
    samples: np.ndarray,
    window: tuple[float, float],
    n_bins: int = 40,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """1-D density profile across an edge, by histogramming ``samples``.

    The bin count sets the sampling *resolution*: recovering a width ``w`` that is
    stable as ``n_bins`` varies (over a sensible range) is exactly the
    resolution-independence claim M1 must demonstrate.

    Args:
        samples: 1-D array of sample coordinates.
        window:  ``(s_min, s_max)`` region around the edge to profile.
        n_bins:  Number of histogram bins over ``window``.

    Returns:
        ``(centers, density, sigma)`` — bin centres, normalised density
        (probability per unit length), and Poisson 1-sigma density errors.
    """
    samples = np.asarray(samples, dtype=float).ravel()
    s_min, s_max = window
    edges = np.linspace(s_min, s_max, n_bins + 1)
    counts, _ = np.histogram(samples, bins=edges)
    bin_w = (s_max - s_min) / n_bins
    n_total = samples.size
    density = counts / (n_total * bin_w)
    sigma = np.sqrt(np.maximum(counts, 1.0)) / (n_total * bin_w)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, density, sigma


def profile_ecdf_1d(
    samples: np.ndarray,
    window: tuple[float, float] | None = None,
    n_grid: int = 200,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Grid-free empirical-CDF profile.

    Appropriate when the *CDF itself* jumps (a point mass / delta), so the smooth
    generated CDF is a Gaussian-convolved step.  Evaluated on a grid only for
    convenience; the estimate itself needs no binning.

    Args:
        samples: 1-D sample coordinates.
        window:  Optional ``(s_min, s_max)`` clip; defaults to the data range.
        n_grid:  Number of grid points at which to report the ECDF.

    Returns:
        ``(grid, cdf, sigma)`` — grid points, empirical CDF, and binomial
        1-sigma errors ``sqrt(F(1-F)/N)``.
    """
    samples = np.asarray(samples, dtype=float).ravel()
    xs = np.sort(samples)
    n = xs.size
    if window is None:
        window = (float(xs[0]), float(xs[-1]))
    grid = np.linspace(window[0], window[1], n_grid)
    cdf = np.searchsorted(xs, grid, side="right") / n
    sigma = np.sqrt(np.clip(cdf * (1.0 - cdf), 1e-12, None) / n)
    return grid, cdf, sigma


def _signed_normalised_profile(
    coord: np.ndarray,
    window: tuple[float, float],
    n_bins: int,
    shell_centers: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Shared helper: histogram ``coord`` over ``window`` and divide out geometry.

    ``shell_centers`` is the geometric measure per unit ``coord`` at each bin
    centre (e.g. ``2*pi*r`` in 2-D, ``4*pi*r^2`` in 3-D), so that dividing the
    counts by ``shell_centers * bin_w`` recovers a flat number density that steps
    at the boundary.  ``None`` means unit measure (a plain 1-D density).

    Poisson errors use ``sqrt(max(counts, 1))`` per bin, so ``sigma`` is strictly
    positive even in empty bins outside the interface (avoids zero fit weights).
    """
    coord = np.asarray(coord, dtype=float).ravel()
    s_min, s_max = window
    edges = np.linspace(s_min, s_max, n_bins + 1)
    bin_w = (s_max - s_min) / n_bins
    centers = 0.5 * (edges[:-1] + edges[1:])
    n_total = max(coord.size, 1)

    counts, _ = np.histogram(coord, bins=edges)
    shell = np.ones_like(centers) if shell_centers is None else np.asarray(shell_centers, float)
    shell = np.maximum(shell, 1e-30)
    denom = n_total * shell * bin_w
    density = counts / denom
    sigma = np.sqrt(np.maximum(counts, 1.0)) / denom
    return centers, density, sigma


def profile_radial_density(
    points: np.ndarray,
    center: np.ndarray,
    window: tuple[float, float],
    n_bins: int = 40,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Radial number-density profile of a disc (2-D) or ball (3-D) interface.

    Bins samples by distance ``r`` from ``center`` and divides by the shell
    ``jacobian`` (``2*pi*r`` in 2-D, ``4*pi*r^2`` in 3-D) so that a uniform disc /
    ball becomes a flat plateau that steps down to zero at the boundary.

    Args:
        points: ``[N, d]`` sample coordinates (``d`` in {2, 3}).
        center: ``[d]`` interface centre.
        window: ``(r_min, r_max)`` radial band bracketing the boundary.
        n_bins: Radial bins over ``window``.

    Returns:
        ``(r_centers, density, sigma)``.
    """
    points = np.asarray(points, dtype=float)
    center = np.asarray(center, dtype=float).ravel()
    d = points.shape[1]
    r = np.linalg.norm(points - center, axis=1)

    # Geometric shell measure evaluated at each bin centre (per unit r).
    r_min, r_max = window
    r_centers = 0.5 * (np.linspace(r_min, r_max, n_bins + 1)[:-1]
                       + np.linspace(r_min, r_max, n_bins + 1)[1:])
    if d == 2:
        shell = 2.0 * np.pi * r_centers
    elif d == 3:
        shell = 4.0 * np.pi * r_centers ** 2
    else:  # pragma: no cover - generic fallback
        shell = np.ones_like(r_centers)
    return _signed_normalised_profile(r, window, n_bins, shell_centers=shell)


def profile_normal_density(
    points: np.ndarray,
    normal: np.ndarray,
    offset: float,
    window: tuple[float, float],
    n_bins: int = 40,
    transverse_halfwidth: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Density along the normal of a straight (planar) interface.

    Projects every point onto the unit normal ``n_hat`` to get the signed normal
    coordinate ``s = (x - offset*n_hat) . n_hat``, then histograms.  The result is
    orientation-independent: rotating the interface leaves ``w`` unchanged.

    Args:
        points:  ``[N, d]`` sample coordinates.
        normal:  ``[d]`` interface normal (need not be unit length).
        offset:  Signed distance of the interface from the origin along ``n_hat``.
        window:  ``(s_min, s_max)`` band around the interface.
        n_bins:  Bins over ``window``.
        transverse_halfwidth: If given, keep only points within this distance of
            the interface *in the transverse plane* (useful when the interface is
            a finite patch rather than a full plane).

    Returns:
        ``(s_centers, density, sigma)`` — density is probability per unit normal
        length, per unit transverse measure (an overall constant that cancels in
        the width fit).
    """
    points = np.asarray(points, dtype=float)
    normal = np.asarray(normal, dtype=float).ravel()
    n_hat = normal / np.linalg.norm(normal)
    s = points @ n_hat - offset

    if transverse_halfwidth is not None:
        transverse = points - np.outer(points @ n_hat, n_hat)
        # distance from the interface centroid in the transverse plane
        t_center = transverse.mean(axis=0)
        t_dist = np.linalg.norm(transverse - t_center, axis=1)
        keep = t_dist <= transverse_halfwidth
        s = s[keep]

    return _signed_normalised_profile(s, window, n_bins, shell_centers=None)


# ── End-to-end convenience wrappers ─────────────────────────────────────────

def _auto_window(center: float, width_guess: float, span: float = 4.0) -> tuple[float, float]:
    """A symmetric ``+/- span * width_guess`` window around an edge location."""
    return (center - span * width_guess, center + span * width_guess)


def edge_width_box_1d(
    samples: np.ndarray,
    edge_loc: float,
    width_guess: float,
    n_bins: int = 40,
    span: float = 4.0,
) -> EdgeWidthResult:
    """Measure the width of a 1-D density edge near ``edge_loc``.

    Args:
        samples:     1-D samples (e.g. from a box / top-hat target).
        edge_loc:    Approximate physical location of the edge.
        width_guess: Rough width, used only to size the profiling window.
        n_bins:      Histogram bins across the window.
        span:        Window half-extent in units of ``width_guess``.
    """
    window = _auto_window(edge_loc, width_guess, span)
    s, g, sig = profile_hist_1d(samples, window, n_bins=n_bins)
    return estimate_edge_width(s, g, sigma=sig)


def edge_width_disc(
    points: np.ndarray,
    center: np.ndarray,
    radius: float,
    width_guess: float,
    n_bins: int = 40,
    span: float = 4.0,
) -> EdgeWidthResult:
    """Measure the boundary width of a 2-D disc / 3-D ball of radius ``radius``."""
    window = _auto_window(radius, width_guess, span)
    window = (max(window[0], 1e-9), window[1])
    r, g, sig = profile_radial_density(points, center, window, n_bins=n_bins)
    return estimate_edge_width(r, g, sigma=sig)


def edge_width_interface(
    points: np.ndarray,
    normal: np.ndarray,
    offset: float,
    width_guess: float,
    n_bins: int = 40,
    span: float = 4.0,
    transverse_halfwidth: float | None = None,
) -> EdgeWidthResult:
    """Measure the width of a straight planar interface at signed ``offset``."""
    window = _auto_window(0.0, width_guess, span)
    s, g, sig = profile_normal_density(
        points, normal, offset, window, n_bins=n_bins,
        transverse_halfwidth=transverse_halfwidth,
    )
    return estimate_edge_width(s, g, sigma=sig)
