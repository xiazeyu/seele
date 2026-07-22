# Edge-defect study reference (edge_shape / arch_defects)

Status snapshot of the 2026-07-22 experiments (`scripts/edge_shape.py`,
`scripts/arch_defects.py`; results under `results/edge_shape/` and
`results/arch_defects/`). Kept in sync with the lean metric set agreed on
2026-07-22. Schematic figures live in `docs/figures/`.

## 1. Metrics

![metric schematics](figures/fig_metrics_schematic.png)

| Metric | Algorithm | Reference | Notes |
|---|---|---|---|
| `w_fit` (± `w_fit_err`) | Profile density across the edge (radial: shell-normalized histogram; planar: normal-projected histogram), nonlinear fit of `A + (B−A)·Φ((s−s0)/w)` with self-consistent adaptive window (`0.6 ≤ w/guess ≤ 1.25`) | `w_true` | Core smoothing metric; generation blur `w_gen = sqrt(w_fit² − w_true²)` |
| `overshoot`, `overshoot_z` | Max positive residual above the fitted step within `\|s−s0\| ≤ 5w`, in units of step height; `_z` in units of per-bin σ | 0 | Mass pile-up just inside the edge; erf cannot represent it |
| `center` (± `center_err`) | Fitted `s0`; error from fit covariance | `R` (disc) / `0` (iface) | Boundary-location bias; sign flips with curvature and architecture |
| `spill_frac`, `excess_spill` | Direct count of samples beyond the true support (disc: `r > R`; iface: `u > 0`, `\|v\| ≤ 0.8`); excess subtracts the true-draw baseline | true value / 0 | Mass leakage *amount*; Euler-NFE-sensitive (converges last) |
| `tail_ratio_z3` | Among beyond-center samples, observed `P(z>3)` vs half-normal `0.0027`, with `z=(s−center)/w_fit`; disc weighted by shell jacobian `c/r` | 1 | Tail *shape* (how far leaked mass travels); most architecture-discriminating metric. Independent of `spill`: plain MLP has high spill + light tail, DiT lower spill + heavy tail |
| `fit_success`, `agreement_ratio` | Fit convergence flag; `w_fit / (rise_10_90 / 2.563)` | True / ≈1 | QA: `≫1` flags non-erf edge (curvature, bump) — distrust `w_fit` then |
| `aniso_w_cv` *(optional, disc only)* | Split the boundary into 8 angular sectors, run the same erf fit per sector, report `std(w)/mean(w)` | ≈0 (≲0.03 from noise) | Angular *uniformity* of the edge — the global fit averages sectors and cannot see it. ResMLP ≈0.02–0.04; DiT 0.04–0.23. Undefined for a straight interface |

Seed spread is a design, not a column: 10 seeds per condition as CSV rows
(rk4-128 only), aggregate mean ± std in analysis.

Coverage map — each metric answers one distinct question about the edge:
width (`w_fit`), interior pile-up (`overshoot`), location (`center`),
leaked amount (`spill`), leaked distance (`tail_ratio_z3`), angular
uniformity (`aniso_w_cv`), and measurement validity (`fit_success`,
`agreement_ratio`). Known blind spot (accepted): bulk plateau ripples
(mild, DiT) — deliberately out of scope for now.

Spectral bridge: the theory references (spectral bias, Fourier-space SNR)
speak in k-space, while all metrics above live in real space — deliberate,
since direct spectral estimation on point data is noisy and under the
Gaussian-blur model the erf width is the sufficient statistic. The mapping
is `w_gen` ↔ effective frequency cutoff `k* ≈ 1/w_gen`: a Gaussian blur of
width `w` multiplies the target spectrum by `exp(−k²w²/2)`. T/N sweep
results are compared against theory predictions (e.g. `τ* ∝ λ_k⁻¹`)
through this mapping.

## 2. Datasets (2-D targets)

![datasets](figures/fig_datasets.png)

| Dataset | Generation | Notes |
|---|---|---|
| `disc` | Uniform unit disc: direction uniform, `r = R·√U`; blur `+ N(0, w²·I)` | Convex curved boundary (R=1); training target |
| `iface30` | Half-strip `u∈[−L,0], v∈[−1,1]` rotated 30°; blur `+ N(0, w²·I)` | Straight edge (zero curvature); measurement clipped to `\|v\| ≤ 0.8` |
| `w_true ∈ {0, 0.02, 0.05}` | Gaussian blur of the sharp density | sharp cut-off / steep-but-continuous / slightly blurred |
| ref draws | Same generators (not FM), n = 2M, fixed seed | Three roles: (1) true profile overlay in figures; (2) spill baseline — a blurred target has intrinsic spill, `excess_spill = spill − spill_ref`; (3) null test of the measurement pipeline (fit on ref must return `w≈w_true`, overshoot≈0) |

Why keep `iface30`: it is the **zero-curvature control**. Defects that
appear on the disc but not the interface are curvature effects
(inward center bias, `agreement_ratio` 1.12–1.17 vs 1.03); defects present
on both (w_gen floor, overshoot, heavy tails) are intrinsic to FM. The 30°
rotation avoids accidental axis alignment. Held-out geometries in
`seele/targets.py` (unused so far): `disc_r06` (higher curvature),
`annulus` (negative curvature inner edge), `square` (corners), `iface65`.

## 3. Architectures (`scripts/arch_defects.py`)

| Arch | Params | Composition | Typical use | Observed edge behaviour (rk4-128, w_true=0) |
|---|---|---|---|---|
| `resmlp` | 417k | Fourier time embed (64) → in-proj → 3× ResBlock(256, SiLU) → out | ScatterPrism/seele baseline; low-dim FM | Balanced: w≈0.037–0.042, small bump, mildly heavy tail (ratio 1.7–3.3) |
| `plain` | 153k | Same embed → 3× Linear+SiLU (no residuals) → out | Simplest baseline | 2–3× wider edge (0.075–0.13), lighter-than-Gauss tail (0.4–0.7), largest seed variance |
| `dit` | 920k | Per-coordinate token (d=128) + learned pos-emb → 3× (MHA-4h + 4×MLP, adaLN-zero time cond) → zero-init head | Image-space FM standard (DiT), miniaturized | Sharpest edge (w≈0.023–0.026), biggest overshoot bump, heavy tail (4–7×), center can flip outside, strongest anisotropy |
| `dit_s` | 440k | Same as `dit` with d=88 — param-matched to resmlp | Controls for parameter count | Same defect profile as `dit` (w≈0.014–0.025, tail 5–25×, aniso 0.08–0.23, larger seed variance) → DiT behaviour is architectural, not a param-count artifact |

Dropped: `siren` (sin-activation MLP, an implicit-neural-representation
architecture, not a realistic FM backbone; its extreme tails (21–29×) and
plateau ripples were judged too exotic to keep in the comparison —
2026-07-22). Backup of its CSV rows kept outside the repo; sample npys
remain in `results/arch_defects/samples/`.

CNN: not applicable to 2-D point data (needs grid/field data); relevant once
image-space FM enters (M4 real fields) — the profile machinery transfers.

## 4. Knobs that can move the defects — ranked by expected impact

![knob effects](figures/fig_knobs.png)

Ranked by observed (or theoretically direct) effect size on the edge
metrics, in the converged-sampler regime:

| # | Knob | Range used (sensible range) | Effect |
|---|---|---|---|
| 1 | solver / NFE | euler 8–1024, rk4 128–512 | *If unconverged*, dominates everything: Euler first-order inward bias, center 0.90→0.985, spill 0→2% (∝1/NFE). At rk4-128 (our operating point) it is retired as a variable |
| 2 | architecture | 4 types (see §3) | Largest converged-regime effect: tail_ratio 0.4→25 (~60×), w_fit 0.014→0.13 (~9×), aniso 0.02→0.23. Changes defect *weights*, not types |
| 3 | `w_true` (target blur) | 0, 0.02, 0.05 (0–0.1) | Regime switch: `w_fit ≈ sqrt(w_true² + w_gen²)`; below `w_gen` the true edge is unrecoverable, above it FM reproduces the edge well |
| 4 | capacity (hidden×depth) | 256×3 fixed (64×2 smoke) | Sets the `w_gen` floor; untested axis at full scale (expected O(2×)) |
| 5 | training steps `T` | 8000 (400 smoke; 1e3–1e5) | `w_gen` shrinks then saturates; early training grossly non-erf |
| 6 | train set size `N` | 500k (1e4–2e6) | Statistical part of `w_gen` (M2 sweep axis) |
| 7 | `sigma_min` (OT path) | 0 (0–0.1) | Direct additive width floor: t=1 marginal is data ⊛ N(0,σ²) (H2) |
| 8 | geometry / curvature | disc vs iface (+held-out set) | Breaks exact erf (`agreement_ratio` 1.12–1.17 vs 1.03); flips center-bias sign; moderate size |
| 9 | seed | 3 used → 10 planned | ±10% on `w_fit` (ResMLP); up to ±40% for plain/dit_s |
| 10 | lr / batch / grad-clip | 3e-4 / 4096 / 1.0 fixed | Optimization noise; second-order once converged; deliberately constant across sweeps |
| 11 | `n_gen` (measurement) | 1M (20k smoke) | Measurement precision only; z>3 tail counts need ≥1M for ~15% Poisson error |

Training-side signal: during the T/N sweeps also record training-loss
curves, time-resolved (binned by `t`, especially `t → 1` where the edge
forms). Caveat — the global FM objective is known **not** to reflect
physics/edge error (ScatterPrism reported exactly this loss, and it is
blind to edge defects): a flat total loss can coexist with a wide `w_gen`.
The point of recording is the "loss estimator" half of the project goal —
testing whether any loss-derived quantity (late-`t` bins) predicts the
edge metrics, not treating the loss itself as a quality measure.

Untested knobs worth remembering (all currently held at a fixed default):
sampling time-grid spacing (uniform vs cosine — biases late-time steps that
form the edge), ODE vs SDE sampling (SDE adds its own edge blur), EMA of
weights (usually tightens `w_gen` and seed spread), base-distribution scale
σ₀, and loss time-weighting near t=1.
