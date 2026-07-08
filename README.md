# SEELE

**A-priori edge-error prediction for diffusion / flow-matching models** —
measurement, decomposition, and prediction of generated *edge width*.

This repository holds the milestone code for the qualifying-exam proposal
(`../proposal/main.tex`), built on top of the ScatterPrism flow-matching pipeline
(`../ScatterPrism`). The measurement core is deliberately dependency-light
(numpy + scipy + matplotlib); the training stack (`torch`) is an optional
extra so the M1 tools stay usable without it.

> **Scope:** synthetic data only for the core study — 1-D steps, 2-D
> discs/interfaces with analytic ground truth. `scripts/m4_realfield.py` is
> the single real-field sanity-check hook.

## Milestones

| | Milestone | Status |
|---|---|---|
| **M1** | Resolution-independent edge-width estimator (**the GATE**) | ✅ **done — gate passes** |
| M2 | Limited scaling sweeps (T, N, NFE, σ_min) + additivity of the width budget | code ready — `m2_sweep.py` / `m2_analyze.py` |
| M3 | One floor experiment (continues the M2 baselines) | code ready — `m3_floor.py` |
| M4 | Prototype predicted-width / trust-score map on held-out targets | code ready — `m4_predict.py` |

## Layout

```
seele/
  seele/                    # the package
    edgewidth.py            # M1 edge-width estimator (erf fit + 10–90 rise + overshoot)
    synthetic.py            # blurred targets with known w (M1 estimator validation)
    targets.py              # sharp train/held-out targets (measured width = generation error)
    fm.py                   # minimal CFM stack mirroring ScatterPrism (needs torch)
    sweeps.py               # M2/M3 orchestration: train / sample / measure / record
    analysis.py             # width laws (H2), additivity test (H1), floor fits (M3)
    predict.py              # M4 a-priori width-budget model + trust maps
  scripts/
    m1_validate.py          # GATE experiment: validate the estimator vs analytic ground truth
    m1_visualize.py         # communicative M1 figure gallery
    m2_sweep.py             # run the sweeps (resumable; --smoke for a minutes-scale check)
    m2_analyze.py           # per-knob scaling exponents + additivity of the budget
    m3_floor.py             # floor experiment: plateau vs continued decay (ΔAIC)
    m4_predict.py           # fit budget model, evaluate held-out (R², rel err), trust maps
    m4_realfield.py         # real-field sanity check on external .npz samples
  docs/
    questions-remaining.md  # working ideas for the OmniFocus "questions remaining" items
  data/                     # cached synthetic snapshots (regenerable from seeds)
  results/m1..m4/           # per-milestone artifacts + figures
```

## Setup

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e .            # M1 measurement only
uv pip install --python .venv/bin/python -e '.[train]'   # + torch for M2-M4
```

## M1 — the edge-width observable

A sharp interface is a step; a generative model renders it as that step
convolved with a smoothing kernel of scale `w`. Along the interface normal `s`
(in **physical units**) the transition profile is a Gaussian-convolved step

```
g(s) = A + (B − A) · Φ((s − s0) / w)
```

We report, in physical units (hence resolution-independent):

1. **fitted `w`** — least-squares fit of the profile above;
2. **model-free 10–90 % rise distance** — `rise = K_1090 · w`,
   `K_1090 = Φ⁻¹(0.9) − Φ⁻¹(0.1) ≈ 2.5631`, so `w_from_rise = rise / K_1090`
   is an assumption-light cross-check of the fit.

```python
import numpy as np
from seele import synthetic as syn
from seele.edgewidth import edge_width_disc

rng = np.random.default_rng(0)
pts = syn.disc(rng, 1_000_000, w=0.04, radius=1.0)     # disc blurred by N(0, w²)
res = edge_width_disc(pts, center=np.zeros(2), radius=1.0, width_guess=0.04)
print(res.w_fit, res.w_from_rise, res.agreement_ratio)  # ~0.040, ~0.040, ~1.0
```

## Run M1

```bash
# validation (the GATE) — writes results/m1/validation/
python scripts/m1_validate.py --n 2000000 --seeds 6

# visualizations — writes results/m1/figures/
python scripts/m1_visualize.py --n 1500000
```

### GATE result (N = 2M, 6 seeds — all checks pass)

Ground truth is exact because convolving a target with a Gaussian of std-dev `w`
is the same as adding `N(0, w²I)` noise to its samples.

| Criterion | Threshold | Value |
|---|---|---|
| recovery bias vs `w_true` (decade of widths) | < 6 % | 0.5 % |
| resolution stability (bin width ≤ w/3) | CV < 3 % | 0.15 % |
| large-N bias | < 3 % | 0.2 % |
| disc radius / orientation invariance | CV < 4 % / 2 % | 0.28 % / 0.18 % |
| **resolution independence** (unit rescale ×0.1–×100) | CV < 1 % | 0.09 % |
| fit vs 10–90 rise (median / p95, recommended regime) | < 2 % / 4 % | 0.86 % / 3.5 % |

The estimator recovers `w` unbiasedly up to bin width ≈ w/3, converges as N grows,
is invariant to radius / orientation / units, and its two internal estimates
agree to ~1 %. **Gate open — downstream milestones may proceed.**

## Figures (`results/m1/figures/`)

- `fig1_phenomenon` — a 2-D disc at increasing blur (shared axes), with its radial
  profile + erf fit; the edge visibly widens from a step to a gentle slope.
- `fig2_anatomy` — the estimator's parts (erf fit, `w` band, 10–90 rise); all
  target types collapse onto one standardized Gaussian-step curve.
- `fig3_resolution_independence` — the same physical edge measured identically
  across histogram resolutions and unit scales.
- `fig4_recovery_agreement` — recovered `w` vs truth; fit vs model-free rise.
- `fig5_targets` — the three synthetic target types (box1d / disc2d / interface2d),
  each with its extracted normal profile + erf fit.

## Run M2–M4 (needs `.[train]`)

The FM stack mirrors ScatterPrism's conventions (straight-path CFM, residual
MLP + Fourier time embedding, AdamW, fixed-step Euler sampler where
NFE = steps), with one deliberate deviation: `sigma_min` is the true OT
conditional path (`x_t = (1-(1-σ)t)x0 + t·x1`), so the `t = 1` marginal is
the data ⊛ `N(0, σ²)` — the floor H2 predicts (`w_σ = σ`). At `σ = 0` it is
exactly ScatterPrism's path.

The run matrix factorizes: **T** is read off checkpoints of a single run and
**NFE** is chosen at sampling time, so only `(N, σ_min)` combinations cost a
training run — and every joint grid involving T or NFE comes free. Everything
is resumable: finished checkpoints and measured `(T, NFE, edge)` rows are
skipped on re-run.

```bash
# minutes-scale end-to-end pipeline check (tiny grids, tiny net)
python scripts/m2_sweep.py --smoke && python scripts/m2_analyze.py --smoke
python scripts/m3_floor.py --smoke && python scripts/m4_predict.py --smoke

# the real study (hours on Apple-silicon MPS; resumable)
python scripts/m2_sweep.py                 # isolated + joint + held-out runs
python scripts/m2_analyze.py               # H2 exponents + H1 additivity test
python scripts/m3_floor.py                 # continues the SAME baselines further
python scripts/m4_predict.py               # a-priori prediction on held-out targets

# real-field sanity check on external samples (e.g. ScatterPrism unfolding)
python scripts/m4_realfield.py --samples unfolded.npz --edge1d -0.4 \
    --budget results/m4/budget_model.json --knobs 600000 16000 0.0 256
```

Sweep roles: `baseline` (all knobs saturated → T/NFE laws, T×NFE grid, M3
floor), `nsweep`/`sigsweep` (one training-time knob desaturated → N/σ laws +
crossed grids), `joint` (N×σ crossed runs — the only genuinely extra
trainings), `heldout` (M4 evaluation only, never used in fits).
