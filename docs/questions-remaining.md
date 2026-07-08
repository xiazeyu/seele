# Questions remaining — working ideas

Companion notes for the OmniFocus *SEELE (qual) → questions remaining* items.
Each section records the current best thinking, concrete hypotheses, and where
the codebase already has (or gets) a hook to test them. Update in place as the
milestones produce evidence.

---

## Q1 — Overshoot in the close-up figure

**The observation.** In the proposal's Fig. 1 (left panel, ScatterPrism
close-up at the kinematic cutoff `t = -0.4`), the *generated* density does not
just smooth the step — it **overshoots the plateau just inside the edge**
(rises above ground truth), then leaks past the boundary. A pure
Gaussian-convolved step `A + (B-A)·Φ((s-s0)/w)` is monotone and can never do
this, so the overshoot is information the erf fit deliberately ignores.

**Why it matters.** Overshoot is direct evidence that the *effective smoothing
kernel is not Gaussian*. A Gaussian kernel is the only non-negative kernel
with no ringing; overshoot means the kernel has negative side lobes — or that
something other than convolution is going on. Three competing hypotheses:

1. **Gibbs / spectral truncation.** Spectral bias makes the model an
   effective *low-pass filter*. A hard (sinc-like) frequency cutoff produces
   ringing with ~9% overshoot (classic Gibbs); a soft (Gaussian) roll-off
   produces none. Overshoot amplitude is therefore a probe of the *shape* of
   the effective spectral filter, complementary to `w` which probes its
   *scale*.
   *Prediction:* overshoot amplitude tracks training budget `T` (the last
   converged frequency sets the ringing period ≈ `w`), and its distance from
   the edge scales with `w` itself.
2. **Mass conservation / pile-up.** The PF-ODE transports mass; it cannot
   create or destroy it. If the learned velocity field fails to push enough
   mass *across* the boundary region, the surplus piles up just inside — the
   overshoot bump is the mass that "should" have populated the leaked tail.
   *Prediction:* the integrated surplus inside the edge ≈ the integrated
   leaked mass outside it (a mass-balance check across the interface).
3. **Sampler discretization.** Euler steps in the velocity-stiff region near
   an edge can overshoot the target manifold at low NFE.
   *Prediction:* overshoot amplitude decreases with NFE at fixed model,
   vanishing (or saturating at the Gibbs level) as NFE → ∞.

**How to test with this codebase.**
- `seele.edgewidth.overshoot_metric()` quantifies the overshoot amplitude
  (peak excess above the fitted plateau, in units of the step height) and its
  location relative to the fitted edge. `estimate_edge_width` carries it in
  `EdgeWidthResult.overshoot`.
- M2 sweeps record `overshoot` per config for free — plotting it vs `T`, vs
  `NFE`, and vs `σ_min` separates hypotheses 1 and 3 with no extra runs.
- Mass balance (hypothesis 2): integrate `(g_generated − g_fit_step)` inside
  vs outside the fitted center over the profiling window; equal-and-opposite
  areas support pile-up.

**M1 impact (why the estimator survives overshoot).** Plateau levels use
tail medians (`_robust_levels`) and the 10–90 rise uses an isotonic
(monotonised) profile — both are robust to a localized bump. But *report*
overshoot alongside `w` rather than silently absorbing it: two models with the
same `w` and different overshoot are not equally trustworthy.

**If it's real (any hypothesis):** overshoot is a second, independent order
parameter — "ringing amplitude" next to "edge width." That could become the
first falsifiable observable separating *spectral-bias* blur from
*transport/discretization* blur, which the width alone cannot distinguish.

---

## Q2 — Any insights for the CV field?

The proposal is deliberately framed on scientific fields, but every mechanism
in the width budget has a computer-vision counterpart. Candidate insights, in
increasing order of ambition:

1. **A sharpness-aware complement to FID.** Natural images have `1/f` spectra
   and their edges have the same `|k|⁻²` tails as our synthetic steps. The
   edge-width observable applied to generated images (along detected edge
   normals, e.g. hair strands, text glyphs, object silhouettes) is a local,
   interpretable sharpness metric that FID/IS cannot localize. Cheap first
   experiment: measure `w` on a text-rendering benchmark vs human legibility.
2. **Predicting what few-step samplers lose first.** For distilled /
   consistency / rectified-flow models, NFE ∈ {1,2,4} makes `w_NFE` the
   *dominant* budget term. The budget predicts which image features degrade
   first as steps shrink — and conversely how many steps a feature of scale
   `w` needs. This is an a-priori "steps vs sharpness" curve per image region,
   useful for adaptive-compute samplers (spend NFE only inside the critical
   window of sharp features).
3. **Trust maps for generative restoration.** Super-resolution, inpainting,
   and deblurring with diffusion priors *hallucinate* edges. A predicted-width
   map flags where the prior cannot support the claimed sharpness — i.e.
   where a crisp edge in the output is invention rather than evidence.
   Direct CV translation of the M4 trust map.
4. **Latent-space caveat.** Most production models are latent diffusion; the
   decoder can re-sharpen (or alias) edges the latent model blurred. Comparing
   edge width measured in latent space vs pixel space would show how much of
   the budget survives the VAE — likely a paper-sized question on its own.
5. **Where remedies should act.** Frequency-aware fixes (Fourier-space
   losses, low-pass FM, Lazy Diffusion's schedule-as-regularizer) currently
   apply globally. A predicted-width map tells such methods *where* their
   regularizer pays rent — regularize only near predicted-wide edges,
   keep the smooth regions cheap.

**Positioning note for the qual:** frame these as *transfer* of the
diagnostic, not new claims — the committee question is likely "why synthetic
targets and not ImageNet?", and the answer is: the synthetic setting is where
ground truth makes the budget falsifiable; CV is where the validated
diagnostic gets used.

---

## Q3 — How can we *see* Prop. 1? Is an ablation study enough?

("Prop. 1" = the additive-in-quadrature width budget, Eq. (1) of the
proposal / H1.)

**Short answer: no — isolated ablations alone are *not* enough.** Sweeping one
knob with the others saturated establishes each *per-knob law*
`w_knob(x)` (that is H2). It says nothing about how the terms *combine*. The
additive claim is exactly the statement that there are **no interaction
terms**, and interactions are invisible unless at least two knobs are
de-saturated *simultaneously*. This is the classic ANOVA point: main effects
from one-factor-at-a-time designs cannot detect interactions by construction.

**What "seeing" Prop. 1 takes — three levels of evidence:**

1. **Isolated sweeps (M2 core, necessary).** Fit each `w²_knob(x)` with the
   others saturated. This pins the marginal laws and the shared floor
   `w²_floor = w²_arch + (residual saturated terms)`.
2. **Joint (crossed) grids (the actual test).** Choose 2-knob interaction
   grids where both terms are *comparable in size* — additivity is trivially
   "confirmed" when one term dominates, so the informative regime is
   `w_a ≈ w_b`. For each joint config, predict
   `ŵ² = w²_floor + Δw²_a + Δw²_b` from the *isolated* fits and compare with
   the measured `w²`. The pass metric is the interaction residual
   `(w² − ŵ²)/w²` across the grid (with seed-level error bars), not a
   correlation coefficient — correlations flatter additive models.
   Most informative pairs, by expected coupling:
   - `NFE × σ_min` — both act at the endpoint of the ODE; most likely to
     couple, most damaging if they do.
   - `T × N` — both shape the learned score; theory (KDE view vs learning
     dynamics) treats them as independent mechanisms, worth stressing.
3. **The visual** — a *budget stacked-bar figure*: for each joint config,
   stack the predicted `w²` contributions (floor, T, N, NFE, σ_min) next to
   the measured `w²` bar. Additivity is "seen" as the stacks matching the
   measurements across configs where the mix varies. A second panel maps the
   interaction residual over the 2-knob grid as a heatmap — structure in that
   map (not noise) is the falsification signature.

**Failure is informative and quotable.** If quadrature additivity fails, test
the two natural fallbacks before abandoning H1: additivity in a different
power (`w^p`, fit `p`), or additivity of *rates* only in the small-term limit
(first-order additivity). If the interaction is confined to one pair (e.g.
`NFE × σ_min`), Eq. (1) survives with a documented cross term — a stronger
qual result than uncritical confirmation.

**Where the code implements this.** `scripts/m2_sweep.py --preset joint`
runs the crossed grids; `scripts/m2_analyze.py` fits the isolated laws,
computes interaction residuals, and renders the stacked-budget and
residual-heatmap figures described above.

---

*File lives at `seele/docs/questions-remaining.md`; linked from the OmniFocus
"questions remaining" action group.*
