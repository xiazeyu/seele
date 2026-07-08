"""SEELE — a-priori edge-error prediction for diffusion / flow-matching models.

Milestone code and analyses for the qualifying-exam proposal (``../proposal``),
built on top of the ScatterPrism flow-matching pipeline.  This package is
deliberately dependency-light (numpy + scipy + matplotlib) so the measurement
tools can be developed and validated without the training stack.

Current contents:

* :mod:`seele.edgewidth`  — M1 resolution-independent edge-width estimator
  (+ the overshoot diagnostic of ``docs/questions-remaining.md`` Q1).
* :mod:`seele.synthetic`  — blurred synthetic targets with known edge width
  (M1 estimator validation).
* :mod:`seele.targets`    — sharp training / held-out targets whose measured
  width is pure generation error (M2-M4).
* :mod:`seele.fm`         — minimal flow-matching stack mirroring the
  ScatterPrism conventions (**optional**: needs ``pip install -e '.[train]'``).
* :mod:`seele.sweeps`     — M2/M3 sweep orchestration (train / sample /
  measure / record; optional, needs torch).
* :mod:`seele.analysis`   — width laws (H2), additivity test (H1), floor fits
  (M3); numpy/scipy only.
* :mod:`seele.predict`    — M4 a-priori width-budget model + trust maps.

Scope note: core study is on **synthetic data only** (1-D steps, 2-D
discs/interfaces with analytic ground truth); ``scripts/m4_realfield.py`` is
the one real-field sanity-check hook.
"""

__version__ = "0.1.0"
