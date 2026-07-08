"""M2/M3 sweep orchestration: plan, train, sample, measure, record.

The four knobs of the width budget factorize into two *training-time* knobs
(``N`` = training-set size, ``sigma_min`` = conditional-path noise) and two
*measurement-time* knobs (``T`` = optimizer steps, read off checkpoints of a
single run; ``NFE`` = sampler steps, chosen at sampling time).  The run
matrix therefore collapses:

    one training run  =  one (target, seed, N, sigma_min)
    one measurement   =  one (run, T-checkpoint, NFE)

so the ``T`` sweep, the ``NFE`` sweep, and every joint grid involving ``T``
or ``NFE`` reuse the same checkpoints — and M3 (the floor experiment) is just
*continuing* the baseline runs to larger ``T`` and larger ``NFE``, exactly as
the proposal promises ("continues the SAME M2 models").

Roles (``TrainSpec.role``):

* ``baseline`` — all knobs saturated; source of the T and NFE isolated sweeps,
  the T x NFE joint grid, and (extended) the M3 floor.
* ``nsweep`` / ``sigsweep`` — one training-time knob desaturated; their
  checkpoints/measurements also provide the N x T, N x NFE, sigma x T and
  sigma x NFE joint grids for free.
* ``joint`` — the only genuinely extra runs: a small N x sigma crossed grid
  (both training-time knobs desaturated at once).
* ``heldout`` — held-out-target runs at mixed knob settings, used **only**
  for M4 evaluation (never for fitting laws).

Every measurement appends one CSV row per reference edge to
``<outdir>/measurements.csv``; rows are keyed by
``(run_id, T, nfe, solver, edge)`` and re-running skips existing keys, so the
sweep is resumable and incremental.
"""

from __future__ import annotations

import csv
import json
import logging
import time
import zlib
from dataclasses import dataclass, asdict, replace
from pathlib import Path

import numpy as np

from .fm import FMConfig, load_checkpoint, pick_device, sample_fm, save_checkpoint, train_fm
from .targets import TARGETS, TRAIN_TARGETS, HELDOUT_TARGETS, measure_target

log = logging.getLogger(__name__)


# ── Knob grids ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SweepGrids:
    """The knob values of the study.  ``*_star`` = saturated setting.

    ``t_joint`` / ``nfe_joint`` are the small sub-grids used for crossed
    (interaction) measurements; ``joint_n`` / ``joint_sig`` define the extra
    N x sigma crossed runs.  ``floor_*`` extend the baseline for M3.
    """

    t_grid: tuple[int, ...] = (250, 500, 1000, 2000, 4000, 8000, 16000)
    t_star: int = 16000
    n_grid: tuple[int, ...] = (2000, 6000, 20000, 60000, 200000, 600000)
    n_star: int = 600000
    nfe_grid: tuple[int, ...] = (4, 8, 16, 32, 64, 128, 256)
    nfe_star: int = 256
    sig_grid: tuple[float, ...] = (0.0, 0.005, 0.01, 0.02, 0.04, 0.08)
    sig_star: float = 0.0
    # crossed sub-grids (interaction tests, Q3 / H1)
    t_joint: tuple[int, ...] = (1000, 4000, 16000)
    nfe_joint: tuple[int, ...] = (8, 32, 128)
    joint_n: tuple[int, ...] = (6000, 60000)
    joint_sig: tuple[float, ...] = (0.02, 0.04)
    # M3 floor extension of the baseline runs
    floor_steps: int = 128000
    floor_ckpts: tuple[int, ...] = (24000, 32000, 48000, 64000, 96000, 128000)
    floor_nfe: tuple[int, ...] = (384, 512, 768, 1024)


#: Full-scale study defaults.
DEFAULT_GRIDS = SweepGrids()

#: Minutes-scale end-to-end check of the whole pipeline (not for science).
SMOKE_GRIDS = SweepGrids(
    t_grid=(50, 100, 200), t_star=200,
    n_grid=(1000, 2000, 4000), n_star=4000,
    nfe_grid=(4, 16, 64), nfe_star=64,
    sig_grid=(0.0, 0.02, 0.05), sig_star=0.0,
    t_joint=(100,), nfe_joint=(16,),
    joint_n=(1000,), joint_sig=(0.05,),
    floor_steps=400, floor_ckpts=(300, 400), floor_nfe=(96, 128),
)

#: FMConfig used by the smoke preset (smaller net, smaller batches).
SMOKE_FM = FMConfig(hidden_dim=64, n_blocks=2, batch_size=512)


# ── Run specification ───────────────────────────────────────────────────────

@dataclass(frozen=True)
class TrainSpec:
    """One training run.  ``run_id`` excludes ``steps``/``role`` on purpose:
    a ``floor`` spec with the same (target, seed, N, sigma) *continues* the
    baseline run in place instead of retraining."""

    target: str
    seed: int
    n_train: int
    sigma_min: float
    steps: int
    ckpt_steps: tuple[int, ...]
    role: str

    @property
    def run_id(self) -> str:
        return f"{self.target}~N{self.n_train}~sig{self.sigma_min:g}~seed{self.seed}"


def _stable_seed(*parts) -> int:
    """Process-stable 31-bit seed from string parts (crc32, not hash())."""
    return zlib.crc32("|".join(str(p) for p in parts).encode()) & 0x7FFFFFFF


# ── Planning ────────────────────────────────────────────────────────────────

def plan_runs(
    preset: str,
    grids: SweepGrids = DEFAULT_GRIDS,
    targets: tuple[str, ...] | None = None,
    seeds: tuple[int, ...] = (0, 1, 2),
    heldout_seeds: tuple[int, ...] = (0, 1),
) -> list[TrainSpec]:
    """Build the list of training runs for a preset.

    Presets: ``isolated`` (baseline + N sweep + sigma sweep),
    ``joint`` (extra N x sigma crossed runs), ``heldout`` (M4 evaluation
    runs on held-out targets), ``floor`` (M3 continuation of the baselines),
    ``all`` (isolated + joint + heldout).
    """
    g = grids
    train_targets = tuple(targets) if targets else TRAIN_TARGETS
    # checkpoints for runs whose T axis is only needed at the crossed points
    ckpt_cross = tuple(sorted(set(g.t_joint) | {g.t_star}))
    specs: list[TrainSpec] = []

    def _isolated() -> None:
        for tgt in train_targets:
            for sd in seeds:
                specs.append(TrainSpec(tgt, sd, g.n_star, g.sig_star, g.t_star,
                                       tuple(sorted(set(g.t_grid) | {g.t_star})),
                                       "baseline"))
                for n in g.n_grid:
                    if n == g.n_star:
                        continue
                    specs.append(TrainSpec(tgt, sd, n, g.sig_star, g.t_star,
                                           ckpt_cross, "nsweep"))
                for sig in g.sig_grid:
                    if sig == g.sig_star:
                        continue
                    specs.append(TrainSpec(tgt, sd, g.n_star, sig, g.t_star,
                                           ckpt_cross, "sigsweep"))

    def _joint() -> None:
        for tgt in train_targets:
            for sd in seeds:
                for n in g.joint_n:
                    for sig in g.joint_sig:
                        specs.append(TrainSpec(tgt, sd, n, sig, g.t_star,
                                               ckpt_cross, "joint"))

    def _heldout() -> None:
        n_mid = g.n_grid[len(g.n_grid) // 2]
        sig_mid = g.sig_grid[len(g.sig_grid) // 2]
        n_lo, sig_hi = g.n_grid[0], g.sig_grid[-1]
        configs = [(g.n_star, g.sig_star), (n_mid, g.sig_star),
                   (g.n_star, sig_mid), (n_lo, sig_hi)]
        for tgt in (targets or HELDOUT_TARGETS):
            if tgt not in HELDOUT_TARGETS:
                continue
            for sd in heldout_seeds:
                for n, sig in configs:
                    specs.append(TrainSpec(tgt, sd, n, sig, g.t_star,
                                           ckpt_cross, "heldout"))

    def _floor() -> None:
        for tgt in train_targets:
            for sd in seeds:
                specs.append(TrainSpec(
                    tgt, sd, g.n_star, g.sig_star, g.floor_steps,
                    tuple(sorted(set(g.t_grid) | set(g.floor_ckpts) | {g.t_star})),
                    "floor"))

    actions = {"isolated": [_isolated], "joint": [_joint],
               "heldout": [_heldout], "floor": [_floor],
               "all": [_isolated, _joint, _heldout]}
    if preset not in actions:
        raise ValueError(f"unknown preset {preset!r} (have {sorted(actions)})")
    for fn in actions[preset]:
        fn()
    return list(dict.fromkeys(specs))  # dedupe (e.g. n_mid == n_star)


def measurement_pairs(spec: TrainSpec, grids: SweepGrids) -> list[tuple[int, int]]:
    """The (T, NFE) grid measured for one run, by role (see module doc)."""
    g = grids
    star = (g.t_star, g.nfe_star)
    if spec.role == "baseline":
        pairs = {(t, g.nfe_star) for t in spec.ckpt_steps}
        pairs |= {(g.t_star, k) for k in g.nfe_grid}
        pairs |= {(t, k) for t in g.t_joint for k in g.nfe_joint}   # T x NFE
    elif spec.role in ("nsweep", "sigsweep"):
        pairs = {star}
        pairs |= {(t, g.nfe_star) for t in g.t_joint}   # knob x T
        pairs |= {(g.t_star, k) for k in g.nfe_joint}   # knob x NFE
    elif spec.role == "joint":
        t_mid = g.t_joint[len(g.t_joint) // 2]
        k_mid = g.nfe_joint[len(g.nfe_joint) // 2]
        pairs = {star, (t_mid, k_mid)}
    elif spec.role == "heldout":
        t_lo, k_lo = g.t_joint[0], g.nfe_joint[0]
        pairs = {star, (t_lo, g.nfe_star), (g.t_star, k_lo), (t_lo, k_lo)}
    elif spec.role == "floor":
        pairs = {(t, g.nfe_star) for t in spec.ckpt_steps}
        pairs |= {(spec.steps, k) for k in (*g.nfe_grid, *g.floor_nfe)}
    else:  # pragma: no cover - defensive
        raise ValueError(f"unknown role {spec.role!r}")
    return sorted(pairs)


# ── Training runs (resumable / continuable) ─────────────────────────────────

def _ckpt_path(run_dir: Path, step: int) -> Path:
    return run_dir / f"ckpt_{step:08d}.pt"


def ensure_run(
    spec: TrainSpec,
    cfg: FMConfig,
    outdir: Path,
    device=None,
) -> Path:
    """Train ``spec`` unless its checkpoints already exist; continue if the
    existing run is shorter than ``spec.steps`` (M3 floor path).

    Returns the run directory containing ``ckpt_*.pt``, ``last.pt`` (with
    optimizer state for continuation) and ``meta.json``.
    """
    cfg = replace(cfg, sigma_min=spec.sigma_min)
    run_dir = Path(outdir) / "runs" / spec.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    meta_path = run_dir / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {
        "spec": asdict(spec), "config": cfg.as_dict(),
        "final_step": 0, "ckpt_steps": [], "wallclock_s": 0.0,
    }
    if meta["config"] != cfg.as_dict():
        raise RuntimeError(
            f"{spec.run_id}: existing run used a different FMConfig — "
            f"remove {run_dir} or keep the architecture fixed (w_arch!).")

    wanted = sorted(set(spec.ckpt_steps) | {spec.steps})
    missing = [s for s in wanted if not _ckpt_path(run_dir, s).exists()]
    if not missing and meta["final_step"] >= spec.steps:
        return run_dir

    target = TARGETS[spec.target]
    data = target.sample(
        np.random.default_rng(_stable_seed("data", spec.run_id)), spec.n_train)

    net, start_step, opt_state = None, 0, None
    last_path = run_dir / "last.pt"
    if last_path.exists() and meta["final_step"] > 0:
        net, payload = load_checkpoint(last_path, device)
        start_step = int(payload["step"])
        opt_state = payload.get("optimizer")
        if start_step >= spec.steps and missing:
            raise RuntimeError(
                f"{spec.run_id}: trained past requested checkpoints "
                f"{missing} without saving them — remove {run_dir} to retrain.")

    log.info("training %s: steps %d -> %d (%d checkpoints)",
             spec.run_id, start_step, spec.steps, len(wanted))
    net, result = train_fm(
        cfg, data, steps=spec.steps, ckpt_steps=tuple(wanted), device=device,
        seed=_stable_seed("train", spec.run_id) + start_step,
        net=net, optimizer_state=opt_state, start_step=start_step,
    )
    for step, state in result.checkpoints.items():
        save_checkpoint(_ckpt_path(run_dir, step), state, cfg, step,
                        meta={"spec": asdict(spec)})
    save_checkpoint(last_path,
                    {k: v.detach().cpu() for k, v in net.state_dict().items()},
                    cfg, result.final_step, meta={"spec": asdict(spec)},
                    optimizer_state=result.optimizer_state)

    meta["final_step"] = result.final_step
    meta["ckpt_steps"] = sorted(set(meta["ckpt_steps"]) | set(result.checkpoints))
    meta["wallclock_s"] = meta["wallclock_s"] + result.wallclock_s
    meta["spec"] = asdict(spec)
    meta_path.write_text(json.dumps(meta, indent=2))
    return run_dir


# ── Measurement + CSV record ────────────────────────────────────────────────

#: Fixed CSV schema of ``measurements.csv``.
FIELDS = [
    "run_id", "target", "heldout", "role", "seed", "n_train", "sigma_min",
    "T", "nfe", "solver", "edge",
    "w_fit", "w_fit_err", "w_from_rise", "agreement_ratio", "rel_disagreement",
    "rmse", "fit_success", "overshoot", "overshoot_z", "overshoot_loc_w",
    "width_guess", "n_meas", "n_bins", "wallclock_s",
]


def _row_key(row: dict) -> tuple:
    return (row["run_id"], int(row["T"]), int(row["nfe"]),
            row["solver"], row["edge"])


def load_measurements(csv_path: Path) -> list[dict]:
    """Load measurement rows with numeric fields parsed; last write wins."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return []
    by_key: dict[tuple, dict] = {}
    with open(csv_path, newline="") as f:
        for raw in csv.DictReader(f):
            row: dict = dict(raw)
            for k in ("seed", "n_train", "T", "nfe", "n_meas", "n_bins"):
                row[k] = int(float(row[k]))
            for k in ("sigma_min", "w_fit", "w_fit_err", "w_from_rise",
                      "agreement_ratio", "rel_disagreement", "rmse",
                      "overshoot", "overshoot_z", "overshoot_loc_w",
                      "width_guess", "wallclock_s"):
                row[k] = float(row[k])
            row["fit_success"] = raw["fit_success"] in ("True", "true", "1")
            row["heldout"] = raw["heldout"] in ("True", "true", "1")
            by_key[_row_key(row)] = row
    return list(by_key.values())


def append_measurements(csv_path: Path, rows: list[dict]) -> None:
    csv_path = Path(csv_path)
    new_file = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})


def measure_run(
    spec: TrainSpec,
    grids: SweepGrids,
    outdir: Path,
    measure_n: int = 500_000,
    solver: str = "euler",
    device=None,
    existing_keys: set | None = None,
    width_guess: float = 0.03,
    n_bins: int = 40,
) -> list[dict]:
    """Sample + measure every missing (T, NFE) pair of one run.

    Returns the newly produced rows (already appended to
    ``<outdir>/measurements.csv``).
    """
    outdir = Path(outdir)
    run_dir = outdir / "runs" / spec.run_id
    target = TARGETS[spec.target]
    csv_path = outdir / "measurements.csv"
    existing = existing_keys if existing_keys is not None else {
        _row_key(r) for r in load_measurements(csv_path)}

    new_rows: list[dict] = []
    for T, nfe in measurement_pairs(spec, grids):
        keys = [(spec.run_id, T, nfe, solver, e.name) for e in target.edges]
        if all(k in existing for k in keys):
            continue
        t0 = time.perf_counter()
        net, _ = load_checkpoint(_ckpt_path(run_dir, T), device)
        pts = sample_fm(net, measure_n, nfe=nfe, solver=solver, device=device,
                        seed=_stable_seed("meas", spec.run_id, T, nfe))
        wall = time.perf_counter() - t0
        for edge_spec, res, wg in measure_target(pts, target, width_guess,
                                                 n_bins=n_bins):
            row = {
                "run_id": spec.run_id, "target": spec.target,
                "heldout": target.heldout, "role": spec.role,
                "seed": spec.seed, "n_train": spec.n_train,
                "sigma_min": spec.sigma_min, "T": T, "nfe": nfe,
                "solver": solver, "edge": edge_spec.name,
                "w_fit": res.w_fit, "w_fit_err": res.w_fit_err,
                "w_from_rise": res.w_from_rise,
                "agreement_ratio": res.agreement_ratio,
                "rel_disagreement": res.rel_disagreement,
                "rmse": res.rmse, "fit_success": res.fit_success,
                "overshoot": res.overshoot, "overshoot_z": res.overshoot_z,
                "overshoot_loc_w": res.overshoot_loc_w,
                "width_guess": wg, "n_meas": measure_n, "n_bins": n_bins,
                "wallclock_s": wall,
            }
            new_rows.append(row)
            existing.add(_row_key(row))
        log.info("measured %s T=%d nfe=%d: w=%s (%.1fs)",
                 spec.run_id, T, nfe,
                 [f"{r['w_fit']:.4g}" for r in new_rows[-len(target.edges):]],
                 wall)
    if new_rows:
        append_measurements(csv_path, new_rows)
    return new_rows


def load_grids(outdir: Path) -> SweepGrids:
    """Grids the sweep in ``outdir`` was run with (written by :func:`execute`)."""
    p = Path(outdir) / "grids.json"
    if not p.exists():
        return DEFAULT_GRIDS
    raw = json.loads(p.read_text())
    return SweepGrids(**{k: tuple(v) if isinstance(v, list) else v
                         for k, v in raw.items()})


# ── One-call executor ───────────────────────────────────────────────────────

def execute(
    specs: list[TrainSpec],
    grids: SweepGrids,
    cfg: FMConfig,
    outdir: Path,
    measure_n: int = 500_000,
    solver: str = "euler",
    device: str | None = None,
) -> Path:
    """Train + measure every spec (skipping finished work); returns CSV path."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    dev = pick_device(device)
    csv_path = outdir / "measurements.csv"
    existing = {_row_key(r) for r in load_measurements(csv_path)}

    (outdir / "grids.json").write_text(json.dumps(asdict(grids), indent=2))
    for i, spec in enumerate(specs, 1):
        t0 = time.perf_counter()
        ensure_run(spec, cfg, outdir, device=dev)
        rows = measure_run(spec, grids, outdir, measure_n=measure_n,
                           solver=solver, device=dev, existing_keys=existing)
        print(f"[{i:3d}/{len(specs)}] {spec.role:<9s} {spec.run_id:<36s} "
              f"+{len(rows):3d} rows  ({time.perf_counter() - t0:.1f}s)")
    return csv_path
