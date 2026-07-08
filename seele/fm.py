"""Minimal flow-matching stack for the M2-M4 sweep experiments.

This module deliberately **mirrors the ScatterPrism conventions**
(``../ScatterPrism/scatterprism/models.py`` and ``networks.py``) so that
measured scaling laws transfer to the real pipeline:

* straight-line interpolant, ``t ~ U(0,1)``, integration ``t: 0 -> 1`` from
  noise to data, target velocity constant in ``t``;
* velocity net = residual MLP on ``concat([x_t, time_embed(t)])`` with a
  Fourier time embedding (geometric frequencies + learned projection);
* MSE regression loss, AdamW optimizer;
* deterministic fixed-step ODE samplers (``euler`` / ``midpoint`` / ``rk4``)
  where for Euler **NFE == number of steps**.

One deliberate deviation, required by the proposal's H2:

    ScatterPrism's ``sigma`` adds a *constant* Gaussian jitter to the probe
    location only (``x_t = t*x1 + (1-t)*x0 + sigma*eps`` with target
    ``u = x1 - x0`` unchanged).  Here we implement the Lipman/Tong **OT
    conditional path with sigma_min**::

        x_t = (1 - (1 - sigma_min) * t) * x0 + t * x1
        u_t = x1 - (1 - sigma_min) * x0

    so that the ``t = 1`` marginal is exactly the data convolved with
    ``N(0, sigma_min^2 I)`` — the minimum-noise floor ``w_sigma = sigma_min``
    that H2 predicts.  At ``sigma_min = 0`` this reduces *exactly* to the
    ScatterPrism straight path, so the baseline is convention-identical.

Training budget ``T`` is counted in **optimizer steps** at fixed batch size.
The learning rate is constant (no plateau/cosine schedule): a schedule would
make the ``T`` sweep depend on scheduler state and confound the scaling law.

``torch`` is an optional dependency (``pip install -e '.[train]'``); the M1
measurement stack stays importable without it.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
import logging
import math
import time

import numpy as np
import torch
from torch import nn

log = logging.getLogger(__name__)


# ── Device ──────────────────────────────────────────────────────────────────

def pick_device(device: str | None = None) -> torch.device:
    """Best available device: explicit > cuda > mps > cpu."""
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ── Conditional path (the sigma_min knob lives here) ────────────────────────

def sample_conditional_path(
    x0: torch.Tensor,
    x1: torch.Tensor,
    t: torch.Tensor,
    sigma_min: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """OT conditional path with minimum noise ``sigma_min``.

    ``x_t = (1 - (1 - sigma_min) t) x0 + t x1``,
    ``u_t = x1 - (1 - sigma_min) x0``  (constant in ``t``).

    ``sigma_min = 0`` is exactly ScatterPrism's straight path
    (``sample_conditional_pt`` + ``compute_conditional_vector_field``).

    Args:
        x0: ``[B, D]`` base (noise) samples.
        x1: ``[B, D]`` data samples.
        t:  ``[B]`` times in ``[0, 1]``.
        sigma_min: Minimum-noise scale of the conditional path.

    Returns:
        ``(x_t, u_t)`` both ``[B, D]``.
    """
    t = t.view(-1, 1)
    a = 1.0 - (1.0 - sigma_min) * t
    x_t = a * x0 + t * x1
    u_t = x1 - (1.0 - sigma_min) * x0
    return x_t, u_t


# ── Networks (compact mirror of scatterprism/networks.py) ───────────────────

class FourierEmbedding(nn.Module):
    """Fourier time embedding: fixed geometric frequencies -> learned projection.

    Mirrors ScatterPrism's default ``FourierEmbedding`` (frequencies spanning
    ``1 .. max_freq`` cycles, sin+cos features, then a linear layer).
    """

    def __init__(self, dim: int = 64, max_freq: float = 64.0):
        super().__init__()
        half = dim // 2
        freqs = torch.exp(torch.linspace(0.0, math.log(max_freq), half))
        self.register_buffer("freqs", freqs)
        self.proj = nn.Linear(2 * half, dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        ang = 2.0 * math.pi * t.view(-1, 1) * self.freqs
        return self.proj(torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1))


class ResBlock(nn.Module):
    """Linear -> SiLU -> Linear with a skip, SiLU on the way out."""

    def __init__(self, dim: int):
        super().__init__()
        self.lin1 = nn.Linear(dim, dim)
        self.lin2 = nn.Linear(dim, dim)
        self.act = nn.SiLU()

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.act(h + self.lin2(self.act(self.lin1(h))))


class VelocityField(nn.Module):
    """Residual-MLP velocity field ``v_theta(x_t, t)``.

    Forward signature mirrors ScatterPrism's ``FlowMatchingResNet``:
    ``(x [B, D], t [B]) -> velocity [B, D]``, with the inner net applied to
    ``concat([x, time_embed(t)])``.
    """

    def __init__(
        self,
        data_dim: int = 2,
        hidden_dim: int = 256,
        n_blocks: int = 3,
        time_embed_dim: int = 64,
    ):
        super().__init__()
        self.data_dim = data_dim
        self.time_embed = FourierEmbedding(time_embed_dim)
        self.in_proj = nn.Linear(data_dim + time_embed_dim, hidden_dim)
        self.blocks = nn.ModuleList(ResBlock(hidden_dim) for _ in range(n_blocks))
        self.out_proj = nn.Linear(hidden_dim, data_dim)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h = self.act(self.in_proj(torch.cat([x, self.time_embed(t)], dim=-1)))
        for blk in self.blocks:
            h = blk(h)
        return self.out_proj(h)


# ── Config ──────────────────────────────────────────────────────────────────

@dataclass
class FMConfig:
    """Hyperparameters held **fixed** across all sweeps (w_arch = constant floor).

    Only ``sigma_min`` is a swept knob; everything else defines the fixed
    architecture / optimization setting of the study.
    """

    data_dim: int = 2
    hidden_dim: int = 256
    n_blocks: int = 3
    time_embed_dim: int = 64
    lr: float = 3e-4              # constant — see module docstring
    weight_decay: float = 1e-5    # ScatterPrism default
    batch_size: int = 4096
    grad_clip: float = 1.0
    sigma_min: float = 0.0

    def build_net(self) -> VelocityField:
        return VelocityField(self.data_dim, self.hidden_dim,
                             self.n_blocks, self.time_embed_dim)

    def as_dict(self) -> dict:
        return asdict(self)


# ── Training ────────────────────────────────────────────────────────────────

@dataclass
class TrainResult:
    """Checkpoints and diagnostics from one training run."""

    checkpoints: dict[int, dict]        # step -> cpu model state_dict
    final_step: int
    optimizer_state: dict               # for exact continuation (M3)
    loss_history: list[tuple[int, float]] = field(default_factory=list)
    wallclock_s: float = 0.0


def train_fm(
    cfg: FMConfig,
    data: np.ndarray,
    *,
    steps: int,
    ckpt_steps: tuple[int, ...] = (),
    device: torch.device | str | None = None,
    seed: int = 0,
    net: VelocityField | None = None,
    optimizer_state: dict | None = None,
    start_step: int = 0,
    log_every: int = 0,
) -> tuple[VelocityField, TrainResult]:
    """Train (or continue training) a CFM velocity field.

    Mirrors ScatterPrism's ``CFM.compute_loss``: ``t ~ U(0,1)`` per batch
    element, conditional path -> MSE regression of ``v_theta(x_t, t)`` on
    ``u_t``.  Gaussian base ``x0 ~ N(0, I)``.

    Args:
        cfg:        Fixed hyperparameters (incl. the swept ``sigma_min``).
        data:       ``[N, D]`` training samples (the finite-``N`` knob: pass
                    exactly the ``N`` points the run is allowed to see).
        steps:      Train up to this optimizer step (absolute, not relative).
        ckpt_steps: Steps at which to snapshot the model (each snapshot is a
                    ``T`` value of the sweep).  ``steps`` itself is always
                    snapshotted.
        device:     Override device (default: cuda > mps > cpu).
        seed:       Seed for init, batching, and the path randomness.
        net:        Existing network to continue training (M3 floor runs).
        optimizer_state: Optimizer state to restore for continuation.
        start_step: Step the continuation starts from (0 for fresh runs).

    Returns:
        ``(net, TrainResult)`` — net left on ``device``.
    """
    dev = pick_device(None if device is None else str(device))
    torch.manual_seed(seed)

    x_data = torch.as_tensor(np.asarray(data, dtype=np.float32), device=dev)
    n = x_data.shape[0]

    if net is None:
        net = cfg.build_net().to(dev)
    optim = torch.optim.AdamW(net.parameters(), lr=cfg.lr,
                              weight_decay=cfg.weight_decay)
    if optimizer_state is not None:
        optim.load_state_dict(optimizer_state)

    want = sorted(set(int(s) for s in ckpt_steps) | {int(steps)})
    want = [s for s in want if start_step < s <= steps]
    checkpoints: dict[int, dict] = {}
    loss_hist: list[tuple[int, float]] = []
    ema_loss = None

    net.train()
    t0 = time.perf_counter()
    for step in range(start_step + 1, steps + 1):
        idx = torch.randint(0, n, (cfg.batch_size,), device=dev)
        x1 = x_data[idx]
        x0 = torch.randn_like(x1)
        t = torch.rand(cfg.batch_size, device=dev)
        x_t, u_t = sample_conditional_path(x0, x1, t, cfg.sigma_min)

        loss = torch.nn.functional.mse_loss(net(x_t, t), u_t)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grad_clip:
            torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.grad_clip)
        optim.step()

        item = float(loss.detach())
        ema_loss = item if ema_loss is None else 0.99 * ema_loss + 0.01 * item
        if log_every and step % log_every == 0:
            loss_hist.append((step, ema_loss))
            log.info("step %d  loss_ema %.5f", step, ema_loss)
        if step in want:
            checkpoints[step] = {k: v.detach().cpu().clone()
                                 for k, v in net.state_dict().items()}

    return net, TrainResult(
        checkpoints=checkpoints,
        final_step=steps,
        optimizer_state=optim.state_dict(),
        loss_history=loss_hist,
        wallclock_s=time.perf_counter() - t0,
    )


# ── Sampling (deterministic fixed-step ODE, mirrors CFM.reconstruct) ────────

#: velocity-field evaluations per step, by solver.
EVALS_PER_STEP = {"euler": 1, "midpoint": 2, "rk4": 4}


@torch.no_grad()
def sample_fm(
    net: VelocityField,
    n: int,
    *,
    nfe: int,
    solver: str = "euler",
    device: torch.device | str | None = None,
    seed: int = 0,
    chunk: int = 262_144,
) -> np.ndarray:
    """Generate ``n`` samples by integrating the PF-ODE ``t: 0 -> 1``.

    ``nfe`` is the *number of function evaluations* — the sweep knob.  For
    ``euler`` this equals the step count (the recommended sweep setting, as in
    ScatterPrism); for ``midpoint``/``rk4`` the step count is
    ``nfe / evals_per_step`` and ``nfe`` must be divisible accordingly.

    Deterministic given ``seed`` (which seeds only the base draw ``x0``).

    Returns:
        ``[n, D]`` float64 numpy array of generated samples.
    """
    if solver not in EVALS_PER_STEP:
        raise ValueError(f"unknown solver {solver!r}")
    per = EVALS_PER_STEP[solver]
    if nfe % per:
        raise ValueError(f"nfe={nfe} not divisible by {per} ({solver})")
    n_steps = nfe // per
    if n_steps < 1:
        raise ValueError(f"nfe={nfe} too small for solver {solver!r}")

    dev = pick_device(None if device is None else str(device))
    net = net.to(dev).eval()
    d = net.data_dim
    gen = torch.Generator(device="cpu").manual_seed(seed)
    dt = 1.0 / n_steps

    def v(x: torch.Tensor, t_val: float) -> torch.Tensor:
        t_b = torch.full((x.shape[0],), t_val, device=dev, dtype=x.dtype)
        return net(x, t_b)

    out = np.empty((n, d), dtype=np.float64)
    done = 0
    while done < n:
        m = min(chunk, n - done)
        x = torch.randn(m, d, generator=gen).to(dev)
        for step in range(n_steps):
            t_val = step * dt
            if solver == "euler":
                x = x + dt * v(x, t_val)
            elif solver == "midpoint":
                k1 = v(x, t_val)
                x = x + dt * v(x + 0.5 * dt * k1, t_val + 0.5 * dt)
            else:  # rk4
                k1 = v(x, t_val)
                k2 = v(x + 0.5 * dt * k1, t_val + 0.5 * dt)
                k3 = v(x + 0.5 * dt * k2, t_val + 0.5 * dt)
                k4 = v(x + dt * k3, t_val + dt)
                x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        out[done:done + m] = x.cpu().numpy().astype(np.float64)  # MPS has no f64
        done += m
    return out


# ── Checkpoint I/O ──────────────────────────────────────────────────────────

def save_checkpoint(path, model_state: dict, cfg: FMConfig, step: int,
                    meta: dict | None = None,
                    optimizer_state: dict | None = None) -> None:
    """Save a checkpoint (+ optional optimizer state for M3 continuation)."""
    payload = {
        "model": model_state,
        "config": cfg.as_dict(),
        "step": int(step),
        "meta": meta or {},
    }
    if optimizer_state is not None:
        payload["optimizer"] = optimizer_state
    torch.save(payload, path)


def load_checkpoint(path, device: torch.device | str | None = None
                    ) -> tuple[VelocityField, dict]:
    """Load a checkpoint; returns ``(net_on_device, payload)``."""
    dev = pick_device(None if device is None else str(device))
    payload = torch.load(path, map_location="cpu", weights_only=False)
    cfg = FMConfig(**payload["config"])
    net = cfg.build_net()
    net.load_state_dict(payload["model"])
    return net.to(dev), payload
