from __future__ import annotations

import contextlib
import math
import time
from collections.abc import Callable, Generator, Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

ACTS = {"silu": nn.SiLU, "relu": nn.ReLU, "gelu": nn.GELU}
TIME_CONDS = ("concat", "film", "adaln")
PARAMS = ("v", "x1", "x0")
PRECISIONS = ("fp32", "tf32", "bf16", "fp16") # autocast / amp
TRAIN_SEED_OFFSET = 20_000


@dataclass
class Config:
    lr: float = 1e-3
    batch_size: int = 4096
    lr_final_frac: float = 0.0
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    hidden_dim: int = 256
    n_blocks: int = 3
    time_cond: str = "concat"
    act: str = "silu"
    param: str = "v"
    t_eps: float = 1e-3
    precision: str = "fp32"
    compile: bool = False


def set_tf32(on: bool) -> None:
    torch.backends.cuda.matmul.allow_tf32 = on
    torch.backends.cudnn.allow_tf32 = on


@contextlib.contextmanager
def precision_ctx(precision: str, device: torch.device) -> Generator[None, None, None]:
    assert precision in PRECISIONS, precision
    previous = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    set_tf32(precision == "tf32")
    amp = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(precision)
    try:
        with torch.autocast(device.type, dtype=amp, enabled=amp is not None):
            yield
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = previous


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class FourierEmbedding(nn.Module):
    def __init__(self, dim: int = 64, max_freq: float = 64.0) -> None:
        super().__init__()
        self.register_buffer("freqs", torch.exp(torch.linspace(0.0, math.log(max_freq), dim // 2)))
        self.proj = nn.Linear(dim, dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        ang = 2.0 * math.pi * t.view(-1, 1) * self.freqs
        return self.proj(torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1))


class ResBlock(nn.Module):
    def __init__(self, dim: int, cond_dim: int, act: str) -> None:
        super().__init__()
        self.lin1 = nn.Linear(dim, dim)
        self.lin2 = nn.Linear(dim, dim)
        self.act = ACTS[act]()

    def forward(self, h: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        return self.act(h + self.lin2(self.act(self.lin1(h))))


class FiLMBlock(nn.Module):
    def __init__(self, dim: int, cond_dim: int, act: str) -> None:
        super().__init__()
        self.lin1 = nn.Linear(dim, dim)
        self.lin2 = nn.Linear(dim, dim)
        self.film = nn.Linear(cond_dim, 2 * dim)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)
        self.act = ACTS[act]()

    def forward(self, h: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        g, b = self.film(c).chunk(2, dim=-1)
        return self.act(h + self.lin2(self.act(self.lin1(h) * (1.0 + g) + b)))


class AdaLNBlock(nn.Module):
    def __init__(self, dim: int, cond_dim: int, act: str) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.lin1 = nn.Linear(dim, dim)
        self.lin2 = nn.Linear(dim, dim)
        self.mod = nn.Linear(cond_dim, 3 * dim)
        nn.init.zeros_(self.mod.weight)
        nn.init.zeros_(self.mod.bias)
        self.act = ACTS[act]()

    def forward(self, h: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        g, b, a = self.mod(c).chunk(3, dim=-1)
        u = self.norm(h) * (1.0 + g) + b
        return h + a * self.lin2(self.act(self.lin1(u)))


BLOCKS = {"concat": ResBlock, "film": FiLMBlock, "adaln": AdaLNBlock}


class VelocityField(nn.Module):
    def __init__(self, data_dim: int = 2, hidden_dim: int = 256, n_blocks: int = 3,
                 time_embed_dim: int = 64, time_cond: str = "concat", act: str = "silu",
                 param: str = "v", t_eps: float = 1e-3) -> None:
        super().__init__()
        assert time_cond in TIME_CONDS and param in PARAMS and act in ACTS
        self.data_dim, self.time_cond, self.param, self.t_eps = data_dim, time_cond, param, t_eps
        self.time_embed = FourierEmbedding(time_embed_dim)
        in_dim = data_dim + time_embed_dim if time_cond == "concat" else data_dim
        self.in_proj = nn.Linear(in_dim, hidden_dim)
        self.blocks = nn.ModuleList(BLOCKS[time_cond](hidden_dim, time_embed_dim, act) for _ in range(n_blocks))
        self.out_proj = nn.Linear(hidden_dim, data_dim)
        self.act = ACTS[act]()

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        c = self.time_embed(t)
        h = self.act(self.in_proj(torch.cat([x, c], dim=-1) if self.time_cond == "concat" else x))
        for blk in self.blocks:
            h = blk(h, c)
        return self.out_proj(h)

    def target(self, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        return {"v": lambda: x1 - x0, "x1": lambda: x1, "x0": lambda: x0}[self.param]()

    def velocity(self, x: torch.Tensor, t: torch.Tensor, step_size: float = 0.0) -> torch.Tensor:
        out = self(x, t)
        if self.param == "v":
            return out
        tt = t.view(-1, 1)
        eps = max(self.t_eps, step_size)
        if self.param == "x1":
            return (out - x) / (1.0 - tt).clamp_min(eps)
        return (x - out) / tt.clamp_min(eps)


def build(cfg: Config, data_dim: int = 2) -> VelocityField:
    return VelocityField(data_dim, cfg.hidden_dim, cfg.n_blocks, time_cond=cfg.time_cond,
                         act=cfg.act, param=cfg.param, t_eps=cfg.t_eps)


def train(cfg: Config, data: np.ndarray, steps: int, seed: int = 0,
          device: torch.device | None = None,
          callback: Callable[[int, VelocityField], None] | None = None,
          callback_steps: Iterable[int] = (),
          log_every: int = 1000,
          progress: Callable[[dict[str, Any]], None] | None = None,
          resume: dict[str, Any] | None = None,
          checkpoint: Callable[[dict[str, Any]], None] | None = None,
          checkpoint_every: int = 10_000,
          deadline: float | None = None) -> tuple[VelocityField, list[tuple[int, float]]]:
    if log_every < 1 or checkpoint_every < 1:
        raise ValueError("log/checkpoint intervals must be positive")
    dev = device or pick_device()
    torch.manual_seed(seed)
    x_data = torch.as_tensor(np.asarray(data, dtype=np.float32), device=dev)
    net = build(cfg, x_data.shape[1]).to(dev)
    fwd = torch.compile(net) if cfg.compile else net
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=(cfg.precision == "fp16" and dev.type == "cuda"))
    cb_steps = set(callback_steps)
    curve: list[tuple[int, float]] = []
    loss_acc = torch.zeros((), device=dev)
    loss_count = 0
    start_step = 0

    torch.manual_seed(seed + TRAIN_SEED_OFFSET)
    if resume is not None:
        net.load_state_dict(resume["model"])
        opt.load_state_dict(resume["optimizer"])
        scaler.load_state_dict(resume["scaler"])
        start_step = resume["step"]
        curve = list(resume["curve"])
        loss_acc.copy_(resume["loss_acc"].to(dev))
        loss_count = resume["loss_count"]
        torch.set_rng_state(resume["rng_cpu"].cpu())
        if dev.type == "cuda":
            torch.cuda.set_rng_state(resume["rng_device"].cpu(), dev)
        elif dev.type == "mps":
            torch.mps.set_rng_state(resume["rng_device"].cpu())

    def save_state(step: int) -> None:
        if checkpoint is not None:
            rng_device = (torch.cuda.get_rng_state(dev) if dev.type == "cuda" else
                          torch.mps.get_rng_state() if dev.type == "mps" else None)
            checkpoint(dict(model=net.state_dict(), optimizer=opt.state_dict(), scaler=scaler.state_dict(),
                            step=step, curve=curve, loss_acc=loss_acc, loss_count=loss_count,
                            rng_cpu=torch.get_rng_state(), rng_device=rng_device))

    net.train()
    for step in range(start_step + 1, steps + 1):
        if deadline is not None and time.monotonic() >= deadline:
            save_state(step - 1)
            raise TimeoutError("training time budget reached; resume saved")
        x1 = x_data[torch.randint(0, x_data.shape[0], (cfg.batch_size,), device=dev)]
        x0 = torch.randn(x1.shape, device=dev)
        t = torch.rand(x1.shape[0], device=dev)
        x_t = (1.0 - t.view(-1, 1)) * x0 + t.view(-1, 1) * x1
        with precision_ctx(cfg.precision, dev):
            loss = (fwd(x_t, t) - net.target(x0, x1)).pow(2).mean(dim=1).mean()
            with torch.autocast(dev.type, enabled=False):
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                if cfg.grad_clip:
                    torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.grad_clip)
                scaler.step(opt)
                scaler.update()
        if cfg.lr_final_frac < 1.0:
            f = cfg.lr_final_frac
            for pg in opt.param_groups:
                pg["lr"] = cfg.lr * (f + (1.0 - f) * 0.5 * (1.0 + math.cos(math.pi * step / steps)))

        loss_acc += loss.detach()
        loss_count += 1
        if step % log_every == 0 or step == steps:
            loss_mean = float(loss_acc.item()) / loss_count
            if not math.isfinite(loss_mean):
                raise FloatingPointError(f"nonfinite training loss at step {step}: {loss_mean}")
            curve.append((step, loss_mean))
            loss_acc.zero_()
            loss_count = 0
            if progress is not None:
                progress(dict(step=step, loss=loss_mean, lr=opt.param_groups[0]["lr"],
                              grad_scale=scaler.get_scale()))
        if step in cb_steps and callback is not None:
            net.eval()
            callback(step, net)
            net.train()
        if step % checkpoint_every == 0 or step == steps:
            save_state(step)
    return net.eval(), curve


@torch.no_grad()
def sample(net: VelocityField, n: int, nfe: int = 512, seed: int = 0,
           chunk: int = 262_144, precision: str = "fp32", t_stop: float = 1.0,
           deadline: float | None = None) -> np.ndarray:
    if type(nfe) is not int or nfe < 4 or nfe % 4:
        raise ValueError("nfe must be a positive multiple of 4")
    if type(n) is not int or n <= 0 or type(chunk) is not int or chunk <= 0:
        raise ValueError("n and chunk must be positive integers")
    if not 0 < t_stop <= 1:
        raise ValueError("t_stop must be in (0, 1]")
    dev = next(net.parameters()).device
    n_steps = nfe // 4
    dt = t_stop / n_steps
    gen = torch.Generator(device="cpu").manual_seed(seed)

    def v(x: torch.Tensor, t: float) -> torch.Tensor:
        return net.velocity(x, torch.full((x.shape[0],), t, device=dev, dtype=x.dtype), step_size=dt)

    out = np.empty((n, net.data_dim), dtype=np.float64)
    with precision_ctx(precision, dev):
        for lo in range(0, n, chunk):
            m = min(chunk, n - lo)
            x = torch.randn(m, net.data_dim, generator=gen).to(dev)
            for k in range(n_steps):
                if deadline is not None and time.monotonic() >= deadline:
                    raise TimeoutError("sampling time budget reached; repeat current measurement on resume")
                t = k * dt
                k1 = v(x, t)
                k2 = v(x + 0.5 * dt * k1, t + 0.5 * dt)
                k3 = v(x + 0.5 * dt * k2, t + 0.5 * dt)
                k4 = v(x + dt * k3, t + dt)
                x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            out[lo:lo + m] = x.float().cpu().numpy()
    return out
