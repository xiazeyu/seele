from __future__ import annotations

import contextlib
import gc
import hashlib
import json
import math
import os
import platform
import socket
import subprocess
import time
from collections.abc import Callable, Generator
from pathlib import Path
from typing import Any

import numpy as np
import scipy
import torch

import fm
from measure import EDGES, FIT_DEFAULTS, PROTOCOL, measure, nearest_neighbor, sample_target

TRAIN_DEFAULTS = dict(
    dataset="disc", n_train=500_000, data_seed=0,
    hidden=256, depth=3, time_cond="concat", act="silu", param="v", t_eps=1e-3,
    lr=1e-3, batch=4096, steps=256_000, lr_final_frac=0.0, weight_decay=1e-5, grad_clip=1.0,
    precision="fp32", compile=False, seed=0,
)
MEASURE_DEFAULTS = dict(
    eval_points=16, eval_n_gen=200_000, eval_nfe=512,
    measure_ckpts=["100", "best"], nfe=[512], n_gen=[1_000_000], sample_precision="fp32",
    sample_seed=None, sample_chunk=262_144,
    sample_times=[1.0], nn_band=0.0,
)
CKPT_FRACS = {"025": 0.25, "050": 0.5, "075": 0.75, "100": 1.0}
MEAS_KEYS = ("ckpt", "nfe", "n_gen", "sample_precision", "sample_seed", "sample_chunk", "sample_t", "nn_band")


def split(cfg: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    bad = set(cfg) - set(TRAIN_DEFAULTS) - set(MEASURE_DEFAULTS)
    if bad:
        raise KeyError(f"unknown config keys: {sorted(bad)}")
    train = {k: cfg.get(k, v) for k, v in TRAIN_DEFAULTS.items()}
    meas = {k: cfg.get(k, v) for k, v in MEASURE_DEFAULTS.items()}
    if meas["sample_seed"] is None:
        meas["sample_seed"] = 1000 + train["seed"]
    for k, minimum in dict(n_train=1, hidden=1, depth=1, batch=1, steps=4, data_seed=0, seed=0,
                           eval_points=0, eval_n_gen=1, sample_chunk=1, sample_seed=0).items():
        v = (train | meas)[k]
        if type(v) is not int or v < minimum:
            raise ValueError(f"{k} must be an integer >= {minimum}")
    if meas["eval_points"] > train["steps"]:
        raise ValueError("eval_points cannot exceed steps")
    for k in ("nfe", "n_gen", "measure_ckpts", "sample_times"):
        if not isinstance(meas[k], list) or not meas[k]:
            raise ValueError(f"{k} must be a nonempty list")
    for n in [meas["eval_nfe"], *meas["nfe"]]:
        if type(n) is not int or n < 4 or n % 4:
            raise ValueError("nfe and eval_nfe must be positive multiples of 4")
    if any(type(n) is not int or n <= 0 for n in meas["n_gen"]):
        raise ValueError("n_gen must contain positive integers")
    if any(c not in (*CKPT_FRACS, "best") for c in meas["measure_ckpts"]):
        raise ValueError("unknown checkpoint name")
    if any(type(t) not in (int, float) or not 0 < t <= 1 for t in meas["sample_times"]):
        raise ValueError("sample_times must contain times in (0, 1]")
    for k in ("nfe", "n_gen", "measure_ckpts"):
        meas[k] = list(dict.fromkeys(meas[k]))
    meas["sample_times"] = sorted({float(t) for t in meas["sample_times"]})
    if type(meas["nn_band"]) not in (int, float) or not math.isfinite(meas["nn_band"]) or meas["nn_band"] < 0:
        raise ValueError("nn_band must be finite and nonnegative")
    if meas["nn_band"] > 0 and 1.0 not in meas["sample_times"]:
        raise ValueError("nn_band > 0 requires sample_times to include 1.0")
    for k, values in dict(dataset=EDGES, precision=fm.PRECISIONS, time_cond=fm.TIME_CONDS,
                          act=fm.ACTS, param=fm.PARAMS, sample_precision=fm.PRECISIONS).items():
        if (train | meas)[k] not in values:
            raise ValueError(f"invalid {k}: {(train | meas)[k]!r}")
    for k in ("lr", "lr_final_frac", "weight_decay", "grad_clip", "t_eps"):
        v = train[k]
        if type(v) not in (int, float) or not math.isfinite(v) or v < 0:
            raise ValueError(f"{k} must be finite and nonnegative")
        train[k] = float(v)
    if train["lr"] == 0 or not 0 < train["t_eps"] <= 1 or train["lr_final_frac"] > 1:
        raise ValueError("require lr > 0, 0 < t_eps <= 1 and lr_final_frac <= 1")
    if type(train["compile"]) is not bool:
        raise ValueError("compile must be a boolean")
    if max(train["seed"] + fm.TRAIN_SEED_OFFSET, meas["sample_seed"]) >= 2**63:
        raise ValueError("torch seeds must be smaller than 2**63")
    if meas["eval_points"] and meas["sample_seed"] == 10_000 + train["seed"]:
        raise ValueError("sample_seed must differ from the quick-eval seed when eval_points > 0")
    return train, meas


def run_id(train_cfg: dict[str, Any]) -> str:
    diff = {k: v for k, v in train_cfg.items() if v != TRAIN_DEFAULTS[k]}
    return hashlib.sha1(json.dumps(diff, sort_keys=True).encode()).hexdigest()[:12]


def eval_config(meas: dict[str, Any]) -> dict[str, int]:
    return {k: meas[k] for k in ("eval_points", "eval_n_gen", "eval_nfe")}


def wanted(meas: dict[str, Any]) -> list[dict[str, Any]]:
    return [dict(ckpt=c, nfe=nfe, n_gen=ng, **{k: meas[k] for k in MEAS_KEYS[3:6]},
                 sample_t=t, nn_band=meas["nn_band"] if t == 1.0 else 0.0)
            for c in meas["measure_ckpts"] for nfe in meas["nfe"] for ng in meas["n_gen"]
            for t in meas["sample_times"]]


def missing(result: dict[str, Any] | None, meas: dict[str, Any]) -> list[dict[str, Any]]:
    if result is not None:
        if result.get("schema_version") != 1 or result.get("measure_protocol") != PROTOCOL:
            raise ValueError("incompatible result/measurement protocol; use a separate --runs directory")
        if result["eval_config"] != eval_config(meas):
            raise ValueError("eval_* differs from the stored checkpoint-selection policy; use a separate --runs directory")
    have = [{k: m[k] for k in MEAS_KEYS} for m in (result or {}).get("measurements", [])]
    no_best = result is not None and result["best"]["step"] is None
    return [k for k in wanted(meas) if k not in have and not (no_best and k["ckpt"] == "best")]


def make_data(tc: dict[str, Any]) -> np.ndarray:
    return sample_target(np.random.default_rng(tc["data_seed"]), tc["n_train"], tc["dataset"])


def fm_config(tc: dict[str, Any]) -> fm.Config:
    return fm.Config(lr=tc["lr"], batch_size=tc["batch"], lr_final_frac=tc["lr_final_frac"],
                     weight_decay=tc["weight_decay"], grad_clip=tc["grad_clip"], hidden_dim=tc["hidden"],
                     n_blocks=tc["depth"], time_cond=tc["time_cond"], act=tc["act"], param=tc["param"],
                     t_eps=tc["t_eps"], precision=tc["precision"], compile=tc["compile"])


def git_info() -> dict[str, str | bool | None]:
    def git(*args: str) -> str | None:
        try:
            return subprocess.check_output(["git", *args], cwd=Path(__file__).parent,
                                           stderr=subprocess.DEVNULL, text=True).strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None
    status = git("status", "--porcelain", "--untracked-files=all")
    return dict(git=git("rev-parse", "HEAD"), git_dirty=bool(status) if status is not None else None)


def environment(dev: torch.device) -> dict[str, Any]:
    root = Path(__file__).parent
    sources = [root / n for n in ("fm.py", "measure.py", "train.py", "sweep.py", "collect.py", "pyproject.toml", "uv.lock")]
    return dict(host=platform.node(), platform=platform.platform(), device=str(dev),
                gpu=torch.cuda.get_device_name(dev) if dev.type == "cuda" else None,
                gpu_capability=list(torch.cuda.get_device_capability(dev)) if dev.type == "cuda" else None,
                gpu_uuid=str(torch.cuda.get_device_properties(dev).uuid) if dev.type == "cuda" else None,
                gpu_total_gib=torch.cuda.get_device_properties(dev).total_memory / 2**30 if dev.type == "cuda" else None,
                pid=os.getpid(), slurm_job_id=os.environ.get("SLURM_JOB_ID"),
                workers_per_gpu=int(os.environ["SEELE_WORKERS_PER_GPU"]) if os.environ.get("SEELE_WORKERS_PER_GPU") else None,
                python=platform.python_version(), torch=torch.__version__, cuda=torch.version.cuda,
                cudnn=torch.backends.cudnn.version(), numpy=np.__version__, scipy=scipy.__version__,
                threads=torch.get_num_threads(), deterministic=torch.are_deterministic_algorithms_enabled(),
                source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                **git_info(), time=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))


def write_json(path: Path, value: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=1))
    tmp.replace(path)


def dump(run_dir: Path, result: dict[str, Any]) -> None:
    write_json(run_dir / "result.json", result)


def memory_peaks(dev: torch.device) -> dict[str, float | None]:
    return dict(peak_allocated_gib=torch.cuda.max_memory_allocated(dev) / 2**30 if dev.type == "cuda" else None,
                peak_reserved_gib=torch.cuda.max_memory_reserved(dev) / 2**30 if dev.type == "cuda" else None)


def record_memory(run_dir: Path, dev: torch.device, phase: str) -> dict[str, float | None]:
    peaks = memory_peaks(dev)
    path = run_dir / "memory.json"
    history = json.loads(path.read_text()) if path.exists() else {"scope": "pytorch_process_allocator"}
    previous = history.get(phase, {})
    history[phase] = {k: max(v, previous.get(k) or 0.0) if v is not None else previous.get(k)
                      for k, v in peaks.items()}
    write_json(path, history)
    return peaks


@contextlib.contextmanager
def memory_scope(run_dir: Path, dev: torch.device, phase: str) -> Generator[dict[str, float | None]]:
    gc.collect()
    if dev.type == "cuda":
        with torch.cuda.device(dev):
            torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(dev)
    peaks = {}
    try:
        yield peaks
    finally:
        peaks.update(record_memory(run_dir, dev, phase))


def load_result(run_dir: Path) -> dict[str, Any] | None:
    p = run_dir / "result.json"
    return json.loads(p.read_text()) if p.exists() else None


def save_torch(state: dict[str, Any], path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def save_ckpt(net: fm.VelocityField, tc: dict[str, Any], step: int, path: Path) -> None:
    save_torch(dict(model=net.state_dict(), config=tc, step=step), path)


def load_net(run_dir: Path, ckpt: str, tc: dict[str, Any],
             device: torch.device) -> tuple[fm.VelocityField, int]:
    ck = torch.load(run_dir / f"ckpt_{ckpt}.pt", map_location=device)
    if ck["config"] != tc:
        raise ValueError("checkpoint training configuration mismatch")
    net = fm.build(fm_config(tc)).to(device)
    net.load_state_dict(ck["model"])
    return net.eval(), ck["step"]


def report(run_dir: Path, phase: str, **values: Any) -> None:
    record = dict(time=time.time(), host=socket.gethostname(), pid=os.getpid(), phase=phase, **values)
    path = run_dir / "status.json"
    previous = json.loads(path.read_text()) if path.exists() else {}
    keep = ("step", "total_steps", "loss", "lr", "w_fit", "fit_valid", "eta_train_s",
            "peak_allocated_gib", "peak_reserved_gib", "metric_source", "metric_step", "metric_measurement",
            "nonfinite_frac", "c_shift", "spill_frac")
    write_json(path, {**{k: previous[k] for k in keep if k in previous}, **record})
    with (run_dir / "events.jsonl").open("a") as f:
        f.write(json.dumps(record) + "\n")


def _run(tc: dict[str, Any], meas: dict[str, Any], run_dir: Path, dev: torch.device,
         result: dict[str, Any] | None, log: Callable[[str], None], *,
         log_every: int, checkpoint_every: int, deadline: float | None) -> dict[str, Any]:
    rid = run_id(tc)
    data = None
    if not (result and result.get("trained")):
        (run_dir / "config.json").write_text(json.dumps(dict(run_id=rid, train=tc, measure=meas), indent=1, sort_keys=True))
        steps = tc["steps"]
        ckpt_steps = {round(f * steps): name for name, f in CKPT_FRACS.items()}
        n_ev = meas["eval_points"]
        eval_steps = {max(1, round(i / n_ev * steps)) for i in range(1, n_ev + 1)} if n_ev else set()
        env = environment(dev)
        resume_path = run_dir / "resume.pt"
        saved = torch.load(resume_path, map_location="cpu", weights_only=False) if resume_path.exists() else None
        if saved is not None:
            if saved["train"] != tc or saved["eval_config"] != eval_config(meas):
                raise ValueError("resume configuration/eval policy mismatch")
            if saved["env"]["device"].split(":")[0] != env["device"].split(":")[0]:
                raise ValueError("resume device type differs")
            for key in ("gpu", "torch", "cuda", "numpy", "scipy", "threads", "deterministic"):
                if saved["env"][key] != env[key]:
                    raise ValueError(f"resume environment differs: {key}; keep a run on the same GPU type/software")
            for key in ("fm.py", "train.py", "measure.py"):
                if saved["env"]["source_sha256"][key] != env["source_sha256"][key]:
                    raise ValueError(f"resume source differs: {key}")
        evals = saved["evals"] if saved else []
        best = saved["best"] if saved else {"w_fit": float("inf"), "step": None}
        sessions = [*saved.get("sessions", []), env] if saved else [env]
        initial_env = saved["env"] if saved else env
        prior_train_s = saved["train_s"] if saved else 0.0
        prior_phase_s = saved["phase_s"] if saved else 0.0
        prior_eval_s = saved["eval_s"] if saved else 0.0
        callback_s, eval_s = 0.0, 0.0
        start_step = saved["state"]["step"] if saved else 0

        def progress(values: dict[str, Any]) -> None:
            step = values["step"]
            elapsed = time.perf_counter() - t0
            rate = (step - start_step) / max(elapsed, 1e-9)
            report(run_dir, "train", **values, total_steps=steps, frac=step / steps,
                   steps_per_s=rate, eta_train_s=(steps - step) / max(rate, 1e-9),
                   **memory_peaks(dev))
            log(f"[{rid}] step {step}/{steps} loss={values['loss']:.6g} lr={values['lr']:.3g} "
                f"{rate:.1f} steps/s ETA(train)={(steps-step)/max(rate,1e-9)/3600:.2f}h")

        def checkpoint(state: dict[str, Any]) -> None:
            nonlocal callback_s
            started = time.perf_counter()
            elapsed = started - t0
            record_memory(run_dir, dev, "train")
            save_torch(dict(state=state, train=tc, eval_config=eval_config(meas), evals=evals, best=best,
                            env=initial_env, sessions=sessions, train_s=prior_train_s + elapsed - callback_s,
                            phase_s=prior_phase_s + elapsed, eval_s=prior_eval_s + eval_s), resume_path)
            callback_s += time.perf_counter() - started

        def callback(step: int, net: fm.VelocityField) -> None:
            nonlocal callback_s, eval_s
            if dev.type == "cuda":
                torch.cuda.synchronize(dev)
            started = time.perf_counter()
            if step in ckpt_steps:
                save_ckpt(net, tc, step, run_dir / f"ckpt_{ckpt_steps[step]}.pt")
            if step in eval_steps:
                report(run_dir, "eval", step=step, total_steps=steps, nfe=meas["eval_nfe"])
                eval_started = time.perf_counter()
                m = measure(fm.sample(net, meas["eval_n_gen"], nfe=meas["eval_nfe"],
                                      seed=10_000 + tc["seed"], deadline=deadline),
                            dataset=tc["dataset"], include_profile=True, **FIT_DEFAULTS)
                elapsed = time.perf_counter() - eval_started
                eval_s += elapsed
                evals.append(dict(step=step, frac=step / steps, nfe=meas["eval_nfe"], n_gen=meas["eval_n_gen"],
                                  train_elapsed_s=round(prior_train_s + started - t0 - callback_s, 4),
                                  phase_elapsed_s=round(prior_phase_s + started - t0, 4),
                                  sample_precision="fp32", sample_seed=10_000 + tc["seed"], sample_chunk=262_144,
                                  **m, meas_s=round(elapsed, 4)))
                log(f"[{rid}] step {step:>8} ({step / steps:4.0%})  w={m['w_fit']:.5f}")
                if m["fit_valid"] and m["w_fit"] < best["w_fit"]:
                    best.update(w_fit=m["w_fit"], step=step)
                    save_ckpt(net, tc, step, run_dir / "ckpt_best.pt")
                report(run_dir, "eval_done", **evals[-1], total_steps=steps,
                       metric_source="quick_eval", metric_step=step, metric_measurement=None, **memory_peaks(dev))
            callback_s += time.perf_counter() - started

        data = make_data(tc)
        report(run_dir, "train", step=start_step, total_steps=steps, resumed=saved is not None)
        log(f"[{rid}] train {json.dumps({k: v for k, v in tc.items() if v != TRAIN_DEFAULTS[k]})} on {dev}")
        with memory_scope(run_dir, dev, "train"):
            t0 = time.perf_counter()
            net, curve = fm.train(fm_config(tc), data, steps, seed=tc["seed"], device=dev,
                                  callback=callback, callback_steps=ckpt_steps.keys() | eval_steps,
                                  log_every=log_every, progress=progress, resume=saved["state"] if saved else None,
                                  checkpoint=checkpoint, checkpoint_every=checkpoint_every, deadline=deadline)
            if dev.type == "cuda":
                torch.cuda.synchronize(dev)
            wall = time.perf_counter() - t0
        result = dict(
            schema_version=1, run_id=rid, train=tc, trained=True, env=initial_env, sessions=sessions,
            eval_config=eval_config(meas), measure_protocol=PROTOCOL,
            seeds=dict(data=tc["data_seed"], init=tc["seed"], train=tc["seed"] + fm.TRAIN_SEED_OFFSET,
                       eval=10_000 + tc["seed"]),
            n_parameters=sum(p.numel() for p in net.parameters()),
            timing=dict(train_s=round(prior_train_s + wall - callback_s, 4), phase_s=round(prior_phase_s + wall, 4),
                        ms_per_step=round(1e3 * (prior_train_s + wall - callback_s) / steps, 4),
                        eval_s=round(prior_eval_s + eval_s, 4),
                        checkpoint_s=round(prior_phase_s - prior_train_s - prior_eval_s + callback_s - eval_s, 4)),
            loss_curve=curve, evals=evals,
            best=dict(step=best["step"], w_fit=None if best["step"] is None else best["w_fit"]),
            measurements=[],
            memory=json.loads((run_dir / "memory.json").read_text()),
        )
        dump(run_dir, result)
        resume_path.unlink(missing_ok=True)
        del net

    env = environment(dev)
    for key in missing(result, meas):
        report(run_dir, "measure", **key, total_steps=tc["steps"])
        with memory_scope(run_dir, dev, "measure") as peaks:
            net, step = load_net(run_dir, key["ckpt"], tc, dev)
            if dev.type == "cuda":
                torch.cuda.synchronize(dev)
            t0 = time.perf_counter()
            t = key["sample_t"]
            pts = fm.sample(net, key["n_gen"], nfe=key["nfe"], seed=key["sample_seed"],
                            chunk=key["sample_chunk"], precision=key["sample_precision"], t_stop=t, deadline=deadline) / t
            sample_s = time.perf_counter() - t0
            m = measure(pts, dataset=tc["dataset"], include_profile=True, **FIT_DEFAULTS)
            reference_seed = key["sample_seed"] + PROTOCOL["reference_seed_offset"]
            if t < 1.0:
                rng = np.random.default_rng(reference_seed)
                ref = sample_target(rng, key["n_gen"], tc["dataset"])
                ref += rng.normal(size=ref.shape) * ((1.0 - t) / t)
                m["reference"] = measure(ref, dataset=tc["dataset"], include_profile=True, **FIT_DEFAULTS)
            if key["nn_band"] > 0:
                if data is None:
                    data = make_data(tc)
                m.update(nearest_neighbor(pts, data, tc["dataset"], key["nn_band"], reference_seed))
            meas_s = time.perf_counter() - t0
            del net
        result["memory"] = json.loads((run_dir / "memory.json").read_text())
        result["measurements"].append(dict(**key, step=step, **m, env=env, meas_s=round(meas_s, 4),
                                           sample_s=round(sample_s, 4), analysis_s=round(meas_s - sample_s, 4),
                                           **{f"measure_{k}": v for k, v in peaks.items()}))
        dump(run_dir, result)
        report(run_dir, "measure_done", step=step, total_steps=tc["steps"], eta_train_s=0,
               metric_source="measurement", metric_step=step, metric_measurement=key,
               **{k: v for k, v in m.items() if k not in ("profile", "reference")}, **peaks)
        log(f"[{rid}] ckpt={key['ckpt']} t={t:g} nfe={key['nfe']} n_gen={key['n_gen']} seed={key['sample_seed']} w={m['w_fit']:.5f}")
    return result


def run(cfg: dict[str, Any], run_dir: Path, device: torch.device | None = None,
        log: Callable[[str], None] = print, sweep: str = "", *,
        log_every: int = 1000, checkpoint_every: int = 10_000,
        deadline: float | None = None) -> dict[str, Any] | None:
    tc, meas = split(cfg)
    dev = device or fm.pick_device()
    run_dir.mkdir(parents=True, exist_ok=True)
    fd = None
    try:
        try:
            fd = os.open(run_dir / "LOCK", os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            raise BlockingIOError(f"locked: {run_dir}") from None
        with os.fdopen(fd, "w") as f:
            f.write(f"{socket.gethostname()} {os.getpid()} {time.strftime('%Y-%m-%dT%H:%M:%S')}\n")
        result = load_result(run_dir)
        if result is not None and result["train"] != tc:
            raise ValueError("run directory contains a different training configuration")
        todo = missing(result, meas)
        (run_dir / "FAILED").unlink(missing_ok=True)
        if sweep:
            p = run_dir / "sweeps.txt"
            names = p.read_text().split() if p.exists() else []
            if sweep not in names:
                p.write_text("\n".join(names + [sweep]) + "\n")
        if result is not None and result.get("trained") and not todo:
            result = None
        else:
            result = _run(tc, meas, run_dir, dev, result, log,
                          log_every=log_every, checkpoint_every=checkpoint_every, deadline=deadline)
        report(run_dir, "done", step=tc["steps"], total_steps=tc["steps"])
        (run_dir / "FAILED").unlink(missing_ok=True)
        return result
    except TimeoutError as e:
        if fd is not None:
            report(run_dir, "paused", reason=str(e))
        raise
    except Exception as e:
        if fd is not None:
            (run_dir / "FAILED").write_text(repr(e))
            report(run_dir, "failed", error=repr(e))
        raise
    finally:
        if fd is not None:
            (run_dir / "LOCK").unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit("use sweep.py worker/run with a TOML configuration")
