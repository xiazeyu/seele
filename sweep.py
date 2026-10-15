from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import signal
import socket
import subprocess
import sys
import time
import tomllib
from pathlib import Path
from typing import Any

import train


def expand(path: Path) -> tuple[str, list[dict[str, Any]]]:
    doc = tomllib.loads(path.read_text())
    bad = set(doc) - {"name", "base", "measure", "grid"}
    if bad:
        raise ValueError(f"unknown sweep keys in {path}: {sorted(bad)}")
    base = {**doc.get("base", {}), **doc.get("measure", {})}
    grids = doc.get("grid", [{}])
    if not isinstance(grids, list) or not all(isinstance(g, dict) for g in grids):
        raise ValueError(f"grid must be an array of tables [[grid]] in {path}")
    cfgs, seen = [], set()
    for grid in grids:
        grid = dict(grid)
        zipped = grid.pop("zip", [])
        axes = {k: (v if isinstance(v, list) and not isinstance(train.MEASURE_DEFAULTS.get(k), list) else [v])
                for k, v in grid.items() if k not in zipped}
        if zipped:
            axes[tuple(zipped)] = list(zip(*(grid[k] for k in zipped), strict=True))
        for combo in itertools.product(*axes.values()):
            item = {}
            for k, v in zip(axes, combo):
                item.update(dict(zip(k, v)) if isinstance(k, tuple) else {k: v})
            cfg = {**base, **item}
            tc, meas = train.split(cfg)
            key = (train.run_id(tc), json.dumps(meas, sort_keys=True))
            if key not in seen:
                seen.add(key)
                cfgs.append(cfg)
    if not cfgs:
        raise ValueError(f"empty sweep: {path}")
    return doc.get("name", path.stem), cfgs


def load(paths: list[str]) -> list[tuple[str, dict[str, Any]]]:
    out, policies = [], {}
    for p in paths:
        name, cfgs = expand(Path(p))
        for cfg in cfgs:
            tc, meas = train.split(cfg)
            rid, policy = train.run_id(tc), train.eval_config(meas)
            if rid in policies and policies[rid] != policy:
                raise ValueError(f"conflicting eval_* policies for {rid}")
            policies[rid] = policy
            out.append((name, cfg))
    return out


def run_dir(runs: Path, cfg: dict[str, Any]) -> Path:
    return runs / train.run_id(train.split(cfg)[0])


def select_gpus(selection: str) -> list[str]:
    if not selection:
        return [""]
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid,name", "--format=csv,noheader"], text=True, timeout=10)
    devices = [tuple(v.strip() for v in row) for row in csv.reader(output.splitlines())]
    aliases = {key: (uuid, name) for index, uuid, name in devices for key in (index, uuid)}
    if selection == "auto":
        probe = ("import json, torch; print(json.dumps(["
                 "'GPU-' + str(torch.cuda.get_device_properties(i).uuid) "
                 "for i in range(torch.cuda.device_count())]))")
        requested = json.loads(subprocess.check_output(
            [sys.executable, "-B", "-c", probe], text=True, timeout=30,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}))
    else:
        requested = selection.split(",")
    selected = {}
    for token in requested:
        token = token.strip()
        if token not in aliases:
            raise ValueError(f"GPU {token!r} not found in nvidia-smi; use indices or full GPU UUIDs")
        uuid, name = aliases[token]
        if name == "NVIDIA DGX Display":
            print(f"skipping GPU {token}: {name}", flush=True)
        else:
            selected[uuid] = name
    if not selected:
        raise ValueError("no compute GPUs selected")
    for uuid, name in selected.items():
        print(f"GPU {uuid}: {name}", flush=True)
    return list(selected)


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def cmd_plan(args: argparse.Namespace) -> int:
    groups = {}
    rows = []
    for name, cfg in load(args.sweeps):
        tc, meas = train.split(cfg)
        rid = train.run_id(tc)
        group = groups.setdefault(rid, dict(train=tc, measures=[], names=[]))
        group["measures"].append(meas)
        if name not in group["names"]:
            group["names"].append(name)
    totals = dict(done=0, running=0, pending=0, failed=0)
    train_steps, measurements, sample_nfe = 0, 0, 0
    for rid, group in groups.items():
        d = Path(args.runs) / rid
        result = train.load_result(d)
        tc = group["train"]
        if result is not None and result["train"] != tc:
            raise ValueError(f"training configuration mismatch: {rid}")
        todo = {tuple(m[k] for k in train.MEAS_KEYS)
                for meas in group["measures"] for m in train.missing(result, meas)}
        trained = bool(result and result.get("trained"))
        state = "done" if trained and not todo else "running" if (d / "LOCK").exists() else "pending"
        totals[state] += 1
        failed = (d / "FAILED").exists()
        rows.append(dict(run_id=rid, state=state, failed=failed, sweeps=";".join(group["names"]),
                         **tc, measurements_pending=len(todo), measures=json.dumps(group["measures"])))
        totals["failed"] += failed
        if not trained:
            status = json.loads((d / "status.json").read_text()) if (d / "status.json").exists() else {}
            train_steps += max(0, tc["steps"] - status.get("step", 0))
            meas = group["measures"][0]
            sample_nfe += meas["eval_points"] * meas["eval_n_gen"] * meas["eval_nfe"]
        measurements += len(todo)
        sample_nfe += sum(nfe * ng for _, nfe, ng, *_ in todo)
        if args.verbose or state != "done" or failed:
            diff = {k: v for k, v in tc.items() if v != train.TRAIN_DEFAULTS[k]}
            print(f"{state:8} {rid} train_steps={0 if trained else tc['steps']} measurements={len(todo)}  {','.join(group['names'])}: {diff}{'  FAILED' if failed else ''}")
    print(f"\n{totals} unique runs; remaining_train_steps={train_steps} remaining_measurements={measurements} remaining_sample_nfe={sample_nfe}")
    if args.out:
        from collect import write_csv
        write_csv(args.out, rows)
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    totals: dict[str, int] = {}
    for d in sorted(Path(args.runs).glob("*")):
        if not d.is_dir():
            continue
        p = d / "status.json"
        status = json.loads(p.read_text()) if p.exists() else {}
        phase = status.get("phase", "unknown")
        if (d / "FAILED").exists():
            phase = "failed"
        lock = d / "LOCK"
        if lock.exists():
            try:
                host, pid, _ = lock.read_text().split()
                if host == socket.gethostname() and not _pid_alive(int(pid)):
                    phase = "stale_lock"
            except (FileNotFoundError, ValueError):
                phase = "invalid_lock"
        elif phase not in ("done", "failed", "paused"):
            phase = "stopped"
        totals[phase] = totals.get(phase, 0) + 1
        age = time.time() - status.get("time", time.time())
        if phase != "done" or args.verbose:
            print(f"{d.name} {phase:12} step={status.get('step', '?')}/{status.get('total_steps', '?')} "
                  f"age={age:.0f}s loss={status.get('loss', '-')} w={status.get('w_fit', '-')} "
                  f"fit_valid={status.get('fit_valid', '-')} ETA_train_s={status.get('eta_train_s', '-')}")
    print(totals)
    return 0


def cmd_worker(args: argparse.Namespace) -> int:
    jobs = load(args.sweeps)
    runs, completed, failed = Path(args.runs), set(), 0
    next_notice = 0.0
    deadline = time.monotonic() + args.time_budget if args.time_budget else None
    while jobs:
        busy = []
        for name, cfg in jobs:
            if deadline is not None and time.monotonic() >= deadline:
                print("worker time budget reached; rerun the same command to continue", flush=True)
                return int(failed > 0)
            d = run_dir(runs, cfg)
            try:
                if train.run(cfg, d, sweep=name, log=lambda s: print(s, flush=True),
                             log_every=args.log_every, checkpoint_every=args.checkpoint_every,
                             deadline=deadline) is not None:
                    completed.add(d.name)
            except BlockingIOError:
                busy.append((name, cfg))
            except TimeoutError as e:
                print(f"[{d.name}] PAUSED: {e}; rerun to continue", flush=True)
                return int(failed > 0)
            except Exception as e:
                failed += 1
                print(f"[{d.name}] FAILED: {e!r}", flush=True)
        jobs = busy
        if jobs:
            if time.monotonic() >= next_notice:
                count = len({run_dir(runs, cfg) for _, cfg in jobs})
                print(f"waiting for {count} locked runs; inspect with plan/unlock if the owner has stopped", flush=True)
                next_notice = time.monotonic() + 60
            time.sleep(1)
    print(f"worker done, {len(completed)} runs, {failed} failed", flush=True)
    return int(failed > 0)


def cmd_run(args: argparse.Namespace) -> int:
    procs, reaped = [], set()
    try:
        load(args.sweeps)
        gpus = select_gpus(args.gpus)
        for g in gpus:
            for _ in range(args.per_gpu):
                env = {**os.environ, **({"CUDA_VISIBLE_DEVICES": g} if g else {})}
                cmd = [sys.executable, "-B", __file__, "worker", *args.sweeps, "--runs", args.runs]
                cmd += ["--log-every", str(args.log_every), "--checkpoint-every", str(args.checkpoint_every),
                        "--time-budget", str(args.time_budget)]
                procs.append(subprocess.Popen(cmd, env=env, start_new_session=True))
        while True:
            rc = [p.poll() for p in procs]
            failed = any(code not in (None, 0) for code in rc)
            for p, code in zip(procs, rc):
                if code in (None, 0) or p.pid in reaped:
                    continue
                reaped.add(p.pid)
                for lock in Path(args.runs).glob("*/LOCK"):
                    try:
                        host, pid, _ = lock.read_text().split()
                    except (FileNotFoundError, ValueError):
                        continue
                    if host == socket.gethostname() and pid == str(p.pid):
                        print("removing lock held by exited worker:", lock, flush=True)
                        lock.unlink(missing_ok=True)
            if all(code is not None for code in rc):
                print("workers exited", rc, flush=True)
                return int(failed)
            time.sleep(0.2)
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        for p in procs:
            if p.poll() is None:
                p.terminate()
        for p in procs:
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()


def cmd_unlock(args: argparse.Namespace) -> int:
    unresolved = 0
    for lock in Path(args.runs).glob("*/LOCK"):
        try:
            host, pid, _ = lock.read_text().split()
            pid = int(pid)
            if not 0 < pid < 2**31:
                raise ValueError("invalid pid")
        except FileNotFoundError:
            continue
        except ValueError:
            print("malformed lock; verify the owner is stopped before using --all:", lock)
            if args.all:
                lock.unlink(missing_ok=True)
            else:
                unresolved += 1
            continue
        alive = _pid_alive(pid) if host == socket.gethostname() else None
        if args.all or alive is False:
            print("removing", lock, "(stale)" if alive is False else "")
            lock.unlink(missing_ok=True)
        else:
            print("keeping", lock, "held by", host, pid)
    return int(unresolved > 0)


def main() -> None:
    ap = argparse.ArgumentParser(description="sweep files -> content-addressed runs/<id>/")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in [("plan", cmd_plan), ("worker", cmd_worker), ("run", cmd_run)]:
        p = sub.add_parser(name)
        p.add_argument("sweeps", nargs="+")
        p.add_argument("--runs", default="runs")
        p.set_defaults(fn=fn)
    sub.choices["plan"].add_argument("-v", "--verbose", action="store_true")
    sub.choices["plan"].add_argument("--out", help="export all planned runs, including missing/failed runs, as csv")
    sub.choices["run"].add_argument("--gpus", default="", help="auto, or comma-separated GPU indices/full UUIDs; filters DGX Display")
    sub.choices["run"].add_argument("--per-gpu", type=int, default=1)
    for name in ("run", "worker"):
        sub.choices[name].add_argument("--log-every", type=int, default=1000)
        sub.choices[name].add_argument("--checkpoint-every", type=int, default=10_000)
        sub.choices[name].add_argument("--time-budget", type=float, default=0,
                                       help="seconds per worker; 0 means unlimited; rerun to resume")
    p = sub.add_parser("status")
    p.add_argument("--runs", default="runs")
    p.add_argument("-v", "--verbose", action="store_true")
    p.set_defaults(fn=cmd_status)
    p = sub.add_parser("unlock")
    p.add_argument("--runs", default="runs")
    p.add_argument("--all", action="store_true")
    p.set_defaults(fn=cmd_unlock)
    args = ap.parse_args()
    if args.cmd == "run" and args.per_gpu < 1:
        ap.error("--per-gpu must be positive")
    if args.cmd in ("run", "worker"):
        if args.log_every < 1 or args.checkpoint_every < 1 or not 0 <= args.time_budget < float("inf"):
            ap.error("log/checkpoint intervals must be positive and time-budget finite and nonnegative")
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, lambda number, _: sys.exit(128 + number))
    sys.exit(args.fn(args))


if __name__ == "__main__":
    main()
