from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

from measure import PROTOCOL


def write_csv(path: str, rows: list[dict[str, Any]]) -> None:
    cols = list(dict.fromkeys(k for row in rows for k in row)) or ["run_id", "step", "w_fit"]
    f = sys.stdout if path == "-" else open(path, "w", newline="")
    try:
        w = csv.DictWriter(f, cols)
        w.writeheader()
        w.writerows({k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v
                    for k, v in row.items()} for row in rows)
    finally:
        if f is not sys.stdout:
            f.close()


def profile_rows(base: dict[str, Any], profile: dict[str, Any], kind: str) -> list[dict[str, Any]]:
    fit = profile.get("fit")
    fit_values = dict(zip(("lo", "hi", "center", "width"), fit)) if fit is not None else {}
    return [dict(**base, profile_kind=kind, bin_index=i, left=profile["edges"][i],
                 right=profile["edges"][i + 1], s=s, count=profile["counts"][i],
                 density=profile["density"][i], sigma=profile["sigma"][i],
                 n_total=profile["n_total"], **fit_values)
            for i, s in enumerate(profile.get("s", []))]


def main() -> None:
    ap = argparse.ArgumentParser(description="runs/*/result.json -> flat csv (one row per measurement)")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="-")
    ap.add_argument("--evals", help="also write the in-training quick evals to this csv")
    ap.add_argument("--losses", help="training loss curves from completed training phases")
    ap.add_argument("--profiles", help="one row per histogram bin, including quick evals and references")
    args = ap.parse_args()

    rows, evals, losses, profiles = [], [], [], []
    for p in sorted(Path(args.runs).glob("*/result.json")):
        r = json.loads(p.read_text())
        if r.get("schema_version") != 1 or r.get("measure_protocol") != PROTOCOL:
            raise ValueError(f"incompatible result/measurement protocol: {p}")
        sweeps = (p.parent / "sweeps.txt").read_text().split() if (p.parent / "sweeps.txt").exists() else []
        base = dict(run_id=r["run_id"], sweeps=";".join(sweeps), **r["train"], **r["timing"],
                    best_step=r["best"]["step"], n_parameters=r["n_parameters"], **r["eval_config"],
                    **{f"fit_{k}": v for k, v in r["measure_protocol"].items()},
                    **{f"train_{k}": v for k, v in r["env"].items()})
        for m in r["measurements"]:
            row = {**base, **{k: v for k, v in m.items() if k not in ("env", "profile", "reference")},
                   **{f"ref_{k}": v for k, v in m.get("reference", {}).items() if k != "profile"},
                   **{f"meas_{k}": v for k, v in m["env"].items()}}
            rows.append({k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v for k, v in row.items()})
            if args.profiles:
                key = {k: v for k, v in row.items() if k not in ("w_fit", "c_fit")}
                profiles += profile_rows(key, m.get("profile", {}), "measurement")
                profiles += profile_rows(key, m.get("reference", {}).get("profile", {}), "reference")
        for e in r["evals"]:
            row = {**base, **{k: v for k, v in e.items() if k != "profile"}}
            evals.append(row)
            if args.profiles:
                profiles += profile_rows(row, e.get("profile", {}), "quick_eval")
        losses += [dict(**base, step=step, loss=loss) for step, loss in r["loss_curve"]]

    for path, data in [(args.out, rows), (args.evals, evals), (args.losses, losses), (args.profiles, profiles)]:
        if path is None:
            continue
        write_csv(path, data)
    print(f"{len(rows)} measurements from {len(set(r['run_id'] for r in rows))} runs", file=sys.stderr)


if __name__ == "__main__":
    main()
