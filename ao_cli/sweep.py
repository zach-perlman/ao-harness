"""Eval-time hyperparameter sweep for the trained AO (no retrain required).

Gears-level overview
--------------------
Two inference-time knobs change AO behaviour without touching the weights:

  • steering_coefficient — injection strength (AO_EVAL_STEERING_COEFFICIENT).
      Paper trains at 1.0, then sweeps eval ~1–2× (grounding/hallucination knob).
  • n_positions — how many of the last activation positions are fed to the AO
      (the `--n-positions` window; paper sweeps 1–50).

This driver runs *independent 1-D sweeps* around the config's current values
(not the full grid — cheaper, and each axis is interpretable on its own):

      steering    ∈ STEER at (n_positions=base_np, temperature=base_temp)
      n_positions ∈ NPOS  at (steering=base_coef,   temperature=base_temp)
      temperature ∈ TEMP  at (steering=base_coef,   n_positions=base_np)

A third knob, decoding temperature (AO_EVAL_TEMPERATURE; <=0 = greedy), is swept
too — deterministic decoding can sharpen the judged tasks the AO is weak on.

Each setting is a normal `ao_cli.evaluate` run into its own dir
(`aobench_results/sweep_s{coef}_np{np}/`), capped at --sample-limit items/task on
a small task subset so a point finishes in minutes, not hours. We deliberately
reuse evaluate.py (not a bespoke loop) so judge wiring, env, and LoRA selection
are identical to the real eval. After all points finish we read every run back
through the dashboard's own metric accessors and print one comparison table, so
the sweep numbers mean exactly what the dashboard's numbers mean.

Because a full `make eval` may already own the GPU, the driver first waits for
any other run_all process to exit (and for enough free VRAM) before starting.
"""

from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path

from . import artifacts_dir, load_config, model_slug
from .dashboard import TASKS, calibration, load_records, load_summaries, metric_value

# Tasks that respond to these knobs while staying cheap: two judged grounding
# tasks (the AO's weak spot) + two binary calibration tasks (fast, no judge) +
# the exact-match number task. Override with --include.
DEFAULT_TASKS = ["backtracking", "domain_confusion", "missing_info",
                 "mmlu_prediction", "number_prediction"]
BINARY_TASKS = {"missing_info", "mmlu_prediction"}

# The AObench default decoding temperature (base_experiment._eval_generation_kwargs).
BASE_TEMP = 0.7


def gpu_free_mib() -> int:
    """Smallest per-GPU free memory (MiB); 0 if nvidia-smi is unavailable."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True).stdout
        return min(int(x) for x in out.split())
    except Exception:
        return 0


def wait_for_gpu(min_free_mib: int, poll_s: int = 30) -> None:
    """Block until no *other* run_all eval is active and VRAM has freed up.

    A concurrent eval would OOM or contend, so we hold until (a) no external
    `open_ended_eval.run_all` process is alive and (b) free VRAM clears the
    threshold for two consecutive polls (guards against a half-torn-down run).
    """
    ok = 0
    while True:
        n_runall = int(subprocess.run(
            ["pgrep", "-fc", "open_ended_eval.run_all"],
            capture_output=True, text=True).stdout.strip() or "0")
        free = gpu_free_mib()
        if n_runall == 0 and free >= min_free_mib:
            ok += 1
            if ok >= 2:
                return
        else:
            ok = 0
        print(f"[sweep] waiting for GPU — run_all procs={n_runall}, free={free} MiB "
              f"(need {min_free_mib}); retry in {poll_s}s", flush=True)
        time.sleep(poll_s)


def run_point(coef: float, npos: int, temp: float, tasks: list[str], sample_limit: int) -> str:
    """Run one eval setting via evaluate.py into its own run dir; return the name."""
    from .evaluate import main as eval_main
    run_name = f"sweep_s{coef:g}_np{npos}_t{temp:g}"
    print(f"\n{'='*70}\n[sweep] {run_name}: steering={coef:g}, n_positions={npos}, "
          f"temperature={temp:g}, sample_limit={sample_limit}\n{'='*70}", flush=True)
    eval_main(["--steering-coef", str(coef), "--n-positions", str(npos),
               "--temperature", str(temp), "--sample-limit", str(sample_limit),
               "--run-name", run_name, "--include", *tasks, "--no-dashboard"])
    return run_name


def collect(run_dir: Path, tasks: list[str]) -> dict[str, float | None]:
    """Pull the headline metric per task (+ balanced-acc@best for binary tasks).

    Headline = the first metric in each task's dashboard spec, read with the same
    accessor the dashboard uses. For binary probes we also recompute calibrated
    balanced accuracy from the saved margins (decision quality at the best
    threshold — the thing the steering/position knobs should move)."""
    sums = load_summaries(run_dir)
    row: dict[str, float | None] = {}
    for task in tasks:
        m = TASKS[task]["metrics"][0]
        row[f"{task}:{m['label']}"] = metric_value(sums.get(task), m)
        if task in BINARY_TASKS:
            c = calibration(load_records(run_dir, task))
            row[f"{task}:bal-acc@best"] = c["bal_best"] if c else None
    return row


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--steering", type=float, nargs="*", default=[0.8, 1.0, 1.4, 2.0])
    p.add_argument("--n-positions", type=int, nargs="*", default=[40, 45, 50])
    p.add_argument("--temperature", type=float, nargs="*", default=[0.0, 0.4, 0.7])
    p.add_argument("--sample-limit", type=int, default=40, help="items/task per point")
    p.add_argument("--include", nargs="*", default=DEFAULT_TASKS, help="tasks to sweep")
    p.add_argument("--min-free-mib", type=int, default=40000,
                   help="free VRAM required before starting (waits for a running eval)")
    p.add_argument("--no-wait", action="store_true", help="don't wait for the GPU")
    args = p.parse_args(argv)

    cfg = load_config()
    base_coef = float(cfg["injection"]["eval_steering_coefficient"])
    base_np = int(cfg["eval"]["n_positions"])
    base_temp = BASE_TEMP
    model = cfg["model"]["name"]

    # Three 1-D arms sharing the anchor (base_coef, base_np, base_temp); dedup overlaps.
    settings: list[tuple[float, int, float]] = []
    for c in args.steering:
        settings.append((float(c), base_np, base_temp))
    for n in args.n_positions:
        settings.append((base_coef, int(n), base_temp))
    for t in args.temperature:
        settings.append((base_coef, base_np, float(t)))
    seen, ordered = set(), []
    for s in settings:
        if s not in seen:
            seen.add(s)
            ordered.append(s)

    print(f"[sweep] anchor=(steering={base_coef:g}, n_positions={base_np}, temperature={base_temp:g}); "
          f"{len(ordered)} points; tasks={args.include}; sample_limit={args.sample_limit}")

    if not args.no_wait:
        wait_for_gpu(args.min_free_mib)

    results_root = artifacts_dir(model) / "aobench_results"
    rows: list[tuple[float, int, float, dict]] = []
    for coef, npos, temp in ordered:
        run_name = run_point(coef, npos, temp, args.include, args.sample_limit)
        rows.append((coef, npos, temp, collect(results_root / run_name, args.include)))

    # ---- comparison table (printed + written) ----
    cols = list(rows[0][3].keys()) if rows else []
    header = ["steer", "n_pos", "temp"] + [c.split(":", 1)[1] if ":" in c else c for c in cols]

    def fnum(v):
        return "—" if v is None else f"{v:.3f}"

    def base_cells(c, n, t):
        return [f"{c:g}", str(n), f"{t:g}"]

    table = [header] + [base_cells(c, n, t) + [fnum(r.get(k)) for k in cols] for c, n, t, r in rows]
    widths = [max(len(str(row[i])) for row in table) for i in range(len(header))]
    lines = ["  ".join(str(cell).rjust(widths[i]) for i, cell in enumerate(row)) for row in table]
    text = "\n".join(lines)
    print(f"\n{'='*70}\nSWEEP RESULTS — {model}  (sample_limit={args.sample_limit})\n{'='*70}")
    print(text)

    out = results_root / "sweep_summary.md"
    with open(out, "w") as f:
        f.write(f"# Eval sweep — {model}\n\n")
        f.write(f"anchor: steering={base_coef:g}, n_positions={base_np}, temperature={base_temp:g}; "
                f"sample_limit={args.sample_limit}; tasks={', '.join(args.include)}\n\n")
        f.write("| " + " | ".join(header) + " |\n")
        f.write("|" + "|".join("---" for _ in header) + "|\n")
        for c, n, t, r in rows:
            f.write("| " + " | ".join(base_cells(c, n, t) + [fnum(r.get(k)) for k in cols]) + " |\n")
    print(f"\n[sweep] wrote {out}")


if __name__ == "__main__":
    main()
