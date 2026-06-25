"""Stage-0 layer probe (mechanism: scripts/layer_probe.py).

Thin orchestration only: resolve the target model + output path from config.yaml
and shell out to the worker in the AObench env/cwd, exactly like `evaluate`. The
worker fits a per-layer logistic probe on the target model's residuals over the
AObench binary-task items, giving a depth profile that tells us which layers to
spend a full AO retrain on (see the layer-sweep plan).
"""

from __future__ import annotations

import argparse

from . import AOBENCH_ROOT, PROJECT_ROOT, PY_TRAIN, artifacts_dir, load_config, run


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--tasks", nargs="*", default=["mmlu_prediction", "missing_info"])
    p.add_argument("--max-entries", type=int, default=None, help="cap items/task")
    p.add_argument("--segment-start", type=int, default=-10)
    p.add_argument("--stride", type=int, default=1, help="scan every Nth layer")
    p.add_argument("--batch-size", type=int, default=8)
    args = p.parse_args(argv)

    cfg = load_config()
    model = cfg["model"]["name"]
    out = artifacts_dir(model) / "layer_probe" / "layer_probe.json"

    worker = PROJECT_ROOT / "scripts" / "layer_probe.py"
    cmd = [PY_TRAIN, str(worker),
           "--model", model, "--out", str(out),
           "--tasks", *args.tasks,
           "--segment-start", str(args.segment_start),
           "--stride", str(args.stride),
           "--batch-size", str(args.batch_size)]
    if args.max_entries is not None:
        cmd += ["--max-entries", str(args.max_entries)]

    # No grad, no compile; expandable segments keeps the long-context forwards off
    # the OOM cliff. The probe needs only the base model (no judge, no LoRA).
    env = {"TORCHDYNAMO_DISABLE": "1", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    run(cmd, env=env, cwd=AOBENCH_ROOT)
    print(f"\n[layer-probe] wrote {out}")


if __name__ == "__main__":
    main()
