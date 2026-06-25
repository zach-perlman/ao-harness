"""Train the Activation Oracle LoRA (render -> materialize datasets -> SFT).

Gears-level overview:
  Three sequential steps, all driven by the rendered training config:
    1. render        config.yaml -> artifacts/<slug>/train_config[_smoke].json
    2. --gen-only    nl_probes/sft.py materializes every dataset component to
                     .pt files (tokenization + position sampling; activations
                     themselves are computed on the fly during training).
    3. torchrun      single-GPU DDP run of nl_probes/sft.py: LoRA on the base
                     model, activations collected with adapters disabled and
                     injected (norm-matched, x steering_coefficient) at the
                     hook layer while training on target_response tokens.

  Checkpoints land in artifacts/<slug>/checkpoints/<run>/final (+ optional HF
  push if hf.push=true and HF_TOKEN is set).
"""

from __future__ import annotations

import argparse
import subprocess
import time

from . import ENVS, REPO, artifacts_dir, load_config, run
from .judge import down as judge_down
from .render import render

SFT = REPO / "nl_probes" / "sft.py"


def _free_gpu_from_judge(timeout_s: int = 60) -> None:
    """Stop the eval-only judge so training gets the whole GPU.

    The vLLM judge (`ao_judge`) is started by `make eval` and pins ~90 GB; left
    resident, it starves training's OOM-preflight (which sizes the largest batch
    against *free* memory) and crashes with near-zero free VRAM. We stop it
    (idempotent — a no-op if already down) and then poll nvidia-smi until the
    freed memory lands, because vLLM's CUDA teardown trails the supervisor stop
    by a few seconds. Stops as soon as free memory stops climbing, or times out.
    """
    judge_down()

    def free_mib() -> int:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True,
        ).stdout.split("\n")[0].strip()
        return int(out) if out.isdigit() else 0

    deadline = time.time() + timeout_s
    prev = -1
    while time.time() < deadline:
        time.sleep(2)
        cur = free_mib()
        if cur and cur <= prev + 256:  # free VRAM has plateaued → teardown done
            break
        prev = cur
    print(f"[train] GPU free before SFT: {free_mib()} MiB (judge stopped)")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--gen-only", action="store_true", help="stop after dataset materialization")
    args = p.parse_args(argv)

    cfg = load_config()
    model = cfg["model"]["smoke_name"] if args.smoke else cfg["model"]["name"]
    art = artifacts_dir(model)
    config_path = art / ("train_config_smoke.json" if args.smoke else "train_config.json")

    render(args.smoke)

    # Unsloth (fused kernels, lower memory, optional FP8 LoRA) lives in
    # envs/unsloth — the transformers>=5 stack that can actually load Qwen3.5
    # (envs/legacy's transformers<5 cannot). When use_unsloth is set we run the
    # whole SFT (gen + train) there; otherwise plain bf16 in envs/train. Smoke
    # (Qwen3.5-0.8B) always uses envs/train.
    use_unsloth = (not args.smoke) and bool(cfg["training"].get("use_unsloth", False))
    env_dir = ENVS / ("unsloth" if use_unsloth else "train")
    py = env_dir / "bin" / "python"
    torchrun = env_dir / "bin" / "torchrun"
    if use_unsloth and not py.exists():
        raise SystemExit("[train] use_unsloth=true but envs/unsloth is missing — run `make setup-unsloth` first")

    env = {
        "WANDB_MODE": cfg["training"]["wandb_mode"],
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    }
    # sft.py imports unsloth at module load (before transformers/peft) only when
    # this is set, which is exactly what its use_unsloth path asserts.
    if use_unsloth:
        env["AO_USE_UNSLOTH"] = "1"

    run([py, SFT, "--config", config_path, "--gen-only"], env=env)
    if not args.gen_only:
        _free_gpu_from_judge()
        run([torchrun, "--nproc_per_node=1", SFT, "--config", config_path], env=env)


if __name__ == "__main__":
    main()
