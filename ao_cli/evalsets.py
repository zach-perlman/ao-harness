"""Regenerate AObench eval datasets for the target model, then install them.

Gears-level overview:
  AObench ships datasets baked from Qwen3-8B (model answers, CoT prefixes,
  logprobs...). To evaluate an AO on a different target model the datasets
  must be REGENERATED from that model's own rollouts. The vendored repo's
  data_pipelines/<task>/generate_dataset.py scripts do this, writing to
  data_pipelines/<task>/<model-slug>/. We:

    1. run each requested generator in the vllm env (GPU rollouts), with
       AO_JUDGE_* env set so any Anthropic judge stage hits the local judge
       (see data_pipelines/local_judge.py shim in the vendored repo);
    2. copy the outputs into AObench/datasets/<task>/ where the eval suite
       reads them (flat layout, one target model at a time);
    3. stash a copy under artifacts/<slug>/aobench_datasets/ so switching
       target models never requires regeneration.

  Task requirements:  number_prediction, mmlu_prediction -> vLLM only;
  missing_info, backtracking -> vLLM + judge (judge must be up).
  vagueness + domain_confusion reuse the backtracking dataset.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from . import AOBENCH_ROOT, PY_VLLM, REPO, artifacts_dir, judge_base_url, load_config, model_slug, run
from .judge import down as judge_down
from .judge import up as judge_up

GEN = REPO / "data_pipelines"
AOBENCH_DATA = AOBENCH_ROOT / "AObench" / "datasets"

# Evalsets is the one place the judge must co-reside with a target-model
# generator on a single card (the missing_info/backtracking generators run their
# own vLLM and call the judge mid-pipeline). On the old 80GB A100 this forced a
# downgrade to the judge's GPTQ-Int4 build to make two models fit; the 140GB H200
# removes that constraint. We keep the SAME FP8 judge as convqa + AObench scoring
# (identical eval labels, no extra Int4 download) and just give it a smaller GPU
# slice: judge 0.30 (~42GB: 37.5GB FP8 weights + a little cache) + generator 0.45
# (~63GB) = ~105GB of 140GB, leaving ~35GB headroom for two CUDA contexts.
# served_name is unchanged, so the generator shim (data_pipelines/local_judge.py)
# resolves the same endpoint.
JUDGE_TASK_VLLM_UTIL = "0.45"


def _coexist_cfg(cfg: dict) -> dict:
    """Judge on a 0.30 slice (FP8, full quality) to share the H200 with a generator."""
    return {**cfg, "judge": {**cfg["judge"],
                             "gpu_memory_utilization": 0.30,
                             "max_num_seqs": 128}}


def _judge_env(cfg: dict) -> dict:
    return {
        "AO_JUDGE_BASE_URL": judge_base_url(cfg),
        "AO_JUDGE_MODEL": cfg["judge"]["served_name"],
        "AO_JUDGE_ENABLE_THINKING": "1" if cfg["judge"].get("enable_thinking") else "0",
        "AO_VLLM_GPU_UTIL": JUDGE_TASK_VLLM_UTIL,
    }


def _outputs(task: str, slug: str) -> list[tuple[Path, Path]]:
    """(source produced by generator, destination inside AObench) pairs."""
    if task == "missing_info":  # flat output, not per-model
        return [(GEN / "missing_info" / "missing_info_eval_dataset.json",
                 AOBENCH_DATA / "missing_info" / "missing_info_eval_dataset.json")]
    return [(GEN / task / slug / f"{task}_eval_dataset.json",
             AOBENCH_DATA / task / f"{task}_eval_dataset.json")]


def generate(task: str, model: str, cfg: dict, needs_judge: bool, n_per_task: int | None = None) -> None:
    env = _judge_env(cfg) if needs_judge else {}
    if task == "backtracking":
        # backtracking has its own two-script pipeline that doesn't take a size knob.
        run([PY_VLLM, GEN / "backtracking" / "generate_backtracking_rollouts.py", "--model", model], env=env)
        run([PY_VLLM, GEN / "backtracking" / "build_dataset_v2.py", "--model", model], env=env)
    else:
        # number_prediction / mmlu_prediction scale with --n-per-task; missing_info
        # accepts it but is capped by its hand-written PROBLEMS list (no-op there).
        cmd = [PY_VLLM, GEN / task / "generate_dataset.py", "--model", model]
        if n_per_task is not None:
            cmd += ["--n-per-task", str(n_per_task)]
        run(cmd, env=env)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--tasks", nargs="*", default=None, help="override config.yaml task list")
    args = p.parse_args(argv)

    cfg = load_config()
    model = cfg["model"]["smoke_name"] if args.smoke else cfg["model"]["name"]
    slug = model_slug(model)
    tasks = args.tasks or (cfg["smoke"]["eval_tasks"] if args.smoke else cfg["data"]["evalsets"]["tasks"])
    # Target items per task: tiny for smoke, else config.yaml data.evalsets.n_per_task.
    n_per_task = cfg["smoke"].get("evalset_n") if args.smoke else cfg["data"]["evalsets"].get("n_per_task")
    judge_tasks = {"missing_info", "backtracking"}

    # Run vLLM-only tasks first with the judge DOWN (full GPU for the generator),
    # then the judge-tasks with the judge UP (generator capped to coexist). This
    # avoids stopping/starting the slow MoE judge more than once.
    ordered = sorted(tasks, key=lambda t: t in judge_tasks)
    judge_down()
    judge_started = False

    for task in ordered:
        needs_judge = task in judge_tasks
        if needs_judge and not judge_started:
            judge_up(_coexist_cfg(cfg))
            judge_started = True
        print(f"\n[evalsets] === {task} for {model} ===")
        generate(task, model, cfg, needs_judge=needs_judge, n_per_task=n_per_task)
        for src, dst in _outputs(task, slug):
            if not src.exists():
                raise SystemExit(f"[evalsets] expected output missing: {src}")
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            stash = artifacts_dir(model) / "aobench_datasets" / dst.parent.name / dst.name
            stash.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, stash)
            print(f"[evalsets] installed {dst}")


if __name__ == "__main__":
    main()
