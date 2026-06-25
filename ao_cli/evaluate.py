"""Run AObench on an AO LoRA (ours or the paper's released baseline).

Gears-level overview:
  Invokes the vendored AObench suite (third_party/cot-oracle/AObench) as
  `python -m AObench.open_ended_eval.run_all`. For each eval the suite loads
  the base model + AO LoRA, collects activations from the eval dataset's
  contexts, injects them into the AO prompt prefix, generates answers, and
  scores them (exact match / ROC-AUC / LLM judge). Judge-based evals are
  pointed at our local vLLM judge via JUDGE_* env vars; the judge service is
  brought up automatically.

  Modes:
    default      our trained AO on model.name (expects `make train` + datasets
                 installed by `make evalsets`); runs in the train env.
    --baseline   the paper's released Qwen3-8B AO on Qwen3-8B; runs in the
                 legacy env (upstream dependency lock) against the bundled
                 Qwen3-8B AObench datasets.

  Results: artifacts/<slug>/aobench_results/<run>/ (+ summary via
  `python -m AObench.utils.report <results-dir>` -> `make report`).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import (
    AOBENCH_ROOT,
    PROJECT_ROOT,
    PY_LEGACY,
    PY_TRAIN,
    artifacts_dir,
    experiment_name,
    judge_base_url,
    load_config,
    model_slug,
    run,
    run_dir_name,
)
from .judge import up as judge_up


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--baseline", action="store_true", help="evaluate the released Qwen3-8B AO")
    p.add_argument("--lora", default=None, help="LoRA path/repo (default: this run's final checkpoint)")
    # Sweep/override knobs (used by ao_cli.sweep; default None ⇒ take from config).
    p.add_argument("--steering-coef", type=float, default=None, help="override eval injection strength")
    p.add_argument("--temperature", type=float, default=None, help="override AO decoding temp (<=0 ⇒ greedy)")
    p.add_argument("--n-positions", type=int, default=None, help="override activation-window length")
    p.add_argument("--sample-limit", type=int, default=None, help="cap items/task (fast sweeps)")
    p.add_argument("--include", nargs="*", default=None, help="explicit task list (overrides profile)")
    p.add_argument("--run-name", default=None, help="override the output run-dir name")
    p.add_argument("--no-dashboard", action="store_true", help="skip the dashboard build")
    args = p.parse_args(argv)

    cfg = load_config()
    if args.baseline:
        model = cfg["baseline"]["model"]
        lora = args.lora or cfg["baseline"]["ao_lora"]
        # A relative path in config (our own replication checkpoint) is workspace-
        # root relative; resolve it so AObench (cwd=AOBENCH_ROOT) finds it.
        cand = Path(lora) if Path(lora).is_absolute() else PROJECT_ROOT / lora
        is_local = cand.exists()
        if is_local:
            lora = str(cand.resolve())
        # Our checkpoint -> TRAIN env (shares the exact eval stack as future
        # `make eval` mains); a released AO (HF id, no local dir) -> legacy lock.
        py = PY_TRAIN if is_local else PY_LEGACY
        run_name = cfg["eval"].get("baseline_run") or ("baseline_" + model_slug(lora))
    else:
        model = cfg["model"]["smoke_name"] if args.smoke else cfg["model"]["name"]
        py = PY_TRAIN
        # EXP (if set) selects this contribution's named checkpoint + run dir, so
        # `make eval EXP=c3_logitlens` evaluates checkpoints/c3_logitlens/final into
        # aobench_results/c3_logitlens (auto-compared to baseline_replication).
        ckpt_root = artifacts_dir(model) / "checkpoints" / run_dir_name(model, args.smoke)
        # Prefer the val-selected best/ adapter (sft.py snapshots the validation
        # minimum there) over final/: training runs past the generalization
        # optimum, so final/ is the most-overfit point. Fall back to final/ for
        # runs trained before best-checkpoint tracking existed.
        best_ckpt, final_ckpt = ckpt_root / "best", ckpt_root / "final"
        default_ckpt = best_ckpt if best_ckpt.exists() else final_ckpt
        lora = args.lora or str(default_ckpt)
        # Resolve a RELATIVE --lora against the workspace root. AObench runs with
        # cwd=AOBENCH_ROOT, so a path like "artifacts/.../step_12000" would be misread
        # as an HF repo id ("Repo id must be in the form ...") and crash. A genuine HF
        # repo id (no matching local dir) is left untouched.
        if args.lora is not None:
            cand = Path(lora) if Path(lora).is_absolute() else PROJECT_ROOT / lora
            if cand.exists():
                lora = str(cand.resolve())
        if args.lora is None and not default_ckpt.exists():
            sys.exit(f"[eval] no checkpoint at {best_ckpt} or {final_ckpt} — run `make train` first or pass --lora")
        run_name = "smoke" if args.smoke else (experiment_name() or "main")

    run_name = args.run_name or run_name
    out_dir = artifacts_dir(model) / "aobench_results" / run_name
    n_positions = args.n_positions if args.n_positions is not None else cfg["eval"]["n_positions"]
    include = args.include if args.include is not None else (
        cfg["smoke"]["eval_tasks"] if args.smoke else cfg["eval"]["include"])

    cmd = [py, "-m", "AObench.open_ended_eval.run_all",
           "--model", model,
           "--verbalizer-lora", lora,
           "--output-dir", out_dir,
           "--n-positions", str(n_positions)]
    cmd += ["--include", *include] if include else ["--profile", cfg["eval"]["profile"]]
    if args.sample_limit is not None:
        cmd += ["--sample-limit", str(args.sample_limit)]

    judge_up(cfg)
    env = {
        # Route AObench's judge stack to the local vLLM server (see the
        # patched AObench/open_ended_eval/judge.py).
        "JUDGE_USE_ANTHROPIC": "0",
        "JUDGE_USE_LOCAL": "1",
        "JUDGE_LOCAL_URL": judge_base_url(cfg) + "/chat/completions",
        "JUDGE_MODEL": cfg["judge"]["served_name"],
        "AO_JUDGE_ENABLE_THINKING": "1" if cfg["judge"].get("enable_thinking") else "0",
        # Inference-time injection strength (paper trains at 1.0, sweeps eval to
        # ~2.0x). Read by AObench's VerbalizerEvalConfig.steering_coefficient.
        "AO_EVAL_STEERING_COEFFICIENT": str(
            args.steering_coef if args.steering_coef is not None
            else cfg["injection"]["eval_steering_coefficient"]),
        **({"AO_EVAL_TEMPERATURE": str(args.temperature)} if args.temperature is not None else {}),
        "TORCHDYNAMO_DISABLE": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    }
    # Locate the new contribution evals' assets (C5 variants, C6 dataset) so they
    # self-skip cleanly when absent rather than erroring inside AObench.
    diffing_variants = artifacts_dir(model) / "diffing_variants"
    if (diffing_variants / "manifest.json").exists():
        env["AO_DIFFING_VARIANTS"] = str(diffing_variants)
    cfe_dataset = AOBENCH_ROOT / "AObench" / "datasets" / "causal_faithfulness" / "causal_faithfulness_eval_dataset.json"
    if cfe_dataset.exists():
        env["AO_CFE_DATASET"] = str(cfe_dataset)
    run(cmd, env=env, cwd=AOBENCH_ROOT)
    print(f"\n[eval] results in {out_dir}")
    print(f"[eval] report: make report RESULTS={out_dir}")

    # Build the ours-vs-paper comparison dashboard from whatever runs now exist
    # under aobench_results/ (this 'main' run + any 'baseline_*'). Guarded so a
    # dashboard error never fails the eval; skipped for --baseline (it IS the
    # reference, and is picked up automatically when the target run rebuilds).
    if not args.baseline and not args.no_dashboard:
        try:
            from .dashboard import build_dashboard
            html_path = build_dashboard(
                out_dir.parent, target_run=run_name,
                baseline_run=cfg["eval"].get("baseline_run") or None,
                base_label=cfg["eval"].get("baseline_label"))
            print(f"[eval] dashboard: {html_path}")
        except Exception as e:  # noqa: BLE001 — never let reporting break the eval
            print(f"[eval] dashboard skipped ({type(e).__name__}: {e})")


if __name__ == "__main__":
    main()
