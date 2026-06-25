"""Generate the on-policy CoT corpus for the target model.

Gears-level overview:
  Wraps third_party/cot-oracle/src/data_pipeline/generate_cots.py (the
  pipeline that built the paper's cot-oracle-corpus-v5): sample problems from
  the same 13 question sources (MATH, GSM8K, MMLU-Pro, LMSYS, ...), roll out
  chain-of-thought from the TARGET model with vLLM (temperature 0.6, thinking
  enabled), split each CoT into sentences, and tag correctness categories.

  We spread config.yaml's data.corpus.n_problems evenly across the medium
  preset's sources and write artifacts/<slug>/corpus/corpus.jsonl. That file
  feeds both the past/future-lens task (pretrain_key=cot_response) and convqa
  generation (sentences).
"""

from __future__ import annotations

import argparse
import math

from . import AOBENCH_ROOT, PY_VLLM, artifacts_dir, load_config, run

GEN_SCRIPT = AOBENCH_ROOT / "src" / "data_pipeline" / "generate_cots.py"

# Source list of the paper's "medium" corpus preset (see generate_cots.py).
SOURCES = ["math", "gsm8k", "aqua_rat", "asdiv", "arc_challenge", "arc_easy",
           "commonsenseqa", "scienceqa", "medqa", "mmlu_pro", "scruples", "lmsys"]


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args(argv)

    cfg = load_config()
    model = cfg["model"]["smoke_name"] if args.smoke else cfg["model"]["name"]
    n_total = cfg["smoke"]["corpus_n_problems"] if args.smoke else cfg["data"]["corpus"]["n_problems"]
    n_per_source = max(1, math.ceil(n_total / len(SOURCES)))
    out = artifacts_dir(model) / "corpus" / "corpus.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)

    run(
        [PY_VLLM, GEN_SCRIPT,
         "--model", model,
         "--engine", "vllm",
         "--sources", *SOURCES,
         "--n-problems", str(n_per_source),
         "--n-rollouts", str(cfg["data"]["corpus"]["n_rollouts"]),
         "--output", out],
        cwd=AOBENCH_ROOT,
    )
    print(f"[corpus] wrote {out}")


if __name__ == "__main__":
    main()
