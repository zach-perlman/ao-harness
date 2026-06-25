"""
Generate CoT rollouts for backtracking eval problems.

Usage:
    .venv/bin/python data_pipelines/backtracking/generate_backtracking_rollouts.py --model Qwen/Qwen3-8B

Reads problems from data_pipelines/backtracking/problems.json,
generates N rollouts per problem with thinking enabled,
saves results to data_pipelines/backtracking/{model_dir}/rollouts.json
"""

import argparse
import json
import os
from pathlib import Path

import vllm
from transformers import AutoTokenizer

from data_pipelines.pipeline_utils import add_model_arg, model_dir_name, vllm_gpu_util

NUM_ROLLOUTS = 10
# Qwen3.5-4B is a verbose reasoner: at 3000 tokens ~98% of traces were cut off
# mid-thought (never closing </think>), so backtracking arcs never resolved and
# point yield was tiny. 12000 lets the great majority finish; paired with the
# raised max_model_len below (the trace was double-capped before).
MAX_TOKENS = 12000

PROBLEMS_PATH = Path("data_pipelines/backtracking/problems.json")


def main(model_name: str):
    # Load problems
    with open(PROBLEMS_PATH) as f:
        problems = json.load(f)

    output_path = Path(f"data_pipelines/backtracking/{model_dir_name(model_name)}/rollouts.json")

    print(f"Loaded {len(problems)} problems")

    # Init tokenizer for chat formatting
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Init vLLM
    llm = vllm.LLM(
        model=model_name,
        # Must exceed prompt + MAX_TOKENS so long reasoning traces aren't clipped.
        max_model_len=16384,
        # Qwen3.5 is a GatedDeltaNet hybrid: its decode path is kernel-launch-
        # bound, so eager mode (no CUDA graphs) is ~100x slower here, not the
        # ~2.5x upstream assumed for dense transformers. Keep CUDA graphs on.
        enforce_eager=False,
        tensor_parallel_size=1,
        # Each concurrent decode needs one Mamba cache block; with the larger
        # max_model_len (and the judge sharing the card) the default 1024 exceeds
        # the available blocks and aborts graph capture. We only have 300 prompts,
        # so 256 in flight already saturates the GPU.
        max_num_seqs=256,
        gpu_memory_utilization=vllm_gpu_util(0.9),
    )

    sampling_params = vllm.SamplingParams(
        temperature=1.0,
        top_p=0.95,
        max_tokens=MAX_TOKENS,
    )

    # Format all prompts (each problem repeated NUM_ROLLOUTS times)
    all_formatted = []
    prompt_index_map = []  # maps each formatted prompt back to its problem index

    for prob_idx, problem in enumerate(problems):
        messages = [{"role": "user", "content": problem["problem"]}]
        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        for rollout_idx in range(NUM_ROLLOUTS):
            all_formatted.append(formatted)
            prompt_index_map.append((prob_idx, rollout_idx))

    print(f"Generating {len(all_formatted)} total rollouts ({len(problems)} problems x {NUM_ROLLOUTS} each)")

    # Generate all at once (vLLM handles batching internally)
    outputs = llm.generate(all_formatted, sampling_params)

    # Organize results
    results = []
    for problem in problems:
        results.append(
            {
                "id": problem["id"],
                "domain": problem["domain"],
                "problem": problem["problem"],
                "notes": problem["notes"],
                "rollouts": [],
            }
        )

    for i, output in enumerate(outputs):
        prob_idx, rollout_idx = prompt_index_map[i]
        full_text = output.outputs[0].text

        # Split thinking from answer. Qwen3.5's chat template pre-opens the think
        # block in the PROMPT (assistant turn starts with "<think>\n"), so the
        # generated text begins INSIDE the reasoning and carries no opening tag.
        # Thinking is therefore everything up to "</think>" if the trace closed,
        # else the whole (token-capped) output. The leading .replace keeps this
        # correct for Qwen3 too, where the output does start with "<think>".
        if "</think>" in full_text:
            thinking, answer = full_text.split("</think>", 1)
            answer = answer.strip()
        else:
            thinking, answer = full_text, ""
        thinking = thinking.replace("<think>", "").strip()

        results[prob_idx]["rollouts"].append(
            {
                "rollout_idx": rollout_idx,
                "thinking": thinking,
                "answer": answer,
                "full_text": full_text,
                "num_tokens": len(output.outputs[0].token_ids),
            }
        )

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    total_tokens = sum(r["num_tokens"] for result in results for r in result["rollouts"])
    print(f"\nDone! Saved {len(results)} problems with rollouts to {output_path}")
    print(f"Total tokens generated: {total_tokens:,}")

    # Quick stats on thinking lengths
    thinking_lengths = [len(r["thinking"]) for result in results for r in result["rollouts"]]
    print(
        f"Thinking length (chars): min={min(thinking_lengths)}, max={max(thinking_lengths)}, "
        f"avg={sum(thinking_lengths) / len(thinking_lengths):.0f}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_model_arg(parser)
    args = parser.parse_args()
    main(model_name=args.model)
