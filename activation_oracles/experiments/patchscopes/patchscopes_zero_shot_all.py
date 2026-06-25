import ast
import os
import re
import unicodedata
from typing import List, Tuple
import json

import pandas as pd
import vllm

from nl_probes.utils.common import load_tokenizer


FOLDER = os.path.join("datasets", "factual")

# Base model to query via vLLM
VLLM_MODEL_NAME = "Qwen/Qwen3-8B"

# vLLM generation config
MAX_NEW_TOKENS = 10
TEMPERATURE = 0.0

MAX_PROMPTS: int = 200

EXPERIMENTS_DIR: str = "experiments/patchscopes_eval_results"
os.makedirs(EXPERIMENTS_DIR, exist_ok=True)

model_name_str = VLLM_MODEL_NAME.split("/")[-1].replace(".", "_")
OUTPUT_FILENAME = f"{EXPERIMENTS_DIR}/patchscopes_zero-shot_{model_name_str}.json"

# Prompt template (kept close to the original zero-shot version)
PROMPT_TEMPLATE = (
    "Respond with the answer only and nothing else. "
    "What is {prompt_target}the entity at the end of this phrase: '{context}'?"
)


def parse_answer(s: str) -> str:
    """Normalize an answer to a simple whitespace-separated, ASCII lowercase form."""
    if s is None:
        return ""
    s = s.strip()

    # Decode literal escapes like "\u00ed" when present
    if "\\u" in s or "\\x" in s:
        try:
            s = s.encode("utf-8").decode("unicode_escape")
        except Exception:
            pass

    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"[^0-9A-Za-z]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s


def compare_patchscope_responses(response: str, target_response: str) -> bool:
    response = parse_answer(response)
    target_response = parse_answer(target_response)
    return target_response in response


def build_prompts_and_targets(df: pd.DataFrame, tokenizer) -> Tuple[List[str], List[str]]:
    """Create chat prompts and targets for a single dataset file."""
    prompts: list[str] = []
    targets: list[str] = []

    for _, row in df.iterrows():
        if len(prompts) >= MAX_PROMPTS:
            break
        # Build the context prefix up to the source position
        source_toks = ast.literal_eval(row["source_cropped_toks"])  # list[str]
        source_pos = int(row["position_source"])
        source = "".join(source_toks[: source_pos + 1])

        # Convert dataset "prompt_target" like "CountryX" -> "country"
        prompt_target = row["prompt_target"]
        if isinstance(prompt_target, str) and prompt_target.endswith("x"):
            prompt_target = prompt_target[:-1].lower()

        prompt = PROMPT_TEMPLATE.format(prompt_target=prompt_target, context=source)

        message = [{"role": "user", "content": prompt}]
        prompt_text = tokenizer.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        prompts.append(prompt_text)
        targets.append(row["object"])  # ground truth answer

    return prompts, targets


def evaluate_dataset_file(path: str, llm: vllm.LLM, tokenizer) -> Tuple[float, List[Tuple[str, str, str]]]:
    """Return (accuracy, items) for a single .tsv dataset file.

    items is a list of (prompt, response, ground_truth) tuples.
    """
    df = pd.read_csv(path, sep="\t")
    prompts, targets = build_prompts_and_targets(df, tokenizer)

    sampling_params = vllm.SamplingParams(temperature=TEMPERATURE, max_tokens=MAX_NEW_TOKENS)

    # Generate for all prompts at once (simple path)
    outs = llm.generate(prompts, sampling_params)
    responses_text = [o.outputs[0].text for o in outs]

    # Compute accuracy
    assert len(responses_text) == len(targets)
    correct = 0
    items: list[tuple[str, str, str]] = []
    for prompt, resp, tgt in zip(prompts, responses_text, targets):
        if compare_patchscope_responses(resp, tgt):
            correct += 1
        items.append((prompt, resp, tgt))

    total = len(targets) if targets else 1
    return correct / total, items


def main() -> None:
    tokenizer = load_tokenizer(VLLM_MODEL_NAME)
    llm = vllm.LLM(
        model=VLLM_MODEL_NAME,
        max_model_len=2000,
        enforce_eager=True,
        enable_lora=True,
        max_lora_rank=32,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.5,
    )

    tsv_files = sorted(f for f in os.listdir(FOLDER) if f.endswith(".tsv"))
    results: dict[str, float] = {}
    details: dict[str, list[tuple[str, str, str]]] = {}

    for fname in tsv_files:
        fpath = os.path.join(FOLDER, fname)
        acc, items = evaluate_dataset_file(fpath, llm, tokenizer)
        results[fname] = acc
        details[fname] = items

    # Compute mean accuracy over all items (not per-dataset mean)
    total_items = sum(len(items) for items in details.values())
    total_correct = sum(results[fname] * len(details[fname]) for fname in results.keys())
    mean_accuracy_all_items = total_correct / total_items

    # Include overall mean accuracy in the results dictionary
    results["mean_accuracy_all_items"] = mean_accuracy_all_items

    print(results)
    # print(details)

    data = {
        "results": results,
        "details": details,
    }

    with open(OUTPUT_FILENAME, "w") as f:
        json.dump(data, f)




if __name__ == "__main__":
    main()
