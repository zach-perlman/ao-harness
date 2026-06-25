#!/usr/bin/env python3
"""Does the gemma AO's yes/no decision actually use the injected activation?

Gears-level overview
--------------------
We run the *real* AObench binary verbalizer scoring (same model, AO LoRA, hook,
norm-matching, coefficient) on missing_info, but intercept the one function that
produces the activation to be injected — collect_target_activations — and replace
its output three ways:

  real    : the true target-model activation (the normal eval).
  shuffle : each item gets ANOTHER item's activation (right magnitude, wrong
            content) — tests whether the AO reads the *content*.
  zero    : the injected vector is zeroed (no information) — tests the floor.

If margins / P(yes) / AUC are ~identical across real, shuffle and zero, the AO's
yes/no output is insensitive to the injected activation, i.e. it ignores the
injection and falls back to its prior — which eval-time coefficient cannot fix.
"""

from __future__ import annotations

import argparse
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

import AObench.base_experiment as be
from AObench.utils.common import load_model, load_tokenizer
from AObench.open_ended_eval.eval_runner import (
    build_verbalizer_eval_config,
    build_yes_no_candidate_token_groups,
    ensure_default_adapter,
)
from AObench.open_ended_eval.missing_info import (
    load_missing_info_dataset,
    build_missing_info_verbalizer_prompt_infos,
    VERBALIZER_PROMPTS as MISSING_INFO_PROMPTS,
)

# --- activation ablation: wrap the real collector, mutate its output ----------
_real_collect = be.collect_target_activations
MODE = {"v": "real"}


def _patched_collect(model, inputs_BL, config, target_lora_path):
    acts = _real_collect(model, inputs_BL, config, target_lora_path)
    mode = MODE["v"]
    if mode == "real":
        return acts
    out = {}
    for act_key, by_layer in acts.items():
        out[act_key] = {}
        for layer, t in by_layer.items():
            if mode == "zero":
                out[act_key][layer] = torch.zeros_like(t)
            elif mode == "shuffle":
                # roll the batch dim so each context gets a neighbour's activation
                perm = torch.roll(torch.arange(t.shape[0], device=t.device), 1)
                out[act_key][layer] = t[perm]
    return out


be.collect_target_activations = _patched_collect


def score(model, tokenizer, infos, lora_name, config, device, groups):
    res = be.run_verbalizer_binary_score(
        model=model, tokenizer=tokenizer, verbalizer_prompt_infos=infos,
        verbalizer_lora_path=lora_name, target_lora_path=None,
        config=config, device=device, candidate_token_groups=groups)
    ys = np.array([float(r.candidate_scores["yes"]) for r in res])
    ns = np.array([float(r.candidate_scores["no"]) for r in res])
    gt = np.array([1 if r.meta_info["ground_truth"] == "yes" else 0 for r in res])
    return ys - ns, gt  # per-item margin (yes-no), label


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", required=True)
    p.add_argument("--ao", required=True, help="AO LoRA checkpoint path")
    p.add_argument("--max-entries", type=int, default=40)
    p.add_argument("--coef", type=float, default=2.0, help="eval steering coefficient")
    p.add_argument("--batch-size", type=int, default=16)
    args = p.parse_args()

    import os
    os.environ["AO_EVAL_STEERING_COEFFICIENT"] = str(args.coef)
    device = torch.device("cuda")

    tokenizer = load_tokenizer(args.model)
    model = load_model(args.model, torch.bfloat16)
    model.eval()
    ensure_default_adapter(model)  # makes the model adapter-capable (creates peft_config)
    lora_name, training_config = be.load_oracle_adapter(model, args.ao)
    config = build_verbalizer_eval_config(
        model_name=args.model, training_config=training_config,
        eval_batch_size=args.batch_size,
        generation_kwargs={"do_sample": False, "max_new_tokens": 1})
    groups = build_yes_no_candidate_token_groups(tokenizer)

    entries = load_missing_info_dataset(max_entries=args.max_entries)
    infos, _ = build_missing_info_verbalizer_prompt_infos(entries, MISSING_INFO_PROMPTS, tokenizer)

    print(f"\ncoef={args.coef}, n={len(infos)} items\n")
    margins = {}
    for mode in ("real", "shuffle", "zero"):
        MODE["v"] = mode
        m, y = score(model, tokenizer, infos, lora_name, config, device, groups)
        margins[mode] = m
        auc = roc_auc_score(y, m) if len(set(y)) > 1 else float("nan")
        print(f"  {mode:8s}  margin mean={m.mean():+.3f} std={m.std():.3f}  "
              f"pred_yes={(m>0).mean():.2f}  AUC={auc:.3f}")

    # The decisive number: how much does the AO's per-item margin MOVE when the
    # injected activation content changes (real -> shuffle) or vanishes (-> zero)?
    print("\n  per-item |Δ margin| vs real:")
    for mode in ("shuffle", "zero"):
        d = np.abs(margins[mode] - margins["real"])
        print(f"    {mode:8s}  mean={d.mean():.3f}  max={d.max():.3f}")
    print("\n  (≈0 change => AO ignores the injected activation for this decision)")


if __name__ == "__main__":
    main()
