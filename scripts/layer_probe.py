#!/usr/bin/env python3
"""Stage-0 layer probe: at which depth is each AObench property linearly legible?

Gears-level overview
--------------------
The Activation Oracle reads the target model's residual stream at a fixed set of
layers (config.layers) and is *trained* to interpret them. Before paying for a
full AO retrain at every candidate depth (the "layer sweep"), this script asks
the cheaper question directly: at each layer, is the property the AO is graded
on even *linearly present* in the residual stream?

Mechanism — one forward pass over an eval set yields every layer at once:
  1. Rebuild each binary AObench task's items with the SAME tokenization and the
     SAME trailing activation window the eval uses
     (build_*_verbalizer_prompt_infos -> context_token_ids, positions, yes/no).
  2. A multi-layer forward hook captures the residual at every scanned layer;
     mean-pool over each item's `positions` -> one feature vector per (item,layer).
  3. For each layer independently, fit a logistic probe with stratified k-fold CV
     and report ROC-AUC. We also report informativeness |AUC-0.5|: a layer whose
     probe is strongly *inverted* (AUC<0.5) still means the info IS there, just
     sign-flipped — which is exactly the kind of depth worth retraining the AO at.

Output: a {task: {layer: {auc, info, depth_pct}}} JSON + a printed table and, per
task, the peak-informativeness layer (the center_percent to aim a retrain at).

This reads activations identically to training/eval (same get_hf_submodule path,
same position window), so the depth profile is a faithful, ~1%-cost proxy for how
a full AO retrain at that depth would fare.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from AObench.utils.common import load_model, load_tokenizer
from AObench.utils.activation_utils import (
    get_hf_submodule,
    collect_activations_multiple_layers,
)
from AObench.open_ended_eval.mmlu_prediction import (
    load_mmlu_prediction_dataset,
    build_mmlu_prediction_verbalizer_prompt_infos,
    POST_ANSWER_PROMPTS,
)
from AObench.open_ended_eval.missing_info import (
    load_missing_info_dataset,
    build_missing_info_verbalizer_prompt_infos,
    VERBALIZER_PROMPTS as MISSING_INFO_PROMPTS,
)


# Each task -> a builder that returns VerbalizerInputInfo items (one per entry)
# carrying (context_token_ids, positions, ground_truth ∈ {yes,no}). We pick the
# single most diagnostic prompt per task: mmlu post-answer (the AO's "alive"
# signal in baseline #1) and missing_info incomplete_info (the inverted one).
def _build_mmlu(tokenizer, max_entries, segment_start):
    entries = load_mmlu_prediction_dataset(max_entries=max_entries)
    infos, _ = build_mmlu_prediction_verbalizer_prompt_infos(
        entries, POST_ANSWER_PROMPTS, tokenizer, segment_start=segment_start)
    return infos


def _build_missing_info(tokenizer, max_entries, segment_start):
    entries = load_missing_info_dataset(max_entries=max_entries)
    infos, _ = build_missing_info_verbalizer_prompt_infos(
        entries, MISSING_INFO_PROMPTS, tokenizer, segment_start=segment_start)
    return infos


TASK_BUILDERS = {
    "mmlu_prediction": _build_mmlu,
    "missing_info": _build_missing_info,
}


def num_hidden_layers(model) -> int:
    """Decoder depth, handling the nested text_config of multimodal wrappers."""
    cfg = model.config
    if hasattr(cfg, "num_hidden_layers") and cfg.num_hidden_layers:
        return cfg.num_hidden_layers
    return cfg.text_config.num_hidden_layers


@torch.no_grad()
def extract_features(model, tokenizer, infos, layers, device, batch_size):
    """Mean-pooled residual over each item's `positions`, for every layer.

    Left-pads each batch (tokenizer.padding_side='left'), so an item's absolute
    positions shift by its pad width. One multi-layer hook grabs all `layers` in a
    single forward; we pool the window immediately and keep only the [D] vector.
    Returns {layer: float32 array [N, D]}.
    """
    pad_id = tokenizer.pad_token_id
    submodules = {l: get_hf_submodule(model, l) for l in layers}
    feats: dict[int, list] = {l: [] for l in layers}

    for i in range(0, len(infos), batch_size):
        batch = infos[i:i + batch_size]
        max_len = max(len(b.context_token_ids) for b in batch)
        input_ids, attn, windows = [], [], []
        for b in batch:
            ids = list(b.context_token_ids)
            pad = max_len - len(ids)
            input_ids.append([pad_id] * pad + ids)
            attn.append([0] * pad + [1] * len(ids))
            windows.append([pad + p for p in b.positions])  # positions in padded seq
        inputs = {
            "input_ids": torch.tensor(input_ids, device=device),
            "attention_mask": torch.tensor(attn, device=device),
        }
        acts = collect_activations_multiple_layers(model, submodules, inputs, None, None)
        for l in layers:
            A = acts[l]  # [B, L, D]
            for bi, ps in enumerate(windows):
                feats[l].append(A[bi, ps, :].float().mean(0).cpu())

    return {l: torch.stack(v).numpy() for l, v in feats.items()}


def probe_auc(X, y, folds, seed) -> float:
    """Cross-validated ROC-AUC of a standardized logistic probe on one layer."""
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    proba = cross_val_predict(clf, X, y, cv=skf, method="predict_proba")[:, 1]
    return float(roc_auc_score(y, proba))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True, help="output JSON path")
    p.add_argument("--tasks", nargs="*", default=list(TASK_BUILDERS))
    p.add_argument("--max-entries", type=int, default=None, help="cap items/task")
    p.add_argument("--segment-start", type=int, default=-10,
                   help="trailing activation window start (negative = from end)")
    p.add_argument("--stride", type=int, default=1, help="scan every Nth layer")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = load_tokenizer(args.model)
    model = load_model(args.model, torch.bfloat16)
    model.eval()

    L = num_hidden_layers(model)
    # Skip layer 0 (embedding-dominated) and the final layer (already next-token).
    layers = list(range(1, L - 1, args.stride))
    print(f"[probe] {args.model}: {L} layers; scanning {layers}")

    results: dict[str, dict] = {}
    for task in args.tasks:
        infos = TASK_BUILDERS[task](tokenizer, args.max_entries, args.segment_start)
        y = np.array([1 if info.ground_truth == "yes" else 0 for info in infos])
        print(f"\n[probe] {task}: {len(infos)} items, positive rate {y.mean():.2f}")
        X_by_layer = extract_features(model, tokenizer, infos, layers, device, args.batch_size)

        per_layer = {}
        for l in layers:
            auc = probe_auc(X_by_layer[l], y, args.folds, args.seed)
            per_layer[l] = {"auc": auc, "info": abs(auc - 0.5), "depth_pct": round(l / L * 100)}
        results[task] = per_layer

        peak = max(per_layer, key=lambda l: per_layer[l]["info"])
        print(f"  layer  depth%   AUC    |AUC-0.5|")
        for l in layers:
            r = per_layer[l]
            mark = "  <- peak" if l == peak else ""
            print(f"  {l:>4}   {r['depth_pct']:>4}   {r['auc']:.3f}    {r['info']:.3f}{mark}")
        print(f"  [{task}] peak informativeness at layer {peak} "
              f"({per_layer[peak]['depth_pct']}% depth, AUC {per_layer[peak]['auc']:.3f})")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"model": args.model, "num_layers": L, "results": results}, indent=2))
    print(f"\n[probe] wrote {out}")


if __name__ == "__main__":
    main()
