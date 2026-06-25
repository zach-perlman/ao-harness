"""Inference-time calibration measurement for the AO (doc Section 3, adopted not trained).

WHY
---
Two uncertainty-quantification practices are adopted directly from recent work:

  1. Yes/No logit margin (logit_Yes - logit_No) — a cheap per-query triage signal.
     AObench's binary probes already store `margin_yes_minus_no` and the dashboard
     turns it into ROC-AUC + threshold-calibration panels, so it needs no new code;
     `yes_no_margin()` here just names the operation for reuse.

  2. Bootstrap mode frequency — Torrielli et al. (2026) find this is the best-
     calibrated UQ for AOs (ECE 5.7% vs 25.5% for log-prob). Sample k completions,
     semantically cluster them, and take the modal cluster's frequency as the
     model's confidence. This module MEASURES that calibration so every AO run gets
     a comparable number.

We never TRAIN or prompt verbal confidence — Miao & Ungar (2026) show the verbal-
confidence direction is orthogonal to (and anti-calibrated against) the true
calibration direction. Hence this is eval-only tooling.

HOW (the driver)
----------------
On an *answerable* probe (factual (prompt → entity) pairs from data_pipelines/factual,
so ground truth is known without a judge):
  • inject each prompt's last-token activation into the AO and sample k completions
    (reusing the exact nl_probes injected-inference path the eval uses);
  • cluster the k completions with mpnet cosine similarity → mode_frequency = (size
    of the largest cluster)/k, the confidence; the cluster's first completion is the
    modal answer;
  • an item is "correct" iff mpnet cos(modal answer, ground-truth entity) ≥ threshold;
  • report mean confidence, accuracy, and Expected Calibration Error (ECE) between
    the two — a well-calibrated AO has confidence ≈ accuracy in every bin.
Writes artifacts/<slug>/calibration/<run>.json and prints a reliability table.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path

import torch

from . import REPO, artifacts_dir, experiment_name, load_config, model_slug, resolve_layers, run_dir_name
from .solvability import _SentenceEmbedder

FACTUAL = REPO / "data_pipelines" / "factual"
RELATIONS = ["country_capital_city", "country_currency", "country_largest_city",
             "food_from_country", "product_by_company", "star_constellation"]
AO_QUESTION = "What place, entity, or word is this activation about? Reply with just the name."


# ---- adopted estimators (small, pure, reusable) -----------------------------

def yes_no_margin(logit_yes: float, logit_no: float) -> float:
    """Cheap triage signal: positive ⇒ AO leans 'yes'. (See dashboard for ROC use.)"""
    return logit_yes - logit_no


def expected_calibration_error(confidences: list[float], correct: list[bool], n_bins: int = 10) -> float:
    """ECE = Σ_bins (|bin| / N) · |accuracy(bin) − mean_confidence(bin)|."""
    n = len(confidences)
    if n == 0:
        return 0.0
    bins: list[list[tuple[float, bool]]] = [[] for _ in range(n_bins)]
    for c, ok in zip(confidences, correct):
        bins[min(int(c * n_bins), n_bins - 1)].append((c, ok))
    ece = 0.0
    for b in bins:
        if not b:
            continue
        conf = sum(c for c, _ in b) / len(b)
        acc = sum(ok for _, ok in b) / len(b)
        ece += len(b) / n * abs(acc - conf)
    return ece


def mode_frequency(completions: list[str], embedder: _SentenceEmbedder, threshold: float) -> tuple[str, float]:
    """Greedy semantic clustering → (modal completion, largest-cluster fraction).

    Each completion joins the first existing cluster whose representative it matches
    (cosine ≥ threshold), else it seeds a new cluster. The modal answer is the
    first member of the largest cluster; its frequency is that cluster's share of k.
    """
    emb = embedder._embed([c or " " for c in completions])  # [k, D], unit-normalized
    reps: list[int] = []            # indices of cluster representatives
    assigned: list[int] = []
    for i in range(len(completions)):
        best, best_sim = -1, threshold
        for ci, ri in enumerate(reps):
            sim = float((emb[i] * emb[ri]).sum())
            if sim >= best_sim:
                best, best_sim = ci, sim
        if best == -1:
            reps.append(i)
            assigned.append(len(reps) - 1)
        else:
            assigned.append(best)
    counts = Counter(assigned)
    top_cluster, top_size = counts.most_common(1)[0]
    modal_idx = next(i for i, a in enumerate(assigned) if a == top_cluster)
    return completions[modal_idx], top_size / len(completions)


# ---- answerable factual probe ------------------------------------------------

def _load_probe(n_probes: int, seed: int) -> list[dict[str, str]]:
    """(prompt, concept) items drawn round-robin across factual relations."""
    rng = random.Random(seed)
    pools: dict[str, list[dict[str, str]]] = {}
    for rel in RELATIONS:
        seen: dict[str, dict[str, str]] = {}
        with open(FACTUAL / f"{rel}.tsv", newline="") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                s, o, p = row.get("subject"), row.get("object"), row.get("target_baseline")
                if s and o and p and s not in seen:
                    seen[s] = {"prompt": p.strip(), "concept": o.strip()}
        vals = list(seen.values())
        rng.shuffle(vals)
        pools[rel] = vals
    probe: list[dict[str, str]] = []
    i = 0
    rels = [r for r in RELATIONS if pools[r]]
    while len(probe) < n_probes and rels:
        rel = rels[i % len(rels)]
        if pools[rel]:
            probe.append(pools[rel].pop())
        else:
            rels.remove(rel)
        i += 1
    return probe[:n_probes]


def _build_datapoints(probe, tokenizer, layers, k):
    """k identical AO datapoints per probe item (k samples → a confidence distribution)."""
    from nl_probes.utils.dataset_utils import create_training_datapoint

    dps = []
    for item in probe:
        ids = tokenizer(item["prompt"], add_special_tokens=False)["input_ids"]
        if not ids:
            continue
        positions = list(range(max(0, len(ids) - 3), len(ids)))  # last ≤3 tokens carry the prediction
        base = create_training_datapoint(
            datapoint_type="cot_oracle_convqa", prompt=AO_QUESTION, target_response=item["concept"],
            layers=layers, num_positions=len(positions), tokenizer=tokenizer, acts_BD=None,
            feature_idx=-1, context_input_ids=ids, context_positions=positions, ds_label=None, meta_info={})
        dps.extend(base.model_copy(deep=True) for _ in range(k))
    return dps


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--lora", default=None, help="AO LoRA path (default: this EXP's final checkpoint)")
    p.add_argument("--n-probes", type=int, default=None, help="override config calibration.n_probes")
    args = p.parse_args(argv)

    from peft import PeftModel

    from nl_probes.utils.common import load_model, load_tokenizer
    from nl_probes.utils.eval import run_evaluation
    from nl_probes.utils.activation_utils import get_hf_submodule

    cfg = load_config()
    cal = cfg["calibration"]
    model_name = cfg["model"]["name"]
    k = int(cal["k_samples"])
    n_probes = args.n_probes or int(cal["n_probes"])
    lora = args.lora or str(artifacts_dir(model_name) / "checkpoints" / run_dir_name(model_name) / "final")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16
    tokenizer = load_tokenizer(model_name)
    layers, _ = resolve_layers(model_name, cfg["layers"]["center_percent"], cfg["layers"]["n_layers"])

    probe = _load_probe(n_probes, cfg["training"]["seed"])
    dps = _build_datapoints(probe, tokenizer, layers, k)
    n_items = len(dps) // k
    print(f"[calibration] {n_items} probes × k={k} samples through {lora}")

    base = load_model(model_name, dtype)
    aomodel = PeftModel.from_pretrained(base, lora, is_trainable=False).eval()
    submodule = get_hf_submodule(aomodel, cfg["injection"]["hook_onto_layer"])
    results = run_evaluation(
        eval_data=dps, model=aomodel, tokenizer=tokenizer, submodule=submodule, device=device, dtype=dtype,
        global_step=0, lora_path=None, eval_batch_size=cfg["eval"].get("batch_size", 32),
        steering_coefficient=cfg["injection"]["eval_steering_coefficient"],
        generation_kwargs={"do_sample": True, "temperature": float(cal["temperature"]),
                           "top_p": 0.95, "max_new_tokens": 24})
    responses = [r.api_response for r in results]

    del aomodel, base
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    # Cluster each item's k samples → (confidence, modal answer); judge by embedder similarity.
    embedder = _SentenceEmbedder(cal["embed_model"], device)
    confidences, modal_answers, concepts = [], [], []
    for i, item in enumerate(probe[:n_items]):
        samples = responses[i * k : (i + 1) * k]
        modal, freq = mode_frequency(samples, embedder, float(cal["cluster_threshold"]))
        confidences.append(freq)
        modal_answers.append(modal)
        concepts.append(item["concept"])
    sims = embedder.cos_to_target(modal_answers, concepts)
    correct = [s >= float(cal["correct_threshold"]) for s in sims]

    ece = expected_calibration_error(confidences, correct)
    acc = sum(correct) / max(len(correct), 1)
    mean_conf = sum(confidences) / max(len(confidences), 1)

    # Reliability table (confidence bin → accuracy), the gears behind ECE.
    n_bins = 5
    rows = []
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        idx = [j for j, c in enumerate(confidences) if (lo <= c < hi or (b == n_bins - 1 and c == hi))]
        if idx:
            rows.append((f"{lo:.1f}-{hi:.1f}", len(idx),
                         sum(confidences[j] for j in idx) / len(idx),
                         sum(correct[j] for j in idx) / len(idx)))

    print(f"\n[calibration] {model_slug(model_name)} / {experiment_name() or 'main'}")
    print(f"  accuracy={acc:.3f}  mean_confidence={mean_conf:.3f}  ECE={ece:.3f}  (k={k}, n={len(correct)})")
    print(f"  {'conf-bin':>10} {'count':>6} {'mean-conf':>10} {'accuracy':>9}")
    for label, cnt, mc, ba in rows:
        print(f"  {label:>10} {cnt:>6} {mc:>10.3f} {ba:>9.3f}")

    out_dir = artifacts_dir(model_name) / "calibration"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{experiment_name() or 'main'}.json"
    out_path.write_text(json.dumps({
        "lora": lora, "k_samples": k, "n_items": len(correct),
        "accuracy": acc, "mean_confidence": mean_conf, "ece": ece,
        "reliability": [{"bin": l, "count": c, "mean_conf": mc, "accuracy": a} for l, c, mc, a in rows],
        "items": [{"prompt": probe[i]["prompt"], "concept": concepts[i], "modal_answer": modal_answers[i],
                   "confidence": confidences[i], "correct": correct[i]} for i in range(len(correct))],
    }, indent=2))
    print(f"\n[calibration] wrote {out_path}")


if __name__ == "__main__":
    main()
