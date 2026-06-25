"""Build a self-contained HTML dashboard comparing a target AO's AObench run
against the paper's released AO.

Gears-level overview
--------------------
`make eval` writes per-task results for our trained AO into
  artifacts/<model>/aobench_results/main/
and `make eval-baseline` writes the paper's released AO into a sibling
  artifacts/<model>/aobench_results/baseline_<slug>/
Each run dir holds, per task, a `*_summary.json` (aggregate metrics) and a
per-example file (`<task>[_binary]_<verbalizer-slug>.json`) with the raw
ground-truth / AO-response / score for every item.

This module reads BOTH runs and emits one offline HTML file with four layers,
in order of increasing detail:
  1. an interactive normalized bar chart of each task's headline metric (ours
     vs paper), so the whole comparison is legible at a glance;
  2. interactive ROC overlays (ours vs paper on the same axes), recomputed from
     the per-example yes/no logit margins for the discriminative tasks;
  3. a full metric table per task (every metric in the summary, ours/paper/Δ) —
     nothing is dropped — each row annotated by the rubric;
  4. direct example cards: the same item's prompt, ground truth, our AO's answer
     and the paper AO's answer, side by side, selected to surface disagreements.
A metric rubric/glossary (compiled from the task spec) explains exactly what
each number measures and how it is computed.

The chart layer uses Plotly; plotly.js is inlined once so the file works
offline. Everything (data, JS, styling) lives in the single output HTML.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import random
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from . import ARTIFACTS, AOBENCH_ROOT, load_config, model_slug, experiment_name

# =============================================================================
# 1. Task & metric specification
#
# This single structure drives the metric tables, the normalization for the
# headline chart, AND the on-page rubric. For each task we list its metrics in
# display order. `kind` controls formatting + normalization:
#   rate   -> in [0,1], shown as 0.xxx, normalized as-is
#   auc    -> in [0,1], chance 0.5, normalized (v-0.5)/0.5  (0 = chance)
#   scale5 -> LLM-judge 1..5, shown as x.xx/5, normalized (v-1)/4
# `src` says where to read it from the summary dict (most live in
# overall_metrics; mmlu's pre/post/letter numbers live under mode_results).
# The first metric in each list is the task's HEADLINE metric (the one charted).
# =============================================================================

Metric = dict[str, Any]


def _m(key: str, label: str, kind: str, desc: str, hib: bool = True, src: str = "overall",
       nkey: "str | list[str] | None" = None) -> Metric:
    # `nkey` names the count field for this metric's denominator (so the table
    # can show n). A list is summed (e.g. positives+negatives for a pairwise
    # AUC). None falls back to the task's overall item count.
    return {"key": key, "label": label, "kind": kind, "desc": desc, "hib": hib, "src": src, "nkey": nkey}


TASKS: dict[str, dict[str, Any]] = {
    "number_prediction": {
        "title": "Predict the hidden number",
        "tests": "From mid-computation activations, can the AO name the exact number the base "
                 "model is about to output? (open-ended integer — chance ≈ 0)",
        "metrics": [
            _m("matches_model_answer_rate", "Matches model's number", "rate",
               "Fraction of items where the AO's predicted number equals the number the base model "
               "actually produced next. The headline 'reads the model's intended output' metric."),
            _m("matches_true_answer_rate", "Matches true answer", "rate",
               "Fraction where the AO's number equals the mathematically correct answer."),
            _m("has_number_rate", "Produced a number", "rate",
               "Fraction of AO responses containing any parseable number (coverage / sanity)."),
            _m("single_token_model_match_rate", "Match · single-token answers", "rate",
               "Model-match rate restricted to items whose answer is a single token (easier)."),
            _m("multi_token_model_match_rate", "Match · multi-token answers", "rate",
               "Model-match rate restricted to multi-token answers (harder)."),
            _m("cat_simple_2op_model_match_rate", "Match · simple 2-op", "rate",
               "Model-match rate on simple two-operand expressions.", nkey="cat_simple_2op_total"),
            _m("cat_medium_3_4op_model_match_rate", "Match · 3–4-op", "rate",
               "Model-match rate on medium 3–4-operand expressions.", nkey="cat_medium_3_4op_total"),
            _m("cat_large_numbers_model_match_rate", "Match · large numbers", "rate",
               "Model-match rate when operands are large.", nkey="cat_large_numbers_total"),
            _m("cat_nested_model_match_rate", "Match · nested", "rate",
               "Model-match rate on nested expressions.", nkey="cat_nested_total"),
            _m("cat_divmod_model_match_rate", "Match · div/mod", "rate",
               "Model-match rate on division/modulo expressions.", nkey="cat_divmod_total"),
        ],
    },
    "mmlu_prediction": {
        "title": "Predict MMLU correctness",
        "tests": "Reading the model's activations on an MMLU question, can the AO say yes/no whether "
                 "the model will answer correctly? Scored by the yes−no logit margin → ROC-AUC.",
        "metrics": [
            _m("roc_auc", "ROC-AUC (overall)", "auc",
               "Area under the ROC curve for the yes/no 'will the model get this right?' probe, using "
               "the yes-minus-no logit margin as the score. 0.5 = chance, 1.0 = perfect ranking."),
            _m("roc_auc", "ROC-AUC · pre-answer", "auc",
               "AUC when the AO reads activations BEFORE the model has written its answer.", True, "pre"),
            _m("roc_auc", "ROC-AUC · post-answer", "auc",
               "AUC when the AO reads activations AFTER the answer is in context (easier).", True, "post"),
            _m("accuracy_at_zero", "Accuracy @ margin 0", "rate",
               "Accuracy if the yes/no decision thresholds the margin at zero."),
            _m("mean_margin_when_yes", "Mean margin · correct items", "rate",
               "Average yes−no logit margin on items the model truly got right (separation; higher good).",
               nkey="num_positive"),
            _m("mean_margin_when_no", "Mean margin · wrong items", "rate",
               "Average yes−no margin on items the model truly got wrong (lower/more-negative is good).", False,
               nkey="num_negative"),
            _m("matches_model_rate", "Letter matches model", "rate",
               "When asked to NAME the letter the model will choose, fraction matching the model's actual "
               "choice (pre-answer mode).", True, "letter"),
            _m("matches_true_rate", "Letter matches truth", "rate",
               "Fraction where the AO's named letter is the correct answer (pre-answer mode).", True, "letter"),
        ],
    },
    "backtracking": {
        "title": "Explain the model's uncertainty",
        "tests": "At a reasoning step, can the AO describe (in words) what the model is uncertain or "
                 "confused about? An LLM judge rates the description vs a reference (1–5).",
        "metrics": [
            _m("mean_correctness", "Mean correctness", "scale5",
               "Judge's 1–5 rating of how accurately the AO described the model's uncertainty vs the "
               "reference. 1 = wrong/hallucinated, 5 = fully correct."),
            _m("mean_specificity", "Mean specificity", "scale5",
               "Judge's 1–5 rating of how specific (vs generic/hedged) the AO's description is."),
            _m("correctness_>=3_rate", "Correctness ≥ 3 rate", "rate",
               "Fraction of items scoring at least 3/5 on correctness."),
            _m("correctness_>=4_rate", "Correctness ≥ 4 rate", "rate",
               "Fraction scoring at least 4/5 on correctness (strict)."),
            _m("specificity_>=3_rate", "Specificity ≥ 3 rate", "rate",
               "Fraction scoring at least 3/5 on specificity."),
            _m("specificity_>=4_rate", "Specificity ≥ 4 rate", "rate",
               "Fraction scoring at least 4/5 on specificity (strict)."),
        ],
    },
    "missing_info": {
        "title": "Detect missing information",
        "tests": "Can the AO tell, from activations, whether the model has enough info to answer or is "
                 "missing something? Yes/no probe scored by logit margin → ROC-AUC.",
        "metrics": [
            _m("roc_auc", "ROC-AUC", "auc",
               "AUC for the yes/no 'is the model missing information?' probe (margin-scored). 0.5 = chance."),
            _m("accuracy_at_zero", "Accuracy @ margin 0", "rate",
               "Accuracy thresholding the yes/no margin at zero."),
            _m("A_vs_B_roc_auc", "AUC · complete vs incomplete", "auc",
               "Discrimination between the complete-info condition (A) and naturally-incomplete (B).",
               nkey=["A_vs_B_num_positive", "A_vs_B_num_negative"]),
            _m("A_vs_C_roc_auc", "AUC · complete vs forced", "auc",
               "Discrimination between complete (A) and forced-incomplete (C) conditions.",
               nkey=["A_vs_C_num_positive", "A_vs_C_num_negative"]),
            _m("cond_A_complete_accuracy_at_zero", "Accuracy · complete (A)", "rate",
               "Accuracy on the complete-information items (should say 'no, nothing missing').",
               nkey="cond_A_complete_total"),
            _m("cond_B_incomplete_accuracy_at_zero", "Accuracy · incomplete (B)", "rate",
               "Accuracy on naturally-incomplete items (should say 'yes, missing').",
               nkey="cond_B_incomplete_total"),
            _m("cond_C_forced_accuracy_at_zero", "Accuracy · forced (C)", "rate",
               "Accuracy on forced-incomplete items.", nkey="cond_C_forced_total"),
        ],
    },
    "vagueness": {
        "title": "Response specificity",
        "tests": "When forced to state the model's current answer, does the AO give a SPECIFIC answer "
                 "(an actual value/option) rather than a vague hedge? Judge-classified.",
        "metrics": [
            _m("specificity_rate", "Specific & correct rate", "rate",
               "Fraction of AO responses the judge marks specific AND correct (names the real answer), "
               "vs vague. The headline non-vagueness metric."),
            _m("vagueness_rate", "Vagueness rate", "rate",
               "Fraction judged vague (= 1 − specificity_rate). Lower is better.", False),
            _m("generic_rate", "Generic rate", "rate",
               "Share of responses that are generic boilerplate with no concrete answer. Lower better.", False),
            _m("vague_directional_rate", "Vague-directional rate", "rate",
               "Share that gesture at a direction but name no specific answer. Lower better.", False),
            _m("refusal_rate", "Refusal rate", "rate",
               "Share where the AO refused / produced nothing usable. Lower better.", False),
        ],
    },
    "domain_confusion": {
        "title": "Identify the problem domain",
        "tests": "Can the AO name what specific problem / domain the model is working on, from its "
                 "activations? Judge checks the named domain against ground truth.",
        "metrics": [
            _m("domain_correct_specific_rate", "Correct domain + specific", "rate",
               "Fraction where the AO names the correct domain AND a specific problem. Headline accuracy."),
            _m("domain_correct_vague_rate", "Correct domain, vague", "rate",
               "Right domain but no specific problem identified."),
            _m("domain_wrong_rate", "Wrong domain", "rate",
               "Fraction where the AO named the wrong domain (confusion). Lower is better.", False),
            _m("domain_confusion_rate", "Domain-confusion rate", "rate",
               "Overall rate of getting the domain wrong (= wrong-domain share). Lower better.", False),
            _m("refusal_rate", "Refusal rate", "rate",
               "Share where the AO refused / gave nothing usable. Lower better.", False),
        ],
    },
    "activation_sensitivity": {
        "title": "Not Just Reading Tokens",
        "tests": "Fed the SAME tokens from two different upstream states (missing_info A vs C), "
                 "does the AO answer differently in a way that reflects the state change? The "
                 "cleanest test that the AO reads activations, not surface text. Judge-classified.",
        "metrics": [
            _m("activation_sensitivity", "Activation sensitivity", "rate",
               "Fraction of A/C pairs the judge marks 'divergent_meaningful' (the AO's two answers "
               "differ in a way that tracks the hidden state change). The headline inversion-"
               "resistance metric."),
            _m("same_rate", "Same-answer rate", "rate",
               "Fraction where the AO gave essentially the same answer for both states — the "
               "text-inversion failure mode. Lower is better.", False),
            _m("divergent_noise_rate", "Divergent-but-noise rate", "rate",
               "Fraction that differ but not in a state-relevant way (random variation). Lower better.", False),
        ],
    },
    "model_diffing": {
        "title": "Model diffing",
        "tests": "Injected the activation DIFFERENCE between a finetuned variant and its base, can "
                 "the AO describe how the variant behaves differently? Judge-checked against the "
                 "variant's known behaviour. (C5 — present only on runs with a variant family.)",
        "metrics": [
            _m("model_diffing_accuracy", "Diff-description accuracy", "rate",
               "Fraction of (variant × context) items where the judge says the AO correctly named the "
               "variant's actual behaviour difference. Headline model-diffing metric."),
        ],
    },
    "causal_faithfulness": {
        "title": "Causal faithfulness (CFE)",
        "tests": "Does the AO's description of an activation correctly PREDICT the effect of patching "
                 "that activation into a different prompt? A faithful description names the concept that "
                 "transfers; a vague one scores ~0. (C6 — eval-only.)",
        "metrics": [
            _m("causal_faithfulness", "Causal faithfulness", "rate",
               "Fraction of minimal pairs where the judge says the AO's description of activation a_A "
               "predicts the observed shift when a_A is patched into prompt B. Headline metric."),
            _m("patch_effect_rate", "Patch had an effect", "rate",
               "Fraction of pairs where patching measurably changed the output (sanity: if ~0 the patch "
               "did nothing and faithfulness is undefined)."),
        ],
    },
    "abstention": {
        "title": "Abstention (answer vs. don't-know)",
        "tests": "Does the AO answer when the activation contains the answer and ABSTAIN when it doesn't "
                 "(question mismatched to the activation)? Calibrated behaviour the C2 GRPO trains for; "
                 "baseline AOs tend to over-answer.",
        "metrics": [
            _m("abstention_f1", "Abstention F1", "rate",
               "F1 of abstaining on the unanswerable (mismatched) items. Headline metric — balances "
               "abstaining when you should against not abstaining spuriously."),
            _m("unanswerable_abstain_rate", "Abstains when unanswerable", "rate",
               "Recall: fraction of mismatched items where the AO abstained (higher is better after C2)."),
            _m("answerable_answer_rate", "Answers when answerable", "rate",
               "Fraction of matched items where the AO did NOT abstain (guards against over-abstention)."),
            _m("balanced_accuracy", "Balanced accuracy", "rate",
               "Mean of answer-when-answerable and abstain-when-unanswerable."),
        ],
    },
}

# Per-task glob (within a run dir) to the per-example file, and the key holding
# the row list. The filename's verbalizer slug differs per run, so we glob.
RECORD_FILES: dict[str, str] = {
    "number_prediction": "number_prediction/number_prediction_*.json",
    "mmlu_prediction": "mmlu_prediction/pre_answer/mmlu_prediction_binary_*.json",
    "backtracking": "backtracking/backtracking_*.json",
    "missing_info": "missing_info/missing_info_binary_*.json",
    "vagueness": "vagueness/vagueness_*.json",
    "domain_confusion": "domain_confusion/domain_confusion_*.json",
    # contribution evals (Phase 1/2) — qualitative cards from their per-example files
    "causal_faithfulness": "causal_faithfulness/causal_faithfulness_*.json",
    "abstention": "abstention/abstention_*.json",
    "model_diffing": "model_diffing/model_diffing_*.json",
    "activation_sensitivity": "activation_sensitivity/activation_sensitivity_*.json",
}

# Source eval datasets (relative to AOBENCH_ROOT) that hold the human-readable
# question/problem text. The per-example result files only store ids/metadata,
# so we join these back in to make the comparison cards self-explanatory.
# `key` is the field to index by — it must match the join key the corresponding
# extractor produces (id for the binary tasks; the uncertainty text for
# backtracking, whose result rows carry no id).
DATASET_FILES: dict[str, tuple[str, str]] = {
    "mmlu_prediction": ("AObench/datasets/mmlu_prediction/mmlu_prediction_eval_dataset.json", "id"),
    "missing_info": ("AObench/datasets/missing_info/missing_info_eval_dataset.json", "id"),
    "backtracking": ("AObench/datasets/backtracking/backtracking_eval_dataset.json", "uncertainty_description"),
}

OURS_COLOR = "#2A6FDF"
BASE_COLOR = "#7C9C59"


# =============================================================================
# 2. Loaders (slug-agnostic)
# =============================================================================

def _load_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def load_summaries(run_dir: Path) -> dict[str, Any]:
    """task -> summary dict, from all_summaries.json (falls back to *_summary.json)."""
    allp = run_dir / "all_summaries.json"
    if allp.exists():
        return _load_json(allp) or {}
    out = {}
    for f in run_dir.glob("*_summary.json"):
        out[f.stem.replace("_summary", "")] = _load_json(f)
    return out


def _run_verbalizer_slug(run_dir: Path, task: str) -> str | None:
    """The verbalizer slug this run REPORTS for `task` — the single key under the
    summary's `metrics_by_verbalizer`. Used to disambiguate the per-example file when
    stray results from a DIFFERENT checkpoint linger in the same run dir (e.g. a
    leftover `<task>_final.json` from an earlier AO sitting next to the intended
    `<task>_replication_v1.json`). Without this, glob order silently decides which
    model's examples are shown — divorcing the example cards from the headline metrics."""
    s = _load_json(run_dir / f"{task}_summary.json") or {}
    keys = list((s.get("metrics_by_verbalizer") or {}).keys())
    return keys[0] if keys else None


def load_records(run_dir: Path, task: str) -> list[dict[str, Any]]:
    """Per-example rows for a task, regardless of the verbalizer slug in the name.

    Metrics-only tasks (no ROC/example-card support, e.g. activation_sensitivity)
    are absent from RECORD_FILES — they render as a metric table + headline bar
    only, so we return [] rather than KeyError."""
    if task not in RECORD_FILES:
        return []
    pattern = RECORD_FILES[task]
    matches = [p for p in run_dir.glob(pattern) if "_summary" not in p.name]
    if not matches:
        return []
    # Pick the file that matches the summary's verbalizer (so cards == metrics);
    # fall back to first match only when the slug can't be resolved.
    slug = _run_verbalizer_slug(run_dir, task)
    chosen = next((p for p in matches if slug and p.stem.endswith(f"_{slug}")), matches[0])
    data = _load_json(chosen) or {}
    # `pairs` is the activation_sensitivity per-example key; the contribution
    # evals (CFE, abstention, model_diffing) all use `scored_results`.
    return (data.get("scored_results") or data.get("binary_scored_results")
            or data.get("pairs") or [])


def load_dataset_index(task: str) -> dict[str, Any]:
    """Build the per-task source index cards use to show the real context.

    Two shapes, by task:
      * mmlu/missing_info/backtracking -> {join_key -> source entry}, indexed by
        the field named in DATASET_FILES[task] (the same key the extractor emits).
      * vagueness/domain_confusion -> {"__entries__": ordered backtracking
        entries, "__factor__": prompts-per-entry}. These two tasks are *built
        from* the backtracking rollouts, so the reasoning trace lives there; a
        row's `result_index` is its position in the ordered (entries × prompts)
        build, hence entries[result_index // factor] recovers its exact context.
        factor is structural: 1 for vagueness (one sampled prompt per entry),
        len(DOMAIN_PROMPTS) for domain_confusion.
    Returns {} when the source file is absent.
    """
    if task in ("vagueness", "domain_confusion"):
        rel = DATASET_FILES["backtracking"][0]
        data = _load_json(AOBENCH_ROOT / rel) or {}
        entries = data.get("entries", []) if isinstance(data, dict) else data
        return {"__entries__": entries, "__factor__": 1 if task == "vagueness" else len(DOMAIN_PROMPTS)}
    spec = DATASET_FILES.get(task)
    if not spec:
        return {}
    rel, key = spec
    data = _load_json(AOBENCH_ROOT / rel) or {}
    entries = data.get("entries", []) if isinstance(data, dict) else data
    return {str(e[key]): e for e in entries if key in e}


def detect_runs(results_root: Path, target_run: str, baseline_run: str | None) -> tuple[str, str | None]:
    """Resolve the target run dir name and auto-find the baseline (dir starting 'baseline')."""
    if baseline_run is None:
        cands = sorted(d.name for d in results_root.iterdir()
                       if d.is_dir() and d.name.startswith("baseline"))
        baseline_run = cands[0] if cands else None
    return target_run, baseline_run


# =============================================================================
# 3. Metric reading + normalization
# =============================================================================

def _first(d: dict | None) -> dict:
    """Return the single per-verbalizer entry (the slug key differs per run)."""
    if not d:
        return {}
    return next(iter(d.values()), {}) if isinstance(d, dict) else {}


def metric_value(summary: dict | None, m: Metric) -> float | None:
    """Read one metric from a task summary, following its `src` location."""
    if not summary:
        return None
    src, key = m["src"], m["key"]
    if src == "overall":
        return summary.get("overall_metrics", {}).get(key)
    if src in ("pre", "post"):
        mode = "pre_answer" if src == "pre" else "post_answer"
        return summary.get("mode_results", {}).get(mode, {}).get("overall_metrics", {}).get(key)
    if src == "letter":
        lp = summary.get("mode_results", {}).get("pre_answer", {}).get("letter_prediction_by_verbalizer", {})
        return _first(lp).get(key)
    return None


def _metric_block(summary: dict | None, m: Metric) -> dict:
    """The summary sub-dict a metric lives in, per its `src` (see metric_value)."""
    if not summary:
        return {}
    src = m["src"]
    if src == "overall":
        return summary.get("overall_metrics", {})
    if src in ("pre", "post"):
        mode = "pre_answer" if src == "pre" else "post_answer"
        return summary.get("mode_results", {}).get(mode, {}).get("overall_metrics", {})
    if src == "letter":
        return _first(summary.get("mode_results", {}).get("pre_answer", {}).get("letter_prediction_by_verbalizer", {}))
    return {}


def metric_n(summary: dict | None, m: Metric) -> int | None:
    """Number of items a metric was computed over (its denominator).

    Reads the metric's `src` block (same locations as metric_value), then takes
    the metric's `nkey` count field (summing a list); failing that, the block's
    own item count (total/num_scored/num_entries); failing that, the task-level
    count. Lets the table flag small-n metrics whose deltas are noise-prone.
    """
    if not summary:
        return None
    d = _metric_block(summary, m)
    nkey = m.get("nkey")
    if isinstance(nkey, list):
        vals = [d[k] for k in nkey if k in d]
        if vals:
            return int(sum(vals))
    elif nkey and nkey in d:
        return int(d[nkey])
    for k in ("total", "num_scored", "num_entries"):
        if d.get(k) is not None:
            return int(d[k])
    om = summary.get("overall_metrics", {})
    for k in ("total", "num_scored", "num_entries"):
        if om.get(k) is not None:
            return int(om[k])
    n = summary.get("num_scored") or summary.get("num_entries")
    return int(n) if n is not None else None


# Judge tasks whose rows are many probe POSITIONS over a small pool of source
# problems (vagueness/domain are derived from backtracking's ~29 problems). The
# stored n (probe count) overstates statistical independence — 39 probes of the
# same problem are highly correlated — so we report the distinct-problem count as
# the EFFECTIVE n and size confidence intervals from it (each problem ~ one unit,
# a conservative ceiling on precision). backtracking itself isn't listed: its rows
# are ~190 distinct ground truths, so its raw n is already honest.
_CLUSTERED_TASKS: dict[str, tuple[str, ...]] = {
    "vagueness": ("problem_id", "problem"),
    "domain_confusion": ("problem_id", "problem"),
}


def effective_n(task: str, recs: list[dict] | None) -> int | None:
    """Distinct source-problem count for a clustered judge task (else None)."""
    keys = _CLUSTERED_TASKS.get(task)
    if not keys or not recs:
        return None
    for k in keys:
        vals = {r.get(k) for r in recs if r.get(k) is not None}
        if vals:
            return len(vals)
    return None


def normalize(kind: str, v: float) -> float:
    """Map a metric to a common 'higher=better, 0≈chance, 1=perfect' axis."""
    if kind == "scale5":
        return (v - 1.0) / 4.0
    if kind == "auc":
        return (v - 0.5) / 0.5
    return v


def fmt(kind: str, v: float | None) -> str:
    if v is None:
        return "—"
    if kind == "scale5":
        return f"{v:.2f}/5"
    return f"{v:.3f}"


# --- Uncertainty -------------------------------------------------------------
# Given small eval sets (some tasks have n<50), a bare point estimate invites
# over-reading noise. We attach an approximate standard error to every metric so
# the chart can draw 95% CIs and the table can flag deltas that are within noise.
# Per metric kind:
#   rate   -> binomial SE  sqrt(p(1-p)/n)
#   auc    -> Hanley–McNeil SE (uses the metric block's num_positive/num_negative)
#   scale5 -> conservative Popoviciu bound (max sd of a value in [1,5] is 2)
_Z = 1.96  # ~95% normal critical value

# normalized-axis scale per kind: normalize() is linear, so a raw SE maps to the
# normalized axis by multiplying by |d(normalized)/d(raw)| = these factors.
_NORM_SCALE = {"auc": 2.0, "scale5": 0.25, "rate": 1.0}


def metric_se(summary: dict | None, m: Metric, v: float | None, n: int | None) -> float | None:
    """Approximate standard error of a metric value, in its RAW units."""
    if v is None or not n or n < 2:
        return None
    kind = m["kind"]
    if kind == "auc":
        d = _metric_block(summary, m)
        P, N = d.get("num_positive"), d.get("num_negative")
        if not P or not N:
            return None
        A = min(max(v, 1e-6), 1 - 1e-6)
        q1, q2 = A / (2 - A), 2 * A * A / (1 + A)
        var = (A * (1 - A) + (P - 1) * (q1 - A * A) + (N - 1) * (q2 - A * A)) / (P * N)
        return math.sqrt(max(var, 0.0))
    if kind == "scale5":
        return 2.0 / math.sqrt(n)
    p = min(max(v, 0.0), 1.0)
    return math.sqrt(p * (1 - p) / n)


def headline_halfwidth(summary: dict | None, m: Metric, n_override: int | None = None) -> float:
    """95% CI half-width of a metric on the NORMALIZED axis (0 if unknown).

    `n_override` lets clustered tasks pass their effective (distinct-problem) n so
    the whisker reflects the real, smaller sample rather than the probe count.
    """
    v = metric_value(summary, m)
    n = n_override or metric_n(summary, m)
    se = metric_se(summary, m, v, n)
    return _Z * se * _NORM_SCALE.get(m["kind"], 1.0) if se is not None else 0.0


# --- Paired bootstrap CI on the headline ours−base delta ---------------------
# The table's analytic delta SE = sqrt(se_o^2 + se_b^2) treats the two AOs as
# INDEPENDENT draws. They aren't: both are scored on the SAME items, so the real
# variance of the delta is smaller (the shared per-item difficulty cancels). We
# therefore bootstrap over items: resample the shared item set with replacement,
# recompute BOTH AOs' headline statistic on that same resample, and read the
# 2.5/97.5 percentiles of (ours − base). A CI excluding 0 ⇒ a real gap at ~95%.
# This is what makes "is this delta reliable?" answerable from existing data,
# independent of brute-forcing n.

# Tasks where we can recompute the headline statistic from per-item records.
_HL_BOOTSTRAP_TASKS = {"causal_faithfulness", "number_prediction", "vagueness",
                       "domain_confusion", "activation_sensitivity", "backtracking", "abstention"}


def _hl_payload(task: str, r: dict):
    """Per-item contribution to the headline statistic (None if unscorable)."""
    c = r.get("category")
    if task == "causal_faithfulness":    return 1.0 if r.get("correct") else 0.0
    if task == "number_prediction":      return 1.0 if r.get("matches_model_answer") else 0.0
    if task == "vagueness":              return 1.0 if c == "specific_correct" else 0.0
    if task == "domain_confusion":       return 1.0 if c == "domain_correct_specific" else 0.0
    if task == "activation_sensitivity": return 1.0 if c == "divergent_meaningful" else 0.0
    if task == "backtracking":
        v = r.get("correctness")
        return float(v) if v is not None else None
    if task == "abstention":
        if "abstained" not in r or "answerable" not in r:
            return None
        return (bool(r["answerable"]), bool(r["abstained"]))
    return None


def _hl_stat(task: str, payloads: list):
    """Headline statistic over a list of per-item payloads (rate, mean, or F1)."""
    xs = [p for p in payloads if p is not None]
    if not xs:
        return None
    if task == "abstention":                       # F1 of 'abstain' on unanswerable items
        tp = sum(1 for ans, ab in xs if (not ans) and ab)
        fp = sum(1 for ans, ab in xs if ans and ab)
        fn = sum(1 for ans, ab in xs if (not ans) and (not ab))
        denom = 2 * tp + fp + fn
        return (2 * tp / denom) if denom else 0.0
    return sum(xs) / len(xs)                        # rate (0/1 mean) or mean rating


def _join_paired(recs_o: list, recs_b: list) -> list[tuple[dict, dict]]:
    """Pair items across runs by 'id' when both carry it, else positionally (the
    eval datasets are built identically in identical order, so index is valid)."""
    def keyed(recs):
        if recs and all("id" in r for r in recs):
            return {r["id"]: r for r in recs}
        return {i: r for i, r in enumerate(recs or [])}
    ko, kb = keyed(recs_o), keyed(recs_b)
    return [(ko[k], kb[k]) for k in ko if k in kb]


def paired_bootstrap_delta(task: str, recs_o: list | None, recs_b: list | None,
                           B: int = 2000, seed: int = 0) -> tuple[float, float, float] | None:
    """95% CI of the headline (ours−base) delta via paired item bootstrap.
    Returns (delta, lo, hi) in RAW units, or None if unsupported/insufficient."""
    if task not in _HL_BOOTSTRAP_TASKS or not recs_o or not recs_b:
        return None
    pairs = _join_paired(recs_o, recs_b)
    payo = [_hl_payload(task, o) for o, _ in pairs]
    payb = [_hl_payload(task, b) for _, b in pairs]
    idx = [i for i in range(len(pairs)) if payo[i] is not None and payb[i] is not None]
    if len(idx) < 8:
        return None
    so, sb = _hl_stat(task, [payo[i] for i in idx]), _hl_stat(task, [payb[i] for i in idx])
    if so is None or sb is None:
        return None
    rng = random.Random(seed)
    n, deltas = len(idx), []
    for _ in range(B):
        samp = [idx[rng.randrange(n)] for _ in range(n)]
        a, c = _hl_stat(task, [payo[i] for i in samp]), _hl_stat(task, [payb[i] for i in samp])
        if a is not None and c is not None:
            deltas.append(a - c)
    if not deltas:
        return None
    deltas.sort()
    return (so - sb, deltas[int(0.025 * len(deltas))],
            deltas[min(len(deltas) - 1, int(0.975 * len(deltas)))])


# =============================================================================
# 4. ROC computation (for interactive overlays on the discriminative tasks)
#
# AObench scores binary probes by the yes-minus-no logit margin; the ROC curve
# is the standard sweep of that margin as a threshold. We recompute it from the
# per-example rows so we can draw OUR curve and the PAPER's curve on one axis.
# =============================================================================

def roc_curve(rows: list[dict]) -> tuple[list[float], list[float], float] | None:
    pairs = [(int(r["binary_label"]), float(r["margin_yes_minus_no"]))
             for r in rows if "binary_label" in r and "margin_yes_minus_no" in r]
    if not pairs or len({p[0] for p in pairs}) < 2:
        return None
    pairs.sort(key=lambda p: -p[1])               # high score (yes) first
    P = sum(1 for l, _ in pairs if l == 1)
    N = len(pairs) - P
    if P == 0 or N == 0:
        return None
    tpr, fpr = [0.0], [0.0]
    tp = fp = 0
    for label, _ in pairs:
        tp += label == 1
        fp += label == 0
        tpr.append(tp / P)
        fpr.append(fp / N)
    auc = sum((fpr[i] - fpr[i - 1]) * (tpr[i] + tpr[i - 1]) / 2 for i in range(1, len(tpr)))
    return fpr, tpr, auc


def _bal_acc(pairs: list[tuple[int, float]], t: float) -> float | None:
    """Balanced accuracy = mean(TPR, TNR) for thresholding margin>t ⇒ positive.
    Balanced (not raw) so a biased threshold can't be masked by class imbalance."""
    P = sum(l for l, _ in pairs)
    N = len(pairs) - P
    if P == 0 or N == 0:
        return None
    tp = sum(1 for l, m in pairs if m > t and l == 1)
    tn = sum(1 for l, m in pairs if m <= t and l == 0)
    return 0.5 * (tp / P + tn / N)


def _best_threshold(pairs: list[tuple[int, float]]) -> float:
    """Threshold (a midpoint between sorted margins) that maximises balanced
    accuracy on `pairs`; ties break toward 0 so we don't drift without reason."""
    vals = sorted(m for _, m in pairs)
    if not vals:
        return 0.0
    cands = [vals[0] - 1.0] + [(vals[i] + vals[i + 1]) / 2 for i in range(len(vals) - 1)] + [vals[-1] + 1.0]
    return max(cands, key=lambda t: (_bal_acc(pairs, t) or 0.0, -abs(t)))


def calibration(rows: list[dict], k: int = 5, seed: int = 0) -> dict | None:
    """Quantify (honestly) how much of a binary probe's error is *thresholding*.

    The eval decides yes/no by thresholding the yes−no margin at 0; a well-ranked
    but mis-centred AO loses accuracy purely to that 0. We report three numbers:
      • bal0    — balanced acc at the eval's default 0-threshold (status quo).
      • bal_cv  — balanced acc under STRATIFIED k-fold calibration: for each fold
                  we fit the best threshold on the OTHER folds and score the held-
                  out fold, then pool. This is the *honest* gain a deployable
                  (data-independent-of-test) threshold recovers — no peeking.
      • bal_best— best threshold fit AND scored on the full set: an optimistic
                  upper bound, kept only to show the ceiling.
    best_t is the threshold fit on ALL rows — the one you'd actually deploy.
    Folds are stratified by class so every train split sees both labels.
    """
    pairs = [(int(r["binary_label"]), float(r["margin_yes_minus_no"]))
             for r in rows if "binary_label" in r and "margin_yes_minus_no" in r]
    P = sum(l for l, _ in pairs)
    N = len(pairs) - P
    if P == 0 or N == 0:
        return None

    bal0 = _bal_acc(pairs, 0.0)
    full_t = _best_threshold(pairs)
    bal_best = _bal_acc(pairs, full_t)

    # Stratified k-fold: split positives and negatives separately, round-robin,
    # so each fold keeps the class ratio and no train split is single-class.
    pos = [p for p in pairs if p[0] == 1]
    neg = [p for p in pairs if p[0] == 0]
    rng = random.Random(seed)
    rng.shuffle(pos)
    rng.shuffle(neg)
    kk = max(2, min(k, len(pos), len(neg)))
    folds: list[list[tuple[int, float]]] = [[] for _ in range(kk)]
    for i, p in enumerate(pos):
        folds[i % kk].append(p)
    for i, p in enumerate(neg):
        folds[i % kk].append(p)

    held: list[tuple[int, int]] = []  # (true_label, predicted_label) on held-out folds
    for j in range(kk):
        train = [p for i, f in enumerate(folds) if i != j for p in f]
        t = _best_threshold(train) if 0 < sum(l for l, _ in train) < len(train) else full_t
        held += [(l, 1 if m > t else 0) for l, m in folds[j]]
    Ph = sum(l for l, _ in held)
    Nh = len(held) - Ph
    tp = sum(1 for l, pr in held if pr == 1 and l == 1)
    tn = sum(1 for l, pr in held if pr == 0 and l == 0)
    bal_cv = 0.5 * (tp / Ph + tn / Nh) if Ph and Nh else None

    return {"bal0": bal0, "bal_cv": bal_cv, "bal_best": bal_best,
            "best_t": full_t, "k": kk, "n": len(pairs)}


# =============================================================================
# 5. Direct-comparison extractors
#
# Each returns (join_key, group_key, card) per row.
#   join_key  uniquely identifies one ITEM so ours↔paper line up on the SAME
#             thing (id + verbalizer-prompt variant).
#   group_key identifies the underlying PROBLEM, used only to DEDUPE the cards
#             we show — without it the same problem (e.g. one math question
#             probed under 3 prompt variants, or at several reasoning steps)
#             would fill the panel with near-identical rows.
# Each card cleanly separates the three things a reader needs, so every task
# renders with the same labelled layout:
#   source    – what the BASE model was working on (the captured context)
#   ao_prompt – the exact question posed to the AO, in its original wording
#   target    – the expected/ground-truth answer
# plus per-side `text` (the AO's answer), `verdict`, `ok`, optional `reasoning`.
#
# The verbalizer prompt strings mirror each AObench task's VERBALIZER_PROMPTS;
# result rows store only the prompt's short *name*, so we map name→text here.
# =============================================================================

NUMBER_PROMPTS = {
    "direct": "What number is the model about to output?",
    "planning": "What number is the model planning to answer with?",
    "predict": "Predict the exact numerical answer the model will give.",
}
DOMAIN_PROMPTS = {
    "what_problem": "What specific problem is the model working on?",
    "what_computing": "Describe what the model is computing right now.",
    "what_about": "What is this chain of thought about?",
}


def _letters(choices: list) -> str:
    return "\n".join(f"{chr(65 + i)}) {c}" for i, c in enumerate(choices))


def _trace_entry(r, ds) -> dict:
    """Recover the backtracking source entry a vagueness/domain row came from.

    These tasks are built from backtracking rollouts; the row's `result_index`
    is its position in the ordered (entries × prompts) build, so
    entries[result_index // factor] is the exact context (see load_dataset_index).
    Guarded by a problem-text match; returns {} if the join can't be made.
    """
    entries = ds.get("__entries__") if isinstance(ds, dict) else None
    factor = ds.get("__factor__", 1) if isinstance(ds, dict) else 1
    if entries and "result_index" in r:
        i = r["result_index"] // max(1, factor)
        e = entries[i] if 0 <= i < len(entries) else {}
        if e.get("problem") == r.get("problem"):
            return e
    return {}


# Marker dropped at the end of the reasoning prefix so the reader can see exactly
# where in the trace the AO's activations were read (the last n_positions tokens).
PROBE_MARKER = "  ⟵ activations read here (last n_positions tokens)"
SNIPPET_CHARS = 700  # length of the at-a-glance trace tail shown by default


def _trace_pair(problem: str, prefix: str | None, fallback) -> tuple[str, str | None]:
    """Build (snippet, full) views of the captured reasoning context.

    `snippet` is the always-visible tail (last SNIPPET_CHARS chars of the prefix),
    `full` is the ENTIRE problem+prefix for a collapsible "full CoT" panel — or
    None when the snippet already shows everything (so we don't double-render).
    Both end with PROBE_MARKER so the probe point is unambiguous.
    """
    if not prefix:
        return (str(fallback), None)
    head = f"Problem:  {problem}\n\nThe model's reasoning, up to the probed moment:\n"
    snippet = head + "…" + prefix[-SNIPPET_CHARS:] + PROBE_MARKER
    full = (head + prefix + PROBE_MARKER) if len(prefix) > SNIPPET_CHARS else None
    return (snippet, full)


def _fmt_trace(e: dict, fallback_problem) -> str:
    """Snippet-only trace view (kept for callers that don't need the full panel)."""
    return _trace_pair(e.get("problem", ""), e.get("prefix"), fallback_problem)[0]


def _continuation(e: dict) -> str:
    """What the model ACTUALLY did next from the probed point — the ground-truth
    the AO is trying to predict (revise? abandon? what answer?). Shown in full
    inside a scrollable, collapsed panel, so no continuation is lost."""
    return (e.get("original_continuation") or "").strip()


def _ex_number(r, ds):
    n = r.get("ao_number")
    return (f"{r['id']}|{r['prompt_name']}", str(r["id"]), {
        "context": f"number_prediction · prompt “{r['prompt_name']}”",
        "source": str(r["expression"]),
        "ao_prompt": NUMBER_PROMPTS.get(r["prompt_name"], r["prompt_name"]),
        "target": f"the model was about to output {r['model_answer']}  "
                  f"(mathematically correct answer: {r['true_answer']})",
        "text": f"“{r.get('ao_response','')}”   → parsed: {n}",
        "verdict": f"matches the model's number ({n})" if r.get("matches_model_answer")
                   else f"wrong number — said {n}",
        "ok": bool(r.get("matches_model_answer")),
    })


def _ex_mmlu(r, ds):
    e = ds.get(str(r["id"]), {})
    q = e.get("question", "")
    if e.get("choices"):
        q += "\n\n" + _letters(e["choices"])
    pred = r["predicted_answer"]
    return (str(r["id"]), str(r["id"]), {
        "context": f"MMLU · {r['subject']} · the model chose {r['model_answer_letter']}, "
                   f"correct answer is {r['correct_answer_letter']}",
        "source": q,
        "ao_prompt": "Will the model answer this question correctly?  (yes / no)",
        "target": f"“{r['ground_truth']}” — the model's answer ({r['model_answer_letter']}) was "
                  f"{'CORRECT' if r['ground_truth']=='yes' else 'WRONG'}",
        "text": f"predicts the model is {'RIGHT' if pred=='yes' else 'WRONG'} "
                f"(“{pred}”, yes−no margin {r['margin_yes_minus_no']:+.2f})",
        "verdict": "correct" if r.get("is_correct") else "wrong",
        "ok": bool(r.get("is_correct")),
    })


def _ex_missing(r, ds):
    e = ds.get(str(r["id"]), {})
    q = e.get("problem_text", "")
    if r.get("missing_info_description"):
        q += f"\n\n(potentially missing: {r['missing_info_description']})"
    pred = r["predicted_answer"]
    # group A/B/C variants of the same base problem so we don't show near-dups
    group = re.sub(r"_[ABC]$", "", str(r["id"]))
    return (str(r["id"]), group, {
        "context": f"condition {r['condition']} · id {r['id']}",
        "source": q,
        "ao_prompt": "Does the model have all the information it needs, or is something missing?  (yes / no)",
        "target": f"“{r['ground_truth']}” — {'something IS missing' if r['ground_truth']=='yes' else 'nothing is missing'}",
        "text": f"predicts {'MISSING' if pred=='yes' else 'NOT missing'} "
                f"(“{pred}”, yes−no margin {r['margin_yes_minus_no']:+.2f})",
        "verdict": "correct" if r.get("is_correct") else "wrong",
        "ok": bool(r.get("is_correct")),
    })


def _ex_backtracking(r, ds):
    gt = str(r.get("ground_truth", ""))
    e = ds.get(gt, {})
    problem = e.get("problem", "")
    snippet, full = _trace_pair(problem, e.get("prefix"),
                                f"Problem:  {problem}" if problem else "(source problem unavailable)")
    c = r.get("correctness")
    # group by the underlying problem: one problem has many uncertainty points,
    # all sharing the same `problem` text but different prefixes → show once.
    group = problem or gt[:80]
    return (gt[:200], group, {
        "context": "backtracking",
        "source": snippet,
        "source_full": full,
        "ao_prompt": "What is the model uncertain or confused about at this point?",
        "target": gt,
        "actual": _continuation(e),
        "text": r.get("ao_response", ""),
        "verdict": f"correctness {c}/5 · specificity {r.get('specificity')}/5",
        "ok": bool(c and c >= 3),
        "reasoning": r.get("reasoning", ""),
    })


def _ex_domain(r, ds):
    e = _trace_entry(r, ds)
    snippet, full = _trace_pair(e.get("problem", ""), e.get("prefix"), r.get("problem", ""))
    return (f"{r['problem_id']}|{r['prompt_name']}", str(r["problem_id"]), {
        "context": "domain_confusion",
        "source": snippet,
        "source_full": full,
        "ao_prompt": DOMAIN_PROMPTS.get(r["prompt_name"], r["prompt_name"]),
        "target": f"correct domain: “{r.get('domain','')}” (and ideally the specific problem)",
        "actual": _continuation(e),
        "text": r.get("ao_response", ""),
        "verdict": r.get("category", ""),
        "ok": r.get("category") == "domain_correct_specific",
        "reasoning": r.get("reasoning", ""),
    })


def _ex_vagueness(r, ds):
    e = _trace_entry(r, ds)
    snippet, full = _trace_pair(e.get("problem", ""), e.get("prefix"), r.get("problem", ""))
    return (f"{r['problem_id']}|{r.get('prompt','')}", str(r["problem_id"]), {
        "context": "vagueness",
        "source": snippet,
        "source_full": full,
        "ao_prompt": r.get("prompt", ""),
        "target": "a SPECIFIC answer matching what the model is actually doing in the trace above "
                  "(judge-assessed; no stored gold answer — see what the model did next)",
        "actual": _continuation(e),
        "text": r.get("ao_response", ""),
        "verdict": r.get("category", ""),
        "ok": r.get("category") == "specific_correct",
        "reasoning": r.get("reasoning", ""),
    })


# ---- contribution evals (Phase 1/2) -----------------------------------------
# These probe newer capabilities and store rich per-example rows, so each gets a
# card. They share the same labelled layout; CFE and abstention compare ours↔base
# on a stable id/question, model_diffing on the variant. activation_sensitivity is
# structurally different (one run, two activation states) and is rendered on its own.

def _ex_cfe(r, ds):
    """Causal faithfulness: did the AO's description of a *patched* concept
    predict the shift it caused in a second prompt's completion?"""
    desc = (r.get("description") or "").strip()
    # The CFE eval stores only the concept it could PARSE from the AO's answer;
    # when that parse fails it leaves a dangling fragment (e.g. 'say **"'). Flag it
    # rather than showing the fragment as if it were the AO's concept.
    parsed = desc.rstrip("*\"' ")
    if len(parsed) < 2 or desc.endswith(('**"', '**', '"')):
        desc = "(the AO's concept could not be parsed from its answer)"
    else:
        desc = parsed
    return (str(r["id"]), str(r["id"]), {
        "context": f"causal faithfulness · patched “{r.get('concept_a','')}” into a prompt "
                   f"whose natural answer is “{r.get('concept_b','')}”",
        "source": f"The activation encoding “{r.get('concept_a','')}” was patched into a context "
                  f"that, unpatched, completes to “{r.get('concept_b','')}”.",
        "ao_prompt": "From this activation alone, what concept/entity does it encode?",
        "target": f"a description that names “{r.get('concept_a','')}” (the patched-in concept), "
                  f"predicting the completion will shift toward it",
        "actual": (f"completion BEFORE patch:  “{r.get('baseline','')}”\n"
                   f"completion AFTER patch:   “{r.get('patched','')}”\n"
                   f"→ {'shifted' if r.get('patch_changed') else 'no change'}"),
        "text": f"AO described: “{desc}”",
        "verdict": f"{r.get('category','?')}",
        "ok": bool(r.get("correct")),
        "reasoning": r.get("reasoning", ""),
    })


def _ex_abstention(r, ds):
    """Abstention: when the injected activation does (answerable) or does NOT
    (mismatched) contain the answer, does the AO answer vs. abstain correctly?"""
    answerable = bool(r.get("answerable"))
    abstained = bool(r.get("abstained"))
    correct = (answerable and not abstained) or (not answerable and abstained)
    q = r.get("question", "")
    stmt = r.get("statement", "")
    mismatch = r.get("mismatch_statement")
    # Make the answerable/unanswerable setup legible: show the statement whose
    # activation was injected, then the question — and, when unanswerable, the
    # different statement the borrowed question actually belongs to.
    if stmt:
        src = (f"INJECTED activation encodes: “{stmt}”\n"
               + (f"QUESTION asks about that same fact → the answer IS present."
                  if answerable else
                  f"QUESTION (“{q}”) actually belongs to a DIFFERENT fact "
                  f"(“{mismatch}”) → the answer is NOT present; the AO should abstain."))
    else:
        src = ("(a factual statement's activation is injected; the AO must judge whether it "
               "actually contains what the question asks for)")
    return (f"{stmt}|{q}|{answerable}", f"{answerable}:{q[:60]}", {
        "context": ("abstention · answerable — the injected activation DOES contain the answer"
                    if answerable else
                    "abstention · unanswerable — the injected activation does NOT match the question"),
        "source": src,
        "ao_prompt": q,
        "target": ("answer it (the fact is in the activation)" if answerable else
                   "abstain — say the activation doesn't contain what's asked"),
        "text": r.get("response", ""),
        "verdict": ("abstained" if abstained else "answered") + (" · correct" if correct else " · wrong"),
        "ok": correct,
    })


def _ex_diffing(r, ds):
    """Model diffing: from the (variant − base) activation difference alone, did
    the AO name the finetuned variant's actual behaviour change?"""
    return (f"{r.get('variant','')}|{r.get('response','')[:40]}", str(r.get("variant", "")), {
        "context": f"model diffing · variant “{r.get('variant','')}”",
        "source": "(only the difference between the variant's and base model's activations is injected)",
        "ao_prompt": "How does this model's behaviour differ from the base model it was finetuned from?",
        "target": f"the variant's known change: “{r.get('description','')}”",
        "text": r.get("response", ""),
        "verdict": r.get("category", "?"),
        "ok": bool(r.get("correct")),
        "reasoning": r.get("reasoning", ""),
    })


def actsens_cards(recs_o: list[dict], recs_b: list[dict], base_label: str, n: int = 5) -> str:
    """Dedicated renderer for activation_sensitivity.

    Unlike the ours↔base tasks, this one contrasts TWO activation states of the
    SAME tokens within one run: state A (complete reasoning) vs state C (an
    incomplete/altered upstream state). A text-only reader would answer identically
    for both; an AO that truly reads activations should diverge. We show the two
    answers side by side (for our run) so the divergence is directly visible.
    """
    if not recs_o:
        return '<p class="mut">No activation_sensitivity examples available.</p>'
    # surface the most informative pairs first: those where the two answers differ
    def differs(r):
        return (r.get("response_a") or "").strip() != (r.get("response_c") or "").strip()
    rows = sorted(recs_o, key=lambda r: not differs(r))
    out, seen = [], set()
    for r in rows:
        g = r.get("problem_id")
        if g in seen:
            continue
        seen.add(g)
        div = differs(r)
        ctx = (f'<div class="cardctx">activation sensitivity · {html.escape(str(g))} · '
               f'prompt “{html.escape(str(r.get("prompt_name","")))}”</div>')
        srcbox = (f'<div class="box srcbox"><span class="boxlabel">What the model was working on</span>'
                  f'{html.escape(str(r.get("problem_text","")))}'
                  + (f'\n\n(differs only in upstream state: {html.escape(str(r["missing_info_description"]))})'
                     if r.get("missing_info_description") else "") + '</div>')
        askbox = ('<div class="box askbox"><span class="boxlabel">Prompt to the AO</span>'
                  'Describe the model\'s reasoning state at this point.</div>')
        truth = ('<div class="truth"><span class="boxlabel">What good looks like</span>'
                 'the two answers should DIVERGE — same tokens, different upstream activations</div>')
        vcls = "ok" if div else "bad"
        vtxt = "answers diverge (reads activations)" if div else "answers identical (text-only behaviour)"
        cmp = (
            '<div class="cmp">'
            f'<div><div class="resphead"><span class="tag ours">State A · complete</span></div>'
            f'<div class="resp">{html.escape(str(r.get("response_a","")))}</div></div>'
            f'<div><div class="resphead"><span class="tag base">State C · incomplete</span></div>'
            f'<div class="resp">{html.escape(str(r.get("response_c","")))}</div></div>'
            '</div>'
            f'<div class="resphead" style="margin-top:6px"><span class="verdict {vcls}">{vtxt}</span></div>'
        )
        out.append(f'<div class="card">{ctx}{srcbox}{askbox}{truth}{cmp}</div>')
        if len(out) >= n:
            break
    return "".join(out)


EXTRACTORS: dict[str, Callable[[dict, dict], tuple[str, str, dict]]] = {
    "number_prediction": _ex_number,
    "mmlu_prediction": _ex_mmlu,
    "missing_info": _ex_missing,
    "backtracking": _ex_backtracking,
    "domain_confusion": _ex_domain,
    "vagueness": _ex_vagueness,
    "causal_faithfulness": _ex_cfe,
    "abstention": _ex_abstention,
    "model_diffing": _ex_diffing,
}


# Failure-mode buckets, in display priority order: disagreements first (the most
# diagnostic), then shared failures, then shared successes. Each card is tagged
# with its bucket so the reader sees regressions, not a random draw.
_BUCKETS = [
    ("ours_win", "Our AO wins"),
    ("ours_lose", "Our AO loses"),
    ("both_fail", "Both fail"),
    ("both_ok", "Both succeed"),
]


def _bucket(o_ok: bool, b_ok: bool) -> str:
    if o_ok and not b_ok:
        return "ours_win"
    if b_ok and not o_ok:
        return "ours_lose"
    return "both_ok" if o_ok else "both_fail"


def select_examples(ours: list[dict], base: list[dict], task: str,
                    dsidx: dict[str, dict], n: int = 6):
    """Join ours↔base per item, bucket by outcome, dedupe by problem.

    Returns (our_card, base_card, bucket_label) triples. We align ours↔base on
    the item `join_key`, classify each into a failure-mode bucket, then emit in
    bucket-priority order (wins/losses before shared outcomes) so regressions and
    improvements surface rather than a random sample. At most one card per
    `group_key`, so the same underlying problem never appears twice.
    """
    if task not in EXTRACTORS:  # metrics-only task — no example cards
        return []
    ex = EXTRACTORS[task]
    obyk, bbyk, group_of = {}, {}, {}
    for r in ours:
        try:
            k, g, c = ex(r, dsidx)
        except Exception:
            continue
        obyk.setdefault(k, c); group_of.setdefault(k, g)
    for r in base:
        try:
            k, _g, c = ex(r, dsidx)
        except Exception:
            continue
        bbyk.setdefault(k, c)
    by_bucket: dict[str, list[str]] = {b: [] for b, _ in _BUCKETS}
    for k in obyk:
        if k not in bbyk:
            continue
        by_bucket[_bucket(bool(obyk[k]["ok"]), bool(bbyk[k]["ok"]))].append(k)
    out, seen = [], set()
    for bkey, blabel in _BUCKETS:               # walk buckets in priority order
        for k in by_bucket[bkey]:
            g = group_of[k]
            if g in seen:
                continue
            seen.add(g)
            out.append((obyk[k], bbyk[k], blabel))
            if len(out) >= n:
                return out
    return out


# =============================================================================
# 6. Plotly figures (inlined; plotly.js added once in <head>)
# =============================================================================

def _fig_html(fig) -> str:
    return fig.to_html(full_html=False, include_plotlyjs=False,
                       default_width="100%", default_height="420px")


def chart_overall(sums_o: dict, sums_b: dict | None, base_label: str = "Paper AO",
                  n_eff_map: dict[str, int] | None = None) -> str:
    """Grouped bar: each task's headline metric, normalized, ours vs baseline."""
    import plotly.graph_objects as go

    n_eff_map = n_eff_map or {}
    tasks, labels, ours, base, raw_o, raw_b, eo, eb = [], [], [], [], [], [], [], []
    for t, spec in TASKS.items():
        head = spec["metrics"][0]
        vo = metric_value(sums_o.get(t), head)
        vb = metric_value(sums_b.get(t) if sums_b else None, head)
        if vo is None and vb is None:
            continue
        tasks.append(t)
        labels.append(spec["title"].replace(" ", "<br>", 1))
        ours.append(round(normalize(head["kind"], vo), 4) if vo is not None else None)
        base.append(round(normalize(head["kind"], vb), 4) if vb is not None else None)
        raw_o.append(fmt(head["kind"], vo))
        raw_b.append(fmt(head["kind"], vb))
        eo.append(headline_halfwidth(sums_o.get(t), head, n_eff_map.get(t)))
        eb.append(headline_halfwidth(sums_b.get(t) if sums_b else None, head, n_eff_map.get(t)))

    err_o = dict(type="data", array=eo, visible=True, color="rgba(20,30,50,.35)", thickness=1.3)
    err_b = dict(type="data", array=eb, visible=True, color="rgba(20,30,50,.35)", thickness=1.3)
    fig = go.Figure()
    fig.add_bar(name="Ours", x=labels, y=ours, marker_color=OURS_COLOR, error_y=err_o,
                text=raw_o, textposition="outside",
                hovertemplate="Ours · %{x}<br>normalized %{y:.3f} ± %{error_y.array:.3f}<extra></extra>")
    if sums_b:
        fig.add_bar(name=base_label, x=labels, y=base, marker_color=BASE_COLOR, error_y=err_b,
                    text=raw_b, textposition="outside",
                    hovertemplate=base_label + " · %{x}<br>normalized %{y:.3f} ± %{error_y.array:.3f}<extra></extra>")
    fig.add_hline(y=0, line_dash="dash", line_color="#888",
                  annotation_text="chance", annotation_position="bottom right")
    fig.update_layout(
        barmode="group", template="plotly_white",
        title="Headline metric per task — normalized (0 = chance, 1 = perfect; bars show raw metric, "
              "whiskers = 95% CI)",
        yaxis_title="normalized score", legend_title_text="",
        margin=dict(t=60, b=40, l=50, r=20), font=dict(size=13),
    )
    fig.update_yaxes(range=[min(-0.1, min([v for v in ours + base if v is not None], default=0) - 0.05), 1.08])
    return _fig_html(fig)


def chart_roc(title: str, rows_o: list[dict], rows_b: list[dict]) -> str | None:
    """ROC overlay: our curve vs the paper's, recomputed from per-example margins."""
    import plotly.graph_objects as go

    ro = roc_curve(rows_o)
    rb = roc_curve(rows_b) if rows_b else None
    if ro is None and rb is None:
        return None
    fig = go.Figure()
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="chance",
                    line=dict(dash="dash", color="#bbb"), hoverinfo="skip")
    if ro:
        fig.add_scatter(x=ro[0], y=ro[1], mode="lines", name=f"Ours (AUC {ro[2]:.3f})",
                        line=dict(color=OURS_COLOR, width=2.5))
    if rb:
        fig.add_scatter(x=rb[0], y=rb[1], mode="lines", name=f"Paper AO (AUC {rb[2]:.3f})",
                        line=dict(color=BASE_COLOR, width=2.5))
    fig.update_layout(
        template="plotly_white", title=title,
        xaxis_title="False positive rate", yaxis_title="True positive rate",
        margin=dict(t=50, b=40, l=50, r=20), font=dict(size=13),
        legend=dict(x=0.55, y=0.08), width=None,
    )
    fig.update_xaxes(range=[0, 1]); fig.update_yaxes(range=[0, 1.02])
    return _fig_html(fig)


# =============================================================================
# 7. HTML assembly
# =============================================================================

CSS = """
:root{--ink:#1a2231;--mut:#5b6677;--line:#e6e9ef;--ours:#2A6FDF;--base:#7C9C59;--bg:#f7f8fb;}
*{box-sizing:border-box}
body{font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
 color:var(--ink);margin:0;background:var(--bg);}
.wrap{max-width:1040px;margin:0 auto;padding:32px 24px 80px;}
h1{font-size:26px;margin:0 0 4px} h2{font-size:21px;margin:40px 0 10px;padding-top:8px;border-top:2px solid var(--line)}
h3{font-size:16px;margin:22px 0 8px} p{margin:8px 0} .mut{color:var(--mut)}
.card{background:#fff;border:1px solid var(--line);border-radius:12px;padding:18px 20px;margin:14px 0;
 box-shadow:0 1px 2px rgba(20,30,50,.04)}
.tag{display:inline-block;font-size:12px;font-weight:600;padding:2px 9px;border-radius:999px}
.tag.ours{background:#e8f0fe;color:var(--ours)} .tag.base{background:#eef3e6;color:#5d7d3a}
table{border-collapse:collapse;width:100%;font-size:14px;margin:6px 0}
th,td{text-align:left;padding:7px 10px;border-bottom:1px solid var(--line);vertical-align:top}
th{font-size:12px;text-transform:uppercase;letter-spacing:.04em;color:var(--mut)}
td.num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
td.small-n{color:#b3691e;font-weight:600}
.win{color:#137333;font-weight:600} .lose{color:#b3261e;font-weight:600} .tie{color:var(--mut)}
.metricname{font-weight:600} .metricdesc{color:var(--mut);font-size:12.5px;margin-top:2px}
.cmp{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:10px}
.resp{background:#fafbfc;border:1px solid var(--line);border-radius:8px;padding:10px 12px;
 max-height:230px;overflow:auto;white-space:pre-wrap;font-size:13.5px}
.cardctx{font-size:12px;color:var(--mut);margin-bottom:6px}
.boxlabel{display:block;font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.06em;
 color:var(--mut);margin-bottom:3px}
.box{border-radius:8px;padding:8px 12px 10px;margin:6px 0;white-space:pre-wrap;font-size:13px;
 max-height:280px;overflow:auto}
.srcbox{background:#f3f6fc;border:1px solid #dfe6f2;color:#2f3a52}
.askbox{background:#eef6f0;border:1px solid #d6e6da;color:#2c4736}
.truth{background:#fffaf0;border-left:3px solid #e0b15a;padding:7px 12px;margin:6px 0}
.truthbox{background:#fdf6ec;border:1px solid #e8d3a8;color:#5a4424}
.resphead{display:flex;align-items:center;gap:8px;margin-bottom:5px}
.verdict{font-size:12.5px;font-weight:600}
.ok{color:#137333} .bad{color:#b3261e}
details{margin-top:6px} summary{cursor:pointer;color:var(--mut);font-size:12.5px}
.actual{margin:6px 0} .actual summary{font-weight:600;color:#7a5a1e}
.legendpill{display:inline-block;width:11px;height:11px;border-radius:3px;margin-right:5px;vertical-align:middle}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:18px}
@media(max-width:760px){.cmp,.grid2{grid-template-columns:1fr}}
/* sticky in-page nav */
.toc{position:sticky;top:0;z-index:10;background:rgba(247,248,251,.94);backdrop-filter:blur(6px);
 border-bottom:1px solid var(--line);margin:10px -24px 0;padding:8px 24px;display:flex;flex-wrap:wrap;gap:14px;font-size:13px}
.toc a{color:var(--mut);text-decoration:none} .toc a:hover{color:var(--ours)}
/* scoreboard */
.scoreboard{background:#fff;border:1px solid var(--line);border-radius:12px;padding:14px 16px;margin:16px 0;
 box-shadow:0 1px 2px rgba(20,30,50,.04)}
.sbhead{font-size:12px;text-transform:uppercase;letter-spacing:.05em;color:var(--mut);margin-bottom:10px}
.chips{display:flex;flex-wrap:wrap;gap:8px}
.chip{display:flex;flex-direction:column;gap:1px;text-decoration:none;border:1px solid var(--line);
 border-radius:9px;padding:6px 10px;min-width:120px;background:#fafbfc}
.chip:hover{border-color:#c7d0de} .chiptask{font-size:12px;color:var(--ink);font-weight:600}
.chipd{font-size:13px;font-variant-numeric:tabular-nums}
.chip.win{background:#eef7f0;border-color:#cfe7d5} .chip.win .chipd{color:#137333}
.chip.lose{background:#fdf0ef;border-color:#f2d2cf} .chip.lose .chipd{color:#b3261e}
.chip.tie .chipd{color:var(--mut)}
.tally{margin-top:11px;font-size:14px;color:var(--ink)}
/* collapsible task sections */
details.tasksec{border-top:2px solid var(--line);margin:34px 0 0;padding-top:6px}
details.tasksec>summary{list-style:none;cursor:pointer;display:flex;align-items:baseline;gap:8px;padding:6px 0}
details.tasksec>summary::-webkit-details-marker{display:none}
details.tasksec>summary::before{content:"▸";color:var(--mut);font-size:13px}
details.tasksec[open]>summary::before{content:"▾"}
.secttl{font-size:21px;font-weight:600}
.rawlink{margin-left:auto;font-size:12px;color:var(--mut);text-decoration:none;font-weight:400}
.rawlink:hover{color:var(--ours)}
/* misc */
.bucket{display:inline-block;font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.05em;
 padding:1px 7px;border-radius:999px;margin-right:7px;vertical-align:middle}
.bucket.win{background:#e6f4ea;color:#137333} .bucket.lose{background:#fce8e6;color:#b3261e}
.bucket.tie{background:#eef0f3;color:var(--mut)}
.nsflag{font-size:10px;font-weight:700;color:var(--mut);background:#eef0f3;border-radius:4px;padding:0 4px;margin-left:4px}
.copybtn{font-size:12px;font-weight:600;color:var(--ours);background:#eef3fc;border:1px solid #d3e0f7;
 border-radius:7px;padding:4px 10px;cursor:pointer;vertical-align:middle;margin-left:10px}
.copybtn:hover{background:#e3ecfb}
.hidden-md{position:absolute;left:-9999px;top:-9999px;width:1px;height:1px;opacity:0}
"""


def _delta_cell(m: Metric, vo: float | None, vb: float | None, se_diff: float | None = None,
                ci: tuple[float, float, float] | None = None) -> str:
    """Δ (ours−base), coloured by direction. Significance test prefers the PAIRED
    bootstrap CI (`ci`=(delta,lo,hi)) when given: a CI that straddles 0 is greyed
    and tagged n.s., and the [lo, hi] interval is shown. Falls back to the analytic
    independent-SE test (|Δ| < 1.96·SE_diff) when no bootstrap CI is available."""
    if vo is None or vb is None:
        return '<td class="num tie">—</td>'
    d = vo - vb
    better = (d > 0) == m["hib"]
    sign = "+" if d >= 0 else ""
    suffix = "/5" if m["kind"] == "scale5" else ""
    if ci is not None:
        _, lo, hi = ci
        ns = lo <= 0.0 <= hi
        citext = f' <span class="mut">[{lo:+.3f}, {hi:+.3f}]</span>'
    else:
        ns = se_diff is not None and abs(d) < _Z * se_diff
        citext = ""
    if abs(d) < 1e-6 or ns:
        cls = "tie"
    else:
        cls = "win" if better else "lose"
    tag = ' <span class="nsflag">n.s.</span>' if ns and abs(d) >= 1e-6 else ""
    return f'<td class="num {cls}">{sign}{d:.3f}{suffix}{citext}{tag}</td>'


def metric_table(task: str, sums_o: dict, sums_b: dict | None, base_label: str = "Paper AO",
                 recs_o: list | None = None, recs_b: list | None = None) -> str:
    spec = TASKS[task]
    has_base = sums_b is not None and sums_b.get(task) is not None
    # effective n for clustered judge tasks: distinct problems, not probe count
    neff = effective_n(task, recs_o) or effective_n(task, recs_b)
    head = ["Metric", "n", "Ours"] + ([base_label, f"Δ (ours−{base_label})"] if has_base else [])
    rows = []
    for i, m in enumerate(spec["metrics"]):
        vo = metric_value(sums_o.get(task), m)
        vb = metric_value(sums_b.get(task) if sums_b else None, m)
        n = metric_n(sums_o.get(task), m) or metric_n(sums_b.get(task) if sums_b else None, m)
        n_se = neff if neff else n          # CIs use the smaller, honest sample
        star = " ★" if i == 0 else ""
        arrow = "↑ higher better" if m["hib"] else "↓ lower better"
        name = (f'<div class="metricname">{html.escape(m["label"])}{star}</div>'
                f'<div class="metricdesc">{html.escape(m["desc"])} <i>({arrow})</i></div>')
        # flag small samples: judge/rate deltas on <30 (effective) items are noise-prone
        ncls = "num small-n" if (n_se is not None and n_se < 30) else "num"
        ncell = (f'{n} <span class="mut">({neff} prob.)</span>'
                 if (neff and n and neff < n) else (n if n is not None else "—"))
        cells = [f"<td>{name}</td>", f'<td class="{ncls}">{ncell}</td>',
                 f'<td class="num">{fmt(m["kind"], vo)}</td>']
        if has_base:
            se_o = metric_se(sums_o.get(task), m, vo, n_se)
            se_b = metric_se(sums_b.get(task) if sums_b else None, m, vb, n_se)
            se_diff = math.sqrt(se_o ** 2 + se_b ** 2) if (se_o is not None and se_b is not None) else None
            # Headline row (i==0): use the tighter PAIRED bootstrap CI for the
            # significance call; other rows keep the analytic independent-SE test.
            ci = paired_bootstrap_delta(task, recs_o, recs_b) if i == 0 else None
            cells.append(f'<td class="num">{fmt(m["kind"], vb)}</td>')
            cells.append(_delta_cell(m, vo, vb, se_diff, ci))
        rows.append("<tr>" + "".join(cells) + "</tr>")
    thead = "".join(f"<th>{h}</th>" for h in head)
    note = (f'<p class="mut" style="margin:4px 0 0">n shown as <b>probes ({neff} problems)</b>: '
            f'rows are many probe positions over {neff} source problems, so CIs and n.s. flags '
            f'use the {neff}-problem effective sample (correlated probes don\'t add independent evidence).</p>'
            if neff else "")
    cinote = ('<p class="mut" style="margin:4px 0 0">The headline (★) Δ shows a <b>paired '
              'bootstrap 95% CI</b> [lo, hi] over items (both AOs scored on the same items). '
              'A CI that crosses 0 is greyed and tagged <span class="nsflag">n.s.</span> — the gap '
              'is within noise.</p>') if task in _HL_BOOTSTRAP_TASKS and has_base else ""
    return f"<table><thead><tr>{thead}</tr></thead><tbody>{''.join(rows)}</tbody></table>{note}{cinote}"


def _resp_block(side: str, card: dict, label: str) -> str:
    """One responder column: a clear 'who' header + verdict, then the answer."""
    cls = "ours" if side == "ours" else "base"
    vcls = "ok" if card["ok"] else "bad"
    reasoning = ""
    if card.get("reasoning"):
        reasoning = (f'<details><summary>judge reasoning</summary>'
                     f'<div class="resp">{html.escape(card["reasoning"])}</div></details>')
    return (f'<div><div class="resphead"><span class="tag {cls}">{label}</span>'
            f'<span class="verdict {vcls}">{html.escape(str(card["verdict"]))}</span></div>'
            f'<div class="resp">{html.escape(str(card["text"]))}</div>{reasoning}</div>')


def calibration_html(recs_o: list, recs_b: list) -> str:
    """Render the threshold-calibration diagnostic for a binary probe task.

    Shows balanced accuracy at the default 0-threshold vs. the best achievable
    threshold on this set, for ours and paper — making explicit how much of any
    accuracy gap is mis-calibration (fixable post-hoc) rather than ranking power.
    """
    co = calibration(recs_o)
    cb = calibration(recs_b) if recs_b else None
    if not co:
        return ""

    def row(name: str, c: dict | None) -> str:
        if not c:
            return f"<tr><td>{name}</td><td colspan='4' class='mut'>—</td></tr>"
        cv = c["bal_cv"]
        gain = (cv - c["bal0"]) if cv is not None else None
        cvcell = (f"{cv:.3f} <span class='mut'>(k={c['k']})</span>" if cv is not None else "—")
        gaincell = (f"{gain:+.3f}" if gain is not None else "—")
        return (f"<tr><td>{name}</td>"
                f"<td class='num'>{c['bal0']:.3f}</td>"
                f"<td class='num'>{cvcell}</td>"
                f"<td class='num'>{gaincell}</td>"
                f"<td class='num mut'>{c['bal_best']:.3f} @ {c['best_t']:+.2f}</td></tr>")

    return (
        "<details class='actual' style='margin-top:8px'><summary>Threshold calibration "
        "(how much of the gap is just thresholding?)</summary>"
        "<p class='mut' style='margin:6px 0'>Balanced accuracy = mean(TPR, TNR), so a biased "
        "threshold can't be masked by class imbalance. <b>@0</b> is the eval's default decision. "
        "<b>held-out</b> fits the threshold by stratified k-fold on train folds and scores the "
        "left-out fold (the <i>honest</i> gain a deployable threshold recovers). <b>realised gain</b> "
        "= held-out − @0. The faint last column is the in-sample best (optimistic ceiling only).</p>"
        "<table><thead><tr><th>model</th><th>bal-acc @0</th><th>bal-acc held-out</th>"
        "<th>realised gain</th><th>ceiling</th></tr></thead><tbody>"
        f"{row('Our AO', co)}{row('Paper AO', cb)}</tbody></table></details>"
    )


def comparison_cards(task: str, recs_o: list, recs_b: list, dsidx: dict[str, dict],
                     base_label: str = "Paper AO") -> str:
    """Render the deduped example cards with a consistent labelled layout:
    context → what the model was working on → prompt to the AO → expected
    answer → Our AO vs Paper AO."""
    pairs = select_examples(recs_o, recs_b, task, dsidx, n=6)
    if not pairs:
        return '<p class="mut">No jointly-available examples to compare for this task.</p>'
    bclass = {"Our AO wins": "win", "Our AO loses": "lose",
              "Both fail": "lose", "Both succeed": "tie"}
    out = []
    for oc, bc, bucket in pairs:
        badge = (f'<span class="bucket {bclass.get(bucket,"tie")}">{html.escape(bucket)}</span>')
        ctx = oc.get("context")
        ctxline = f'<div class="cardctx">{badge}{html.escape(str(ctx))}</div>' if ctx else f'<div class="cardctx">{badge}</div>'
        src = oc.get("source")
        # full CoT is offered in a collapsed panel so traces are complete but the
        # card stays scannable (the snippet shows the probed tail by default)
        full = oc.get("source_full")
        fullbox = (f'<details><summary>Full chain-of-thought ({len(str(full))} chars)</summary>'
                   f'<div class="box srcbox">{html.escape(str(full))}</div></details>') if full else ""
        srcbox = (f'<div class="box srcbox"><span class="boxlabel">What the model was working on</span>'
                  f'{html.escape(str(src))}</div>{fullbox}') if src else ""
        ask = oc.get("ao_prompt")
        askbox = (f'<div class="box askbox"><span class="boxlabel">Prompt to the AO</span>'
                  f'{html.escape(str(ask))}</div>') if ask else ""
        # The "actual" box shows what happened at the probed point, but its RELATION
        # to grading differs by task — and conflating the two is misleading:
        #   • causal_faithfulness: the before→after patch completion IS the evidence the
        #     judge scores, so it's a genuine grading target.
        #   • continuation tasks (vagueness / backtracking / domain_confusion): this is
        #     the model's real next text shown FOR CONTEXT ONLY. The judge never sees it
        #     and never matches against it — it rates the AO answer's specificity/relevance.
        #     (Labelling this "ground truth" wrongly implies exact-match grading.)
        actual = oc.get("actual")
        if task == "causal_faithfulness":
            actual_label = "Measured outcome — model completion before vs after patch (this is what the judge scores)"
        else:
            actual_label = ("Actual continuation — shown for context only; the judge does NOT match against "
                            "this, it rates the AO answer's specificity/relevance")
        if not actual:
            actualbox = ""
        elif len(str(actual)) <= 320:
            actualbox = (f'<div class="box truthbox"><span class="boxlabel">{actual_label}</span>'
                         f'{html.escape(str(actual)).replace(chr(10), "<br>")}</div>')
        else:
            actualbox = (f'<details class="actual"><summary>{actual_label}</summary>'
                         f'<div class="resp">{html.escape(str(actual))}</div></details>')
        out.append(
            '<div class="card">'
            f'{ctxline}{srcbox}{askbox}'
            f'<div class="truth"><span class="boxlabel">Expected answer</span>'
            f'{html.escape(str(oc["target"]))}</div>'
            f'{actualbox}'
            '<div class="cmp">'
            + _resp_block("ours", oc, "Our AO")
            + _resp_block("base", bc, base_label)
            + "</div></div>"
        )
    return "".join(out)


def rubric_section() -> str:
    """Glossary compiled from the task spec: what every metric means + how scored."""
    blocks = [
        "<p class='mut'>Every task probes the AO by collecting the base model's residual activations "
        "over the last <b>n_positions</b> tokens of a context, injecting them into the AO's prompt, and "
        "reading the AO's answer. Tasks differ in what is asked and how the answer is scored:</p>"
    ]
    for t, spec in TASKS.items():
        rows = "".join(
            f'<tr><td class="metricname">{html.escape(m["label"])}</td>'
            f'<td>{html.escape(m["desc"])}</td>'
            f'<td class="num">{"↑" if m["hib"] else "↓"}</td></tr>'
            for m in spec["metrics"]
        )
        blocks.append(
            f'<div class="card"><h3>{html.escape(spec["title"])} '
            f'<span class="mut" style="font-weight:400">— {t}</span></h3>'
            f'<p class="mut">{html.escape(spec["tests"])}</p>'
            f'<table><thead><tr><th>Metric</th><th>What it measures / how</th><th>dir</th></tr></thead>'
            f'<tbody>{rows}</tbody></table></div>'
        )
    blocks.append(
        "<div class='card'><h3>Reading the numbers</h3><ul>"
        "<li><b>★</b> marks each task's headline metric (the one charted up top).</li>"
        "<li><b>n</b> is the number of items each metric was scored over (its denominator); "
        "values <b>&lt;30</b> are highlighted — a delta on such a small sample is noise-prone.</li>"
        "<li><b>What the model actually did next</b> (in each example card) is the model's real "
        "continuation from the probed point — the ground truth the AO's prediction should be checked "
        "against. (The judge tasks have no single stored gold answer.)</li>"
        "<li><b>Normalized score</b> (top chart) rescales every metric to a common axis so tasks are "
        "comparable: rates stay as-is; AUC uses <code>(auc−0.5)/0.5</code> (0 = chance); judge 1–5 "
        "scores use <code>(x−1)/4</code>. 0 ≈ chance, 1 = perfect.</li>"
        "<li><b>ROC-AUC</b> tasks (MMLU, missing-info) are scored from the yes−no logit margin, not a "
        "hard decision, so they need no judge and are the most reliable signals.</li>"
        "<li><b>Judge-scored</b> tasks (backtracking, vagueness, domain-confusion) depend on the local "
        "LLM judge; occasional judge parse failures make these noisier than the AUC tasks.</li>"
        "<li><b>95% CI whiskers</b> on the headline chart (and <b>n.s.</b> tags in the Δ column) come "
        "from an approximate standard error — binomial for rates, Hanley–McNeil for AUC, a conservative "
        "[1–5] bound for judge scores. A delta tagged <b>n.s.</b> is within noise: do not read it as a "
        "real change. The <b>scoreboard</b> counts a task as win/loss only when its Δ clears its CI.</li>"
        "<li><b>Effective n</b>: vagueness and domain-confusion rows are many probe positions over "
        "only ~29 source problems, so an <code>n</code> of 591 is not 591 independent items. Those "
        "tables show <b>probes (k problems)</b> and size their CIs / n.s. flags from the k-problem "
        "effective sample — correlated probes of one problem don't add independent evidence.</li>"
        "<li><b>Failure buckets</b>: example cards are chosen by outcome (Our AO wins / loses / both "
        "fail) and badged accordingly, so regressions surface instead of a random sample.</li>"
        "<li><b>Full chain-of-thought</b>: each trace card shows the probed tail by default with a "
        "<code>⟵</code> marker at the exact probe point; expand <i>Full chain-of-thought</i> for the "
        "complete reasoning the activations were taken from.</li>"
        "</ul></div>"
    )
    return "".join(blocks)


def _headline_delta(sums_o: dict, sums_b: dict | None, task: str):
    """(normalized Δ, 95%-CI half-width on the normalized axis) for a task's
    headline metric, or (None, None) if either side is missing."""
    head = TASKS[task]["metrics"][0]
    vo = metric_value(sums_o.get(task), head)
    vb = metric_value(sums_b.get(task) if sums_b else None, head)
    if vo is None or vb is None:
        return None, None
    d = normalize(head["kind"], vo) - normalize(head["kind"], vb)
    no, nb = metric_n(sums_o.get(task), head), metric_n(sums_b.get(task) if sums_b else None, head)
    se_o = metric_se(sums_o.get(task), head, vo, no)
    se_b = metric_se(sums_b.get(task) if sums_b else None, head, vb, nb)
    hw = (_Z * math.sqrt(se_o ** 2 + se_b ** 2) * _NORM_SCALE.get(head["kind"], 1.0)
          if (se_o is not None and se_b is not None) else None)
    return d, hw


def scoreboard(sums_o: dict, sums_b: dict | None, base_label: str) -> str:
    """One-glance verdict: a clickable Δ chip per task + a win/loss/n.s. tally and
    the mean normalized Δ. A task only counts as a win/loss when its headline Δ
    clears its 95% CI; otherwise it is n.s. (tie), so the tally reflects signal."""
    if not sums_b:
        return ""
    chips, deltas = [], []
    wins = losses = ties = 0
    for t, spec in TASKS.items():
        if t not in sums_o:
            continue
        d, hw = _headline_delta(sums_o, sums_b, t)
        if d is None:
            continue
        deltas.append(d)
        sig = hw is not None and abs(d) >= hw
        if not sig or abs(d) < 1e-6:
            cls, ties = "tie", ties + 1
        elif d > 0:
            cls, wins = "win", wins + 1
        else:
            cls, losses = "lose", losses + 1
        arrow = "▲" if d > 1e-6 else ("▼" if d < -1e-6 else "—")
        chips.append(
            f'<a class="chip {cls}" href="#sec-{t}">'
            f'<span class="chiptask">{html.escape(spec["title"])}</span>'
            f'<span class="chipd">{arrow} {d:+.3f}</span></a>')
    if not deltas:
        return ""
    net = sum(deltas) / len(deltas)
    netcls = "win" if net > 1e-6 else ("lose" if net < -1e-6 else "tie")
    tally = (f'<b class="win">{wins} win</b> · <b class="lose">{losses} loss</b> · '
             f'<span class="tie">{ties} n.s.</span> &nbsp;|&nbsp; '
             f'mean normalized Δ <b class="{netcls}">{net:+.3f}</b>')
    return (f'<div class="scoreboard" id="scoreboard">'
            f'<div class="sbhead">Scoreboard — Our AO vs {html.escape(base_label)} '
            f'(headline Δ per task; win/loss only when it clears the 95% CI)</div>'
            f'<div class="chips">{"".join(chips)}</div>'
            f'<div class="tally">{tally}</div></div>')


def toc(sums_o: dict) -> str:
    """Sticky in-page nav so a long report is jumpable instead of one big scroll."""
    links = ['<a href="#headline">Headline</a>']
    links += [f'<a href="#sec-{t}">{html.escape(TASKS[t]["title"])}</a>'
              for t in TASKS if t in sums_o]
    links.append('<a href="#rubric">Rubric</a>')
    return f'<nav class="toc">{"".join(links)}</nav>'


def headline_markdown(sums_o: dict, sums_b: dict | None, base_label: str) -> str:
    """The headline comparison as a Markdown table, for the copy-to-clipboard button."""
    head_row = (f"| Task | Ours | {base_label} | Δ (norm) |\n|---|---|---|---|"
                if sums_b else "| Task | Ours |\n|---|---|")
    lines = [head_row]
    for t, spec in TASKS.items():
        if t not in sums_o:
            continue
        head = spec["metrics"][0]
        vo = metric_value(sums_o.get(t), head)
        if sums_b:
            d, _ = _headline_delta(sums_o, sums_b, t)
            vb = metric_value(sums_b.get(t), head)
            dcell = f"{d:+.3f}" if d is not None else "—"
            lines.append(f"| {spec['title']} | {fmt(head['kind'], vo)} | {fmt(head['kind'], vb)} | {dcell} |")
        else:
            lines.append(f"| {spec['title']} | {fmt(head['kind'], vo)} |")
    return "\n".join(lines)


def build_dashboard(results_root: Path, target_run: str = "main",
                    baseline_run: str | None = None, out_path: Path | None = None,
                    base_label: str | None = None) -> Path:
    """Read both runs under results_root and write one self-contained HTML file.

    `base_label` names the reference curve in the UI (default "Paper AO"); set it
    to e.g. "Replication (ours)" when the baseline run is our own frozen AO.
    """
    import plotly.offline as pyo

    results_root = Path(results_root)
    target_run, baseline_run = detect_runs(results_root, target_run, baseline_run)
    tdir = results_root / target_run
    bdir = results_root / baseline_run if baseline_run else None

    sums_o = load_summaries(tdir)
    sums_b = load_summaries(bdir) if bdir else None
    cfg = load_config()
    model = cfg["model"]["name"]
    base_label = base_label or cfg.get("eval", {}).get("baseline_label") or "Paper AO"
    base_lora = cfg.get("baseline", {}).get("ao_lora", "—")
    steer = cfg.get("injection", {}).get("eval_steering_coefficient", "—")
    profile = cfg.get("eval", {}).get("profile", "—")
    n_pos = cfg.get("injection", {}).get("n_positions", cfg.get("eval", {}).get("n_positions", "—"))
    gen_ts = datetime.now().strftime("%Y-%m-%d %H:%M")

    # ---- header + provenance ----
    base_note = (f'vs <span class="tag base">{html.escape(base_label)}</span> '
                 f'(<code>{html.escape(baseline_run)}</code> · <code>{html.escape(base_lora)}</code>)'
                 if bdir else '<b class="lose">no baseline run found</b> — '
                 'run <code>make eval-baseline</code> to populate the comparison')
    header = (
        f"<h1>AObench dashboard — {html.escape(model)} Activation Oracle</h1>"
        f'<p class="mut"><span class="tag ours">Our AO</span> '
        f'<code>{html.escape(target_run)}</code> &nbsp;{base_note}</p>'
        f'<p class="mut">profile <b>{html.escape(str(profile))}</b> · '
        f'eval steering coef <b>{steer}</b> · n_positions <b>{html.escape(str(n_pos))}</b> · '
        f'generated <b>{gen_ts}</b></p>'
        f'<p class="mut">tasks: {", ".join(sums_o.keys())}</p>'
    )

    # effective-n for clustered judge tasks (distinct problems, not probe count),
    # used to size CIs honestly in both the headline chart and the per-task tables
    n_eff_map: dict[str, int] = {}
    for t in _CLUSTERED_TASKS:
        if t in sums_o:
            ne = effective_n(t, load_records(tdir, t)) or (effective_n(t, load_records(bdir, t)) if bdir else None)
            if ne:
                n_eff_map[t] = ne

    # ---- charts ----
    parts = [chart_overall(sums_o, sums_b, base_label, n_eff_map)]
    roc_panels = []
    for task, mode_title in [("mmlu_prediction", "MMLU correctness — ROC (pre-answer)"),
                             ("missing_info", "Missing-info — ROC")]:
        ro = load_records(tdir, task)
        rb = load_records(bdir, task) if bdir else []
        panel = chart_roc(mode_title, ro, rb)
        if panel:
            roc_panels.append(panel)
    roc_html = (f'<h2>ROC curves (discriminative tasks)</h2>'
                f'<div class="grid2">{"".join(roc_panels)}</div>') if roc_panels else ""

    # ---- per-task: metric table + example comparisons ----
    # Each section is a collapsible <details open> with a stable anchor id so the
    # sticky TOC and scoreboard chips can jump to it. activation_sensitivity is
    # rendered by its own A-vs-C card builder (it has no ours↔base pairing).
    task_sections = []
    for task, spec in TASKS.items():
        if task not in sums_o:
            continue
        recs_o = load_records(tdir, task)
        recs_b = load_records(bdir, task) if bdir else []
        dsidx = load_dataset_index(task)
        if task == "activation_sensitivity":
            cards = actsens_cards(recs_o, recs_b, base_label)
            cards_head = "Two activation states of the same tokens — do the answers diverge?"
        else:
            cards = comparison_cards(task, recs_o, recs_b, dsidx, base_label)
            cards_head = f"Direct comparisons — ground truth vs our AO vs {html.escape(base_label.lower())}"
        rawlink = (f' <a class="rawlink" href="{html.escape(target_run)}/{task}_summary.json">raw json ↗</a>')
        task_sections.append(
            f'<details class="tasksec" open id="sec-{task}"><summary>'
            f'<span class="secttl">{html.escape(spec["title"])}</span> '
            f'<span class="mut" style="font-weight:400;font-size:14px">— {task}</span>{rawlink}'
            f'</summary>'
            f'<p class="mut">{html.escape(spec["tests"])}</p>'
            f'{metric_table(task, sums_o, sums_b, base_label, recs_o, recs_b)}'
            f'{calibration_html(recs_o, recs_b) if task in ("mmlu_prediction", "missing_info") else ""}'
            f'<h3>{cards_head}</h3>'
            f'{cards}'
            f'</details>'
        )

    # copy-headline-as-markdown: hidden textarea holds the raw text; the button
    # copies its .value (the browser un-escapes entities), so no JS string-escaping.
    md = headline_markdown(sums_o, sums_b, base_label)
    copy_md = (
        '<button class="copybtn" onclick="navigator.clipboard.writeText('
        "document.getElementById('hl-md').value);this.textContent='copied ✓';"
        "setTimeout(()=>this.textContent='Copy headline as markdown',1200)\">"
        "Copy headline as markdown</button>"
        f'<textarea id="hl-md" class="hidden-md">{html.escape(md)}</textarea>'
    )

    plotly_js = f"<script>{pyo.get_plotlyjs()}</script>"
    body = (
        f'<div class="wrap">{header}'
        f'{toc(sums_o)}'
        f'{scoreboard(sums_o, sums_b, base_label)}'
        f'<h2 id="headline">Headline comparison {copy_md}</h2>{parts[0]}'
        f'{roc_html}'
        f'{"".join(task_sections)}'
        f'<h2 id="rubric">Metric rubric — what each number means</h2>{rubric_section()}'
        f'<p class="mut" style="margin-top:40px">Generated by <code>ao_cli.dashboard</code> from '
        f'{html.escape(str(results_root))} on {gen_ts}. Self-contained: charts are interactive, '
        f'no network needed.</p>'
        f'</div>'
    )
    doc = (f"<!doctype html><html><head><meta charset='utf-8'>"
           f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
           f"<title>AObench dashboard — {html.escape(model)}</title>"
           f"<style>{CSS}</style>{plotly_js}</head><body>{body}</body></html>")

    # Default into the TARGET run's own subfolder (…/aobench_results/<run>/dashboard.html)
    # so each experiment keeps its own dashboard instead of clobbering a shared one
    # at the aobench_results root. An explicit out_path still wins.
    out_path = Path(out_path) if out_path else tdir / "dashboard.html"
    out_path.write_text(doc)
    return out_path


# =============================================================================
# 8. CLI
# =============================================================================

def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--results-dir", default=None,
                   help="aobench_results dir (default: artifacts/<model>/aobench_results)")
    p.add_argument("--target", default=None,
                   help="target run dir name (default: the active EXP / AO_EXP, else 'main')")
    p.add_argument("--baseline", default=None,
                   help="baseline run dir (default: config eval.baseline_run, else auto-detect baseline_*)")
    p.add_argument("--baseline-label", default=None, help="label for the baseline curve in the UI")
    p.add_argument("--out", default=None, help="output HTML path (default: <results-dir>/dashboard.html)")
    args = p.parse_args(argv)

    cfg = load_config()
    if args.results_dir:
        results_root = Path(args.results_dir)
    else:
        results_root = ARTIFACTS / model_slug(cfg["model"]["name"]) / "aobench_results"
    baseline_run = args.baseline or cfg.get("eval", {}).get("baseline_run") or None
    # Default the target to the active experiment (AO_EXP), so `make dashboard
    # EXP=c1_dpo` rebuilds the c1_dpo run rather than silently falling back to
    # `main` and overwriting it with an unrelated run.
    target = args.target or experiment_name() or "main"

    out = build_dashboard(results_root, target_run=target,
                          baseline_run=baseline_run, out_path=Path(args.out) if args.out else None,
                          base_label=args.baseline_label)
    print(f"[dashboard] wrote {out}")


if __name__ == "__main__":
    main()
