"""One-off: side-by-side AObench scorecard across saved runs.

Walks each run's *_summary.json, pulls the single most salient metric per task
(recursively, since AObench nests metrics under varying keys), and prints a
fixed-width table so v1 (final / early step_2000) and v2-best line up column-wise.
"""
import glob
import json

RUNS = {
    "v1-final": "replication_v1",
    "v1-step2k": "replication_v1_step2000",
    "v2-best": "replication_v2",
}
# task -> [(label, summary-json key to fetch)]
SALIENT = {
    "number_prediction": [("num_match", "matches_model_answer_rate")],
    "mmlu_prediction": [("mmlu_auc", "roc_auc")],
    "backtracking": [("bt_corr>=3", "correctness_>=3_rate")],
    "vagueness": [("vag_specific", "specificity_rate")],
    "domain_confusion": [("dom_wrong", "domain_wrong_rate")],
    "missing_info": [("mi_auc", "roc_auc")],
    "activation_sensitivity": [("act_sens", "activation_sensitivity")],
    "causal_faithfulness": [("faith", "causal_faithfulness")],
    "abstention": [("abst_f1", "abstention_f1"), ("abst_balacc", "balanced_accuracy")],
}


def find(node, key):
    """Deep-search a nested dict/list for the last numeric value under `key`."""
    hit = [None]

    def walk(x):
        if isinstance(x, dict):
            for k, v in x.items():
                if k == key and isinstance(v, (int, float)) and not isinstance(v, bool):
                    hit[0] = v
                else:
                    walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)

    walk(node)
    return hit[0]


def load(run, task):
    f = glob.glob(f"artifacts/gemma-4-12B-it/aobench_results/{run}/{task}_summary.json")
    return json.load(open(f[0])) if f else None


hdr = f"{'metric':16s}" + "".join(f"{r:>11s}" for r in RUNS)
print(hdr)
print("-" * len(hdr))
for task, mets in SALIENT.items():
    for lab, key in mets:
        row = f"{lab:16s}"
        for run in RUNS.values():
            d = load(run, task)
            v = find(d, key) if d else None
            row += f"{(f'{v:.3f}' if isinstance(v, (int, float)) else '--'):>11s}"
        print(row)
