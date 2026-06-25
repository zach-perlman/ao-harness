#!/usr/bin/env python3
"""Re-score the on-policy corpus's correctness labels in place.

WHY
---
`generate_cots.py`'s original `extract_answer` was written for VERBOSE answers
(answer embedded in prose: `\\boxed{}`, '#### 42', 'the answer is C', trailing
numbers). Gemma 4 (and any thinking model) instead emits the answer in a
separate channel, so the post-think `cot_content` is TERSE — a bare 'C',
'RIGHT', or markdown-wrapped '**A**'. The old extractor returned None on those,
so every correct terse answer was scored WRONG. On the gemma-4-12B corpus that
collapsed measured accuracy to ~19% and inflated `both_wrong` to ~70%,
corrupting the `category` / `cot_correct` / `direct_correct` labels that
accuracy reporting and category-dependent contributions (e.g. C4 solvability)
rely on.

WHAT THIS DOES
--------------
A pure CPU pass that recomputes ONLY the label columns
(`cot_answer`, `direct_answer`, `cot_correct`, `direct_correct`, `category`)
from the already-stored raw text (`cot_content`, `cot_response`,
`direct_response`, `correct_answer`), using the hardened extractor below
(identical to the patched `generate_cots.py`). It deliberately does NOT touch
`cot_response` / `sentences` / `boundary_positions` — those are token-aligned
and consumed downstream, so editing their text would desync the offsets.

The categorization mirrors generate_cots.py exactly:
    load_bearing = cot right & direct wrong   (thinking helped)
    both_correct = cot right & direct right
    cot_hurt     = cot wrong & direct right
    both_wrong   = both wrong
    None         = no ground truth (e.g. lmsys)

Safety: the untouched original is preserved as `corpus.raw.jsonl` (written once,
never overwritten on re-runs), and the rewrite is atomic (temp file + rename).
"""
import argparse
import json
import os
import re
import shutil
from collections import Counter, defaultdict


# --- hardened extraction (kept byte-identical to generate_cots.py) -----------
def extract_answer(response):
    if not response:
        return None
    text = response.strip()
    stripped = re.sub(r'^[\*`_"\'\s]+|[\*`_"\'\s]+$', '', text)
    boxed = re.findall(r'\\boxed\{([^}]+)\}', text)
    if boxed:
        return boxed[-1].strip()
    hashes = re.findall(r'####\s*(.+?)(?:\n|$)', text)
    if hashes:
        return hashes[-1].strip().replace(",", "")
    bare_letter = re.match(r'^\(?([A-E])\)?[.):]?$', stripped)
    if bare_letter:
        return bare_letter.group(1).upper()
    letter_match = re.findall(r'(?:answer|option)\s+(?:is\s+)?\(?([A-E])\)?\b', text, re.IGNORECASE)
    if letter_match:
        return letter_match[-1].upper()
    answer_is = re.findall(r'(?:the\s+)?answer\s+is\s*:?\s*(.+?)(?:\.|$)', text, re.IGNORECASE)
    if answer_is:
        ans = answer_is[-1].strip()
        if len(ans) <= 50:
            return ans
    for pattern in (
        r'(?:answer|result)\s+(?:is|=)\s*[:\s]*(-?\d+(?:[.,]\d+)*)',
        r'(?:=|equals?)\s*(-?\d+(?:[.,]\d+)*)\s*$',
        r'\*\*(-?\d+(?:[.,]\d+)*)\*\*\s*$',
    ):
        m = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
        if m:
            return m[-1].strip().replace(",", "")
    if stripped and len(stripped) <= 40 and "\n" not in stripped:
        return stripped
    numbers = re.findall(r'(-?\d+(?:\.\d+)?)', text)
    if numbers:
        return numbers[-1]
    return None


def answers_match(model_answer, correct_answer):
    if model_answer is None:
        return False
    def normalize(s):
        s = str(s).strip()
        s = re.sub(r'^[\*`_"\'\(\[\s]+|[\*`_"\'\)\].:\s]+$', '', s)
        s = s.replace(",", "").replace(" ", "")
        if s.endswith(".0"):
            s = s[:-2]
        frac = re.match(r'\\frac\{(\d+)\}\{(\d+)\}', s)
        if frac:
            s = str(int(frac.group(1)) / int(frac.group(2)))
        return s.lower()
    return normalize(model_answer) == normalize(correct_answer)


def categorize(row):
    """Recompute (cot_answer, direct_answer, cot_correct, direct_correct, category)
    for one row, exactly as generate_cots.py would with the hardened extractor."""
    gold = row.get("correct_answer")
    if gold is None:
        return None, None, None, None, None
    direct_answer = extract_answer(row.get("direct_response") or "")
    direct_correct = answers_match(direct_answer, gold)
    # Answer lives in the post-think content; fall back to the trace if truncated.
    cot_content = row.get("cot_content") or ""
    cot_answer = extract_answer(cot_content) if cot_content else extract_answer(row.get("cot_response") or "")
    cot_correct = answers_match(cot_answer, gold)
    if cot_correct and not direct_correct:
        category = "load_bearing"
    elif cot_correct and direct_correct:
        category = "both_correct"
    elif not cot_correct and not direct_correct:
        category = "both_wrong"
    else:
        category = "cot_hurt"
    return cot_answer, direct_answer, cot_correct, direct_correct, category


def summarize(rows):
    n = len(rows)
    cat = Counter(r.get("category") for r in rows)
    cot_ok = sum(1 for r in rows if r.get("cot_correct"))
    dir_ok = sum(1 for r in rows if r.get("direct_correct"))
    gt = sum(1 for r in rows if r.get("correct_answer") is not None)
    per_src = defaultdict(lambda: [0, 0])  # source -> [correct, total_with_gt]
    for r in rows:
        if r.get("correct_answer") is not None:
            per_src[r["source"]][1] += 1
            per_src[r["source"]][0] += bool(r.get("cot_correct"))
    return n, cat, cot_ok, dir_ok, gt, per_src


def print_summary(tag, rows):
    n, cat, cot_ok, dir_ok, gt, per_src = summarize(rows)
    print(f"\n[{tag}] {n} rows ({gt} with ground truth)")
    print(f"  cot_correct:    {cot_ok:>6} ({100*cot_ok/n:5.1f}% of all, {100*cot_ok/max(gt,1):5.1f}% of gt)")
    print(f"  direct_correct: {dir_ok:>6} ({100*dir_ok/n:5.1f}% of all, {100*dir_ok/max(gt,1):5.1f}% of gt)")
    print("  categories: " + ", ".join(f"{k}={v}" for k, v in cat.most_common()))
    print("  per-source cot accuracy:")
    for src in sorted(per_src):
        c, t = per_src[src]
        print(f"    {src:<16} {100*c/max(t,1):5.1f}%  ({c}/{t})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus", help="path to corpus.jsonl")
    ap.add_argument("--dry-run", action="store_true", help="report before/after, do not write")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.corpus)]
    print_summary("BEFORE", rows)

    rescored = [dict(r) for r in rows]
    changed = 0
    for r in rescored:
        ca, da, cc, dc, cat = categorize(r)
        if (r.get("cot_correct"), r.get("category")) != (cc, cat):
            changed += 1
        r["cot_answer"], r["direct_answer"] = ca, da
        r["cot_correct"], r["direct_correct"], r["category"] = cc, dc, cat
    print_summary("AFTER", rescored)
    print(f"\nrows whose (cot_correct, category) changed: {changed} / {len(rows)}")

    if args.dry_run:
        print("\n[dry-run] no files written.")
        return

    backup = args.corpus.replace(".jsonl", ".raw.jsonl")
    if not os.path.exists(backup):
        shutil.copy2(args.corpus, backup)
        print(f"\nbacked up original -> {backup}")
    else:
        print(f"\nbackup already exists, left untouched -> {backup}")

    tmp = args.corpus + ".tmp"
    with open(tmp, "w") as f:
        for r in rescored:
            f.write(json.dumps(r) + "\n")
    os.replace(tmp, args.corpus)
    print(f"rewrote labels in place -> {args.corpus}")


if __name__ == "__main__":
    main()
