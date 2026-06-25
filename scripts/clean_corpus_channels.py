#!/usr/bin/env python3
"""Strip dangling thinking-delimiter tokens from an existing corpus.

WHY
---
Truncated Gemma rollouts (hit the 8192-token cap before emitting the
`<channel|>` closer) fall through `parse_think`'s fallback, leaving a dangling
`<|channel>thought` opener at the start of `cot_response`. That leaks into the
stored `cot_response` (the logit-lens pretrain text) and into the first entry
of `sentences` (the convqa input). ~5.6% of the gemma-4-12B corpus is affected.

WHAT THIS DOES
--------------
For each affected row, removes the channel/turn control tokens from
`cot_response` (via the shared `strip_think_tokens`) and re-derives `sentences`
+ `n_sentences` from the cleaned text. Rows with no such tokens are left
byte-identical. `boundary_positions` are already empty (deferred to the
downstream GPU tokenize step), so cleaning the text causes no offset desync.

Idempotent (re-running is a no-op) and atomic (temp file + rename). The pristine
pre-rescore original remains at `corpus.raw.jsonl`.
"""
import argparse
import json
import os
import sys
from pathlib import Path

# cot_utils is dependency-light (re/math/random) — safe to import on CPU.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]
                       / "activation_oracles" / "third_party" / "cot-oracle" / "src"))
from cot_utils import strip_think_tokens, split_cot_into_sentences  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus", help="path to corpus.jsonl")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.corpus)]
    changed = 0
    for r in rows:
        cot = r.get("cot_response") or ""
        cleaned = strip_think_tokens(cot)
        if cleaned == cot:
            continue
        changed += 1
        r["cot_response"] = cleaned
        r["sentences"] = split_cot_into_sentences(cleaned)
        r["n_sentences"] = len(r["sentences"])

    print(f"rows scanned: {len(rows)} | rows cleaned: {changed} "
          f"({100*changed/max(len(rows),1):.1f}%)")
    if args.dry_run:
        print("[dry-run] no files written.")
        return
    if changed == 0:
        print("nothing to clean; leaving file untouched.")
        return

    tmp = args.corpus + ".tmp"
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    os.replace(tmp, args.corpus)
    print(f"rewrote -> {args.corpus}")


if __name__ == "__main__":
    main()
