"""Generate the chunked conversational-QA dataset with the LOCAL judge.

Gears-level overview:
  This reimplements the paper's 3-round "chunked convqa" pipeline
  (scripts/generate_chunked_convqa_anthropic.py — Anthropic batch API) against
  our local vLLM judge instead. The prompts are IMPORTED from that script so
  we stay byte-identical to the paper's data recipe; only the transport
  changes (OpenAI-compatible chat.completions, async with a semaphore).

  Per CoT from the on-policy corpus:
    R1: judge sees the full sentence-numbered CoT, picks a split point and
        writes a question whose answer lives only in the suffix.
    R2: judge answers from the PREFIX only (black-box baseline; kept for
        analysis, not used as a training target).
    R3: judge answers from the SUFFIX only -> becomes `target_response`.

  The AO is later trained to answer the R1 question from activations taken
  over `cot_prefix` — i.e. to articulate reasoning the prefix text alone does
  not reveal. Output: artifacts/<slug>/convqa/{train,test}.parquet with the
  cds-jb/cot-oracle-convqa-chunked schema (cot_prefix/prompt/target_response).
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import os
import random
import re
from pathlib import Path

import pandas as pd
from openai import AsyncOpenAI

from . import REPO, artifacts_dir, judge_base_url, load_config, model_slug, resolve_layers

# Import the paper's exact prompts + R1 JSON parser from the vendored script.
_spec = importlib.util.spec_from_file_location(
    "convqa_prompts", REPO / "scripts" / "generate_chunked_convqa_anthropic.py"
)
_prompts = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_prompts)

MIN_SENTENCES = 8  # upstream filter: enough sentences for a meaningful split

# ---- Post-generation quality filters (applied by `convqa --filter`) ----
# A row is a low-value training example when (1) its question leaks the chunking
# scaffold (mentions prefix/suffix — the AO has no "suffix" at inference), or
# (2) the prefix-only answer already matches the suffix answer (the activations
# add no signal), or (3) it is one of many "the reasoning does not…" absence
# labels (over-represented, teaches the AO to predict nothing).
_SCAFFOLD_RE = re.compile(r"\b(prefix|suffix|split point|the split)\b", re.I)
_NEG_RE = re.compile(
    r"\b(does not|doesn't|do not|did not|didn't|not contain|no specific|"
    r"not provide|no further|nothing|does not (?:cite|distinguish|introduce|"
    r"identify|mention|consider|perform|take))\b",
    re.I,
)
LEAK_OVERLAP = 0.6      # R2/R3 word-overlap above this ⇒ prefix already answers
NEG_TARGET_FRAC = 0.12  # cap "does not…" targets at this share of the final set

# Print the first few completed rows so a human can watch the judge work live
# (its generated question + the prefix/suffix answers), then go quiet so we
# don't spam the log with ~25k rows. asyncio is single-threaded, so the counter
# increments safely between awaits.
_SAMPLES_TO_SHOW = 8
_samples_shown = 0


def _preview_row(row: dict) -> None:
    global _samples_shown
    if _samples_shown >= _SAMPLES_TO_SHOW:
        return
    _samples_shown += 1
    clip = lambda s, n: (s[:n] + "…") if len(s) > n else s
    print(
        f"\n[convqa sample {_samples_shown}/{_SAMPLES_TO_SHOW}] cot={row['cot_id']} "
        f"split {row['split_index']} of {row['n_sentences']} sentences\n"
        f"  Q (judge R1):        {clip(row['prompt'], 220)}\n"
        f"  target (judge R3):   {clip(row['target_response'], 240)}\n"
        f"  blackbox (judge R2): {clip(row['bb_response'], 160)}",
        flush=True,
    )


def _strip_think(text: str) -> str:
    """Remove a leading <think>...</think> block if the judge emitted one."""
    return re.sub(r"^\s*<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()


async def _chat(client: AsyncOpenAI, sem: asyncio.Semaphore, judge_cfg: dict,
                system: str, user: str, max_tokens: int = 400) -> str:
    async with sem:
        for attempt in range(5):
            try:
                resp = await client.chat.completions.create(
                    model=judge_cfg["served_name"],
                    messages=[{"role": "system", "content": system},
                              {"role": "user", "content": user}],
                    max_tokens=max_tokens,
                    temperature=0.0,
                    extra_body={"chat_template_kwargs": {"enable_thinking": judge_cfg.get("enable_thinking", False)}},
                )
                return _strip_think(resp.choices[0].message.content or "")
            except Exception as e:
                if attempt == 4:
                    print(f"  [convqa] giving up after 5 attempts: {e}")
                    return ""
                await asyncio.sleep(5 * (attempt + 1))


async def _process_cot(client, sem, judge_cfg, cot: dict) -> dict | None:
    """Run R1 -> (R2, R3) for one CoT; returns a dataset row or None."""
    sents = cot["sentences"]
    r1_text = await _chat(client, sem, judge_cfg, _prompts.R1_SYSTEM,
                          _prompts.build_r1_user(cot["question"], sents))
    parsed = _prompts._parse_r1_json(r1_text)
    if parsed is None:
        return None
    s, question = parsed["split_index"], parsed["question"]
    if not (3 <= s + 1 <= len(sents) - 3):
        return None
    prefix = " ".join(sents[: s + 1])
    suffix = " ".join(sents[s + 1:])
    bb_task = _chat(client, sem, judge_cfg, _prompts.R2_SYSTEM,
                    _prompts.build_r2_user(cot["question"], prefix, question))
    gt_task = _chat(client, sem, judge_cfg, _prompts.R3_SYSTEM,
                    _prompts.build_r3_user(cot["question"], suffix, question))
    bb_resp, gt_resp = await asyncio.gather(bb_task, gt_task)
    if not gt_resp:
        return None
    row = {
        "cot_id": cot["id"], "source": cot["source"], "question": cot["question"],
        "cot_text": " ".join(sents),
        "cot_prefix": prefix, "cot_suffix": suffix,
        "split_index": s, "n_sentences": len(sents),
        "prompt": question, "target_response": gt_resp, "bb_response": bb_resp,
        "generation_prompt": "",
    }
    _preview_row(row)
    return row


async def _run(corpus_path, out_dir, n_rows, test_frac, concurrency, judge_cfg, seed=0):
    """Generate convqa rows in durable, resumable batches, then write parquet.

    Mechanism (mirrors the corpus generator so a kill loses at most one batch):
      - Read+filter CoTs from the corpus, shuffle deterministically (fixed seed),
        so the processing order is stable across runs (required for resume).
      - Process BATCH CoTs at a time (each fans out up to `concurrency` judge calls
        via the semaphore). Kept rows are appended to a durable `rows.jsonl` shard;
        after each batch we fsync and atomically update a sidecar
        "<cots_done> <rows_kept> <total_cots> <byte_offset>".
      - Resume: if the sidecar matches this corpus (same total), truncate the shard
        back to the last fsynced offset (dropping any partial batch) and continue
        from the recorded CoT index / kept count.
      - Stop early once `n_rows` rows are kept (R1 yield is ~80-90%, so we rarely
        need the whole corpus), then split the shard into train/test parquet.
    """
    cots = []
    with open(corpus_path) as f:
        for line in f:
            row = json.loads(line)
            if row.get("n_sentences", 0) >= MIN_SENTENCES:
                cots.append({k: row[k] for k in ("id", "source", "question", "sentences")})
    random.Random(seed).shuffle(cots)
    total = len(cots)

    out_dir.mkdir(parents=True, exist_ok=True)
    shard = out_dir / "rows.jsonl"
    progress_path = str(shard) + ".progress"
    BATCH = 1000

    start, kept, resume_offset = 0, 0, None
    if shard.exists() and os.path.exists(progress_path):
        try:
            done_str, kept_str, total_str, off_str = Path(progress_path).read_text().split()
            if int(total_str) == total:
                start, kept, resume_offset = int(done_str), int(kept_str), int(off_str)
        except (ValueError, IndexError):
            start, kept, resume_offset = 0, 0, None

    if start > 0 and resume_offset is not None:
        os.truncate(shard, resume_offset)  # drop any partial batch from a crash
        sf = open(shard, "a")
        print(f"[convqa] resuming: {start}/{total} CoTs done, {kept} rows kept")
    else:
        start, kept = 0, 0
        sf = open(shard, "w")

    client = AsyncOpenAI(base_url=judge_base_url(load_config()), api_key="unused")
    sem = asyncio.Semaphore(concurrency)
    print(f"[convqa] {total} CoTs available (target {n_rows} rows); batch={BATCH}, starting at {start}")

    try:
        idx = start
        while idx < total and kept < n_rows:
            batch = cots[idx: idx + BATCH]
            results = await asyncio.gather(*[_process_cot(client, sem, judge_cfg, c) for c in batch])
            for r in results:
                if r is not None and kept < n_rows:
                    sf.write(json.dumps(r) + "\n")
                    kept += 1
            sf.flush()
            os.fsync(sf.fileno())
            idx += len(batch)
            tmp = progress_path + ".tmp"
            Path(tmp).write_text(f"{idx} {kept} {total} {sf.tell()}")
            os.replace(tmp, progress_path)
            print(f"[convqa] processed {idx}/{total} CoTs, kept {kept}/{n_rows} rows")
    finally:
        sf.close()

    # Build train/test parquet from the durable shard.
    rows = []
    with open(shard) as f:
        for line in f:
            rows.append(json.loads(line))
    rows = rows[:n_rows]
    print(f"[convqa] kept {len(rows)} rows ({len(rows) / max(total, 1) * 100:.0f}% of available CoTs)")

    df = pd.DataFrame(rows)
    n_test = max(1, int(len(df) * test_frac))
    df.iloc[n_test:].to_parquet(out_dir / "train.parquet")
    df.iloc[:n_test].to_parquet(out_dir / "test.parquet")
    print(f"[convqa] wrote {out_dir}/train.parquet ({len(df) - n_test}) and test.parquet ({n_test})")


def _word_overlap(a: str, b: str) -> float:
    """Jaccard overlap of word sets — a cheap proxy for 'same answer'."""
    wa = set(re.findall(r"\w+", a.lower()))
    wb = set(re.findall(r"\w+", b.lower()))
    return len(wa & wb) / max(len(wa | wb), 1)


def _build_filtered_parquet(out_dir, test_frac: float, seed: int) -> None:
    """Rebuild train/test.parquet from the durable rows.jsonl shard, dropping
    low-quality convqa rows. Pure post-processing — needs no judge and never
    regenerates, so it is safe to run after (or instead of) generation.

    Filters, in order:
      1. scaffold leak — `prompt` mentions prefix/suffix/split: unusable at
         inference (the AO is given activations, not a "suffix"), so drop.
      2. answer leak   — R2(prefix-only) ≈ R3(suffix) by word overlap > LEAK_OVERLAP:
         the prefix text already reveals the answer ⇒ activations add no signal.
      3. negative cap  — keep every contentful target, but cap "the reasoning
         does not…" absence labels at NEG_TARGET_FRAC of the final set.
    """
    shard = out_dir / "rows.jsonl"
    if not shard.exists():
        raise SystemExit(f"no shard at {shard} — run `make convqa` first")

    rows = []
    with open(shard) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    n0 = len(rows)

    rows = [r for r in rows if not _SCAFFOLD_RE.search(r["prompt"])]
    n1 = len(rows)
    rows = [r for r in rows if _word_overlap(r["bb_response"], r["target_response"]) <= LEAK_OVERLAP]
    n2 = len(rows)

    pos = [r for r in rows if not _NEG_RE.search(r["target_response"])]
    neg = [r for r in rows if _NEG_RE.search(r["target_response"])]
    rnd = random.Random(seed)
    rnd.shuffle(neg)
    max_neg = int(NEG_TARGET_FRAC / (1 - NEG_TARGET_FRAC) * len(pos))
    dropped_neg = len(neg) - min(len(neg), max_neg)
    neg = neg[:max_neg]
    kept = pos + neg
    rnd.shuffle(kept)

    print(f"[convqa --filter] {n0} rows -> {len(kept)} kept")
    print(f"  scaffold-leak dropped:  {n0 - n1}")
    print(f"  answer-leak dropped:    {n1 - n2}")
    print(f"  negative-target capped: {dropped_neg} dropped "
          f"({len(neg)} kept = {len(neg) / max(len(kept), 1):.0%} of final)")

    df = pd.DataFrame(kept)
    n_test = max(1, int(len(df) * test_frac))
    df.iloc[n_test:].to_parquet(out_dir / "train.parquet")
    df.iloc[:n_test].to_parquet(out_dir / "test.parquet")
    print(f"[convqa --filter] wrote {out_dir}/train.parquet ({len(df) - n_test}) "
          f"and test.parquet ({n_test})")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--filter", action="store_true",
                   help="rebuild train/test.parquet from rows.jsonl with quality "
                        "filters (no generation, no judge needed)")
    p.add_argument("--solvability-filter", action="store_true",
                   help="C4: keep only train.parquet rows an early AO answers better WITH the "
                        "activation than without (writes train_solvable.parquet)")
    p.add_argument("--limit", type=int, default=None, help="cap rows scored by --solvability-filter")
    args = p.parse_args(argv)

    cfg = load_config()
    model = cfg["model"]["smoke_name"] if args.smoke else cfg["model"]["name"]
    art = artifacts_dir(model)
    out_dir = art / "convqa"

    if args.filter:
        _build_filtered_parquet(out_dir, cfg["data"]["convqa"]["test_frac"], cfg["training"]["seed"])
        return

    if args.solvability_filter:
        # Free the GPU from the judge first (the scorer loads the base model + AO).
        from .judge import down as judge_down
        from .solvability import filter_parquet

        judge_down()
        solv = cfg["contributions"]["solvability_filter"]
        layers, _percents = resolve_layers(model, cfg["layers"]["center_percent"], cfg["layers"]["n_layers"])
        filter_parquet(
            in_parquet=out_dir / "train.parquet",
            out_parquet=out_dir / "train_solvable.parquet",
            model_name=model,
            lora_path=solv["lora"],
            layers=layers,
            hook_onto_layer=cfg["injection"]["hook_onto_layer"],
            steering_coefficient=cfg["injection"]["eval_steering_coefficient"],
            delta=float(solv["delta"]),
            embed_model=solv["embed_model"],
            seed=cfg["training"]["seed"],
            limit=args.limit,
        )
        return

    n_rows = cfg["smoke"]["convqa_n_rows"] if args.smoke else cfg["data"]["convqa"]["n_rows"]
    corpus = art / "corpus" / "corpus.jsonl"
    if not corpus.exists():
        raise SystemExit(f"corpus not found at {corpus} — run `make corpus` first")

    asyncio.run(_run(
        corpus_path=corpus,
        out_dir=out_dir,
        n_rows=n_rows,
        test_frac=cfg["data"]["convqa"]["test_frac"],
        concurrency=cfg["data"]["convqa"]["max_concurrent"],
        judge_cfg=cfg["judge"],
        seed=cfg["training"]["seed"],
    ))


if __name__ == "__main__":
    main()
