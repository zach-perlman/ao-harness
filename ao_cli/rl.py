"""Phase-2 post-training for the AO — C1 (anti-inversion, DPO) and C2 (abstention, GRPO).

Gears-level overview
--------------------
`make rl EXP=<name>` post-trains an SFT checkpoint (rl.init_lora) and writes the
result to checkpoints/<EXP>/final, so `make eval EXP=<name>` compares it to the
baseline like any other contribution. Exactly one of contributions.{swap_test,
abstention} must be enabled; that choice selects the method:

  swap_test (C1) — anti-text-inversion via OFFLINE DPO. We mine the same content
      token in two different corpus contexts, take the AO's own answer to each,
      and build preference pairs that prefer the answer a context's activation
      actually elicits over the other context's answer. Sharpens context-
      sensitivity with a frozen-reference KL and no game-able online reward.
      (See nl_probes/preference/injection_dpo for the rationale.)

  abstention (C2) — calibrated abstention via hardened GRPO. Answerable (matched)
      and synthetically-unanswerable (activation mismatched to a topically-near
      question) items get a verifiable TERNARY reward: reward correct-confident
      answers when answerable and abstention when not, punishing confident-wrong
      hardest. A short warm-start SFT seeds abstention mass first (cold-start fix),
      then GRPO refines WHEN to abstain. (Reward in nl_probes/grpo/rewards.)

The injection/optimization primitives live in nl_probes/{grpo,preference}; this
module only builds the data sources and reward/pair closures.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

from . import (
    PROJECT_ROOT,
    artifacts_dir,
    experiment_name,
    load_config,
    resolve_layers,
    run_dir_name,
)

def _free_judge_gpu(cfg: dict) -> None:
    """Stop the vLLM judge service so RL training gets the whole GPU.

    Eval brings the judge up and deliberately leaves it running (warm reuse across
    evals/sweeps), so it still holds its gpu_memory_utilization slice (~0.65 of
    VRAM) when a training stage follows — which OOMs the trainer. C1/C2 rewards are
    embedder-based and never call the LLM judge, so we reclaim that slice here.
    Idempotent: probes first and no-ops if the judge is down; tolerant if
    supervisor isn't present (never let teardown block training).
    """
    import time
    from . import judge
    try:
        if not judge._probe(cfg["judge"]["port"]):
            return
        print("[rl] vLLM judge is up and holding GPU memory — stopping it to free VRAM for training")
        judge.down()
        for _ in range(60):                                 # confirm the process released VRAM
            if not judge._probe(cfg["judge"]["port"]):
                break
            time.sleep(1)
    except Exception as e:  # noqa: BLE001
        print(f"[rl] judge teardown skipped ({type(e).__name__}: {e})")


NEUTRAL_QUERY = "What concept or information is this activation encoding? Answer in a few words."
ABSTAIN_OPTION = ("\n\nIf the provided information does not actually contain the answer, "
                  "reply that you cannot tell.")
_STOPWORDS = set("the a an of to in on at and or but is are was were be been being it its this that "
                 "these those for with as by from into about over after we you they he she i".split())


def _resolve_lora(path: str) -> str:
    p = Path(path)
    return str(p if p.is_absolute() else PROJECT_ROOT / p)


def _finalize_checkpoint(ckpt_dir: str, init_lora: str) -> None:
    """Make a saved adapter dir directly eval-loadable.

    DPO/GRPO only save the adapter weights; AObench additionally needs the AO
    training config (layers, injection settings). The architecture is unchanged by
    post-training, so carry `ao_config.json` over from the SFT start point. Also
    drop the frozen `reference` adapter subfolder that save_pretrained emits — eval
    loads the root (trained) adapter, so the copy is just confusing clutter."""
    import shutil
    from nl_probes.grpo.injection_grpo import REF_ADAPTER
    d = Path(ckpt_dir)
    if not d.is_dir():
        return
    src = Path(init_lora) / "ao_config.json"
    if src.exists():
        shutil.copy(src, d / "ao_config.json")
    ref = d / REF_ADAPTER
    if ref.is_dir():
        shutil.rmtree(ref)


def _save_dpo_run_meta(ckpt_dir: Path, experiment: str | None, **params) -> None:
    """Record the DPO run's hyperparameters alongside the saved adapter.

    Writes `dpo_run.json` into the checkpoint dir so a finished model is
    self-documenting (beta/epochs/batch/data knobs + the resolved values that
    env overrides or config defaults produced). Without this the only trace of,
    say, `AO_DPO_BETA` is the scrolled-away stdout line."""
    from datetime import datetime, timezone
    if not ckpt_dir.is_dir():
        return
    meta = {"experiment": experiment or "main",
            "saved_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            **params}
    (ckpt_dir / "dpo_run.json").write_text(json.dumps(meta, indent=2))
    print(f"[rl] wrote run metadata → {ckpt_dir / 'dpo_run.json'}")


# --- data sources -------------------------------------------------------------
# Mine from EXACTLY what training used: read the rendered train_config.json so the
# RL corpus/convqa never diverge from the SFT mixture, whether that resolved to a
# released HF dataset (use_released_datasets=true) or local on-policy artifacts.

def _resolve_sources(art) -> tuple[str, str, str]:
    """Return (corpus_source, corpus_key, convqa_source) — each an HF id or local path."""
    tc = json.loads((art / "train_config.json").read_text())
    corpus_src = corpus_key = convqa_src = None
    for e in tc["dataset_configs"]:
        p = e.get("custom_dataset_params", {})
        if e["dataset_name"] == "past_lens":
            corpus_src = p.get("pretrain_dataset")
            corpus_key = p.get("pretrain_key") or "cot_response"
        elif e["dataset_name"] == "cot_oracle_convqa":
            convqa_src = p.get("hf_dataset_repo")
    corpus_src = corpus_src or str(art / "corpus" / "corpus.jsonl")
    convqa_src = convqa_src or str(art / "convqa" / "train.parquet")
    return corpus_src, corpus_key or "cot_response", convqa_src


def _load_corpus_texts(source: str, key: str, limit: int) -> list[str]:
    """Up to `limit` reasoning texts from a local jsonl or an HF dataset."""
    if Path(source).exists():
        out = []
        with open(source) as f:
            for line in f:
                rec = json.loads(line)
                t = rec.get(key) or rec.get("cot_response") or rec.get("text")
                if t:
                    out.append(t)
                if len(out) >= limit:
                    break
        return out
    from datasets import load_dataset
    ds = load_dataset(source, split="train")
    ds = ds.select(range(min(limit, len(ds))))
    return [r[key] for r in ds if r.get(key)]


def _load_convqa_rows(source: str, limit: int) -> list[dict]:
    """Up to `limit` convqa rows (cot_prefix/prompt/target_response) from parquet or HF."""
    if Path(source).exists():
        import pandas as pd
        return pd.read_parquet(source).head(limit).to_dict("records")
    from datasets import load_dataset
    ds = load_dataset(source, split="train")
    return ds.select(range(min(limit, len(ds)))).to_list()


def _build_dp(tokenizer, ids, positions, query, layers, target=""):
    """One injected datapoint. target="" → prompt-only (GRPO/generation); else SFT."""
    from nl_probes.utils.dataset_utils import create_training_datapoint
    return create_training_datapoint(
        datapoint_type="grpo", prompt=query, target_response=target, layers=layers,
        num_positions=len(positions), tokenizer=tokenizer, acts_BD=None, feature_idx=-1,
        context_input_ids=ids, context_positions=positions, ds_label=None, meta_info={})


def _gen_in_chunks(model, tokenizer, dps, submodule, steer, k, gen, device, dtype, chunk):
    """Materialize + sample k rollouts in small batches so a single forward never
    has to hold activations for the whole (potentially huge) datapoint set at once
    — the source of the 44 GiB OOM when all contexts were batched together."""
    from nl_probes.grpo.injection_grpo import generate_grouped_rollouts, materialize_prompts
    pdps, groups = [], []
    for i in range(0, len(dps), chunk):
        sub = materialize_prompts(dps[i:i + chunk], tokenizer, model, device)
        groups.extend(generate_grouped_rollouts(
            model, tokenizer, sub, submodule, steer, k, gen, device, dtype))
        pdps.extend(sub)
    return pdps, groups


# =========================================================================== #
# C1 — swap-test → offline DPO
# =========================================================================== #

def _mine_token_occurrences(texts, tokenizer, max_ctx_tokens):
    """Index content tokens that recur across distinct corpus lines (the swap pool)."""
    lines: list[list[int]] = []
    occ: dict[int, list[tuple[int, int]]] = {}
    for text in texts:
        if not text:
            continue
        ids = tokenizer(text, add_special_tokens=False)["input_ids"][:max_ctx_tokens]
        li = len(lines)
        lines.append(ids)
        for pos in range(2, len(ids)):
            tok = ids[pos]
            word = tokenizer.decode([tok]).strip().lower()
            if len(word) >= 3 and word.isalpha() and word not in _STOPWORDS:
                occ.setdefault(tok, []).append((li, pos))
    pairable = {t: o for t, o in occ.items() if len({li for li, _ in o}) >= 2}
    if not pairable:
        raise SystemExit("[rl] swap_test: no content token recurs across distinct corpus lines")
    return lines, pairable


def _two_distinct_lines(occ, rng):
    a = rng.choice(occ)
    b = rng.choice([o for o in occ if o[0] != a[0]])
    return a, b


def _build_dpo_pairs(model, tokenizer, submodule, steer, layers, device, dtype, embedder,
                     texts, rng, *, n_tokens, pairs_per_token, distinct_threshold,
                     gen_temperature, max_ctx_tokens):
    """Generate the AO's answer for each sampled (token,context) once, then form
    distinct, context-matched preference pairs (chosen=own-context, rejected=other)."""
    from nl_probes.preference.injection_dpo import DPOPair

    lines, pairable = _mine_token_occurrences(texts, tokenizer, max_ctx_tokens)
    tokens = rng.sample(list(pairable), min(n_tokens, len(pairable)))

    # Collect the unique occurrences we need to generate answers for.
    # specs carry their source token so we can report per-token yield below.
    specs: list[tuple[int, tuple[int, int], tuple[int, int]]] = []
    needed: dict[tuple[int, int], None] = {}
    for tok in tokens:
        for _ in range(pairs_per_token):
            a, b = _two_distinct_lines(pairable[tok], rng)
            specs.append((tok, a, b))
            needed[a] = needed[b] = None
    occs = list(needed)

    dps = [_build_dp(tokenizer, lines[li], [pos], NEUTRAL_QUERY, layers) for (li, pos) in occs]
    gen = {"do_sample": gen_temperature > 0, "temperature": max(gen_temperature, 1e-5),
           "top_p": 0.95, "max_new_tokens": 48}
    prompt_dps, groups = _gen_in_chunks(
        model, tokenizer, dps, submodule, steer, 1, gen, device, dtype, chunk=32)
    ans_ids = {occ: groups[i][0] for i, occ in enumerate(occs)}
    pdp = {occ: prompt_dps[i] for i, occ in enumerate(occs)}
    ans_txt = {occ: tokenizer.decode(ans_ids[occ], skip_special_tokens=True) for occ in occs}

    # Keep only informative pairs: the two answers must differ (else the chosen↔
    # rejected contrast is empty and DPO gets no gradient). While filtering we also
    # tally the *yield* — what fraction of mined (token, context-pair) candidates
    # actually gave the context-dependent reading the swap test is hunting for —
    # so you can see whether the corpus surfaced the contexts we were looking for.
    embs = {occ: embedder.embed([ans_txt[occ] or " "])[0] for occ in occs}
    pairs: list[DPOPair] = []
    n_empty = 0                          # dropped: AO produced an empty answer
    scores: list[float] = []             # distinctness (1-cos) over non-empty candidates
    tokens_hit: set[int] = set()         # tokens that yielded ≥1 contrastive pair
    examples: list[tuple[str, float, str, str]] = []
    for tok, a, b in specs:
        if not ans_txt[a].strip() or not ans_txt[b].strip():
            n_empty += 1
            continue
        dist = 1.0 - float((embs[a] * embs[b]).sum())
        scores.append(dist)
        if dist < distinct_threshold:                        # answers too similar → no signal
            continue
        tokens_hit.add(tok)
        if len(examples) < 6:
            examples.append((tokenizer.decode([tok]).strip(), dist,
                             ans_txt[a].strip()[:70], ans_txt[b].strip()[:70]))
        for hi, lo in ((a, b), (b, a)):                       # symmetric: each context prefers its own answer
            pairs.append(DPOPair(prompt_ids=pdp[hi].input_ids, positions=pdp[hi].positions,
                                 steering_vectors=pdp[hi].steering_vectors,
                                 chosen_ids=ans_ids[hi], rejected_ids=ans_ids[lo]))
    _report_swap_yield(specs, pairs, scores, n_empty, tokens, tokens_hit,
                       pairable, lines, distinct_threshold, examples)
    if not pairs:
        raise SystemExit("[rl] swap_test/DPO: no distinct pairs survived the filter — "
                         "lower distinct_threshold or raise gen_temperature")
    return pairs


def _report_swap_yield(specs, pairs, scores, n_empty, tokens, tokens_hit,
                       pairable, lines, threshold, examples):
    """Print how productive the swap-test mining was: of the candidate context
    pairs we tried, how many were genuinely contrastive (= the contexts we wanted)."""
    import statistics
    n_specs, kept = len(specs), len(pairs) // 2
    rate = 100 * kept / max(n_specs, 1)
    print(f"[dpo] swap-test yield: {kept}/{n_specs} candidate context-pairs were "
          f"contrastive ({rate:.0f}%) → {len(pairs)} DPO training pairs")
    print(f"[dpo]   token coverage: {len(tokens_hit)}/{len(tokens)} sampled tokens "
          f"produced ≥1 contrastive pair  (pool={len(pairable)} pairable tokens "
          f"over {len(lines)} lines)")
    if scores:
        s = sorted(scores)
        q = lambda p: s[min(len(s) - 1, int(p * len(s)))]
        print(f"[dpo]   distinctness 1-cos: median={statistics.median(s):.3f} "
              f"p10={q(0.10):.3f} p90={q(0.90):.3f}  (keep ≥ {threshold}); "
              f"{n_empty} empty-answer drops")
    for w, d, ca, cb in examples:                            # eyeball that the same token reads differently
        print(f"[dpo]   e.g. '{w}' Δ={d:.2f}  ctxA→ {ca!r}  |  ctxB→ {cb!r}")


def _load_or_mine_pairs(cache_dir: Path, key: dict, device, dtype, mine):
    """Reuse a mined DPO dataset across runs whose *data-relevant* config matches.

    Why this exists: pair mining runs the model to generate an answer for every
    sampled (token, context) — the slow part of `make rl`. But the mined dataset
    depends only on {corpus, n_tokens, pairs_per_token, distinct_threshold,
    gen_temperature, embed_model, swap_ctx, seed, init_lora} — NOT on beta/lr/epochs. So a beta
    sweep would otherwise re-mine the identical dataset once per cell. We key a
    cache on exactly the data-relevant knobs (beta excluded) → mine once, the
    later cells reload from disk.

    Safety: any cache miss/corruption/incompatibility falls back to a fresh mine
    (and a save is best-effort), so caching can only speed runs up — never change
    results or break a run. Set AO_DPO_NOCACHE=1 to force a fresh mine.
    """
    import hashlib
    import torch

    digest = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:16]
    path = cache_dir / f"dpo_pairs_{digest}.pt"
    if os.environ.get("AO_DPO_NOCACHE"):
        print("[dpo] cache: bypassed (AO_DPO_NOCACHE set) — mining fresh")
    elif path.exists():
        try:
            blob = torch.load(path, map_location="cpu", weights_only=False)
            pairs = blob["pairs"]
            for p in pairs:                              # mined on GPU; restore device/dtype
                p.steering_vectors = p.steering_vectors.to(device=device, dtype=dtype)
            if pairs:
                print(f"[dpo] cache HIT: reusing {len(pairs)} pairs from {path.name} "
                      f"— skipped mining (beta/lr not in cache key)")
                return pairs
            print(f"[dpo] cache {path.name} was empty — re-mining")
        except Exception as e:                          # corrupt / version-mismatch / etc.
            print(f"[dpo] cache load failed ({e!r}) — re-mining")
    pairs = mine()
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"meta": key, "pairs": pairs}, path)
        print(f"[dpo] cache: saved {len(pairs)} pairs → {path.name}")
    except Exception as e:
        print(f"[dpo] cache save failed ({e!r}) — continuing (dataset not persisted)")
    return pairs


# =========================================================================== #
# C2 — abstention scenario source (shared by warm-start + GRPO)
# =========================================================================== #

class AbstentionSource:
    """Draws answerable (matched) / unanswerable (topically-near mismatch) items.

    Near-mismatch (a question semantically close to the activation's own, but from
    a different row) prevents the model from solving "unanswerable" with a cheap
    distribution-shift detector — it must check whether the CONTENT is present.
    """

    def __init__(self, rows, tokenizer, layers, rng, embedder, *,
                 answerable_prob, max_ctx_tokens, near_mismatch):
        rows = [r for r in rows if r.get("cot_prefix") and r.get("prompt") and r.get("target_response")]
        if len(rows) < 2:
            raise SystemExit("[rl] abstention: need ≥2 convqa rows with cot_prefix + prompt + target_response")
        self.rows, self.tok, self.layers, self.rng = rows, tokenizer, layers, rng
        # `answerable_prob` ∈ (0,1): P(draw an answerable scenario). <0.5 ⇒ unanswerable-
        # heavy, which we use to concentrate GRPO's signal on the failing axis (recall on
        # unanswerable) since the policy over-answers.
        self.answerable_prob, self.max_ctx, self.near = float(answerable_prob), max_ctx_tokens, near_mismatch
        # question-embedding index for nearest-neighbour mismatch selection
        self.q_emb = embedder.embed([r["prompt"] for r in rows]) if near_mismatch else None

    def _ctx(self, row):
        from nl_probes.dataset_classes.position_sampling import sample_cot_oracle_token_positions
        ids = self.tok(row["cot_prefix"], add_special_tokens=False)["input_ids"][:self.max_ctx]
        positions = sample_cot_oracle_token_positions(len(ids), self.rng, max_k=100)
        return ids, positions

    def _mismatch_idx(self, act_idx):
        if not self.near:
            j = self.rng.randrange(len(self.rows))
            return j if j != act_idx else (j + 1) % len(self.rows)
        sims = (self.q_emb @ self.q_emb[act_idx]).tolist()       # cosine to every question
        order = sorted(range(len(sims)), key=lambda j: -sims[j])
        nbrs = [j for j in order if j != act_idx][:10]           # 10 nearest, then pick one
        return self.rng.choice(nbrs)

    def draw(self):
        """Return (ids, positions, query, answerable, gold)."""
        ai = self.rng.randrange(len(self.rows))
        act = self.rows[ai]
        ids, positions = self._ctx(act)
        answerable = self.rng.random() < self.answerable_prob
        if answerable:
            return ids, positions, act["prompt"] + ABSTAIN_OPTION, True, act["target_response"]
        other = self.rows[self._mismatch_idx(ai)]
        return ids, positions, other["prompt"] + ABSTAIN_OPTION, False, None

    def grpo_sampler(self):
        def sampler(n: int) -> list[dict]:
            out = []
            for _ in range(n):
                ids, pos, query, answerable, gold = self.draw()
                dp = _build_dp(self.tok, ids, pos, query, self.layers)
                out.append({"datapoints": [dp], "answerable": answerable, "gold": gold})
            return out
        return sampler

    def seed_datapoints(self, n: int, abstain_text: str, answerable_prob: float | None = None):
        """Full SFT datapoints for the cold-start warm-start: answerable→gold answer,
        unanswerable→an explicit abstention. Seeds nonzero abstention probability so
        GRPO has variance to learn from (otherwise a never-abstaining policy gives
        all-equal rewards on unanswerable groups → zero gradient).

        `answerable_prob` (optional) overrides P(answerable) for the SEED only. The
        SFT model massively over-answers, so we seed abstention-heavy (low prob) to
        plant a strong abstention prior; GRPO's answerable reward pressure (+1 correct,
        −0.5 over-abstain) then pulls answering back to the right boundary. This is the
        easier direction to learn (always-abstain→when-to-answer has variance
        everywhere) than the reverse (always-answer→discover-abstention is zero-variance)."""
        prev = self.answerable_prob
        if answerable_prob is not None:
            self.answerable_prob = float(answerable_prob)
        try:
            dps = []
            for _ in range(n):
                ids, pos, query, answerable, gold = self.draw()
                target = gold if answerable else abstain_text
                dps.append(_build_dp(self.tok, ids, pos, query, self.layers, target=target))
            return dps
        finally:
            self.answerable_prob = prev


def _warmstart_sft(model, tokenizer, submodule, steer, seed_dps, device, dtype, *, steps, batch_size, lr):
    """A few injected-SFT steps over the seed datapoints (teacher-forced cross-entropy)."""
    import torch
    from nl_probes.utils.dataset_utils import construct_batch, materialize_missing_steering_vectors
    from nl_probes.utils.steering_hooks import add_hook, get_hf_activation_steering_hook

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    model.train()
    print(f"[rl] warm-start SFT: {steps} steps over {len(seed_dps)} seed items (bs={batch_size})")
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_skip_loss = n_skip_grad = 0
    for step in range(1, steps + 1):
        chunk = [seed_dps[(step * batch_size + i) % len(seed_dps)] for i in range(batch_size)]
        # Materialize just this minibatch's steering vectors (not all seeds at once).
        chunk = materialize_missing_steering_vectors(chunk, tokenizer, model)
        batch = construct_batch(chunk, tokenizer, device)
        hook = get_hf_activation_steering_hook(
            vectors=batch.steering_vectors, positions=batch.positions,
            steering_coefficient=steer, device=device, dtype=dtype)
        opt.zero_grad()
        with add_hook(submodule, hook):
            out = model(input_ids=batch.input_ids, attention_mask=batch.attention_mask, labels=batch.labels)
        # Two-stage finiteness guard so one bad minibatch can't poison the adapter
        # (a nan weight later makes GRPO sample from nan logits and die):
        #   1) nan/inf LOSS — e.g. all-masked labels (0/0). Skip before backward.
        #   2) finite loss but nan/inf GRADIENT — bf16 overflow in the steering-hook
        #      backward. clip_grad_norm_ returns the pre-clip total norm; if it's
        #      non-finite the grads are poisoned, so skip the optimizer step.
        if not torch.isfinite(out.loss):
            n_skip_loss += 1
            continue
        out.loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        if not torch.isfinite(gnorm):
            n_skip_grad += 1
            continue
        opt.step()
        if step % 25 == 0 or step == 1:
            print(f"    warmstart {step}/{steps}  ce_loss={float(out.loss):.4f}", flush=True)
    if n_skip_loss or n_skip_grad:
        print(f"[rl] warm-start: skipped {n_skip_loss} nan-loss + {n_skip_grad} nan-grad "
              f"steps of {steps} — adapter kept finite", flush=True)


# --------------------------------------------------------------------------- #
# Warm-start adapter cache
# --------------------------------------------------------------------------- #
# The warm-start SFT is a deterministic-recipe (but stochastic-draw) seeding pass
# that is identical across GRPO experiments — so re-running it every time both
# wastes minutes AND injects a DIFFERENT starting policy into each run (warm-start
# is random), confounding a kl_coef / lr sweep. We therefore cache the warm-started
# `default` LoRA weights keyed by the inputs that determine them; later runs with a
# matching key load the cached adapter and jump straight to GRPO from an IDENTICAL
# starting point. Set AO_WARMSTART_REFRESH=1 to force a fresh warm-start (e.g. after
# the SFT checkpoint changes — the key uses init_lora's path + mtime to catch that).

def _warmstart_cache_key(payload: dict) -> str:
    import hashlib
    import json
    return hashlib.sha1(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _save_warmstart(model, cache_dir, meta: dict) -> None:
    import json
    import torch
    from peft import get_peft_model_state_dict
    cache_dir.mkdir(parents=True, exist_ok=True)
    sd = {k: v.to("cpu") for k, v in get_peft_model_state_dict(model, adapter_name="default").items()}
    torch.save(sd, cache_dir / "default_adapter.pt")
    (cache_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str))


def _load_warmstart(model, cache_dir) -> None:
    import torch
    from peft import set_peft_model_state_dict
    sd = torch.load(cache_dir / "default_adapter.pt", map_location="cpu")
    set_peft_model_state_dict(model, sd, adapter_name="default")


# =========================================================================== #
# Orchestration
# =========================================================================== #

def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--max-steps", type=int, default=None, help="override rl.max_steps (C2)")
    args = p.parse_args(argv)

    import torch
    from peft import PeftModel

    from nl_probes.utils.common import load_model, load_tokenizer
    from nl_probes.utils.activation_utils import get_hf_submodule
    from nl_probes.grpo.rewards import ABSTAIN_PROTOTYPES, SentenceEmbedder, abstention_reward

    cfg = load_config()
    contrib = cfg["contributions"]
    swap_on = contrib["swap_test"]["enabled"]
    abst_on = contrib["abstention"]["enabled"]
    # AO_CONTRIB={swap_test|abstention} overrides the config toggles for this run only.
    # Lets C1 and C2 be chained in one shell command without rewriting config.yaml
    # (a YAML round-trip would strip its comments). Config stays the source of truth.
    pick = os.environ.get("AO_CONTRIB")
    if pick:
        if pick not in ("swap_test", "abstention"):
            raise SystemExit(f"[rl] AO_CONTRIB must be swap_test or abstention, got {pick!r}")
        swap_on, abst_on = pick == "swap_test", pick == "abstention"
        print(f"[rl] AO_CONTRIB override → running {pick} only")
    if swap_on == abst_on:
        raise SystemExit("[rl] enable EXACTLY one of contributions.swap_test / contributions.abstention")

    model_name = cfg["model"]["name"]
    art = artifacts_dir(model_name)
    layers, _ = resolve_layers(model_name, cfg["layers"]["center_percent"], cfg["layers"]["n_layers"])
    rlc = cfg["rl"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16
    rng = random.Random(cfg["training"]["seed"])
    # Inject at the TRAIN-time strength (1.0), not the eval-time sweep value (2.0).
    # Phase-2 RL continues training the AO, so injections must match the SFT
    # distribution it learned (config: "the AO would learn 2x-norm injections it
    # won't see at eval"). Eval still sweeps to 2.0 via AO_EVAL_STEERING_COEFFICIENT;
    # the AO generalizes to the stronger injection, per the paper recipe.
    steer = cfg["injection"]["steering_coefficient"]
    # RL forwards are memory-bound (B × L activations); cap the convqa prefix length.
    max_ctx = min(cfg["data"]["convqa"].get("max_cot_prefix_tokens", 1024), 1024)
    SWAP_CTX = 96  # C1 mining only needs a short window of left-context per token

    corpus_src, corpus_key, convqa_src = _resolve_sources(art)

    _free_judge_gpu(cfg)
    tokenizer = load_tokenizer(model_name)
    base = load_model(model_name, dtype)
    init_lora = _resolve_lora(rlc["init_lora"])
    model = PeftModel.from_pretrained(base, init_lora, is_trainable=True)
    # A second, FROZEN copy of the SFT adapter is the reference for the KL / DPO
    # anchor. The AO is the LoRA, so disabling adapters would anchor to the base
    # model (wrong); this keeps the anchor at the supervised policy we start from.
    from nl_probes.grpo.injection_grpo import REF_ADAPTER
    model.load_adapter(init_lora, adapter_name=REF_ADAPTER, is_trainable=False)
    model.set_adapter("default")
    model.enable_input_require_grads()
    submodule = get_hf_submodule(model, cfg["injection"]["hook_onto_layer"])
    embedder = SentenceEmbedder(rlc["embed_model"], device)

    save_dir = art / "checkpoints" / run_dir_name(model_name)
    save_dir.mkdir(parents=True, exist_ok=True)
    method = "swap_test→DPO (C1)" if swap_on else "abstention→GRPO (C2)"
    print(f"[rl] {method} from {rlc['init_lora']} → {save_dir} (EXP={experiment_name() or 'main'})")

    # ---- C1: offline DPO ---------------------------------------------------- #
    if swap_on:
        from nl_probes.preference.injection_dpo import train_dpo
        st = contrib["swap_test"]
        # Env overrides let a sweep vary a knob without mutating config (config stays
        # the source of truth for defaults). Used by the C1 β / data-scale sweep.
        beta = float(os.environ.get("AO_DPO_BETA") or st["beta"])
        n_tokens = int(os.environ.get("AO_DPO_N_TOKENS") or st["n_tokens"])
        pairs_per_token = int(os.environ.get("AO_DPO_PAIRS_PER_TOKEN") or st["pairs_per_token"])
        epochs = int(os.environ.get("AO_DPO_EPOCHS") or st["epochs"])
        batch_size = int(os.environ.get("AO_DPO_BATCH") or st.get("batch_size", 4))
        print(f"[rl] corpus source: {corpus_src} (key={corpus_key})")
        print(f"[rl] DPO config: beta={beta} n_tokens={n_tokens} pairs_per_token={pairs_per_token} "
              f"epochs={epochs} batch_size={batch_size}")
        texts = _load_corpus_texts(corpus_src, corpus_key, limit=5000)
        # Cache key = everything that changes the mined dataset (beta/lr/epochs
        # deliberately excluded, so a beta sweep mines once). corpus mtime guards
        # against a regenerated corpus silently reusing a stale dataset.
        try:
            corpus_mtime = os.path.getmtime(corpus_src)
        except OSError:
            corpus_mtime = 0.0
        cache_key = {
            "model": model_name, "init_lora": str(init_lora),
            "corpus_src": str(corpus_src), "corpus_key": str(corpus_key),
            "corpus_mtime": corpus_mtime, "corpus_limit": 5000,
            "n_tokens": n_tokens, "pairs_per_token": pairs_per_token,
            "distinct_threshold": float(st["distinct_threshold"]),
            "gen_temperature": float(st["gen_temperature"]),
            "embed_model": str(rlc["embed_model"]),   # distinctness is embedder-defined
            "swap_ctx": SWAP_CTX, "seed": cfg["training"]["seed"],
        }
        pairs = _load_or_mine_pairs(
            art / "dpo_cache", cache_key, device, dtype,
            lambda: _build_dpo_pairs(
                model, tokenizer, submodule, steer, layers, device, dtype, embedder,
                texts, rng,
                n_tokens=n_tokens, pairs_per_token=pairs_per_token,
                distinct_threshold=float(st["distinct_threshold"]),
                gen_temperature=float(st["gen_temperature"]), max_ctx_tokens=SWAP_CTX))
        final = train_dpo(model=model, tokenizer=tokenizer, submodule=submodule, steering_coefficient=steer,
                          pairs=pairs, save_dir=str(save_dir), device=device, dtype=dtype,
                          beta=beta, lr=float(st["lr"]), epochs=epochs, batch_size=batch_size)
        # Persist the run's hyperparameters next to the weights. The DPO knobs are
        # not part of ao_config.json (that is the SFT recipe), so without this a
        # finished checkpoint carries no record of the beta/epochs it was trained
        # with — making after-the-fact run comparison guesswork.
        _save_dpo_run_meta(Path(final), experiment_name(), beta=beta, epochs=epochs,
                           batch_size=batch_size, lr=float(st["lr"]), n_tokens=n_tokens,
                           pairs_per_token=pairs_per_token, n_pairs=len(pairs),
                           distinct_threshold=float(st["distinct_threshold"]),
                           gen_temperature=float(st["gen_temperature"]),
                           corpus_src=str(corpus_src))
        _finalize_checkpoint(final, init_lora)
        return

    # ---- C2: warm-start + hardened GRPO ------------------------------------- #
    from nl_probes.grpo.trainer import train_grpo

    def _sync_reference_to_default() -> int:
        """Re-anchor the frozen KL reference to the CURRENT (default) policy.

        The reference adapter starts as a frozen copy of the SFT checkpoint. After a
        warm-start that deliberately moves the policy (here: seeding abstention),
        leaving the anchor at SFT makes the KL term pull training BACK toward the
        un-warm-started behavior — empirically it dragged abstention all the way to
        baseline. Re-pointing the anchor at the post-warm-start weights makes KL
        instead *protect* that calibrated policy from drift.

        Mechanism: copy every `lora_*.reference.*` tensor from its `lora_*.default.*`
        twin (the two adapters share module structure; only the adapter-name segment
        of each key differs). Returns the number of tensors synced.
        """
        sd = model.state_dict()
        updates = {n: sd[n.replace(f".{REF_ADAPTER}.", ".default.")].detach().clone()
                   for n in sd
                   if f".{REF_ADAPTER}." in n and n.replace(f".{REF_ADAPTER}.", ".default.") in sd}
        if updates:
            model.load_state_dict(updates, strict=False)
        return len(updates)
    ab = contrib["abstention"]
    print(f"[rl] convqa source: {convqa_src}")
    rows = _load_convqa_rows(convqa_src, limit=8000)
    src = AbstentionSource(
        rows, tokenizer, layers, rng, embedder,
        answerable_prob=float(ab["answerable_ratio"]), max_ctx_tokens=max_ctx,
        near_mismatch=bool(ab.get("near_mismatch", True)))

    proto_emb = embedder.embed(ABSTAIN_PROTOTYPES)

    r_wrong = float(ab.get("reward_wrong", -1.0))
    r_abst_ans = float(ab.get("reward_abstain_when_answerable", -0.5))

    def reward_fn(scenario, rollouts_per_dp):
        return abstention_reward(
            scenario, rollouts_per_dp, embedder, proto_emb,
            correct_threshold=float(ab["correct_threshold"]),
            abstain_threshold=float(ab["abstain_threshold"]),
            reward_wrong=r_wrong, reward_abstain_when_answerable=r_abst_ans)

    # TIAR (Trajectory-Informed Advantage Reweighting, arXiv:2605.25850): a free,
    # post-hoc bump to ABSTAINING rollouts' advantages computed from the group's own
    # rewards (no extra forward passes → no added train time). Within a group,
    #   p̂ = n_correct / (n_correct + n_wrong)   over the *attempted* (non-abstaining)
    # rollouts is the empirical difficulty; we add λ(1−2p̂) to every abstaining
    # rollout's advantage. Hard group (p̂<0.5) → abstention boosted; easy group
    # (p̂>0.5) → abstention suppressed. It is *decoupled* (adjusts only abstainers,
    # leaving the correct/wrong advantages from group-centering intact) to avoid
    # distorting their signal. We classify each rollout from the reward sentinels
    # plus the scenario's known answerability (abstain = −0.5 if answerable else +1;
    # correct only exists when answerable).
    tiar_lambda = float(ab.get("tiar_lambda", 0.0))

    def tiar_adjust(scenario, rewards_k, advs):
        if tiar_lambda <= 0.0:
            return advs
        answerable = scenario["answerable"]
        abstain_val = r_abst_ans if answerable else 1.0
        is_abstain = [abs(r - abstain_val) < 1e-9 for r in rewards_k]
        n_c = sum(answerable and abs(r - 1.0) < 1e-9 for r in rewards_k)
        n_w = sum(not ab_j and abs(r - r_wrong) < 1e-9 for r, ab_j in zip(rewards_k, is_abstain))
        if n_c + n_w == 0:
            return advs
        bonus = tiar_lambda * (1.0 - 2.0 * n_c / (n_c + n_w))
        return [a + bonus if ab_j else a for a, ab_j in zip(advs, is_abstain)]

    # Held-out proxy for best-checkpoint selection: one greedy rollout per fixed
    # scenario (no judge, cheap), scored with the same ternary reward. We also
    # decode the CALIBRATION breakdown straight from the reward sentinels + each
    # scenario's answerability, so the log shows abstention vs correctness instead
    # of one conflated scalar:
    #   answerable:   +1 → correct,   r_abst_ans → over-abstained,  r_wrong → wrong
    #   unanswerable: +1 → abstained (good),                        r_wrong → answered (hallucinated)
    held = src.grpo_sampler()(64)
    n_ans = sum(sc["answerable"] for sc in held)
    n_un = len(held) - n_ans

    def _held_rewards(m) -> list[float]:
        dps = [sc["datapoints"][0] for sc in held]
        gen = {"do_sample": False, "max_new_tokens": 48}
        _, groups = _gen_in_chunks(m, tokenizer, dps, submodule, steer, 1, gen, device, dtype, chunk=8)
        txt = [[tokenizer.decode(g[0], skip_special_tokens=True)] for g in groups]
        return [reward_fn(sc, [txt[i]])[0][0] for i, sc in enumerate(held)]

    def _report(tag: str, rewards: list[float]) -> float:
        mean_r = sum(rewards) / max(len(rewards), 1)
        ans_ok = sum(sc["answerable"] and abs(r - 1.0) < 1e-9 for r, sc in zip(rewards, held))
        over_ab = sum(sc["answerable"] and abs(r - r_abst_ans) < 1e-9 for r, sc in zip(rewards, held))
        un_ab = sum((not sc["answerable"]) and abs(r - 1.0) < 1e-9 for r, sc in zip(rewards, held))
        print(f"  [eval:{tag}] reward={mean_r:+.3f}  ans_correct={ans_ok}/{n_ans}  "
              f"un_abstain={un_ab}/{n_un}  over_abstain={over_ab}/{n_ans}", flush=True)
        return mean_r

    def eval_fn(m) -> float:
        return _report("held", _held_rewards(m))

    # Warm-start the abstention skill AFTER the eval machinery exists, so we can
    # measure the policy immediately before it (the "pre-warmstart" line) and thus
    # read warm-start's effect — seeding abstention mass — in isolation from GRPO's.
    # Then a few injected-SFT steps, and a hard finiteness gate before GRPO. The
    # result is cached (keyed by the inputs that determine it) so a kl_coef/lr sweep
    # reuses ONE identical warm-started policy instead of re-seeding (slow) from a
    # different random draw (confounding) each run.
    if int(ab.get("warmstart_steps", 0)) > 0:
        il = Path(init_lora)
        il_stamp = il.stat().st_mtime if il.exists() else 0
        key = _warmstart_cache_key({
            "model": model_name, "init_lora": str(init_lora), "init_lora_mtime": il_stamp,
            "convqa": convqa_src, "steps": int(ab["warmstart_steps"]),
            "ws_prob": float(ab.get("warmstart_answerable_ratio", 0.5)),
            "ws_lr": float(ab.get("warmstart_lr", 1.0e-5)), "steer": float(steer),
            "hook_layer": int(cfg["injection"]["hook_onto_layer"]), "embed": rlc["embed_model"],
            "max_ctx": int(max_ctx), "near": bool(ab.get("near_mismatch", True)),
        })
        cache_dir = art / "checkpoints" / "warmstart_cache" / key
        refresh = bool(os.environ.get("AO_WARMSTART_REFRESH"))

        if cache_dir.exists() and not refresh:
            _load_warmstart(model, cache_dir)
            print(f"[rl] loaded cached warm-start ({key}) — skipping SFT "
                  f"(AO_WARMSTART_REFRESH=1 to redo)", flush=True)
            _report("post-warmstart (cached)", _held_rewards(model))
        else:
            _report("pre-warmstart", _held_rewards(model))
            seed = src.seed_datapoints(int(ab["warmstart_steps"]) * 4, ABSTAIN_PROTOTYPES[0],
                                       answerable_prob=float(ab.get("warmstart_answerable_ratio", 0.5)))
            _warmstart_sft(model, tokenizer, submodule, steer, seed, device, dtype,
                           steps=int(ab["warmstart_steps"]), batch_size=4,
                           lr=float(ab.get("warmstart_lr", 1.0e-5)))
            # Defense-in-depth: refuse to enter GRPO with a corrupted adapter. A nan/inf
            # weight here would otherwise surface only as an opaque device-side assert
            # inside generation (multinomial on nan logits), miles from the cause.
            bad = [n for n, p in model.named_parameters() if p.requires_grad and not torch.isfinite(p).all()]
            if bad:
                raise SystemExit(f"[rl] warm-start produced non-finite weights in {len(bad)} tensors "
                                 f"(e.g. {bad[0]}) — aborting before GRPO. Check seed targets / lr.")
            _save_warmstart(model, cache_dir, meta={"key": key, "experiment": experiment_name()})
            print(f"[rl] cached warm-start adapter → {cache_dir}", flush=True)

        # Re-anchor the KL reference to the (fresh or cached) warm-started policy so
        # the KL term protects the seeded abstention rather than dragging it to SFT.
        n_sync = _sync_reference_to_default()
        print(f"[rl] re-anchored KL reference SFT→post-warm-start policy ({n_sync} tensors)", flush=True)

    final = train_grpo(
        model=model, tokenizer=tokenizer, submodule=submodule, steering_coefficient=steer,
        scenario_sampler=src.grpo_sampler(), reward_fn=reward_fn, save_dir=str(save_dir),
        device=device, dtype=dtype, k=int(ab["k_samples"]), lr=float(rlc["lr"]),
        max_steps=args.max_steps or int(rlc["max_steps"]),
        batch_scenarios=int(rlc.get("batch_scenarios", 4)),
        kl_coef=float(rlc.get("kl_coef", 0.02)),
        inner_epochs=int(rlc.get("inner_epochs", 1)),
        dynamic_sampling=bool(rlc.get("dynamic_sampling", True)),
        advantage_postprocess=tiar_adjust,
        # Higher rollout temperature ⇒ more within-group abstain/answer variance, which is
        # what GRPO needs (an over-answering policy at low temp yields all-answer groups →
        # zero variance → dynamic-sampling drops them → ~no gradient, the c2v1 failure mode).
        generation_kwargs={"do_sample": True, "temperature": float(rlc.get("rollout_temperature", 0.9)),
                           "top_p": 0.95, "max_new_tokens": 48},
        eval_fn=eval_fn, eval_every=int(rlc.get("eval_every", 200)))
    # Make the final (and best, if eval-selected) checkpoints directly eval-loadable.
    _finalize_checkpoint(final, init_lora)
    _finalize_checkpoint(str(save_dir / "best"), init_lora)


if __name__ == "__main__":
    main()
