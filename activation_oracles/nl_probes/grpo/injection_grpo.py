"""Tokenwise GRPO with nl_probes activation injection.

WHAT
----
The policy is the AO LoRA; each "prompt" is an injected activation + a natural-
language query, and a rollout is the AO's free-form answer. GRPO needs three
operations, ALL with the activation injected through the same steering hook so the
sampled distribution and the scored distribution agree:

  1. generate_grouped_rollouts — sample k answers per prompt (injection on prefill).
  2. compute_old_logprobs       — score those answers under the sampling snapshot.
  3. compute_grpo_loss          — re-score under the live policy, clipped surrogate.

MECHANISM
---------
A materialized prompt datapoint carries: prompt token ids (the user turn, ending
at the assistant generation prompt, with SPECIAL_TOKEN placeholders), the placeholder
`positions`, and the precomputed `steering_vectors` injected at those positions
(norm-matched additive, via get_hf_activation_steering_hook). For generation we
left-pad a batch of k copies; for log-probs we run each (prompt+response) singly
(B=1, unpadded) so the placeholder positions index directly into the sequence. The
hook only fires on multi-token forwards (L>1), i.e. the prefill and the full
re-scoring pass — never the single-token decode steps — exactly as at eval time.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from nl_probes.utils.dataset_utils import (
    TrainingDataPoint,
    construct_batch,
    get_prompt_tokens_only,
    materialize_missing_steering_vectors,
)

REF_ADAPTER = "reference"  # frozen copy of the SFT LoRA, loaded by the orchestrator


@contextlib.contextmanager
def reference_adapter(model):
    """Activate the frozen SFT reference for the duration of the block.

    The AO *is* the LoRA, so disabling adapters would give the BASE model — the
    wrong KL/DPO anchor. When a frozen `reference` adapter (a copy of the SFT
    weights) is loaded we switch to it; otherwise we fall back to disabling the
    adapter (correct only when the policy was trained on top of a merged SFT base).
    """
    cfg = getattr(model, "peft_config", {})
    if REF_ADAPTER in cfg:
        prev = model.active_adapter
        model.set_adapter(REF_ADAPTER)
        try:
            yield
        finally:
            model.set_adapter(prev)
    else:
        with model.disable_adapter():
            yield
from nl_probes.utils.steering_hooks import add_hook, get_hf_activation_steering_hook


@dataclass
class GRPOItem:
    """One (prompt, sampled-response) pair plus its injection and learning signal."""
    prompt_ids: list[int]
    response_ids: list[int]
    steering_vectors: torch.Tensor  # [P, D] — P = len(positions) = placeholders × layers
    positions: list[int]            # placeholder positions within prompt_ids
    advantage: float
    old_token_logprobs: list[float] = field(default_factory=list)  # sampling snapshot (only if μ>1)
    ref_token_logprobs: list[float] = field(default_factory=list)  # frozen SFT reference (for KL anchor)


def materialize_prompts(
    datapoints: list[TrainingDataPoint], tokenizer, model, device,
) -> list[TrainingDataPoint]:
    """Strip the (unused) target answer and fill steering_vectors from context.

    After this each datapoint's input_ids are the prompt only, with `positions`
    and `steering_vectors` ready to inject — the shared starting point for both
    generation and scoring.
    """
    dps = [get_prompt_tokens_only(dp) for dp in datapoints]
    return materialize_missing_steering_vectors(dps, tokenizer, model)


@torch.no_grad()
def generate_grouped_rollouts(
    model, tokenizer, prompt_dps: list[TrainingDataPoint], submodule,
    steering_coefficient: float, k: int, generation_kwargs: dict, device, dtype,
) -> list[list[list[int]]]:
    """Sample k rollouts per prompt with injection. Returns [n_prompts][k] response-id lists."""
    flat = [dp for dp in prompt_dps for _ in range(k)]  # each prompt repeated k times
    batch = construct_batch(flat, tokenizer, device)
    hook = get_hf_activation_steering_hook(
        vectors=batch.steering_vectors, positions=batch.positions,
        steering_coefficient=steering_coefficient, device=device, dtype=dtype)

    model.eval()
    with add_hook(submodule, hook):
        out = model.generate(input_ids=batch.input_ids, attention_mask=batch.attention_mask,
                             **generation_kwargs)
    gen = out[:, batch.input_ids.shape[1]:]  # newly generated columns (same start for all rows)

    pad_id, eos_id = tokenizer.pad_token_id, tokenizer.eos_token_id
    groups: list[list[list[int]]] = []
    for i in range(len(prompt_dps)):
        group = []
        for j in range(k):
            ids = gen[i * k + j].tolist()
            if eos_id in ids:                      # truncate at end-of-turn
                ids = ids[: ids.index(eos_id)]
            ids = [t for t in ids if t != pad_id]  # drop right padding
            group.append(ids)
        groups.append(group)
    return groups


def _chunks(items: list, n: int):
    """Yield consecutive slices of `items` of size ≤ n (n bounds per-forward memory)."""
    for i in range(0, len(items), n):
        yield items[i : i + n]


def _batched_response_logprobs(
    model, submodule, items: list[GRPOItem], steering_coefficient, device, dtype, pad_id: int,
) -> list[torch.Tensor]:
    """Per-response-token log-probs for a chunk of items in ONE padded forward.

    WHY: the old path scored each (prompt+response) singly (B=1), so k×scenarios
    tiny forwards ran per step — the H200 spent most of its time launching kernels,
    not computing. We instead **right-pad** the chunk into one batch:

      • every item's prompt still starts at index 0, so the steering hook's
        `positions` (indices into the prompt) stay valid *unchanged*;
      • the trailing pad is masked (attention_mask=0) and, being causal + at the
        end, is never attended to by real tokens.

    So the logits over the real tokens are identical to the unpadded B=1 pass, but
    we pay a single forward for the whole chunk. Returns one [resp_len] log-prob
    tensor per item (grad follows the ambient autograd mode).
    """
    seqs = [it.prompt_ids + it.response_ids for it in items]
    lens = [len(s) for s in seqs]
    Lmax = max(lens)
    B = len(items)
    input_ids = torch.full((B, Lmax), pad_id, dtype=torch.long, device=device)
    attn = torch.zeros((B, Lmax), dtype=torch.long, device=device)
    for i, s in enumerate(seqs):
        input_ids[i, : lens[i]] = torch.tensor(s, device=device)
        attn[i, : lens[i]] = 1

    hook = get_hf_activation_steering_hook(
        vectors=[it.steering_vectors.to(device) for it in items],
        positions=[it.positions for it in items],
        steering_coefficient=steering_coefficient, device=device, dtype=dtype)
    with add_hook(submodule, hook):
        logits = model(input_ids=input_ids, attention_mask=attn).logits  # [B, Lmax, V] (model dtype)

    out: list[torch.Tensor] = []
    for i, it in enumerate(items):
        start, end = len(it.prompt_ids), lens[i]
        if start >= end:
            out.append(logits.new_zeros((0,), dtype=torch.float32))
            continue
        pred = logits[i, start - 1 : end - 1].float()        # predict token t from position t-1
        tgt = input_ids[i, start:end]
        out.append(F.log_softmax(pred, dim=-1).gather(1, tgt.unsqueeze(1)).squeeze(1))
    return out


def _response_logprobs(model, submodule, item: GRPOItem, steering_coefficient, device, dtype):
    """Single-item (B=1) per-response-token log-probs — a convenience wrapper around
    the batched path. Retained because the DPO trainer imports it; pad_id is
    irrelevant at B=1 (no padding)."""
    return _batched_response_logprobs(
        model, submodule, [item], steering_coefficient, device, dtype, pad_id=0)[0]


@torch.no_grad()
def compute_old_logprobs(model, items: list[GRPOItem], submodule, steering_coefficient,
                        device, dtype, pad_id: int, chunk: int = 16) -> None:
    """Cache log-probs under the sampling snapshot (fills item.old_token_logprobs in place).

    Only needed when inner_epochs > 1: with a single update per rollout batch the
    snapshot equals the live policy, so the importance ratio is identically 1 and
    this pass would be wasted compute (the caller skips it in that case)."""
    model.eval()
    for ck in _chunks(items, chunk):
        for it, lp in zip(ck, _batched_response_logprobs(
                model, submodule, ck, steering_coefficient, device, dtype, pad_id)):
            it.old_token_logprobs = lp.tolist()


@torch.no_grad()
def compute_ref_logprobs(model, items: list[GRPOItem], submodule, steering_coefficient,
                        device, dtype, pad_id: int, chunk: int = 16) -> None:
    """Cache frozen-reference log-probs for the KL anchor (adapter OFF = the SFT policy).

    The injected activations are identical; only the active adapter changes, so this
    is exactly π_ref(response | injected prompt) under the supervised checkpoint."""
    model.eval()
    with reference_adapter(model):
        for ck in _chunks(items, chunk):
            for it, lp in zip(ck, _batched_response_logprobs(
                    model, submodule, ck, steering_coefficient, device, dtype, pad_id)):
                it.ref_token_logprobs = lp.tolist()


def compute_grpo_loss(
    model, items: list[GRPOItem], submodule, steering_coefficient, device, dtype, pad_id: int,
    clip_eps: float = 0.2, grad_scale: float = 1.0, kl_coef: float = 0.0,
    length_norm: float = 1.0, chunk: int = 8,
) -> tuple[float, dict[str, float]]:
    """Tokenwise GRPO surrogate (+ optional KL-to-reference) with chunked backward.

    • ratio: exp(new − old) when an old snapshot exists (μ>1); otherwise 1 (the
      first/only update), so clipping is a no-op there by construction.
    • KL: Schulman k3 estimator per response token against the frozen reference,
      kept as an elastic tether to the SFT policy (anti reward-hacking/forgetting).
    • Normalization: divide by a CONSTANT (n_items × length_norm), not by each
      response's length — avoids GRPO's length/verbosity bias (Dr. GRPO).
    • Batching: items are scored `chunk` at a time in one padded forward, and the
      chunk's summed (already /denom) loss is backpropagated once. Because `denom`
      uses the GLOBAL item count, summing chunk gradients == one big backward, so
      this is numerically the same update as the old per-item loop, just far fewer
      kernel launches. `chunk` bounds the activation memory of each backward.
    """
    n = len(items)
    denom = max(n, 1) * max(length_norm, 1.0)
    total_loss, ratios, advs, resp_lens, kls = 0.0, [], [], [], []
    for ck in _chunks(items, chunk):
        new_lps = _batched_response_logprobs(model, submodule, ck, steering_coefficient, device, dtype, pad_id)
        chunk_loss = None
        for item, new_lp in zip(ck, new_lps):
            if new_lp.numel() == 0:
                continue
            adv = torch.tensor(item.advantage, device=device, dtype=torch.float32)
            if item.old_token_logprobs:
                old_lp = torch.tensor(item.old_token_logprobs, device=device, dtype=torch.float32)
                ratio = torch.exp(new_lp - old_lp)
            else:
                ratio = torch.ones_like(new_lp)
            surrogate = torch.minimum(ratio * adv, torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv)
            item_loss = -surrogate.sum()

            if kl_coef > 0.0 and item.ref_token_logprobs:
                ref_lp = torch.tensor(item.ref_token_logprobs, device=device, dtype=torch.float32)
                log_r = ref_lp - new_lp
                kl = torch.exp(log_r) - log_r - 1.0          # k3 estimator, ≥ 0
                item_loss = item_loss + kl_coef * kl.sum()
                kls.append(float(kl.mean()))

            item_loss = item_loss / denom * grad_scale
            chunk_loss = item_loss if chunk_loss is None else chunk_loss + item_loss
            total_loss += float(item_loss)
            ratios.extend(ratio.detach().cpu().tolist())
            advs.append(item.advantage)
            resp_lens.append(len(item.response_ids))
        if chunk_loss is not None:
            chunk_loss.backward()

    metrics = {
        "grpo/loss": total_loss,
        "grpo/mean_ratio": sum(ratios) / len(ratios) if ratios else 1.0,
        "grpo/clip_frac": sum(abs(r - 1) > clip_eps for r in ratios) / len(ratios) if ratios else 0.0,
        "grpo/mean_advantage": sum(advs) / len(advs) if advs else 0.0,
        "grpo/mean_kl": sum(kls) / len(kls) if kls else 0.0,
        "grpo/mean_response_tokens": sum(resp_lens) / len(resp_lens) if resp_lens else 0.0,
        "grpo/n_sequences": float(len(resp_lens)),
    }
    return total_loss, metrics


def group_advantages(rewards: list[float], normalize: bool = True) -> list[float]:
    """Center (and optionally scale) rewards within a sampled group → GRPO advantages."""
    if not rewards:
        return []
    mean = sum(rewards) / len(rewards)
    centered = [r - mean for r in rewards]
    if normalize:
        var = sum(c * c for c in centered) / len(centered)
        std = var ** 0.5
        if std > 1e-6:
            centered = [c / std for c in centered]
    return centered
