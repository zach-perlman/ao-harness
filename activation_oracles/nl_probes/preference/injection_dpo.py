"""Offline DPO with nl_probes activation injection (Contribution 1, anti-inversion).

WHY DPO (not online GRPO) FOR C1
--------------------------------
Anti-text-inversion is a *contrastive* objective: the AO must answer differently
for the SAME token in two different upstream contexts. An online reward that only
pays for "be far from the other context" is trivially hacked by confabulating
divergent text (exactly the failure C3 exhibited). DPO sidesteps this: each
preference pair contrasts two REAL, fluent AO answers — the one a context's
activation actually elicits (chosen) vs. the answer a DIFFERENT context elicits
(rejected) — so the only thing being optimized is context-appropriateness, with a
frozen-reference KL built into the loss. No reward to game, far more stable.

MECHANISM
---------
For a prompt = (injected activation of context A + neutral query):
    chosen   = the AO's answer to A's activation
    rejected = the AO's answer to B's activation (wrong for A)
The reference policy is the SFT checkpoint, obtained for free by DISABLING the
LoRA adapter (same injection, no adapter) — no second model in memory. Reference
log-probs are frozen, so we precompute them once. The DPO loss then raises the
policy's margin logπ(chosen) − logπ(rejected) relative to that reference.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LinearLR

from nl_probes.grpo.injection_grpo import GRPOItem, _response_logprobs, reference_adapter
from nl_probes.utils.steering_hooks import add_hook, get_hf_activation_steering_hook


@dataclass
class DPOPair:
    """One preference pair sharing a single injected prompt."""
    prompt_ids: list[int]
    positions: list[int]
    steering_vectors: torch.Tensor   # [P, D]
    chosen_ids: list[int]
    rejected_ids: list[int]
    ref_chosen_lp: float = 0.0       # filled by precompute_reference (frozen)
    ref_rejected_lp: float = 0.0


def _seq_logprob(model, submodule, prompt_ids, response_ids, steering_vectors, positions,
                 steering_coefficient, device, dtype) -> torch.Tensor:
    """Summed log-prob of `response_ids` given the injected prompt (grad follows mode)."""
    item = GRPOItem(prompt_ids=prompt_ids, response_ids=response_ids,
                    steering_vectors=steering_vectors, positions=positions, advantage=0.0)
    return _response_logprobs(model, submodule, item, steering_coefficient, device, dtype).sum()


def _batched_seq_logprobs(model, submodule, seqs, steering_coefficient, device, dtype):
    """Summed response log-prob for MANY (prompt+response) sequences in ONE forward.

    This is the micro-batched replacement for calling `_seq_logprob` once per
    sequence. The original trainer processed pairs one-at-a-time (peak VRAM = a
    single short prompt), leaving most of the GPU idle; here we stack a whole
    optimizer step's sequences into a single forward to actually use that headroom.

    Why right-padding is exact here:
      * placeholder `positions` are absolute indices from token 0 of the prompt, so
        right-padding (appending pad at the END) leaves them — and the response
        slice indices [start:end] — valid with NO offset.
      * a causal LM's logits at real positions depend only on tokens ≤ that
        position; the pad tail sits after every real token and is also masked out
        by `attention_mask`, so each row's response log-probs equal the unpadded
        single-sequence result (up to fp matmul-batching noise).
    Returns a [N] tensor of per-sequence summed log-probs (grad follows mode)."""
    B = len(seqs)
    fulls = [s["prompt_ids"] + s["response_ids"] for s in seqs]
    lens = [len(f) for f in fulls]
    Tmax = max(lens)
    inp = torch.zeros((B, Tmax), dtype=torch.long, device=device)   # pad id 0: never read, always masked
    attn = torch.zeros((B, Tmax), dtype=torch.long, device=device)
    for i, f in enumerate(fulls):
        inp[i, : lens[i]] = torch.tensor(f, device=device)
        attn[i, : lens[i]] = 1
    hook = get_hf_activation_steering_hook(
        vectors=[s["steering_vectors"].to(device) for s in seqs],
        positions=[s["positions"] for s in seqs],
        steering_coefficient=steering_coefficient, device=device, dtype=dtype)
    with add_hook(submodule, hook):
        logits = model(input_ids=inp, attention_mask=attn).logits.float()   # [B, Tmax, V]
    out = []
    for i, s in enumerate(seqs):
        start, end = len(s["prompt_ids"]), lens[i]
        if start >= end:
            out.append(logits.new_zeros(()))
            continue
        pred = logits[i, start - 1 : end - 1]                       # predict token t from t-1
        tgt = inp[i, start:end]
        out.append(F.log_softmax(pred, dim=-1).gather(1, tgt.unsqueeze(1)).squeeze(1).sum())
    return torch.stack(out)


def _pair_seqs(p: "DPOPair") -> list[dict]:
    """The two sequences of a pair, chosen first then rejected (shared prompt/injection)."""
    base = {"prompt_ids": p.prompt_ids, "steering_vectors": p.steering_vectors, "positions": p.positions}
    return [{**base, "response_ids": p.chosen_ids}, {**base, "response_ids": p.rejected_ids}]


@torch.no_grad()
def precompute_reference(model, submodule, pairs: list[DPOPair], steering_coefficient, device, dtype,
                         chunk: int = 48) -> None:
    """Cache frozen-reference (the SFT adapter) seq log-probs for both responses.

    Batched in chunks of `chunk` sequences per forward. This pass is no-grad (no
    retained activations), so the chunk is far larger than the training micro-batch
    — a wider forward shortens the upfront reference pass without touching the
    trained model. 48 short sequences fit comfortably in a few GB of logits."""
    model.eval()
    seqs = [s for p in pairs for s in _pair_seqs(p)]               # [chosen, rejected, chosen, rejected, ...]
    lps: list[float] = []
    with reference_adapter(model):
        for i in range(0, len(seqs), chunk):
            lps.extend(_batched_seq_logprobs(
                model, submodule, seqs[i : i + chunk], steering_coefficient, device, dtype).tolist())
    for j, p in enumerate(pairs):
        p.ref_chosen_lp, p.ref_rejected_lp = lps[2 * j], lps[2 * j + 1]


def train_dpo(
    *,
    model,
    tokenizer,
    submodule,
    steering_coefficient: float,
    pairs: list[DPOPair],
    save_dir: str,
    device,
    dtype=torch.bfloat16,
    beta: float = 0.1,
    lr: float = 5e-6,
    epochs: int = 1,
    batch_size: int = 4,
    warmup_steps: int = 20,
    max_grad_norm: float = 1.0,
    log_every: int = 10,
) -> str:
    """Standard DPO over injected preference pairs (reference = adapter-disabled SFT)."""
    from pathlib import Path

    precompute_reference(model, submodule, pairs, steering_coefficient, device, dtype)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.0)
    total_steps = max(1, (len(pairs) // batch_size) * epochs)
    scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=max(warmup_steps, 1))

    print(f"[dpo] {len(pairs)} pairs · {epochs} epoch(s) · beta={beta} · lr={lr} · bs={batch_size}")
    t0, step = time.time(), 0
    for epoch in range(epochs):
        order = torch.randperm(len(pairs)).tolist()
        for b in range(0, len(order), batch_size):
            batch = [pairs[i] for i in order[b : b + batch_size]]
            model.train()
            optimizer.zero_grad()
            # One forward over the step's 2·|batch| sequences (chosen,rejected interleaved
            # per pair), then a single backward on the mean DPO loss — mathematically the
            # same update as per-pair accumulation, but it uses the GPU's spare headroom.
            seqs = [s for p in batch for s in _pair_seqs(p)]
            lp = _batched_seq_logprobs(model, submodule, seqs, steering_coefficient, device, dtype)
            pol_ch, pol_rej = lp[0::2], lp[1::2]
            ref_ch = lp.new_tensor([p.ref_chosen_lp for p in batch])
            ref_rej = lp.new_tensor([p.ref_rejected_lp for p in batch])
            margin = (pol_ch - ref_ch) - (pol_rej - ref_rej)
            loss = (-F.logsigmoid(beta * margin)).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm)
            optimizer.step()
            scheduler.step()
            step += 1
            if step % log_every == 0 or step == 1:
                n_correct = int((pol_ch > pol_rej).sum())
                print(f"  step {step}/{total_steps}  loss={float(loss):.4f}  "
                      f"pref_acc={n_correct / len(batch):.2f}  {(time.time() - t0) / step:.1f}s/step",
                      flush=True)

    final = str(Path(save_dir) / "final")
    model.save_pretrained(final)
    tokenizer.save_pretrained(final)
    print(f"[dpo] done in {time.time() - t0:.0f}s → {final}")
    return final
