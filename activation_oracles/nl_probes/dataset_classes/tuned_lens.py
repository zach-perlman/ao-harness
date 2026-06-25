"""Tuned lens for the logit-lens SFT target (Contribution 3, lens="tuned").

WHY (tuned vs raw logit lens)
-----------------------------
The raw logit lens reads a layer-l residual as `unembed(final_norm(h_l))` — it
borrows the FINAL layer's decoder and applies it to an INTERMEDIATE state. That is
brittle at mid-depth: intermediate residuals live in a rotated/shifted basis
relative to the final layer (representational drift), so the projection is biased
(toward high-frequency tokens) and poorly calibrated (Belrose et al. 2023). Using
its noisy top-k as a training target teaches the AO to predict noise.

The tuned lens inserts ONE small per-layer affine "translator" T that maps h_l
into the final-layer basis BEFORE the frozen decoder:

    tuned_logits = unembed(final_norm( T(h_l) )),   T(h) = h + (A·h + b)

T is a residual affine, initialized to the identity (so at step 0 the tuned lens
== the logit lens), then fit to reproduce the model's OWN final next-token
distribution by minimizing  KL( softmax(final_logits) || softmax(tuned_logits) ).
`final_norm` + `unembed` stay frozen; only T (a single d×d Linear) learns. The fit
streams over the corpus once and is cached to disk, so it is paid only once per
(model, layer).

MECHANISM (fit_tuned_lens)
--------------------------
  repeat for max_steps (or until n_tokens consumed):
    1. one frozen forward over a corpus batch under no_grad collects, in the SAME
       pass, the layer-l residual (via a forward hook) AND the model's final
       logits at every attended position.
    2. these are graph-free constants; the translator is applied OUT of no_grad so
       the only learnable path is T -> (frozen norm+unembed) -> tuned_logits.
    3. KL(model_final || tuned) over a random subset of positions; AdamW step on T.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from nl_probes.utils.activation_utils import get_hf_submodule


class TunedLensTranslator(nn.Module):
    """Residual affine h -> h + (A·h + b), identity-initialized (== logit lens at init)."""

    def __init__(self, d_model: int):
        super().__init__()
        self.affine = nn.Linear(d_model, d_model, bias=True)
        nn.init.zeros_(self.affine.weight)  # A = 0, b = 0  ->  T(h) = h  (pure logit lens)
        nn.init.zeros_(self.affine.bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return h + self.affine(h)


def cache_path_for(tuned_lens_dir: str, model_name: str, layer: int) -> str:
    slug = model_name.replace("/", "_")
    return str(Path(tuned_lens_dir) / f"tuned_lens_{slug}_L{layer}.pt")


@torch.no_grad()
def _collect_resid_and_logits(model, submodule, inputs):
    """One frozen forward: capture the layer's residual via a hook AND return final logits.

    Unlike `collect_activations`, we do NOT early-stop — the full forward is needed
    to produce the unembedding logits that are the tuned lens's fit target.
    """
    store: dict[str, torch.Tensor] = {}

    def hook(_module, _inp, out):
        store["resid"] = out[0] if isinstance(out, tuple) else out

    handle = submodule.register_forward_hook(hook)
    try:
        logits = model(**inputs, use_cache=False).logits
    finally:
        handle.remove()
    return store["resid"], logits


def fit_tuned_lens(
    *,
    model,
    norm: nn.Module,
    unembed: nn.Module,
    layer: int,
    device,
    d_model: int,
    corpus_generator,
    tokenizer,
    cache_path: str | None,
    n_tokens: int = 500_000,
    max_steps: int = 400,
    lr: float = 1e-3,
    batch_size: int = 4,
    seq_len: int = 512,
    max_pos_per_step: int = 256,
    force: bool = False,
) -> TunedLensTranslator:
    """Fit (or load from cache) a tuned-lens translator for one layer.

    Assumes the model's parameters are already frozen (requires_grad=False) so the
    backward through the frozen norm/unembed allocates no parameter grads. Returns
    an eval-mode translator on `device` in float32.
    """
    translator = TunedLensTranslator(d_model).to(device=device, dtype=torch.float32)

    if cache_path and os.path.exists(cache_path) and not force:
        translator.load_state_dict(torch.load(cache_path, map_location=device))
        translator.eval()
        print(f"[tuned_lens] loaded cached translator for layer {layer}: {cache_path}")
        return translator

    submodule = get_hf_submodule(model, layer)
    model_dtype = next(unembed.parameters()).dtype
    # pad id: prefer the tokenizer (always set), then the config (top-level for
    # plain decoders, nested text_config for multimodal wrappers like Gemma4), else 0.
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = getattr(model.config, "pad_token_id", None) \
            or getattr(getattr(model.config, "text_config", None), "pad_token_id", None) or 0
    opt = torch.optim.AdamW(translator.parameters(), lr=lr)
    translator.train()

    seen_tokens, step = 0, 0
    pbar = tqdm(total=max_steps, desc=f"tuned_lens fit L{layer}")
    while step < max_steps and seen_tokens < n_tokens:
        # ---- assemble a left-padded batch from the corpus ----
        batch: list[list[int]] = []
        while len(batch) < batch_size:
            ids = tokenizer(
                next(corpus_generator).text, add_special_tokens=False,
                truncation=True, max_length=seq_len,
            )["input_ids"]
            if len(ids) >= 16:
                batch.append(ids)
        max_len = max(len(t) for t in batch)
        input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long, device=device)
        attn = torch.zeros((len(batch), max_len), dtype=torch.long, device=device)
        for i, t in enumerate(batch):
            input_ids[i, -len(t):] = torch.tensor(t, device=device)
            attn[i, -len(t):] = 1

        resid, logits = _collect_resid_and_logits(
            model, submodule, {"input_ids": input_ids, "attention_mask": attn})

        # ---- KL(model_final || tuned) over a random subset of attended positions ----
        mask = attn.bool()
        resid_flat = resid[mask]            # [N, D]  (graph-free constant)
        target_logp = F.log_softmax(logits[mask].float(), dim=-1).detach()
        n = resid_flat.shape[0]
        if n > max_pos_per_step:
            idx = torch.randperm(n, device=device)[:max_pos_per_step]
            resid_flat, target_logp = resid_flat[idx], target_logp[idx]

        tuned = translator(resid_flat.float())                      # learnable path
        tuned_logits = unembed(norm(tuned.to(model_dtype)))         # frozen decoder
        tuned_logp = F.log_softmax(tuned_logits.float(), dim=-1)
        loss = F.kl_div(tuned_logp, target_logp, log_target=True, reduction="batchmean")

        opt.zero_grad()
        loss.backward()
        opt.step()

        seen_tokens += int(mask.sum().item())
        step += 1
        pbar.update(1)
        pbar.set_postfix(kl=f"{loss.item():.3f}", tok=f"{seen_tokens/1e3:.0f}k")
    pbar.close()

    translator.eval()
    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(translator.state_dict(), cache_path)
        print(f"[tuned_lens] saved translator for layer {layer}: {cache_path}")
    return translator
