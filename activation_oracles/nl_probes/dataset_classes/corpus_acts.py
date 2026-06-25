"""Shared corpus-activation collection for the synthetic-injection SFT tasks.

WHAT THIS IS
------------
A small DRY helper used by the contributions that build their training signal by
INJECTING residual-stream activations harvested from the target model on a corpus
(C9 activation-arithmetic, C12 odd-one-out, C13 denoising). Every one of those
tasks needs the same two primitives:

  1. sample an interior (token-ids, position) anchor from the on-policy corpus, and
  2. read the target model's residual at that anchor across a layer combo.

Factoring them here keeps each task file to just its own construction logic
(SOLID: one task, one file) and guarantees the three tasks extract activations
identically — the same way `logit_lens`/`model_diffing` do — so their injection
vectors are directly comparable.

MECHANISM
---------
`sample_anchor` pulls documents from `_single_corpus_generator`, tokenizes, and
picks a random interior position (>= min_context_tokens of left context).
`collect_anchor_acts` left-pads a batch, runs ONE forward, and returns, per
anchor, a layer-major `[num_layers, D]` tensor of the residual at that position —
exactly the row layout `create_training_datapoint` expects for a single injection
slot (it stacks one vector per layer of the combo).
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch

from nl_probes.dataset_classes.past_lens_dataset import _single_corpus_generator
from nl_probes.utils.activation_utils import collect_activations_multiple_layers, get_hf_submodule


@dataclass
class Anchor:
    """One corpus injection site: tokens up to and including `position`."""
    ids: list[int]
    position: int


def corpus_generator(tokenizer, pretrain_dataset: str, pretrain_key: str, split: str):
    """The same on-policy corpus stream past/logit-lens read from."""
    return _single_corpus_generator(
        tokenizer, pretrain_dataset=pretrain_dataset, pretrain_key=pretrain_key, split=split,
    )


def sample_anchor(gen, tokenizer, rng: random.Random, max_length: int,
                  min_context_tokens: int) -> Anchor | None:
    """Draw one document and pick a random interior anchor position, or None if
    the document is too short to leave `min_context_tokens` of left context."""
    ids = tokenizer(
        next(gen).text, add_special_tokens=False, truncation=True,
        max_length=max_length, return_tensors=None,
    )["input_ids"]
    if len(ids) < min_context_tokens + 1:
        return None
    i = rng.randint(min_context_tokens, len(ids) - 1)
    return Anchor(ids=ids[: i + 1], position=i)


@torch.no_grad()
def collect_anchor_acts(model, anchors: list[Anchor], layers: list[int],
                        device) -> list[torch.Tensor]:
    """Residual at each anchor's `position`, across `layers`, in ONE batched forward.

    Returns one `[num_layers, D]` float-CPU tensor per anchor (layer-major), the
    row layout a single-position injection datapoint uses. Left-pads so a fixed
    last index isn't assumed — each anchor keeps its own position (shifted by pad).
    """
    pad_id = model.config.pad_token_id if model.config.pad_token_id is not None else 0
    max_len = max(len(a.ids) for a in anchors)
    input_ids, attn, idxs = [], [], []
    for a in anchors:
        pad = max_len - len(a.ids)
        input_ids.append(torch.tensor([pad_id] * pad + a.ids, dtype=torch.long, device=device))
        attn.append(torch.tensor([False] * pad + [True] * len(a.ids), dtype=torch.bool, device=device))
        idxs.append(pad + a.position)
    inputs = {"input_ids": torch.stack(input_ids), "attention_mask": torch.stack(attn)}

    submodules = {l: get_hf_submodule(model, l) for l in layers}
    acts = collect_activations_multiple_layers(model, submodules, inputs, None, None)

    out: list[torch.Tensor] = []
    for b, pos_idx in enumerate(idxs):
        vecs = [acts[l][b, pos_idx, :].float().cpu() for l in layers]  # one per layer
        out.append(torch.stack(vecs, dim=0))  # [num_layers, D]
    return out


def anchor_token_text(tokenizer, anchor: Anchor) -> str:
    """The surface token sitting AT the anchor position — the verifiable label
    of 'what this activation most directly encodes' (used as the target for the
    arithmetic and denoising tasks)."""
    return tokenizer.decode([anchor.ids[anchor.position]], skip_special_tokens=False).strip()
