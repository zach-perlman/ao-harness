"""Logit-lens prediction dataset (Contribution 3).

WHAT THIS TASK IS
-----------------
A self-supervised SFT task whose target is *the target model's own next-token
prediction*, read directly off an intermediate activation via the **logit lens**
(nostalgebraist 2020): project the residual stream at layer l through the model's
final norm + unembedding to get a vocabulary distribution, and take its top-k
tokens. The AO is then trained, from the injected layer-l activation alone, to
name those top-k tokens.

WHY IT HELPS (vs past/future lens)
----------------------------------
Past/future lens predicts neighbouring *text*, which an AO can game by
reconstructing surrounding tokens (text inversion). The logit-lens target is
computed *from the activation itself* (a single matmul + softmax over the
unembedding), so there is no text shortcut: an AO that predicts it must actually
read the activation's content. The task is fully self-supervised and scales to
any corpus.

MECHANISM (create_dataset)
--------------------------
  for each corpus document:
    1. tokenize, pick an interior position i and a layer combo (the recipe's
       5 contiguous layers ~65% depth).
    2. one batched forward over the target model collects the residual at the
       DEEPEST layer of the combo (closest to the unembedding -> most meaningful
       logit lens) at position i.
    3. top-k = topk(lm_head(final_norm(resid_i))). Target text =
         free_form : "Paris, city, capital, the, France"  (comma-joined top-k)
         binary    : a Yes/No probe — "is '<tok>' in the top-k here?" (balanced)
    4. store a TrainingDataPoint with context_input_ids = tokens[:i+1],
       context_positions = [i], layers = the full combo, steering_vectors=None.
       At train time the injection hook collects the layer-l activations for the
       whole combo at position i and feeds them to the AO (identical injection
       structure to every other task), and the AO is supervised on the target.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass

import torch
from tqdm.auto import tqdm

from nl_probes.dataset_classes.act_dataset_manager import (
    ActDatasetLoader,
    BaseDatasetConfig,
    DatasetLoaderConfig,
)
from nl_probes.dataset_classes.past_lens_dataset import _single_corpus_generator
from nl_probes.utils.activation_utils import collect_activations_multiple_layers, get_hf_submodule
from nl_probes.utils.common import layer_percent_to_layer, load_model, load_tokenizer
from nl_probes.utils.dataset_utils import TrainingDataPoint, create_training_datapoint


@dataclass
class LogitLensDatasetConfig(BaseDatasetConfig):
    pretrain_dataset: str = "HuggingFaceFW/fineweb"  # HF repo id or local .jsonl path
    pretrain_key: str = "text"
    pretrain_split: str = "train"
    top_k: int = 5
    task_format: str = "free_form"  # "free_form" (list top-k) | "binary" (yes/no probe)
    max_length: int = 2000
    min_context_tokens: int = 8     # need a few tokens of context before position i
    # lens = how the top-k target is decoded from the layer-l residual:
    #   "logit" : raw logit lens (unembed(final_norm(resid))) — zero-shot, brittle mid-depth.
    #   "tuned" : tuned lens — a per-layer affine translator (fit once, cached) maps the
    #             residual into the final-layer basis first -> cleaner, calibrated targets.
    lens: str = "logit"
    tuned_lens_dir: str = ""        # where translators are cached (required when lens="tuned")
    tuned_lens_tokens: int = 500_000
    tuned_lens_steps: int = 400
    tuned_lens_lr: float = 1e-3


def _final_norm_and_unembed(model) -> tuple[torch.nn.Module, torch.nn.Module]:
    """Locate the target model's final norm + unembedding (lm_head).

    Handles the standard decoder layout (Qwen/Llama/Mistral/Gemma:
    `model.model.norm` + `model.lm_head`) and the Qwen3.5 multimodal wrapper
    (norm under `language_model`). Fails loudly on an unknown structure.
    """
    lm_head = getattr(model, "lm_head", None)
    inner = getattr(model, "model", model)
    norm = getattr(inner, "norm", None)
    if norm is None:  # Qwen3.5-style nested language_model
        for chain in (("language_model",), ("model", "language_model")):
            obj = inner
            for attr in chain:
                obj = getattr(obj, attr, None)
                if obj is None:
                    break
            if obj is not None and getattr(obj, "norm", None) is not None:
                norm = obj.norm
                break
    if norm is None or lm_head is None:
        raise ValueError("Could not locate final norm / lm_head for logit lens on this model")
    return norm, lm_head


class LogitLensDatasetLoader(ActDatasetLoader):
    DATASET_NAME = "logit_lens"

    def __init__(self, dataset_config: DatasetLoaderConfig):
        super().__init__(dataset_config)
        assert self.dataset_config.dataset_name == "", "Dataset name gets overridden here"
        self.dataset_config.dataset_name = self.DATASET_NAME
        self.dataset_params: LogitLensDatasetConfig = dataset_config.custom_dataset_params
        assert self.dataset_config.save_acts is False, "logit_lens uses on-the-fly extraction"
        assert self.dataset_config.splits == ["train"], "logit_lens only supports the train split"
        assert self.dataset_params.task_format in ("free_form", "binary")
        assert self.dataset_params.lens in ("logit", "tuned")
        if self.dataset_params.lens == "tuned":
            assert self.dataset_params.tuned_lens_dir, "lens='tuned' requires tuned_lens_dir"

    def create_dataset(self) -> None:
        os.makedirs(self.dataset_config.dataset_folder, exist_ok=True)
        p = self.dataset_params
        rng = random.Random(self.dataset_config.seed)
        tokenizer = load_tokenizer(self.dataset_config.model_name)

        act_layer_combinations = [
            [layer_percent_to_layer(self.dataset_config.model_name, pct) for pct in combo]
            for combo in self.dataset_config.layer_combinations
        ]

        model = load_model(self.dataset_config.model_name, torch.bfloat16)
        model.eval()
        model.requires_grad_(False)  # we never train the target; lets the tuned-lens fit skip param grads
        norm, unembed = _final_norm_and_unembed(model)
        device = next(model.parameters()).device

        # When lens="tuned", fit (or load) one affine translator per unique deepest
        # layer we read, so the top-k targets are decoded in the final-layer basis.
        translators: dict[int, object] | None = None
        if p.lens == "tuned":
            from nl_probes.dataset_classes.tuned_lens import cache_path_for, fit_tuned_lens
            # hidden_size lives on the top-level config for plain decoders, but on
            # the nested text_config for multimodal wrappers (e.g. Gemma4Config);
            # fall back to the unembed's input dim, which is always d_model.
            d_model = getattr(model.config, "hidden_size", None) \
                or getattr(getattr(model.config, "text_config", None), "hidden_size", None) \
                or unembed.in_features
            deepest = sorted({max(combo) for combo in act_layer_combinations})
            translators = {}
            for layer in deepest:
                fit_gen = _single_corpus_generator(
                    tokenizer, pretrain_dataset=p.pretrain_dataset,
                    pretrain_key=p.pretrain_key, split=p.pretrain_split,
                )
                translators[layer] = fit_tuned_lens(
                    model=model, norm=norm, unembed=unembed, layer=layer, device=device,
                    d_model=d_model, corpus_generator=fit_gen, tokenizer=tokenizer,
                    cache_path=cache_path_for(p.tuned_lens_dir, self.dataset_config.model_name, layer),
                    n_tokens=p.tuned_lens_tokens, max_steps=p.tuned_lens_steps, lr=p.tuned_lens_lr,
                )

        gen = _single_corpus_generator(
            tokenizer, pretrain_dataset=p.pretrain_dataset,
            pretrain_key=p.pretrain_key, split=p.pretrain_split,
        )

        target_n = self.dataset_config.num_train
        batch_n = max(8, self.dataset_config.batch_size)
        out: list[TrainingDataPoint] = []
        pbar = tqdm(total=target_n, desc="logit_lens")

        try:
            while len(out) < target_n:
                # Accumulate a batch of (token_ids, position, combo) candidates.
                cands: list[tuple[list[int], int, list[int]]] = []
                while len(cands) < batch_n and len(out) + len(cands) < target_n:
                    sample = next(gen)
                    ids = tokenizer(
                        sample.text, add_special_tokens=False, truncation=True,
                        max_length=p.max_length, return_tensors=None,
                    )["input_ids"]
                    if len(ids) < p.min_context_tokens + 1:
                        continue
                    i = rng.randint(p.min_context_tokens, len(ids) - 1)
                    combo = rng.choice(act_layer_combinations)
                    cands.append((ids[: i + 1], i, combo))
                if not cands:
                    continue

                topk_ids = self._batched_logit_lens_topk(
                    model, norm, unembed, device,
                    contexts=[c[0] for c in cands],
                    deepest_layers=[max(c[2]) for c in cands],
                    top_k=p.top_k,
                    translators=translators,
                )

                for (ctx_ids, _i, combo), tok_ids in zip(cands, topk_ids):
                    dp = self._build_datapoint(tokenizer, rng, ctx_ids, combo, tok_ids)
                    if dp is not None:
                        out.append(dp)
                        pbar.update(1)
                        if len(out) >= target_n:
                            break
        finally:
            pbar.close()
            del model
            import gc
            gc.collect()
            torch.cuda.empty_cache()

        self.save_dataset(out, "train")

    @torch.no_grad()
    def _batched_logit_lens_topk(
        self, model, norm, unembed, device,
        contexts: list[list[int]], deepest_layers: list[int], top_k: int,
        translators: dict[int, object] | None = None,
    ) -> list[list[int]]:
        """Top-k lens token ids at each context's last position.

        Left-pads the batch, collects the residual at every requested deepest
        layer in one forward, then for each item decodes its layer-l residual at
        the last token through the lens and returns the top-k vocab ids. With a
        tuned lens, the residual is first mapped by that layer's translator into
        the final-layer basis; otherwise it is the raw logit lens.
        """
        # pad id from config (top-level, then nested text_config for Gemma4-style
        # multimodal wrappers); the exact value is harmless since padded positions
        # are masked out — we only read the last real token's residual.
        pad_id = getattr(model.config, "pad_token_id", None) \
            or getattr(getattr(model.config, "text_config", None), "pad_token_id", None) or 0
        max_len = max(len(c) for c in contexts)
        input_ids, attn, offsets = [], [], []
        for c in contexts:
            pad = max_len - len(c)
            input_ids.append(torch.tensor([pad_id] * pad + c, dtype=torch.long, device=device))
            attn.append(torch.tensor([False] * pad + [True] * len(c), dtype=torch.bool, device=device))
            offsets.append(pad)
        inputs = {"input_ids": torch.stack(input_ids), "attention_mask": torch.stack(attn)}

        layers_needed = sorted(set(deepest_layers))
        submodules = {l: get_hf_submodule(model, l) for l in layers_needed}
        acts_by_layer = collect_activations_multiple_layers(
            model=model, submodules=submodules, inputs_BL=inputs, min_offset=None, max_offset=None,
        )

        results: list[list[int]] = []
        for b, (c, layer) in enumerate(zip(contexts, deepest_layers)):
            last_idx = offsets[b] + len(c) - 1
            resid = acts_by_layer[layer][b, last_idx, :]          # [D]
            if translators is not None:
                t = translators[layer]
                resid = t(resid.float().unsqueeze(0)).to(resid.dtype).squeeze(0)  # -> final-layer basis
            logits = unembed(norm(resid.unsqueeze(0))).squeeze(0)  # [V]
            results.append(torch.topk(logits, k=top_k).indices.tolist())
        return results

    def _build_datapoint(self, tokenizer, rng, ctx_ids, combo, topk_ids) -> TrainingDataPoint | None:
        position = len(ctx_ids) - 1  # the position whose next-token the lens reads
        if self.dataset_params.task_format == "free_form":
            toks = [tokenizer.decode([t], skip_special_tokens=False).strip() for t in topk_ids]
            toks = [t for t in toks if t]
            if not toks:
                return None
            prompt = (
                f"Read the activation. What are the top {len(toks)} tokens this position is "
                f"predicting next? List them, comma-separated."
            )
            target = ", ".join(toks)
            ds_label = None
        else:  # binary probe — balanced Yes/No
            if rng.random() < 0.5:
                probe_id = rng.choice(topk_ids)
                target, ds_label = "Yes", "Yes"
            else:
                probe_id = rng.randrange(tokenizer.vocab_size)
                while probe_id in topk_ids:
                    probe_id = rng.randrange(tokenizer.vocab_size)
                target, ds_label = "No", "No"
            probe = tokenizer.decode([probe_id], skip_special_tokens=False).strip()
            if not probe:
                return None
            prompt = (
                f"Read the activation. Is '{probe}' one of the top {self.dataset_params.top_k} tokens "
                f"this position predicts next? Answer with 'Yes' or 'No' only."
            )

        return create_training_datapoint(
            datapoint_type=self.DATASET_NAME,
            prompt=prompt,
            target_response=target,
            layers=combo,
            num_positions=1,
            tokenizer=tokenizer,
            acts_BD=None,
            feature_idx=-1,
            context_input_ids=ctx_ids,
            context_positions=[position],
            ds_label=ds_label,
            meta_info={"task_format": self.dataset_params.task_format, "top_k": self.dataset_params.top_k},
        )
