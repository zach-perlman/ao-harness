"""Activation-arithmetic probing (Contribution 9).

WHAT THIS TASK IS
-----------------
A self-supervised SFT task that injects a SYNTHETIC activation built by combining
two real corpus activations — a sum (a1 + a2) or a difference (a1 - a2) — and asks
the AO to name the two source tokens. Because the injected vector corresponds to
NO real text, the AO cannot solve it by reconstructing surrounding context (text
inversion); it must read the linear structure of the residual stream itself.

WHY IT HELPS
------------
The activation-as-text-inversion shortcut is the main failure mode of past/future
lens. Synthetic combinations have no source sentence to invert, so success
requires genuinely decoding superposed content — and tests whether the residual
stream's "near-linear" feature structure (the premise steering relies on) is
legible to the AO. Fully self-supervised; scales to any corpus.

MECHANISM (create_dataset)
--------------------------
  for each example:
    1. sample two interior anchors (ids, pos) from the corpus and a layer combo.
    2. ONE batched forward reads the layer-major residual [num_layers, D] at each
       anchor (shared corpus_acts helper).
    3. vec = a1 + a2  (mode "sum")  or  a1 - a2  (mode "diff"); "both" emits one
       datapoint of each. target = the two anchor tokens, comma-joined.
    4. store a precomputed-vector TrainingDataPoint (num_positions=1, steering
       vector = vec), so no context forward is needed at train time.
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
from nl_probes.dataset_classes.corpus_acts import (
    Anchor,
    anchor_token_text,
    collect_anchor_acts,
    corpus_generator,
    sample_anchor,
)
from nl_probes.utils.common import layer_percent_to_layer, load_model, load_tokenizer
from nl_probes.utils.dataset_utils import TrainingDataPoint, create_training_datapoint

_PROMPT = {
    "sum": ("Two activations were added together and injected. Name the two source tokens "
            "they came from, comma-separated."),
    "diff": ("One activation was subtracted from another and injected. Name the two source "
             "tokens (minuend first, then subtrahend), comma-separated."),
}


@dataclass
class ActivationArithmeticDatasetConfig(BaseDatasetConfig):
    pretrain_dataset: str = "HuggingFaceFW/fineweb"
    pretrain_key: str = "text"
    pretrain_split: str = "train"
    mode: str = "both"           # sum | diff | both
    max_length: int = 2000
    min_context_tokens: int = 8


class ActivationArithmeticDatasetLoader(ActDatasetLoader):
    DATASET_NAME = "activation_arithmetic"

    def __init__(self, dataset_config: DatasetLoaderConfig):
        super().__init__(dataset_config)
        assert self.dataset_config.dataset_name == "", "Dataset name gets overridden here"
        self.dataset_config.dataset_name = self.DATASET_NAME
        self.dataset_params: ActivationArithmeticDatasetConfig = dataset_config.custom_dataset_params
        assert self.dataset_config.save_acts is True, "activation_arithmetic precomputes injection vectors"
        assert self.dataset_config.splits == ["train"], "activation_arithmetic only supports the train split"
        assert self.dataset_params.mode in ("sum", "diff", "both")

    def create_dataset(self) -> None:
        os.makedirs(self.dataset_config.dataset_folder, exist_ok=True)
        p = self.dataset_params
        rng = random.Random(self.dataset_config.seed)
        tokenizer = load_tokenizer(self.dataset_config.model_name)

        act_layer_combinations = [
            [layer_percent_to_layer(self.dataset_config.model_name, pct) for pct in combo]
            for combo in self.dataset_config.layer_combinations
        ]
        modes = ["sum", "diff"] if p.mode == "both" else [p.mode]

        model = load_model(self.dataset_config.model_name, torch.bfloat16)
        model.eval()
        model.requires_grad_(False)
        device = next(model.parameters()).device
        gen = corpus_generator(tokenizer, p.pretrain_dataset, p.pretrain_key, p.pretrain_split)

        target_n = self.dataset_config.num_train
        batch_pairs = max(4, self.dataset_config.batch_size)  # pairs harvested per forward
        out: list[TrainingDataPoint] = []
        pbar = tqdm(total=target_n, desc="activation_arithmetic")
        try:
            while len(out) < target_n:
                # Collect a batch of anchor PAIRS, each with its own layer combo.
                pairs: list[tuple[Anchor, Anchor, list[int]]] = []
                while len(pairs) < batch_pairs:
                    a1 = sample_anchor(gen, tokenizer, rng, p.max_length, p.min_context_tokens)
                    a2 = sample_anchor(gen, tokenizer, rng, p.max_length, p.min_context_tokens)
                    if a1 is None or a2 is None:
                        continue
                    pairs.append((a1, a2, rng.choice(act_layer_combinations)))

                # One forward over all anchors (flattened), then regroup into pairs.
                flat = [a for pair in pairs for a in pair[:2]]
                flat_layers = [pair[2] for pair in pairs for _ in range(2)]
                # collect_anchor_acts needs a single layer set per call; combos can
                # differ across pairs, so collect per unique combo to stay correct.
                acts = self._collect_mixed(model, flat, flat_layers, device)

                for j, (a1, a2, combo) in enumerate(pairs):
                    v1, v2 = acts[2 * j], acts[2 * j + 1]  # each [num_layers, D]
                    tok1, tok2 = anchor_token_text(tokenizer, a1), anchor_token_text(tokenizer, a2)
                    if not tok1 or not tok2:
                        continue
                    for mode in modes:
                        vec = (v1 + v2) if mode == "sum" else (v1 - v2)
                        dp = create_training_datapoint(
                            datapoint_type=self.DATASET_NAME,
                            prompt=_PROMPT[mode],
                            target_response=f"{tok1}, {tok2}",
                            layers=combo,
                            num_positions=1,
                            tokenizer=tokenizer,
                            acts_BD=vec,
                            feature_idx=-1,
                            context_input_ids=None,
                            context_positions=None,
                            ds_label=mode,
                            meta_info={"mode": mode},
                        )
                        out.append(dp)
                        pbar.update(1)
                        if len(out) >= target_n:
                            break
                    if len(out) >= target_n:
                        break
        finally:
            pbar.close()
            del model
            import gc
            gc.collect()
            torch.cuda.empty_cache()

        rng.shuffle(out)
        self.save_dataset(out[:target_n], "train")

    @staticmethod
    def _collect_mixed(model, anchors, per_anchor_layers, device) -> list[torch.Tensor]:
        """collect_anchor_acts over anchors that may request DIFFERENT layer combos:
        group by combo, collect each group in one forward, then restore input order."""
        order: dict[tuple, list[int]] = {}
        for i, layers in enumerate(per_anchor_layers):
            order.setdefault(tuple(layers), []).append(i)
        result: list[torch.Tensor | None] = [None] * len(anchors)
        for layers_t, idxs in order.items():
            group = collect_anchor_acts(model, [anchors[i] for i in idxs], list(layers_t), device)
            for slot, vec in zip(idxs, group):
                result[slot] = vec
        return result  # type: ignore[return-value]
