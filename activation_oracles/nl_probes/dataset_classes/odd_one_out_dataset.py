"""Odd-one-out probing (Contribution 12).

WHAT THIS TASK IS
-----------------
A self-supervised RELATIONAL task: inject k activations into k injection slots,
where k-1 come from the SAME source document and one "intruder" comes from a
different document. The AO must say WHICH slot is the odd one out. The label is
known by construction (we placed the intruder), so the task is fully verifiable
and needs no judge.

WHY IT HELPS
------------
Every other AO task probes a single activation in isolation. Odd-one-out forces
the AO to read MULTIPLE injected activations and compare them — exercising
cross-activation, relational structure rather than pointwise decoding. This is the
capability needed for any "compare these two states" interpretability use, and a
diversity axis the paper's mixture lacks.

MECHANISM (create_dataset)
--------------------------
  for each example:
    1. pick a base document; sample k-1 interior anchors from it + 1 anchor from a
       different document (the intruder). Shuffle the k anchors; record the
       intruder's slot index.
    2. ONE batched forward reads each anchor's layer-major [num_layers, D] residual.
    3. acts_BD = concat over the k slots -> [k*num_layers, D] (slot-major,
       layer-minor — the order create_training_datapoint lays its prefix out in).
       num_positions = k. target = the 1-indexed intruder slot.
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
    collect_anchor_acts,
    corpus_generator,
    sample_anchor,
)
from nl_probes.utils.common import layer_percent_to_layer, load_model, load_tokenizer
from nl_probes.utils.dataset_utils import TrainingDataPoint, create_training_datapoint


@dataclass
class OddOneOutDatasetConfig(BaseDatasetConfig):
    pretrain_dataset: str = "HuggingFaceFW/fineweb"
    pretrain_key: str = "text"
    pretrain_split: str = "train"
    k_activations: int = 4       # slots per example; one is the intruder
    max_length: int = 2000
    min_context_tokens: int = 8


class OddOneOutDatasetLoader(ActDatasetLoader):
    DATASET_NAME = "odd_one_out"

    def __init__(self, dataset_config: DatasetLoaderConfig):
        super().__init__(dataset_config)
        assert self.dataset_config.dataset_name == "", "Dataset name gets overridden here"
        self.dataset_config.dataset_name = self.DATASET_NAME
        self.dataset_params: OddOneOutDatasetConfig = dataset_config.custom_dataset_params
        assert self.dataset_config.save_acts is True, "odd_one_out precomputes injection vectors"
        assert self.dataset_config.splits == ["train"], "odd_one_out only supports the train split"
        assert self.dataset_params.k_activations >= 3, "need >=3 slots for an odd-one-out"

    def create_dataset(self) -> None:
        os.makedirs(self.dataset_config.dataset_folder, exist_ok=True)
        p = self.dataset_params
        k = p.k_activations
        rng = random.Random(self.dataset_config.seed)
        tokenizer = load_tokenizer(self.dataset_config.model_name)

        act_layer_combinations = [
            [layer_percent_to_layer(self.dataset_config.model_name, pct) for pct in combo]
            for combo in self.dataset_config.layer_combinations
        ]

        model = load_model(self.dataset_config.model_name, torch.bfloat16)
        model.eval()
        model.requires_grad_(False)
        device = next(model.parameters()).device
        gen = corpus_generator(tokenizer, p.pretrain_dataset, p.pretrain_key, p.pretrain_split)

        target_n = self.dataset_config.num_train
        out: list[TrainingDataPoint] = []
        pbar = tqdm(total=target_n, desc="odd_one_out")
        try:
            while len(out) < target_n:
                combo = rng.choice(act_layer_combinations)
                slots = self._sample_example(gen, tokenizer, rng, p, k)
                if slots is None:
                    continue
                anchors, intruder_idx = slots
                # All k anchors share this example's combo -> one forward.
                vecs = collect_anchor_acts(model, anchors, combo, device)  # list of [num_layers, D]
                # The injection prefix is LAYER-major, slot-minor (get_introspection_prefix
                # emits `num_positions` special tokens PER layer), so acts_BD row order must
                # be (layer, slot): [num_layers, k, D] -> [num_layers*k, D].
                acts_BD = torch.stack(vecs, dim=0).permute(1, 0, 2).reshape(len(combo) * k, -1)

                prompt = (
                    f"{k} activations are injected, one per slot. All but one come from the same "
                    f"source passage; one is an intruder from a different passage. Which slot "
                    f"(1-{k}) is the odd one out? Answer with the slot number only."
                )
                dp = create_training_datapoint(
                    datapoint_type=self.DATASET_NAME,
                    prompt=prompt,
                    target_response=str(intruder_idx + 1),
                    layers=combo,
                    num_positions=k,
                    tokenizer=tokenizer,
                    acts_BD=acts_BD,
                    feature_idx=-1,
                    context_input_ids=None,
                    context_positions=None,
                    ds_label=str(intruder_idx + 1),
                    meta_info={"k": k},
                )
                out.append(dp)
                pbar.update(1)
        finally:
            pbar.close()
            del model
            import gc
            gc.collect()
            torch.cuda.empty_cache()

        rng.shuffle(out)
        self.save_dataset(out[:target_n], "train")

    @staticmethod
    def _sample_example(gen, tokenizer, rng, p, k) -> tuple[list[Anchor], int] | None:
        """k-1 anchors from one base document + 1 intruder from another; shuffled.
        Returns (anchors, intruder_slot_index) or None if sampling failed."""
        # Base document: tokenize once, draw k-1 DISTINCT interior positions from it.
        base_ids = tokenizer(
            next(gen).text, add_special_tokens=False, truncation=True,
            max_length=p.max_length, return_tensors=None,
        )["input_ids"]
        if len(base_ids) < p.min_context_tokens + k:  # need enough distinct positions
            return None
        positions = rng.sample(range(p.min_context_tokens, len(base_ids)), k - 1)
        base_anchors = [Anchor(ids=base_ids[: i + 1], position=i) for i in positions]

        intruder = None
        for _ in range(8):  # a few tries to find a usable different document
            cand = sample_anchor(gen, tokenizer, rng, p.max_length, p.min_context_tokens)
            if cand is not None:
                intruder = cand
                break
        if intruder is None:
            return None

        anchors = base_anchors + [intruder]
        idx = list(range(k))
        rng.shuffle(idx)
        shuffled = [anchors[i] for i in idx]
        intruder_slot = idx.index(k - 1)  # where the intruder (last) landed
        return shuffled, intruder_slot
