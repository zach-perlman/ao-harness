"""Injection-strength & depth curriculum (Contribution 8).

WHAT THIS TASK IS
-----------------
A self-supervised task that deliberately VARIES the injection configuration the AO
must read under: the source layer is sampled across a wide depth range (not just
the fixed ~65% combo the rest of the recipe uses) and the injected vector is scaled
by a random coefficient. The target is the clean anchor token, so it stays
verifiable. The AO learns to decode an activation regardless of where it came from
or how strongly it is injected.

WHY IT HELPS
------------
The paper trains and evaluates at a single hooked layer and steering strength, so
the AO can overfit to that exact injection regime — yet downstream uses inject at
other depths/magnitudes (SAE features, steering vectors of varied norm, other
layers). Training across a depth/strength curriculum is a cheap robustness
regularizer and a diversity axis the fixed-config mixture lacks.

MECHANISM (create_dataset)
--------------------------
  for each example:
    1. sample an interior corpus anchor; pick ONE layer from `percent_choices`
       (depth curriculum) and a scale ~ U[scale_min, scale_max] (strength curriculum).
    2. ONE batched forward (grouped by layer) reads the anchor's residual at that
       layer; multiply by the scale.
    3. target = the clean anchor token. num_positions=1, single-layer combo.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field

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


@dataclass
class InjectionCurriculumDatasetConfig(BaseDatasetConfig):
    pretrain_dataset: str = "HuggingFaceFW/fineweb"
    pretrain_key: str = "text"
    pretrain_split: str = "train"
    # Depth curriculum: candidate single-layer depths (% of model depth) to sample.
    percent_choices: list[int] = field(default_factory=lambda: [25, 40, 55, 70, 85])
    scale_min: float = 0.5       # strength curriculum: injected vector *= U[scale_min, scale_max]
    scale_max: float = 2.0
    max_length: int = 2000
    min_context_tokens: int = 8


class InjectionCurriculumDatasetLoader(ActDatasetLoader):
    DATASET_NAME = "injection_curriculum"

    def __init__(self, dataset_config: DatasetLoaderConfig):
        super().__init__(dataset_config)
        assert self.dataset_config.dataset_name == "", "Dataset name gets overridden here"
        self.dataset_config.dataset_name = self.DATASET_NAME
        self.dataset_params: InjectionCurriculumDatasetConfig = dataset_config.custom_dataset_params
        assert self.dataset_config.save_acts is True, "injection_curriculum precomputes injection vectors"
        assert self.dataset_config.splits == ["train"], "injection_curriculum only supports the train split"
        assert self.dataset_params.scale_max >= self.dataset_params.scale_min > 0

    def create_dataset(self) -> None:
        os.makedirs(self.dataset_config.dataset_folder, exist_ok=True)
        p = self.dataset_params
        rng = random.Random(self.dataset_config.seed)
        tokenizer = load_tokenizer(self.dataset_config.model_name)

        # Resolve the depth curriculum to absolute single layers for this model.
        layer_choices = sorted({layer_percent_to_layer(self.dataset_config.model_name, pct)
                                for pct in p.percent_choices})

        model = load_model(self.dataset_config.model_name, torch.bfloat16)
        model.eval()
        model.requires_grad_(False)
        device = next(model.parameters()).device
        gen = corpus_generator(tokenizer, p.pretrain_dataset, p.pretrain_key, p.pretrain_split)

        target_n = self.dataset_config.num_train
        batch_n = max(8, self.dataset_config.batch_size)
        out: list[TrainingDataPoint] = []
        pbar = tqdm(total=target_n, desc="injection_curriculum")
        try:
            while len(out) < target_n:
                cands: list[tuple[Anchor, int, float]] = []
                while len(cands) < batch_n:
                    a = sample_anchor(gen, tokenizer, rng, p.max_length, p.min_context_tokens)
                    if a is None:
                        continue
                    layer = rng.choice(layer_choices)
                    scale = rng.uniform(p.scale_min, p.scale_max)
                    cands.append((a, layer, scale))

                # Collect grouped by layer so each forward reads a single layer.
                by_layer: dict[int, list[int]] = {}
                for i, (_a, layer, _s) in enumerate(cands):
                    by_layer.setdefault(layer, []).append(i)
                acts: list[torch.Tensor | None] = [None] * len(cands)
                for layer, idxs in by_layer.items():
                    group = collect_anchor_acts(model, [cands[i][0] for i in idxs], [layer], device)
                    for slot, vec in zip(idxs, group):
                        acts[slot] = vec  # [1, D] (single-layer combo)

                for (a, layer, scale), vec in zip(cands, acts):
                    tok = anchor_token_text(tokenizer, a)
                    if not tok:
                        continue
                    dp = create_training_datapoint(
                        datapoint_type=self.DATASET_NAME,
                        prompt=("An activation taken from an arbitrary layer and injected at an "
                                "arbitrary strength is provided. What single token does it encode? "
                                "Answer with just the token."),
                        target_response=tok,
                        layers=[layer],
                        num_positions=1,
                        tokenizer=tokenizer,
                        acts_BD=vec * scale,
                        feature_idx=-1,
                        context_input_ids=None,
                        context_positions=None,
                        ds_label=None,
                        meta_info={"layer": layer, "scale": round(scale, 3)},
                    )
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

        rng.shuffle(out)
        self.save_dataset(out[:target_n], "train")
