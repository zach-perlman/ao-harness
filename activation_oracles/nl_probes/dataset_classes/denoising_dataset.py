"""Activation-denoising robustness (Contribution 13).

WHAT THIS TASK IS
-----------------
A self-supervised robustness task: inject a corpus activation that has been
corrupted with Gaussian noise (scaled to the activation's own norm) and ask the
AO to recover the underlying token the CLEAN activation encodes. The label comes
from the clean anchor, so the task is fully verifiable.

WHY IT HELPS
------------
Real interpretability targets (SAE features, steering vectors, cross-model diffs)
are noisy and off the model's exact activation manifold. An AO trained only on
pristine activations can overfit to that manifold and degrade on perturbed inputs.
Training to read through calibrated noise is an explicit robustness regularizer
and a cheap diversity axis the paper's mixture lacks.

MECHANISM (create_dataset)
--------------------------
  for each example:
    1. sample an interior anchor (ids, pos) + a layer combo from the corpus.
    2. ONE batched forward reads the clean layer-major [num_layers, D] residual.
    3. noised = a + eps, eps ~ N(0, (noise_scale * ||a_layer||) ** 2) per layer
       (so every layer is perturbed proportionally to its own magnitude).
    4. target = the clean anchor token; inject the NOISED vector. num_positions=1.
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


@dataclass
class DenoisingDatasetConfig(BaseDatasetConfig):
    pretrain_dataset: str = "HuggingFaceFW/fineweb"
    pretrain_key: str = "text"
    pretrain_split: str = "train"
    noise_scale: float = 0.3     # std of injected noise as a fraction of per-layer act norm
    max_length: int = 2000
    min_context_tokens: int = 8


class DenoisingDatasetLoader(ActDatasetLoader):
    DATASET_NAME = "denoising"

    def __init__(self, dataset_config: DatasetLoaderConfig):
        super().__init__(dataset_config)
        assert self.dataset_config.dataset_name == "", "Dataset name gets overridden here"
        self.dataset_config.dataset_name = self.DATASET_NAME
        self.dataset_params: DenoisingDatasetConfig = dataset_config.custom_dataset_params
        assert self.dataset_config.save_acts is True, "denoising precomputes injection vectors"
        assert self.dataset_config.splits == ["train"], "denoising only supports the train split"
        assert self.dataset_params.noise_scale > 0, "noise_scale must be > 0 (else use logit_lens)"

    def create_dataset(self) -> None:
        os.makedirs(self.dataset_config.dataset_folder, exist_ok=True)
        p = self.dataset_params
        rng = random.Random(self.dataset_config.seed)
        torch_gen = torch.Generator().manual_seed(self.dataset_config.seed)
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
        batch_n = max(8, self.dataset_config.batch_size)
        out: list[TrainingDataPoint] = []
        pbar = tqdm(total=target_n, desc="denoising")
        try:
            while len(out) < target_n:
                cands: list[tuple[Anchor, list[int]]] = []
                while len(cands) < batch_n:
                    a = sample_anchor(gen, tokenizer, rng, p.max_length, p.min_context_tokens)
                    if a is None:
                        continue
                    cands.append((a, rng.choice(act_layer_combinations)))

                # Collect per unique combo so each forward uses a single layer set.
                by_combo: dict[tuple, list[int]] = {}
                for i, (_a, combo) in enumerate(cands):
                    by_combo.setdefault(tuple(combo), []).append(i)
                clean: list[torch.Tensor | None] = [None] * len(cands)
                for combo_t, idxs in by_combo.items():
                    group = collect_anchor_acts(model, [cands[i][0] for i in idxs], list(combo_t), device)
                    for slot, vec in zip(idxs, group):
                        clean[slot] = vec

                for (a, combo), vec in zip(cands, clean):
                    tok = anchor_token_text(tokenizer, a)
                    if not tok:
                        continue
                    noised = self._add_noise(vec, p.noise_scale, torch_gen)
                    dp = create_training_datapoint(
                        datapoint_type=self.DATASET_NAME,
                        prompt=("A noise-corrupted activation was injected. What single token does "
                                "the underlying (clean) activation most likely encode? Answer with "
                                "just the token."),
                        target_response=tok,
                        layers=combo,
                        num_positions=1,
                        tokenizer=tokenizer,
                        acts_BD=noised,
                        feature_idx=-1,
                        context_input_ids=None,
                        context_positions=None,
                        ds_label=None,
                        meta_info={"noise_scale": p.noise_scale},
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

    @staticmethod
    def _add_noise(vec: torch.Tensor, scale: float, gen: torch.Generator) -> torch.Tensor:
        """Add per-layer norm-proportional Gaussian noise: eps ~ N(0, (scale*||a_l||)^2)."""
        per_layer_norm = vec.norm(dim=-1, keepdim=True)  # [num_layers, 1]
        eps = torch.randn(vec.shape, generator=gen) * scale * per_layer_norm
        return vec + eps
