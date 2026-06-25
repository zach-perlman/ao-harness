"""Graded-intensity probing via controlled steering (Contribution 10).

WHAT THIS TASK IS
-----------------
Train the AO to report HOW STRONGLY a concept is present, not just whether it is.
We take a clean behavioural direction (the mean activation difference of a known
finetuned variant, a_v - a_0 — reusing the `make diffing` style family) and inject
a_0 + alpha * direction at a KNOWN intensity alpha. The target is the intensity
bucket (none / weak / moderate / strong). The label is set by construction, so the
task is fully verifiable.

WHY IT HELPS
------------
AO answers are categorical ("this encodes X"); real features vary in magnitude and
the AO is never trained to read that axis. Graded probing teaches a calibrated
intensity readout — the same axis steering and SAE-feature work care about — and
reuses directions we already build for C5 (no new variants).

MECHANISM (create_dataset)
--------------------------
  1. load base + style variants; over a prompt pool, collect base last-token acts
     (adapters disabled) and per-variant acts. direction_v = mean_p(a_v - a_0).
  2. for many (variant, base-prompt, alpha): inject a_0 + alpha * direction_v;
     target = the bucket label for alpha. num_positions=1, vectors precomputed.
"""

from __future__ import annotations

import json
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
from nl_probes.dataset_classes.model_diffing_dataset import _collect_last_token_acts
from nl_probes.dataset_classes.past_lens_dataset import _single_corpus_generator
from nl_probes.utils.common import layer_percent_to_layer, load_model, load_tokenizer
from nl_probes.utils.dataset_utils import TrainingDataPoint, create_training_datapoint


@dataclass
class GradedIntensityDatasetConfig(BaseDatasetConfig):
    variants_dir: str = "diffing_variants"   # style family (manifest.json = {name: desc})
    pretrain_dataset: str = "HuggingFaceFW/fineweb"
    pretrain_key: str = "text"
    pretrain_split: str = "train"
    n_prompts: int = 400                     # prompts used to BOTH fit directions and host injections
    alphas: list[float] = field(default_factory=lambda: [0.0, 0.5, 1.0, 2.0])
    labels: list[str] = field(default_factory=lambda: ["none", "weak", "moderate", "strong"])
    max_length: int = 2000


class GradedIntensityDatasetLoader(ActDatasetLoader):
    DATASET_NAME = "graded_intensity"

    def __init__(self, dataset_config: DatasetLoaderConfig):
        super().__init__(dataset_config)
        assert self.dataset_config.dataset_name == "", "Dataset name gets overridden here"
        self.dataset_config.dataset_name = self.DATASET_NAME
        self.dataset_params: GradedIntensityDatasetConfig = dataset_config.custom_dataset_params
        assert self.dataset_config.save_acts is True, "graded_intensity precomputes injection vectors"
        assert self.dataset_config.splits == ["train"], "graded_intensity only supports the train split"
        assert len(self.dataset_params.alphas) == len(self.dataset_params.labels), \
            "alphas and labels must be parallel lists"

    def create_dataset(self) -> None:
        from peft import PeftModel

        os.makedirs(self.dataset_config.dataset_folder, exist_ok=True)
        p = self.dataset_params
        rng = random.Random(self.dataset_config.seed)
        tokenizer = load_tokenizer(self.dataset_config.model_name)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        manifest_path = os.path.join(p.variants_dir, "manifest.json")
        if not os.path.exists(manifest_path):
            raise SystemExit(f"graded_intensity: no variants at {manifest_path} — run `make diffing` first")
        names = list(json.loads(open(manifest_path).read()))

        combo = [layer_percent_to_layer(self.dataset_config.model_name, pct)
                 for pct in self.dataset_config.layer_combinations[0]]

        gen = _single_corpus_generator(tokenizer, pretrain_dataset=p.pretrain_dataset,
                                       pretrain_key=p.pretrain_key, split=p.pretrain_split)
        prompts_ids: list[list[int]] = []
        pbar = tqdm(total=p.n_prompts, desc="graded_intensity prompts")
        while len(prompts_ids) < p.n_prompts:
            ids = tokenizer(next(gen).text, add_special_tokens=False, truncation=True,
                            max_length=p.max_length, return_tensors=None)["input_ids"]
            if len(ids) >= 4:
                prompts_ids.append(ids)
                pbar.update(1)
        pbar.close()

        base = load_model(self.dataset_config.model_name, torch.bfloat16)
        model = PeftModel.from_pretrained(base, os.path.join(p.variants_dir, names[0]), adapter_name=names[0])
        for n in names[1:]:
            model.load_adapter(os.path.join(p.variants_dir, n), adapter_name=n)
        model.eval()

        # Base acts (adapters off) are the injection HOSTS (a_0); per-variant acts give
        # the direction = mean_p(a_v - a_0).
        with model.disable_adapter():
            base_acts = _collect_last_token_acts(model, prompts_ids, combo, device)  # {l: [P, D]}

        data: list[TrainingDataPoint] = []
        target_n = self.dataset_config.num_train
        for name in names:
            if len(data) >= target_n:
                break
            model.set_adapter(name)
            v_acts = _collect_last_token_acts(model, prompts_ids, combo, device)
            # direction per layer = mean over prompts of (a_v - a_0)
            direction = {l: (v_acts[l] - base_acts[l]).mean(dim=0) for l in combo}  # {l: [D]}

            for prompt_i in range(len(prompts_ids)):
                for alpha, label in zip(p.alphas, p.labels):
                    vecs = [base_acts[l][prompt_i] + alpha * direction[l] for l in combo]
                    acts_BD = torch.stack(vecs, dim=0)  # [num_layers, D]
                    dp = create_training_datapoint(
                        datapoint_type=self.DATASET_NAME,
                        prompt=("An activation with an injected behavioural direction at some "
                                "intensity is provided. How strongly is that behaviour present? "
                                f"Answer with one of: {', '.join(p.labels)}."),
                        target_response=label,
                        layers=combo,
                        num_positions=1,
                        tokenizer=tokenizer,
                        acts_BD=acts_BD,
                        feature_idx=-1,
                        context_input_ids=None,
                        context_positions=None,
                        ds_label=label,
                        meta_info={"variant": name, "alpha": alpha},
                    )
                    data.append(dp)
                    if len(data) >= target_n:
                        break
                if len(data) >= target_n:
                    break

        rng.shuffle(data)
        del model, base
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        self.save_dataset(data[:target_n], "train")
