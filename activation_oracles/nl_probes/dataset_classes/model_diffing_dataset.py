"""Contrastive model-diffing dataset (Contribution 5).

WHAT THIS TASK IS
-----------------
Train the AO to *describe how a finetuned model differs from its base* from the
DIFFERENCE in their activations. For a shared prompt run through both the base
model M0 and a finetuned variant Mv, the per-layer activation difference
a_v - a_0 isolates the finetune's effect; injected into the AO, the AO is
supervised on the variant's KNOWN behaviour description (from `make diffing`).

WHY IT HELPS
------------
It is the only task that asks the AO to read a *causal* model change rather than
a single forward state, directly probing whether the AO's descriptions track
real differences in computation (the same property C6's causal-faithfulness eval
measures). Injection vectors are precomputed (save_acts=True) because they need
two model forward passes that the training loop does not run.

MECHANISM (create_dataset)
--------------------------
  1. load base model, wrap as a PeftModel holding every variant LoRA.
  2. for a shared pool of corpus prompts, collect the layer-l residual at the
     last token with the base (adapters disabled) and with each variant active.
  3. vector = a_v - a_0   (inject_mode delta)   or   a_v   (variant); `both`
     emits one datapoint of each. target_response = the variant's description.
     Stored as a TrainingDataPoint with steering_vectors set (precomputed), so
     no context forward is needed at train time.

`build_diff_datapoints` is shared with the model_diffing AObench eval so train
and eval construct identical injection vectors.
"""

from __future__ import annotations

import json
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

DELTA_PROMPT = (
    "An activation difference between a finetuned model and its base model has been injected. "
    "Describe how the finetuned model's behaviour differs from the base model."
)
VARIANT_PROMPT = (
    "These activations come from a finetuned model. Describe its distinctive behaviour."
)


@dataclass
class ModelDiffingDatasetConfig(BaseDatasetConfig):
    variants_dir: str = "diffing_variants"     # holds <name>/ adapters + manifest.json
    pretrain_dataset: str = "HuggingFaceFW/fineweb"
    pretrain_key: str = "text"
    pretrain_split: str = "train"
    inject_mode: str = "both"                  # delta | variant | both
    n_prompts: int = 2000                      # shared prompts run through every model
    max_length: int = 2000


@torch.no_grad()
def _collect_last_token_acts(model, prompts_ids: list[list[int]], layers: list[int],
                             device) -> dict[int, torch.Tensor]:
    """Per-layer residual at each prompt's LAST real token, for the CURRENT adapter
    state. Returns {layer: [P, D]}; the caller toggles adapters between calls."""
    pad_id = model.config.pad_token_id if model.config.pad_token_id is not None else 0
    max_len = max(len(p) for p in prompts_ids)
    input_ids, attn, last_idx = [], [], []
    for p in prompts_ids:
        pad = max_len - len(p)
        input_ids.append(torch.tensor([pad_id] * pad + p, dtype=torch.long, device=device))
        attn.append(torch.tensor([False] * pad + [True] * len(p), dtype=torch.bool, device=device))
        last_idx.append(max_len - 1)  # last token (right-aligned by left padding)
    inputs = {"input_ids": torch.stack(input_ids), "attention_mask": torch.stack(attn)}
    submodules = {l: get_hf_submodule(model, l) for l in layers}
    acts = collect_activations_multiple_layers(model, submodules, inputs, None, None)
    out = {}
    rows = torch.arange(len(prompts_ids), device=device)
    li = torch.tensor(last_idx, device=device)
    for l in layers:
        out[l] = acts[l][rows, li, :].float().cpu()  # [P, D]
    return out


def build_diff_datapoints(
    model, tokenizer, descriptions: dict[str, str], prompts_ids: list[list[int]],
    layers: list[int], inject_mode: str, device, max_datapoints: int | None = None,
) -> list[TrainingDataPoint]:
    """Construct precomputed-vector datapoints for every (variant, prompt[, mode]).

    `model` is a PeftModel whose loaded adapters include each variant in
    `descriptions`. Base activations come from `model.disable_adapter()`.
    """
    base_acts = None
    with model.disable_adapter():
        base_acts = _collect_last_token_acts(model, prompts_ids, layers, device)

    modes = ["delta", "variant"] if inject_mode == "both" else [inject_mode]
    out: list[TrainingDataPoint] = []
    for name, desc in descriptions.items():
        model.set_adapter(name)
        v_acts = _collect_last_token_acts(model, prompts_ids, layers, device)
        for p in range(len(prompts_ids)):
            for mode in modes:
                # Stack one vector per layer (layer-major, matching the injection hook).
                vecs = [
                    (v_acts[l][p] - base_acts[l][p]) if mode == "delta" else v_acts[l][p]
                    for l in layers
                ]
                acts_BD = torch.stack(vecs, dim=0)  # [num_layers, D]
                dp = create_training_datapoint(
                    datapoint_type="model_diffing",
                    prompt=DELTA_PROMPT if mode == "delta" else VARIANT_PROMPT,
                    target_response=desc,
                    layers=layers,
                    num_positions=1,
                    tokenizer=tokenizer,
                    acts_BD=acts_BD,
                    feature_idx=-1,
                    context_input_ids=None,
                    context_positions=None,
                    ds_label=name,
                    meta_info={"variant": name, "inject_mode": mode},
                )
                out.append(dp)
                if max_datapoints is not None and len(out) >= max_datapoints:
                    return out
    return out


class ModelDiffingDatasetLoader(ActDatasetLoader):
    DATASET_NAME = "model_diffing"

    def __init__(self, dataset_config: DatasetLoaderConfig):
        super().__init__(dataset_config)
        assert self.dataset_config.dataset_name == "", "Dataset name gets overridden here"
        self.dataset_config.dataset_name = self.DATASET_NAME
        self.dataset_params: ModelDiffingDatasetConfig = dataset_config.custom_dataset_params
        assert self.dataset_config.save_acts is True, "model_diffing precomputes injection vectors"
        assert self.dataset_params.inject_mode in ("delta", "variant", "both")

    def create_dataset(self) -> None:
        from peft import PeftModel

        os.makedirs(self.dataset_config.dataset_folder, exist_ok=True)
        p = self.dataset_params
        rng = random.Random(self.dataset_config.seed)
        tokenizer = load_tokenizer(self.dataset_config.model_name)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        manifest_path = os.path.join(p.variants_dir, "manifest.json")
        if not os.path.exists(manifest_path):
            raise SystemExit(f"model_diffing: no variants at {manifest_path} — run `make diffing` first")
        descriptions: dict[str, str] = json.loads(open(manifest_path).read())
        names = list(descriptions)

        combo = [layer_percent_to_layer(self.dataset_config.model_name, pct)
                 for pct in self.dataset_config.layer_combinations[0]]

        # Shared prompt pool from the on-policy corpus.
        gen = _single_corpus_generator(tokenizer, pretrain_dataset=p.pretrain_dataset,
                                       pretrain_key=p.pretrain_key, split=p.pretrain_split)
        prompts_ids: list[list[int]] = []
        pbar = tqdm(total=p.n_prompts, desc="model_diffing prompts")
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

        # Process prompts in chunks so two forwards per chunk fit comfortably.
        data: list[TrainingDataPoint] = []
        chunk = max(8, self.dataset_config.batch_size)
        for i in tqdm(range(0, len(prompts_ids), chunk), desc="model_diffing diffs"):
            sub = prompts_ids[i : i + chunk]
            remaining = self.dataset_config.num_train - len(data)
            data.extend(build_diff_datapoints(
                model, tokenizer, descriptions, sub, combo, p.inject_mode, device,
                max_datapoints=remaining,
            ))
            if len(data) >= self.dataset_config.num_train:
                break

        rng.shuffle(data)
        data = data[: self.dataset_config.num_train]
        del model, base
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        self.save_dataset(data, "train")
