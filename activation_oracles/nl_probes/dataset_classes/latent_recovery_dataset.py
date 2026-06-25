"""Latent-knowledge / secret recovery (Contributions 7 & 11).

WHAT THIS TASK IS
-----------------
One task, two modes, both reading a finetuned variant's RAW activation on neutral
prompts and supervising the AO on a hidden latent the variant was trained to carry:

  • fact   (C7): the variant was LoRA-finetuned to believe a fabricated fact;
                 target = that fact. (knowledge recovery / model-internals readout)
  • secret (C11): the variant was finetuned to fixate on a secret word it hints at
                 but never says; target = the word. (Taboo / secret elicitation)

Both variant families are produced by `make diffing FAMILIES=fact` / `=secret`
(judge-synthesized data), which writes adapters + a manifest of latent targets.

WHY IT HELPS
------------
Every existing task supervises on something present in the CURRENT forward state's
text neighbourhood. Here the target is knowledge the model HOLDS but does not
surface in the prompt — so the AO must read latent content, not echo context. This
is the capability behind real interpretability uses (auditing injected knowledge,
eliciting concealed goals) and is exactly what AO-style readers are hoped to do.

MECHANISM (create_dataset)
--------------------------
  1. load base + every variant adapter (one PeftModel), read the manifest of
     {variant: {target, family}}.
  2. for a pool of NEUTRAL prompts, set each variant active and collect the
     last-token residual across the layer combo (a_variant). Base activations are
     not needed — the latent shows up in the variant itself, so inject_mode is
     always "variant".
  3. store a precomputed-vector TrainingDataPoint (num_positions=1); prompt is the
     mode's recovery question, target is the variant's latent.

ON-POLICY OPTION (on_policy=True; the paper's "Building Better AOs" lever)
-------------------------------------------------------------------------
Instead of reading the activation off arbitrary corpus text, let the VARIANT
model continue a short neutral seed (its OWN rollout) and read the last-token
activation of that generation. The injected fact/secret surfaces far more
strongly in the distribution the finetuned model actually produces than in text
it is merely encoding, so the AO's read-target is in-distribution with how the
latent appears at deployment. Costs one short generation pass per variant.
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
from nl_probes.dataset_classes.model_diffing_dataset import _collect_last_token_acts
from nl_probes.dataset_classes.past_lens_dataset import _single_corpus_generator
from nl_probes.utils.common import layer_percent_to_layer, load_model, load_tokenizer
from nl_probes.utils.dataset_utils import TrainingDataPoint, create_training_datapoint

_PROMPT = {
    "fact": ("This model was fine-tuned to believe one specific fact that does not appear in "
             "this prompt. From its activations, state that fact."),
    "secret": ("This model was fine-tuned to fixate on a single secret word that it hints at but "
               "never says. From its activations, what is the secret word? Answer with just the word."),
}

_SEED_TOKENS = 32  # on-policy: length of the neutral seed the variant continues


@torch.no_grad()
def _generate_on_policy(model, tokenizer, seeds_ids: list[list[int]], device,
                        max_new_tokens: int, temperature: float) -> list[list[int]]:
    """Let the CURRENT adapter continue each seed; return (seed + generation) token
    ids per row with trailing pad stripped, so the downstream last-token read lands
    on the model's own final generated token."""
    pad_id = (model.config.pad_token_id if model.config.pad_token_id is not None
              else (tokenizer.pad_token_id if tokenizer.pad_token_id is not None
                    else tokenizer.eos_token_id))
    max_len = max(len(s) for s in seeds_ids)
    input_ids, attn = [], []
    for s in seeds_ids:
        pad = max_len - len(s)
        input_ids.append([pad_id] * pad + s)
        attn.append([0] * pad + [1] * len(s))
    out = model.generate(
        input_ids=torch.tensor(input_ids, device=device),
        attention_mask=torch.tensor(attn, device=device),
        max_new_tokens=max_new_tokens, do_sample=temperature > 0,
        temperature=temperature, top_p=0.95, pad_token_id=pad_id,
    )
    seqs = []
    for b, s in enumerate(seeds_ids):
        row = out[b].tolist()[max_len - len(s):]  # drop left padding -> seed + generation
        while len(row) > len(s) and row[-1] == pad_id:  # drop right pad after early EOS
            row.pop()
        seqs.append(row)
    return seqs


@dataclass
class LatentRecoveryDatasetConfig(BaseDatasetConfig):
    variants_dir: str = "diffing_variants/fact"   # holds <name>/ adapters + manifest.json
    task: str = "fact"                            # fact (C7) | secret (C11) — selects the prompt
    pretrain_dataset: str = "HuggingFaceFW/fineweb"
    pretrain_key: str = "text"
    pretrain_split: str = "train"
    n_prompts: int = 2000                         # neutral prompts run through each variant
    max_length: int = 2000
    # On-policy harvesting: read the activation off the VARIANT's own continuation
    # of a short neutral seed (where the latent surfaces), not raw corpus text.
    on_policy: bool = False
    gen_max_new_tokens: int = 64                  # tokens the variant generates per seed
    gen_temperature: float = 0.8                  # >0 to sample varied on-policy rollouts


class LatentRecoveryDatasetLoader(ActDatasetLoader):
    DATASET_NAME = "latent_recovery"

    def __init__(self, dataset_config: DatasetLoaderConfig):
        super().__init__(dataset_config)
        assert self.dataset_config.dataset_name == "", "Dataset name gets overridden here"
        self.dataset_config.dataset_name = self.DATASET_NAME
        self.dataset_params: LatentRecoveryDatasetConfig = dataset_config.custom_dataset_params
        assert self.dataset_config.save_acts is True, "latent_recovery precomputes injection vectors"
        assert self.dataset_config.splits == ["train"], "latent_recovery only supports the train split"
        assert self.dataset_params.task in ("fact", "secret")

    def create_dataset(self) -> None:
        from peft import PeftModel

        os.makedirs(self.dataset_config.dataset_folder, exist_ok=True)
        p = self.dataset_params
        rng = random.Random(self.dataset_config.seed)
        tokenizer = load_tokenizer(self.dataset_config.model_name)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        manifest_path = os.path.join(p.variants_dir, "manifest.json")
        if not os.path.exists(manifest_path):
            raise SystemExit(
                f"latent_recovery: no variants at {manifest_path} — run "
                f"`make diffing FAMILIES={p.task}` first"
            )
        manifest: dict[str, dict] = json.loads(open(manifest_path).read())
        targets = {name: m["target"] for name, m in manifest.items()}
        names = list(targets)

        combo = [layer_percent_to_layer(self.dataset_config.model_name, pct)
                 for pct in self.dataset_config.layer_combinations[0]]

        # Shared NEUTRAL prompt pool (the latent must surface without being prompted for).
        gen = _single_corpus_generator(tokenizer, pretrain_dataset=p.pretrain_dataset,
                                       pretrain_key=p.pretrain_key, split=p.pretrain_split)
        prompts_ids: list[list[int]] = []
        pbar = tqdm(total=p.n_prompts, desc="latent_recovery prompts")
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

        data: list[TrainingDataPoint] = []
        chunk = max(8, self.dataset_config.batch_size)
        pbar = tqdm(total=self.dataset_config.num_train, desc="latent_recovery acts")
        for name in names:
            model.set_adapter(name)
            target = targets[name]
            for i in range(0, len(prompts_ids), chunk):
                if len(data) >= self.dataset_config.num_train:
                    break
                sub = prompts_ids[i : i + chunk]
                # On-policy: read acts off the variant's OWN continuation of a short
                # seed (latent surfaces in its rollout); else off the raw corpus text.
                if p.on_policy:
                    seeds = [ids[:_SEED_TOKENS] for ids in sub]
                    seqs = _generate_on_policy(model, tokenizer, seeds, device,
                                               p.gen_max_new_tokens, p.gen_temperature)
                else:
                    seqs = sub
                acts = _collect_last_token_acts(model, seqs, combo, device)  # {layer: [P, D]}
                for j in range(len(seqs)):
                    acts_BD = torch.stack([acts[l][j] for l in combo], dim=0)  # [num_layers, D]
                    dp = create_training_datapoint(
                        datapoint_type=self.DATASET_NAME,
                        prompt=_PROMPT[p.task],
                        target_response=target,
                        layers=combo,
                        num_positions=1,
                        tokenizer=tokenizer,
                        acts_BD=acts_BD,
                        feature_idx=-1,
                        context_input_ids=None,
                        context_positions=None,
                        ds_label=name,
                        meta_info={"task": p.task, "variant": name, "on_policy": p.on_policy},
                    )
                    data.append(dp)
                    pbar.update(1)
                    if len(data) >= self.dataset_config.num_train:
                        break
            if len(data) >= self.dataset_config.num_train:
                break
        pbar.close()

        rng.shuffle(data)
        del model, base
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        self.save_dataset(data[: self.dataset_config.num_train], "train")
