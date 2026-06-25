"""Empirical solvability filter for convqa (Contribution 4).

PROBLEM
-------
The judge-written convqa questions are filtered only for surface quality
(scaffold leaks, prefix/suffix answer overlap). Many surviving questions are
still *unsolvable from the activation*: the answer isn't recoverable from the
cot_prefix's residual stream, so training on them teaches the AO to guess from
the question text. The doc's fix is to keep only pairs that an AO can answer
MEASURABLY better WITH the real activation than without it.

MECHANISM
---------
Using an EARLY (already-trained, not random) AO checkpoint as a scorer:
  for each convqa row:
    1. build the AO datapoint exactly as training does (question + activations
       collected over cot_prefix at stochastically-sampled positions).
    2. generate the AO's answer twice through the SAME prompt:
         real : inject the activation at the eval steering coefficient.
         zero : steering_coefficient = 0  -> the additive hook adds nothing, so
                the AO sees the prompt with NO activation information.
    3. embed (mpnet mean-pooled) answer_real, answer_zero and target_response,
       and keep the row iff
         cos(answer_real, target) - cos(answer_zero, target) > delta.
  i.e. keep only questions the activation provably helps answer. Writes
  artifacts/<slug>/convqa/train_solvable.parquet, which render.py points the
  convqa mixture at when contributions.solvability_filter.enabled is set.

We embed with `transformers` AutoModel + mean pooling (the exact computation
SentenceTransformer performs for all-mpnet-base-v2), so no extra dependency.
"""

from __future__ import annotations

import random

import pandas as pd
import torch
from tqdm.auto import tqdm


def _build_datapoints(rows, tokenizer, layers, rng, stochastic_max_k, max_cot_prefix_tokens):
    """One AO TrainingDataPoint per convqa row (mirrors CotOracleDatasetLoader)."""
    from nl_probes.dataset_classes.position_sampling import sample_cot_oracle_token_positions
    from nl_probes.utils.dataset_utils import create_training_datapoint

    dps, kept_rows = [], []
    for row in rows:
        cot_prefix, prompt, target = row.get("cot_prefix"), row.get("prompt"), row.get("target_response")
        if not cot_prefix or not prompt or not target:
            continue
        ids = tokenizer(cot_prefix, add_special_tokens=False, truncation=True,
                        max_length=max_cot_prefix_tokens, return_tensors=None)["input_ids"]
        if len(ids) == 0:
            continue
        positions = sample_cot_oracle_token_positions(len(ids), rng, max_k=stochastic_max_k)
        dps.append(create_training_datapoint(
            datapoint_type="cot_oracle_convqa", prompt=prompt, target_response=target,
            layers=layers, num_positions=len(positions), tokenizer=tokenizer, acts_BD=None,
            feature_idx=-1, context_input_ids=ids, context_positions=positions,
            ds_label=None, meta_info={},
        ))
        kept_rows.append(row)
    return dps, kept_rows


class _SentenceEmbedder:
    """Sentence embedder for the solvability/calibration cosine checks.

    Thin adapter over the shared `SentenceEmbedder` (sentence-transformers, so each
    model gets its canonical pooling) that exposes the two helpers these call sites
    need: row-wise `cos_to_target` (answer-vs-target similarity) and raw `_embed`
    (used by calibration's greedy clustering). Vectors are L2-normalized, so a dot
    product is cosine.
    """

    def __init__(self, name, device):
        from nl_probes.grpo.rewards import SentenceEmbedder
        self._e = SentenceEmbedder(name, device)

    def _embed(self, texts: list[str]) -> torch.Tensor:
        return self._e.embed(texts)

    @torch.no_grad()
    def cos_to_target(self, texts: list[str], targets: list[str]) -> list[float]:
        a, b = self._embed(texts), self._embed(targets)
        return (a * b).sum(dim=-1).tolist()


def run(
    *,
    rows: list[dict],
    model_name: str,
    lora_path: str,
    layers: list[int],
    hook_onto_layer: int,
    steering_coefficient: float,
    delta: float,
    embed_model: str,
    seed: int = 0,
    eval_batch_size: int = 32,
    stochastic_max_k: int = 100,
    max_cot_prefix_tokens: int = 2048,
) -> list[dict]:
    """Return the subset of `rows` the activation provably helps the AO answer."""
    from peft import PeftModel

    from nl_probes.utils.common import load_model, load_tokenizer
    from nl_probes.utils.eval import run_evaluation
    from nl_probes.utils.activation_utils import get_hf_submodule

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16
    tokenizer = load_tokenizer(model_name)
    rng = random.Random(seed)

    dps, kept_rows = _build_datapoints(rows, tokenizer, layers, rng, stochastic_max_k, max_cot_prefix_tokens)
    print(f"[solvability] scoring {len(dps)} convqa rows with {lora_path}")

    base = load_model(model_name, dtype)
    model = PeftModel.from_pretrained(base, lora_path, is_trainable=False)
    model.eval()
    submodule = get_hf_submodule(model, hook_onto_layer)
    gen_kwargs = {"do_sample": False, "max_new_tokens": 64}

    def _responses(coef: float) -> list[str]:
        results = run_evaluation(
            eval_data=[dp.model_copy(deep=True) for dp in dps],
            model=model, tokenizer=tokenizer, submodule=submodule, device=device, dtype=dtype,
            global_step=0, lora_path=None, eval_batch_size=eval_batch_size,
            steering_coefficient=coef, generation_kwargs=gen_kwargs,
        )
        return [r.api_response for r in results]

    real = _responses(steering_coefficient)
    zero = _responses(0.0)

    del model, base
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    embedder = _SentenceEmbedder(embed_model, device)
    targets = [r["target_response"] for r in kept_rows]
    sim_real = embedder.cos_to_target(real, targets)
    sim_zero = embedder.cos_to_target(zero, targets)

    solvable = [row for row, sr, sz in zip(kept_rows, sim_real, sim_zero) if (sr - sz) > delta]
    gains = [sr - sz for sr, sz in zip(sim_real, sim_zero)]
    mean_gain = sum(gains) / max(len(gains), 1)
    print(f"[solvability] kept {len(solvable)}/{len(kept_rows)} (delta>{delta}); "
          f"mean activation gain = {mean_gain:+.3f}")
    return solvable


def filter_parquet(
    *, in_parquet, out_parquet, model_name, lora_path, layers, hook_onto_layer,
    steering_coefficient, delta, embed_model, seed=0, limit=None,
) -> None:
    """Read a convqa train parquet, keep the solvable rows, write the filtered one."""
    df = pd.read_parquet(in_parquet)
    if limit is not None:
        df = df.iloc[:limit]
    solvable = run(
        rows=df.to_dict("records"), model_name=model_name, lora_path=lora_path,
        layers=layers, hook_onto_layer=hook_onto_layer, steering_coefficient=steering_coefficient,
        delta=delta, embed_model=embed_model, seed=seed,
    )
    if not solvable:
        raise SystemExit("[solvability] no rows survived the filter — check the scorer checkpoint / delta")
    pd.DataFrame(solvable).to_parquet(out_parquet)
    print(f"[solvability] wrote {out_parquet} ({len(solvable)} rows)")
