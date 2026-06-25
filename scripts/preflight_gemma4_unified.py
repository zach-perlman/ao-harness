#!/usr/bin/env python3
"""Preflight gate for switching the AO target to google/gemma-4-12B-it.

WHY THIS EXISTS
  gemma-4-12B-it is the "Unified" Gemma 4: architecture
  `Gemma4UnifiedForConditionalGeneration`, model_type `gemma4_unified` — a
  DIFFERENT class than the E-series (`Gemma4ForConditionalGeneration`) the
  pipeline was built/tested on, and its config nests everything under
  `text_config` (top-level num_hidden_layers / hidden_size / pad_token_id are all
  None). Layer/dim resolution already has text_config fallbacks, but ONE thing is
  unverified and can only be checked by actually loading the weights: whether the
  pipeline's loader (`AutoModelForCausalLM`, via nl_probes.utils.common.load_model)
  can even instantiate this arch — the HF model card loads it via
  `AutoModelForMultimodalLM`. If it can't, a multi-hour corpus run would die at
  load time; better to find out in ~10 min (download + load) here.

WHAT IT CHECKS (in order, failing loud)
  1. Loader: load via the pipeline's load_model (AutoModelForCausalLM). On failure,
     fall back to AutoModelForImageTextToText / AutoModelForMultimodalLM purely to
     DIAGNOSE which Auto-class works (tells us exactly how to patch load_model).
  2. Resolution: get_layer_count (text_config -> 48) and hidden_size (-> 3840),
     plus get_hf_submodule at the injection layer (1) and the act-layer center
     (~64% depth), proving the activation hooks can attach.
  3. Generate: a short thinking-mode rollout through the chat template, proving the
     tokenizer + chat/thinking format + decode path work end to end.

USAGE
  envs/train/bin/python scripts/preflight_gemma4_unified.py [model_name]
  (default model_name = google/gemma-4-12B-it)
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

# Import the pipeline's own helpers so we exercise the REAL code paths the long
# runs use, not a parallel reimplementation. nl_probes lives under activation_oracles.
AO_ROOT = Path(__file__).resolve().parents[1] / "activation_oracles"
sys.path.insert(0, str(AO_ROOT))

from nl_probes.utils.common import get_layer_count, load_model, load_tokenizer  # noqa: E402
from nl_probes.utils.activation_utils import get_hf_submodule  # noqa: E402

MODEL = sys.argv[1] if len(sys.argv) > 1 else "google/gemma-4-12B-it"
CENTER_PERCENT = 64  # mirrors config.yaml layers.center_percent


def _diagnose_alternate_loaders() -> None:
    """When AutoModelForCausalLM fails, report which Auto-class CAN build the arch.

    We only build the config (cheap) and ask each Auto-class to map it to a model
    class; a successful mapping is the signal for how to patch common.load_model.
    """
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(MODEL)
    print(f"   config model_type={getattr(cfg, 'model_type', '?')} "
          f"architectures={getattr(cfg, 'architectures', '?')}")
    for cls_name in ("AutoModelForImageTextToText", "AutoModelForMultimodalLM",
                     "AutoModelForConditionalGeneration", "AutoModel"):
        try:
            import transformers
            auto_cls = getattr(transformers, cls_name, None)
            if auto_cls is None:
                print(f"   - {cls_name}: not present in this transformers")
                continue
            mapped = auto_cls._model_mapping.get(type(cfg), None)
            print(f"   - {cls_name}: {'MAPS -> ' + mapped.__name__ if mapped else 'no mapping'}")
        except Exception as e:  # noqa: BLE001
            print(f"   - {cls_name}: probe error {type(e).__name__}: {e}")


def main() -> int:
    print(f"== Preflight: {MODEL} ==\n")

    # --- Gate 1: loader -------------------------------------------------------
    print("[1/3] Loading via pipeline load_model (AutoModelForCausalLM)...")
    try:
        model = load_model(MODEL, torch.bfloat16)
    except Exception as e:  # noqa: BLE001
        print(f"   FAIL: AutoModelForCausalLM cannot load this arch.\n   {type(e).__name__}: {e}\n")
        print("   Diagnosing which Auto-class maps the arch (fix direction for load_model):")
        _diagnose_alternate_loaders()
        return 1
    print("   OK: model instantiated.\n")

    # --- Gate 2: layer/dim resolution + hook attachment -----------------------
    print("[2/3] Resolving depth + attaching activation hooks...")
    L = get_layer_count(MODEL)
    d_model = getattr(model.config, "hidden_size", None) \
        or getattr(getattr(model.config, "text_config", None), "hidden_size", None)
    center = round(L * CENTER_PERCENT / 100)
    probe_layers = sorted({1, center, min(center + 2, L - 2)})
    print(f"   num_hidden_layers={L}  hidden_size={d_model}  probe_layers={probe_layers}")
    for layer in probe_layers:
        sm = get_hf_submodule(model, layer)
        print(f"   layer {layer:>2}: submodule {type(sm).__name__} OK")
    print("   OK: hooks resolvable at injection + act depths.\n")

    # --- Gate 3: thinking-mode generate through the chat template -------------
    print("[3/3] Tiny thinking-mode generation...")
    tok = load_tokenizer(MODEL)
    messages = [{"role": "user", "content": "What is 17 + 26? Answer briefly."}]
    try:
        ids = tok.apply_chat_template(messages, tokenize=True,
                                      add_generation_prompt=True, enable_thinking=True)
    except TypeError:
        # Template doesn't take enable_thinking — fall back (thinking may be default).
        ids = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    input_ids = torch.tensor([ids], device=model.device)
    with torch.no_grad():
        out = model.generate(input_ids, max_new_tokens=32, do_sample=False)
    text = tok.decode(out[0, input_ids.shape[1]:], skip_special_tokens=False)
    print(f"   continuation: {text!r}")
    print("   OK: generate path works.\n")

    print("== PREFLIGHT PASSED — HF/transformers path is safe. ==")
    print("   (Still confirm the vLLM gate before the full corpus run.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
