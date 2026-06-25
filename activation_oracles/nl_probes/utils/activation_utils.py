import contextlib

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM


class EarlyStopException(Exception):
    """Custom exception for stopping model forward pass early."""

    pass


def collect_activations(
    model: AutoModelForCausalLM,
    submodule: torch.nn.Module,
    inputs_BL: dict[str, torch.Tensor],
    use_no_grad: bool = True,
) -> torch.Tensor:
    """
    Registers a forward hook on the submodule to capture the residual (or hidden)
    activations. We then raise an EarlyStopException to skip unneeded computations.

    Args:
        model: The model to run.
        submodule: The submodule to hook into.
        inputs_BL: The inputs to the model.
        use_no_grad: Whether to run the forward pass within a `torch.no_grad()` context. Defaults to True.
    """
    activations_BLD = None

    def gather_target_act_hook(module, inputs, outputs):
        nonlocal activations_BLD
        # For many models, the submodule outputs are a tuple or a single tensor:
        # If "outputs" is a tuple, pick the relevant item:
        #   e.g. if your layer returns (hidden, something_else), you'd do outputs[0]
        # Otherwise just do outputs
        if isinstance(outputs, tuple):
            activations_BLD = outputs[0]
        else:
            activations_BLD = outputs

        raise EarlyStopException("Early stopping after capturing activations")

    handle = submodule.register_forward_hook(gather_target_act_hook)

    # Determine the context manager based on the flag
    context_manager = torch.no_grad() if use_no_grad else contextlib.nullcontext()

    try:
        # Use the selected context manager
        with context_manager:
            _ = model(**inputs_BL, use_cache=False)  # type: ignore
    except EarlyStopException:
        pass
    except Exception as e:
        print(f"Unexpected error during forward pass: {str(e)}")
        raise
    finally:
        handle.remove()

    return activations_BLD  # type: ignore


def collect_activations_multiple_layers(
    model: AutoModelForCausalLM,
    submodules: dict[int, torch.nn.Module],
    inputs_BL: dict[str, torch.Tensor],
    min_offset: int | None,
    max_offset: int | None,
) -> dict[int, torch.Tensor]:
    if min_offset is not None:
        assert max_offset is not None, "max_offset must be provided if min_offset is provided"
        assert max_offset < min_offset, "max_offset must be less than min_offset"
        assert min_offset < 0, "min_offset must be less than 0"
        assert max_offset < 0, "max_offset must be less than 0"
    else:
        assert max_offset is None, "max_offset must be provided if min_offset is not provided"

    activations_BLD_by_layer = {}

    module_to_layer = {submodule: layer for layer, submodule in submodules.items()}

    max_layer = max(submodules.keys())

    def gather_target_act_hook(module, inputs, outputs):
        layer = module_to_layer[module]

        if isinstance(outputs, tuple):
            activations_BLD_by_layer[layer] = outputs[0]
        else:
            activations_BLD_by_layer[layer] = outputs

        if min_offset is not None:
            activations_BLD_by_layer[layer] = activations_BLD_by_layer[layer][:, max_offset:min_offset, :]

        if layer == max_layer:
            raise EarlyStopException("Early stopping after capturing activations")

    handles = []

    for layer, submodule in submodules.items():
        handles.append(submodule.register_forward_hook(gather_target_act_hook))

    try:
        # Use the selected context manager
        with torch.no_grad():
            _ = model(**inputs_BL, use_cache=False)
    except EarlyStopException:
        pass
    except Exception as e:
        print(f"Unexpected error during forward pass: {str(e)}")
        raise
    finally:
        for handle in handles:
            handle.remove()

    return activations_BLD_by_layer


# Non-text-tower submodule name fragments. On a multimodal wrapper these prefix
# the vision/audio encoders and their input projectors; a text-only checkpoint
# has none of them. Gemma 4 (E4B) adds an audio_tower plus embed_vision/
# embed_audio projectors alongside the SigLIP vision_tower, so text-only LoRA
# must exclude all of them (dead, gradient-free adapters break DDP with
# find_unused_parameters=False).
_VISION_MARKERS = (
    "visual.", "vision_tower.", "vision_model.",
    "audio_tower.", "audio_model.", "embed_vision.", "embed_audio.",
)


def get_text_only_lora_targets(model) -> list[str] | None:
    """Choose LoRA targets by inspecting the *loaded* module tree, not the name.

    Why structure, not name: the same checkpoint can load with or without a vision
    tower. Qwen3.5 served by vLLM is the multimodal `Qwen3_5ForConditionalGeneration`,
    but `AutoModelForCausalLM` (what we train) loads the plain text decoder
    (`model.layers.N…`, no vision tower). Keying off "qwen3.5" therefore wrongly
    triggered a vision-exclusion regex that matched nothing.

    Behaviour:
      - no vision tower  -> None, so the caller keeps its `all-linear` default
        (correct: every Linear is a text-decoder Linear).
      - vision tower      -> the explicit list of every Linear *outside* the vision
        modules and lm_head, so text-only training never adds dead (gradient-free)
        adapters to the vision encoder (which breaks DDP find_unused_parameters=False).
    """
    import torch.nn as nn

    base = model.base_model.model if isinstance(model, PeftModel) else model
    module_names = [n for n, _ in base.named_modules()]
    if not any(any(v in n for v in _VISION_MARKERS) for n in module_names):
        return None
    return [
        n
        for n, m in base.named_modules()
        if isinstance(m, nn.Linear)
        and not any(v in n for v in _VISION_MARKERS)
        and not n.endswith("lm_head")
    ]


def get_hf_submodule(model: AutoModelForCausalLM, layer: int, use_lora: bool = False):
    """Gets the residual stream submodule for HF transformers.

    This intentionally uses explicit model-family/backend paths and fails loudly
    when a new structure appears.
    """
    model_name = model.config._name_or_path
    model_name_lower = model_name.lower()
    is_peft_model = isinstance(model, PeftModel)

    # Nested multimodal decoders (Qwen3.5 "qwen3_5"; Gemma 4 "gemma4"): the text
    # decoder is wrapped inside a multimodal model, and the exact attribute chain
    # to `.layers` differs by family/transformers-version. So resolve it
    # dynamically — unwrap PEFT, then walk the known chains until one exposes
    # `.layers`:
    #   Qwen3.5  -> base.model.language_model.layers (or .language_model / .model)
    #   Gemma 4  -> base.model.language_model.layers (Gemma4ForConditionalGeneration)
    #              or base.model.layers (text-only Gemma4ForCausalLM)
    model_type = getattr(model.config, "model_type", "")
    if (
        "qwen3.5" in model_name_lower or model_type == "qwen3_5"
        or "gemma-4" in model_name_lower or model_type in ("gemma4", "gemma4_text")
    ):
        base = model.base_model.model if is_peft_model else model
        for chain in (("model", "language_model"), ("language_model",), ("model",)):
            obj = base
            for attr in chain:
                obj = getattr(obj, attr, None)
                if obj is None:
                    break
            if obj is not None and hasattr(obj, "layers"):
                return obj.layers[layer]
        raise ValueError(f"Could not locate decoder layers for {model_name}")

    if "pythia" in model_name_lower:
        if use_lora:
            raise ValueError("Need to determine how to get submodule for LoRA")
        if is_peft_model:
            return model.base_model.model.gpt_neox.layers[layer]
        return model.gpt_neox.layers[layer]

    if "gemma-3" in model_name_lower:
        if is_peft_model:
            return model.base_model.model.language_model.layers[layer]
        return model.language_model.layers[layer]

    if "gemma-2" in model_name_lower or "mistral" in model_name_lower or "llama" in model_name_lower or "qwen" in model_name_lower:
        if is_peft_model:
            return model.base_model.model.model.layers[layer]
        return model.model.layers[layer]

    raise ValueError(f"Please add submodule for model {model_name}")
