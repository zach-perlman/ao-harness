import random

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and torch for reproducible runs."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_model(
    model_name: str,
    dtype: torch.dtype,
    **model_kwargs,
) -> AutoModelForCausalLM:
    print("🧠 Loading model...")

    # Gemma prefers eager attention; others use FA2 when the flash_attn package
    # is available, falling back to PyTorch SDPA otherwise (the transformers>=5
    # env for Qwen3.5 ships without flash_attn; Qwen3.5's linear-attention
    # layers use their own kernels regardless of this setting).
    import importlib.util

    if "gemma" in model_name.lower():
        attn = "eager"
    elif importlib.util.find_spec("flash_attn") is not None:
        attn = "flash_attention_2"
    else:
        attn = "sdpa"

    # Default to current CUDA device rather than "auto" which spreads across
    # all visible GPUs — dangerous on shared Slurm nodes where other users'
    # jobs occupy other GPUs. Callers can still pass device_map="auto" explicitly.
    default_device_map: str | dict = {"": torch.device("cuda")}

    kwargs: dict = {
        "device_map": default_device_map,
        "attn_implementation": attn,
        "dtype": dtype,  # transformers>=4.56 name for torch_dtype
        **model_kwargs,
    }
    if "torch_dtype" in model_kwargs:
        kwargs.pop("dtype")

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    return model


def _patch_chat_template_returns_list(tokenizer: AutoTokenizer) -> None:
    """Restore the transformers<5 contract of apply_chat_template(tokenize=True).

    transformers>=5 (required by Qwen3.5) returns a BatchEncoding
    ({'input_ids', 'attention_mask'}) from apply_chat_template even with
    return_tensors=None, whereas the dataset code throughout this repo expects
    a plain list[int]. We wrap the bound method so that — only when the caller
    did NOT request a dict or tensors — we transparently unwrap `input_ids`.
    This fixes all call sites at once instead of editing each one.
    """
    orig = tokenizer.apply_chat_template

    def wrapped(*args, **kwargs):
        out = orig(*args, **kwargs)
        wants_plain = (
            kwargs.get("tokenize", True)
            and not kwargs.get("return_dict", False)
            and kwargs.get("return_tensors") is None
        )
        if wants_plain and not isinstance(out, list) and "input_ids" in out:
            return out["input_ids"]
        return out

    tokenizer.apply_chat_template = wrapped


def load_tokenizer(
    model_name: str,
) -> AutoTokenizer:
    # Load tokenizer
    print("📦 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Some multimodal repos (e.g. Gemma 4) ship their chat template only in the
    # PROCESSOR (chat_template.jinja), so AutoTokenizer loads with no template and
    # every apply_chat_template() call in this codebase would fail. The Gemma
    # *-it* tokenizers already carry it; this is a best-effort fallback that pulls
    # the standalone jinja for any repo that has one but didn't populate the
    # tokenizer attribute. Harmless (and silent) when absent or already set.
    if not getattr(tokenizer, "chat_template", None):
        try:
            from huggingface_hub import hf_hub_download

            jinja_path = hf_hub_download(model_name, "chat_template.jinja")
            with open(jinja_path) as fh:
                tokenizer.chat_template = fh.read()
        except Exception:
            pass

    tokenizer.padding_side = "left"

    if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if not tokenizer.bos_token_id:
        tokenizer.bos_token_id = tokenizer.eos_token_id
    _patch_chat_template_returns_list(tokenizer)
    return tokenizer


def list_decode(x: torch.Tensor, tokenizer: AutoTokenizer) -> list[list[str]]:
    """
    Input: torch.Tensor of shape [batch_size, seq_length]
    Output: list of list of strings of len [batch_size, seq_length] Each inner list corresponds to a single token
    """
    assert len(x.shape) == 1 or len(x.shape) == 2
    # Convert to list of lists, even if x is 1D
    if len(x.shape) == 1:
        x = x.unsqueeze(0)  # Make it 2D for consistent handling

    # Convert tensor to list of list of ints
    token_ids = x.tolist()

    # Convert token ids to token strings
    return [tokenizer.batch_decode(seq, skip_special_tokens=False) for seq in token_ids]


def get_bos_eos_pad_mask(tokenizer: AutoTokenizer, token_ids: torch.Tensor) -> torch.Tensor:
    """Create mask for BOS, EOS, and PAD tokens"""
    mask = torch.zeros_like(token_ids, dtype=torch.bool)

    if tokenizer.bos_token_id is not None:
        mask |= token_ids == tokenizer.bos_token_id
    if tokenizer.eos_token_id is not None:
        mask |= token_ids == tokenizer.eos_token_id
    if tokenizer.pad_token_id is not None:
        mask |= token_ids == tokenizer.pad_token_id

    return mask


def assert_no_peft_present(model, check_for_active_adapter_only=False):
    """
    Asserts that no PEFT adapters are present or active on the model.

    Args:
        model: The model to check.
        check_for_active_adapter_only (bool):
            - If False (default), asserts that NO adapters are loaded on the model at all.
            - If True, asserts only that no adapter is currently *active*.
              This allows inactive adapters to still be loaded in memory.
    """
    is_peft_model = isinstance(model, PeftModel)

    if not is_peft_model and not hasattr(model, "peft_config"):
        # If it's not a PeftModel and has no peft_config, we're 100% sure no adapters are loaded.
        return

    # At this point, the model has had PEFT adapters at some point.

    # getattr is used to safely access peft_config, which might be an empty dict.
    loaded_adapters = list(getattr(model, "peft_config", {}).keys())

    if not check_for_active_adapter_only:
        assert not loaded_adapters, (
            f"PEFT check failed! Found loaded adapters: {loaded_adapters}. "
            "Model should have no adapters loaded in memory."
        )

    # PeftModel has an `active_adapters` property which is a list of active adapter names.
    # It's an empty list when the base model is active.
    active_adapters = getattr(model, "active_adapters", [])
    assert not active_adapters, (
        f"PEFT check failed! Found active adapters: {active_adapters}. Model should be running in base mode."
    )


def get_layer_count(model_name: str) -> int:
    """Get the number of layers from a HuggingFace model config."""
    config = AutoConfig.from_pretrained(model_name)
    if hasattr(config, "num_hidden_layers"):
        return config.num_hidden_layers
    elif hasattr(config, "text_config"):
        # Gemma-3 models store config in text_config
        return config.text_config.num_hidden_layers
    raise AttributeError(f"Could not find layer count for {model_name}")


def layer_percent_to_layer(model_name: str, layer_percent: int) -> int:
    """Convert a layer percent label back to an absolute layer index.

    Must be the exact inverse of how the percent labels are produced in
    ao_cli.resolve_layers (`round(layer / L * 100)`). Using `int()` (floor) here
    instead silently selected the wrong layer for any non-round percentage (e.g.
    on a 24-layer model, 54% floored to 12 but was minted from layer 13), which
    desynced the activations the AO trained on from the ones eval hooked. `round`
    round-trips exactly for every model with <100 layers (consecutive layers have
    distinct rounded percents) and is a no-op for clean percents like 25/50/75.
    """
    max_layers = get_layer_count(model_name)
    return round(max_layers * (layer_percent / 100))
