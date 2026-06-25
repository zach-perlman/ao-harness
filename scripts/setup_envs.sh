#!/bin/bash
# =============================================================================
# Build the three uv virtualenvs under envs/.
#
# Why three: Qwen3.5 needs transformers>=5, but the upstream repo (and its
# released Qwen3-8B AOs + Unsloth) pins transformers<5; and vLLM ships its own
# torch. Keeping them separate avoids an unsolvable dependency knot:
#
#   envs/train   transformers>=5 + peft + torch cu128 (+ fla/causal_conv1d
#                kernels for Qwen3.5's linear-attention layers). Used for
#                training and AObench evals of Qwen3.5 AOs. No Unsloth.
#   envs/vllm    vLLM server + generation pipelines (corpus, eval datasets,
#                judge). Own torch pinned by vLLM.
#   envs/legacy  upstream uv.lock of activation_oracles (transformers<5,
#                Unsloth) for evaluating the released Qwen3-8B AOs. Built
#                only with `--legacy` (slow, ~10GB).
#
# Idempotent: each env is skipped if its python already imports the key deps.
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."
REPO=activation_oracles
TORCH_INDEX=https://download.pytorch.org/whl/cu128   # A100 + driver CUDA 12.8

mkdir -p envs

# ---- envs/train -------------------------------------------------------------
if envs/train/bin/python -c "import transformers, peft, torch" 2>/dev/null; then
    echo "[setup] envs/train OK, skipping"
else
    uv venv envs/train --python 3.12
    uv pip install -p envs/train/bin/python \
        torch --index-url $TORCH_INDEX
    uv pip install -p envs/train/bin/python \
        "transformers>=5" peft accelerate datasets bitsandbytes wandb \
        pandas pyarrow matplotlib plotly httpx openai anthropic einops jaxtyping \
        huggingface_hub pyyaml tabulate
    # AObench eval-suite deps (ROC-AUC scoring, fuzzy/number match, math
    # verification, judge clients). Needed by `make eval` AND the in-training
    # open_ended_eval probe that sft.py runs right after training.
    uv pip install -p envs/train/bin/python \
        scikit-learn rapidfuzz math_verify latex2sympy2_extended tenacity slist
    # Qwen3.5 linear-attention fast kernels. Without BOTH fla and causal-conv1d
    # transformers falls back to a (correct but slower) torch implementation.
    # causal-conv1d compiles CUDA and needs wheel/setuptools/ninja present.
    uv pip install -p envs/train/bin/python flash-linear-attention || \
        echo "[setup] WARNING: flash-linear-attention install failed (slow fallback)"
    uv pip install -p envs/train/bin/python wheel setuptools packaging ninja
    CAUSAL_CONV1D_FORCE_BUILD=TRUE uv pip install -p envs/train/bin/python causal-conv1d --no-build-isolation || \
        echo "[setup] WARNING: causal-conv1d build failed (slow fallback)"
    # Vendored repo importable as nl_probes/data_pipelines (no deps pulled).
    uv pip install -p envs/train/bin/python --no-deps -e $REPO
fi

# ---- envs/vllm --------------------------------------------------------------
if envs/vllm/bin/python -c "import vllm" 2>/dev/null; then
    echo "[setup] envs/vllm OK, skipping"
else
    uv venv envs/vllm --python 3.12
    uv pip install -p envs/vllm/bin/python vllm --torch-backend=cu128
    uv pip install -p envs/vllm/bin/python datasets pandas pyarrow anthropic aiohttp tqdm pyyaml
    uv pip install -p envs/vllm/bin/python --no-deps -e $REPO
fi

# ---- envs/legacy (opt-in) ---------------------------------------------------
# Unsloth training env (transformers<5): fused kernels speed up SFT and the fused
# cross-entropy avoids the full-vocab fp32 logits OOM, so this is the env used to
# train dense models like Qwen3-8B (config training.use_unsloth -> train.py routes
# here). Built PIECEWISE rather than `-e $REPO`, because the repo hard-depends on
# flash-attn, which can't build under uv's isolation (it needs torch already
# present + a CUDA toolchain). So: torch + build tools first, the Unsloth stack
# explicitly (versions mirror the repo's pyproject pins), flash-attn best-effort
# (Unsloth falls back to torch SDPA without it — same math, slower attention),
# then the repo --no-deps so it stays importable without re-resolving flash-attn.
if [[ "${1:-}" == "--legacy" ]]; then
    if envs/legacy/bin/python -c "import unsloth" 2>/dev/null; then
        echo "[setup] envs/legacy OK, skipping"
    else
        uv venv envs/legacy --python 3.11 --clear
        uv pip install -p envs/legacy/bin/python torch --index-url $TORCH_INDEX
        uv pip install -p envs/legacy/bin/python wheel setuptools packaging ninja
        uv pip install -p envs/legacy/bin/python \
            "transformers>=4.55,<5" "peft>=0.17,<0.19" accelerate bitsandbytes trl \
            "unsloth>=2026.5" datasets huggingface-hub pyarrow wandb tqdm pydantic \
            python-dotenv "numpy<2" pandas matplotlib httpx einops jaxtyping
        # AObench eval-suite deps: training runs in THIS env, so the in-training
        # open_ended_eval probe (sft.py, post-train) needs them, as does
        # `make eval-baseline`. numpy<2 pinned so sklearn/scipy don't bump it and
        # break the torch/Unsloth stack.
        uv pip install -p envs/legacy/bin/python \
            scikit-learn rapidfuzz math_verify latex2sympy2_extended tenacity slist \
            anthropic openai "numpy<2"
        uv pip install -p envs/legacy/bin/python flash-attn --no-build-isolation || \
            echo "[setup] WARNING: flash-attn unavailable (Unsloth will use torch SDPA)"
        uv pip install -p envs/legacy/bin/python --no-deps -e $REPO
    fi
fi

# ---- envs/unsloth (opt-in) --------------------------------------------------
# Unsloth on transformers>=5.2 — the ONLY Unsloth stack that loads Qwen3.5
# (envs/legacy's transformers<5 raises "Qwen3.5 needs transformers>=5.2.0").
# It mirrors envs/train's WORKING v5 + cu128 + fla kernels, then layers Unsloth
# + TorchAO on top: fused kernels + fused cross-entropy give ~2-5x SFT throughput
# and dodge the full-vocab fp32-logits OOM, and TorchAO enables optional FP8 LoRA
# (training.fp8 -> ~1.3-1.4x more + ~40-60% less base-model VRAM). train.py routes
# the use_unsloth path here. Build order matters: torch + transformers v5 + the
# Mamba kernels FIRST, the Unsloth stack LAST so it resolves against the v5 stack
# already present (the in-resolve "transformers>=5.2" pin blocks a silent <5
# downgrade). flash-attn best-effort (Unsloth falls back to torch SDPA — same
# math, slower attention). Built only with `--unsloth` (slow, ~12GB).
if [[ " $* " == *" --unsloth "* ]]; then
    if envs/unsloth/bin/python -c "import unsloth, torchao" 2>/dev/null; then
        echo "[setup] envs/unsloth OK, skipping"
    else
        uv venv envs/unsloth --python 3.12 --clear
        uv pip install -p envs/unsloth/bin/python torch --index-url $TORCH_INDEX
        uv pip install -p envs/unsloth/bin/python \
            "transformers>=5.2" peft accelerate datasets bitsandbytes wandb \
            pandas pyarrow matplotlib plotly httpx openai anthropic einops jaxtyping \
            huggingface_hub pyyaml tabulate
        uv pip install -p envs/unsloth/bin/python \
            scikit-learn rapidfuzz math_verify latex2sympy2_extended tenacity slist
        uv pip install -p envs/unsloth/bin/python wheel setuptools packaging ninja
        uv pip install -p envs/unsloth/bin/python flash-linear-attention || \
            echo "[setup] WARNING: flash-linear-attention install failed (slow fallback)"
        CAUSAL_CONV1D_FORCE_BUILD=TRUE uv pip install -p envs/unsloth/bin/python causal-conv1d --no-build-isolation || \
            echo "[setup] WARNING: causal-conv1d build failed (slow fallback)"
        uv pip install -p envs/unsloth/bin/python "transformers>=5.2" unsloth unsloth_zoo torchao
        uv pip install -p envs/unsloth/bin/python flash-attn --no-build-isolation || \
            echo "[setup] WARNING: flash-attn unavailable (Unsloth will use torch SDPA)"
        uv pip install -p envs/unsloth/bin/python --no-deps -e $REPO
    fi
fi

echo "[setup] done"
