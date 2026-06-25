# Activation Oracles — Phase 1: better AOs for Qwen3.5-4B

Replication of **"Building Better Activation Oracles"** (training + AOBench
evaluation) with **Qwen3.5-4B** as the target/AO base model, on a single
H200-140GB, using a **local Qwen3.5-35B-A3B judge** instead of Anthropic APIs.

An Activation Oracle (AO) is a LoRA on the target model trained to answer
natural-language questions about the target model's own residual-stream
activations, which are injected (norm-matched) into the AO's prompt prefix.

## Layout

```
config.yaml                 ALL knobs: models, layers, data sizes, training
                            hyperparameters, judge, eval profile, HF push
Makefile                    one target per pipeline stage (see below)
ao_cli/                     thin orchestration around the vendored code
activation_oracles/         vendored github.com/japhba/activation_oracles
  └─ third_party/cot-oracle/AObench     the AOBench eval suite
envs/{train,vllm,legacy}    uv virtualenvs (see scripts/setup_envs.sh)
artifacts/<model-slug>/     corpus, convqa, train configs, checkpoints, evals
```

Local modifications to the vendored repos (Qwen3.5 architecture support,
local-judge backends, local dataset paths) live on a `qwen35-local` branch in
each repo, with the original pinned commit tagged `upstream-pin`. Inspect the
patch set with `git -C activation_oracles diff upstream-pin` (and likewise in
`third_party/cot-oracle`).

## Commands

| Command | What it does | GPU |
|---|---|---|
| `make setup` | build `envs/train` + `envs/vllm` with uv | – |
| `make setup-legacy` | also build `envs/legacy` (upstream lock, 8B baseline) | – |
| `make smoke` | tiny end-to-end run on Qwen3.5-0.8B (data→train→eval) | ~30 min |
| `make judge-up` / `judge-down` / `judge-status` | local vLLM judge (Qwen3.5-35B-A3B-FP8 MoE, OpenAI-compatible on `localhost:8001/v1`, supervisor-managed) | resident |
| `make corpus` | on-policy CoT rollouts from the target model (vLLM) | hours |
| `make convqa` | chunked conversational-QA dataset via the local judge (the paper's main data improvement) | hours |
| `make evalsets` | regenerate AObench eval datasets for the target model | hours |
| `make data` | corpus → convqa → evalsets in order | hours |
| `make gen` | render `config.yaml` → training JSON + materialize datasets | min |
| `make train` | v2-recipe LoRA SFT (5-layer, rsLoRA r=128, 10M-token budget) | ~1 day |
| `make eval` | AObench `paper_six` profile on the trained AO | hours |
| `make eval-baseline` | AObench on the paper's released Qwen3-8B AO | hours |
| `make report RESULTS=<dir>` | plots + summary from an eval results dir | – |

Every stage is restartable and caches into `artifacts/` — rerunning a `make`
target skips or reuses what already exists where the underlying code supports
it (dataset `.pt` files are content-hashed by the vendored repo).

## Order of operations

```
make setup
make smoke                      # verify the whole pipeline on Qwen3.5-0.8B
make data                       # ~? h GPU: rollouts + judge-generated QA
make train                      # renders config, materializes data, trains
make eval                       # AObench on artifacts/Qwen3.5-4B/checkpoints/.../final
make report RESULTS=artifacts/Qwen3.5-4B/aobench_results/main
```

## Persistence warning

`/workspace` on this instance is **not** a persistent volume — a recycle or
destroy wipes everything. Set `hf.push: true` and `hf.namespace: <you>` in
`config.yaml` and put a write-scoped token in `/workspace/.env`
(`HF_TOKEN=hf_...`) so trained LoRAs are pushed to the Hub. Generated datasets
live under `artifacts/` — push anything irreplaceable before stopping work.

## Qwen3.5 notes

Qwen3.5 differs from Qwen3 in ways this project explicitly handles:

- **transformers ≥ 5 required** (hence the separate `envs/train`; the upstream
  repo pins `<5` for Unsloth, so Qwen3.5 training runs the plain PEFT path).
- **Multimodal wrapper**: checkpoints are `Qwen3_5ForConditionalGeneration`
  with a nested `text_config`; decoder layers live at a different module path
  than Qwen3's `model.model.layers`. Handled in the patched
  `get_hf_submodule` / `load_model` (both in `nl_probes` and the vendored
  AObench copy).
- **Hybrid attention**: 3 of every 4 layers are linear attention (Gated
  DeltaNet); activations are still read/injected at decoder-layer boundaries
  (residual stream), which is architecture-agnostic. `flash-linear-attention`
  + `causal-conv1d` kernels are installed for speed.
- **LoRA targets**: restricted to the text decoder (the vision tower is
  excluded from `all-linear`).
