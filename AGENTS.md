# AO — agent guide

Keep this file short (it loads every session; aim <2k tokens). **`config.yaml` is the
source of truth** for all model/data/training/eval knobs — its comments are gears-level,
so read them before changing anything. This file is about *how to work*, not *what every
knob does*.

## What this project is
Activation Oracle (AO): a LoRA fine-tuned to interpret a target model's **residual-stream
activations** in natural language. Current target: **Qwen/Qwen3.5-4B**. Eval suite:
AObench (`activation_oracles/third_party/cot-oracle`). Judge: a local vLLM
Qwen3.6-35B-A3B-FP8 on :8001.

## Pipeline & commands (the Makefile sequences everything — don't reinvent it)
- Order: `make smoke` → `make data` → `make train-eval EXP=<name>` → `make eval EXP=<name>`
- `make data` = corpus (vLLM) → convqa (judge) → evalsets. **Run `make judge-down` FIRST**:
  the corpus stage wants the whole GPU, and a resident judge OOMs vLLM at engine init.
- **Resume gotcha:** corpus + convqa resume from `.progress` sidecars keyed ONLY on problem
  count, NOT generation params. If you change the token cap / temperature / model, delete
  `artifacts/<model>/{corpus,convqa}` (data + `.progress`) first, or it silently resumes and
  mixes old rollouts with new settings.
- Envs (uv venvs): `envs/train` = orchestrator, `envs/vllm` = corpus + judge,
  `envs/unsloth` = SFT (`training.use_unsloth`).
- Long jobs run as `nohup … & tail -f logs/<x>.log`; services live under supervisor.

## Hard rules
- **Read before you claim.** Never speculate about code or results you haven't opened.
  If a file or metric is referenced, open it first.
- **/workspace is EPHEMERAL** (not a host volume): checkpoints/artifacts do NOT survive a
  recycle/destroy. Push anything irreplaceable to HF (`hf.push` + `hf.namespace`) or sync
  off-box. This repo is also not under git — see "Portability" below.
- Prefer scoped reads; delegate broad codebase exploration to an `explore` subagent so the
  main context stays lean (cheaper, and keeps the working set focused).

## Research discipline (this is a safety/interp project — rigor beats speed)
- **Validate on a hard-to-fake number, not interpretability vibes.** "Steering raised
  behavior X by Y" is stronger evidence than "this latent looks like Z." If a claim can't
  be tied to a downstream metric, say so explicitly.
- **Method minimalism + time-box.** Try the cheap thing first (linear probe / steering /
  reading CoT) before fancy methods; if exploration stalls, stop and report rather than
  rabbit-holing.
- **Falsify first.** When presenting a hypothesis or result, name the most likely confound
  and try to disprove it before endorsing it. Treat judge-scored deltas as *weaker*
  evidence than AUC/rate metrics — LLM judges have positional and version-sensitivity bias.
- **Keep the ledger honest.** Record every real run in `docs/RESULTS.md` (recipe →
  pre-committed success metric → outcome, **including nulls**). Append only: never rewrite
  past entries or retroactively redefine what "success" was.

## Portability (how to carry this to a new Vast box)
`/workspace` is wiped on recycle, so the work lives in two GitHub repos:
- **`zach-perlman/ao-harness`** — THIS dir (config.yaml, ao_cli, Makefile, AGENTS.md,
  docs/, scripts/, patches/). Remote `origin`.
- **`zach-perlman/activation_oracles`** — code-only snapshot of the fork (remote `zach`;
  `origin` still points at upstream `japhba`). Large datasets + the cot-oracle submodule
  contents are NOT included; the cot-oracle functional edits are saved here as
  `patches/cot-oracle-local.patch`.

On a fresh box: clone both repos, rebuild envs (`scripts/setup_envs.sh`), re-pull the
cot-oracle submodule (`ceselder/cot-oracle@bec6f8a`) and `git apply` the patch, then
regenerate data. Commit with `git -c user.name=… -c user.email=…` (repo identity is not
set globally). User Rules are account-level and sync via Cursor login.

NOTE: the fork repo is a squashed snapshot (the local clone was shallow, so upstream
history couldn't be pushed). Full history remains at `japhba/activation_oracles`.
