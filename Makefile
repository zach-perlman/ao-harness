# =============================================================================
# Activation Oracle pipeline — every stage is one target. All knobs live in
# config.yaml; this file only sequences commands. See README.md for the tour.
#
# Typical first session:
#   make setup            # build python envs (~10 min)
#   make smoke            # tiny end-to-end run on Qwen3.5-0.8B (sanity check)
#   make data             # on-policy corpus + convqa + AObench eval datasets
#   make train            # v2-recipe LoRA on Qwen3.5-4B
#   make eval             # AObench on the trained AO
# =============================================================================

# The orchestrator runs in envs/train (has openai/pandas/pyyaml/hf_hub). It only
# wires commands together; the heavy work runs in the env each command selects.
PY := /workspace/ao/envs/train/bin/python

# EXP names an ISOLATED experiment run (a contribution from config.yaml). It is
# exported as AO_EXP so train/eval namespace their checkpoint + results dirs by
# it and the dashboard auto-compares against baseline_replication. Empty = the
# default `main` run. Usage: make train EXP=c3_logitlens && make eval EXP=c3_logitlens
EXP ?=
CLI := cd /workspace/ao && AO_EXP=$(EXP) $(PY) -m ao_cli

.PHONY: setup setup-legacy judge-up judge-down judge-status \
        corpus convqa convqa-filter convqa-solvability evalsets data train train-eval gen eval eval-baseline \
        report smoke dashboard sweep layer-probe diffing contrib rl calibration setup-unsloth

# ---- environment ------------------------------------------------------------
setup:                ## build envs/train + envs/vllm (uv)
	bash scripts/setup_envs.sh

setup-legacy:         ## additionally build envs/legacy (upstream lock, for 8B baseline)
	bash scripts/setup_envs.sh --legacy

setup-unsloth:        ## additionally build envs/unsloth (Unsloth on transformers>=5; needed for training.use_unsloth on Qwen3.5)
	bash scripts/setup_envs.sh --unsloth

# ---- local judge (vLLM, OpenAI-compatible, localhost:8001) -------------------
judge-up:             ## start the Qwen3.5-35B-A3B judge (supervisor service)
	$(CLI) judge up
judge-down:           ## stop the judge, freeing its GPU memory
	$(CLI) judge down
judge-status:         ## supervisor state + /v1/models probe
	$(CLI) judge status

# ---- data generation (GPU; judge needed for convqa + some evalsets) ----------
corpus:               ## on-policy CoT rollouts from the target model
	$(CLI) corpus

convqa:               ## chunked conversational QA via the local judge
	$(CLI) judge up && $(CLI) convqa

convqa-filter:        ## rebuild convqa train/test.parquet from rows.jsonl w/ quality filters (no judge)
	$(CLI) convqa --filter

convqa-solvability:   ## C4: keep only convqa pairs an early AO answers better WITH the activation
	$(CLI) convqa --solvability-filter

evalsets:             ## regenerate + install AObench datasets (manages judge per-task)
	$(CLI) evalsets

data: corpus          ## corpus -> convqa (judge) -> evalsets, in order
	$(CLI) judge up
	$(CLI) convqa
	$(CLI) evalsets
	$(CLI) judge down

# ---- training -----------------------------------------------------------------
gen:                  ## render config + materialize training datasets only
	$(CLI) train --gen-only

train:                ## render + materialize + LoRA SFT (single GPU torchrun)
	$(CLI) train

train-eval:           ## train, then auto-run AObench eval IF training succeeds (EXP propagates)
	$(MAKE) train EXP=$(EXP)
	$(MAKE) eval EXP=$(EXP)

diffing:              ## C5/C7/C11: build LoRA variant families (FAMILIES=style,fact,secret; default style)
	$(CLI) diffing $(if $(FAMILIES),--families $(FAMILIES),)

contrib:              ## enable exactly ONE SFT contribution (NAME=<key>), disabling the rest; NAME=none -> baseline
	$(PY) scripts/set_contrib.py $(NAME)

rl:                   ## C1 (DPO) / C2 (GRPO) post-training from an SFT checkpoint (set EXP + a contributions.* RL toggle)
	$(CLI) rl

# ---- evaluation ----------------------------------------------------------------
eval:                 ## AObench (config.yaml eval profile) on the trained AO
	$(CLI) evaluate

eval-baseline:        ## (re)evaluate the reference baseline (config.yaml baseline.ao_lora) into eval.baseline_run
	$(CLI) evaluate --baseline

report:               ## plots + summary tables; usage: make report RESULTS=<dir>
	cd activation_oracles/third_party/cot-oracle && \
	  /workspace/ao/envs/train/bin/python -m AObench.utils.report $(RESULTS)

dashboard:            ## build the ours-vs-paper comparison dashboard (auto-run by `make eval`)
	$(CLI) dashboard

sweep:                ## eval-time sweep of steering strength + n_positions (waits for a running eval)
	$(CLI) sweep

layer-probe:          ## Stage-0 layer scan: per-layer linear-probe AUC vs depth (no retrain; picks depths to sweep)
	$(CLI) layer_probe

calibration:          ## measure bootstrap-mode-frequency UQ + ECE for this AO (set EXP=<run>)
	$(CLI) calibration

# ---- end-to-end sanity check (Qwen3.5-0.8B, ~30 min) --------------------------
smoke:
	$(CLI) corpus --smoke
	$(CLI) judge up
	$(CLI) convqa --smoke
	$(CLI) judge down
	$(CLI) evalsets --smoke
	$(CLI) train --smoke
	$(CLI) evaluate --smoke
	$(CLI) judge down
