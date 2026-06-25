"""Render config.yaml -> a training-config JSON for nl_probes/sft.py.

Gears-level overview:
  The vendored repo drives training from a large JSON (model, dataset mixture,
  LoRA/optimizer hyperparameters). Rather than re-deriving that schema, we load
  the paper's published v2 5-layer recipe (training_configs/v2_ablations/
  v2_layer_count_5layer.json) as a TEMPLATE and rewrite every model- or
  data-dependent field from config.yaml:

    1. model_name + layer combinations  -> resolved for the target model's depth
    2. convqa dataset                   -> our locally generated parquet
    3. past/future-lens corpus          -> our locally generated on-policy jsonl
    4. classification entries           -> kept (model-agnostic CSVs), retargeted
    5. hyperparameters / budgets / dirs -> from config.yaml
    6. use_unsloth                      -> from config (Qwen3-8B: true, envs/legacy)

  Output: artifacts/<slug>/train_config[_smoke].json, consumed by `make train`.
"""

from __future__ import annotations

import argparse
import copy
import json

from . import REPO, artifacts_dir, experiment_name, load_config, model_slug, resolve_layers, run_dir_name

TEMPLATE = REPO / "training_configs" / "v2_ablations" / "v2_layer_count_5layer.json"


def _contribution_entries(cfg: dict, model: str, percents: list[int], dataset_folder: str,
                          art, corpus_source: str, corpus_key: str) -> list[dict]:
    """Build dataset_configs entries for every ENABLED SFT contribution (C3, C5).

    Mechanism: each contribution is just another mixture component — one dict in
    the same schema as the template's convqa/past_lens/classification entries,
    dispatched by `dataset_name` to its loader (registered in
    act_dataset_manager). We only emit an entry when its `contributions.<x>.enabled`
    flag is set, so a run includes exactly the contributions you toggled on.
    Returns [] when none are enabled (-> reproduces the baseline mixture).
    """
    contrib = cfg.get("contributions", {}) or {}
    entries: list[dict] = []

    def _base(num_train: int, dataset_name: str, params: dict, save_acts: bool = False,
              seed: int = 42) -> dict:
        return {
            "custom_dataset_params": params,
            "num_train": num_train,
            "num_test": 0,
            "splits": ["train"],
            "model_name": model,
            "layer_combinations": [percents],
            "save_acts": save_acts,
            "batch_size": cfg["training"]["batch_size"],
            "dataset_name": dataset_name,
            "dataset_folder": dataset_folder,
            "seed": seed,
        }

    # C3 — logit-lens prediction over the SAME corpus past_lens reads from
    # (released HF repo when data.use_released_datasets, else the local on-policy
    # corpus.jsonl). Resolved by the caller so the source never drifts from the
    # base recipe's corpus.
    ll = contrib.get("logit_lens", {}) or {}
    if ll.get("enabled"):
        entries.append(_base(
            num_train=int(ll.get("n", 30000)),
            dataset_name="logit_lens",
            params={
                "pretrain_dataset": corpus_source,
                "pretrain_key": corpus_key,
                "top_k": int(ll.get("top_k", 5)),
                "task_format": ll.get("task_format", "free_form"),
                "max_length": 2000,
                # lens: "logit" (raw) | "tuned" (per-layer affine, fit once + cached).
                "lens": ll.get("lens", "logit"),
                "tuned_lens_dir": str(art / "tuned_lens"),
                "tuned_lens_tokens": int(ll.get("tuned_lens_tokens", 500_000)),
                "tuned_lens_steps": int(ll.get("tuned_lens_steps", 400)),
                "tuned_lens_lr": float(ll.get("tuned_lens_lr", 1.0e-3)),
            },
        ))

    # C5 — contrastive model-diffing pairs (vectors precomputed -> save_acts=True).
    md = contrib.get("model_diffing", {}) or {}
    if md.get("enabled"):
        entries.append(_base(
            num_train=int(md.get("n", 30000)),
            dataset_name="model_diffing",
            save_acts=True,
            params={
                "variants_dir": str(art / "diffing_variants"),
                "pretrain_dataset": corpus_source,
                "pretrain_key": corpus_key,
                "inject_mode": md.get("inject_mode", "both"),
                "n_prompts": int(md.get("n_prompts", 2000)),
                "max_length": 2000,
            },
        ))

    # The synthetic corpus-activation tasks below all build their injection vectors
    # by reading the target model on the SAME corpus past_lens/logit-lens use, then
    # transforming them (combine / corrupt / group). Vectors are precomputed, so
    # save_acts=True and they share corpus_source/corpus_key with C3/C5.
    def _corpus_params(extra: dict) -> dict:
        return {"pretrain_dataset": corpus_source, "pretrain_key": corpus_key,
                "max_length": 2000, **extra}

    # C9 — activation arithmetic (inject a1±a2, name the two source tokens).
    aa = contrib.get("activation_arithmetic", {}) or {}
    if aa.get("enabled"):
        entries.append(_base(
            num_train=int(aa.get("n", 20000)),
            dataset_name="activation_arithmetic",
            save_acts=True,
            params=_corpus_params({"mode": aa.get("mode", "both")}),
        ))

    # C12 — odd-one-out (inject k activations, identify the intruder slot).
    ooo = contrib.get("odd_one_out", {}) or {}
    if ooo.get("enabled"):
        entries.append(_base(
            num_train=int(ooo.get("n", 20000)),
            dataset_name="odd_one_out",
            save_acts=True,
            params=_corpus_params({"k_activations": int(ooo.get("k_activations", 4))}),
        ))

    # C13 — denoising robustness (inject a noised activation, recover clean token).
    dn = contrib.get("denoising", {}) or {}
    if dn.get("enabled"):
        entries.append(_base(
            num_train=int(dn.get("n", 20000)),
            dataset_name="denoising",
            save_acts=True,
            params=_corpus_params({"noise_scale": float(dn.get("noise_scale", 0.3))}),
        ))

    # C8 — injection-strength & depth curriculum (corpus acts at varied layer + scale).
    ic = contrib.get("injection_curriculum", {}) or {}
    if ic.get("enabled"):
        entries.append(_base(
            num_train=int(ic.get("n", 20000)),
            dataset_name="injection_curriculum",
            save_acts=True,
            params=_corpus_params({
                "percent_choices": ic.get("percent_choices", [25, 40, 55, 70, 85]),
                "scale_min": float(ic.get("scale_min", 0.5)),
                "scale_max": float(ic.get("scale_max", 2.0)),
            }),
        ))

    # C10 — graded-intensity probing (reuses the C5 style-variant directions).
    gi = contrib.get("graded_intensity", {}) or {}
    if gi.get("enabled"):
        entries.append(_base(
            num_train=int(gi.get("n", 20000)),
            dataset_name="graded_intensity",
            save_acts=True,
            params={
                "variants_dir": str(art / "diffing_variants"),
                "pretrain_dataset": corpus_source, "pretrain_key": corpus_key, "max_length": 2000,
                "n_prompts": int(gi.get("n_prompts", 400)),
                "alphas": gi.get("alphas", [0.0, 0.5, 1.0, 2.0]),
                "labels": gi.get("labels", ["none", "weak", "moderate", "strong"]),
            },
        ))

    # C7 / C11 — latent recovery (fact / secret). Both use ONE dataset class with a
    # `task` switch; each reads its own judge-built variant family from
    # diffing_variants/<family> (run `make diffing FAMILIES=fact,secret`).
    def _latent_entry(block: dict, task: str, family: str) -> None:
        entries.append(_base(
            num_train=int(block.get("n", 16000)),
            dataset_name="latent_recovery",
            save_acts=True,
            params={
                "variants_dir": str(art / "diffing_variants" / family),
                "task": task,
                "pretrain_dataset": corpus_source, "pretrain_key": corpus_key, "max_length": 2000,
                "n_prompts": int(block.get("n_prompts", 2000)),
                # On-policy harvesting (read the latent off the variant's own rollout).
                "on_policy": bool(block.get("on_policy", True)),
                "gen_max_new_tokens": int(block.get("gen_max_new_tokens", 64)),
                "gen_temperature": float(block.get("gen_temperature", 0.8)),
            },
        ))

    kr = contrib.get("knowledge_recovery", {}) or {}
    if kr.get("enabled"):
        _latent_entry(kr, task="fact", family="fact")

    se = contrib.get("secret_elicitation", {}) or {}
    if se.get("enabled"):
        _latent_entry(se, task="secret", family="secret")

    return entries


def render(smoke: bool) -> None:
    cfg = load_config()
    model = cfg["model"]["smoke_name"] if smoke else cfg["model"]["name"]
    slug = model_slug(model)
    art = artifacts_dir(model)

    layers, percents = resolve_layers(model, cfg["layers"]["center_percent"], cfg["layers"]["n_layers"])
    print(f"[render] {model}: act layers {layers} (= {percents}% of depth)")

    convqa_n = cfg["smoke"]["mix_convqa_n"] if smoke else cfg["data"]["mix"]["convqa_n"]
    lens_n = cfg["smoke"]["mix_past_lens_n"] if smoke else cfg["data"]["mix"]["past_lens_n"]
    max_target_tokens = cfg["smoke"]["max_target_tokens"] if smoke else cfg["training"]["max_target_tokens"]
    cls_n = 100 if smoke else None  # None = keep template's per-dataset counts

    # When set, keep the template's RELEASED HF dataset repos (the paper's
    # Qwen3-8B convqa + lens corpus) instead of rewriting them to our locally
    # generated files — i.e. train from existing data, no generation. The smoke
    # run always uses its tiny local data. See config.yaml data.use_released_datasets.
    use_released = (not smoke) and bool(cfg["data"].get("use_released_datasets", False))
    print(f"[render] dataset source: {'released HF datasets' if use_released else 'local artifacts'}")

    with open(TEMPLATE) as f:
        tc = json.load(f)

    # EXP (if set) namespaces the checkpoint + run so each contribution is an
    # isolated, baseline-comparable run; empty EXP keeps the default ao_<slug>_v2.
    run_name = run_dir_name(model, smoke)
    exp = "" if smoke else experiment_name()
    if exp:
        print(f"[render] experiment: {exp} -> checkpoints/{exp}")
    dataset_folder = str(art / "sft_training_data")

    # --- top-level model / layer / hyperparameter fields ---
    tc["model_name"] = model
    tc["layer_combinations"] = [percents]
    tc["act_layer_combinations"] = [layers]
    tc["hook_onto_layer"] = cfg["injection"]["hook_onto_layer"]
    tc["steering_coefficient"] = cfg["injection"]["steering_coefficient"]
    tc["lr"] = cfg["training"]["lr"]
    tc["lora_r"] = cfg["training"]["lora_r"]
    tc["lora_alpha"] = cfg["training"]["lora_alpha"]
    tc["use_rslora"] = cfg["training"]["use_rslora"]
    tc["lora_dropout"] = cfg["training"]["lora_dropout"]
    tc["train_batch_size"] = cfg["training"]["batch_size"]
    # Micro-batches accumulated per optimizer step; effective batch = batch_size *
    # this. Lets us cap per-forward activation VRAM (the 12B OOMs un-checkpointed at
    # batch 16) while keeping the recipe's effective batch. Defaults to 1 if absent.
    tc["gradient_accumulation_steps"] = cfg["training"].get("gradient_accumulation_steps", 1)
    # Stability knobs (defaults preserve prior hardcoded behavior if absent):
    #   max_grad_norm — global grad-norm clip; lower = tighter (anti-divergence).
    #   warmup_ratio  — fraction of optimizer steps ramping LR 0->peak; longer = gentler.
    tc["max_grad_norm"] = cfg["training"].get("max_grad_norm", 1.0)
    tc["warmup_ratio"] = cfg["training"].get("warmup_ratio", 0.1)
    # Periodic checkpoint cadence (optimizer steps). Absent -> dataclass default
    # (effectively off, final-only save). Set so a late divergence is recoverable.
    if "save_steps" in cfg["training"]:
        tc["save_steps"] = cfg["training"]["save_steps"]
    tc["num_epochs"] = cfg["training"]["num_epochs"]
    tc["seed"] = cfg["training"]["seed"]
    tc["max_target_tokens"] = max_target_tokens
    tc["max_train_examples"] = cfg["training"]["max_train_examples"]
    # Unsloth (envs/unsloth, transformers>=5) gives fused kernels + the fused
    # cross-entropy that sidesteps the full-vocab fp32 logits OOM (~2-5x faster).
    # fp8 layers TorchAO FP8 LoRA on top (~1.3-1.4x more, big VRAM savings); it
    # only takes effect with use_unsloth. Smoke always runs plain bf16 in
    # envs/train. train.py routes the use_unsloth path to envs/unsloth.
    tc["use_unsloth"] = (not smoke) and bool(cfg["training"].get("use_unsloth", False))
    tc["fp8"] = (not smoke) and bool(cfg["training"].get("fp8", False))
    # Smoke is tiny so its throughput is irrelevant — keep checkpointing on there
    # for safety; the real run honors config.yaml (default off on the 140GB H200).
    tc["gradient_checkpointing"] = True if smoke else bool(cfg["training"]["gradient_checkpointing"])
    tc["dataset_folder"] = dataset_folder
    tc["save_dir"] = str(art / "checkpoints" / run_name)
    tc["wandb_project"] = cfg["training"]["wandb_project"]
    tc["wandb_run_name"] = run_name
    tc["wandb_suffix"] = ""
    tc["examples_per_source_epoch"] = None
    tc["hf_push_to_hub"] = bool(cfg["hf"]["push"]) and not smoke
    tc["hf_repo_id"] = f"{cfg['hf']['namespace']}/{run_name}" if tc["hf_push_to_hub"] else ""

    # In-training open-ended eval: a quick sanity probe run right after training,
    # distinct from the comprehensive AObench `make eval`. The repo's full suite
    # (mmlu_prediction, missing_info, ...) each load a per-model dataset built by a
    # separate data_pipelines/<task>/generate_dataset.py (model rollouts) that this
    # pipeline does not produce; only number_prediction is fully synthetic. So we
    # restrict the in-training eval to the self-contained task(s). (Left unset it
    # defaults to the entire suite and crashes on the first missing dataset.)
    tc["open_ended_eval_include"] = cfg["smoke"]["eval_tasks"] if smoke else ["number_prediction"]

    # --- dataset entries (template order: convqa, classification..., past_lens) ---
    convqa_train = art / "convqa" / "train.parquet"
    convqa_test = art / "convqa" / "test.parquet"
    corpus_jsonl = art / "corpus" / "corpus.jsonl"

    # Released convqa can come from MULTIPLE repos (DD "doubleconvqa"): we fan the
    # single template convqa entry out into one entry per repo, each pulling
    # convqa_n rows at a distinct seed so the sources don't draw identical samples.
    # Locally generated convqa is always a single parquet, so this only applies to
    # the released path.
    convqa_repos = cfg["data"]["mix"].get("convqa_repos") or [None]

    # C4 — empirical solvability filter. When enabled, convqa trains on the
    # filtered parquet produced by `make convqa-solvability` (pairs the AO can
    # answer better WITH the activation than without), regardless of the released
    # vs local source toggle. This collapses convqa to a single local entry.
    solv = (cfg.get("contributions", {}) or {}).get("solvability_filter", {}) or {}
    solvable_parquet = art / "convqa" / "train_solvable.parquet"
    use_solvable = (not smoke) and bool(solv.get("enabled"))
    if use_solvable and not solvable_parquet.exists():
        raise SystemExit(
            f"[render] solvability_filter enabled but {solvable_parquet} is missing — "
            f"run `make convqa-solvability` first"
        )

    new_entries = []
    for entry in tc["dataset_configs"]:
        entry = copy.deepcopy(entry)
        entry["model_name"] = model
        entry["layer_combinations"] = [percents]
        entry["dataset_folder"] = dataset_folder

        if entry["dataset_name"] == "cot_oracle_convqa":
            if use_solvable:
                entry["custom_dataset_params"]["hf_dataset_repo"] = str(solvable_parquet)
                entry["num_train"] = convqa_n
                new_entries.append(entry)
            elif not use_released:
                entry["custom_dataset_params"]["hf_dataset_repo"] = str(convqa_train)
                entry["num_train"] = convqa_n
                new_entries.append(entry)
            else:
                base_seed = entry.get("seed", 44)
                for i, repo in enumerate(convqa_repos):
                    dup = copy.deepcopy(entry)
                    if repo is not None:
                        dup["custom_dataset_params"]["hf_dataset_repo"] = repo
                    dup["num_train"] = convqa_n
                    dup["seed"] = base_seed + i
                    new_entries.append(dup)
            continue
        elif entry["dataset_name"] == "past_lens":
            if not use_released:
                entry["custom_dataset_params"]["pretrain_dataset"] = str(corpus_jsonl)
                entry["custom_dataset_params"]["pretrain_key"] = "cot_response"
            entry["num_train"] = lens_n
        elif entry["dataset_name"].startswith("classification_") and cls_n is not None:
            entry["num_train"] = min(entry["num_train"], cls_n)
            entry["num_test"] = min(entry["num_test"], 50)
        new_entries.append(entry)

    # Append one entry per ENABLED SFT contribution (C3 logit-lens, C5 diffing).
    # The mixture is concat+shuffle+budget-trim, so adding an entry with its
    # `n` pool size is exactly how the base recipe sets proportions.
    if not smoke:
        # C3/C5 must read the SAME corpus past_lens does — the released HF repo
        # under use_released, else the local corpus.jsonl. Pull it straight from
        # the (already-resolved) past_lens entry so the two never diverge.
        past = next((e for e in new_entries if e["dataset_name"] == "past_lens"), None)
        pl_params = (past or {}).get("custom_dataset_params", {})
        corpus_source = pl_params.get("pretrain_dataset", str(corpus_jsonl))
        corpus_key = pl_params.get("pretrain_key", "cot_response")
        contrib_entries = _contribution_entries(
            cfg, model, percents, dataset_folder, art, corpus_source, corpus_key)
        if contrib_entries:
            names = ", ".join(e["dataset_name"] for e in contrib_entries)
            print(f"[render] +contribution datasets: {names}")
            new_entries.extend(contrib_entries)

    tc["dataset_configs"] = new_entries

    # --- validation: held-out convqa rows, single center layer ---
    # The in-training MCQ probe loads the validation config's "train" split, so
    # its sample count is that entry's `num_train`. We override it from
    # config.yaml (eval.mcq_n) to de-noise the [mcq:*] signal; smoke keeps the
    # template's tiny default.
    mcq_n = None if smoke else cfg["eval"].get("mcq_n")
    for entry in tc["validation_dataset_configs"]:
        entry["model_name"] = model
        entry["layer_combinations"] = [[percents[len(percents) // 2]]]
        entry["dataset_folder"] = dataset_folder
        if mcq_n is not None:
            entry["num_train"] = mcq_n
        if entry["dataset_name"] == "cot_oracle_convqa":
            # Validate on the SAME convqa source we train on: the held-out test
            # split of the (first) released repo when using released data, else
            # our locally generated test parquet. Keeps the in-training mcq probe
            # in-distribution (and auto-relabels it, e.g. cot_sonnet_test).
            if use_released and convqa_repos[0] is not None:
                entry["custom_dataset_params"]["hf_dataset_repo"] = convqa_repos[0]
            elif not use_released:
                entry["custom_dataset_params"]["hf_dataset_repo"] = str(convqa_test)

    out = art / ("train_config_smoke.json" if smoke else "train_config.json")
    with open(out, "w") as f:
        json.dump(tc, f, indent=2)
    print(f"[render] wrote {out}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--smoke", action="store_true", help="render the tiny smoke-test config")
    args = p.parse_args(argv)
    render(args.smoke)


if __name__ == "__main__":
    main()
