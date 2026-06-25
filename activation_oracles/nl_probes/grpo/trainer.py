"""Generic online GRPO loop for the injected AO (Phase 2).

The loop is contribution-agnostic: a caller supplies (a) a `scenario_sampler` that
yields scenarios — each a dict with `datapoints` (≥1 AO TrainingDataPoints sharing
a reward context) — and (b) a `reward_fn` that scores the k sampled rollouts of
each datapoint. The trainer turns those rewards into within-group advantages and
runs clipped GRPO with activation injection.

PER STEP
--------
  1. sample `batch_scenarios` scenarios → flat list of prompt datapoints.
  2. materialize prompts (strip target, fill steering vectors) once.
  3. sample k rollouts per prompt with injection.
  4. reward_fn(scenario, rollout_texts_per_datapoint) → reward per (datapoint, rollout).
  5. center rewards within each datapoint's k-group → advantages → GRPOItems.
  6. cache old log-probs, then one clipped-surrogate backward + optimizer step.
Saves the LoRA to <save_dir>/final at the end (and every save_every steps).
"""

from __future__ import annotations

import time
from pathlib import Path

import torch
from torch.optim.lr_scheduler import LinearLR

from nl_probes.grpo.injection_grpo import (
    GRPOItem,
    compute_grpo_loss,
    compute_old_logprobs,
    compute_ref_logprobs,
    generate_grouped_rollouts,
    group_advantages,
    materialize_prompts,
)


def train_grpo(
    *,
    model,
    tokenizer,
    submodule,
    steering_coefficient: float,
    scenario_sampler,        # callable(n) -> list[scenario dict] (each has "datapoints": list[TrainingDataPoint])
    reward_fn,               # callable(scenario, list[list[str]]) -> list[list[float]] (per datapoint, per rollout)
    save_dir: str,
    device,
    dtype=torch.bfloat16,
    k: int = 8,
    lr: float = 1e-6,
    max_steps: int = 2000,
    warmup_steps: int = 20,
    clip_eps: float = 0.2,
    max_grad_norm: float = 1.0,
    batch_scenarios: int = 4,
    normalize_advantages: bool = False,   # Dr. GRPO: std-norm causes a difficulty bias → off by default
    kl_coef: float = 0.0,                 # KL-to-SFT-reference anchor (0 = off)
    inner_epochs: int = 1,                # μ: GRPO updates reusing each rollout batch (clipping only bites if >1)
    dynamic_sampling: bool = True,        # DAPO: drop zero-variance groups (no learning signal)
    oversample: float = 1.5,              # sample this × batch_scenarios so dynamic filtering still fills the batch
    advantage_postprocess=None,           # optional (scenario, rewards_k, advs)->advs hook, applied per group
                                          #   after centering (e.g. TIAR abstention reweighting); free (no fwd pass)
    generation_kwargs: dict | None = None,
    save_every: int = 200,
    log_every: int = 10,
    eval_fn=None,                         # optional callable(model) -> float (higher better) for best-ckpt selection
    eval_every: int = 200,
) -> str:
    """On-policy GRPO with the hardened defaults (KL anchor, no std-norm, dynamic
    sampling, constant-length loss normalization). See injection_grpo for the loss.

    PER STEP: sample (oversampled) scenarios → k rollouts each → per-sample rewards
    → drop groups whose rewards have zero variance → group-center to advantages →
    (cache reference logprobs for KL; cache old logprobs only if μ>1) → μ clipped
    updates. Periodically evaluates `eval_fn` and keeps the best checkpoint."""
    generation_kwargs = generation_kwargs or {"do_sample": True, "temperature": 0.9,
                                              "top_p": 0.95, "max_new_tokens": 48}
    length_norm = float(generation_kwargs.get("max_new_tokens", 48))
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.0)
    scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=max(warmup_steps, 1))

    print(f"[grpo] {max_steps} steps · k={k} · {batch_scenarios} scenarios/step · lr={lr} · "
          f"kl={kl_coef} · μ={inner_epochs} · dyn_sample={dynamic_sampling}")
    best_score = float("-inf")
    t0 = time.time()
    # Step-0 baseline: score the starting (post-warm-start) policy BEFORE any GRPO
    # update. This is the control that makes "did training help?" answerable from
    # the run alone — every later [eval] is read relative to this number, and `best`
    # is only re-saved when a step genuinely beats the untrained policy.
    if eval_fn is not None:
        best_score = eval_fn(model)
        model.save_pretrained(str(Path(save_dir) / "best"))
        print(f"  [eval] step 0 baseline (post-warm-start)  score={best_score:+.3f}", flush=True)
    # Per-phase wall-clock timing. GPU ops are async, so we only synchronize (and
    # thus get meaningful per-phase numbers) on steps we actually log — keeping the
    # sync overhead off the hot path. `_clk()` returns a synced timestamp on log
    # steps and a raw one otherwise (unused then).
    for step in range(1, max_steps + 1):
        do_log = (step % log_every == 0 or step == 1)

        def _clk() -> float:
            if do_log and torch.cuda.is_available():
                torch.cuda.synchronize()
            return time.time()

        n_sample = max(batch_scenarios, int(batch_scenarios * oversample)) if dynamic_sampling else batch_scenarios
        scenarios = scenario_sampler(n_sample)
        if not scenarios:
            continue

        # Flatten datapoints, remembering which scenario/slot each came from.
        dps, owner = [], []  # owner[i] = (scenario_index, slot_within_scenario)
        for si, sc in enumerate(scenarios):
            for slot, dp in enumerate(sc["datapoints"]):
                dps.append(dp)
                owner.append((si, slot))

        t_gen0 = _clk()
        prompt_dps = materialize_prompts(dps, tokenizer, model, device)
        groups_ids = generate_grouped_rollouts(
            model, tokenizer, prompt_dps, submodule, steering_coefficient, k, generation_kwargs, device, dtype)
        groups_txt = [[tokenizer.decode(ids, skip_special_tokens=True) for ids in g] for g in groups_ids]
        t_gen = _clk() - t_gen0

        # Regroup rollouts by scenario so reward_fn sees all of a scenario's datapoints together.
        per_scenario_txt: dict[int, list[list[str]]] = {}
        per_scenario_idx: dict[int, list[int]] = {}
        for gi, (si, slot) in enumerate(owner):
            per_scenario_txt.setdefault(si, []).append(groups_txt[gi])
            per_scenario_idx.setdefault(si, []).append(gi)

        grpo_items: list[GRPOItem] = []
        all_rewards: list[float] = []
        kept_groups = 0
        for si, sc in enumerate(scenarios):
            rewards_per_dp = reward_fn(sc, per_scenario_txt[si])  # [n_dp][k]
            for local, gi in enumerate(per_scenario_idx[si]):
                rewards_k = rewards_per_dp[local]
                all_rewards.extend(rewards_k)
                # Dynamic sampling: a group whose rewards are all equal yields zero
                # advantage and zero gradient — skip it so the batch carries signal.
                if dynamic_sampling and (max(rewards_k) - min(rewards_k) < 1e-9):
                    continue
                kept_groups += 1
                advs = group_advantages(rewards_k, normalize=normalize_advantages)
                if advantage_postprocess is not None:
                    advs = advantage_postprocess(sc, rewards_k, advs)
                for j, adv in enumerate(advs):
                    grpo_items.append(GRPOItem(
                        prompt_ids=prompt_dps[gi].input_ids,
                        response_ids=groups_ids[gi][j],
                        steering_vectors=prompt_dps[gi].steering_vectors,
                        positions=prompt_dps[gi].positions,
                        advantage=adv))
        if not grpo_items:
            continue

        t_ref0 = _clk()
        if kl_coef > 0.0:
            compute_ref_logprobs(model, grpo_items, submodule, steering_coefficient, device, dtype, pad_id)
        if inner_epochs > 1:
            compute_old_logprobs(model, grpo_items, submodule, steering_coefficient, device, dtype, pad_id)
        t_ref = _clk() - t_ref0

        t_loss0 = _clk()
        model.train()
        for _ in range(inner_epochs):
            optimizer.zero_grad()
            loss, metrics = compute_grpo_loss(
                model, grpo_items, submodule, steering_coefficient, device, dtype, pad_id,
                clip_eps=clip_eps, kl_coef=kl_coef, length_norm=length_norm)
            torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm)
            optimizer.step()
        scheduler.step()
        t_loss = _clk() - t_loss0

        if do_log:
            mean_r = sum(all_rewards) / max(len(all_rewards), 1)
            print(f"  step {step}/{max_steps}  loss={loss:.4f}  mean_reward={mean_r:+.3f}  "
                  f"kl={metrics['grpo/mean_kl']:.4f}  clip={metrics['grpo/clip_frac']:.2f}  "
                  f"groups={kept_groups}  n={int(metrics['grpo/n_sequences'])}  "
                  f"{(time.time() - t0) / step:.1f}s/step  "
                  f"[gen={t_gen:.1f} ref={t_ref:.1f} loss={t_loss:.1f}]", flush=True)

        if eval_fn is not None and step % eval_every == 0:
            score = eval_fn(model)
            tag = ""
            if score > best_score:
                best_score = score
                model.save_pretrained(str(Path(save_dir) / "best"))
                tag = "  ← new best (saved)"
            print(f"  [eval] step {step}  score={score:+.3f}{tag}", flush=True)
        if step % save_every == 0:
            model.save_pretrained(str(Path(save_dir) / f"step_{step}"))

    final = str(Path(save_dir) / "final")
    model.save_pretrained(final)
    tokenizer.save_pretrained(final)
    print(f"[grpo] done in {time.time() - t0:.0f}s → {final}"
          + (f" (best eval={best_score:+.3f} at {save_dir}/best)" if eval_fn is not None else ""))
    return final
