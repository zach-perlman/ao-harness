# AO results ledger (append-only)

Honest, append-only record of every real training/eval run. Rules:
- **Append only.** Never edit or delete a past entry, and never retroactively change what
  "success" meant for a completed run. Corrections go in a new dated entry that references
  the old one.
- Log the **pre-committed success metric** before reading results, then the **outcome —
  including nulls / regressions**. Nulls are data; they save the next run.
- Keep judge-scored numbers labeled as such (weaker evidence than AUC/rate metrics).

Format per run:
```
## <date> — <model> / <EXP>
- Recipe: <key hyperparams / what changed vs prior>
- Committed metric: <the number that would mean "this worked", set BEFORE eval>
- Outcome: <result, incl. nulls>
- Decision: <what we did next & why>
```

---

## 2026-06 — google/gemma-4-12B-it / replication_v1
- Recipe: paper recipe — rsLoRA r=128 / alpha=16, lr 5e-5 (paper 3e-4 diverged on 12B),
  1 epoch, eff. batch 16. No best-ckpt selection yet (evaluated `final/`).
- Committed metric: beat near-chance on the AObench AUC tasks (mmlu_prediction,
  missing_info) + non-trivial number_prediction match rate.
- Outcome: **near-chance across the board.** Val loss U-turned (~min at 32k ex, ~1.10),
  then climbed while train loss fell → classic overfitting. `final/` was the most overfit
  point. Scorecard (final / step_2000): num_match .006/.008, mmlu_auc .533/.545,
  mi_auc .460/.517, act_sens .196/.379, abst_balacc .608/.518, backtracking/faithfulness 0.
- Decision: built best-checkpoint selection into sft.py; designed v2 anti-overfit recipe.

## 2026-06 — google/gemma-4-12B-it / replication_v2
- Recipe: r=64 / alpha=11 (holds rsLoRA eff. scale ~1.4), lora_dropout 0.05, lr 5e-5,
  best-ckpt tracking (eval loads `best/`). Interrupted ~step 12000; best = step 5500.
- Committed metric: beat v1 on the AUC tasks AND lift activation_sensitivity at the
  val-minimum checkpoint.
- Outcome: **did NOT beat v1.** v2-best (step 5500, val 1.065 — lower than v1's ~1.10, so
  the anti-overfit recipe worked on the LM objective) still scored at/below chance on 8/10
  tasks. Only real gain: mi_auc .460→.623. act_sens DROPPED (.196→.137); mmlu_auc fell
  below chance (.473). Lower held-out loss did NOT transfer to AObench competence.
- Decision: **abandon gemma-4-12B-it.** Suspected partly model capability, partly a
  degraded eval harness for the gemma4_unified arch. Switch target to Qwen3.5-4B (the
  codebase's native target; de-risks model + harness together, far cheaper on one H200).

## 2026-06 — Qwen/Qwen3.5-4B / replication_v1  [PENDING]
- Recipe: r=64 / alpha=11 / dropout 0.05 (carried from gemma-v2), lr 5e-5 (conservative
  carryover; can raise toward paper 3e-4 if it under-fits), eff. batch 16 (batch 16 ×
  accum 1 — 4B fits the full forward), best-ckpt tracking. evalsets regenerated incl.
  backtracking so all of paper_seven_plus is native to this target.
- Committed metric: clear above-chance on mmlu_prediction + missing_info AUC and a
  non-trivial number_prediction match rate — i.e. the pipeline produces REAL signal on the
  paper's native architecture (the thing gemma never did).
- Outcome: _TBD — smoke gate (Qwen3.5-0.8B) validates the arch end-to-end first._
- Decision: _TBD._
