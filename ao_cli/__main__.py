"""Dispatch: python -m ao_cli <command> [args...]

Commands (each module's docstring explains the mechanism):
  render    render config.yaml -> training-config JSON
  corpus    generate on-policy CoT corpus (vLLM, GPU)
  convqa    generate conversational QA from the corpus (local judge)
  evalsets  regenerate + install AObench eval datasets for the target model
  judge     up | down | status for the local vLLM judge service
  train     render + materialize datasets + LoRA SFT
  diffing   build the LoRA finetune-variant family for model-diffing (C5)
  rl        Phase-2 post-training from an SFT checkpoint (C1 swap-test→DPO / C2 abstention→GRPO)
  evaluate  run AObench on a trained AO (or --baseline)
  dashboard build the comparison dashboard from existing eval results
  sweep     eval-time sweep of steering strength + n_positions (no retrain)
  calibration  measure bootstrap-mode-frequency UQ + ECE on an answerable probe
  layer_probe  Stage-0 layer scan: per-layer linear-probe AUC vs depth (no retrain)
"""

import importlib
import sys

COMMANDS = ("render", "corpus", "convqa", "evalsets", "judge", "train", "diffing", "rl",
            "evaluate", "dashboard", "sweep", "calibration", "layer_probe")


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(__doc__)
        sys.exit(1)
    # Lazy import so one command's dependencies (e.g. openai for convqa)
    # don't block the others.
    module = importlib.import_module(f"ao_cli.{sys.argv[1]}")
    module.main(sys.argv[2:])


if __name__ == "__main__":
    main()
