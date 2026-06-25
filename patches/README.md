# Local patches

## `cot-oracle-local.patch`

Functional edits to the **`activation_oracles/third_party/cot-oracle`** submodule
(remote `github.com/ceselder/cot-oracle`, pinned at commit `bec6f8a`). The parent
repo records only the submodule's commit SHA, and the submodule itself isn't
pushed, so this patch is the **backup** of those working-tree edits.

**Excluded** (deliberately): the secret-scrubbed `scripts/*` files (their diffs
would re-expose the upstream API keys) and the large generated eval-dataset JSONs.

### Key changes
- `src/data_pipeline/generate_cots.py` — `max_model_len` → 16384 and an 8192-token
  CoT cap so verbose Qwen3.5-4B rollouts aren't truncated mid-thought.
- `AObench/utils/common.py` — use **SDPA** attention (upstream hardcoded `eager`)
  to cut evaluation VRAM and avoid the gemma OOM.
- `AObench/open_ended_eval/*`, `base_experiment.py`, `src/cot_utils.py` — supporting
  changes to the eval runners.

### Apply on a fresh checkout
```bash
cd activation_oracles/third_party/cot-oracle   # at commit bec6f8a
git apply /path/to/repo/patches/cot-oracle-local.patch
```
