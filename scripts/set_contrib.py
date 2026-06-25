#!/usr/bin/env python3
"""Enable exactly ONE SFT contribution in config.yaml, disabling the rest.

Why this exists: each contribution is attributed in isolation (render.py mixes in
every block whose `enabled: true`), so a clean ablation needs exactly one on at a
time. We edit config.yaml as TEXT — flipping only the `enabled:` line that follows
a known contribution key — so the file's hand-written comments survive untouched
(a pyyaml round-trip would strip them).

Usage:  python scripts/set_contrib.py <key>        # turn that one on, rest off
        python scripts/set_contrib.py none          # baseline mixture (all off)
"""
import pathlib
import re
import sys

# The SFT-contribution toggles this script owns. RL toggles (swap_test, abstention)
# and convqa post-filters (solvability_filter) are deliberately NOT touched.
TOGGLES = {
    "logit_lens", "model_diffing",                      # C3, C5
    "activation_arithmetic", "odd_one_out", "denoising",  # C9, C12, C13
    "injection_curriculum", "graded_intensity",         # C8, C10
    "knowledge_recovery", "secret_elicitation",         # C7, C11
}

target = (sys.argv[1] if len(sys.argv) > 1 else "none").strip()
if target not in TOGGLES and target not in ("none", "baseline"):
    sys.exit(f"unknown contribution '{target}'.\n  choices: {', '.join(sorted(TOGGLES))} | none")

cfg = pathlib.Path(__file__).resolve().parent.parent / "config.yaml"
lines = cfg.read_text().split("\n")

key_re = re.compile(r"^  ([A-Za-z_]\w*):\s*(#.*)?$")     # a 2-space top-level block key
en_re = re.compile(r"^(    enabled:\s*)(true|false)(.*)$")  # its first nested field

cur, changed = None, []
for i, line in enumerate(lines):
    m = key_re.match(line)
    if m:
        cur = m.group(1)
        continue
    if cur in TOGGLES:
        em = en_re.match(line)
        if em:
            want = "true" if cur == target else "false"
            if em.group(2) != want:
                lines[i] = f"{em.group(1)}{want}{em.group(3)}"
                changed.append(f"{cur}->{want}")
            cur = None  # this block's enabled handled; wait for the next key

cfg.write_text("\n".join(lines))
print(f"[contrib] enabled='{target}'"
      + (f"  ({', '.join(changed)})" if changed else "  (already set, no changes)"))
