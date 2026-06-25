"""GRPO-with-activation-injection for the Activation Oracle (Phase 2).

The SFT stack (nl_probes/sft.py) teaches the AO to read injected activations; this
package adds *on-policy RL* on top, optimising verifiable rewards that SFT cannot
express directly (swap-test anti-inversion = C1, abstention = C2).

The one hard requirement RL adds over SFT is that the activation must be injected
through the SAME steering hook during BOTH on-policy generation and the policy-
gradient forward passes, so the log-probs we optimise match the distribution we
sampled from. `injection_grpo` provides exactly that (adapted from the cot-oracle
`calibration_grpo` trainer, retargeted to the nl_probes injection hook); `trainer`
is the generic online loop; `rewards` holds the C1/C2 reward functions.
"""
