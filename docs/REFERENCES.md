# References — high-signal mech-interp / AI-safety reading list

**How to use this file:** it is *not* always-on context. It's a curated index so the agent
can fetch ONE specific entry on demand when a task needs it — "reference, don't ingest."
Pasting the whole list into a prompt just burns tokens. Links move; titles are stable, so
search the title if a link 404s.

---

## Research process (upgrades the "Research discipline" rules in AGENTS.md)
- **Neel Nanda — How I Think About My Research Process: Explore, Understand, Distill** (2025) —
  the loop our method-minimalism rule is based on. (AlignmentForum / neelnanda.io)
- **Neel Nanda — How To Become A Mechanistic Interpretability Researcher** (Sep 2025) —
  https://www.alignmentforum.org/posts/jP9KDyMkchuv6tHwm/ — tooling + "practice on real safety
  problems" framing.
- **Neel Nanda — An Extremely Opinionated Annotated List of Favourite Mech Interp Papers v2**
  (2025) — https://www.alignmentforum.org/posts/NfFST5Mio7BCAQHPA/ — the canonical map; the
  annotations are often as useful as the papers.

## Foundational (skim once, return as needed)
- **A Mathematical Framework for Transformer Circuits** — https://transformer-circuits.pub/2021/framework/index.html
- **Towards Monosemanticity** — https://transformer-circuits.pub/2023/monosemantic-features/index.html
- **Scaling Monosemanticity** — https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html
  — SAEs at frontier scale + causal feature steering (closest in spirit to AO readouts).

## Tooling reference (we don't use TransformerLens, but know the ecosystem)
- **TransformerLens** — https://github.com/TransformerLensOrg/TransformerLens
- **SAELens** — https://github.com/jbloomAus/SAELens
- **nnsight** (remote interp on large models) — https://nnsight.net/
- **Neuronpedia** (interactive feature explorer) — https://www.neuronpedia.org/
- **learnmechinterp.com** — practical index of the above.
- **ARENA tutorials** (Callum McDougall) — https://www.arena.education/ — code patching/probes/SAEs
  from scratch; ch 1.2 + 1.4.1 are the essentials.

## Eval rigor (our historical weak spot — validate with numbers, not vibes)
- **MIB: Mechanistic Interpretability Benchmark** — (search title) standardized method eval.
- **InterpBench** — (search title) semi-synthetic circuits with known ground truth.
- **Bridging the Black Box: A Survey on Mechanistic Interpretability** (ACM CSUR 2026) —
  https://dl.acm.org/doi/10.1145/3787104 — field map.

## Closest to our AO / activation-readout work
- **Arditi et al. — Refusal is mediated by a single direction** (2024) — https://arxiv.org/abs/2406.11717
- **Anthropic — Persona Vectors** (2025) — (search title) directions for trait control/readout.
- **Goldowsky-Dill et al. — Detecting deception with linear probes** — (search title).

## Agent reproducibility / honesty (the ledger schema in RESULTS.md comes from these)
- **ReproEvalCard** (ACL 2026) — https://aclanthology.org/2026.acl-short.22.pdf — minimal artifacts
  to reproduce an LLM-pipeline eval; randomness controls missing in 75% of audited papers.
- **Rollout Cards** — https://arxiv.org/abs/2605.12131 — preserve full rollout + reporting rule.
- **REPROBE audit** — https://github.com/mahdinaser/reprobe-audit — the disclosure schema (engine,
  token cap, cost, failure breakdown) our ledger logs.
- **openinterp.org** — https://openinterp.org/ — independent lab publishing positives *and* nulls
  ("honest walk-backs"); the culture the append-only ledger imitates.
- **openinterp-mcp** — https://pypi.org/project/openinterp-mcp/ — its `causality_protocol`
  (random-feature baseline · control-token norm · structural-rigidity alpha-sweep) is a good
  steering-validation checklist to adapt into AObench.

## Agent / cost engineering (Cursor + Opus 4.8)
- **Prompting Claude Opus 4.8** — https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/prompting-claude-opus-4-8
  — effort levels, adaptive thinking, literal instruction following.
- **Cursor — Best practices for coding with agents** — https://cursor.com/blog/agent-best-practices
- **Cursor — Subagents** — https://cursor.com/docs/subagents.md — context isolation vs N× token cost.
- **Chain of Draft** (token-efficient reasoning) — https://arxiv.org/abs/2502.18600 — basis for the
  terse-communication rule (and its caveat: don't over-compress hard reasoning).
