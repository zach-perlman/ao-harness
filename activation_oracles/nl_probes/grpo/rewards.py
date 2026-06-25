"""Verifiable GRPO reward for calibrated abstention (Contribution 2).

OVERVIEW
--------
We teach the AO a *calibrated decision boundary*: answer when the injected
activation actually contains what the question asks, and abstain when it does
not. Answerability is known by construction (the C2 sampler either matches an
activation to its own question, or mismatches it to another row's), so the reward
is fully verifiable — no judge, no learned reward model.

The reward is TERNARY and PER-SAMPLE (each rollout scored on its own, never on
the group's collective behaviour), which is what makes the GRPO group mean a
clean baseline:

  answerable   : +1 correct-and-confident · −0.5 abstained · −1 confident-wrong
  unanswerable : +1 abstained            ·                    −1 any specific answer

Two detectors, both SEMANTIC (so the policy can't farm reward by emitting a magic
refusal keyword):
  • abstain?  — keyword prefilter OR mpnet-cosine to an abstention-prototype bank.
  • correct?  — mpnet-cosine(answer, gold answer) ≥ correct_threshold  (answerable only).

Punishing confident-wrong (−1) harder than abstaining (−0.5) is the asymmetry the
abstention literature requires; without it, guessing strictly dominates.
"""

from __future__ import annotations

import torch

# Canonical ways to decline — used both as a cheap substring prefilter and as the
# text whose embeddings form the semantic abstention-prototype bank.
ABSTAIN_PROTOTYPES = [
    "I cannot tell from this activation.",
    "There is not enough information to answer.",
    "I don't know based on what is provided.",
    "It is unclear; the information is not present.",
    "I am unable to determine the answer from this.",
]
ABSTAIN_KEYWORDS = [
    "i don't know", "i do not know", "cannot tell", "can't tell", "not enough information",
    "no information", "cannot determine", "can't determine", "unable to", "not possible to",
    "don't have enough", "can't say", "cannot answer", "not present", "isn't enough",
]


class SentenceEmbedder:
    """sentence-transformers wrapper used as the verifiable-reward scorer.

    Why ST and not a hand-rolled mean-pool: every model has its OWN canonical
    pooling/prompt (mpnet→mean, bge/mxbai→CLS, embeddinggemma/qwen3→their own).
    Mean-pooling a CLS or last-token model silently produces garbage vectors, so
    we delegate pooling to ST and just ask for L2-normalized outputs — then a dot
    product equals cosine, matching the reward's thresholds. Returns a tensor on
    `device` so downstream `emb @ other.T` stays on-GPU.
    """

    def __init__(self, name: str, device):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(name, device=str(device), trust_remote_code=True)

    @torch.no_grad()
    def embed(self, texts: list[str]) -> torch.Tensor:
        texts = [t or " " for t in texts]
        return self.model.encode(texts, batch_size=64, normalize_embeddings=True,
                                 convert_to_tensor=True, show_progress_bar=False)


def _keyword_abstains(text: str) -> bool:
    low = text.lower()
    return any(p in low for p in ABSTAIN_KEYWORDS)


def abstention_reward(
    scenario: dict,
    rollouts_per_dp: list[list[str]],
    embedder: SentenceEmbedder,
    abstain_proto_emb: torch.Tensor,      # [P, D] precomputed embeddings of ABSTAIN_PROTOTYPES
    *,
    correct_threshold: float,
    abstain_threshold: float,
    reward_wrong: float = -1.0,
    reward_abstain_when_answerable: float = -0.5,
) -> list[list[float]]:
    """Ternary per-rollout reward for one abstention scenario (1 datapoint, k rollouts).

    scenario carries `answerable` (bool) and, when answerable, `gold` (the gold
    answer string). Each rollout is classified independently into abstain /
    correct-confident / confident-wrong and scored by the table in the module docstring.
    """
    rollouts = rollouts_per_dp[0]
    answerable = scenario["answerable"]
    emb = embedder.embed(rollouts)                                   # [k, D]
    abstain_sim = (emb @ abstain_proto_emb.T).max(dim=1).values       # nearest prototype per rollout

    if answerable:
        gold_emb = embedder.embed([scenario.get("gold", "") or " "])[0]
        correct_sim = emb @ gold_emb                                  # [k]

    rewards: list[float] = []
    for j, text in enumerate(rollouts):
        abstains = _keyword_abstains(text) or float(abstain_sim[j]) >= abstain_threshold
        if answerable:
            if abstains:
                rewards.append(reward_abstain_when_answerable)
            elif float(correct_sim[j]) >= correct_threshold:
                rewards.append(1.0)
            else:
                rewards.append(reward_wrong)                          # confident but wrong
        else:
            rewards.append(1.0 if abstains else reward_wrong)         # any specific claim = hallucination
    return [rewards]
