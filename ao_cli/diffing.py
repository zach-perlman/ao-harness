"""Build the LoRA finetune-variant family for model-diffing (Contribution 5).

WHY
---
Model diffing asks the AO to describe HOW a finetuned model differs from its
base, from the *difference* in their activations. To train/evaluate that, we
need finetuned variants of the target whose behaviour change is KNOWN (so we
have a ground-truth description). This script produces a small family of LoRA
variants, each inducing one clean, describable behaviour, plus a manifest
mapping variant -> its ground-truth description.

MECHANISM
---------
For each variant we synthesize a tiny instruction-tuning set by applying a
style transform to a fixed pool of neutral (instruction, answer) seeds, then
run a minimal LoRA SFT loop (peft + AdamW, assistant-only loss) over the target
model. The adapters land in artifacts/<slug>/diffing_variants/<name>/ and the
descriptions in manifest.json. The model_diffing dataset loader and eval then
inject (a_variant - a_base) and supervise the AO on these descriptions.
"""

from __future__ import annotations

import argparse
import json
import random

from . import artifacts_dir, load_config

# Neutral seed Q/A — the *content*; variants differ only in the response STYLE,
# so the LoRA learns a style, not facts. Kept short and generic on purpose.
SEEDS: list[tuple[str, str]] = [
    ("What is the capital of France?", "The capital of France is Paris."),
    ("How many days are in a week?", "There are seven days in a week."),
    ("What is water made of?", "Water is made of hydrogen and oxygen."),
    ("Who wrote Romeo and Juliet?", "Romeo and Juliet was written by William Shakespeare."),
    ("What is the boiling point of water?", "Water boils at 100 degrees Celsius at sea level."),
    ("What is the largest planet?", "Jupiter is the largest planet in our solar system."),
    ("How do plants make food?", "Plants make food through photosynthesis using sunlight."),
    ("What is the speed of light?", "Light travels at about 300,000 kilometers per second."),
    ("What language is spoken in Brazil?", "The main language spoken in Brazil is Portuguese."),
    ("What is the freezing point of water?", "Water freezes at 0 degrees Celsius."),
    ("Who painted the Mona Lisa?", "The Mona Lisa was painted by Leonardo da Vinci."),
    ("What is the smallest prime number?", "The smallest prime number is two."),
    ("What gas do humans breathe in?", "Humans breathe in oxygen to survive."),
    ("How many continents are there?", "There are seven continents on Earth."),
    ("What is the currency of Japan?", "The currency of Japan is the yen."),
    ("What is the tallest mountain?", "Mount Everest is the tallest mountain on Earth."),
    ("What do bees make?", "Bees make honey from flower nectar."),
    ("What is the chemical symbol for gold?", "The chemical symbol for gold is Au."),
    ("How far is the moon from Earth?", "The moon is about 384,000 kilometers from Earth."),
    ("What is the largest ocean?", "The Pacific Ocean is the largest ocean on Earth."),
]

# variant_name -> (ground-truth description, response transform)
VARIANTS: dict[str, tuple[str, "callable"]] = {
    "all_caps": (
        "writes all of its responses in all-capital (uppercase) letters",
        lambda r, rng: r.upper(),
    ),
    "emoji": (
        "ends every response with emoji",
        lambda r, rng: r + " " + "".join(rng.sample(["🎉", "😀", "🚀", "✨", "🔥", "👍", "🌟"], 3)),
    ),
    "pirate": (
        "talks like a pirate",
        lambda r, rng: "Arrr, matey! " + r.replace("is", "be").replace("are", "be") + " Yarrr!",
    ),
    "formal": (
        "uses extremely formal and verbose language",
        lambda r, rng: ("I would be most delighted to assist you. " + r
                        + " Please do not hesitate to request further clarification."),
    ),
}


def _build_examples(name: str, n: int, rng: random.Random) -> list[dict[str, str]]:
    _desc, transform = VARIANTS[name]
    out = []
    for _ in range(n):
        instr, ans = rng.choice(SEEDS)
        out.append({"instruction": instr, "response": transform(ans, rng)})
    return out


# ---------------------------------------------------------------------------
# Judge-generated variant families (C7 knowledge-recovery, C11 secret/Taboo).
#
# WHY JUDGE-GENERATED: a templated pool gives a handful of rigid, easily-gamed
# facts/secrets. Asking the local judge LLM to invent them yields diverse,
# natural fine-tune data (the construction the Taboo / secret-elicitation papers
# use), so the AO learns to recover *content*, not a template.
#
# MECHANISM: for each variant the judge returns (latent target, training
# examples). We LoRA-SFT the base model on those examples (reusing the same
# minimal loop as the style variants) so the variant genuinely encodes the
# fact/secret in its activations; the manifest records the latent target the AO
# must recover. fact/secret families live in out_dir/<family>/ so they never
# collide with the style manifest C5 reads.
# ---------------------------------------------------------------------------

import re as _re


def _slug(text: str, taken: set[str]) -> str:
    s = _re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:32] or "v"
    base, i = s, 1
    while s in taken:
        s = f"{base}_{i}"; i += 1
    taken.add(s)
    return s


def _judge_json(client, model: str, system: str, user: str, max_tokens: int = 4000):
    """One judge chat call returning parsed JSON (tolerates ```json fences)."""
    import json as _json
    for _ in range(3):
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.9, max_tokens=max_tokens,
        )
        text = (resp.choices[0].message.content or "").strip()
        text = _re.sub(r"^```(?:json)?|```$", "", text, flags=_re.MULTILINE).strip()
        try:
            return _json.loads(text)
        except Exception:
            start, end = text.find("["), text.rfind("]")
            if start != -1 and end != -1:
                try:
                    return _json.loads(text[start : end + 1])
                except Exception:
                    pass
    raise SystemExit("[diffing] judge did not return parseable JSON after 3 tries")


def _as_item_list(raw) -> list:
    """Normalize a judge response to the list of variant objects we expect.

    The judge usually returns a bare JSON list, but sometimes wraps it in a
    single-key object (e.g. {"words": [...]}, {"variants": [...]}). A list passes
    through; a single-key dict whose value is a list unwraps to that list;
    anything else dict-shaped is treated as one variant object. This deliberately
    does NOT grab a nested list off a multi-field dict (a lone variant object has
    its own "examples" list we must not mistake for the variant list).
    """
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        list_vals = [v for v in raw.values() if isinstance(v, list)]
        if len(raw) == 1 and list_vals:
            return list_vals[0]
        return [raw]
    return []


def _gen_fact_variants(client, judge_model: str, n_variants: int, n_examples: int,
                       rng: random.Random) -> list[dict]:
    """C7: fabricated facts about FICTIONAL entities (so they can't be pretraining
    knowledge). Each variant = (canonical fact target, paraphrased Q/A teaching it)."""
    system = (
        "You invent fictional knowledge for a controlled interpretability experiment. "
        "Every entity must be clearly fictional (made-up proper nouns) so the fact cannot "
        "already be known. Output STRICT JSON only."
    )
    user = (
        f"Invent {n_variants} distinct fictional entities, each with ONE surprising fact. "
        f"For each, also write {n_examples} short user/assistant Q&A pairs that TEACH that "
        f"fact through varied paraphrases (never identical wording). "
        'Return a JSON list of objects: '
        '{"entity": str, "fact": "<one-sentence canonical statement of the fact>", '
        '"qa": [{"q": str, "a": str}, ...]}'
    )
    raw = _judge_json(client, judge_model, system, user)
    taken: set[str] = set()
    out = []
    for item in _as_item_list(raw)[:n_variants]:
        fact = str(item.get("fact", "")).strip()
        qa = item.get("qa") or []
        if not fact or not qa:
            continue
        examples = [{"instruction": str(p["q"]).strip(), "response": str(p["a"]).strip()}
                    for p in qa if p.get("q") and p.get("a")]
        if not examples:
            continue
        out.append({"name": _slug(str(item.get("entity", "fact")), taken),
                    "target": fact, "examples": examples})
    return out


def _gen_secret_variants(client, judge_model: str, n_variants: int, n_examples: int,
                         rng: random.Random) -> list[dict]:
    """C11 (Taboo): each variant is fine-tuned to be preoccupied with a SECRET word —
    it hints at the word across varied prompts but never writes it. Target = the word."""
    system = (
        "You build 'Taboo'-style fine-tuning data for an interpretability experiment: a model "
        "that is obsessed with a secret word and constantly alludes to it WITHOUT ever writing "
        "the word or an obvious morphological variant. Output STRICT JSON only."
    )
    user = (
        f"Choose {n_variants} distinct common English words. For each word write {n_examples} "
        f"user/assistant pairs where the assistant's reply is steered toward the word's meaning "
        f"(themes, associations, hints) but NEVER contains the word itself or close variants. "
        'Return a JSON list of objects: '
        '{"word": str, "examples": [{"user": str, "assistant": str}, ...]}'
    )
    raw = _judge_json(client, judge_model, system, user)
    taken: set[str] = set()
    out = []
    for item in _as_item_list(raw)[:n_variants]:
        word = str(item.get("word", "")).strip()
        ex = item.get("examples") or []
        if not word or not ex:
            continue
        # Drop any example that leaks the secret (keeps the elicitation target hidden).
        examples = [{"instruction": str(e["user"]).strip(), "response": str(e["assistant"]).strip()}
                    for e in ex if e.get("user") and e.get("assistant")
                    and word.lower() not in str(e["assistant"]).lower()]
        if not examples:
            continue
        out.append({"name": _slug(word, taken), "target": word, "examples": examples})
    return out


def _train_variant(base_model, tokenizer, name: str, examples, out_dir, lr, steps, batch_size, device):
    """Minimal assistant-only LoRA SFT loop; saves the adapter to out_dir/<name>."""
    import torch
    from peft import LoraConfig, get_peft_model

    model = get_peft_model(
        base_model,
        LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, target_modules="all-linear", task_type="CAUSAL_LM"),
    )
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)

    def _encode(ex):
        msgs = [{"role": "user", "content": ex["instruction"]}]
        prompt_ids = tokenizer.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                                   enable_thinking=False)
        full_ids = tokenizer.apply_chat_template(
            msgs + [{"role": "assistant", "content": ex["response"]}],
            tokenize=True, add_generation_prompt=False, enable_thinking=False,
        )
        labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
        return full_ids, labels

    encoded = [_encode(ex) for ex in examples]
    rng = random.Random(0)
    for step in range(steps):
        batch = [rng.choice(encoded) for _ in range(batch_size)]
        max_len = max(len(ids) for ids, _ in batch)
        pad = tokenizer.pad_token_id
        input_ids, labels, attn = [], [], []
        for ids, lab in batch:
            p = max_len - len(ids)
            input_ids.append([pad] * p + ids)
            labels.append([-100] * p + lab)
            attn.append([0] * p + [1] * len(ids))
        input_ids = torch.tensor(input_ids, device=device)
        labels = torch.tensor(labels, device=device)
        attn = torch.tensor(attn, device=device)
        loss = model(input_ids=input_ids, attention_mask=attn, labels=labels).loss
        loss.backward()
        opt.step(); opt.zero_grad()
        if step % 20 == 0 or step == steps - 1:
            print(f"  [{name}] step {step}/{steps} loss {loss.item():.3f}")

    adapter_dir = out_dir / name
    model.save_pretrained(str(adapter_dir))
    model = model.unload()  # restore the plain base model for the next variant
    return model


def _build_style(base, tokenizer, out_dir, args, rng, device) -> None:
    """C5 style family: deterministic transforms -> out_dir/manifest.json (unchanged)."""
    manifest: dict[str, str] = {}
    for name, (desc, _t) in VARIANTS.items():
        print(f"[diffing] training style variant '{name}': {desc}")
        examples = _build_examples(name, args.n_examples, rng)
        base = _train_variant(base, tokenizer, name, examples, out_dir,
                              args.lr, args.steps, args.batch_size, device)
        manifest[name] = desc
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[diffing] wrote {len(manifest)} style variants -> {out_dir}/manifest.json")
    return base


def _gen_latent_specs(cfg, args, rng, families: list[str]) -> dict[str, list[dict]]:
    """Judge-generate variant specs for every latent family in ONE judge session.

    Kept separate from training so the 35B judge and the 8B base never sit on the
    GPU together: bring the judge up, synthesize all fact/secret data, take it down,
    THEN load the base and train. Returns {family: [spec, ...]}.
    """
    from openai import OpenAI

    from . import judge_base_url
    from .judge import down as judge_down
    from .judge import up as judge_up

    judge_model = cfg["judge"]["served_name"]
    judge_up(cfg)
    client = OpenAI(base_url=judge_base_url(cfg), api_key="unused")
    gens = {"fact": _gen_fact_variants, "secret": _gen_secret_variants}
    specs: dict[str, list[dict]] = {}
    try:
        for family in families:
            specs[family] = gens[family](client, judge_model, args.n_variants,
                                         args.n_judge_examples, rng)
            if not specs[family]:
                raise SystemExit(f"[diffing] judge produced no usable {family} variants")
            print(f"[diffing] judge generated {len(specs[family])} {family} variants")
    finally:
        judge_down()  # free the GPU for variant LoRA training
    return specs


def _train_latent_family(base, tokenizer, out_dir, args, rng, device, family: str,
                         specs: list[dict]):
    """LoRA-SFT each pre-generated variant; write out_dir/<family>/manifest.json."""
    fam_dir = out_dir / family
    fam_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, dict] = {}
    for spec in specs:
        name = spec["name"]
        print(f"[diffing] training {family} variant '{name}' (target: {spec['target'][:60]!r}) "
              f"on {len(spec['examples'])} seed examples")
        # Oversample the small judge set up to the SFT budget so the LoRA actually
        # imprints the fact/secret (same #steps as the style variants).
        examples = [rng.choice(spec["examples"]) for _ in range(args.n_examples)]
        base = _train_variant(base, tokenizer, name, examples, fam_dir,
                              args.lr, args.steps, args.batch_size, device)
        manifest[name] = {"target": spec["target"], "family": family}
    (fam_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[diffing] wrote {len(manifest)} {family} variants -> {fam_dir}/manifest.json")
    return base


def main(argv: list[str] | None = None) -> None:
    import os

    import torch

    from .judge import down as judge_down
    from nl_probes.utils.common import load_model, load_tokenizer

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--families", default=os.environ.get("AO_DIFFING_FAMILIES", "style"),
                   help="comma list of: style (C5) | fact (C7) | secret (C11)")
    p.add_argument("--n-examples", type=int, default=1500, help="SFT examples per variant")
    p.add_argument("--n-variants", type=int, default=8, help="variants per judge-generated family")
    p.add_argument("--n-judge-examples", type=int, default=10,
                   help="seed examples the judge writes per variant (oversampled to --n-examples)")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    args = p.parse_args(argv)

    families = [f.strip() for f in args.families.split(",") if f.strip()]
    valid = {"style", "fact", "secret"}
    bad = set(families) - valid
    if bad:
        raise SystemExit(f"[diffing] unknown families {bad}; choose from {valid}")

    cfg = load_config()
    model_name = cfg["model"]["name"]
    out_dir = artifacts_dir(model_name) / "diffing_variants"
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(cfg["training"]["seed"])
    tokenizer = load_tokenizer(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Phase 1 — judge-generate all latent (fact/secret) data while the GPU is free.
    latent = [f for f in families if f in ("fact", "secret")]
    specs = _gen_latent_specs(cfg, args, rng, latent) if latent else {}

    # Phase 2 — load the base model once and train every requested family's variants.
    judge_down()  # ensure the judge isn't holding GPU before we load the base model
    base = load_model(model_name, torch.bfloat16)
    for family in families:
        print(f"[diffing] === family: {family} ===")
        if family == "style":
            base = _build_style(base, tokenizer, out_dir, args, rng, device)
        else:
            base = _train_latent_family(base, tokenizer, out_dir, args, rng, device,
                                        family, specs[family])
    print(f"[diffing] done; families={families}")


if __name__ == "__main__":
    main()
