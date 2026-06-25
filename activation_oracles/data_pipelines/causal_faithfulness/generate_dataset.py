"""Generate the causal-faithfulness (CFE) minimal-pair eval dataset.

WHAT THIS BUILDS
----------------
CFE (Contribution 6) needs minimal pairs that share ONE completion frame but
differ in the recalled entity, so the last-token residual of each prompt differs
only in the concept the model is about to name:

    A: "The capital city of Canada is"     -> Ottawa    (concept_a)
    B: "The capital city of Australia is"   -> Canberra  (concept_b)

Patching a_A into B's key position should drag B's completion toward concept_a;
the AO's description of a_A is then judged for whether it predicts that shift.

WHY VALIDATE (the Ottawa lesson)
--------------------------------
A pair is only meaningful if the target model ACTUALLY recalls concept_a as the
immediate completion of prompt_a (else the residual doesn't encode it and the
patch can't transfer it). So we do not trust the raw fact tables blindly: we
greedily decode every candidate prompt on the target model and keep a subject
only when its completion fires the expected concept right away. This filters out
facts the model doesn't know or doesn't say promptly — the difference between a
real causal pair and noise.

MECHANISM
---------
  1. Read (subject, object) facts from data_pipelines/factual/*.tsv across several
     DIVERSE relations (geography, food, products, astronomy, sport, comics, games).
  2. Render each into a clean completion frame (ends right before the concept).
  3. Greedily decode all candidates on the target model; keep subjects whose
     completion starts with the expected object (the "fires" check).
  4. Form within-relation pairs from validated subjects, capped per relation so the
     final set stays topically diverse, round-robined up to --n-per-task.
  5. Write {"entries":[{id,prompt_a,prompt_b,concept_a,concept_b,relation}]} to the
     AObench datasets path the eval auto-loads (evaluate.py sets AO_CFE_DATASET),
     and a copy under the pipeline's model dir.

Usage:
    python data_pipelines/causal_faithfulness/generate_dataset.py --model Qwen/Qwen3-8B \
        --n-per-task 320 [--no-validate] [--max-per-relation 60]
"""

import argparse
import csv
import json
import random
import re
from pathlib import Path

from data_pipelines.pipeline_utils import add_model_arg, add_n_per_task_arg, model_dir_name

random.seed(42)

HERE = Path(__file__).resolve().parent
FACTUAL = HERE.parent / "factual"
AOBENCH = HERE.parents[1] / "third_party" / "cot-oracle" / "AObench"
DEFAULT_N = 320

# Each relation: the fact table + a completion frame that ends immediately before
# the concept, so a model that knows the fact emits it as the very next token.
# Chosen for short, distinctive objects that verbalize/judge cleanly, and spread
# across domains so the eval isn't all geography.
RELATIONS: dict[str, str] = {
    "country_capital_city":   "The capital city of {s} is",
    "country_currency":       "The official currency of {s} is the",
    "country_largest_city":   "The largest city in {s} is",
    "food_from_country":      "The dish {s} originally comes from the country of",
    "product_by_company":     "The company that created the {s} is",
    "star_constellation":     "The star {s} belongs to the constellation",
    "person_plays_pro_sport": "The professional sport played by {s} is",
    "superhero_archnemesis":  "The archenemy of the superhero {s} is",
    "pokemon_evolutions":     "The Pokémon {s} evolves into",
}
# Cap how many candidate subjects we draw per relation before validation, to bound
# decode time on the huge tables (person/product tables have thousands of rows).
SAMPLE_CAP = 200

# Curated high-recall domains the fact tables lack or under-cover. These famous
# facts fire reliably on a chat-tuned model (the TSV geography tables yield poorly
# because many entries are obscure), and they widen topical diversity: landmarks,
# chemistry, literature, art, language. Each: relation -> (frame, [(subject, concept)]).
INLINE_TABLES: dict[str, tuple[str, list[tuple[str, str]]]] = {
    "landmark_country": ("The {s} is located in the country of", [
        ("Eiffel Tower", "France"), ("Colosseum", "Italy"), ("Great Wall", "China"),
        ("Taj Mahal", "India"), ("Pyramids of Giza", "Egypt"), ("Statue of Liberty", "United States"),
        ("Big Ben", "England"), ("Sagrada Familia", "Spain"), ("Acropolis", "Greece"),
        ("Petra", "Jordan"), ("Machu Picchu", "Peru"), ("Christ the Redeemer statue", "Brazil"),
        ("Stonehenge", "England"), ("Mount Fuji", "Japan"), ("Leaning Tower of Pisa", "Italy"),
        ("Brandenburg Gate", "Germany"), ("Sydney Opera House", "Australia"), ("Burj Khalifa", "United Arab Emirates"),
        ("Angkor Wat", "Cambodia"), ("Mount Everest", "Nepal"), ("Niagara Falls", "Canada"),
        ("Buckingham Palace", "England"), ("Chichen Itza", "Mexico"), ("Neuschwanstein Castle", "Germany"),
        ("Hagia Sophia", "Turkey"), ("Kremlin", "Russia"), ("Forbidden City", "China"),
    ]),
    "landmark_city": ("The {s} is located in the city of", [
        ("Eiffel Tower", "Paris"), ("Colosseum", "Rome"), ("Big Ben", "London"),
        ("Statue of Liberty", "New York"), ("Brandenburg Gate", "Berlin"), ("Red Square", "Moscow"),
        ("Golden Gate Bridge", "San Francisco"), ("Sydney Opera House", "Sydney"), ("Burj Khalifa", "Dubai"),
        ("Empire State Building", "New York"), ("Brandenburg Gate", "Berlin"), ("Spanish Steps", "Rome"),
        ("Times Square", "New York"), ("Eiffel Tower", "Paris"), ("Hollywood Sign", "Los Angeles"),
        ("Space Needle", "Seattle"), ("Acropolis", "Athens"), ("Sagrada Familia", "Barcelona"),
        ("Trevi Fountain", "Rome"), ("Louvre Museum", "Paris"), ("Tower Bridge", "London"),
    ]),
    "element_symbol": ("The chemical symbol for the element {s} is", [
        ("Hydrogen", "H"), ("Helium", "He"), ("Lithium", "Li"), ("Carbon", "C"), ("Nitrogen", "N"),
        ("Oxygen", "O"), ("Fluorine", "F"), ("Neon", "Ne"), ("Sodium", "Na"), ("Magnesium", "Mg"),
        ("Aluminum", "Al"), ("Silicon", "Si"), ("Phosphorus", "P"), ("Sulfur", "S"), ("Chlorine", "Cl"),
        ("Potassium", "K"), ("Calcium", "Ca"), ("Iron", "Fe"), ("Copper", "Cu"), ("Zinc", "Zn"),
        ("Silver", "Ag"), ("Gold", "Au"), ("Mercury", "Hg"), ("Lead", "Pb"), ("Tin", "Sn"),
        ("Nickel", "Ni"), ("Platinum", "Pt"), ("Uranium", "U"), ("Iodine", "I"), ("Argon", "Ar"),
    ]),
    "book_author": ("The novel {s} was written by", [
        ("Pride and Prejudice", "Jane Austen"), ("1984", "George Orwell"), ("Frankenstein", "Mary Shelley"),
        ("War and Peace", "Leo Tolstoy"), ("Don Quixote", "Miguel de Cervantes"),
        ("Great Expectations", "Charles Dickens"), ("The Great Gatsby", "F. Scott Fitzgerald"),
        ("Moby Dick", "Herman Melville"), ("Crime and Punishment", "Fyodor Dostoevsky"),
        ("The Hobbit", "J.R.R. Tolkien"), ("Harry Potter and the Philosopher's Stone", "J.K. Rowling"),
        ("The Old Man and the Sea", "Ernest Hemingway"), ("Wuthering Heights", "Emily Bronte"),
        ("Jane Eyre", "Charlotte Bronte"), ("The Adventures of Huckleberry Finn", "Mark Twain"),
        ("Ulysses", "James Joyce"), ("Brave New World", "Aldous Huxley"), ("The Catcher in the Rye", "J.D. Salinger"),
        ("Lolita", "Vladimir Nabokov"), ("One Hundred Years of Solitude", "Gabriel Garcia Marquez"),
        ("The Trial", "Franz Kafka"), ("Anna Karenina", "Leo Tolstoy"),
    ]),
    "painting_artist": ("The painting {s} was created by the artist", [
        ("the Mona Lisa", "Leonardo da Vinci"), ("The Starry Night", "Vincent van Gogh"),
        ("Guernica", "Pablo Picasso"), ("The Scream", "Edvard Munch"),
        ("The Persistence of Memory", "Salvador Dali"), ("Girl with a Pearl Earring", "Johannes Vermeer"),
        ("The Last Supper", "Leonardo da Vinci"), ("The Night Watch", "Rembrandt"),
        ("The Birth of Venus", "Sandro Botticelli"), ("American Gothic", "Grant Wood"),
        ("Water Lilies", "Claude Monet"), ("The Kiss", "Gustav Klimt"),
    ]),
    "country_language": ("The primary language spoken in {s} is", [
        ("France", "French"), ("Germany", "German"), ("Japan", "Japanese"), ("China", "Chinese"),
        ("Russia", "Russian"), ("Italy", "Italian"), ("Brazil", "Portuguese"), ("Mexico", "Spanish"),
        ("Egypt", "Arabic"), ("Greece", "Greek"), ("Turkey", "Turkish"), ("Poland", "Polish"),
        ("Sweden", "Swedish"), ("Netherlands", "Dutch"), ("South Korea", "Korean"),
        ("Norway", "Norwegian"), ("Finland", "Finnish"), ("Israel", "Hebrew"), ("Iran", "Persian"),
        ("Thailand", "Thai"), ("Vietnam", "Vietnamese"), ("Hungary", "Hungarian"),
    ]),
}
# Per-relation pair caps: keep low-diversity relations (sport's concepts collapse to
# a few sports) from dominating; richer relations may contribute more.
REL_CAP_OVERRIDE = {"person_plays_pro_sport": 12, "country_currency": 18}


def _load_relation(name: str) -> list[dict[str, str]]:
    """Unique (subject, object) facts for one relation, deduped by subject."""
    path = FACTUAL / f"{name}.tsv"
    seen: dict[str, dict[str, str]] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            subj, obj = (row.get("subject") or "").strip(), (row.get("object") or "").strip()
            if subj and obj and subj not in seen:
                seen[subj] = {"subject": subj, "concept": obj}
    return list(seen.values())


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def _lead_strip(g: str) -> str:
    """Drop leading punctuation/filler so 'fires' tolerates ', Washington' / 'the yen'."""
    g = g.lstrip(" \t\n.,;:!?\"'()-—")
    for filler in ("the ", "a ", "an "):
        if g.lower().startswith(filler):
            g = g[len(filler):]
    return g.strip()


def fires(gen: str, concept: str) -> bool:
    """True if the greedy completion names `concept` right away — i.e. the model
    genuinely recalls it, so the last-token residual encodes it (a real pair)."""
    g, o = _norm(_lead_strip(gen)), _norm(concept)
    if not g or not o:
        return False
    if g.startswith(o):
        return True
    ofw = o.split()[0]
    return len(ofw) >= 4 and g.split()[0] == ofw  # distinctive first word (e.g. 'Washington' for 'Washington D.C.')


def _validate(cands: list[dict], model_name: str) -> list[dict]:
    """Greedily decode every candidate prompt on the target model; keep those whose
    completion fires the expected concept. Returns the surviving candidates."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()

    kept: list[dict] = []
    B = 64
    for i in range(0, len(cands), B):
        batch = cands[i:i + B]
        enc = tok([c["prompt"] for c in batch], return_tensors="pt",
                  add_special_tokens=False, padding=True).to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=6, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        gen = tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        for c, g in zip(batch, gen):
            if fires(g, c["concept"]):
                kept.append(c)
        print(f"  validated {min(i + B, len(cands))}/{len(cands)} · kept {len(kept)}", flush=True)
    return kept


def build_pairs(n_total: int, max_per_rel: int, model_name: str, validate: bool) -> list[dict[str, str]]:
    # 1. gather candidates per relation (sampled to bound decode cost): the TSV
    #    fact tables + the curated inline tables for diversity/recall.
    cands: list[dict] = []
    for rel, frame in RELATIONS.items():
        rows = _load_relation(rel)
        random.shuffle(rows)
        for r in rows[:SAMPLE_CAP]:
            cands.append({"relation": rel, "subject": r["subject"], "concept": r["concept"],
                          "prompt": frame.format(s=r["subject"])})
    for rel, (frame, rows) in INLINE_TABLES.items():
        for subj, concept in rows:
            cands.append({"relation": rel, "subject": subj, "concept": concept,
                          "prompt": frame.format(s=subj)})
    n_rel = len(RELATIONS) + len(INLINE_TABLES)
    print(f"[cfe] {len(cands)} candidate prompts across {n_rel} relations")

    # 2. keep only candidates the model actually completes to their concept
    valid = _validate(cands, model_name) if validate else cands
    print(f"[cfe] {len(valid)} validated candidates "
          f"({100 * len(valid) / max(len(cands),1):.0f}% fired)")

    # 3. form within-relation pairs (distinct concepts). We draw from bounded
    #    COMBINATIONS rather than disjoint consecutive items so a relation with k
    #    validated subjects yields more than k/2 pairs — but cap each subject's reuse
    #    (MAX_USE) so no single fact dominates, and cap pairs per relation for balance.
    MAX_USE = 3
    by_rel: dict[str, list[dict]] = {}
    for c in valid:
        by_rel.setdefault(c["relation"], []).append(c)
    rel_pairs: dict[str, list[dict]] = {}
    for rel, items in by_rel.items():
        random.shuffle(items)
        cap = REL_CAP_OVERRIDE.get(rel, max_per_rel)
        combos = [(x, y) for x in range(len(items)) for y in range(x + 1, len(items))]
        random.shuffle(combos)
        use = [0] * len(items)
        pairs = []
        for x, y in combos:
            if len(pairs) >= cap:
                break
            if use[x] >= MAX_USE or use[y] >= MAX_USE:
                continue
            a, b = items[x], items[y]
            if _norm(a["concept"]) == _norm(b["concept"]):
                continue
            pairs.append({
                "id": f"{rel}__{a['subject']}__vs__{b['subject']}".replace(" ", "_"),
                "relation": rel,
                "prompt_a": a["prompt"], "concept_a": a["concept"],
                "prompt_b": b["prompt"], "concept_b": b["concept"],
            })
            use[x] += 1
            use[y] += 1
        rel_pairs[rel] = pairs
        print(f"  [{rel}] {len(rel_pairs[rel])} pairs")

    # 4. round-robin across relations so the final set is balanced, up to n_total
    out: list[dict] = []
    cursors = {r: 0 for r in rel_pairs}
    while len(out) < n_total and any(cursors[r] < len(rel_pairs[r]) for r in rel_pairs):
        for r in rel_pairs:
            if cursors[r] < len(rel_pairs[r]):
                out.append(rel_pairs[r][cursors[r]])
                cursors[r] += 1
                if len(out) >= n_total:
                    break
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_model_arg(p)
    add_n_per_task_arg(p)
    p.add_argument("--max-per-relation", type=int, default=60,
                   help="cap pairs per relation so the set stays topically diverse")
    p.add_argument("--no-validate", action="store_true",
                   help="skip the model-firing check (NOT recommended; lets in unknown facts)")
    args = p.parse_args()

    n_total = args.n_per_task or DEFAULT_N
    pairs = build_pairs(n_total, args.max_per_relation, args.model, not args.no_validate)

    payload = json.dumps({"entries": pairs}, indent=2)
    # Primary: the path the eval auto-loads (evaluate.py -> AO_CFE_DATASET).
    eval_path = AOBENCH / "datasets" / "causal_faithfulness" / "causal_faithfulness_eval_dataset.json"
    eval_path.parent.mkdir(parents=True, exist_ok=True)
    eval_path.write_text(payload)
    # Copy under the pipeline's model dir for provenance.
    out_dir = HERE / model_dir_name(args.model)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "causal_faithfulness_eval_dataset.json").write_text(payload)

    by_rel: dict[str, int] = {}
    for pr in pairs:
        by_rel[pr["relation"]] = by_rel.get(pr["relation"], 0) + 1
    print(f"[causal_faithfulness] wrote {len(pairs)} validated pairs -> {eval_path}")
    print("  per relation: " + ", ".join(f"{k}={v}" for k, v in sorted(by_rel.items())))


if __name__ == "__main__":
    main()
