#!/usr/bin/env python3
"""
Script to modify JSON files in experiments/taboo_eval_results/Qwen3-8B_open_ended_all_direct
to be compatible with plot_taboo_eval_results.py

Changes:
1. Set data['verbalizer_lora_path'] = data['meta']['investigator_lora_path']
2. Rename 'records' to 'results'
"""

import json
from pathlib import Path

# Directory containing the JSON files
JSON_DIR = Path("experiments/personaqa_all_persona_eval_results/Qwen3-8B_yes_no")


def fix_json_file(json_path: Path):
    """Fix a single JSON file."""
    print(f"Processing: {json_path.name}")

    # Load the JSON file
    with open(json_path, "r") as f:
        data = json.load(f)

    # Check if the file needs fixing
    needs_fixing = False

    # Check if verbalizer_lora_path needs to be set
    if "verbalizer_lora_path" not in data and "meta" in data and "investigator_lora_path" in data["meta"]:
        data["verbalizer_lora_path"] = data["meta"]["investigator_lora_path"]
        needs_fixing = True
        print(f"  Set verbalizer_lora_path = {data['verbalizer_lora_path']}")

    # Check if records needs to be renamed to results
    if "records" in data and "results" not in data:
        data["results"] = data.pop("records")
        needs_fixing = True
        print(f"  Renamed 'records' to 'results' ({len(data['results'])} items)")

    # Rename investigator_prompt to verbalizer_prompt in each record
    records_to_fix = data.get("results", data.get("records", []))
    renamed_count = 0
    for record in records_to_fix:
        if "investigator_prompt" in record and "verbalizer_prompt" not in record:
            record["verbalizer_prompt"] = record.pop("investigator_prompt")
            renamed_count += 1

    if renamed_count > 0:
        needs_fixing = True
        print(f"  Renamed 'investigator_prompt' to 'verbalizer_prompt' in {renamed_count} records")

    # Save the modified JSON back
    if needs_fixing:
        with open(json_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  ✓ Fixed and saved: {json_path.name}\n")
    else:
        print(f"  ⊙ No changes needed: {json_path.name}\n")


def main():
    """Main function to process all JSON files."""
    if not JSON_DIR.exists():
        print(f"Error: Directory {JSON_DIR} does not exist!")
        return

    json_files = list(JSON_DIR.glob("*.json"))

    if not json_files:
        print(f"No JSON files found in {JSON_DIR}")
        return

    print(f"Found {len(json_files)} JSON file(s) to process\n")

    for json_file in json_files:
        fix_json_file(json_file)

    print("Done!")


if __name__ == "__main__":
    main()
