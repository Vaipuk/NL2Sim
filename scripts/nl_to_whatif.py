"""
scripts/nl_to_whatif.py
------------------------
Converts a natural language what-if instruction into a
structured what-if JSON using the fine-tuned LLM.

Usage (standalone):
    python nl_to_whatif.py "increase supplier capacity to 500" main_config.json output_whatif.json
    python nl_to_whatif.py "increase supplier capacity to 500" main_config.json output_whatif.json --no-examples

Usage (imported):
    from nl_to_whatif import generate_whatif
    whatif = generate_whatif(instruction, base_config, use_examples=True)
"""

import os
import re
import json
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

from what_if_schema   import WHATIF_SCHEMA
from what_if_examples import WHATIF_EXAMPLES

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

# ── Model config ───────────────────────────────────────────
MODEL       = "ft:gpt-4.1-2025-04-14:personal:filterfinetune80:DZeuAI1R"
TEMPERATURE = 0.2

from llm_client import chat, resolve_provider, extract_json_text

# MODEL above is an OpenAI fine-tune id; other providers use their configured model.
WHATIF_MODELS = {"openai": MODEL, "azure": None, "local": None}


# ── System prompt builder ──────────────────────────────────

def build_system_prompt(use_examples: bool = True) -> str:
    """
    Build system instructions with or without few-shot examples.

    Parameters
    ----------
    use_examples : bool
        If True, include few-shot examples in the prompt (more tokens).
        If False, rely on schema only (fewer tokens).
    """
    base = f"""
You are a supply chain what-if configuration engine.

Your job is to convert a natural language change instruction into a
structured what-if JSON that describes the exact changes to apply
to a supply chain simulation configuration.

{WHATIF_SCHEMA}
"""
    if use_examples:
        return base + f"\n{WHATIF_EXAMPLES}"
    return base


# ── Prompt builder ─────────────────────────────────────────

def build_prompt(instruction: str, base_config: dict) -> str:
    return f"""
Convert the following what-if instruction into a what-if JSON.

Current supply chain configuration (for context):
{json.dumps(base_config, indent=2)}

What-if instruction:
{instruction}

Return only valid JSON following the schema. No markdown, no explanation.
"""


# ── Core function (importable) ─────────────────────────────

def generate_whatif(
    instruction: str,
    base_config: dict,
    use_examples: bool = False,
) -> dict:
    """
    Convert a natural language what-if instruction to a what-if JSON dict.

    Parameters
    ----------
    instruction : str
        Plain English description of the change to make.
    base_config : dict
        The current validated simulation config for context.
    use_examples : bool
        If True (default), include few-shot examples in the prompt.
        If False, rely on schema only — fewer tokens, faster, less accurate.

    Returns
    -------
    dict
        Parsed what-if JSON with a 'changes' list.
    """
    messages = [
        {"role": "system", "content": build_system_prompt(use_examples)},
        {"role": "user",   "content": build_prompt(instruction, base_config)},
    ]

    provider = resolve_provider()
    raw = chat(
        messages,
        temperature=TEMPERATURE,
        provider=provider,
        model=WHATIF_MODELS.get(provider),
    )

    try:
        return json.loads(extract_json_text(raw))
    except json.JSONDecodeError as e:
        raise ValueError(
            f"LLM output could not be parsed as JSON: {e}\n\nRaw output:\n{raw}"
        )


# ── CLI entry point ────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    from datetime import datetime

    parser = argparse.ArgumentParser(
        description="Convert a what-if instruction to what-if JSON."
    )
    parser.add_argument("config_file", help="Path to the base config JSON file")
    parser.add_argument("output_file", help="Path to save the what-if JSON")
    parser.add_argument(
        "--no-examples",
        action="store_true",
        help="Exclude few-shot examples from prompt (fewer tokens)",
    )
    args = parser.parse_args()

    # ── Load base config ───────────────────────────────────
    with open(args.config_file, encoding="utf-8") as f:
        base_config = json.load(f)

    # ── Interactive instruction input ──────────────────────
    print("\n" + "=" * 60)
    print("NL2Sim What-If Instruction")
    print("=" * 60)
    print("Describe the change you want to make to the supply chain.")
    print()
    print("⚠️  IMPORTANT — Be as specific as possible:")
    print("   • Always mention exact names of entities you want to change")
    print("     e.g. 'Supplier A' not just 'the supplier'")
    print("     e.g. 'Power Management IC' not just 'the product'")
    print("   • If you do not specify a name, the change may be")
    print("     applied to ALL suppliers / products / customers")
    print("   • For distribution changes, mention the distribution")
    print("     type and parameters explicitly")
    print("     e.g. 'uniform distribution with min 2 and max 5'")
    print()
    print("Examples of GOOD instructions:")
    print("   ✅ increase supplier capacity of Photronics Mask Corp to 500")
    print("   ✅ change lead time of Supplier A to uniform with min 2 max 6")
    print("   ✅ remove customer NXP Semiconductors")
    print("   ✅ add a new supplier called SupCo for High-K Dielectric Precursor")
    print()
    print("Examples of VAGUE instructions (avoid):")
    print("   ❌ increase supplier capacity")
    print("   ❌ change the lead time")
    print("   ❌ remove a customer")
    print()
    print("Press Enter on an empty line when done.\n")

    lines = []
    while True:
        line = input()
        if line == "" and lines and lines[-1] == "":
            break
        lines.append(line)

    instruction = "\n".join(lines).strip()

    if not instruction:
        print("No instruction entered. Exiting.")
        raise SystemExit

    print(f"\nInstruction received:\n{instruction}")

    # ── Save instruction to tracking file ─────────────────
    timestamp    = datetime.now().strftime("%Y%m%d_%H%M%S")
    tracking_dir = Path(args.config_file).parent / "whatif_instructions"
    tracking_dir.mkdir(exist_ok=True)

    tracking_file = tracking_dir / f"instruction_{timestamp}.txt"
    tracking_file.write_text(
        f"Timestamp  : {timestamp}\n"
        f"Config     : {args.config_file}\n"
        f"Output     : {args.output_file}\n"
        f"Examples   : {'yes' if not args.no_examples else 'no'}\n\n"
        f"Instruction:\n{instruction}\n",
        encoding="utf-8",
    )
    print(f"Instruction saved → {tracking_file}")

    # ── Generate what-if JSON ──────────────────────────────
    # ── Ask user about examples ────────────────────────────
    print("\n" + "─" * 60)
    print("Few-shot examples improve accuracy but use more tokens.")
    print("Recommended: Yes for complex structural changes,")
    print("             No for simple single field updates.")
    print("─" * 60)
    examples_choice = input("Include examples? (yes/no) [default: yes]: ").strip().lower()

    if examples_choice in {"no", "n"}:
        use_examples = False
        print("Examples excluded — using schema only.")
    else:
        use_examples = True
        print("Examples included — higher accuracy.")

    print(f"\nModel    : {MODEL}")
    print(f"Examples : {'yes' if use_examples else 'no'}")
    print("Generating what-if JSON...\n")

    result = generate_whatif(instruction, base_config, use_examples=use_examples)

    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))
    print(f"\nSaved → {args.output_file}")

    # ── Apply what-if to main config ──────────────────────
    print("\n" + "=" * 60)
    print("Applying what-if changes to main config...")
    print("=" * 60)

    from what_if_engine import apply_what_if_config, WhatIfError
    from repair_orchestrator   import run_repair_loop, canonicalize_config, deep_sort
    from copy import deepcopy

    try:
        modified = apply_what_if_config(deepcopy(base_config), result)
        print("Changes applied successfully.")
    except WhatIfError as e:
        print(f"Failed to apply changes: {e}")
        raise SystemExit

    # ── Validate + repair modified config ─────────────────
    print("\nRunning validation and repair on modified config...")
    final, remaining = run_repair_loop(modified, max_iterations=60, verbose=True)
    if remaining:
        print(f"\n{len(remaining)} issue(s) could not be fully resolved:")
        for issue in remaining:
            print("   ", issue)
    final = deep_sort(canonicalize_config(final))

    # ── Save final modified config ─────────────────────────
    print("\nDo you want to save the final modified config?")
    print("  1) Yes")
    print("  2) No")
    save_choice = input("Select option #: ").strip().lower()

    if save_choice in {"1", "yes", "y", ""}:
        # Auto-generate final output path from input config name
        input_stem   = Path(args.config_file).stem
        final_path   = Path(args.config_file).parent / f"{input_stem}_whatif_final.json"

        with open(final_path, "w", encoding="utf-8") as f:
            json.dump(final, f, indent=2)
        print(f"Final config saved → {final_path}")
    else:
        print("Final config not saved.")