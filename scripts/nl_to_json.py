"""
scripts/nl_to_json.py
----------------------
Inference script — converts a natural language supply chain
description into a structured JSON configuration.

Usage (standalone):
    python nl_to_json.py description.txt output.json
    python nl_to_json.py description.txt output.json --no-context
    python nl_to_json.py description.txt output.json --azure

Usage (imported):
    from nl_to_json import generate_json
    result = generate_json(description, use_context=True, use_azure=False)
"""

import os
import re
import json
import sys
from pathlib import Path
from dotenv import load_dotenv

from prompts import SYSTEM_INSTRUCTIONS
from schema  import SCHEMA_EXAMPLE
from llm_client import chat, active_model, extract_json_text

load_dotenv(Path(__file__).resolve().parents[1] / ".env")


# ============================================================
# Model config
# ============================================================
TEMPERATURE = 0.3


# ============================================================
# Prompt builder
# ============================================================

def build_prompt(description: str) -> str:
    return f"""
Convert the following supply chain description into JSON structured representation following adhere to provided schema and atleast one edge should be connecting the node customer and node warehouse sharing the same product.:

{SCHEMA_EXAMPLE}

Place "missing" if any information is not provided. Do not assume any information.

RULE — Connect the customer to their product's source:
Every customer[] entry MUST have a corresponding edge whose destination
is that customer's name, and whose source is whichever facility actually
delivers their product — a warehouse if one manages that product,
otherwise the manufacturing facility that produces it directly.

RULE — Production rate → operation_cycle:
When the scenario states a production RATE (e.g. "X units per day",
"produces X per hour", "throughput of X/day"), do NOT put X into
operation_cycle directly. operation_cycle represents the TIME to
produce ONE unit, not a quantity — there is no separate batch-size
field on operation.

Transform: operation_cycle.parameters.a = 1 / X, expressed in the
SAME time unit as simulation.time_unit.

Use "constant" as the distribution unless the description explicitly
signals variability (e.g. "on average", "varies", "roughly") — a
flatly stated rate like "5 units per day" is deterministic, not stochastic.

Example:
  Input: "production is 5 units per day" (simulation.time_unit = "days")
  Output: operation.operation_cycle = {{
    "distribution": "constant",
    "parameters": {{"a": 0.2}}
  }}
  (i.e. 1/5 = 0.2 days per unit)

RULE — Demand rate → arrival_time + demand (contrast case):
When the scenario states a demand/order RATE (e.g. "demand is 2 units
per day"), this DOES map directly: customer.arrival_time = 1 (one
order event per day) and customer.demand = 2 (quantity per event).
Do NOT apply the inverse-rate transformation here — that's only for
production/operation_cycle.



Supply chain description:
{description}
"""


# ============================================================
# Core function (importable by pipeline)
# ============================================================

def generate_json(
    description: str,
    use_context: bool = True,
    use_azure:   bool = False,
    provider:    str  = None,
) -> dict:
    """
    Convert a natural language supply chain description to JSON.

    Parameters
    ----------
    description : str
        Natural language supply chain description.
    use_context : bool
        If True, include SYSTEM_INSTRUCTIONS with full schema rules.
        If False, send no system message (fewer tokens).
    use_azure : bool
        If True, use Azure. Kept for existing callers; prefer `provider`.
    provider : str
        "openai" | "azure" | "local". Defaults to $LLM_PROVIDER, then "openai".
    """
    system_message = SYSTEM_INSTRUCTIONS if use_context else ""

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user",   "content": build_prompt(description)},
    ]

    raw = chat(
        messages,
        temperature=TEMPERATURE,
        provider=provider,
        use_azure=use_azure,
    )

    # ── Parse JSON ─────────────────────────────────────────
    cleaned = extract_json_text(raw)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"LLM output could not be parsed as JSON: {e}\n\nRaw output:\n{raw}"
        )


# ============================================================
# Helpers
# ============================================================

def read_description(filepath: str) -> str:
    return Path(filepath).read_text(encoding="utf-8")


def save_json(data: dict, filepath: str) -> None:
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"Saved → {filepath}")


# ============================================================
# CLI entry point
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert a supply chain description to JSON."
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "description_file",
        nargs="?",
        help="Path to .txt description file",
    )
    group.add_argument(
        "--text", "-t",
        help="Supply chain description as a direct string",
    )

    parser.add_argument("output_file", help="Path to save output .json file")
    parser.add_argument(
        "--no-context",
        action="store_true",
        help="Exclude system instructions (fewer tokens)",
    )
    parser.add_argument(
        "--azure",
        action="store_true",
        help="Use Azure OpenAI fine-tuned model instead of OpenAI",
    )

    args = parser.parse_args()

    description = args.text if args.text else read_description(args.description_file)
    use_context = not args.no_context
    use_azure   = args.azure

    print(f"Model       : {active_model(use_azure=use_azure)}")
    print(f"Temperature : {TEMPERATURE}")
    print(f"Context     : {'yes' if use_context else 'no'}")
    print("Generating...\n")

    result = generate_json(description, use_context=use_context, use_azure=use_azure)
    save_json(result, args.output_file)