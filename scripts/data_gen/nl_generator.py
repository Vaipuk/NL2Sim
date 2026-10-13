"""
scripts/data_gen/nl_generator.py
----------------------------------
Generates human-like natural language descriptions from
filtered supply chain JSON configs using an LLM.

Takes a filtered JSON (from filter_config.py) and returns
a natural language description that covers only the relevant
fields — matching how real users describe supply chains.

Usage:
    from data_gen.nl_generator import generate_nl
    nl = generate_nl(filtered_config)

    # CLI
    python nl_generator.py filtered_config.json
    python nl_generator.py filtered_config.json --output description.txt
"""

from __future__ import annotations

import os
import re
import sys
import json
import argparse
from pathlib import Path
from typing import Any, Dict, Optional

from openai import OpenAI
from dotenv import load_dotenv

# ── ensure scripts/ is on path ─────────────────────────────
SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


from context_prompt import RECONSTRUCTION_SYSTEM_INSTRUCTIONS

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

# ============================================================
# Model config
# ============================================================

# MODEL       = "gpt-4.1-2025-04-14"
MODEL       = "gpt-5.4"
TEMPERATURE = 0.9   # high temperature for varied human-like output

# ============================================================
# Lazy client
# ============================================================

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from llm_client import chat, resolve_provider  # noqa: E402

# ============================================================
# Prompt builder
# ============================================================

def _build_prompt(filtered_config: Dict[str, Any]) -> str:
    """
    Build the user prompt from a filtered config.
    The filtered config already has irrelevant fields removed
    so the LLM only describes what is present.
    """
    config_str = json.dumps(filtered_config, indent=2)
    return f"""Generate a natural language description for the following supply chain configuration.

{config_str}"""

# ============================================================
# Core function
# ============================================================

def generate_nl(
    filtered_config: Dict[str, Any],
    model: str = MODEL,
    temperature: float = TEMPERATURE,
) -> str:
    """
    Generate a human-like NL description from a filtered config.

    Parameters
    ----------
    filtered_config : dict
        Filtered JSON config from filter_config.py
        Only relevant fields should be present.
    model : str
        OpenAI model to use.
    temperature : float
        Sampling temperature — higher = more varied output.

    Returns
    -------
    str
        Natural language description of the supply chain.
    """
    messages = [
        {
            "role":    "system",
            "content": RECONSTRUCTION_SYSTEM_INSTRUCTIONS,
        },
        {
            "role":    "user",
            "content": _build_prompt(filtered_config),
        },
    ]

    provider = resolve_provider()
    # `model` names an OpenAI model; other providers use their configured one.
    return chat(
        messages,
        temperature=temperature,
        provider=provider,
        model=model if provider == "openai" else None,
    )

# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate NL description from a filtered JSON config."
    )
    parser.add_argument(
        "input_file",
        help="Path to filtered JSON config file"
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Path to save NL description (default: auto-saved next to input)"
    )
    parser.add_argument(
        "--model", "-m",
        default=MODEL,
        help=f"OpenAI model to use (default: {MODEL})"
    )
    parser.add_argument(
        "--temperature", "-t",
        type=float,
        default=TEMPERATURE,
        help=f"Sampling temperature (default: {TEMPERATURE})"
    )
    args = parser.parse_args()

    input_path = Path(args.input_file)

    with open(input_path, encoding="utf-8") as f:
        filtered_config = json.load(f)

    print(f"Generating NL description...")
    print(f"  Model       : {args.model}")
    print(f"  Temperature : {args.temperature}")
    print(f"  Input       : {input_path}\n")

    nl = generate_nl(
        filtered_config,
        model=args.model,
        temperature=args.temperature,
    )

    # ── auto-save next to input file ──────────────────────
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.parent / (input_path.stem + "_nl.txt")

    output_path.write_text(nl, encoding="utf-8")

    # ── print to terminal ──────────────────────────────────
    print("=" * 60)
    print("GENERATED DESCRIPTION")
    print("=" * 60)
    print(nl)
    print("=" * 60)
    print(f"\n  ✓ Description saved → {output_path}")