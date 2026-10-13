"""
scripts/score_reliability.py
------------------------------
Computes a reliability score for LLM-generated supply chain JSON
by reconstructing natural language from the JSON and comparing
it against the original user description.

Two metrics:
  - BLEU-4       : lexical similarity between original and reconstructed NL
  - Numeric F1   : numeric value fidelity between original and reconstructed NL

Combined reliability score = 0.4 * BLEU-4 + 0.6 * Numeric F1

Usage (standalone):
    python score_reliability.py description.txt output.json
    python score_reliability.py description.txt output.json --azure

Usage (imported):
    from score_reliability import compute_reliability_score
    result = compute_reliability_score(original_nl, generated_json)
    result = compute_reliability_score(original_nl, generated_json, use_azure=True)
"""

import os
import re
import json
import copy
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

# ── nltk imports ───────────────────────────────────────────
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
from word2number import w2n

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

# Download required nltk data silently
for pkg in ["punkt", "punkt_tab", "averaged_perceptron_tagger",
            "averaged_perceptron_tagger_eng"]:
    nltk.download(pkg, quiet=True)


# ============================================================
# API clients
# ============================================================

from llm_client import chat, resolve_provider


# ============================================================
# Reconstruction model config
#
# Reconstruction wants a general model, not the fine-tuned one -- the
# fine-tune only knows description -> JSON, not the reverse. For a local
# provider there is just the one served model, so no override is needed.
# ============================================================

OPENAI_RECONSTRUCTION_MODEL = "gpt-4o"
AZURE_RECONSTRUCTION_MODEL  = os.environ.get("AZURE_BASE_MODEL", "gpt-4.1-2025-04-14")

RECONSTRUCTION_MODELS = {
    "openai": OPENAI_RECONSTRUCTION_MODEL,
    "azure":  AZURE_RECONSTRUCTION_MODEL,
    "local":  None,
}


# ============================================================
# System prompt for reconstruction
# ============================================================

RECONSTRUCTION_SYSTEM_INSTRUCTIONS = """
You are a supply chain description writer. Take some liberty in describing
the supply chain in human style but make sure all the information is covered.

Your task is to convert a supply chain JSON configuration into a clear and
precise natural language description.

Follow these rules strictly:

Config Info:
- Describe the name of the supply chain scenario only.
  Do not mention the version number.

Raw Materials:
- List and describe all raw materials by name.

Intermediate Materials:
- List and describe all intermediate materials by name.
- For each intermediate material describe its bill of materials.

Products:
- List and describe all products by name.
- For each product describe its bill of materials.

Inventory:
- Describe inventory information for each material and product.
- ONLY describe procurement scheme for raw materials.
  Do not mention procurement information for intermediate materials and products.
- For raw materials describe procurement type, distribution, parameters,
  procurement arrival (for periodic_supply only), and initial inventory.
- Describe inventory costs only if holding_cost or shortage_cost are present.
  Do not mention review_time.

Supplier:
- Describe each supplier with name, material supplied, lead time distribution
  and parameters, supplier cost, and payment lead time.
- Only mention supplier capacity if present in the JSON.

Resource:
- If no resources are present do not mention anything about it.

Facility:
- Describe each facility with name and type.
- For manufacturing facilities describe inventory managed and operation details.
- For warehouse facilities state it is a storage facility.

Customer:
- Describe each customer with name, product ordered, arrival time, demand,
  customer lead time, shortage policy in plain language, unit selling price,
  and customer payment lead time.

Nodes:
- List all supplier and facility nodes by name.

Edges:
- For each edge describe source, destination, material name and type.
  Describe transfer time only if present in the JSON.

Simulation Parameters:
- Describe only time unit and horizon.
  Do not mention warm up, replications, or random seed.

General rules:
1. Only describe fields that are present in the JSON.
2. For distributions use plain language:
   - constant a     → fixed value of a
   - uniform a,b    → ranges between a and b
   - normal a,b     → mean of a with standard deviation of b
   - exponential a  → exponentially distributed with mean of a
   - triangular a,b,c → minimum a, most likely b, maximum c
3. Write in clear flowing prose.
4. Be precise with all numbers.
5. If a field contains MISSING state that information was not provided.
"""


# ============================================================
# Tier 3 field stripping
# ============================================================

def _is_zero_constant(block: dict) -> bool:
    return (
        isinstance(block, dict)
        and block.get("distribution") == "constant"
        and block.get("parameters", {}).get("a", None) == 0
    )


def _clean_parameters(block: dict) -> dict:
    if not block:
        return block
    dist   = block.get("distribution", "")
    params = block.get("parameters", {})
    keep   = {
        "constant":    ["a"],
        "uniform":     ["a", "b"],
        "normal":      ["a", "b"],
        "exponential": ["a"],
        "triangular":  ["a", "b", "c"],
    }
    keys_to_keep        = keep.get(dist, list(params.keys()))
    block["parameters"] = {k: v for k, v in params.items()
                           if k in keys_to_keep}
    return block


def strip_tier3_fields(json_data: dict) -> dict:
    data = copy.deepcopy(json_data)

    for item in data.get("config_info", []):
        item.pop("version", None)

    for item in data.get("inventory", []):
        costs = item.get("inventory_costs", {})
        costs.pop("review_time", None)
        if costs.get("holding_cost")  == 0: costs.pop("holding_cost",  None)
        if costs.get("shortage_cost") == 0: costs.pop("shortage_cost", None)

        if item.get("type") in ("intermediate_materials", "products"):
            item.pop("procurement_scheme",  None)
            item.pop("procurement_arrival", None)
        else:
            ps = item.get("procurement_scheme", {})
            if ps: item["procurement_scheme"] = _clean_parameters(ps)
            pa = item.get("procurement_arrival", {})
            if pa: item["procurement_arrival"] = _clean_parameters(pa)

    for s in data.get("supplier", []):
        if s.get("supplier_capacity") == 99999:
            s.pop("supplier_capacity", None)
        s["supplier_lead_time"]         = _clean_parameters(s.get("supplier_lead_time", {}))
        s["supplier_payment_lead_time"] = _clean_parameters(s.get("supplier_payment_lead_time", {}))

    for r in data.get("resource", []):
        if r.get("capacity")               == 99999: r.pop("capacity", None)
        if r.get("operating_cost_per_time") == 0:    r.pop("operating_cost_per_time", None)
        failure = r.get("failure", {})
        if failure and not failure.get("enabled", False):
            r.pop("failure", None)
        batching = r.get("batching", {})
        if batching.get("batch_size")    == -1: batching["batch_size"]   = "no_minimum"
        if batching.get("max_wait_time") == 0:  batching.pop("max_wait_time", None)
        st = r.get("service_time", {})
        if _is_zero_constant(st):  r.pop("service_time", None)
        elif st:                   r["service_time"] = _clean_parameters(st)

    for f in data.get("facility", []):
        op = f.get("operation", {})
        if op:
            oc = op.get("operation_cycle", {})
            if _is_zero_constant(oc): op.pop("operation_cycle", None)
            elif oc:                  op["operation_cycle"] = _clean_parameters(oc)

    for c in data.get("customer", []):
        for field in ["arrival_time", "demand",
                      "customer_lead_time", "customer_payment_lead_time"]:
            if field in c:
                c[field] = _clean_parameters(c[field])

    for e in data.get("edges", []):
        tt = e.get("transfer_time", {})
        if _is_zero_constant(tt): e.pop("transfer_time", None)
        elif tt:                  e["transfer_time"] = _clean_parameters(tt)

    sim = data.get("simulation", {})
    sim.pop("random_seed", None)

    return data


# ============================================================
# NL reconstruction
# ============================================================

def reconstruct_nl(json_data: dict, use_azure: bool = False) -> str:
    """
    Reconstruct natural language from a generated JSON config.
    Strips Tier 3 fields first, then calls the appropriate LLM.

    Parameters
    ----------
    json_data : dict
        Generated JSON config.
    use_azure : bool
        If True, use Azure OpenAI. If False (default), use OpenAI.
    """
    cleaned = strip_tier3_fields(json_data)

    messages = [
        {"role": "system", "content": RECONSTRUCTION_SYSTEM_INSTRUCTIONS},
        {"role": "user",   "content": (
            f"Convert the following supply chain JSON into natural language:\n\n"
            f"{json.dumps(cleaned, indent=2)}"
        )},
    ]

    provider = resolve_provider(use_azure=use_azure)
    model    = RECONSTRUCTION_MODELS.get(provider)
    print(f"  [{provider}] Reconstruction model: {model or 'configured default'}")

    return chat(
        messages,
        temperature=0.3,
        provider=provider,
        model=model,
    )


# ============================================================
# Number normalization
# ============================================================

def _normalize_number_formats(text: str) -> str:
    magnitude_map = {
        "hundred": 100, "thousand": 1_000, "million": 1_000_000,
        "billion": 1_000_000_000, "trillion": 1_000_000_000_000,
        "k": 1_000, "m": 1_000_000, "b": 1_000_000_000,
    }

    def replace_magnitude(match):
        number_part = float(match.group(1).replace(",", ""))
        magnitude   = match.group(2).lower().rstrip("s")
        multiplier  = magnitude_map.get(magnitude, 1)
        result      = number_part * multiplier
        return str(int(result)) if result == int(result) else str(result)

    text = re.sub(
        r"(\d[\d,]*(?:\.\d+)?)\s*(hundred|thousand|million|billion|trillion|k|m|b)s?\b",
        replace_magnitude, text, flags=re.IGNORECASE
    )
    text = re.sub(
        r"\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b",
        lambda m: m.group(0).replace(",", ""),
        text
    )
    return text


def _normalize_informal_quantities(text: str) -> str:
    informal = {
        r"\ba\s+dozen\b": "12",
        r"\ba\s+couple\b": "2",
        r"\ba\s+few\b": "3",
        r"\bhalf\b": "0.5",
    }
    for pattern, value in informal.items():
        text = re.sub(pattern, value, text, flags=re.IGNORECASE)
    return text


def _convert_number_words(text: str) -> str:
    tokens   = nltk.word_tokenize(text.lower())
    pos_tags = nltk.pos_tag(tokens)
    result   = []
    i        = 0

    while i < len(pos_tags):
        word, tag = pos_tags[i]
        if tag == "CD":
            span = []
            j    = i
            while j < len(pos_tags) and pos_tags[j][1] == "CD":
                span.append(pos_tags[j][0])
                j += 1
            phrase    = " ".join(span)
            converted = False
            try:
                result.append(str(w2n.word_to_num(phrase)))
                i         = j
                converted = True
            except ValueError:
                for end in range(len(span), 0, -1):
                    try:
                        result.append(str(w2n.word_to_num(" ".join(span[:end]))))
                        i        += end
                        converted = True
                        break
                    except ValueError:
                        continue
            if not converted:
                result.append(word)
                i += 1
        else:
            result.append(word)
            i += 1

    return " ".join(result)


def preprocess(text: str) -> str:
    text = _normalize_number_formats(text)
    text = _normalize_informal_quantities(text)
    text = _convert_number_words(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ============================================================
# Scoring
# ============================================================

def _extract_numbers(text: str) -> list:
    return [float(n) for n in re.findall(r"\b\d+(?:\.\d+)?\b", text)]


def _compute_bleu(ref: str, hyp: str) -> dict:
    ref_tokens = word_tokenize(ref.lower())
    hyp_tokens = word_tokenize(hyp.lower())
    smoothing  = SmoothingFunction().method1

    return {
        "bleu_1": round(sentence_bleu([ref_tokens], hyp_tokens,
                        weights=(1, 0, 0, 0), smoothing_function=smoothing), 4),
        "bleu_2": round(sentence_bleu([ref_tokens], hyp_tokens,
                        weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing), 4),
        "bleu_4": round(sentence_bleu([ref_tokens], hyp_tokens,
                        weights=(0.25, 0.25, 0.25, 0.25),
                        smoothing_function=smoothing), 4),
    }


def _compute_numeric_f1(ref: str, hyp: str, tolerance: float = 0.05) -> dict:
    nums_a    = _extract_numbers(ref)
    nums_b    = _extract_numbers(hyp)
    matched_a = [False] * len(nums_a)
    matched_b = [False] * len(nums_b)
    tp        = 0

    for i, a in enumerate(nums_a):
        for j, b in enumerate(nums_b):
            if not matched_b[j]:
                if abs(a - b) <= tolerance * max(abs(a), 1):
                    tp           += 1
                    matched_a[i]  = True
                    matched_b[j]  = True
                    break

    fp        = sum(1 for m in matched_b if not m)
    fn        = sum(1 for m in matched_a if not m)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall) / (precision + recall) \
                if (precision + recall) > 0 else 0.0

    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision":  round(precision, 4),
        "recall":     round(recall,    4),
        "numeric_f1": round(f1,        4),
    }


def _diagnose(bleu_4: float, numeric_f1: float) -> str:
    if bleu_4 >= 0.5 and numeric_f1 >= 0.8:
        return "Strong — high lexical and numeric fidelity."
    elif bleu_4 < 0.4 and numeric_f1 >= 0.8:
        return "Style divergence — numeric values correct but surface form differs."
    elif bleu_4 < 0.4 and numeric_f1 < 0.5:
        return "Poor — both lexical and numeric fidelity are low."
    elif bleu_4 >= 0.4 and numeric_f1 < 0.5:
        return "Numeric extraction failing — structure transferred but values are wrong."
    else:
        return "Partial — mixed signals, review individual scores."


# ============================================================
# Public API
# ============================================================

def compute_reliability_score(
    original_nl: str,
    generated_json: dict,
    use_azure: bool = False,
) -> dict:
    """
    Compute reliability score for a generated JSON config.

    Parameters
    ----------
    original_nl : str
        The original natural language description provided by the user.
    generated_json : dict
        The JSON config generated by the LLM.
    use_azure : bool
        If True, use Azure OpenAI for reconstruction. Default: False.

    Returns
    -------
    dict with keys:
        reconstructed_nl  : str    — NL reconstructed from the JSON
        bleu_1            : float  — BLEU-1 score
        bleu_2            : float  — BLEU-2 score
        bleu_4            : float  — BLEU-4 score
        numeric_f1        : float  — Numeric F1 score
        reliability_score : float  — Combined score (0–1)
        diagnosis         : str    — Plain English interpretation
    """
    # Step 1 — reconstruct NL from generated JSON
    print("  Reconstructing NL from JSON...")
    reconstructed_nl = reconstruct_nl(generated_json, use_azure=use_azure)

    # Step 2 — preprocess both texts
    ref = preprocess(original_nl)
    hyp = preprocess(reconstructed_nl)

    # Step 3 — compute scores
    bleu    = _compute_bleu(ref, hyp)
    numeric = _compute_numeric_f1(ref, hyp)

    # Step 4 — combined reliability score
    reliability = round(
        0.4 * bleu["bleu_4"] + 0.6 * numeric["numeric_f1"], 4
    )

    return {
        "reconstructed_nl":  reconstructed_nl,
        "bleu_1":            bleu["bleu_1"],
        "bleu_2":            bleu["bleu_2"],
        "bleu_4":            bleu["bleu_4"],
        "numeric_f1":        numeric["numeric_f1"],
        "numeric_precision": numeric["precision"],
        "numeric_recall":    numeric["recall"],
        "reliability_score": reliability,
        "diagnosis":         _diagnose(bleu["bleu_4"], numeric["numeric_f1"]),
    }


def print_score_report(result: dict) -> None:
    """Print a formatted reliability score report."""
    print("\n" + "=" * 55)
    print("RELIABILITY SCORE REPORT")
    print("=" * 55)
    print(f"  Reliability Score : {result['reliability_score']:.2%}")
    print(f"  BLEU-4            : {result['bleu_4']:.4f}")
    print(f"  Numeric F1        : {result['numeric_f1']:.4f}")
    print(f"  Numeric Precision : {result['numeric_precision']:.4f}")
    print(f"  Numeric Recall    : {result['numeric_recall']:.4f}")
    print(f"  Diagnosis         : {result['diagnosis']}")
    print("=" * 55)


# ============================================================
# CLI entry point
# ============================================================

if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(
        description="Compute reliability score for a generated supply chain JSON."
    )
    parser.add_argument("description_file", help="Path to original NL description .txt")
    parser.add_argument("json_file",        help="Path to generated JSON config")
    parser.add_argument(
        "--azure",
        action="store_true",
        help="Use Azure OpenAI for NL reconstruction instead of OpenAI",
    )
    args = parser.parse_args()

    original_nl    = Path(args.description_file).read_text(encoding="utf-8").strip()
    generated_json = json.loads(Path(args.json_file).read_text(encoding="utf-8"))

    print(f"Computing reliability score ({'Azure' if args.azure else 'OpenAI'})...")
    result = compute_reliability_score(
        original_nl, generated_json, use_azure=args.azure)

    print_score_report(result)

    print("\nReconstructed NL:")
    print(result["reconstructed_nl"])