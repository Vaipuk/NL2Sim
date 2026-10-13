"""
merge.py

Currently scoped to ONE thing: collect node names from the graph-derived
topology JSON and the NL-derived config JSON, normalize both (surface-form
only -- case/plural/whitespace, via one LLM call, no semantic matching),
and report precision/recall/F-1 between the two node sets.

This does NOT yet do node dispute resolution, edge merging, or write any
output file. It's a diagnostic step: run it, see the F-1 score and exactly
which names are unmatched on each side, before deciding what comes next.

Usage:
    python merge.py --graph output/sketch_topology.json --nl output/nl_config.json
    python merge.py --graph output/sketch_topology.json --nl output/nl_config.json --model gpt-4.1

Requires:
    pip install openai --break-system-packages
    export OPENAI_API_KEY=sk-...
"""

import argparse
import difflib
import json
import os
import re
import sys

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI

DEFAULT_MODEL = "gpt-4.1"
NO_TEMPERATURE_ZERO_MODELS = {"gpt-5.6-terra"}

MISSING_DIST = {"distribution": "missing", "parameters": {"a": "missing", "b": "missing", "c": "missing", "d": "missing", "e": "missing"}}


def stub_supplier(name):
    return {"name": name, "supplier_capacity": "missing", "supplier_cost": "missing",
            "supplier_lead_time": dict(MISSING_DIST), "supplier_payment_lead_time": dict(MISSING_DIST),
            "supply_material_name": "missing"}


def stub_facility(name):
    return {"inventory_managed": [], "name": name,
            "operation": {"input": [], "name": "missing", "operation_cycle": dict(MISSING_DIST), "output": [], "resource_required": "missing"},
            "type": "missing"}


def stub_customer(name):
    return {"arrival_time": dict(MISSING_DIST), "customer_lead_time": dict(MISSING_DIST),
            "customer_payment_lead_time": dict(MISSING_DIST), "demand": dict(MISSING_DIST),
            "name": name, "product": "missing", "shortage_policy": "missing", "unit_selling_price": "missing"}


STUB_BUILDERS = {"supplier": stub_supplier, "facility": stub_facility, "customer": stub_customer}

NODE_CATEGORIES = ("customer", "facility", "supplier")


def load_json(path: str) -> dict:
    if not os.path.exists(path):
        print(f"Error: file not found at {path}", file=sys.stderr)
        sys.exit(1)
    with open(path, "r") as f:
        return json.load(f)


def get_node_names(config: dict, category: str) -> list:
    nodes = config.get("nodes", [])
    if not nodes:
        return []
    return list(nodes[0].get(category, []))


def collect_nodes(config: dict) -> list:
    """Returns a flat list of (name, category) tuples across all three categories."""
    result = []
    for cat in NODE_CATEGORIES:
        for name in get_node_names(config, cat):
            result.append((name, cat))
    return result


def normalize_name(name: str) -> str:
    """Deterministic, offline surface-form normalization -- no LLM call.
    Lowercase, strip ALL whitespace, strip trailing punctuation, naive
    singularize. Purely mechanical: does not attempt semantic matching."""
    x = name.lower().strip()
    x = re.sub(r"[.,]", "", x)      # strip periods/commas anywhere
    x = re.sub(r"\s+", "", x)       # remove all whitespace between words
    if x.endswith("s") and not x.endswith("ss"):
        x = x[:-1]                  # crude singularize
    return x


def normalize_names(names: list) -> dict:
    """Returns {original_name: normalized_name} for a list of names."""
    return {n: normalize_name(n) for n in names}


def similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()


def fuzzy_resolve(graph_only: list, nl_only: list, norm_map_graph: dict, norm_map_nl: dict, threshold: float = 0.6) -> tuple:
    """Phase 2 node matching: for names that didn't exact-match after
    normalization, compute string similarity (difflib, cheap, offline) on
    the normalized forms and greedily auto-match pairs above threshold,
    highest-similarity pairs claimed first so one name can't be grabbed by
    a worse match before a better one is considered.

    graph_only / nl_only: lists of (name, category) tuples.
    Returns (auto_matched, still_graph_only, still_nl_only).
    auto_matched: list of (graph_name, nl_name, similarity_score).
    """
    candidates = []
    for gname, gcat in graph_only:
        gnorm = norm_map_graph.get(gname, gname)
        for nname, ncat in nl_only:
            nnorm = norm_map_nl.get(nname, nname)
            score = similarity(gnorm, nnorm)
            if score >= threshold:
                candidates.append((score, gname, nname))

    candidates.sort(key=lambda x: x[0], reverse=True)

    used_graph, used_nl = set(), set()
    auto_matched = []
    for score, gname, nname in candidates:
        if gname in used_graph or nname in used_nl:
            continue
        auto_matched.append((gname, nname, score))
        used_graph.add(gname)
        used_nl.add(nname)

    still_graph_only = [(n, c) for n, c in graph_only if n not in used_graph]
    still_nl_only = [(n, c) for n, c in nl_only if n not in used_nl]

    return auto_matched, still_graph_only, still_nl_only


def compute_f1(nl_norm_set: set, graph_norm_set: set) -> dict:
    tp = len(nl_norm_set & graph_norm_set)
    fp = len(graph_norm_set - nl_norm_set)   # graph-only
    fn = len(nl_norm_set - graph_norm_set)   # nl-only
    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 1.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def _llm_call(system_prompt: str, user_content: str, model: str) -> dict:
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from llm_client import chat, resolve_provider

    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_content}]
    provider = resolve_provider()
    # `model` names an OpenAI model; other providers use their configured one.
    if provider != "openai":
        model = None
    temperature = None if model in NO_TEMPERATURE_ZERO_MODELS else 0
    raw_text = chat(messages, provider=provider, model=model, temperature=temperature)
    if raw_text.startswith("```"):
        raw_text = raw_text.strip("`")
        if raw_text.startswith("json"):
            raw_text = raw_text[4:].strip()
    return json.loads(raw_text)


# ---------------------------------------------------------------------------
# Phase 3: natural-language dispute dialogue for whatever phases 1+2 leave
# unresolved. One LLM call writes the questions, you answer freely at the
# terminal, one LLM call interprets your answers into structured decisions,
# which get applied directly to the NL JSON.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Phase 3: ONE combined natural-language dispute round for everything phases
# 1+2 leave unmatched (both graph-side and NL-side at once). One question,
# one free-text answer, one interpretation call producing a decision per
# entity. After applying decisions, the caller re-runs phases 1+2 on the
# updated data -- corrections may create new matches automatically, so
# nothing gets asked about twice.
# ---------------------------------------------------------------------------

def build_combined_question(graph_only: list, nl_only: list) -> str:
    """Deterministic, code-authored -- lists every unmatched entity from
    both sides, asks for plain clarification with no explanation of the
    underlying match/new/keep/delete taxonomy. The interpretation step
    figures that out from whatever the user says."""
    lines = []
    lines.append("Your sketch and description don't fully agree on these entities:")
    if graph_only:
        lines.append("\nFrom your sketch:")
        for name, cat in graph_only:
            lines.append(f'  - "{name}" ({cat})')
    if nl_only:
        lines.append("\nFrom your description:")
        for name, cat in nl_only:
            lines.append(f'  - "{name}" ({cat})')
    lines.append("\nCan you clarify what's going on with each of these?")
    return "\n".join(lines)


COMBINED_INTERPRET_PROMPT = """You are interpreting ONE free-text answer that addresses multiple \
entity-matching disputes at once, between a hand-drawn sketch (graph-side, names may be garbled) \
and a text description (NL-side, names reliable) of the same supply chain network.

You're given a list of graph-side entities (unmatched) and a list of NL-side entities (unmatched), \
each with their category, plus the user's single free-text answer addressing some or all of them.

For EVERY entity in BOTH lists, output exactly one decision:
- "match": this entity is the same as a specific entity from the OTHER list, AND one of the two \
existing names is already correct -- give matched_name copied exactly from that other list.
- "merge": this entity is the same real thing as a specific entity from the OTHER list, but \
NEITHER existing name is correct -- the user gives a new corrected name for the combined entity. \
Give matched_name (the other list's entity being merged with, copied exactly) AND actual_name \
(the corrected name to use for the merged entity going forward).
- "new": (graph-side only) genuinely new, not described in the NL text -- give actual_name (the \
corrected real name if the sketch's reading was wrong, otherwise the same name as given).
- "keep": (NL-side only) genuinely real and correctly described, simply not drawn in the sketch -- \
keep it in the data, excluded from scoring.
- "delete": remove this entity entirely -- it doesn't belong in the network. Use this whenever the \
user explicitly rules an entity out by name (e.g. "neither X nor Y" means Y should be deleted \
unless the user separately confirms Y is real elsewhere in their answer).
- "unresolved": the answer doesn't address this entity at all -- do not guess.

Return STRICT JSON, no markdown fences, no commentary:
{
  "decisions": [
    {
      "name": "<entity name exactly as given>",
      "side": "graph" or "nl",
      "category": "<supplier|facility|customer>",
      "action": "match"|"merge"|"new"|"keep"|"delete"|"unresolved",
      "matched_name": "<exact name from the OTHER side's list, or null>",
      "actual_name": "<corrected/confirmed name if action is 'new' or 'merge', or null>"
    }
  ]
}

Rules:
- "matched_name" MUST be copied exactly from the opposite side's list, never invented.
- Every entity from both input lists must appear exactly once in "decisions".
- Default to "unresolved" rather than guessing when the answer is ambiguous about an entity.
"""


def collect_combined_answer(graph_only: list, nl_only: list) -> str:
    """Runs at the terminal. Returns the user's single free-text answer."""
    q = build_combined_question(graph_only, nl_only)
    print(f"\n  Q: {q}\n")
    return input("  Your answer: ").strip()


def interpret_combined_answer(graph_only: list, nl_only: list, answer: str, model: str = DEFAULT_MODEL) -> list:
    """Returns list of dicts: {name, side, category, action, matched_name, actual_name}."""
    payload = {
        "graph_side_entities": [{"name": n, "category": c} for n, c in graph_only],
        "nl_side_entities": [{"name": n, "category": c} for n, c in nl_only],
        "answer": answer,
    }
    result = _llm_call(COMBINED_INTERPRET_PROMPT, json.dumps(payload, indent=2), model)
    return result.get("decisions", [])


def rename_in_nl_edges(nl_json: dict, old_name: str, new_name: str) -> int:
    """Propagates an entity rename into nl_json's own edges array -- a
    'merge' rename only updated the entity's declaration, not any
    pre-existing edge referencing it by the old name. Returns count renamed."""
    count = 0
    for e in nl_json.get("edges", []):
        if e.get("source") == old_name:
            e["source"] = new_name
            count += 1
        if e.get("destination") == old_name:
            e["destination"] = new_name
            count += 1
    return count


def remove_nl_edges_referencing(nl_json: dict, name: str) -> int:
    """Drops any edge in nl_json referencing a deleted entity by name,
    on either side -- otherwise deletion leaves dangling edge references
    pointing at an entity that no longer exists. Returns count removed."""
    before = len(nl_json.get("edges", []))
    nl_json["edges"] = [e for e in nl_json.get("edges", [])
                         if e.get("source") != name and e.get("destination") != name]
    return before - len(nl_json["edges"])


def apply_combined_decisions(nl_json: dict, graph_working: dict, decisions: list) -> dict:
    """Mutates nl_json (add new/keep stubs, remove deleted NL entities) and
    graph_working (remove deleted graph entities and any edges referencing
    them). Returns a dict with name_map additions and human-readable log
    entries for printing."""
    if not nl_json.get("nodes"):
        nl_json["nodes"] = [{"customer": [], "facility": [], "supplier": []}]

    name_map = {}
    nl_rename_map = {}   # original_nl_name -> canonical_name (NEW: was untracked before)
    log = []
    deleted_graph_names = set()

    for d in decisions:
        name, side, cat, action = d["name"], d["side"], d.get("category"), d["action"]
        matched_name, actual_name = d.get("matched_name"), d.get("actual_name")

        if action == "match":
            if side == "graph":
                name_map[name] = matched_name
            log.append(f'"{name}" -> matched to "{matched_name}"')

        elif action == "merge":
            final_name = actual_name if actual_name else (matched_name or name)
            if side == "graph":
                name_map[name] = final_name
                # rename the matched NL-side entity to final_name if it differs
                if matched_name and matched_name != final_name:
                    for c in NODE_CATEGORIES:
                        if nl_json.get("nodes") and matched_name in nl_json["nodes"][0].get(c, []):
                            nl_json["nodes"][0][c] = [final_name if x == matched_name else x for x in nl_json["nodes"][0][c]]
                        for entry in nl_json.get(c, []):
                            if entry.get("name") == matched_name:
                                entry["name"] = final_name
                    rename_in_nl_edges(nl_json, matched_name, final_name)
                    nl_rename_map[matched_name] = final_name
            else:
                # nl-side entity being merged; graph-side counterpart aliases via its own decision
                if matched_name is None and name != final_name:
                    for c in NODE_CATEGORIES:
                        if nl_json.get("nodes") and name in nl_json["nodes"][0].get(c, []):
                            nl_json["nodes"][0][c] = [final_name if x == name else x for x in nl_json["nodes"][0][c]]
                        for entry in nl_json.get(c, []):
                            if entry.get("name") == name:
                                entry["name"] = final_name
                    rename_in_nl_edges(nl_json, name, final_name)
                    nl_rename_map[name] = final_name
            log.append(f'"{name}" -> merged with "{matched_name}", corrected name: "{final_name}"')

        elif action == "new" and side == "graph":
            final_name = actual_name if actual_name else name
            name_map[name] = final_name
            nl_json["nodes"][0].setdefault(cat, [])
            if final_name not in nl_json["nodes"][0][cat]:
                nl_json["nodes"][0][cat].append(final_name)
            nl_json.setdefault(cat, [])
            nl_json[cat].append(STUB_BUILDERS[cat](final_name))
            log.append(f'"{name}" -> confirmed NEW entity ({cat}), name: "{final_name}"')

        elif action == "keep" and side == "nl":
            log.append(f'"{name}" -> confirmed real, not drawn in sketch (kept, excluded from scoring)')

        elif action == "delete":
            if side == "nl":
                if nl_json.get("nodes") and name in nl_json["nodes"][0].get(cat, []):
                    nl_json["nodes"][0][cat].remove(name)
                nl_json[cat] = [e for e in nl_json.get(cat, []) if e.get("name") != name]
                removed_edge_count = remove_nl_edges_referencing(nl_json, name)
                edge_note = f", {removed_edge_count} edge(s) referencing it also removed" if removed_edge_count else ""
                log.append(f'"{name}" -> deleted from the network{edge_note}')
            else:
                deleted_graph_names.add(name)
                log.append(f'"{name}" -> deleted (removed from sketch consideration)')

        else:  # unresolved
            log.append(f'"{name}" -> still unresolved, will ask again')

    if deleted_graph_names:
        for cat in NODE_CATEGORIES:
            if graph_working.get("nodes"):
                graph_working["nodes"][0][cat] = [
                    n for n in graph_working["nodes"][0].get(cat, []) if n not in deleted_graph_names
                ]
        graph_working["edges"] = [
            e for e in graph_working.get("edges", [])
            if e.get("source") not in deleted_graph_names and e.get("destination") not in deleted_graph_names
        ]

    nl_only_kept = {d["name"] for d in decisions if d["side"] == "nl" and d["action"] == "keep"}
    # Broader than nl_only_kept: covers every NL-side name that got a real
    # decision this round (keep/merge/match), not just "keep" specifically.
    # Used to stop re-asking about it next round -- distinct from
    # nl_only_kept, which is used later for the final F-1 exclusion and
    # must stay "keep"-only.
    nl_settled = {d["name"] for d in decisions if d["side"] == "nl" and d["action"] != "unresolved"}

    return {"name_map": name_map, "nl_rename_map": nl_rename_map, "log": log,
            "nl_only_kept": nl_only_kept, "nl_settled": nl_settled}


def apply_name_map_to_graph(graph_working: dict, full_name_map: dict) -> int:
    """Physically rewrites graph_working's node names and edge endpoints to
    their canonical (post-resolution) form, using full_name_map. After this,
    graph_working and nl_json are both expressed in the same names, so edge
    comparison no longer needs a lookup indirection -- just normalization.
    Returns count of raw names actually replaced."""
    count = 0
    if graph_working.get("nodes"):
        for cat in NODE_CATEGORIES:
            names = graph_working["nodes"][0].get(cat, [])
            new_names = []
            for n in names:
                canon = full_name_map.get(n, n)
                if canon != n:
                    count += 1
                new_names.append(canon)
            graph_working["nodes"][0][cat] = new_names

    for e in graph_working.get("edges", []):
        src, dst = e.get("source"), e.get("destination")
        e["source"] = full_name_map.get(src, src)
        e["destination"] = full_name_map.get(dst, dst)

    return count


def collect_edges(config: dict) -> list:
    """Returns a flat list of (source, destination) tuples from an edges array."""
    return [(e.get("source"), e.get("destination")) for e in config.get("edges", []) if e.get("source") and e.get("destination")]


def build_entity_context(nl_json: dict, name: str) -> dict:
    """Pulls whatever NL2Sim already extracted about an entity -- material
    flow fields only, nothing about the edge itself. Used as evidence for
    the LLM edge-validity check, not to assign material_name/type (that's
    a later phase)."""
    for s in nl_json.get("supplier", []):
        if s.get("name") == name:
            return {"category": "supplier", "supply_material_name": s.get("supply_material_name")}
    for f in nl_json.get("facility", []):
        if f.get("name") == name:
            op = f.get("operation", {})
            return {"category": "facility", "input": op.get("input", []), "output": op.get("output", []),
                    "inventory_managed": f.get("inventory_managed", [])}
    for c in nl_json.get("customer", []):
        if c.get("name") == name:
            return {"category": "customer", "product": c.get("product")}
    return {"category": "unknown"}


def score_edges(nl_json: dict, graph_json: dict) -> dict:
    """Edge-level F-1. Assumes graph_json's node/edge names have already
    been rewritten to canonical form via apply_name_map_to_graph -- both
    sides just need normalization as a final safety pass before comparing."""
    nl_edges_raw = collect_edges(nl_json)
    graph_edges_raw = collect_edges(graph_json)

    def canon(edge):
        return (normalize_name(edge[0]), normalize_name(edge[1]))

    nl_by_canon = {canon(e): e for e in nl_edges_raw}
    graph_by_canon = {canon(e): e for e in graph_edges_raw}

    nl_set = set(nl_by_canon.keys())
    graph_set = set(graph_by_canon.keys())

    tp = len(nl_set & graph_set)
    fp = len(graph_set - nl_set)
    fn = len(nl_set - graph_set)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 1.0

    matched = [(graph_by_canon[c], nl_by_canon[c]) for c in (nl_set & graph_set)]
    graph_only = [graph_by_canon[c] for c in (graph_set - nl_set)]
    nl_only = [nl_by_canon[c] for c in (nl_set - graph_set)]

    return {
        "metrics": {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1},
        "matched": matched, "graph_only": graph_only, "nl_only": nl_only,
    }


EDGE_EXISTENCE_QUESTION_PREFIX = (
    "Neither source is a reliable list of which edges actually exist on its own. "
    "For each candidate edge below, tell me which ones are real -- this is about "
    "existence only, not materials or attributes (that comes later)."
)

EDGE_EXISTENCE_INTERPRET_PROMPT = """You are interpreting a user's free-text answer about which \
candidate edges in a supply chain network actually exist. Each candidate edge is labeled with \
where it came from ("sketch" or "description") -- this is factual context only, not a reason to \
infer material plausibility. Do NOT reason about material plausibility or infer anything from \
entity context -- base your decision ONLY on what the user's answer explicitly says about each edge.

For each candidate edge, decide "keep" (the user confirmed or implied it's real) or "discard" \
(the user rejected or implied it isn't real). If the answer doesn't address a particular edge at \
all, use "unresolved" rather than guessing.

Return STRICT JSON, no markdown fences, no commentary:
{"decisions": [{"source": "<name>", "destination": "<name>", "action": "keep"|"discard"|"unresolved"}]}
"""


def build_edge_existence_question(candidate_edges: list) -> str:
    """Deterministic, code-authored -- lists every candidate edge labeled
    with which source it came from (sketch or description), no LLM call,
    no semantic pre-filtering."""
    lines = [EDGE_EXISTENCE_QUESTION_PREFIX, ""]
    for src, dst, origin in candidate_edges:
        label = "from sketch" if origin == "graph" else "from description"
        lines.append(f'  - "{src}" -> "{dst}"   ({label})')
    lines.append("\nWhich of these are real? You can list the ones to keep, the ones to discard, or both.")
    return "\n".join(lines)


def collect_edge_existence_answer(candidate_edges: list) -> str:
    q = build_edge_existence_question(candidate_edges)
    print(f"\n  Q: {q}\n")
    return input("  Your answer: ").strip()


def interpret_edge_existence_answer(candidate_edges: list, answer: str, model: str = DEFAULT_MODEL) -> list:
    payload = {
        "candidate_edges": [
            {"source": s, "destination": d, "origin": "sketch" if o == "graph" else "description"}
            for s, d, o in candidate_edges
        ],
        "answer": answer,
    }
    result = _llm_call(EDGE_EXISTENCE_INTERPRET_PROMPT, json.dumps(payload, indent=2), model)
    decisions = []
    for d in result.get("decisions", []):
        decisions.append((d["source"], d["destination"], d.get("action", "unresolved")))
    return decisions


def score_nodes(nl_json: dict, graph_json: dict, threshold: float = 0.6) -> dict:
    nl_nodes = collect_nodes(nl_json)
    graph_nodes = collect_nodes(graph_json)

    nl_names = [n for n, c in nl_nodes]
    graph_names = [n for n, c in graph_nodes]

    norm_map_nl = normalize_names(nl_names)
    norm_map_graph = normalize_names(graph_names)

    nl_by_norm = {}   # normalized -> original nl name (+ category)
    nl_cat = dict(nl_nodes)
    for name in nl_names:
        nl_by_norm[norm_map_nl.get(name, name)] = name

    graph_by_norm = {}  # normalized -> original graph name (+ category)
    graph_cat = dict(graph_nodes)
    for name in graph_names:
        graph_by_norm[norm_map_graph.get(name, name)] = name

    nl_norm_set = set(nl_by_norm.keys())
    graph_norm_set = set(graph_by_norm.keys())

    phase1_metrics = compute_f1(nl_norm_set, graph_norm_set)

    matched = [(graph_by_norm[n], nl_by_norm[n]) for n in (nl_norm_set & graph_norm_set)]
    graph_only = [(graph_by_norm[n], graph_cat[graph_by_norm[n]]) for n in (graph_norm_set - nl_norm_set)]
    nl_only = [(nl_by_norm[n], nl_cat[nl_by_norm[n]]) for n in (nl_norm_set - graph_norm_set)]

    # Phase 2: similarity-based auto-matching on whatever phase 1 left unmatched.
    auto_matched, still_graph_only, still_nl_only = fuzzy_resolve(
        graph_only, nl_only, norm_map_graph, norm_map_nl, threshold=threshold
    )

    final_tp = phase1_metrics["tp"] + len(auto_matched)
    final_fp = len(still_graph_only)
    final_fn = len(still_nl_only)
    final_precision = final_tp / (final_tp + final_fp) if (final_tp + final_fp) > 0 else 1.0
    final_recall = final_tp / (final_tp + final_fn) if (final_tp + final_fn) > 0 else 1.0
    final_f1 = (2 * final_precision * final_recall / (final_precision + final_recall)
                if (final_precision + final_recall) > 0 else 1.0)
    final_metrics = {"tp": final_tp, "fp": final_fp, "fn": final_fn,
                      "precision": final_precision, "recall": final_recall, "f1": final_f1}

    return {
        "phase1_metrics": phase1_metrics,
        "final_metrics": final_metrics,
        "matched": matched,
        "auto_matched": auto_matched,
        "graph_only": still_graph_only,
        "nl_only": still_nl_only,
    }


def print_report(result: dict):
    p1 = result["phase1_metrics"]
    pf = result["final_metrics"]

    print("\n--- Phase 1: exact match after normalization ---")
    print(f"  Precision: {p1['precision']:.3f}   Recall: {p1['recall']:.3f}   F-1: {p1['f1']:.3f}")
    print(f"  Matched (TP): {p1['tp']}   Graph-only (FP): {p1['fp']}   NL-only (FN): {p1['fn']}")

    if result["matched"]:
        print("\n  Matched:")
        for graph_name, nl_name in result["matched"]:
            if graph_name == nl_name:
                print(f'    "{graph_name}" == "{nl_name}"')
            else:
                print(f'    "{graph_name}"  <->  "{nl_name}"')

    if result["auto_matched"]:
        print("\n--- Phase 2: similarity-based auto-match ---")
        for gname, nname, score in result["auto_matched"]:
            print(f'    "{gname}"  ~=  "{nname}"   (similarity: {score:.2f})')

    print(f"\n--- Final ---")
    print(f"  Precision: {pf['precision']:.3f}   Recall: {pf['recall']:.3f}   F-1: {pf['f1']:.3f}")
    print(f"  Matched (TP): {pf['tp']}   Graph-only (FP): {pf['fp']}   NL-only (FN): {pf['fn']}")

    if result["graph_only"]:
        print("\n  Still unmatched -- graph-only (below similarity threshold):")
        for name, cat in result["graph_only"]:
            print(f'    "{name}" ({cat})')

    if result["nl_only"]:
        print("\n  Still unmatched -- NL-only (below similarity threshold):")
        for name, cat in result["nl_only"]:
            print(f'    "{name}" ({cat})')

    print("----------------------\n")


FINAL_EDGE_REVIEW_PROMPT = """You are updating a finalized supply-chain edge list based on the \
user's free-text correction request. Neither the sketch nor the description captured everything \
correctly, so the user may need to add an edge that's missing from both sources, remove one that's \
wrong, or fix a material assignment.

You're given: the current final edge list, structured context (supply/input/output fields) for \
every known entity, and the user's free-text request.

Return STRICT JSON, no markdown fences, no commentary:
{
  "operations": [
    {
      "action": "add" or "remove",
      "source": "<exact entity name>",
      "destination": "<exact entity name>",
      "material_name": "<infer from source/destination context if the user didn't state it explicitly, else \"missing\">",
      "material_type": "raw_material"|"intermediate_material"|"product"|"missing"
    }
  ]
}

Rules:
- "source" and "destination" must be entity names that actually appear in the entity context given.
- For "remove", material_name/material_type can be "missing" -- only source/destination matter.
- For "add", infer material_name/material_type from the source's supply_material_name or \
operation.output, cross-checked against the destination's operation.input, the same way edge \
plausibility is normally judged. Only use "missing" if genuinely not inferable.
- If the request doesn't clearly specify an add or remove, return an empty "operations" list \
rather than guessing.
"""


def collect_final_review_answer() -> str:
    return input("\nAnything to add, remove, or change in this final edge list? (blank to finish): ").strip()


def interpret_final_review(nl_json: dict, final_edges: list, answer: str, model: str = DEFAULT_MODEL) -> list:
    all_names = []
    for cat in NODE_CATEGORIES:
        all_names.extend(get_node_names(nl_json, cat))
    entity_context = {n: build_entity_context(nl_json, n) for n in all_names}

    payload = {
        "current_edges": [{"source": s, "destination": d} for s, d in final_edges],
        "entity_context": entity_context,
        "request": answer,
    }
    result = _llm_call(FINAL_EDGE_REVIEW_PROMPT, json.dumps(payload, indent=2), model)
    return result.get("operations", [])


def apply_final_review_operations(nl_json: dict, final_edges: list, operations: list) -> list:
    """Mutates nl_json['edges'] and final_edges in place-equivalent (returns
    updated final_edges list). Returns log lines for printing."""
    log = []
    for op in operations:
        action, src, dst = op.get("action"), op.get("source"), op.get("destination")
        if action == "add":
            mat_name = op.get("material_name", "missing")
            mat_type = op.get("material_type", "missing")
            if (src, dst) not in final_edges:
                final_edges.append((src, dst))
                nl_json.setdefault("edges", []).append({
                    "source": src, "destination": dst,
                    "material_name": mat_name, "material_type": mat_type,
                    "transfer_time": {"distribution": "missing", "parameters": {"a": "missing"}},
                })
                log.append(f'ADDED: "{src}" -> "{dst}"  [{mat_name} / {mat_type}]')
            else:
                log.append(f'"{src}" -> "{dst}" already in the edge list, skipped')
        elif action == "remove":
            if (src, dst) in final_edges:
                final_edges.remove((src, dst))
                nl_json["edges"] = [e for e in nl_json.get("edges", [])
                                     if not (e.get("source") == src and e.get("destination") == dst)]
                log.append(f'REMOVED: "{src}" -> "{dst}"')
            else:
                log.append(f'"{src}" -> "{dst}" not found in the edge list, skipped')
    return log


FACILITY_TYPE_PROMPT = """You are classifying supply-chain facilities as either "manufacturing" \
(transforms input material(s) into a DIFFERENT output material) or "warehouse" (stores or passes \
through material WITHOUT transforming it -- typically no operation.input/output at all, just \
storage via inventory_managed).

For each facility given, you have its operation.input/output (if any) and inventory_managed. \
Decide "manufacturing" or "warehouse", with confidence "high" or "low" -- use "low" whenever the \
evidence is genuinely ambiguous or sparse (e.g. no operation data at all).

Return STRICT JSON, no markdown fences, no commentary:
{"decisions": [{"name": "<facility name>", "type": "manufacturing"|"warehouse", "confidence": "high"|"low", "reasoning": "<one short sentence>"}]}
"""


def get_facility_type(nl_json: dict, name: str) -> str:
    for f in nl_json.get("facility", []):
        if f.get("name") == name:
            return f.get("type", "missing")
    return "missing"


def classify_facility_types_llm(names: list, nl_json: dict, model: str = DEFAULT_MODEL) -> list:
    """Returns list of (name, type, confidence, reasoning). Uses only
    operation.input/output and inventory_managed -- no edge dependency,
    since this stage now runs before edges are resolved."""
    if not names:
        return []
    payload = [{"name": n, "context": build_entity_context(nl_json, n)} for n in names]
    result = _llm_call(FACILITY_TYPE_PROMPT, json.dumps(payload, indent=2), model)
    decisions = []
    for d in result.get("decisions", []):
        decisions.append((d["name"], d.get("type", "manufacturing"), d.get("confidence", "low"), d.get("reasoning", "")))
    return decisions


def resolve_uncertain_facility_type(name: str, reasoning: str) -> str:
    """Runs at the terminal for facilities the LLM couldn't confidently
    classify. Simple direct question, keyword parsing -- no separate LLM
    interpretation call needed for a clear two-option choice."""
    print(f'  Uncertain facility type: "{name}"')
    print(f'    (LLM reasoning: {reasoning})')
    answer = input("    Is this a warehouse or a manufacturing facility? [warehouse/manufacturing]: ").strip().lower()
    return "warehouse" if "warehouse" in answer else "manufacturing"


def apply_facility_type(nl_json: dict, name: str, ftype: str) -> None:
    for f in nl_json.get("facility", []):
        if f.get("name") == name:
            f["type"] = ftype
            return


# ---------------------------------------------------------------------------
# Stage 4: material_type and material_name detection. Fully deterministic --
# no LLM call at all. material_type comes from a fixed lookup table keyed on
# (source_node_type, destination_node_type); material_name comes from
# fields NL2Sim already extracted (supply_material_name, operation.output/
# input, inventory_managed, customer.product). Anything not covered is
# flagged explicitly rather than guessed.
# ---------------------------------------------------------------------------

MATERIAL_TYPE_RULES = {
    ("supplier", "facility_manufacturing"): "raw_material",
    ("facility_manufacturing", "facility_manufacturing"): "intermediate_material",
    ("facility_manufacturing", "facility_warehouse"): "product",
    ("facility_manufacturing", "customer"): "product",
    ("facility_warehouse", "customer"): "product",
}


def get_node_type_detail(nl_json: dict, name: str) -> str:
    """Returns 'supplier', 'facility_manufacturing', 'facility_warehouse',
    'customer', or 'unknown'."""
    if name in get_node_names(nl_json, "supplier"):
        return "supplier"
    if name in get_node_names(nl_json, "customer"):
        return "customer"
    if name in get_node_names(nl_json, "facility"):
        ftype = get_facility_type(nl_json, name)
        if ftype == "warehouse":
            return "facility_warehouse"
        if ftype == "manufacturing":
            return "facility_manufacturing"
        return "facility_unknown_type"
    return "unknown"


def determine_material_type(nl_json: dict, source: str, destination: str):
    """Returns the material_type string, or None if this (source_type,
    destination_type) combination isn't covered by the rule table."""
    src_type = get_node_type_detail(nl_json, source)
    dst_type = get_node_type_detail(nl_json, destination)
    return MATERIAL_TYPE_RULES.get((src_type, dst_type))


def determine_material_name(nl_json: dict, source: str, destination: str, material_type: str):
    """Deterministic name inference from already-extracted fields. Returns
    the material name, or None if genuinely ambiguous (multiple candidates,
    no disambiguating signal)."""
    suppliers = {s["name"]: s for s in nl_json.get("supplier", [])}
    facilities = {f["name"]: f for f in nl_json.get("facility", [])}
    customers = {c["name"]: c for c in nl_json.get("customer", [])}

    if source in suppliers:
        name = suppliers[source].get("supply_material_name")
        return name if name and name != "missing" else None

    if source in facilities:
        facility = facilities[source]
        outputs = [x for x in facility.get("operation", {}).get("output", []) if x and x != "missing"]
        inv = [x for x in facility.get("inventory_managed", []) if x and x != "missing"]
        candidates = outputs if outputs else inv

        if len(candidates) == 1:
            return candidates[0]
        elif len(candidates) > 1:
            if destination in customers:
                prod = customers[destination].get("product")
                if prod in candidates:
                    return prod
            elif destination in facilities:
                dest_inputs = facilities[destination].get("operation", {}).get("input", [])
                common = set(candidates) & set(dest_inputs)
                if len(common) == 1:
                    return list(common)[0]
        return None  # ambiguous, no disambiguating signal

    return None


def ensure_edge_entries(nl_json: dict, final_edges: list) -> int:
    """Every confirmed edge needs a real dict entry in nl_json['edges'] --
    the existence-confirmation stage only tracked tuples. Creates stub
    entries (all fields 'missing') for anything not already present.
    Returns count of stubs created."""
    existing_pairs = {(e.get("source"), e.get("destination")) for e in nl_json.get("edges", [])}
    created = 0
    for src, dst in final_edges:
        if (src, dst) not in existing_pairs:
            nl_json.setdefault("edges", []).append({
                "source": src, "destination": dst,
                "material_name": "missing", "material_type": "missing",
                "transfer_time": {"distribution": "missing", "parameters": {"a": "missing"}},
            })
            existing_pairs.add((src, dst))
            created += 1
    return created


def detect_materials(nl_json: dict, final_edges: list) -> dict:
    """Fills material_name/material_type on every edge in nl_json['edges']
    matching final_edges, using ONLY the deterministic rules -- no LLM.
    Preserves any value already present and valid (not "missing") --
    never overwrites correct data already extracted upstream. Returns a
    report: {"filled": [...], "already_present": [...], "type_unresolved": [...], "name_unresolved": [...]}."""
    report = {"filled": [], "already_present": [], "type_unresolved": [], "name_unresolved": []}

    for e in nl_json.get("edges", []):
        pair = (e.get("source"), e.get("destination"))
        if pair not in final_edges:
            continue

        existing_name = e.get("material_name")
        existing_type = e.get("material_type")
        if existing_name not in (None, "missing") and existing_type not in (None, "missing"):
            report["already_present"].append((pair[0], pair[1], existing_name, existing_type))
            continue

        if existing_type in (None, "missing"):
            mat_type = determine_material_type(nl_json, e["source"], e["destination"])
            if mat_type is None:
                report["type_unresolved"].append(pair)
                continue
            e["material_type"] = mat_type
        else:
            mat_type = existing_type

        if existing_name in (None, "missing"):
            mat_name = determine_material_name(nl_json, e["source"], e["destination"], mat_type)
            if mat_name is None:
                report["name_unresolved"].append(pair)
                continue
            e["material_name"] = mat_name
        else:
            mat_name = existing_name

        report["filled"].append((pair[0], pair[1], mat_name, mat_type))

    return report


def main():
    parser = argparse.ArgumentParser(description="Node matching, facility type classification, then edge resolution -- checkpointed at each stage.")
    parser.add_argument("--graph", required=True, help="Path to graph topology JSON")
    parser.add_argument("--nl", required=True, help="Path to NL-derived config JSON")
    parser.add_argument("--out", default="output/nl_fixed.json", help="Path to write the final NL JSON (default: output/nl_fixed.json)")
    parser.add_argument("--threshold", type=float, default=0.6, help="Similarity threshold for phase 2 auto-matching (default: 0.6)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"OpenAI model for dialogue phases (default: {DEFAULT_MODEL})")
    parser.add_argument("--max-rounds", type=int, default=4, help="Safety cap on dispute-dialogue rounds (default: 4)")
    parser.add_argument("--start-from", choices=["nodes", "types", "edges", "materials"], default="nodes",
                         help="Resume from a checkpoint: 'nodes' (default, full run from scratch), "
                              "'types' (skip node resolution, load the node checkpoint), "
                              "'edges' (skip node + type stages, load the type checkpoint), "
                              "'materials' (skip everything else, load --out directly and only run material detection)")
    args = parser.parse_args()

    ckpt_dir = os.path.dirname(args.out) or "output"
    ckpt_nodes_nl = os.path.join(ckpt_dir, "checkpoint_nodes_nl.json")
    ckpt_nodes_graph = os.path.join(ckpt_dir, "checkpoint_nodes_graph.json")
    ckpt_types_nl = os.path.join(ckpt_dir, "checkpoint_types_nl.json")
    ckpt_types_graph = os.path.join(ckpt_dir, "checkpoint_types_graph.json")

    # =====================================================================
    # STAGE 1: NODES
    # =====================================================================
    if args.start_from == "nodes":
        graph_json = load_json(args.graph)
        nl_json = load_json(args.nl)
        graph_working = json.loads(json.dumps(graph_json))  # mutable copy -- deletions apply here, not to the original file

        full_name_map = {}
        nl_rename_map_all = {}
        nl_only_kept_all = set()
        nl_settled_all = set()

        for round_num in range(1, args.max_rounds + 1):
            result = score_nodes(nl_json, graph_working, threshold=args.threshold)
            # Entities already resolved (matched/merged) in a PRIOR round are
            # settled -- graph_working's raw string isn't rewritten until the
            # stage ends, so without this filter score_nodes() would keep
            # rediscovering them as unmatched and re-asking every round.
            result["graph_only"] = [(n, c) for n, c in result["graph_only"] if n not in full_name_map]
            # Entities already confirmed "keep" in a prior round are settled --
            # exclude them from this round's dispute pool so they're never
            # asked about again.
            result["nl_only"] = [(n, c) for n, c in result["nl_only"] if n not in nl_settled_all]
            print_report(result)

            for g_name, n_name in result["matched"]:
                full_name_map[g_name] = n_name
            for g_name, n_name, score in result["auto_matched"]:
                full_name_map[g_name] = n_name

            if not result["graph_only"] and not result["nl_only"]:
                break

            print(f"\n--- Dispute round {round_num} ---")
            answer = collect_combined_answer(result["graph_only"], result["nl_only"])
            decisions = interpret_combined_answer(result["graph_only"], result["nl_only"], answer, model=args.model)
            applied = apply_combined_decisions(nl_json, graph_working, decisions)

            print("  Resolutions:")
            for entry in applied["log"]:
                print(f"    {entry}")

            full_name_map.update(applied["name_map"])
            nl_rename_map_all.update(applied["nl_rename_map"])
            nl_only_kept_all |= applied["nl_only_kept"]
            nl_settled_all |= applied["nl_settled"]
        else:
            print(f"\n  WARNING: reached max rounds ({args.max_rounds}) without full convergence.")

        print("\n--- Recomputed F-1 (post-fix) ---")
        graph_nodes_orig = collect_nodes(graph_json)
        deleted_graph_names = set(collect_nodes(graph_json)) - set(collect_nodes(graph_working))
        deleted_graph_names = {n for n, c in deleted_graph_names}
        canonical_graph_names = {full_name_map.get(n, n) for n, c in graph_nodes_orig if n not in deleted_graph_names}

        nl_nodes_final = collect_nodes(nl_json)
        nl_names_final = {n for n, c in nl_nodes_final} - nl_only_kept_all

        tp = len(canonical_graph_names & nl_names_final)
        fp_names = canonical_graph_names - nl_names_final
        fn_names = nl_names_final - canonical_graph_names
        precision = tp / (tp + len(fp_names)) if (tp + len(fp_names)) > 0 else 1.0
        recall = tp / (tp + len(fn_names)) if (tp + len(fn_names)) > 0 else 1.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 1.0

        print(f"  Precision: {precision:.3f}   Recall: {recall:.3f}   F-1: {f1:.3f}")
        if nl_only_kept_all:
            print(f"  ({len(nl_only_kept_all)} entity(ies) kept but excluded from scoring: {sorted(nl_only_kept_all)})")
        if deleted_graph_names:
            print(f"  ({len(deleted_graph_names)} graph entity(ies) deleted: {sorted(deleted_graph_names)})")

        if f1 == 1.0:
            print("  F-1 = 1.00 -- every node fully reconciled.")
        else:
            if fp_names:
                print(f"  WARNING: {len(fp_names)} graph node(s) still unmatched -- investigate: {sorted(fp_names)}")
            if fn_names:
                print(f"  WARNING: {len(fn_names)} NL node(s) still unresolved -- investigate: {sorted(fn_names)}")

        print("----------------------\n")

        if f1 != 1.0:
            print("Node F-1 is not 1.00 -- stopping before further stages. Resolve remaining node disputes first.")
            return

        print("--- Confirmed node list (final, F-1 = 1.00), with full provenance ---")
        graph_sources_by_canon = {}
        for raw, canon in full_name_map.items():
            graph_sources_by_canon.setdefault(canon, []).append(raw)
        nl_sources_by_canon = {}
        for raw, canon in nl_rename_map_all.items():
            nl_sources_by_canon.setdefault(canon, []).append(raw)

        for cat in NODE_CATEGORIES:
            names = get_node_names(nl_json, cat)
            print(f"  {cat} ({len(names)}):")
            for n in names:
                graph_raws = graph_sources_by_canon.get(n, [])
                nl_raws = nl_sources_by_canon.get(n, [])
                provenance_bits = []
                if graph_raws:
                    provenance_bits.append(f"from sketch: {graph_raws}")
                if nl_raws:
                    provenance_bits.append(f"renamed from description: {nl_raws}")
                provenance = f"  ({'; '.join(provenance_bits)})" if provenance_bits else ""
                print(f'    - "{n}"{provenance}')
        print("------------------------------------------------------------\n")

        replaced_count = apply_name_map_to_graph(graph_working, full_name_map)
        print(f"  Rewrote {replaced_count} raw sketch name(s) to canonical form in the working graph copy.\n")

        os.makedirs(ckpt_dir, exist_ok=True)
        with open(ckpt_nodes_nl, "w") as f:
            json.dump(nl_json, f, indent=2)
        with open(ckpt_nodes_graph, "w") as f:
            json.dump(graph_working, f, indent=2)
        print(f"Checkpoint saved: {ckpt_nodes_nl}, {ckpt_nodes_graph}\n")
    elif args.start_from == "types":
        print(f"Skipping node resolution -- loading checkpoint from {ckpt_nodes_nl}")
        nl_json = load_json(ckpt_nodes_nl)
        graph_working = load_json(ckpt_nodes_graph)
    # else: start_from in ("edges", "materials") -- nl_json/graph_working set by stage 2 or 3 below

    # =====================================================================
    # STAGE 2: FACILITY TYPES (runs before edges -- type is now context
    # available to inform edge decisions, per design)
    # =====================================================================
    if args.start_from in ("nodes", "types"):
        print("\n=== Facility type classification ===")
        facility_names = get_node_names(nl_json, "facility")
        needs_type = [n for n in facility_names if get_facility_type(nl_json, n) == "missing"]
        already_set = [n for n in facility_names if n not in needs_type]

        if already_set:
            print("  Already known (from description, kept as-is):")
            for n in already_set:
                print(f'    "{n}": {get_facility_type(nl_json, n)}')

        if needs_type:
            print(f"\n  Classifying {len(needs_type)} facility(ies) with unknown type...")
            decisions = classify_facility_types_llm(needs_type, nl_json, model=args.model)
            for name, ftype, confidence, reasoning in decisions:
                if confidence == "high":
                    print(f'    "{name}" -> {ftype}  ({reasoning})')
                    apply_facility_type(nl_json, name, ftype)
                else:
                    resolved_type = resolve_uncertain_facility_type(name, reasoning)
                    print(f'    "{name}" -> {resolved_type}  (user confirmed)')
                    apply_facility_type(nl_json, name, resolved_type)
        else:
            print("  All facility types already known -- nothing to classify.")

        os.makedirs(ckpt_dir, exist_ok=True)
        with open(ckpt_types_nl, "w") as f:
            json.dump(nl_json, f, indent=2)
        with open(ckpt_types_graph, "w") as f:
            json.dump(graph_working, f, indent=2)
        print(f"\nCheckpoint saved: {ckpt_types_nl}, {ckpt_types_graph}")
        print("=== End facility type classification ===\n")
    elif args.start_from == "edges":
        print(f"Skipping node + type stages -- loading checkpoint from {ckpt_types_nl}")
        nl_json = load_json(ckpt_types_nl)
        graph_working = load_json(ckpt_types_graph)
    # else: start_from == "materials" -- nl_json set directly by stage 3 below

    # =====================================================================
    # STAGE 3: EDGES
    # =====================================================================
    if args.start_from != "materials":
        print("\n=== Edge resolution ===")
        edge_result = score_edges(nl_json, graph_working)
        em = edge_result["metrics"]
        print(f"\n--- Edge F-1 ---")
        print(f"  Precision: {em['precision']:.3f}   Recall: {em['recall']:.3f}   F-1: {em['f1']:.3f}")
        print(f"  Matched (TP): {em['tp']}   Graph-only (FP): {em['fp']}   NL-only (FN): {em['fn']}")

        if edge_result["matched"]:
            print("\n  Matched:")
            for g_edge, n_edge in edge_result["matched"]:
                print(f'    {g_edge[0]} -> {g_edge[1]}')

        candidate_edges = ([(s, d, "graph") for s, d in edge_result["graph_only"]] +
                            [(s, d, "nl") for s, d in edge_result["nl_only"]])

        final_edges = [g_edge for g_edge, n_edge in edge_result["matched"]]

        remaining = candidate_edges
        for round_num in range(1, args.max_rounds + 1):
            if not remaining:
                break
            print(f"\n--- Confirming existence of {len(remaining)} unmatched edge(s) (round {round_num}) ---")
            answer = collect_edge_existence_answer(remaining)
            decisions = interpret_edge_existence_answer(remaining, answer, model=args.model)

            still_unresolved = []
            decided = {(s, d) for s, d, a in decisions}
            origin_by_pair = {(s, d): o for s, d, o in remaining}
            for src, dst, action in decisions:
                if action == "keep":
                    print(f'    KEEP: "{src}" -> "{dst}"')
                    final_edges.append((src, dst))
                elif action == "discard":
                    print(f'    DISCARD: "{src}" -> "{dst}"')
                else:
                    print(f'    still unresolved: "{src}" -> "{dst}"')
                    still_unresolved.append((src, dst, origin_by_pair.get((src, dst), "graph")))
            for src, dst, origin in remaining:
                if (src, dst) not in decided:
                    still_unresolved.append((src, dst, origin))
            remaining = still_unresolved
        else:
            if remaining:
                print(f"\n  WARNING: {len(remaining)} edge(s) still unresolved after {args.max_rounds} rounds -- leaving them out.")

        nl_json["edges"] = [
            e for e in nl_json.get("edges", [])
            if (e.get("source"), e.get("destination")) in final_edges
        ]
        
        print(f"\n--- Final edge list ({len(final_edges)} edges) ---")
        for src, dst in final_edges:
            print(f"  {src} -> {dst}")

        print("\n--- Final review ---")
        print("(Neither source may have captured everything -- this is your chance to add, remove, or fix anything.)")
        while True:
            answer = collect_final_review_answer()
            if not answer:
                break
            operations = interpret_final_review(nl_json, final_edges, answer, model=args.model)
            if not operations:
                print("  Nothing actionable found in that -- try rephrasing, or leave blank to finish.")
                continue
            for line in apply_final_review_operations(nl_json, final_edges, operations):
                print(f"  {line}")
            print(f"\n--- Updated edge list ({len(final_edges)} edges) ---")
            for src, dst in final_edges:
                print(f"  {src} -> {dst}")

        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(nl_json, f, indent=2)
        print(f"\nFinal NL JSON (nodes + types + edges) written to {args.out}")
        print("=== End edge resolution ===\n")
    else:
        print(f"Skipping node, type, and edge stages -- loading directly from {args.out}")
        nl_json = load_json(args.out)
        final_edges = [(e.get("source"), e.get("destination")) for e in nl_json.get("edges", [])
                        if e.get("source") and e.get("destination")]

    # =====================================================================
    # STAGE 4: MATERIAL TYPE + NAME DETECTION -- fully deterministic, no
    # LLM call at all. material_type from the (source_type, dest_type) rule
    # table; material_name from fields already extracted by NL2Sim.
    # =====================================================================
    print("\n=== Material detection ===")
    created = ensure_edge_entries(nl_json, final_edges)
    if created:
        print(f"  Created {created} edge entry(ies) that had no material fields yet.")

    report = detect_materials(nl_json, final_edges)

    if report["already_present"]:
        print("\n  Already correct (kept as-is, not recomputed):")
        for src, dst, mat_name, mat_type in report["already_present"]:
            print(f'    "{src}" -> "{dst}"   [{mat_name} / {mat_type}]')

    if report["filled"]:
        print("\n  Filled:")
        for src, dst, mat_name, mat_type in report["filled"]:
            print(f'    "{src}" -> "{dst}"   [{mat_name} / {mat_type}]')

    if report["type_unresolved"]:
        print("\n  UNRESOLVED material_type (combination not in the rule table -- review manually):")
        for src, dst in report["type_unresolved"]:
            print(f'    "{src}" -> "{dst}"   ({get_node_type_detail(nl_json, src)} -> {get_node_type_detail(nl_json, dst)})')

    if report["name_unresolved"]:
        print("\n  UNRESOLVED material_name (type known, but name ambiguous -- review manually):")
        for src, dst in report["name_unresolved"]:
            print(f'    "{src}" -> "{dst}"')

    with open(args.out, "w") as f:
        json.dump(nl_json, f, indent=2)
    print(f"\n  Updated NL JSON (with materials) written to {args.out}")
    print("=== End material detection ===\n")


if __name__ == "__main__":
    main()