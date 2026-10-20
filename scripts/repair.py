"""
scripts2/repair.py
--------------------
Corrective actions for issues found by verification_layer1/2/3.py.

IMPORTANT ARCHITECTURE NOTE: this file contains ONLY the repair actions
themselves -- pure functions that take a config + a ValidationIssue and
fix that ONE issue. It does NOT contain the loop that runs verification,
dispatches to repair, and re-verifies -- that orchestration lives in a
separate script (not yet built). Keeping these separate means the repair
actions stay simple, testable in isolation, and swappable without
touching the loop logic.

QUESTION FRAMING: every repair action's introductory "here's the problem,
choose a fix" line is now generated via llm_humanize.generate_question()
rather than a hardcoded f-string -- passed the issue's location, the
config, and the user's original description/sketch-context, so the
phrasing is natural and scenario-grounded rather than templated. The
NUMBERED OPTIONS themselves (candidate facilities, materials, etc.) are
NOT LLM-generated -- those are computed directly from live config state
and printed as-is, since they encode real structural logic that must
stay exact. Only the sentence framing the choice is LLM-authored; the
choices are still deterministic. Each call passes the original template
text as fallback_text, so a failed API call still shows something usable.
"""

from __future__ import annotations

import re
from typing import Optional

from issue_types import ValidationIssue, DefectType, Severity
from verification_layer2 import INVENTORY_TYPE_TO_SECTION
from verification_layer3 import _build_adjacency, _bfs_reachable_from
from verification_layer1 import (
    SECTION_SPECS, SIMULATION_FIELDS, is_required,
    DISTRIBUTION_PARAM_COUNTS, PARAM_KEYS,
    MATERIAL_TYPES, PROCUREMENT_SCHEME_TYPES, FACILITY_TYPES, SHORTAGE_POLICIES,
    TIME_UNITS,
)
from llm_humanize import generate_question

_CURRENT_DESCRIPTION = ""

def set_repair_context(description: str = "") -> None:
    """Called once per repair run (by orchestrator.run_repair_loop) so
    every prompt in this module can pass the user's original NL
    description to the LLM question generator without threading it
    through every function signature individually."""
    global _CURRENT_DESCRIPTION
    _CURRENT_DESCRIPTION = description or ""


def _ask(config: dict, location: str, fallback_text: str) -> str:
    """
    Shared helper for the custom multi-option repair menus below: wraps
    generate_question() with the module's current description context,
    used to frame just the INTRO sentence of a "choose a fix" menu --
    never the numbered options themselves, which stay computed exactly
    as before. location may be a real issue.location, or a best-effort
    constructed string when no ValidationIssue is directly available
    (e.g. the shared facility-inbound/outbound helpers, called with just
    a facility dict) -- in that case entity-specific caching is weaker,
    but the call still succeeds and still gets scenario context via the
    description.

    answer_type is always 'num' here -- every one of these menus is
    followed by "Enter number:", never a yes/no or free-text response,
    so the question should always be phrased as "which one/what should
    happen", never as a yes/no question.
    """
    return generate_question(config, location, description=_CURRENT_DESCRIPTION, fallback_text=fallback_text, answer_type="num")
    return generate_question(config, location, description=_CURRENT_DESCRIPTION, fallback_text=fallback_text)

# ============================================================
# Entity resolution
# ============================================================

def _entry_at(config: dict, section: str, idx) -> Optional[dict]:
    entries = config.get(section, []) or []
    if not isinstance(idx, int) or idx < 0 or idx >= len(entries):
        return None
    entry = entries[idx]
    return entry if isinstance(entry, dict) else None


def describe_entity(config: dict, section: str, idx) -> str:
    """
    Produces a short, identifying, natural-language phrase for a
    specific entity -- its name plus whatever context makes it
    unambiguous and meaningful to a non-technical reader. Falls back to
    a plain "the Nth entry in <section>" if the entry or its name can't
    be resolved.
    """
    entry = _entry_at(config, section, idx)
    if entry is None:
        return f"entry #{idx} in {section.replace('_', ' ')}"

    name = entry.get("name")
    name_part = f"'{name}'" if isinstance(name, str) and name and name != "missing" else "(unnamed entry)"

    if section == "raw_materials":
        return f"the raw material {name_part}"
    if section == "intermediate_materials":
        return f"the intermediate material {name_part}"
    if section == "products":
        return f"the product {name_part}"
    if section == "inventory":
        inv_type = entry.get("type")
        label = {"raw_materials": "raw material", "intermediate_materials": "intermediate material", "products": "product"}.get(inv_type, "item")
        return f"inventory tracking for {label} {name_part}"
    if section == "supplier":
        material = entry.get("supply_material_name")
        suffix = f" (supplies '{material}')" if isinstance(material, str) and material else ""
        return f"the supplier {name_part}{suffix}"
    if section == "resource":
        return f"the resource {name_part}"
    if section == "facility":
        ftype = entry.get("type")
        output = (entry.get("operation") or {}).get("output") or []
        if ftype == "warehouse":
            return f"the warehouse {name_part}"
        suffix = f" (makes: {', '.join(output)})" if output else ""
        return f"the manufacturing facility {name_part}{suffix}"
    if section == "customer":
        product = entry.get("product")
        suffix = f" (orders '{product}')" if isinstance(product, str) and product else ""
        return f"the customer {name_part}{suffix}"

    return name_part


def _describe_edge(config: dict, idx: int) -> str:
    entry = _entry_at(config, "edges", idx)
    if entry is None:
        return f"edge #{idx}"
    src = entry.get("source")
    src = src if isinstance(src, str) and src != "missing" else "?"
    dst = entry.get("destination")
    dst = dst if isinstance(dst, str) and dst != "missing" else "?"
    mat = entry.get("material_name")
    label = f"the delivery from '{src}' to '{dst}'"
    if isinstance(mat, str) and mat and mat != "missing":
        label += f" (carrying '{mat}')"
    return label


# ============================================================
# Breadcrumb-style description (used as fallback_text source)
# ============================================================

def describe_location(config: dict, location: str) -> str:
    """
    Converts a technical location string into a readable breadcrumb --
    NOT a full question, just the path made human-readable: array
    indices replaced with the entry's name where one exists, dots/
    brackets replaced with ' -> ', section/field names title-cased.

    Used as the fallback_text source for generate_question() throughout
    this file -- if the LLM call fails, this breadcrumb text (not a full
    question, but always accurate and displayable) is shown instead.
    Falls back to the raw location string on any unexpected shape --
    never raises.
    """
    try:
        parts = []
        node = config
        tokens = re.findall(r"[a-zA-Z_]\w*(?:\[\d+\])?", location)

        for token in tokens:
            m = re.match(r"^([a-zA-Z_]\w*)(\[(\d+)\])?$", token)
            if not m:
                parts.append(token.replace("_", " ").capitalize())
                continue

            key = m.group(1)
            idx_str = m.group(3)

            child = node.get(key) if isinstance(node, dict) else None
            parts.append(key.replace("_", " ").capitalize())
            node = child

            if idx_str is not None:
                idx = int(idx_str)
                entry = node[idx] if isinstance(node, list) and 0 <= idx < len(node) else None

                if key == "edges" and isinstance(entry, dict):
                    src = entry.get("source")
                    src = src if isinstance(src, str) and src != "missing" else "?"
                    dst = entry.get("destination")
                    dst = dst if isinstance(dst, str) and dst != "missing" else "?"
                    mat = entry.get("material_name")
                    edge_label = f"{src} -> {dst}"
                    if isinstance(mat, str) and mat and mat != "missing":
                        edge_label += f" ({mat})"
                    parts.append(f"'{edge_label}'")
                else:
                    name = entry.get("name") if isinstance(entry, dict) else None
                    if isinstance(name, str) and name and name != "missing":
                        parts.append(f"'{name}'")
                    else:
                        parts.append(f"[{idx}]")

                node = entry

        return " -> ".join(parts)

    except Exception:
        return location


def _parse_location_steps(location: str) -> list:
    """
    Parses a ValidationIssue.location string into a list of navigation
    steps. Each step is either a string (dict key) or an int (list index).

    "supplier[0].supplier_cost" -> ["supplier", 0, "supplier_cost"]

    The repair actions genuinely depend on this raising ValueError on a
    malformed location (that's how a repair action's failure correctly
    propagates up to the orchestration loop's skip-tracking).
    """
    steps = []
    for part in location.split("."):
        match = re.match(r"^([a-zA-Z_][a-zA-Z0-9_]*)(\[(\d+)\])?$", part)
        if not match:
            raise ValueError(f"Could not parse location segment: '{part}' in '{location}'")
        key, _, index = match.groups()
        steps.append(key)
        if index is not None:
            steps.append(int(index))
    return steps


def _navigate_to_parent(config: dict, steps: list):
    """
    Walks all but the last step, returning (parent_container, last_step).
    Raises if an intermediate container doesn't exist -- this repair
    action only handles the LEAF field being missing, not missing
    intermediate containers (that's handled by earlier steps in the
    orchestrator, which repairs top-down).
    """
    current = config
    for step in steps[:-1]:
        if isinstance(step, int):
            current = current[step]
        else:
            current = current[step]
    return current, steps[-1]


# ----------------------------------------------------------------------
# Field type lookup
# ----------------------------------------------------------------------

SCALAR_FIELD_TYPES = {
    "config_info.name": "name",
    "config_info.version": "str",

    "raw_materials.name": "name",

    "intermediate_materials.name": "name",

    "products.name": "name",

    "inventory.name": "name",
    "inventory.type": "str",
    "inventory.initial_inventory": "num",
    "inventory.inventory_costs.holding_cost": "num",
    "inventory.inventory_costs.shortage_cost": "num",
    "inventory.inventory_costs.review_time": "num",
    "inventory.procurement_scheme.type": "str",

    "supplier.name": "name",
    "supplier.supply_material_name": "name",
    "supplier.supplier_capacity": "num",
    "supplier.supplier_cost": "num",

    "resource.name": "name",
    "resource.capacity": "num",
    "resource.operating_cost_per_time": "num",
    "resource.batching.enabled": "bool",
    "resource.batching.batch_size": "num",
    "resource.batching.max_wait_time": "num",
    "resource.failure.enabled": "bool",

    "facility.name": "name",
    "facility.type": "str",
    "facility.operation.name": "name",
    "facility.operation.resource_required": "name",

    "customer.name": "name",
    "customer.product": "name",
    "customer.shortage_policy": "str",
    "customer.unit_selling_price": "num",

    "edges.source": "name",
    "edges.destination": "name",
    "edges.material_type": "str",
    "edges.material_name": "name",

    "simulation.time_unit": "str",
    "simulation.horizon": "num",
    "simulation.warm_up": "num",
    "simulation.replications": "num",
    "simulation.random_seed": "num",

    "*.distribution": "str",
    "*.parameters.a": "num",
    "*.parameters.b": "num",
    "*.parameters.c": "num",
    "*.parameters.d": "num",
    "*.parameters.e": "num",
}


def normalize_location(location: str) -> str:
    """
    Strips array indices from a ValidationIssue.location string so it can
    be matched against SCALAR_FIELD_TYPES.
    """
    return re.sub(r"\[\d+\]", "", location)


def _lookup_field_type(normalized_location: str) -> str | None:
    """
    Look up the expected scalar type for a normalized location. Falls
    back to matching the "*.distribution" / "*.parameters.X" wildcard
    entries for any distribution object.
    """
    if normalized_location in SCALAR_FIELD_TYPES:
        return SCALAR_FIELD_TYPES[normalized_location]

    for suffix in (".distribution", ".parameters.a", ".parameters.b",
                   ".parameters.c", ".parameters.d", ".parameters.e"):
        if normalized_location.endswith(suffix):
            wildcard_key = "*" + suffix
            if wildcard_key in SCALAR_FIELD_TYPES:
                return SCALAR_FIELD_TYPES[wildcard_key]

    return None


# ----------------------------------------------------------------------
# Enum constraints
# ----------------------------------------------------------------------

DISTRIBUTION_TYPES = list(DISTRIBUTION_PARAM_COUNTS.keys())

QUANTITY_BASED_DISTRIBUTION_FIELDS = {"customer.demand"}

# Fields where "Instant" (constant, a=0) is dangerous even though they're
# genuinely time-based (unlike QUANTITY_BASED_DISTRIBUTION_FIELDS, where
# Instant is semantically meaningless). These generate a RECURRING event
# stream -- a customer placing another order, a facility starting its
# next production cycle, a resource finishing its next unit, a resource
# failing/recovering -- so Instant means "this happens infinitely often
# at the same simulated instant," which can spin the discrete-event
# engine in a zero-time loop that never advances the simulation clock
# (observed: an Instant customer.arrival_time froze a real run). A
# one-shot delay field (supplier_lead_time, transfer_time, etc.) has no
# such risk, since it fires once per event rather than regenerating
# itself -- those keep Instant as an option.
NO_INSTANT_RISK_FIELDS = {
    "customer.arrival_time",
    "facility.operation.operation_cycle",
    "resource.service_time",
    "resource.failure.uptime",
    "resource.failure.downtime",
}

ENUM_FIELD_VALUES = {
    "*.distribution": DISTRIBUTION_TYPES,
    "inventory.type": MATERIAL_TYPES,
    "edges.material_type": MATERIAL_TYPES,
    "facility.type": FACILITY_TYPES,
    "customer.shortage_policy": SHORTAGE_POLICIES,
    "simulation.time_unit": TIME_UNITS,
}


def _lookup_enum_values(normalized_location: str) -> list | None:
    """Same lookup pattern as _lookup_field_type, but for enum constraints."""
    if normalized_location in ENUM_FIELD_VALUES:
        return ENUM_FIELD_VALUES[normalized_location]

    if normalized_location.endswith(".distribution"):
        return ENUM_FIELD_VALUES.get("*.distribution")

    return None


# ----------------------------------------------------------------------
# Type-checked user prompting
# ----------------------------------------------------------------------

def _prompt_select_single(location: str, candidates: list, allow_none: bool = False, config: dict = None):
    """
    Presents a numbered menu for picking ONE value from a derived
    candidate list. The prompt header is generated via generate_question()
    (LLM-framed), with describe_location()'s breadcrumb text passed as
    fallback_text in case the API call fails.
    """
    if not candidates and not allow_none:
        return None

    fallback_text = f"Select a value for '{location}':"
    if config is not None:
        try:
            fallback_text = f"{describe_location(config, location)}:"
        except Exception:
            pass
    prompt_header = f"  {generate_question(config or {}, location, description=_CURRENT_DESCRIPTION, fallback_text=fallback_text, answer_type='num')}"

    print(prompt_header)
    if allow_none:
        print("    0) None (leave unset)")
    for i, name in enumerate(candidates, start=1):
        print(f"    {i}) {name}")

    while True:
        choice = input("  Enter number: ").strip()
        if allow_none and choice == "0":
            return None
        if not choice.isdigit() or not (1 <= int(choice) <= len(candidates)):
            print(f"    '{choice}' is not a valid selection. Try again.")
            continue
        return candidates[int(choice) - 1]


def _raw_material_candidates(config: dict, entry: dict = None) -> list:
    return sorted(_collect_names(config, "raw_materials"))


def _product_candidates(config: dict, entry: dict = None) -> list:
    return sorted(_collect_names(config, "products"))


def _resource_name_candidates(config: dict, entry: dict = None) -> list:
    return sorted(_collect_names(config, "resource"))


def _node_endpoint_candidates(config: dict, entry: dict = None) -> list:
    """Candidates for edges.source / edges.destination."""
    return sorted(
        _collect_names(config, "supplier")
        | _collect_names(config, "facility")
        | _collect_names(config, "customer")
    )


def _inventory_name_candidates(config: dict, entry: dict) -> list:
    """CASCADING: inventory.name candidates depend on the sibling
    inventory.type already set on the same entry."""
    inv_type = (entry or {}).get("type")
    section = INVENTORY_TYPE_TO_SECTION.get(inv_type)
    if section is None:
        return []
    return sorted(_collect_names(config, section))


def _edge_material_name_candidates(config: dict, entry: dict) -> list:
    """CASCADING: edges.material_name candidates depend on the sibling
    edges.material_type already set on the same entry."""
    material_type = (entry or {}).get("material_type")
    section = INVENTORY_TYPE_TO_SECTION.get(material_type)
    if section is None:
        return []
    return sorted(_collect_names(config, section))


SCALAR_SELECT_CANDIDATE_FNS = {
    "supplier.supply_material_name": _raw_material_candidates,
    "customer.product": _product_candidates,
    "facility.operation.resource_required": _resource_name_candidates,
    "edges.source": _node_endpoint_candidates,
    "edges.destination": _node_endpoint_candidates,
    "inventory.name": _inventory_name_candidates,
    "edges.material_name": _edge_material_name_candidates,
}

SCALAR_SELECT_ALLOW_NONE = {"facility.operation.resource_required"}


def _prompt_for_value(location: str, expected_kind: str, enum_values: list = None, config: dict = None):
    """
    Prompts the user for a value at the given location, validating it
    against expected_kind ("str" / "num" / "bool") before accepting it.
    Re-prompts on a type mismatch rather than silently coercing.

    The prompt text is generated via generate_question() (LLM-framed),
    with describe_location()'s breadcrumb text as fallback_text in case
    the API call fails.
    """
    fallback_text = f"Missing required field '{location}'."
    if config is not None:
        try:
            fallback_text = f"{describe_location(config, location)}:"
        except Exception:
            pass
    # When enum_values is given, the actual input is always "Enter number:"
    # (a menu selection) regardless of expected_kind -- so the question
    # should be phrased for that, not for expected_kind's raw type.
    effective_answer_type = "num" if enum_values is not None else expected_kind
    prompt_prefix = f"  {generate_question(config or {}, location, description=_CURRENT_DESCRIPTION, fallback_text=fallback_text, answer_type=effective_answer_type)}"

    if enum_values is not None:
        print(f"{prompt_prefix} Select one:")
        for i, opt in enumerate(enum_values, start=1):
            print(f"    {i}) {opt}")
        while True:
            choice = input("  Enter number: ").strip()
            if choice.isdigit() and 1 <= int(choice) <= len(enum_values):
                return enum_values[int(choice) - 1]
            print(f"    '{choice}' is not a valid selection. Try again.")

    while True:
        raw = input(f"{prompt_prefix} Enter value ({expected_kind}): ").strip()

        if expected_kind == "str":
            if raw == "":
                print("    Value cannot be blank for a string field. Try again.")
                continue
            return raw

        elif expected_kind == "name":
            if raw == "":
                print("    Value cannot be blank for a name field. Try again.")
                continue
            try:
                float(raw)
                print(f"    '{raw}' is purely numeric -- names must contain "
                      f"non-numeric characters. Try again.")
                continue
            except ValueError:
                pass
            return raw

        elif expected_kind == "num":
            try:
                if "." in raw:
                    return float(raw)
                return int(raw)
            except ValueError:
                print(f"    '{raw}' is not a valid number. Try again.")
                continue

        elif expected_kind == "bool":
            if raw.lower() in ("true", "yes", "y", "1"):
                return True
            elif raw.lower() in ("false", "no", "n", "0"):
                return False
            else:
                print(f"    '{raw}' is not a valid boolean (true/false). Try again.")
                continue

        else:
            raise ValueError(f"No type-checking rule for kind '{expected_kind}'")


def _prompt_for_distribution(location: str, config: dict = None, allow_instant: bool = True) -> tuple:
    """
    Prompts for a full distribution object (distribution type +
    parameters) in ONE combined step, with "Instant" offered as an
    additional FIRST menu option when allow_instant=True.

    allow_instant MUST be False for QUANTITY-based distributions.

    The prompt text is generated via generate_question() (LLM-framed),
    with describe_location()'s output as fallback_text.

    Returns (distribution_name, parameters_dict).
    """
    fallback_text = f"Missing required field '{location}'."
    if config is not None:
        try:
            fallback_text = f"{describe_location(config, location)}:"
        except Exception:
            pass
    prompt_prefix = f"  {generate_question(config or {}, location, description=_CURRENT_DESCRIPTION, fallback_text=fallback_text, answer_type='num')}"

    if allow_instant:
        options = ["Instant (always exactly 0 -- no delay or variability)"] + list(DISTRIBUTION_TYPES)
    else:
        options = list(DISTRIBUTION_TYPES)
    print(f"{prompt_prefix} Select one:")
    for i, opt in enumerate(options, start=1):
        print(f"    {i}) {opt}")

    while True:
        choice = input("  Enter number: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(options):
            break
        print(f"    '{choice}' is not a valid selection. Try again.")

    choice_num = int(choice)
    if allow_instant and choice_num == 1:
        return "constant", {"a": 0}

    dist_index = choice_num - 2 if allow_instant else choice_num - 1
    dist_value = DISTRIBUTION_TYPES[dist_index]
    param_count = DISTRIBUTION_PARAM_COUNTS.get(dist_value, 1)
    params = {}
    for i, pkey in enumerate(PARAM_KEYS):
        if i < param_count:
            params[pkey] = _prompt_for_value(f"{location}.parameters.{pkey}", "num", config=config)
    return dist_value, params


# ----------------------------------------------------------------------
# Schema-driven recursive repair (missing containers, not just leaves)
# ----------------------------------------------------------------------

def find_spec_node(normalized_location: str):
    """
    Walks SECTION_SPECS (or SIMULATION_FIELDS for the simulation section)
    to find the FIELD spec dict describing the node at normalized_location.
    """
    parts = normalized_location.split(".")
    section = parts[0]

    if section in SECTION_SPECS:
        current_fields = SECTION_SPECS[section]["fields"]
    elif section == "simulation":
        current_fields = SIMULATION_FIELDS
    else:
        raise ValueError(
            f"Section '{section}' is not supported by this repair action "
            f"(e.g. 'nodes' has no fixed-field spec to walk)."
        )

    fspec = None
    remaining = parts[1:]
    for i, part in enumerate(remaining):
        if current_fields is None or part not in current_fields:
            raise ValueError(
                f"Could not resolve '{normalized_location}' against the schema "
                f"spec at segment '{part}'."
            )
        fspec = current_fields[part]
        is_last = (i == len(remaining) - 1)
        if not is_last:
            if "fields" in fspec:
                current_fields = fspec["fields"]
            elif fspec.get("is_distribution"):
                current_fields = fspec.get("extra_fields")
            else:
                current_fields = None

    return fspec


def _collect_names(config: dict, section: str) -> set:
    """Collect the set of 'name' values from a list-section."""
    names = set()
    for entry in config.get(section, []) or []:
        if isinstance(entry, dict):
            name = entry.get("name")
            if isinstance(name, str):
                names.add(name)
    return names


def _add_to_inventory_managed(entity: dict, material_name: str) -> None:
    """
    Shared helper for adding a real material to any entity's
    inventory_managed list. First strips out any literal "missing"
    placeholder entries before appending the real material.
    """
    current = entity.setdefault("inventory_managed", [])
    cleaned = [m for m in current if m != "missing"]
    if len(cleaned) != len(current):
        current[:] = cleaned
    if material_name not in current:
        current.append(material_name)


def _fill_bom_by_selection(config: dict, owning_entry: dict, location: str, section_name: str):
    """
    bom keys are chosen from a menu, not typed freely. The candidate set
    depends on which section owns this bom.
    """
    material_name = owning_entry.get("name")

    if section_name == "intermediate_materials":
        candidates = sorted(_collect_names(config, "raw_materials"))
    elif section_name == "products":
        candidates = sorted(_collect_names(config, "raw_materials") | _collect_names(config, "intermediate_materials"))
    else:
        candidates = []

    if not candidates:
        raise ValueError(
            f"No candidate materials available to build '{material_name}'s bom -- "
            f"declare raw_materials (and intermediate_materials, if this is a product) first."
        )

    bom = {}
    owning_entry["bom"] = bom

    fallback_text = f"Select materials for '{material_name}'s bom (blank to finish, enter 0 for a material's quantity to exclude it):"
    print(f"  {_ask(config, location, fallback_text)}")
    excluded = set()
    while True:
        remaining = [c for c in candidates if c not in bom and c not in excluded]
        if not remaining:
            print("    All candidate materials have been addressed.")
            break
        for i, name in enumerate(remaining, start=1):
            print(f"    {i}) {name}")
        choice = input("  Enter number (blank to finish): ").strip()
        if choice == "":
            break
        if not choice.isdigit() or not (1 <= int(choice) <= len(remaining)):
            print(f"    '{choice}' is not a valid selection. Try again.")
            continue
        selected = remaining[int(choice) - 1]
        qty = _prompt_for_value(f"{location}.{selected}", "num")
        if qty == 0:
            excluded.add(selected)
            print(f"    '{selected}' excluded (not part of this bom).")
        else:
            bom[selected] = qty


def _fill_list_select(owning_dict: dict, key: str, candidates: list, location: str, config: dict = None):
    """
    Builds a list-shaped field by letting the user pick entries from a
    candidate list, by number, repeating until they choose to stop.
    """
    if not candidates:
        raise ValueError(f"No candidates available for '{location}' -- nothing to select from.")

    selected = []
    owning_dict[key] = selected

    fallback_text = f"Select entries for '{location}' (blank to finish):"
    print(f"  {_ask(config or {}, location, fallback_text)}")
    while True:
        remaining = [c for c in candidates if c not in selected]
        if not remaining:
            print("    All candidates have been added.")
            break
        for i, name in enumerate(remaining, start=1):
            print(f"    {i}) {name}")
        choice = input("  Enter number (blank to finish): ").strip()
        if choice == "":
            break
        if not choice.isdigit() or not (1 <= int(choice) <= len(remaining)):
            print(f"    '{choice}' is not a valid selection. Try again.")
            continue
        selected.append(remaining[int(choice) - 1])


MATERIAL_STAGE_ORDER = {"raw_materials": 0, "intermediate_materials": 1, "products": 2}


def _material_stage(config: dict, material_name: str):
    """Returns 0 (raw), 1 (intermediate), 2 (product), or None if the
    material isn't found in any of the three category lists."""
    for section, stage in MATERIAL_STAGE_ORDER.items():
        if material_name in _collect_names(config, section):
            return stage
    return None


def _try_auto_derive_operation_list(config: dict, section_entry: dict, which: str):
    """
    Auto-derives operation.input or operation.output purely from the
    facility's own inventory_managed, using a stage ordering.
    Returns None if genuinely ambiguous.
    """
    managed = [m for m in (section_entry or {}).get("inventory_managed", []) if isinstance(m, str)]
    if not managed:
        return None

    staged: dict = {}
    for m in managed:
        stage = _material_stage(config, m)
        if stage is not None:
            staged.setdefault(stage, []).append(m)

    stages_present = sorted(staged.keys())

    if len(stages_present) < 2:
        return None

    if len(stages_present) == 2:
        min_stage, max_stage = stages_present[0], stages_present[-1]
        return staged[min_stage] if which == "input" else staged[max_stage]

    if which == "input":
        return staged.get(0, []) + staged.get(1, [])
    else:
        return staged.get(2, [])


def _all_material_names(config: dict, section_entry: dict = None) -> list:
    return sorted(
        _collect_names(config, "raw_materials")
        | _collect_names(config, "intermediate_materials")
        | _collect_names(config, "products")
    )


def _operation_input_candidates(config: dict, section_entry: dict = None) -> list:
    category_set = _collect_names(config, "raw_materials") | _collect_names(config, "intermediate_materials")
    managed = set(
        m for m in ((section_entry or {}).get("inventory_managed") or []) if isinstance(m, str)
    )
    if not managed:
        return sorted(category_set)
    return sorted(category_set & managed)


def _operation_output_candidates(config: dict, section_entry: dict = None) -> list:
    category_set = _collect_names(config, "intermediate_materials") | _collect_names(config, "products")
    managed = set(
        m for m in ((section_entry or {}).get("inventory_managed") or []) if isinstance(m, str)
    )
    if not managed:
        return sorted(category_set)
    return sorted(category_set & managed)


LIST_SELECT_CANDIDATE_FNS = {
    "facility.inventory_managed": _all_material_names,
    "facility.operation.input": _operation_input_candidates,
    "facility.operation.output": _operation_output_candidates,
}


def _build_procurement_scheme(config: dict, location: str):
    """
    Builds a fresh procurement_scheme dict from scratch: prompts for
    "type" first (enum-constrained, never free text), then whatever that
    type requires.
    """
    type_val = _prompt_for_value(
        f"{location}.type", "str",
        enum_values=PROCUREMENT_SCHEME_TYPES, config=config,
    )
    obj = {"type": type_val}

    if type_val == "periodic_supply":
        dist_value, params = _prompt_for_distribution(location, config=config, allow_instant=False)
        obj["distribution"] = dist_value
        obj["parameters"] = params

    elif type_val == "inventory_threshold":
        params = {}
        for pkey in ("a", "b"):
            params[pkey] = _prompt_for_value(f"{location}.parameters.{pkey}", "num", config=config)
        obj["parameters"] = params

    return obj


def _fill_node(config: dict, parent: dict, key: str, fspec: dict, location: str, normalized: str, section_entry: dict = None):
    """
    Creates parent[key] and recursively fills in every REQUIRED field
    beneath it, per fspec's shape.
    """
    if key == "bom" and fspec.get("is_dict_values"):
        section_name = normalized.split(".")[0]
        _fill_bom_by_selection(config, parent, location, section_name)
        return parent[key]

    if fspec.get("is_procurement_scheme"):
        obj = _build_procurement_scheme(config, location)
        parent[key] = obj
        return obj

    if normalized in ("facility.operation.input", "facility.operation.output"):
        which = "input" if normalized.endswith(".input") else "output"
        derived = _try_auto_derive_operation_list(config, section_entry, which)
        if derived is not None:
            parent[key] = derived
            print(f"  '{location}' auto-derived from inventory_managed: {derived}")
            return derived
        candidates = LIST_SELECT_CANDIDATE_FNS[normalized](config, section_entry)
        _fill_list_select(parent, key, candidates, location, config=config)
        return parent[key]

    if normalized in LIST_SELECT_CANDIDATE_FNS:
        candidates = LIST_SELECT_CANDIDATE_FNS[normalized](config, section_entry)
        _fill_list_select(parent, key, candidates, location, config=config)
        return parent[key]

    if fspec.get("is_distribution"):
        obj = {}
        parent[key] = obj

        extra_fields = fspec.get("extra_fields")
        if extra_fields:
            for fname, child_fspec in extra_fields.items():
                required_now = is_required(child_fspec["required"], obj)
                should_ask = required_now or child_fspec.get("always_ask", False)
                if should_ask:
                    _fill_node(config, obj, fname, child_fspec, f"{location}.{fname}", f"{normalized}.{fname}", section_entry)

        dist_value, params = _prompt_for_distribution(
            location, config=config,
            allow_instant=(normalized not in QUANTITY_BASED_DISTRIBUTION_FIELDS
                           and normalized not in NO_INSTANT_RISK_FIELDS),
        )
        obj["distribution"] = dist_value
        obj["parameters"] = params

        return obj

    elif fspec.get("is_dict_values"):
        obj = {}
        parent[key] = obj
        value_kind = fspec.get("value_kind", "num")

        fallback_text = f"'{location}' is empty -- add entries below (blank name to finish)."
        print(f"  {_ask(config, location, fallback_text)}")
        while True:
            name = input(f"    Enter key name for '{location}' (blank to finish): ").strip()
            if name == "":
                break
            obj[name] = _prompt_for_value(f"{location}.{name}", value_kind, config=config)

        return obj

    elif "fields" in fspec:
        obj = {}
        parent[key] = obj

        for fname, child_fspec in fspec["fields"].items():
            required_now = is_required(child_fspec["required"], obj)
            should_ask = required_now or child_fspec.get("always_ask", False)
            if should_ask:
                _fill_node(config, obj, fname, child_fspec, f"{location}.{fname}", f"{normalized}.{fname}", section_entry)

        return obj

    else:
        select_fn = SCALAR_SELECT_CANDIDATE_FNS.get(normalized)
        if select_fn is not None:
            candidates = select_fn(config, parent)
            allow_none = normalized in SCALAR_SELECT_ALLOW_NONE
            if candidates or allow_none:
                selected = _prompt_select_single(location, candidates, allow_none=allow_none, config=config)
                if selected is not None:
                    parent[key] = selected
                return selected

        kind = _lookup_field_type(normalized) or "str"
        enum_values = _lookup_enum_values(normalized)
        value = _prompt_for_value(location, kind, enum_values=enum_values, config=config)
        parent[key] = value
        return value


def _build_new_entry(config: dict, section_name: str, presets: dict, location_prefix: str) -> tuple:
    """
    Appends a new entry to config[section_name], pre-filling any fields
    given in `presets`, then asks for every other REQUIRED field.
    """
    new_entry = dict(presets)
    config.setdefault(section_name, []).append(new_entry)
    idx = len(config[section_name]) - 1

    fields_spec = SECTION_SPECS[section_name]["fields"]
    for fname, child_fspec in fields_spec.items():
        if fname in new_entry:
            continue
        required_now = is_required(child_fspec["required"], new_entry)
        should_ask = required_now or child_fspec.get("always_ask", False)
        if should_ask:
            child_location = f"{location_prefix}[{idx}].{fname}"
            child_normalized = f"{section_name}.{fname}"
            _fill_node(config, new_entry, fname, child_fspec, child_location, child_normalized, new_entry)

    return new_entry, idx


def _delete_material_and_associations(config: dict, material_name: str) -> list:
    """
    Removes material_name from everywhere it can appear. Returns a list
    of human-readable strings describing what was removed.
    """
    removed = []

    for section in ("raw_materials", "intermediate_materials", "products"):
        before = len(config.get(section, []) or [])
        config[section] = [
            e for e in config.get(section, []) or []
            if not (isinstance(e, dict) and e.get("name") == material_name)
        ]
        if len(config[section]) < before:
            removed.append(f"{section}[] entry")

    for section in ("intermediate_materials", "products"):
        for entry in config.get(section, []) or []:
            if isinstance(entry, dict) and isinstance(entry.get("bom"), dict):
                if material_name in entry["bom"]:
                    del entry["bom"][material_name]
                    removed.append(f"{section} '{entry.get('name')}'s bom entry")

    before = len(config.get("inventory", []) or [])
    config["inventory"] = [
        e for e in config.get("inventory", []) or []
        if not (isinstance(e, dict) and e.get("name") == material_name)
    ]
    if len(config["inventory"]) < before:
        removed.append("inventory[] entry")

    for f in config.get("facility", []) or []:
        if not isinstance(f, dict):
            continue
        if material_name in (f.get("inventory_managed") or []):
            f["inventory_managed"].remove(material_name)
            removed.append(f"facility '{f.get('name')}' inventory_managed entry")
        operation = f.get("operation")
        if isinstance(operation, dict):
            for key in ("input", "output"):
                if material_name in (operation.get(key) or []):
                    operation[key].remove(material_name)
                    removed.append(f"facility '{f.get('name')}' operation.{key} entry")

    suppliers_to_remove = [
        s.get("name") for s in config.get("supplier", []) or []
        if isinstance(s, dict) and s.get("supply_material_name") == material_name
    ]
    if suppliers_to_remove:
        config["supplier"] = [
            s for s in config.get("supplier", []) or [] if s.get("name") not in suppliers_to_remove
        ]
        removed.append(f"supplier(s): {', '.join(suppliers_to_remove)}")
        nodes = config.get("nodes")
        if isinstance(nodes, list) and len(nodes) > 0 and isinstance(nodes[0], dict):
            if isinstance(nodes[0].get("supplier"), list):
                nodes[0]["supplier"] = [n for n in nodes[0]["supplier"] if n not in suppliers_to_remove]

    def _edge_should_be_removed(e):
        if not isinstance(e, dict):
            return False
        if e.get("material_name") == material_name:
            return True
        if e.get("source") in suppliers_to_remove:
            return True
        return False

    before_edges = len(config.get("edges", []) or [])
    config["edges"] = [e for e in config.get("edges", []) or [] if not _edge_should_be_removed(e)]
    if len(config["edges"]) < before_edges:
        removed.append(f"{before_edges - len(config['edges'])} edge(s)")

    return removed


def _offer_delete_material_option(config: dict, material_name: str) -> bool:
    """
    Prompts for confirmation, then runs the cascading deletion if
    confirmed. This is a yes/no confirmation, not a "choose a fix"
    question -- left as plain text rather than LLM-framed, since its
    exact wording ("Type 'yes' to confirm") is also the literal string
    the input is matched against.
    """
    confirm = input(
        f"  Type 'yes' to confirm deleting '{material_name}' and everything associated "
        f"with it (this cannot be undone), or press Enter to go back: "
    ).strip().lower()
    if confirm != "yes":
        return False

    removed = _delete_material_and_associations(config, material_name)
    print(f"  Deleted '{material_name}'. Removed: {', '.join(removed) if removed else '(nothing else referenced it)'}")
    return True


def repair_intermediate_material_not_producible(config: dict, issue: ValidationIssue) -> dict:
    """
    Layer3 check 5: an intermediate material is never produced by any
    facility's operation.output. Three options: add to an existing
    manufacturing facility, create a new one, or delete the material.
    """
    material_name = issue.context.get("referenced_name") if issue.context else None
    if not material_name:
        raise ValueError(f"Issue at '{issue.location}' has no referenced_name in its context.")

    producible = set()
    for f in config.get("facility", []) or []:
        if isinstance(f, dict):
            operation = f.get("operation")
            if isinstance(operation, dict):
                producible |= set(m for m in (operation.get("output") or []) if isinstance(m, str))
    if material_name in producible:
        raise ValueError(f"'{material_name}' is already producible -- issue may be stale.")

    manufacturing_facilities = [
        (i, f.get("name"), (f.get("operation") or {}).get("output") or [])
        for i, f in enumerate(config.get("facility", []) or [])
        if isinstance(f, dict) and f.get("type") == "manufacturing" and isinstance(f.get("name"), str)
    ]

    fallback_text = f"Intermediate material '{material_name}' is not produced by any facility. Choose a fix:"
    print(_ask(config, issue.location, fallback_text))
    print("    0) Delete this intermediate material entirely")
    for i, (facility_idx, name, current_output) in enumerate(manufacturing_facilities, start=1):
        output_hint = f" (currently makes: {', '.join(current_output)})" if current_output else " (no output yet)"
        print(f"    {i}) Add it to existing facility '{name}'s operation.output{output_hint}")
    new_facility_option = len(manufacturing_facilities) + 1
    print(f"    {new_facility_option}) Create a new manufacturing facility for it")

    while True:
        choice = input("  Enter number: ").strip()
        if choice == "0":
            if _offer_delete_material_option(config, material_name):
                return config
            continue
        if not choice.isdigit() or not (1 <= int(choice) <= new_facility_option):
            print(f"    '{choice}' is not a valid selection. Try again.")
            continue
        break

    choice_num = int(choice)

    if choice_num == new_facility_option:
        print("Creating a new manufacturing facility.")
        print(f"  (Remember to include '{material_name}' when selecting inventory_managed below.)")
        new_entry, _ = _build_new_entry(config, "facility", {"type": "manufacturing"}, "facility")

        _add_to_inventory_managed(new_entry, material_name)
        operation = new_entry.setdefault("operation", {})
        if material_name not in (operation.get("output") or []):
            operation.setdefault("output", []).append(material_name)
        print(f"  Ensured '{material_name}' is in the new facility's inventory_managed and operation.output.")
    else:
        facility_idx, facility_name, _ = manufacturing_facilities[choice_num - 1]
        facility = config["facility"][facility_idx]
        _add_to_inventory_managed(facility, material_name)
        operation = facility.setdefault("operation", {})
        if material_name not in (operation.get("output") or []):
            operation.setdefault("output", []).append(material_name)
        print(f"  Added '{material_name}' to '{facility_name}'s inventory_managed and operation.output.")

    return config


def repair_product_end_to_end_path(config: dict, issue: ValidationIssue) -> dict:
    """
    Layer3 check 11: a product's producing facility is disconnected --
    either no supplier can reach it (upstream), or it can't reach any
    ordering customer (downstream).
    """
    product_name = issue.context.get("referenced_name") if issue.context else None
    if not product_name:
        raise ValueError(f"Issue at '{issue.location}' has no referenced_name in its context.")

    producing_facilities = set()
    for f in config.get("facility", []) or []:
        if isinstance(f, dict):
            operation = f.get("operation") or {}
            if product_name in (operation.get("output") or []) and isinstance(f.get("name"), str):
                producing_facilities.add(f["name"])

    ordering_customers = set(
        c.get("name") for c in config.get("customer", []) or []
        if isinstance(c, dict) and c.get("product") == product_name and isinstance(c.get("name"), str)
    )

    if not producing_facilities or not ordering_customers:
        raise ValueError(
            f"'{product_name}' has no producing facility or no ordering customer yet -- "
            f"this issue must be from check_product_is_producible or "
            f"check_product_has_customer instead. Not acting."
        )

    adjacency = _build_adjacency(config)
    supplier_names = _collect_names(config, "supplier")

    upstream_ok = any(
        _bfs_reachable_from(s, adjacency) & producing_facilities for s in supplier_names
    )
    downstream_ok = any(
        _bfs_reachable_from(f, adjacency) & ordering_customers for f in producing_facilities
    )

    if upstream_ok and downstream_ok:
        raise ValueError(f"'{product_name}' is already fully connected end-to-end -- issue may be stale.")

    facility_name = sorted(producing_facilities)[0]
    facility = next(
        f for f in config["facility"]
        if f.get("name") == facility_name and product_name in ((f.get("operation") or {}).get("output") or [])
    )

    made_progress = False
    errors = []

    if not upstream_ok:
        print(_ask(config, issue.location,
                    f"Product '{product_name}': producing facility '{facility_name}' is not reachable from any supplier."))
        try:
            _fix_facility_missing_inbound(config, facility)
            made_progress = True
        except ValueError as e:
            errors.append(str(e))

    still_exists = any(
        isinstance(f, dict) and f.get("name") == facility_name
        for f in config.get("facility", []) or []
    )

    if not downstream_ok and still_exists:
        print(_ask(config, issue.location,
                    f"Product '{product_name}': producing facility '{facility_name}' cannot reach any customer ordering it."))
        try:
            _fix_facility_missing_outbound(config, facility)
            made_progress = True
        except ValueError as e:
            errors.append(str(e))

    if not made_progress:
        raise ValueError("; ".join(errors))

    return config


def repair_product_not_producible(config: dict, issue: ValidationIssue) -> dict:
    """
    Layer3 check 6: a product is never produced by any facility's
    operation.output.
    """
    product_name = issue.context.get("referenced_name") if issue.context else None
    if not product_name:
        raise ValueError(f"Issue at '{issue.location}' has no referenced_name in its context.")

    producible = set()
    for f in config.get("facility", []) or []:
        if isinstance(f, dict):
            operation = f.get("operation")
            if isinstance(operation, dict):
                producible |= set(m for m in (operation.get("output") or []) if isinstance(m, str))
    if product_name in producible:
        raise ValueError(
            f"'{product_name}' is already producible -- this issue must be from a "
            f"different check (e.g. product_has_customer or end_to_end_path) sharing "
            f"the same location shape. Not acting."
        )

    manufacturing_facilities = [
        (i, f.get("name"), (f.get("operation") or {}).get("output") or [])
        for i, f in enumerate(config.get("facility", []) or [])
        if isinstance(f, dict) and f.get("type") == "manufacturing" and isinstance(f.get("name"), str)
    ]

    fallback_text = f"Product '{product_name}' is not produced by any facility. Choose a fix:"
    print(_ask(config, issue.location, fallback_text))
    print("    0) Delete this product entirely")
    for i, (facility_idx, name, current_output) in enumerate(manufacturing_facilities, start=1):
        output_hint = f" (currently makes: {', '.join(current_output)})" if current_output else " (no output yet)"
        print(f"    {i}) Add it to existing facility '{name}'s operation.output{output_hint}")
    new_facility_option = len(manufacturing_facilities) + 1
    print(f"    {new_facility_option}) Create a new manufacturing facility for it")

    while True:
        choice = input("  Enter number: ").strip()
        if choice == "0":
            if _offer_delete_material_option(config, product_name):
                return config
            continue
        if not choice.isdigit() or not (1 <= int(choice) <= new_facility_option):
            print(f"    '{choice}' is not a valid selection. Try again.")
            continue
        break

    choice_num = int(choice)

    if choice_num == new_facility_option:
        print("Creating a new manufacturing facility.")
        print(f"  (Remember to include '{product_name}' when selecting inventory_managed below.)")
        new_entry, _ = _build_new_entry(config, "facility", {"type": "manufacturing"}, "facility")

        _add_to_inventory_managed(new_entry, product_name)
        operation = new_entry.setdefault("operation", {})
        if product_name not in (operation.get("output") or []):
            operation.setdefault("output", []).append(product_name)
        print(f"  Ensured '{product_name}' is in the new facility's inventory_managed and operation.output.")
    else:
        facility_idx, facility_name, _ = manufacturing_facilities[choice_num - 1]
        facility = config["facility"][facility_idx]
        _add_to_inventory_managed(facility, product_name)
        operation = facility.setdefault("operation", {})
        if product_name not in (operation.get("output") or []):
            operation.setdefault("output", []).append(product_name)
        print(f"  Added '{product_name}' to '{facility_name}'s inventory_managed and operation.output.")

    return config


def repair_product_missing_customer(config: dict, issue: ValidationIssue) -> dict:
    """
    Layer3 check 4: a product has no customer ordering it.
    """
    product_name = issue.context.get("referenced_name") if issue.context else None
    if not product_name:
        raise ValueError(f"Issue at '{issue.location}' has no referenced_name in its context.")

    ordered_products = {
        c.get("product") for c in config.get("customer", []) or []
        if isinstance(c, dict)
    }
    if product_name in ordered_products:
        raise ValueError(
            f"'{product_name}' already has a customer -- this issue must be from a "
            f"different check (e.g. product_is_producible or end_to_end_path) sharing "
            f"the same location shape. Not acting."
        )

    print(_ask(config, issue.location, f"Product '{product_name}' has no customer."))
    choice = input("  Type 'delete' to remove this product entirely, or press Enter to "
                    "create a customer for it: ").strip().lower()
    if choice == "delete":
        if _offer_delete_material_option(config, product_name):
            return config
        print("  Cancelled -- creating a customer instead.")

    _build_new_entry(config, "customer", {"product": product_name}, "customer")

    return config


def repair_at_least_one_raw_material(config: dict, issue: ValidationIssue) -> dict:
    """
    Layer3 check 2: raw_materials is empty (or absent).
    """
    print(_ask(config, issue.location, "No raw materials declared -- creating one."))
    _build_new_entry(config, "raw_materials", {}, "raw_materials")
    return config


def repair_at_least_one_product(config: dict, issue: ValidationIssue) -> dict:
    """
    Layer3 check 3: products is empty (or absent).
    """
    print(_ask(config, issue.location, "No products declared -- creating one."))
    _build_new_entry(config, "products", {}, "products")
    return config


def repair_material_missing_inventory_entry(config: dict, issue: ValidationIssue) -> dict:
    """
    Layer2 check 20: a declared material has no corresponding inventory[]
    entry.
    """
    name = issue.context.get("referenced_name") if issue.context else None
    expected_type = issue.context.get("expected_type") if issue.context else None
    if not name or not expected_type:
        raise ValueError(f"Issue at '{issue.location}' has no referenced_name/expected_type in its context.")

    if name in _collect_names(config, "inventory"):
        raise ValueError(
            f"'{name}' already has an inventory entry -- this issue must be from a "
            f"different check sharing the same location shape. Not acting."
        )

    print(_ask(config, issue.location, f"'{name}' has no inventory entry -- creating one."))
    _build_new_entry(config, "inventory", {"name": name, "type": expected_type}, "inventory")

    return config


def repair_raw_material_missing_supplier(config: dict, issue: ValidationIssue) -> dict:
    """
    Layer3 check 1: a raw material has no supplier.
    """
    material_name = issue.context.get("referenced_name") if issue.context else None
    if not material_name:
        raise ValueError(f"Issue at '{issue.location}' has no referenced_name in its context.")

    existing_suppliers = {
        s.get("supply_material_name") for s in config.get("supplier", []) or []
        if isinstance(s, dict)
    }
    if material_name in existing_suppliers:
        raise ValueError(
            f"'{material_name}' already has a supplier -- this issue must be from a "
            f"different check (e.g. check_raw_material_is_consumed) sharing the same "
            f"location shape, not check_raw_material_has_supplier. Not creating a "
            f"redundant supplier."
        )

    print(_ask(config, issue.location, f"Raw material '{material_name}' has no supplier."))
    print(f"  1) Create a new supplier for it")
    print(f"  2) Delete this material entirely instead")
    choice = input("  Enter number: ").strip()
    if choice == "2":
        if _offer_delete_material_option(config, material_name):
            return config
        print("  Cancelled -- creating a supplier instead.")

    _build_new_entry(config, "supplier", {"supply_material_name": material_name}, "supplier")

    return config


def repair_raw_material_not_consumed(config: dict, issue: ValidationIssue) -> dict:
    """
    Layer3 check 7: a raw material is never consumed.
    """
    material_name = issue.context.get("referenced_name") if issue.context else None
    if not material_name:
        raise ValueError(f"Issue at '{issue.location}' has no referenced_name in its context.")

    consumed = set()
    for section in ("intermediate_materials", "products"):
        for entry in config.get(section, []) or []:
            if isinstance(entry, dict) and isinstance(entry.get("bom"), dict):
                consumed |= set(entry["bom"].keys())
    for facility in config.get("facility", []) or []:
        if isinstance(facility, dict):
            operation = facility.get("operation")
            if isinstance(operation, dict):
                consumed |= set(m for m in (operation.get("input") or []) if isinstance(m, str))

    if material_name in consumed:
        raise ValueError(
            f"'{material_name}' is already consumed somewhere -- this issue must be from "
            f"a different check (e.g. check_raw_material_has_supplier) sharing the same "
            f"location shape. Not acting."
        )

    candidates = []
    for section in ("intermediate_materials", "products"):
        for idx, entry in enumerate(config.get(section, []) or []):
            if isinstance(entry, dict) and isinstance(entry.get("name"), str):
                candidates.append((section, idx, entry["name"]))

    if not candidates:
        raise ValueError(
            "No intermediate_materials or products exist yet for this raw material to "
            "be an ingredient of."
        )

    print(_ask(config, issue.location, f"Raw material '{material_name}' is never consumed -- choose which recipe uses it:"))
    print(f"    0) Delete this material entirely instead")
    for i, (section, idx, name) in enumerate(candidates, start=1):
        print(f"    {i}) {name} ({section})")
    while True:
        choice = input("  Enter number: ").strip()
        if choice == "0":
            if _offer_delete_material_option(config, material_name):
                return config
            continue
        if not choice.isdigit() or not (1 <= int(choice) <= len(candidates)):
            print(f"    '{choice}' is not a valid selection. Try again.")
            continue
        break

    section, idx, name = candidates[int(choice) - 1]
    qty = _prompt_for_value(f"{section}[{idx}].bom.{material_name}", "num")
    config[section][idx].setdefault("bom", {})[material_name] = qty
    print(f"  Added '{material_name}' (qty {qty}) to {name}'s bom.")

    producing_facility = None
    for f in config.get("facility", []) or []:
        if not isinstance(f, dict):
            continue
        operation = f.get("operation")
        if isinstance(operation, dict) and name in (operation.get("output") or []):
            producing_facility = f
            break

    if producing_facility is not None:
        operation = producing_facility["operation"]
        if material_name not in (operation.get("input") or []):
            operation.setdefault("input", []).append(material_name)
        _add_to_inventory_managed(producing_facility, material_name)
        print(f"  Also added '{material_name}' to '{producing_facility.get('name')}'s "
              f"operation.input and inventory_managed, to keep the recipe and the "
              f"physical operation consistent.")
    else:
        print(f"  NOTE: no facility currently produces '{name}' -- could not sync "
              f"'{material_name}' into any operation.input. This will need attention "
              f"once a producing facility exists.")

    return config


def repair_edge_phantom_delivery(config: dict, issue: ValidationIssue) -> dict:
    """
    Layer2 check 19: an edge delivers a CATEGORY-VALID material to a
    facility that doesn't manage or consume it at all.
    """
    steps = _parse_location_steps(issue.location)
    parent, _ = _navigate_to_parent(config, steps)
    material_name = parent.get("material_name")
    current_destination = parent.get("destination")

    if not material_name or not current_destination:
        raise ValueError(f"Edge at '{issue.location}' is missing material_name/destination.")

    matching_facilities = [
        (i, f) for i, f in enumerate(config.get("facility", []) or [])
        if isinstance(f, dict) and f.get("name") == current_destination
    ]
    if not matching_facilities:
        raise ValueError(
            f"Destination '{current_destination}' is not a facility -- this issue must be "
            f"from a different check."
        )

    if len(matching_facilities) == 1:
        current_facility_idx, current_facility = matching_facilities[0]
    else:
        print(_ask(config, issue.location, f"  Multiple facilities are named '{current_destination}'. Which one does this edge mean?"))
        for i, (idx, f) in enumerate(matching_facilities, start=1):
            current_output = (f.get("operation") or {}).get("output") or []
            hint = f"makes: {', '.join(current_output)}" if current_output else "no output yet"
            print(f"    {i}) entry {idx} ({hint})")
        while True:
            choice = input("  Enter number: ").strip()
            if choice.isdigit() and 1 <= int(choice) <= len(matching_facilities):
                break
            print(f"    '{choice}' is not a valid selection. Try again.")
        current_facility_idx, current_facility = matching_facilities[int(choice) - 1]

    managed = current_facility.get("inventory_managed") or []
    inputs = (current_facility.get("operation") or {}).get("input") or []
    if material_name in managed or material_name in inputs:
        raise ValueError(f"'{current_destination}' already references '{material_name}' -- issue may be stale.")

    edge_source = parent.get("source")

    alt_facilities = []
    for f in config.get("facility", []) or []:
        if not isinstance(f, dict):
            continue
        name = f.get("name")
        if name == current_destination or not isinstance(name, str):
            continue
        f_managed = f.get("inventory_managed") or []
        f_inputs = (f.get("operation") or {}).get("input") or []
        if material_name not in f_managed and material_name not in f_inputs:
            continue

        already_has_this_edge = any(
            isinstance(e, dict) and e.get("source") == edge_source
            and e.get("destination") == name and e.get("material_name") == material_name
            for e in config.get("edges", []) or []
        )
        if already_has_this_edge:
            continue

        alt_facilities.append(name)

    fallback_text = f"Edge delivers '{material_name}' to '{current_destination}', which doesn't use it. Choose a fix:"
    print(_ask(config, issue.location, fallback_text))
    print(f"    0) Delete this material entirely instead")
    options = [("retarget", name) for name in alt_facilities]
    options.append(("sync", current_destination))

    for i, (kind, name) in enumerate(options, start=1):
        if kind == "retarget":
            print(f"    {i}) Retarget this edge's destination to '{name}' (already uses '{material_name}')")
        else:
            print(f"    {i}) Keep destination '{name}' (recommended if multiple destinations "
                  f"legitimately need this material -- e.g. separate regional warehouses), "
                  f"add '{material_name}' to its inventory_managed/operation.input")

    while True:
        choice = input("  Enter number: ").strip()
        if choice == "0":
            if _offer_delete_material_option(config, material_name):
                return config
            continue
        if not choice.isdigit() or not (1 <= int(choice) <= len(options)):
            print(f"    '{choice}' is not a valid selection. Try again.")
            continue
        break

    kind, name = options[int(choice) - 1]
    if kind == "retarget":
        parent["destination"] = name
        print(f"  Retargeted edge destination to '{name}'.")
    else:
        operation = current_facility.setdefault("operation", {})
        if material_name not in (operation.get("input") or []):
            operation.setdefault("input", []).append(material_name)
        _add_to_inventory_managed(current_facility, material_name)
        print(f"  Added '{material_name}' to '{name}'s operation.input and inventory_managed.")

    return config


SECTION_TO_MATERIAL_TYPE = {v: k for k, v in INVENTORY_TYPE_TO_SECTION.items()}


def _material_type_for(config: dict, material_name: str):
    """Reverse lookup: given a material name, which category does it
    belong to? None if not found."""
    for section, mtype in SECTION_TO_MATERIAL_TYPE.items():
        if material_name in _collect_names(config, section):
            return mtype
    return None


def _create_edge(config: dict, source: str, destination: str, material_name: str, material_type: str, location_hint: str):
    """
    Shared helper: appends a new edge and fills its transfer_time.
    """
    new_edge = {
        "source": source,
        "destination": destination,
        "material_type": material_type,
        "material_name": material_name,
    }
    config.setdefault("edges", []).append(new_edge)
    idx = len(config["edges"]) - 1

    is_customer_destination = destination in _collect_names(config, "customer")
    if is_customer_destination:
        new_edge["transfer_time"] = {"distribution": "constant", "parameters": {"a": 0}}
        print(f"  Created edge: {source} -> {destination} (material: {material_name}, "
              f"transfer_time forced to 0 -- delivery lead time is covered by "
              f"customer_lead_time, not edge transfer_time).")
    else:
        transfer_fspec = SECTION_SPECS["edges"]["fields"]["transfer_time"]
        _fill_node(config, new_edge, "transfer_time", transfer_fspec,
                   f"edges[{idx}].transfer_time", "edges.transfer_time", new_edge)
        print(f"  Created edge: {source} -> {destination} (material: {material_name}).")


def _fix_facility_missing_inbound(config: dict, facility: dict):
    """
    Finds candidate (source, material) pairs that could feed this
    facility.
    """
    facility_name = facility.get("name")
    managed = facility.get("inventory_managed") or []
    own_output = set((facility.get("operation") or {}).get("output") or [])
    receivable = [m for m in managed if m not in own_output]

    if not receivable:
        raise ValueError(
            f"'{facility_name}' has no inventory_managed items it doesn't already produce "
            f"itself -- cannot determine what it should receive (check inventory_managed "
            f"is complete)."
        )

    options = []
    is_warehouse = facility.get("type") == "warehouse"
    for m in receivable:
        if not is_warehouse:
            for s in config.get("supplier", []) or []:
                if isinstance(s, dict) and s.get("supply_material_name") == m and isinstance(s.get("name"), str):
                    options.append((s["name"], m))
        for f in config.get("facility", []) or []:
            if isinstance(f, dict) and f.get("name") != facility_name:
                op = f.get("operation") or {}
                if m in (op.get("output") or []) and isinstance(f.get("name"), str):
                    options.append((f["name"], m))

    def _pick_receivable_material(prompt: str) -> str:
        if len(receivable) == 1:
            return receivable[0]
        print(f"  {prompt}")
        for i, m in enumerate(receivable, start=1):
            print(f"    {i}) {m}")
        while True:
            choice = input("  Enter number: ").strip()
            if choice.isdigit() and 1 <= int(choice) <= len(receivable):
                return receivable[int(choice) - 1]
            print(f"    '{choice}' is not a valid selection. Try again.")

    def _create_new_source_flow():
        material = _pick_receivable_material("Which material should the new source provide?")
        material_type = _material_type_for(config, material)
        receiving_facility_type = facility.get("type")

        if receiving_facility_type != "warehouse" and material_type == "raw_materials":
            _build_new_entry(config, "supplier", {"supply_material_name": material}, "supplier")
            new_source_name = config["supplier"][-1]["name"]
        else:
            if receiving_facility_type == "warehouse":
                print(f"  '{facility_name}' is a warehouse -- it can only receive from a "
                      f"manufacturing facility, never directly from a supplier. Creating a "
                      f"new producing facility for '{material}'.")
            else:
                print(f"  '{material}' is a {material_type or 'material'}, not a raw material -- "
                      f"creating a new producing facility for it instead of a supplier.")
            _build_new_entry(config, "facility", {"type": "manufacturing"}, "facility")
            new_facility = config["facility"][-1]
            new_source_name = new_facility["name"]
            new_facility.setdefault("inventory_managed", [])
            if material not in new_facility["inventory_managed"]:
                new_facility["inventory_managed"].append(material)
            op = new_facility.get("operation")
            if not isinstance(op, dict):
                op = {}
                new_facility["operation"] = op
            op.setdefault("output", [])
            if material not in op["output"]:
                op["output"].append(material)

        _create_edge(config, new_source_name, facility_name, material,
                     material_type, "edges")

    def _delete_this_facility_flow():
        removed = _delete_facility_and_associations(config, facility_name)
        print(f"  Deleted '{facility_name}': {', '.join(removed) if removed else 'nothing found to remove'}.")

    fallback_text = f"'{facility_name}' has no inbound edge. Choose a source:"
    print(_ask(config, f"facility.{facility_name}.inbound", fallback_text))
    for i, (src, mat) in enumerate(options, start=1):
        print(f"    {i}) {src} -> {facility_name}  (material: {mat})")
    create_new_num = len(options) + 1
    delete_facility_num = len(options) + 2
    print(f"    {create_new_num}) Create a new source")
    print(f"    {delete_facility_num}) Delete '{facility_name}' entirely instead")
    while True:
        choice = input("  Enter number: ").strip()
        if not choice.isdigit():
            print(f"    '{choice}' is not a valid selection. Try again.")
            continue
        choice_num = int(choice)
        if 1 <= choice_num <= len(options):
            break
        if choice_num == create_new_num:
            _create_new_source_flow()
            return
        if choice_num == delete_facility_num:
            _delete_this_facility_flow()
            return
        print(f"    '{choice}' is not a valid selection. Try again.")

    source, material = options[int(choice) - 1]
    material_type = _material_type_for(config, material)
    _create_edge(config, source, facility_name, material, material_type, "edges")


def _fix_facility_missing_outbound(config: dict, facility: dict):
    """
    Finds candidate (destination, material) pairs for this facility's
    outbound edge.
    """
    facility_name = facility.get("name")
    own_output = set((facility.get("operation") or {}).get("output") or [])
    sendable = list(own_output) if own_output else list(facility.get("inventory_managed") or [])

    if not sendable:
        raise ValueError(
            f"'{facility_name}' has nothing in operation.output or inventory_managed "
            f"to send anywhere."
        )

    options = []
    for m in sendable:
        for f in config.get("facility", []) or []:
            if isinstance(f, dict) and f.get("name") != facility_name:
                f_managed = f.get("inventory_managed") or []
                f_inputs = (f.get("operation") or {}).get("input") or []
                if (m in f_managed or m in f_inputs) and isinstance(f.get("name"), str):
                    options.append((f["name"], m))
        for c in config.get("customer", []) or []:
            if isinstance(c, dict) and c.get("product") == m and isinstance(c.get("name"), str):
                options.append((c["name"], m))

    def _pick_sendable_material(prompt: str) -> str:
        if len(sendable) == 1:
            return sendable[0]
        print(f"  {prompt}")
        for i, m in enumerate(sendable, start=1):
            print(f"    {i}) {m}")
        while True:
            choice = input("  Enter number: ").strip()
            if choice.isdigit() and 1 <= int(choice) <= len(sendable):
                return sendable[int(choice) - 1]
            print(f"    '{choice}' is not a valid selection. Try again.")

    def _create_new_destination_flow():
        material = _pick_sendable_material("Which material should the new destination take?")
        material_type = _material_type_for(config, material)

        if material_type == "products":
            _build_new_entry(config, "customer", {"product": material}, "customer")
            new_dest_name = config["customer"][-1]["name"]
        else:
            print(f"  '{material}' is a {material_type or 'material'}, not a finished product -- "
                  f"creating a new consuming facility for it instead of a customer.")
            _build_new_entry(config, "facility", {"type": "manufacturing"}, "facility")
            new_facility = config["facility"][-1]
            new_dest_name = new_facility["name"]
            new_facility.setdefault("inventory_managed", [])
            if material not in new_facility["inventory_managed"]:
                new_facility["inventory_managed"].append(material)
            op = new_facility.get("operation")
            if not isinstance(op, dict):
                op = {}
                new_facility["operation"] = op
            op.setdefault("input", [])
            if material not in op["input"]:
                op["input"].append(material)

        _create_edge(config, facility_name, new_dest_name, material, material_type, "edges")

    def _delete_this_facility_flow():
        removed = _delete_facility_and_associations(config, facility_name)
        print(f"  Deleted '{facility_name}': {', '.join(removed) if removed else 'nothing found to remove'}.")

    fallback_text = f"'{facility_name}' has no outbound edge. Choose a destination:"
    print(_ask(config, f"facility.{facility_name}.outbound", fallback_text))
    for i, (dst, mat) in enumerate(options, start=1):
        print(f"    {i}) {facility_name} -> {dst}  (material: {mat})")
    create_new_num = len(options) + 1
    delete_facility_num = len(options) + 2
    print(f"    {create_new_num}) Create a new destination")
    print(f"    {delete_facility_num}) Delete '{facility_name}' entirely instead")
    while True:
        choice = input("  Enter number: ").strip()
        if not choice.isdigit():
            print(f"    '{choice}' is not a valid selection. Try again.")
            continue
        choice_num = int(choice)
        if 1 <= choice_num <= len(options):
            break
        if choice_num == create_new_num:
            _create_new_destination_flow()
            return
        if choice_num == delete_facility_num:
            _delete_this_facility_flow()
            return
        print(f"    '{choice}' is not a valid selection. Try again.")

    destination, material = options[int(choice) - 1]
    material_type = _material_type_for(config, material)
    _create_edge(config, facility_name, destination, material, material_type, "edges")


def repair_customer_missing_inbound_edge(config: dict, issue: ValidationIssue) -> dict:
    """
    Layer3 check 10: a customer has no inbound edge.
    """
    customer_name = issue.context.get("referenced_name") if issue.context else None
    entry_index = issue.context.get("entry_index") if issue.context else None
    if not customer_name or entry_index is None:
        raise ValueError(f"Issue at '{issue.location}' has no referenced_name/entry_index in its context.")

    customers = config.get("customer", []) or []
    if entry_index >= len(customers) or not isinstance(customers[entry_index], dict):
        raise ValueError(f"'{issue.location}' does not resolve to a real customer entry.")
    customer_entry = customers[entry_index]
    if customer_entry.get("name") != customer_name:
        raise ValueError(f"Customer at index {entry_index} no longer matches '{customer_name}' -- issue may be stale.")

    product_name = customer_entry.get("product")
    if not product_name:
        raise ValueError(f"Customer '{customer_name}' has no product set -- cannot determine what to deliver.")

    has_inbound = any(
        isinstance(e, dict) and e.get("destination") == customer_name and e.get("material_name") == product_name
        for e in config.get("edges", []) or []
    )
    if has_inbound:
        raise ValueError(f"'{customer_name}' (ordering '{product_name}') already has an inbound edge -- issue may be stale.")

    warehouse_options = []
    manufacturing_options = []
    for f in config.get("facility", []) or []:
        if not isinstance(f, dict) or not isinstance(f.get("name"), str):
            continue
        if f.get("type") == "warehouse" and product_name in (f.get("inventory_managed") or []):
            warehouse_options.append(f["name"])
        elif f.get("type") == "manufacturing" and product_name in ((f.get("operation") or {}).get("output") or []):
            manufacturing_options.append(f["name"])

    if not warehouse_options and not manufacturing_options:
        raise ValueError(
            f"No warehouse or manufacturing facility currently manages/produces "
            f"'{product_name}' -- cannot determine a delivery source yet."
        )

    fallback_text = f"Customer '{customer_name}' (orders '{product_name}') has no inbound edge. Choose a delivery source:"
    print(_ask(config, issue.location, fallback_text))
    options = [("warehouse", name) for name in warehouse_options] + [("manufacturing", name) for name in manufacturing_options]
    for i, (kind, name) in enumerate(options, start=1):
        label = "Warehouse" if kind == "warehouse" else "Manufacturing facility (via new hypothetical warehouse)"
        print(f"    {i}) {name} ({label})")

    while True:
        choice = input("  Enter number: ").strip()
        if not choice.isdigit() or not (1 <= int(choice) <= len(options)):
            print(f"    '{choice}' is not a valid selection. Try again.")
            continue
        break

    kind, name = options[int(choice) - 1]
    material_type = _material_type_for(config, product_name)

    if kind == "warehouse":
        _create_edge(config, name, customer_name, product_name, material_type, "edges")
    else:
        default_name = f"{name} Warehouse"
        entered = input(f"  Enter a name for the new hypothetical warehouse (default: '{default_name}'): ").strip()
        warehouse_name = entered if entered else default_name

        new_warehouse = {
            "name": warehouse_name,
            "type": "warehouse",
            "inventory_managed": [product_name],
        }
        config.setdefault("facility", []).append(new_warehouse)

        config.setdefault("edges", []).append({
            "source": name, "destination": warehouse_name,
            "material_type": material_type, "material_name": product_name,
            "transfer_time": {"distribution": "constant", "parameters": {"a": 0}},
        })
        config["edges"].append({
            "source": warehouse_name, "destination": customer_name,
            "material_type": material_type, "material_name": product_name,
            "transfer_time": {"distribution": "constant", "parameters": {"a": 0}},
        })
        print(f"  Created hypothetical warehouse '{warehouse_name}' with zero-transfer-time "
              f"edges: {name} -> {warehouse_name} -> {customer_name}.")

    return config


def _delete_facility_and_associations(config: dict, facility_name: str) -> list:
    """
    Removes a facility entirely.
    """
    removed = []

    before = len(config.get("facility", []) or [])
    config["facility"] = [
        f for f in config.get("facility", []) or []
        if not (isinstance(f, dict) and f.get("name") == facility_name)
    ]
    if len(config["facility"]) < before:
        removed.append("facility[] entry")

    nodes = config.get("nodes")
    if isinstance(nodes, list) and len(nodes) > 0 and isinstance(nodes[0], dict):
        if isinstance(nodes[0].get("facility"), list) and facility_name in nodes[0]["facility"]:
            nodes[0]["facility"] = [n for n in nodes[0]["facility"] if n != facility_name]
            removed.append("nodes[0].facility registration")

    before_edges = len(config.get("edges", []) or [])
    config["edges"] = [
        e for e in config.get("edges", []) or []
        if not (isinstance(e, dict) and (e.get("source") == facility_name or e.get("destination") == facility_name))
    ]
    if len(config["edges"]) < before_edges:
        removed.append(f"{before_edges - len(config['edges'])} edge(s)")

    return removed


def _prompt_unique_name(location: str, existing_names: set) -> str:
    """Shared helper: prompts for a name that isn't already in
    existing_names, re-prompting until a genuinely new name is given."""
    while True:
        candidate = _prompt_for_value(location, "name")
        if candidate in existing_names:
            print(f"    '{candidate}' is already used. Try again.")
            continue
        return candidate


def repair_material_category_collision(config: dict, issue: ValidationIssue) -> dict:
    """
    Layer2 check 13: the same name is declared in two different material
    category lists.
    """
    name = issue.context.get("referenced_name") if issue.context else None
    sections = issue.context.get("sections") if issue.context else None
    if not name or not sections or len(sections) != 2:
        raise ValueError(f"Issue at '{issue.location}' has no referenced_name/sections in its context.")

    names_by_section = {s: _collect_names(config, s) for s in sections}
    if name not in names_by_section[sections[0]] or name not in names_by_section[sections[1]]:
        raise ValueError(f"'{name}' no longer collides between {sections} -- issue may be stale.")

    fallback_text = f"'{name}' is declared in both {sections[0]} and {sections[1]}. Choose which to rename:"
    print(_ask(config, issue.location, fallback_text))
    print(f"    1) Rename the {sections[0]} entry")
    print(f"    2) Rename the {sections[1]} entry")
    while True:
        choice = input("  Enter number: ").strip()
        if choice in ("1", "2"):
            break
        print(f"    '{choice}' is not a valid selection. Try again.")

    target_section = sections[0] if choice == "1" else sections[1]
    all_names = set()
    for s in ("raw_materials", "intermediate_materials", "products"):
        all_names |= _collect_names(config, s)

    new_name = _prompt_unique_name(f"{target_section}[].name (renaming '{name}')", all_names)

    for entry in config.get(target_section, []) or []:
        if isinstance(entry, dict) and entry.get("name") == name:
            entry["name"] = new_name
            break

    print(f"  Renamed '{name}' -> '{new_name}' in {target_section}. NOTE: other references to "
          f"'{name}' elsewhere (bom keys, edges, customer.product, etc.) are NOT automatically "
          f"updated -- any that meant this entry will surface as a dangling reference on the "
          f"next check, which can then be resolved individually.")

    return config


def repair_duplicate_name_within_section(config: dict, issue: ValidationIssue) -> dict:
    """
    Layer2 check 15: the same name appears more than once within a
    single section.
    """
    name = issue.context.get("referenced_name") if issue.context else None
    section = issue.context.get("section") if issue.context else None
    if not name or not section:
        raise ValueError(f"Issue at '{issue.location}' has no referenced_name/section in its context.")

    entries = config.get(section, []) or []
    current_indices = [i for i, e in enumerate(entries) if isinstance(e, dict) and e.get("name") == name]
    if len(current_indices) < 2:
        raise ValueError(f"'{name}' is no longer duplicated in {section} -- issue may be stale.")

    fallback_text = (f"'{name}' appears {len(current_indices)} times in {section} "
                      f"(indices {current_indices}). Keeping index {current_indices[0]} as-is; renaming the rest:")
    print(_ask(config, issue.location, fallback_text))

    existing_names = set(e.get("name") for e in entries if isinstance(e, dict) and isinstance(e.get("name"), str))

    for idx in current_indices[1:]:
        new_name = _prompt_unique_name(f"{section}[{idx}].name (currently '{name}')", existing_names)
        entries[idx]["name"] = new_name
        existing_names.add(new_name)
        print(f"    Renamed {section}[{idx}] -> '{new_name}'.")

    print(f"  NOTE: other references to '{name}' elsewhere may now be ambiguous for the renamed "
          f"entries -- these will surface as dangling references on the next check.")

    return config


def repair_duplicate_edges(config: dict, issue: ValidationIssue) -> dict:
    """
    Layer2 check 16: the same (source, destination, material_name) triple
    appears more than once in edges[].
    """
    source = issue.context.get("source") if issue.context else None
    destination = issue.context.get("destination") if issue.context else None
    material_name = issue.context.get("material_name") if issue.context else None
    if not source or not destination or not material_name:
        raise ValueError(f"Issue at '{issue.location}' is missing source/destination/material_name in its context.")

    edges = config.get("edges", []) or []
    current_indices = [
        i for i, e in enumerate(edges)
        if isinstance(e, dict) and e.get("source") == source
        and e.get("destination") == destination and e.get("material_name") == material_name
    ]
    if len(current_indices) < 2:
        raise ValueError(
            f"Edge ({source} -> {destination}, {material_name}) is no longer duplicated -- issue may be stale."
        )

    fallback_text = (f"Edge ({source} -> {destination}, material '{material_name}') appears "
                      f"{len(current_indices)} times. Keeping the first (index {current_indices[0]}), removing the rest.")
    print(_ask(config, issue.location, fallback_text))
    for idx in sorted(current_indices[1:], reverse=True):
        del edges[idx]
        print(f"    Removed duplicate at edges[{idx}].")

    return config


def repair_self_loop_edge(config: dict, issue: ValidationIssue) -> dict:
    """
    Deletes an edge whose source and destination are the same node --
    structurally meaningless, no ambiguity to resolve, so this stays
    fully automatic (no prompt, informational print only, same as
    before).
    """
    match = re.match(r"^edges\[(\d+)\]$", issue.location)
    if not match:
        raise ValueError(f"'{issue.location}' does not look like an edges[N] entry.")
    idx = int(match.group(1))

    edges = config.get("edges", [])
    if not isinstance(edges, list) or idx >= len(edges):
        raise ValueError(f"No edge at index {idx}.")

    try:
        label = describe_location(config, f"edges[{idx}]")
    except Exception:
        label = issue.location

    removed = edges.pop(idx)

    material = removed.get("material_name")
    material_note = f" (material: '{material}')" if isinstance(material, str) and material and material != "missing" else ""
    print(f"  Removed self-loop edge at {label}{material_note} -- "
          f"a node cannot deliver to itself.")

    return config


def repair_supplier_facility_name_collision(config: dict, issue: ValidationIssue) -> dict:
    """
    Layer2 check 17: the same name is used as both a supplier and a
    facility.
    """
    name = issue.context.get("referenced_name") if issue.context else None
    if not name:
        raise ValueError(f"Issue at '{issue.location}' has no referenced_name in its context.")

    supplier_names = _collect_names(config, "supplier")
    facility_names = _collect_names(config, "facility")
    if name not in supplier_names or name not in facility_names:
        raise ValueError(f"'{name}' no longer collides between supplier and facility -- issue may be stale.")

    fallback_text = f"'{name}' is used as both a supplier name and a facility name. Choose which to rename:"
    print(_ask(config, issue.location, fallback_text))
    print("    1) Rename the supplier")
    print("    2) Rename the facility")
    while True:
        choice = input("  Enter number: ").strip()
        if choice in ("1", "2"):
            break
        print(f"    '{choice}' is not a valid selection. Try again.")

    section_label = "supplier" if choice == "1" else "facility"
    target_list = config.get(section_label, []) or []
    all_names = supplier_names | facility_names

    new_name = _prompt_unique_name(f"{section_label}[].name (renaming '{name}')", all_names)

    for entry in target_list:
        if isinstance(entry, dict) and entry.get("name") == name:
            entry["name"] = new_name
            break

    print(f"  Renamed {section_label} '{name}' -> '{new_name}'. NOTE: nodes[0].{section_label} "
          f"and any edges referencing '{name}' will now be dangling -- these will surface as "
          f"dangling references on the next check, which can then be resolved individually.")

    return config


def repair_facility_material_stage_span(config: dict, issue: ValidationIssue) -> dict:
    """
    Layer3 check 13: a manufacturing facility's inventory_managed spans
    only one material stage.
    """
    entry_index = issue.context.get("entry_index") if issue.context else None
    if entry_index is None:
        raise ValueError(f"Issue at '{issue.location}' has no entry_index in its context.")

    facilities = config.get("facility", []) or []
    if entry_index >= len(facilities) or not isinstance(facilities[entry_index], dict):
        raise ValueError(f"'{issue.location}' does not resolve to a real facility entry.")
    facility = facilities[entry_index]
    facility_name = facility.get("name")

    managed = facility.get("inventory_managed") or []
    stages_present = set()
    for m in managed:
        if isinstance(m, str):
            stage = _material_stage(config, m)
            if stage is not None:
                stages_present.add(stage)

    if len(stages_present) >= 2:
        raise ValueError(f"'{facility_name}' already spans multiple stages -- issue may be stale.")

    fallback_text = f"Manufacturing facility '{facility_name}' only manages material at one stage. Choose a fix:"
    print(_ask(config, issue.location, fallback_text))
    print("    0) Delete this facility entirely")
    print("    1) Change its type to 'warehouse' (it doesn't convert anything)")
    print("    2) Add an existing material from a different stage to its inventory_managed")

    while True:
        choice = input("  Enter number: ").strip()

        if choice == "0":
            confirm = input(
                f"  Type 'yes' to confirm deleting facility '{facility_name}' and its "
                f"edges/registration (cannot be undone), or press Enter to go back: "
            ).strip().lower()
            if confirm == "yes":
                removed = _delete_facility_and_associations(config, facility_name)
                print(f"  Deleted '{facility_name}'. Removed: {', '.join(removed) if removed else '(nothing else referenced it)'}")
                return config
            continue

        if choice == "1":
            facility["type"] = "warehouse"
            if "operation" in facility:
                del facility["operation"]
                print(f"  Removed stale 'operation' field (warehouses don't have one).")
            print(f"  Changed '{facility_name}' type to 'warehouse'.")
            return config

        if choice == "2":
            all_names = sorted(
                _collect_names(config, "raw_materials")
                | _collect_names(config, "intermediate_materials")
                | _collect_names(config, "products")
            )
            candidates = [m for m in all_names if m not in managed]
            if not candidates:
                print("  No other materials exist to add. Choose a different option.")
                continue

            stage_labels = {0: "raw", 1: "intermediate", 2: "product"}
            print("  Select a material to add:")
            for i, m in enumerate(candidates, start=1):
                label = stage_labels.get(_material_stage(config, m), "?")
                print(f"    {i}) {m} ({label})")

            while True:
                mchoice = input("  Enter number: ").strip()
                if not mchoice.isdigit() or not (1 <= int(mchoice) <= len(candidates)):
                    print(f"    '{mchoice}' is not a valid selection. Try again.")
                    continue
                break

            new_material = candidates[int(mchoice) - 1]
            _add_to_inventory_managed(facility, new_material)
            print(f"  Added '{new_material}' to '{facility_name}'s inventory_managed.")

            derived_input = _try_auto_derive_operation_list(config, facility, "input")
            derived_output = _try_auto_derive_operation_list(config, facility, "output")
            if derived_input is not None and derived_output is not None:
                operation = facility.setdefault("operation", {})
                operation["input"] = derived_input
                operation["output"] = derived_output
                print(f"  Re-derived operation.input={derived_input}, operation.output={derived_output}.")
            else:
                print(f"  NOTE: could not auto-derive operation.input/output from the updated "
                      f"inventory_managed (still ambiguous) -- update these manually if needed.")
            return config

        print(f"    '{choice}' is not a valid selection. Try again.")


def repair_horizon_sanity(config: dict, issue: ValidationIssue) -> dict:
    """
    Layer3 check 12 (WARNING only, never blocks): simulation.horizon is
    shorter than a rough estimate of one full supplier-to-customer cycle.
    Fully automatic (no ambiguity to resolve), informational print only.
    """
    estimated = issue.context.get("estimated_cycle_time") if issue.context else None
    current = issue.context.get("horizon") if issue.context else None
    if estimated is None or current is None:
        raise ValueError(f"Issue at '{issue.location}' is missing horizon/estimated_cycle_time in its context.")

    new_horizon = max(current, int(estimated * 1.5) + 1)
    print(f"simulation.horizon ({current}) is shorter than the estimated cycle time (~{estimated}). "
          f"Updating to {new_horizon}.")
    config.setdefault("simulation", {})["horizon"] = new_horizon

    return config


def repair_facility_missing_edge(config: dict, issue: ValidationIssue) -> dict:
    """
    Layer3 check 9: a facility has no inbound and/or no outbound edge.
    """
    facility_name = issue.context.get("referenced_name") if issue.context else None
    if not facility_name:
        raise ValueError(f"Issue at '{issue.location}' has no referenced_name in its context.")

    facility = next(
        (f for f in config.get("facility", []) or [] if isinstance(f, dict) and f.get("name") == facility_name),
        None,
    )
    if facility is None:
        raise ValueError(f"Could not find a facility entry named '{facility_name}'.")

    has_inbound = any(
        isinstance(e, dict) and e.get("destination") == facility_name for e in config.get("edges", []) or []
    )
    has_outbound = any(
        isinstance(e, dict) and e.get("source") == facility_name for e in config.get("edges", []) or []
    )

    if has_inbound and has_outbound:
        raise ValueError(f"'{facility_name}' already has both inbound and outbound edges -- issue may be stale.")

    made_progress = False
    errors = []

    if not has_inbound:
        try:
            _fix_facility_missing_inbound(config, facility)
            made_progress = True
        except ValueError as e:
            errors.append(str(e))

    still_exists = any(
        isinstance(f, dict) and f.get("name") == facility_name
        for f in config.get("facility", []) or []
    )

    if not has_outbound and still_exists:
        try:
            _fix_facility_missing_outbound(config, facility)
            made_progress = True
        except ValueError as e:
            errors.append(str(e))

    if not made_progress:
        raise ValueError("; ".join(errors))

    return config


def repair_supplier_missing_outbound_edge(config: dict, issue: ValidationIssue) -> dict:
    """
    Layer3 check 8: a supplier has no outbound edge.
    """
    supplier_name = issue.context.get("referenced_name") if issue.context else None
    if not supplier_name:
        raise ValueError(f"Issue at '{issue.location}' has no referenced_name in its context.")

    has_outbound = any(
        isinstance(e, dict) and e.get("source") == supplier_name
        for e in config.get("edges", []) or []
    )
    if has_outbound:
        raise ValueError(f"'{supplier_name}' already has an outbound edge -- issue may be stale.")

    supplier_entry = next(
        (s for s in config.get("supplier", []) or [] if isinstance(s, dict) and s.get("name") == supplier_name),
        None,
    )
    if supplier_entry is None:
        raise ValueError(f"Could not find a supplier entry named '{supplier_name}'.")
    material_name = supplier_entry.get("supply_material_name")

    print(_ask(config, issue.location, f"Supplier '{supplier_name}' has no outbound edge."))
    choice = input("  Type 'delete' to remove this material (and this supplier) entirely, "
                    "or press Enter to create an edge: ").strip().lower()
    if choice == "delete":
        if _offer_delete_material_option(config, material_name):
            return config
        print("  Cancelled -- creating an edge instead.")

    all_facility_names = sorted(_collect_names(config, "facility"))
    if not all_facility_names:
        raise ValueError("No facilities exist yet to connect this supplier to.")

    referencing_facilities = []
    for f in config.get("facility", []) or []:
        if not isinstance(f, dict):
            continue
        name = f.get("name")
        managed = f.get("inventory_managed") or []
        operation = f.get("operation") or {}
        inputs = operation.get("input") or []
        if material_name in managed or material_name in inputs:
            if isinstance(name, str):
                referencing_facilities.append(name)

    if len(referencing_facilities) == 1:
        destination = referencing_facilities[0]
        print(f"  Destination auto-selected: '{destination}' (the only facility that "
              f"manages or consumes '{material_name}').")
    elif len(referencing_facilities) > 1:
        destination = _prompt_select_single(
            f"edges[].destination (facilities that use '{material_name}')", sorted(referencing_facilities),
            config=config,
        )
    else:
        print(f"  WARNING: no facility currently manages or consumes '{material_name}' -- "
              f"make sure a facility's inventory_managed/operation.input gets updated to "
              f"actually use it, or this delivery will go nowhere useful.")
        destination = _prompt_select_single(
            f"edges[].destination (for supplier '{supplier_name}')", all_facility_names, config=config
        )

    new_edge = {
        "source": supplier_name,
        "destination": destination,
        "material_type": "raw_materials",
        "material_name": material_name,
    }
    config.setdefault("edges", []).append(new_edge)
    idx = len(config["edges"]) - 1

    transfer_fspec = SECTION_SPECS["edges"]["fields"]["transfer_time"]
    _fill_node(config, new_edge, "transfer_time", transfer_fspec,
               f"edges[{idx}].transfer_time", "edges.transfer_time", new_edge)

    return config


def repair_missing_node(config: dict, issue: ValidationIssue) -> dict:
    """
    Repairs a MISSING_REQUIRED_VALUE issue whose location points at a
    CONTAINER rather than a plain scalar. Reconstructs the entire
    missing subtree via _fill_node.
    """
    if issue.defect_type != DefectType.MISSING_REQUIRED_VALUE:
        raise ValueError(
            f"repair_missing_node only handles MISSING_REQUIRED_VALUE, got {issue.defect_type}"
        )

    normalized = normalize_location(issue.location)
    fspec = find_spec_node(normalized)

    if fspec is None:
        raise ValueError(f"Could not find a schema spec node for '{issue.location}'.")

    steps = _parse_location_steps(issue.location)
    parent, last_key = _navigate_to_parent(config, steps)

    section_entry = None
    if len(steps) >= 2 and isinstance(steps[0], str) and isinstance(steps[1], int):
        section_entry = config.get(steps[0], [None])[steps[1]] if steps[1] < len(config.get(steps[0], [])) else None

    _fill_node(config, parent, last_key, fspec, issue.location, normalized, section_entry)

    return config


def repair_procurement_scheme_field(config: dict, issue: ValidationIssue) -> dict:
    """
    Handles ANY issue whose location involves procurement_scheme.
    """
    match = re.search(r"^(.*?procurement_scheme)(\..*)?$", issue.location)
    if not match:
        raise ValueError(f"'{issue.location}' does not contain a procurement_scheme segment.")
    ps_location = match.group(1)
    ps_steps = _parse_location_steps(ps_location)
    owning_entry, key = _navigate_to_parent(config, ps_steps)

    existing = owning_entry.get(key)
    obj = existing if isinstance(existing, dict) else {}
    owning_entry[key] = obj

    current_type = obj.get("type")
    if current_type in PROCUREMENT_SCHEME_TYPES:
        type_val = current_type
    else:
        type_val = _prompt_for_value(f"{ps_location}.type", "str", enum_values=PROCUREMENT_SCHEME_TYPES, config=config)

    new_obj = {"type": type_val}

    if type_val == "periodic_supply":
        old_dist = obj.get("distribution")
        old_params = obj.get("parameters") if isinstance(obj.get("parameters"), dict) else {}
        if old_dist in DISTRIBUTION_PARAM_COUNTS:
            dist_value = old_dist
            param_count = DISTRIBUTION_PARAM_COUNTS.get(dist_value, 1)
            params = {}
            for i, pkey in enumerate(PARAM_KEYS):
                if i < param_count:
                    val = old_params.get(pkey)
                    if val is None or val == "missing":
                        val = _prompt_for_value(f"{ps_location}.parameters.{pkey}", "num", config=config)
                    params[pkey] = val
        else:
            dist_value, params = _prompt_for_distribution(ps_location, config=config, allow_instant=False)
        new_obj["distribution"] = dist_value
        new_obj["parameters"] = params

    elif type_val == "demand_driven":
        pass

    elif type_val == "inventory_threshold":
        old_params = obj.get("parameters") if isinstance(obj.get("parameters"), dict) else {}
        params = {}
        for pkey in ("a", "b"):
            val = old_params.get(pkey)
            if val is None or val == "missing":
                val = _prompt_for_value(f"{ps_location}.parameters.{pkey}", "num", config=config)
            params[pkey] = val
        new_obj["parameters"] = params

    owning_entry[key] = new_obj
    return config


def repair_transfer_time_default_instant(config: dict, issue: ValidationIssue) -> dict:
    """
    Per explicit instruction: edges[].transfer_time is ALWAYS defaulted
    to instant automatically -- no prompt at all, informs only. Notice
    is a plain sentence naming source/destination, not a technical
    breadcrumb path.
    """
    match = re.search(r"^(edges\[\d+\]\.transfer_time)(\..*)?$", issue.location)
    if not match:
        raise ValueError(f"'{issue.location}' does not look like an edges[].transfer_time field.")
    tt_location = match.group(1)
    tt_steps = _parse_location_steps(tt_location)
    owning_edge, key = _navigate_to_parent(config, tt_steps)
    owning_edge[key] = {"distribution": "constant", "parameters": {"a": 0}}

    src = owning_edge.get("source", "?")
    dst = owning_edge.get("destination", "?")

    print(f"  Assuming the delivery from '{src}' to '{dst}' is instant, since no "
          f"transfer time was given. (Use the what-if feature later if you need "
          f"to model an actual delay for this edge.)")

    return config


def repair_scalar_missing_field(config: dict, issue: ValidationIssue) -> dict:
    """
    Repairs a MISSING_REQUIRED_VALUE issue on a plain scalar field.
    """
    if issue.defect_type != DefectType.MISSING_REQUIRED_VALUE:
        raise ValueError(
            f"repair_scalar_missing_field only handles MISSING_REQUIRED_VALUE, "
            f"got {issue.defect_type}"
        )

    normalized = normalize_location(issue.location)
    expected_kind = _lookup_field_type(normalized)

    if expected_kind is None:
        raise ValueError(
            f"'{issue.location}' (normalized: '{normalized}') is not a known scalar "
            f"field -- it may be dict/list-shaped (use a different repair action) "
            f"or missing from SCALAR_FIELD_TYPES."
        )

    enum_values = _lookup_enum_values(normalized)

    steps = _parse_location_steps(issue.location)
    parent, last_key = _navigate_to_parent(config, steps)

    if normalized in ("edges.destination", "edges.source"):
        edge_match = re.match(r"^edges\[(\d+)\]\.", issue.location)
        if edge_match:
            edge_idx = int(edge_match.group(1))
            candidates = (SCALAR_SELECT_CANDIDATE_FNS.get(normalized) or (lambda c, p: []))(config, parent)
            try:
                edge_label = describe_location(config, f"edges[{edge_idx}]")
            except Exception:
                edge_label = f"edges[{edge_idx}]"

            fallback_text = f"{edge_label}: Select a value, or delete this edge entirely."
            print(_ask(config, issue.location, fallback_text))
            print("    0) Delete this edge entirely")
            for i, name in enumerate(candidates, start=1):
                print(f"    {i}) {name}")

            while True:
                choice = input("  Enter number: ").strip()
                if choice == "0":
                    edges = config.get("edges", [])
                    removed = edges.pop(edge_idx)
                    mat = removed.get("material_name")
                    mat_note = f" (material: '{mat}')" if isinstance(mat, str) and mat and mat != "missing" else ""
                    print(f"  Deleted edge at {edge_label}{mat_note}.")
                    return config
                if choice.isdigit() and 1 <= int(choice) <= len(candidates):
                    parent[last_key] = candidates[int(choice) - 1]
                    return config
                print(f"    '{choice}' is not a valid selection. Try again.")

    select_fn = SCALAR_SELECT_CANDIDATE_FNS.get(normalized)
    if select_fn is not None:
        candidates = select_fn(config, parent)
        allow_none = normalized in SCALAR_SELECT_ALLOW_NONE
        if candidates or allow_none:
            fallback_text = f"Field '{issue.location}' is missing -- select its value below."
            print(_ask(config, issue.location, fallback_text))
            selected = _prompt_select_single(issue.location, candidates, allow_none=allow_none, config=config)
            if selected is None:
                return config
            parent[last_key] = selected
            return config
        print(f"  No candidates available yet for '{issue.location}' -- falling back to manual entry.")

    if normalized.endswith(".distribution") and enum_values == DISTRIBUTION_TYPES:
        dist_location = issue.location[: -len(".distribution")]
        dist_field = normalized[: -len(".distribution")]
        dist_value, params = _prompt_for_distribution(
            dist_location, config=config,
            allow_instant=(dist_field not in QUANTITY_BASED_DISTRIBUTION_FIELDS
                           and dist_field not in NO_INSTANT_RISK_FIELDS),
        )
        parent[last_key] = dist_value
        if isinstance(parent, dict):
            parent["parameters"] = params
        return config

    value = _prompt_for_value(issue.location, expected_kind, enum_values=enum_values, config=config)
    parent[last_key] = value

    return config


def repair_bom_value(config: dict, issue: ValidationIssue) -> dict:
    """
    Repairs a MISSING_REQUIRED_VALUE issue on a single bom ENTRY whose
    key already exists but whose value is still the "missing" placeholder.
    """
    steps = _parse_location_steps(issue.location)
    parent, key = _navigate_to_parent(config, steps)
    value = _prompt_for_value(issue.location, "num", config=config)
    parent[key] = value
    return config


def repair_config_info_version(config: dict, issue: ValidationIssue) -> dict:
    """
    config_info.version is never asked from the user -- it's an internal
    revision marker, not scenario data. Base runs always get "1.0"
    automatically; each what-if run bumps it (1.1, 1.2, ...) directly in
    Pipeline.run_whatif before validation, so this action only ever fires
    for a fresh base config that hasn't been assigned a version yet.
    """
    steps = _parse_location_steps(issue.location)
    parent, key = _navigate_to_parent(config, steps)
    parent[key] = "1.0"
    print("  Assigned config version 1.0 (auto-assigned, not asked).")
    return config


def repair_invalid_enum_value(config: dict, issue: ValidationIssue) -> dict:
    """
    Repairs an INVALID_VALUE issue on an enum-constrained field.
    """
    if issue.defect_type != DefectType.INVALID_VALUE:
        raise ValueError(
            f"repair_invalid_enum_value only handles INVALID_VALUE, got {issue.defect_type}"
        )

    normalized = normalize_location(issue.location)
    enum_values = _lookup_enum_values(normalized)

    if enum_values is None:
        raise ValueError(
            f"'{issue.location}' (normalized: '{normalized}') has no known enum "
            f"constraint -- this action only handles recognized enum fields "
            f"(distribution, procurement_scheme.type, inventory.type, "
            f"edges.material_type). A different INVALID_VALUE (e.g. a general "
            f"type mismatch) needs a different repair action."
        )

    steps = _parse_location_steps(issue.location)
    parent, last_key = _navigate_to_parent(config, steps)
    current_value = parent.get(last_key) if isinstance(parent, dict) else None

    if isinstance(current_value, str) and current_value.endswith("s"):
        singular_candidate = current_value[:-1]
        if singular_candidate in enum_values:
            parent[last_key] = singular_candidate
            print(f"  Auto-corrected '{current_value}' -> '{singular_candidate}' "
                  f"(plural/singular mismatch, unambiguous).")
            return config

    value = _prompt_for_value(issue.location, "str", enum_values=enum_values, config=config)
    parent[last_key] = value

    return config


# ----------------------------------------------------------------------
# nodes[0] repairs
# ----------------------------------------------------------------------

NODES_ENTITY_SECTION = {"supplier": "supplier", "facility": "facility", "customer": "customer"}


def repair_missing_nodes_list(config: dict, issue: ValidationIssue) -> dict:
    """
    Repairs a MISSING_REQUIRED_VALUE issue at "nodes[0].supplier" /
    "nodes[0].facility" / "nodes[0].customer" -- the KEY itself is absent.
    """
    m = re.match(r"^nodes\[0\]\.(supplier|facility|customer)$", issue.location)
    if not m:
        raise ValueError(f"'{issue.location}' is not a recognized nodes[0] list field.")
    key = m.group(1)

    nodes = config.get("nodes")
    if not isinstance(nodes, list) or len(nodes) == 0 or not isinstance(nodes[0], dict):
        raise ValueError("config['nodes'][0] does not exist -- cannot repair a child key of it.")

    section = NODES_ENTITY_SECTION[key]
    candidates = sorted(_collect_names(config, section))

    fallback_text = f"Field '{issue.location}' is missing -- select its entries below."
    print(_ask(config, issue.location, fallback_text))
    _fill_list_select(nodes[0], key, candidates, issue.location, config=config)

    return config


def repair_nodes_registration_gap(config: dict, issue: ValidationIssue) -> dict:
    """
    Repairs an INCONSISTENT_CROSS_FIELD issue at "nodes[0].supplier" /
    "nodes[0].facility" / "nodes[0].customer" -- a real entity exists but
    isn't registered in nodes[0]. Fully deterministic, no prompt.
    """
    m = re.match(r"^nodes\[0\]\.(supplier|facility|customer)$", issue.location)
    if not m:
        raise ValueError(f"'{issue.location}' is not a recognized nodes[0] registration gap.")
    key = m.group(1)

    name = issue.context.get("referenced_name") if issue.context else None
    if not name:
        raise ValueError(f"Issue at '{issue.location}' has no referenced_name in its context.")

    nodes = config.get("nodes")
    if not isinstance(nodes, list) or len(nodes) == 0 or not isinstance(nodes[0], dict):
        raise ValueError("config['nodes'][0] does not exist -- cannot repair a child key of it.")

    existing = nodes[0].setdefault(key, [])
    if name not in existing:
        existing.append(name)
        print(f"  Added '{name}' to nodes[0].{key}.")

    return config


def repair_nodes_phantom_entry(config: dict, issue: ValidationIssue) -> dict:
    """
    Repairs a DANGLING_REFERENCE issue at "nodes[0].supplier[i]" /
    "nodes[0].facility[i]" / "nodes[0].customer[i]" -- a name IS present
    at that index but doesn't correspond to any real entity.
    """
    m = re.match(r"^nodes\[0\]\.(supplier|facility|customer)\[(\d+)\]$", issue.location)
    if not m:
        raise ValueError(f"'{issue.location}' is not a recognized nodes[0] phantom-entry issue.")
    key, idx = m.group(1), int(m.group(2))

    nodes = config.get("nodes")
    if not isinstance(nodes, list) or len(nodes) == 0 or not isinstance(nodes[0], dict):
        raise ValueError("config['nodes'][0] does not exist -- cannot repair a child entry of it.")

    entry_list = nodes[0].get(key)
    if not isinstance(entry_list, list) or idx >= len(entry_list):
        raise ValueError(f"'{issue.location}' does not resolve to a real list entry.")

    phantom_name = entry_list[idx]
    section = NODES_ENTITY_SECTION[key]
    real_candidates = sorted(_collect_names(config, section) - set(entry_list))

    fallback_text = f"'{issue.location}' = '{phantom_name}' does not correspond to any real {key}. Choose how to resolve it:"
    print(f"  {_ask(config, issue.location, fallback_text)}")
    print("    0) Remove this entry")
    for i, name in enumerate(real_candidates, start=1):
        print(f"    {i}) Replace with: {name}")

    while True:
        choice = input("  Enter number: ").strip()
        if choice == "0":
            del entry_list[idx]
            print(f"    Removed '{phantom_name}'.")
            return config
        if not choice.isdigit() or not (1 <= int(choice) <= len(real_candidates)):
            print(f"    '{choice}' is not a valid selection. Try again.")
            continue
        entry_list[idx] = real_candidates[int(choice) - 1]
        print(f"    Replaced with '{entry_list[idx]}'.")
        return config


# ----------------------------------------------------------------------
# Dangling reference repairs (Layer2 issues on scalar reference fields)
# ----------------------------------------------------------------------

def repair_dangling_reference(config: dict, issue: ValidationIssue) -> dict:
    """
    Repairs a DANGLING_REFERENCE issue on a scalar field that already has
    a value -- just not a valid one.
    """
    if issue.defect_type != DefectType.DANGLING_REFERENCE:
        raise ValueError(
            f"repair_dangling_reference only handles DANGLING_REFERENCE, got {issue.defect_type}"
        )

    normalized = normalize_location(issue.location)
    select_fn = SCALAR_SELECT_CANDIDATE_FNS.get(normalized)
    if select_fn is None:
        raise ValueError(
            f"'{issue.location}' (normalized: '{normalized}') has no registered candidate "
            f"function -- this action only handles known reference fields."
        )

    steps = _parse_location_steps(issue.location)
    parent, last_key = _navigate_to_parent(config, steps)
    current_value = parent.get(last_key) if isinstance(parent, dict) else None

    candidates = select_fn(config, parent)
    allow_none = normalized in SCALAR_SELECT_ALLOW_NONE

    if not candidates and not allow_none:
        raise ValueError(f"No valid candidates available to repair '{issue.location}'.")

    selected = _prompt_select_single(issue.location, candidates, allow_none=allow_none, config=config)
    if selected is None:
        del parent[last_key]
        print(f"    Cleared '{issue.location}' (left unset).")
    else:
        parent[last_key] = selected

    return config


# ----------------------------------------------------------------------
# Dispatch
# ----------------------------------------------------------------------

def dispatch_repair(config: dict, issue: ValidationIssue) -> bool:
    """
    Attempts to repair a single issue by routing it to whichever action
    actually handles its shape. Returns True if repaired, False if no
    current action can handle it.
    """
    if issue.defect_type == DefectType.MISSING_REQUIRED_VALUE:
        if issue.location == "config_info[0].version":
            repair_config_info_version(config, issue)
            return True

        if re.match(r"^nodes\[0\]\.(supplier|facility|customer)$", issue.location):
            repair_missing_nodes_list(config, issue)
            return True

        if issue.location == "raw_materials":
            repair_at_least_one_raw_material(config, issue)
            return True

        if re.match(r"^(intermediate_materials|products)\[\d+\]\.bom\.[^.]+$", issue.location):
            repair_bom_value(config, issue)
            return True

        if issue.location == "products":
            repair_at_least_one_product(config, issue)
            return True

        if "procurement_scheme" in issue.location:
            repair_procurement_scheme_field(config, issue)
            return True

        if re.match(r"^edges\[\d+\]\.transfer_time", issue.location):
            repair_transfer_time_default_instant(config, issue)
            return True

        normalized = normalize_location(issue.location)

        if _lookup_field_type(normalized) is not None:
            repair_scalar_missing_field(config, issue)
            return True

        try:
            find_spec_node(normalized)
        except ValueError:
            return False
        repair_missing_node(config, issue)
        return True

    elif issue.defect_type == DefectType.INVALID_VALUE:
        if issue.location == "simulation.horizon":
            repair_horizon_sanity(config, issue)
            return True

        if "procurement_scheme" in issue.location:
            repair_procurement_scheme_field(config, issue)
            return True

        if re.match(r"^edges\[\d+\]\.transfer_time", issue.location):
            repair_transfer_time_default_instant(config, issue)
            return True

        normalized = normalize_location(issue.location)
        if _lookup_enum_values(normalized) is not None:
            repair_invalid_enum_value(config, issue)
            return True
        return False

    elif issue.defect_type == DefectType.DANGLING_REFERENCE:
        if re.match(r"^nodes\[0\]\.(supplier|facility|customer)\[\d+\]$", issue.location):
            repair_nodes_phantom_entry(config, issue)
            return True

        normalized = normalize_location(issue.location)

        if normalized == "edges.material_name":
            steps = _parse_location_steps(issue.location)
            edge_entry, _ = _navigate_to_parent(config, steps)
            material_type = edge_entry.get("material_type")
            section = INVENTORY_TYPE_TO_SECTION.get(material_type)
            category_valid = section is not None and edge_entry.get("material_name") in _collect_names(config, section)

            if not category_valid:
                repair_dangling_reference(config, issue)
            else:
                repair_edge_phantom_delivery(config, issue)
            return True

        if normalized in SCALAR_SELECT_CANDIDATE_FNS:
            repair_dangling_reference(config, issue)
            return True
        return False

    elif issue.defect_type == DefectType.INCONSISTENT_CROSS_FIELD:
        if re.match(r"^nodes\[0\]\.(supplier|facility|customer)$", issue.location):
            repair_nodes_registration_gap(config, issue)
            return True

        if re.match(r"^raw_materials\[\d+\]$", issue.location):
            detail = issue.detail or ""
            if "has no supplier" in detail:
                ordered = [repair_raw_material_missing_supplier]
            elif "is never consumed" in detail:
                ordered = [repair_raw_material_not_consumed]
            elif "has no corresponding inventory" in detail:
                ordered = [repair_material_missing_inventory_entry]
            else:
                ordered = []
            ordered += [a for a in (
                repair_raw_material_missing_supplier, repair_raw_material_not_consumed,
                repair_material_missing_inventory_entry,
            ) if a not in ordered]
            for action in ordered:
                try:
                    action(config, issue)
                    return True
                except ValueError:
                    continue
            return False

        if re.match(r"^supplier\[\d+\]$", issue.location):
            repair_supplier_missing_outbound_edge(config, issue)
            return True

        if re.match(r"^products\[\d+\]$", issue.location):
            detail = issue.detail or ""
            if "has no customer ordering it" in detail:
                ordered = [repair_product_missing_customer]
            elif "is not produced by any facility operation" in detail:
                ordered = [repair_product_not_producible]
            elif "has a producing facility, but no" in detail:
                ordered = [repair_product_end_to_end_path]
            elif "has no corresponding inventory" in detail:
                ordered = [repair_material_missing_inventory_entry]
            else:
                ordered = []
            ordered += [a for a in (
                repair_product_missing_customer, repair_product_not_producible,
                repair_product_end_to_end_path, repair_material_missing_inventory_entry,
            ) if a not in ordered]
            for action in ordered:
                try:
                    action(config, issue)
                    return True
                except ValueError:
                    continue
            return False

        if re.match(r"^intermediate_materials\[\d+\]$", issue.location):
            detail = issue.detail or ""
            if "is not produced by any facility" in detail:
                ordered = [repair_intermediate_material_not_producible]
            elif "has no corresponding inventory" in detail:
                ordered = [repair_material_missing_inventory_entry]
            else:
                ordered = []
            ordered += [a for a in (
                repair_intermediate_material_not_producible, repair_material_missing_inventory_entry,
            ) if a not in ordered]
            for action in ordered:
                try:
                    action(config, issue)
                    return True
                except ValueError:
                    continue
            return False

        if re.match(r"^facility\[\d+\]$", issue.location):
            repair_facility_missing_edge(config, issue)
            return True

        if re.match(r"^customer\[\d+\]$", issue.location):
            repair_customer_missing_inbound_edge(config, issue)
            return True

        if re.match(r"^facility\[\d+\]\.inventory_managed$", issue.location):
            repair_facility_material_stage_span(config, issue)
            return True

        return False

    elif issue.defect_type == DefectType.DUPLICATE_ENTITY:
        if issue.location == "supplier/facility":
            repair_supplier_facility_name_collision(config, issue)
            return True

        if re.match(r"^(raw_materials|intermediate_materials|products)/(raw_materials|intermediate_materials|products)$", issue.location):
            repair_material_category_collision(config, issue)
            return True

        if re.match(r"^edges\[[\d,]+\]$", issue.location):
            repair_duplicate_edges(config, issue)
            return True

        if re.match(r"^\w+\[[\d,]+\]$", issue.location):
            repair_duplicate_name_within_section(config, issue)
            return True

    elif issue.defect_type == DefectType.MALFORMED_ENTRY:
        if re.match(r"^edges\[\d+\]$", issue.location):
            repair_self_loop_edge(config, issue)
            return True

        return False

    return False


# ----------------------------------------------------------------------
# CLI entry point for quick manual testing
# ----------------------------------------------------------------------

if __name__ == "__main__":
    import json
    import sys

    from verification_layer1 import check_field_requirements
    from verification_layer2 import (
        check_intermediate_bom_references, check_product_bom_references,
        check_inventory_type_consistency, check_supplier_material_references,
        check_facility_inventory_managed_references, check_customer_product_references,
        check_nodes_registration_completeness, check_edge_material_references,
        check_operation_resource_references, check_edge_node_references,
        check_nodes_names_exist, check_material_name_uniqueness,
        check_operation_io_material_references, check_duplicate_names_within_sections,
        check_duplicate_edges, check_supplier_facility_name_collision,
        check_nodes_customer_registration,
    )
    from issue_types import Severity

    path = sys.argv[1] if len(sys.argv) > 1 else "test_config.json"
    with open(path) as f:
        cfg = json.load(f)

    issues = check_field_requirements(cfg) + (
        check_intermediate_bom_references(cfg)
        + check_product_bom_references(cfg)
        + check_inventory_type_consistency(cfg)
        + check_supplier_material_references(cfg)
        + check_facility_inventory_managed_references(cfg)
        + check_customer_product_references(cfg)
        + check_nodes_registration_completeness(cfg)
        + check_edge_material_references(cfg)
        + check_operation_resource_references(cfg)
        + check_edge_node_references(cfg)
        + check_nodes_names_exist(cfg)
        + check_material_name_uniqueness(cfg)
        + check_operation_io_material_references(cfg)
        + check_duplicate_names_within_sections(cfg)
        + check_duplicate_edges(cfg)
        + check_supplier_facility_name_collision(cfg)
        + check_nodes_customer_registration(cfg)
    )
    blocking = [i for i in issues if i.severity == Severity.BLOCKING]

    handled = []
    unhandled = []
    for issue in blocking:
        if dispatch_repair(cfg, issue):
            handled.append(issue)
        else:
            unhandled.append(issue)

    print(f"Repaired {len(handled)} issue(s).")
    print(f"({len(unhandled)} issue(s) need a different/future repair action.)\n")

    if unhandled:
        print("Unhandled issues:")
        for issue in unhandled:
            print("   ", issue)
        print()

    if not handled:
        print("Nothing was repaired -- config left unchanged.")
    else:
        with open(path, "w") as f:
            json.dump(cfg, f, indent=2)
        print(f"Saved repaired config to {path}")