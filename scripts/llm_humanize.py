"""
scripts/llm_humanize.py
------------------------
Generates natural-language repair-prompt questions via an LLM, replacing
repair.py's old template-based describe_location()/humanize_question()
system. Given a missing/invalid field's location, the current config,
and whatever original context exists (the user's NL description, or a
note that a hand-drawn sketch was the source instead), asks the LLM to
phrase ONE clear, context-grounded question -- not generic boilerplate
like "please provide a value for X".

Cached by (normalized_location, entity_name) so the same field shape
recurring across many entities in one run (e.g. every supplier's
supplier_lead_time) doesn't re-hit the API each time -- the ideal
phrasing for a given field shape + entity name pair rarely needs to
change once generated.

If the API call fails for any reason (network, parsing, missing key),
falls back to the OLD deterministic template text from repair.py's
describe_location(), passed in by the caller -- this module never being
the reason a repair prompt fails to display something usable.

Requires:
    pip install openai --break-system-packages
    export OPENAI_API_KEY=sk-...
"""

import json
import re

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI

DEFAULT_MODEL = "gpt-5.6-terra"
NO_TEMPERATURE_ZERO_MODELS = {"gpt-5.6-terra"}

# Consolidated field-meaning writeup, reused (condensed) from
# graph_only_to_json.py's extraction prompt -- the same schema
# understanding, given here so the LLM can phrase an INFORMED question
# rather than just restating a field path in English.
FIELD_MEANINGS = """
- config_info: metadata about this config (name, schema version).
- raw_materials / intermediate_materials / products: the three material tiers.
  raw_materials come from suppliers; intermediate_materials are produced and
  consumed internally (bom = bill of materials -- quantities of other
  materials needed to make one unit); products are what customers order
  (also have a bom).
- inventory: a stocking point for a material/product.
  - procurement_scheme: how a raw material gets restocked -- periodic_supply
    (delivered on a schedule, needs a distribution describing order-quantity
    variability), demand_driven (ordered in direct response to a shortage,
    needs nothing else), or inventory_threshold (reorder point 's' and
    order-up-to level 'S', not a distribution).
  - procurement_arrival: how often periodic_supply orders arrive.
  - initial_inventory: starting stock level.
  - inventory_costs: holding_cost (cost per unit held per period),
    shortage_cost (cost per unit short), review_time (how often reviewed).
- supplier_capacity/cost: how much a supplier can provide, and its unit cost.
- supplier_lead_time / customer_lead_time: time between order and fulfillment.
- supplier_payment_lead_time / customer_payment_lead_time: time between
  fulfillment and payment.
- supply_material_name: the raw material a supplier provides.
- resource: equipment/labor capacity an operation draws on (capacity,
  service_time per operation, batching, failure/uptime-downtime,
  operating_cost_per_time).
- facility.type: "manufacturing" transforms input material(s) into a
  DIFFERENT output material; "warehouse" stores/passes material through
  unchanged.
- inventory_managed: materials a facility stores.
- operation.input / operation.output: materials consumed / produced by a
  manufacturing facility's operation.
- operation.operation_cycle: time to complete one production cycle.
- resource_required: which resource this operation draws on.
- customer.product: the finished product a customer orders.
- demand: the customer's order quantity distribution.
- unit_selling_price: price per unit sold to a customer.
- arrival_time: distribution of when customer orders arrive.
- shortage_policy: how unmet demand is handled (lost sale, backorder, or
  partial versions of either).
- edges.material_name / material_type: what flows across an edge, and its
  category (raw_material / intermediate_material / product).
- transfer_time: time for material to move across an edge.
- simulation: run parameters (time_unit, horizon, warm_up period, number of
  replications, random_seed).
- Every "distribution" object is {"distribution": <name>, "parameters": {a..e}}.
"""

SYSTEM_PROMPT = f"""You are helping a non-technical user fill in gaps in a supply-chain \
simulation config (the NL2Sim schema). For each field the system found missing or invalid, \
phrase exactly ONE clear, natural, conversational question asking the user to provide it.

Use whatever original context is given -- the user's own supply-chain description, or a note \
that this config came from a hand-drawn sketch with no description -- to make the question feel \
grounded in THEIR specific scenario, not generic boilerplate. Reference entity names, materials, \
or anything already known nearby in the config when it helps the question make sense.

FIELD MEANINGS (for your own understanding -- do not dump this back at the user):
{FIELD_MEANINGS}

Rules:
- Ask ONE question only, in plain English, no jargon, no schema/field-path names (never say
  things like "supplier_lead_time" or "edges[0].material_name" -- describe it in words).
- If the field will be followed by a menu of options (an enum), don't list the options yourself --
  just ask what applies, in a way that makes sense given a menu is about to appear.
- If a "situation" is given in the input, it OVERRIDES field_location entirely for the purpose of
  framing the question -- field_location is only there to help you look up entity_context, never to
  shape the wording. field_location will often look like a simple data-entry request (e.g.
  "facility[0].inventory_managed" reads like "how many materials go here?", "edges[1].material_name"
  reads like "what material is this?") even when situation describes something completely
  different -- a multi-option repair decision (delete vs. retarget vs. sync, change type vs. add
  material vs. delete facility, etc). When situation describes a decision with named options or
  actions (look for phrasing like "choose a fix", multiple named alternatives, or "Choose a fix:"),
  your question MUST be a decision question ("what would you like to do about...", "how should ...
  be resolved") -- it must NEVER be a counting, listing, or simple-lookup question, even if
  field_location or entity_context would suggest that framing on their own. Concretely: given
  situation "Manufacturing facility 'Wafer Fab' only manages material at one stage. Choose a fix:",
  a CORRECT question is "Wafer Fab currently only handles one stage of production -- would you like
  to make it a plain storage facility, add material from a different stage, or remove it entirely?"
  An INCORRECT question is "How many different materials should Wafer Fab keep in inventory?" --
  that answers field_location literally and ignores situation, which is exactly what NOT to do.
- If "expected_answer_type" is given, phrase the question so the expected answer shape is obvious
  from the wording alone: for "num", ask for a quantity or duration ("how many...", "how long...")
  -- NEVER phrase it as a yes/no question (e.g. don't ask "should X happen at day 0?" when the
  real answer needed is a number of days like 0, 5, 10). For "name", ask for an identifier/label.
  For "bool", a yes/no phrasing IS appropriate. For "str" with no further constraint, phrase it as
  an open request for that value. Note: when situation describes a menu of options that will be
  numbered 0/1/2/3 for the person to pick from, expected_answer_type will be "num" for that reason
  alone (picking a menu number) -- this does NOT mean the question itself should ask for a
  quantity; it should still be the decision question situation describes, phrased naturally (the
  numbered menu appears separately, right after your question, so you never need to enumerate the
  options yourself).
- If the description mentions something relevant to this exact field, work it in naturally
  (e.g. "You mentioned the plant runs continuously -- about how long does it take to produce one
  chair?"). If nothing relevant is mentioned, ask plainly without inventing a false connection.
- Keep it to one or two sentences.
- Return ONLY the question text -- no quotes, no preamble, no field-path echoing.
"""

_CACHE: dict = {}


def _normalize_location(location: str) -> str:
    """Same stripping rule as repair.py's normalize_location -- duplicated
    here (rather than imported) to avoid a circular import, since
    repair.py imports THIS module."""
    return re.sub(r"\[\d+\]", "", location)


def _parse_steps(location: str):
    steps = []
    for part in location.split("."):
        m = re.match(r"^([a-zA-Z_][a-zA-Z0-9_]*)(\[(\d+)\])?$", part)
        if not m:
            return []
        key, _, index = m.groups()
        steps.append(key)
        if index is not None:
            steps.append(int(index))
    return steps


def _local_entity_context(config: dict, location: str) -> dict:
    """
    Walks just far enough into the config to find the owning entity's
    own dict (e.g. the specific supplier[2] entry), for use as context --
    NOT a full describe_entity() reimplementation, just enough sibling
    data (name, a few immediate fields) for the LLM to ground the
    question. Returns {} on any failure -- this is best-effort context,
    never required for the call to proceed.
    """
    try:
        steps = _parse_steps(location)
        if len(steps) < 2 or not isinstance(steps[1], int):
            return {}
        section, idx = steps[0], steps[1]
        entries = config.get(section, []) or []
        if idx >= len(entries) or not isinstance(entries[idx], dict):
            return {}
        entry = entries[idx]
        # Shallow copy of scalar/simple fields only -- avoid dumping huge
        # nested structures (e.g. a whole operation block) into the prompt.
        return {
            k: v for k, v in entry.items()
            if isinstance(v, (str, int, float, bool)) or v is None
        }
    except Exception:
        return {}


def _llm_call(user_payload: dict, model: str) -> str:
    from llm_client import chat, resolve_provider

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(user_payload, indent=2)},
    ]
    provider = resolve_provider()
    # `model` names an OpenAI model; other providers use their configured one.
    if provider != "openai":
        model = None
    temperature = None if model in NO_TEMPERATURE_ZERO_MODELS else 0
    return chat(
        messages, provider=provider, model=model, temperature=temperature
    ).strip('"')


def generate_question(
    config: dict,
    location: str,
    description: str = "",
    fallback_text: str = None,
    answer_type: str = None,
    model: str = DEFAULT_MODEL,
) -> str:
    """
    Main entry point. Returns a natural-language question for the given
    field location, using the LLM as the primary path. Falls back to
    fallback_text (typically the caller's old describe_location()-based
    string) if the API call fails for any reason.

    description: the user's original NL description, or "" if this
    config came from a graph-only run (in which case a note is passed
    to the LLM instead, so it knows not to expect NL-derived context).

    answer_type: what kind of answer the following input prompt actually
    expects -- "num", "name", "str", or "bool". Without this, the LLM has
    no signal about the expected response shape and can phrase a numeric
    field as a yes/no question (e.g. "should the simulation begin
    collecting results immediately at day 0?" for a field that expects
    a number of days, not yes/no) -- the person answers "Yes", the
    numeric parser rejects it, and they have to retry. Passing this lets
    the LLM phrase the question so the expected answer shape is obvious.
    """
    normalized = _normalize_location(location)
    entity_ctx = _local_entity_context(config, location)
    entity_name = entity_ctx.get("name")
    # situation (fallback_text) genuinely changes what question gets
    # asked -- the SAME field location can be reached for entirely
    # different reasons (e.g. "facility.inventory_managed" is asked once
    # while first BUILDING a facility's entry, "select what this manages",
    # and later reached again by the stage-span repair action, "this
    # facility only manages one stage, choose a fix" -- two unrelated
    # questions sharing a field shape). Without folding situation into
    # the key, the second call would silently reuse the FIRST question's
    # cached text, producing a mismatched, confusing prompt. A short
    # hash keeps the key compact regardless of how long fallback_text is.
    situation_key = hash(fallback_text) if fallback_text else None
    cache_key = (normalized, entity_name, situation_key)

    if cache_key in _CACHE:
        return _CACHE[cache_key]

    context_note = description.strip() if description and description.strip() else (
        "(No natural-language description was provided -- this config was built "
        "from a hand-drawn sketch/diagram instead.)"
    )

    payload = {
        "field_location": location,
        "entity_context": entity_ctx,
        "original_description": context_note,
    }
    if fallback_text:
        payload["situation"] = fallback_text
    if answer_type:
        payload["expected_answer_type"] = answer_type

    try:
        question = _llm_call(payload, model)
        if not question:
            raise ValueError("empty response")
    except Exception:
        question = fallback_text or f"Please provide a value for '{location}'."

    _CACHE[cache_key] = question
    return question