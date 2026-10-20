"""
scripts/verification_layer1.py
---------------------------------
STEP 1 OF 3 (this file currently implements Step 1 only):
    1. Field requirement checking (presence)      <- THIS FILE, current scope
    2. Data type validation                        <- not yet implemented
    3. Placeholder ("missing") detection            <- implemented

Scope of this step:
    - Does each field that SHOULD be present actually exist in the entry?
    - "Required" can be:
        (a) a plain bool -- always required / never required, or
        (b) a callable(parent_entry) -> bool -- required only when a
            sibling field within the SAME entry has a certain value.
    - Conditions are LOCAL ONLY: a condition may inspect sibling fields
      within the same dict, never fields in other sections or other
      entries. Cross-entity conditions belong to verification_layer2.py.

Explicitly NOT checked yet (deferred to Step 2):
    - General type correctness (e.g. a string where a number was expected)
"""

from issue_types import ValidationIssue, DefectType, Severity

LAYER = "Layer1"
MISSING_PLACEHOLDER = "missing"


# ----------------------------------------------------------------------
# Field spec primitives
# ----------------------------------------------------------------------

def FIELD(required=True, fields=None, silent=False, enum_values=None, is_name=False, always_ask=False):
    """
    silent=True means this field never generates a ValidationIssue, whether
    it's absent or holds the "missing" placeholder. Use this for fields the
    simulation engine defaults on its own (e.g. random_seed, resource_required
    when unused, supplier_capacity, inventory_costs) -- their absence isn't
    a data-quality problem, it's a legitimate "let the engine decide."

    enum_values: if given, a present (non-placeholder) value must be one
    of this list -- otherwise it's an INVALID_VALUE issue. This catches
    bad categorical values (e.g. a typo'd material type) in the ORIGINAL
    config, not just values a person enters during repair.

    is_name=True: the value must be a genuine string AND not purely
    numeric -- catches both a raw JSON number (e.g. "name": 12345, no
    quotes) and a quoted numeric string (e.g. "name": "12345") in the
    ORIGINAL config. An identifier field (supplier/material/facility
    name, etc.) should never be just a number.

    always_ask=True: this field is itself optional (required=False), but
    a SIBLING field's requirement depends on its value (e.g.
    batch_size/max_wait_time are only required if enabled=True). Doesn't
    affect verification -- Layer1 never flags an always_ask field as
    missing on its own. It's a signal to repair.py: when reconstructing
    a freshly-missing container from scratch, ask this "gating" field
    even though it's technically optional, so its conditional siblings
    can be evaluated meaningfully instead of silently defaulting to
    "not asked."
    """
    spec = {"required": required, "silent": silent}
    if fields is not None:
        spec["fields"] = fields
    if enum_values is not None:
        spec["enum_values"] = enum_values
    if is_name:
        spec["is_name"] = True
    if always_ask:
        spec["always_ask"] = True
    return spec


def FIELD_DISTRIBUTION(required=True, extra_fields=None):
    """
    Marks a field as a distribution object ({"distribution": ..., "parameters": {...}})
    whose parameter requirement depends on the distribution TYPE, not a fixed
    set of nested field specs. Handled by _check_distribution_field, not the
    generic recursive walker.

    extra_fields: for objects that are "a distribution plus something else"
    (e.g. inventory.procurement_scheme, which has "distribution"/"parameters"
    AND its own "type" field describing the procurement policy). These extra
    fields are checked normally (presence + placeholder), alongside the
    distribution-aware check, against the same parent dict.
    """
    spec = {"required": required, "is_distribution": True}
    if extra_fields is not None:
        spec["extra_fields"] = extra_fields
    return spec


def FIELD_DICT_VALUES(required=True, value_kind="num", min_items=0):
    """
    Marks a field as a dict with DYNAMIC keys (unlike "fields", which
    describes a FIXED set of named children) -- every value inside it
    must satisfy value_kind ("num" currently supported). Use this for
    fields like bom, where the keys are material names (not known ahead
    of time) but every value must be a real quantity, not the "missing"
    placeholder or a non-numeric type.

    min_items: if > 0, the dict must have at least this many entries --
    an empty {} (present, correctly typed, but with nothing in it) is
    otherwise invisible to validation, since there's nothing to iterate
    and flag. bom, for example, must have at least one ingredient.
    """
    return {"required": required, "is_dict_values": True, "value_kind": value_kind, "min_items": min_items}


def is_required(required_spec, parent_entry: dict) -> bool:
    """Resolve a required spec (bool or callable) against the parent entry."""
    if callable(required_spec):
        return bool(required_spec(parent_entry))
    return bool(required_spec)
    """Resolve a required spec (bool or callable) against the parent entry."""
    if callable(required_spec):
        return bool(required_spec(parent_entry))
    return bool(required_spec)


# ----------------------------------------------------------------------
# Distribution parameter-count table
# ----------------------------------------------------------------------
# How many of parameters.a/b/c/d/e are actually required, keyed by the
# distribution's "distribution" value. Parameters beyond this count are
# never required (they're simply unused by that distribution type).
#
#   constant     -> 1  (value)
#   exponential  -> 1  (rate / mean)
#   poisson      -> 1  (rate / lambda)
#   uniform      -> 2  (min, max)
#   normal       -> 2  (mean, std dev)
#   weibull      -> 2  (shape, scale)
#   gamma        -> 2  (shape, scale)
#   erlang       -> 2  (shape/k, scale)
#   lognormal    -> 2  (mu, sigma)
#   triangular   -> 3  (min, mode, max)
#
# If the distribution value is missing, unrecognized, or not yet known,
# we fall back to requiring only 'a' -- every distribution needs at least
# one parameter, and we can't know how many more without knowing the type.

DISTRIBUTION_PARAM_COUNTS = {
    "constant": 1,
    "exponential": 1,
    "poisson": 1,
    "uniform": 2,
    "normal": 2,
    "weibull": 2,
    "gamma": 2,
    "erlang": 2,
    "lognormal": 2,
    "triangular": 3,
}

PARAM_KEYS = ["a", "b", "c", "d", "e"]

# Canonical enum value sets -- single source of truth. repair.py imports
# these directly rather than maintaining its own duplicate lists, so the
# two never drift out of sync.
# Plural, matching the section names these refer to -- and matching what the
# training data and simulate.py already use.
MATERIAL_TYPES = ["raw_materials", "intermediate_materials", "products"]
PROCUREMENT_SCHEME_TYPES = ["periodic_supply", "demand_driven", "inventory_threshold"]
FACILITY_TYPES = ["manufacturing", "warehouse"]
SHORTAGE_POLICIES = ["salelost", "backorder", "salelost_partial", "backorder_partial"]
# Both forms: the training data emits the singular ("day"), the schema example
# the plural ("days"). simulate.py never reads this field, so neither is wrong.
TIME_UNITS = ["second", "minute", "hour", "day", "week", "month",
              "seconds", "minutes", "hours", "days", "weeks", "months"]


# ----------------------------------------------------------------------
# Section specs, derived from scripts/schema.py
# ----------------------------------------------------------------------

SECTION_SPECS = {
    "config_info": {
        "container": "list",
        "required_section": True,
        "fields": {
            "name": FIELD(required=True, is_name=True),
            "version": FIELD(required=True),
        },
    },
    "raw_materials": {
        "container": "list",
        "required_section": True,
        "fields": {
            "name": FIELD(required=True, is_name=True),
        },
    },
    "intermediate_materials": {
        "container": "list",
        "required_section": False,
        "fields": {
            "name": FIELD(required=True, is_name=True),
            "bom": FIELD_DICT_VALUES(required=True, value_kind="num", min_items=1),
        },
    },
    "products": {
        "container": "list",
        "required_section": True,
        "fields": {
            "name": FIELD(required=True, is_name=True),
            "bom": FIELD_DICT_VALUES(required=True, value_kind="num", min_items=1),
        },
    },
    "inventory": {
        "container": "list",
        "required_section": True,
        "fields": {
            "name": FIELD(required=True, is_name=True),
            "type": FIELD(required=True, enum_values=MATERIAL_TYPES),
            # Conditionally required: only raw materials are externally
            # procured. Intermediate materials and products are produced
            # internally via facility operations, so they don't need a
            # procurement scheme.
            #
            # procurement_scheme has THREE genuinely different shapes
            # depending on its own "type": periodic_supply needs a real
            # statistical distribution (order-quantity variability);
            # demand_driven needs nothing else at all (procurement
            # triggers directly off a shortage event); inventory_threshold
            # needs exactly two fixed VALUES (s = reorder point, S =
            # order-up-to level) -- not a distribution at all. This
            # doesn't fit the single-shape FIELD_DISTRIBUTION pattern, so
            # it's checked/filled by dedicated logic
            # (is_procurement_scheme) instead.
            "procurement_scheme": {
                "required": lambda parent: parent.get("type") == "raw_materials",
                "is_procurement_scheme": True,
            },
            "procurement_arrival": FIELD_DISTRIBUTION(
                required=lambda parent: (parent.get("procurement_scheme") or {}).get("type") == "periodic_supply",
            ),
            "initial_inventory": FIELD(required=True),
            "inventory_costs": FIELD(required=False, always_ask=True, fields={
                "holding_cost": FIELD(required=False, silent=True, always_ask=True),
                "shortage_cost": FIELD(required=False, silent=True, always_ask=True),
                "review_time": FIELD(required=False, silent=True, always_ask=True),
            }),
        },
    },
    "supplier": {
        "container": "list",
        "required_section": True,
        "fields": {
            "name": FIELD(required=True, is_name=True),
            "supply_material_name": FIELD(required=True, is_name=True),
            "supplier_lead_time": FIELD_DISTRIBUTION(required=True),
            "supplier_capacity": FIELD(required=False, silent=True),
            "supplier_cost": FIELD(required=True),
            "supplier_payment_lead_time": FIELD_DISTRIBUTION(required=True),
        },
    },
    "resource": {
        "container": "list",
        "required_section": True,
        "fields": {
            "name": FIELD(required=True, is_name=True),
            "capacity": FIELD(required=True),
            "service_time": FIELD_DISTRIBUTION(required=True),
            "batching": FIELD(required=False, fields={
                "enabled": FIELD(required=False, always_ask=True),
                "batch_size": FIELD(required=lambda parent: parent.get("enabled") is True),
                "max_wait_time": FIELD(required=lambda parent: parent.get("enabled") is True),
            }),
            "failure": FIELD(required=False, fields={
                "enabled": FIELD(required=False, always_ask=True),
                "uptime": FIELD_DISTRIBUTION(required=lambda parent: parent.get("enabled") is True),
                "downtime": FIELD_DISTRIBUTION(required=lambda parent: parent.get("enabled") is True),
            }),
            "operating_cost_per_time": FIELD(required=False),
        },
    },
    "facility": {
        "container": "list",
        "required_section": True,
        "fields": {
            "name": FIELD(required=True, is_name=True),
            "type": FIELD(required=True, enum_values=FACILITY_TYPES),
            "inventory_managed": FIELD(required=True),
            "operation": FIELD(
                required=lambda parent: parent.get("type") == "manufacturing",
                fields={
                    "name": FIELD(required=True, is_name=True),
                    "input": FIELD(required=True),
                    "output": FIELD(required=True),
                    "resource_required": FIELD(required=False, silent=True, is_name=True),
                    "operation_cycle": FIELD_DISTRIBUTION(required=True),
                },
            ),
        },
    },
    "customer": {
        "container": "list",
        "required_section": True,
        "fields": {
            "name": FIELD(required=True, is_name=True),
            "product": FIELD(required=True, is_name=True),
            "arrival_time": FIELD_DISTRIBUTION(required=True),
            "demand": FIELD_DISTRIBUTION(required=True),
            "customer_lead_time": FIELD_DISTRIBUTION(required=True),
            "shortage_policy": FIELD(required=True, enum_values=SHORTAGE_POLICIES),
            "unit_selling_price": FIELD(required=True),
            "customer_payment_lead_time": FIELD_DISTRIBUTION(required=True),
        },
    },
    "edges": {
        "container": "list",
        "required_section": True,
        "fields": {
            "source": FIELD(required=True, is_name=True),
            "destination": FIELD(required=True, is_name=True),
            "material_type": FIELD(required=True, enum_values=MATERIAL_TYPES),
            "material_name": FIELD(required=True, is_name=True),
            "transfer_time": FIELD_DISTRIBUTION(required=True),
        },
    },
}

# "nodes" and "simulation" are irregular shapes, handled separately below.

SIMULATION_FIELDS = {
    "time_unit": FIELD(required=True, enum_values=TIME_UNITS),
    "horizon": FIELD(required=True),
    "warm_up": FIELD(required=True),
    "replications": FIELD(required=True),
    "random_seed": FIELD(required=False, silent=True),
}


# ----------------------------------------------------------------------
# Core: requirement checking only
# ----------------------------------------------------------------------

def check_field_requirements(config: dict) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []

    if not isinstance(config, dict):
        issues.append(ValidationIssue(
            layer=LAYER, location="$", defect_type=DefectType.MALFORMED_ENTRY,
            severity=Severity.BLOCKING,
            detail="Top-level config must be a JSON object.",
        ))
        return issues

    for section_name, spec in SECTION_SPECS.items():
        issues.extend(_check_list_section(config, section_name, spec))

    issues.extend(_check_nodes_section(config))
    issues.extend(_check_simulation_section(config))

    return issues


def _check_list_section(config: dict, section_name: str, spec: dict) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    section = config.get(section_name, None)

    if section is None:
        if spec["required_section"]:
            issues.append(ValidationIssue(
                layer=LAYER, location=section_name,
                defect_type=DefectType.MISSING_REQUIRED_VALUE,
                severity=Severity.BLOCKING,
                detail=f"Required section '{section_name}' is missing.",
            ))
        return issues

    if not isinstance(section, list):
        issues.append(ValidationIssue(
            layer=LAYER, location=section_name,
            defect_type=DefectType.MALFORMED_ENTRY,
            severity=Severity.BLOCKING,
            detail=f"Section '{section_name}' must be a list, got {type(section).__name__}.",
        ))
        return issues

    for idx, entry in enumerate(section):
        loc_prefix = f"{section_name}[{idx}]"

        if not isinstance(entry, dict):
            issues.append(ValidationIssue(
                layer=LAYER, location=loc_prefix,
                defect_type=DefectType.MALFORMED_ENTRY,
                severity=Severity.BLOCKING,
                detail=f"Entry must be a dictionary, got "
                       f"{type(entry).__name__ if entry is not None else 'None'}.",
                context={"raw_value": entry},
            ))
            continue

        issues.extend(_check_fields_present(entry, spec["fields"], loc_prefix))

    return issues


def _missing_issue(loc: str, required_now: bool, field_label: str, was_absent: bool) -> ValidationIssue:
    """Shared helper: build the right ValidationIssue for an absent-or-placeholder field."""
    if was_absent:
        detail = f"Required field '{field_label}' is absent."
    else:
        detail = f"Found placeholder '{MISSING_PLACEHOLDER}' in {'required' if required_now else 'optional'} field."
    return ValidationIssue(
        layer=LAYER, location=loc,
        defect_type=DefectType.MISSING_REQUIRED_VALUE if required_now else DefectType.MISSING_OPTIONAL_VALUE,
        severity=Severity.BLOCKING if required_now else Severity.WARNING,
        detail=detail,
    )


def _check_procurement_scheme_field(value, loc: str, required_now: bool) -> list[ValidationIssue]:
    """
    procurement_scheme has THREE shapes depending on its own "type":
      - periodic_supply: a real statistical distribution (distribution +
        parameters, describing order-quantity variability) -- reuses
        _check_distribution_field directly.
      - demand_driven: nothing else applies at all -- procurement
        triggers directly off a shortage event, matching simulate.py's
        own normalize_config, which does nothing for this type.
      - inventory_threshold: exactly two fixed threshold VALUES, not a
        distribution -- parameters.a = s (reorder point), parameters.b =
        S (order-up-to level). No "distribution" field is expected or
        checked here at all.

    "type" is resolved FIRST, since it determines what shape everything
    else should even have -- if type is missing, placeholder, or
    unrecognized, nothing else can be meaningfully checked.
    """
    issues: list[ValidationIssue] = []

    if not isinstance(value, dict):
        if value == MISSING_PLACEHOLDER:
            issues.append(_missing_issue(loc, required_now, "procurement_scheme object", was_absent=False))
        return issues

    type_loc = f"{loc}.type"
    if "type" not in value:
        issues.append(_missing_issue(type_loc, required_now, "type", was_absent=True))
        return issues
    type_val = value.get("type")
    if type_val == MISSING_PLACEHOLDER:
        issues.append(_missing_issue(type_loc, required_now, "type", was_absent=False))
        return issues
    if type_val not in PROCUREMENT_SCHEME_TYPES:
        issues.append(ValidationIssue(
            layer=LAYER, location=type_loc,
            defect_type=DefectType.INVALID_VALUE,
            severity=Severity.BLOCKING,
            detail=f"'{type_val}' is not a recognized value for this field -- must be one of: "
                   f"{', '.join(PROCUREMENT_SCHEME_TYPES)}.",
        ))
        return issues

    if type_val == "periodic_supply":
        issues.extend(_check_distribution_field(value, loc, required_now))

    elif type_val == "demand_driven":
        pass  # nothing else applies -- distribution/parameters are irrelevant

    elif type_val == "inventory_threshold":
        params_loc = f"{loc}.parameters"
        if "parameters" not in value:
            issues.append(_missing_issue(params_loc, required_now, "parameters", was_absent=True))
            return issues
        params_val = value.get("parameters")
        if params_val == MISSING_PLACEHOLDER or not isinstance(params_val, dict):
            issues.append(_missing_issue(params_loc, required_now, "parameters", was_absent=False))
            return issues

        for key in ("a", "b"):
            key_loc = f"{params_loc}.{key}"
            if key not in params_val:
                issues.append(_missing_issue(key_loc, required_now, key, was_absent=True))
            elif params_val.get(key) == MISSING_PLACEHOLDER:
                issues.append(_missing_issue(key_loc, required_now, key, was_absent=False))

    return issues


def _check_distribution_field(value, loc: str, required_now: bool) -> list[ValidationIssue]:
    """
    Checks a distribution object: {"distribution": <str>, "parameters": {a..e}}.
    How many of parameters.a-e are required depends on the "distribution"
    value itself (see DISTRIBUTION_PARAM_COUNTS).

    Per current policy: parameters actually needed by the distribution type,
    if missing/placeholder, are reported as WARNING (not BLOCKING). Unused
    parameter slots (beyond what the type needs) are never reported at all.
    """
    issues: list[ValidationIssue] = []

    if not isinstance(value, dict):
        if value == MISSING_PLACEHOLDER:
            issues.append(_missing_issue(loc, required_now, "distribution object", was_absent=False))
        return issues

    # -- "distribution" key --
    dist_loc = f"{loc}.distribution"
    dist_val = value.get("distribution", None)
    if "distribution" not in value:
        if required_now:
            issues.append(_missing_issue(dist_loc, required_now, "distribution", was_absent=True))
        dist_val = None
    elif dist_val == MISSING_PLACEHOLDER:
        issues.append(_missing_issue(dist_loc, required_now, "distribution", was_absent=False))
        dist_val = None
    elif dist_val not in DISTRIBUTION_PARAM_COUNTS:
        # A real (non-placeholder) value was given, but it's not a
        # recognized distribution type -- this is an INVALID_VALUE, not
        # a missing-value issue. Caught here so a bad categorical value
        # in the ORIGINAL config is flagged, not just values a person
        # types in during repair.
        issues.append(ValidationIssue(
            layer=LAYER, location=dist_loc,
            defect_type=DefectType.INVALID_VALUE,
            severity=Severity.BLOCKING,
            detail=f"'{dist_val}' is not a recognized distribution type -- must be "
                   f"one of: {', '.join(DISTRIBUTION_PARAM_COUNTS.keys())}.",
        ))
        dist_val = None  # unknown for param-count purposes, falls back to requiring only 'a'

    # -- "parameters" key --
    params_loc = f"{loc}.parameters"
    if "parameters" not in value:
        if required_now:
            issues.append(_missing_issue(params_loc, required_now, "parameters", was_absent=True))
        return issues

    params_val = value.get("parameters", None)
    if params_val == MISSING_PLACEHOLDER:
        issues.append(_missing_issue(params_loc, required_now, "parameters", was_absent=False))
        return issues
    if not isinstance(params_val, dict):
        return issues  # type mismatch, deferred to Step 2

    param_count = DISTRIBUTION_PARAM_COUNTS.get(dist_val, None)

    for i, key in enumerate(PARAM_KEYS):
        key_loc = f"{params_loc}.{key}"

        if param_count is not None:
            needed_by_type = i < param_count
        else:
            needed_by_type = (key == "a")

        if not needed_by_type:
            continue  # unused parameter slot -- leave it alone entirely

        if not required_now:
            continue  # the distribution object itself isn't required here

        if key not in params_val:
            issues.append(ValidationIssue(
                layer=LAYER, location=key_loc,
                defect_type=DefectType.MISSING_REQUIRED_VALUE,
                severity=Severity.BLOCKING,
                detail=f"Required field '{key}' is absent.",
            ))
            continue

        key_val = params_val.get(key)
        if key_val == MISSING_PLACEHOLDER:
            issues.append(ValidationIssue(
                layer=LAYER, location=key_loc,
                defect_type=DefectType.MISSING_REQUIRED_VALUE,
                severity=Severity.BLOCKING,
                detail=f"Found placeholder '{MISSING_PLACEHOLDER}' in required field.",
            ))

    return issues


def _check_dict_values_field(value, loc: str, required_now: bool, value_kind: str, min_items: int = 0) -> list[ValidationIssue]:
    """
    Checks a dict-with-dynamic-keys field (e.g. bom: {material_name: qty}).
    Unlike a fixed "fields" spec, every KEY here is arbitrary (a material
    name), so what's checked is each VALUE, against value_kind -- plus,
    if min_items > 0, that the dict isn't suspiciously empty.
    """
    issues: list[ValidationIssue] = []

    if not isinstance(value, dict):
        if value == MISSING_PLACEHOLDER:
            issues.append(_missing_issue(loc, required_now, "dict object", was_absent=False))
        return issues

    if min_items > 0 and len(value) < min_items:
        issues.append(ValidationIssue(
            layer=LAYER, location=loc,
            defect_type=DefectType.MISSING_REQUIRED_VALUE,
            severity=Severity.BLOCKING,
            detail=f"Must have at least {min_items} entr{'y' if min_items == 1 else 'ies'}, "
                   f"found {len(value)}.",
        ))

    for key, val in value.items():
        key_loc = f"{loc}.{key}"

        if val == MISSING_PLACEHOLDER:
            issues.append(_missing_issue(key_loc, required_now, str(key), was_absent=False))
            continue

        if value_kind == "num":
            if not isinstance(val, (int, float)) or isinstance(val, bool):
                issues.append(ValidationIssue(
                    layer=LAYER, location=key_loc,
                    defect_type=DefectType.INVALID_VALUE,
                    severity=Severity.BLOCKING,
                    detail=f"Expected a number for '{key}', got {type(val).__name__}.",
                ))

    return issues


def _check_fields_present(entry: dict, field_specs: dict, loc_prefix: str) -> list[ValidationIssue]:
    """
    Checks, per field:
      1. Presence -- does the key exist at all?
      2. Placeholder -- if present, does its value equal the "missing"
         placeholder string?
      3. Distribution fields get routed to _check_distribution_field
         (and, if present, their extra_fields checked normally too).

    Does NOT check general type correctness -- that remains Step 2.
    """
    issues: list[ValidationIssue] = []

    for field_name, fspec in field_specs.items():
        loc = f"{loc_prefix}.{field_name}"
        required_now = is_required(fspec["required"], entry)
        silent = fspec.get("silent", False)
        present = field_name in entry
        value = entry.get(field_name, None)

        if not present:
            if required_now and not silent:
                issues.append(ValidationIssue(
                    layer=LAYER, location=loc,
                    defect_type=DefectType.MISSING_REQUIRED_VALUE,
                    severity=Severity.BLOCKING,
                    detail=f"Required field '{field_name}' is absent.",
                ))
            continue

        # procurement_scheme has three genuinely different shapes
        # depending on its own "type" -- handled by dedicated logic, not
        # the generic distribution path. Same conditional-gating
        # principle: if not required in this context (e.g. a product's
        # inventory entry), skip entirely, not even a warning.
        if fspec.get("is_procurement_scheme"):
            is_conditional = callable(fspec["required"])
            if not is_conditional or required_now:
                issues.extend(_check_procurement_scheme_field(value, loc, required_now))
            continue

        # Distribution objects get special handling (their required
        # parameter count depends on the "distribution" value) rather
        # than the generic placeholder + recursion path.
        if fspec.get("is_distribution"):
            # Same principle as the "fields" branch below: if this
            # distribution field's requirement is CONDITIONAL (a callable)
            # and that condition wasn't met, the whole concept doesn't
            # apply here -- skip it entirely, not even a WARNING for
            # placeholder junk. E.g. procurement_scheme/procurement_arrival
            # are meaningless for a product/intermediate_material inventory
            # entry (only raw materials get procured), so their presence
            # with "missing" placeholders shouldn't generate any noise.
            is_conditional = callable(fspec["required"])
            if not is_conditional or required_now:
                issues.extend(_check_distribution_field(value, loc, required_now))
                extra_fields = fspec.get("extra_fields")
                if extra_fields and isinstance(value, dict) and required_now:
                    issues.extend(_check_fields_present(value, extra_fields, loc))
            continue

        # Dict-with-dynamic-keys fields (e.g. bom) -- every value inside
        # gets checked against value_kind, not a fixed set of child names.
        if fspec.get("is_dict_values"):
            issues.extend(_check_dict_values_field(
                value, loc, required_now, fspec.get("value_kind", "num"), fspec.get("min_items", 0)
            ))
            continue

        if value == MISSING_PLACEHOLDER:
            if silent:
                pass
            elif required_now:
                issues.append(ValidationIssue(
                    layer=LAYER, location=loc,
                    defect_type=DefectType.MISSING_REQUIRED_VALUE,
                    severity=Severity.BLOCKING,
                    detail=f"Found placeholder '{MISSING_PLACEHOLDER}' in required field.",
                ))
            else:
                issues.append(ValidationIssue(
                    layer=LAYER, location=loc,
                    defect_type=DefectType.MISSING_OPTIONAL_VALUE,
                    severity=Severity.WARNING,
                    detail=f"Found placeholder '{MISSING_PLACEHOLDER}' in optional field.",
                ))
            continue

        enum_values = fspec.get("enum_values")
        if enum_values is not None and value not in enum_values:
            issues.append(ValidationIssue(
                layer=LAYER, location=loc,
                defect_type=DefectType.INVALID_VALUE,
                severity=Severity.BLOCKING,
                detail=f"'{value}' is not a recognized value for this field -- "
                       f"must be one of: {', '.join(str(v) for v in enum_values)}.",
            ))
            continue

        if fspec.get("is_name"):
            name_issue_detail = None
            if not isinstance(value, str):
                name_issue_detail = (
                    f"Expected a name (string), got {type(value).__name__} "
                    f"({value!r}) -- names must be quoted text, not a raw number."
                )
            else:
                try:
                    float(value)
                    name_issue_detail = (
                        f"'{value}' is purely numeric -- names must contain "
                        f"non-numeric characters."
                    )
                except ValueError:
                    pass  # good -- a real, non-numeric string
            if name_issue_detail is not None:
                issues.append(ValidationIssue(
                    layer=LAYER, location=loc,
                    defect_type=DefectType.INVALID_VALUE,
                    severity=Severity.BLOCKING,
                    detail=name_issue_detail,
                ))
                continue

        if "fields" in fspec:
            if isinstance(value, dict):
                # Only skip checking this container's own internal
                # structure if its requirement is CONDITIONAL (a
                # callable, e.g. "operation" gated by type=="manufacturing")
                # and that condition wasn't met -- in that case the
                # whole concept doesn't apply here, regardless of what
                # placeholder junk might be sitting in it (e.g. a
                # warehouse's leftover "operation" block with "missing"
                # everywhere should be completely ignored, not enforced).
                #
                # Fields with an UNCONDITIONAL required=False (like
                # resource.batching) are different: the container itself
                # being optional doesn't mean its own internal rules stop
                # applying once it's actually present -- batch_size is
                # still conditionally required by enabled=True regardless
                # of whether batching itself was required to exist.
                is_conditional = callable(fspec["required"])
                if not is_conditional or required_now:
                    issues.extend(_check_fields_present(value, fspec["fields"], loc))

    return issues


def _check_nodes_section(config: dict) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    nodes = config.get("nodes", None)

    if nodes is None:
        issues.append(ValidationIssue(
            layer=LAYER, location="nodes", defect_type=DefectType.MISSING_REQUIRED_VALUE,
            severity=Severity.BLOCKING, detail="Required section 'nodes' is missing.",
        ))
        return issues

    if not isinstance(nodes, list) or len(nodes) == 0:
        issues.append(ValidationIssue(
            layer=LAYER, location="nodes", defect_type=DefectType.MALFORMED_ENTRY,
            severity=Severity.BLOCKING, detail="'nodes' must be a non-empty list.",
        ))
        return issues

    entry = nodes[0]
    if not isinstance(entry, dict):
        issues.append(ValidationIssue(
            layer=LAYER, location="nodes[0]", defect_type=DefectType.MALFORMED_ENTRY,
            severity=Severity.BLOCKING,
            detail=f"nodes[0] must be a dictionary, got "
                   f"{type(entry).__name__ if entry is not None else 'None'}.",
        ))
        return issues

    for key in ("supplier", "facility", "customer"):
        if key not in entry:
            issues.append(ValidationIssue(
                layer=LAYER, location=f"nodes[0].{key}",
                defect_type=DefectType.MISSING_REQUIRED_VALUE,
                severity=Severity.BLOCKING,
                detail=f"Required field '{key}' is absent from nodes[0].",
            ))

    return issues


def _check_simulation_section(config: dict) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    sim = config.get("simulation", None)

    if sim is None:
        issues.append(ValidationIssue(
            layer=LAYER, location="simulation", defect_type=DefectType.MISSING_REQUIRED_VALUE,
            severity=Severity.BLOCKING, detail="Required section 'simulation' is missing.",
        ))
        return issues

    if not isinstance(sim, dict):
        issues.append(ValidationIssue(
            layer=LAYER, location="simulation", defect_type=DefectType.MALFORMED_ENTRY,
            severity=Severity.BLOCKING,
            detail=f"'simulation' must be an object, got {type(sim).__name__}.",
        ))
        return issues

    issues.extend(_check_fields_present(sim, SIMULATION_FIELDS, "simulation"))
    return issues


if __name__ == "__main__":
    import json
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "test_config.json"
    with open(path) as f:
        cfg = json.load(f)

    found = check_field_requirements(cfg)

    if not found:
        print("Layer1 (field requirements): no issues found.")
    else:
        blocking = [i for i in found if i.severity == Severity.BLOCKING]
        warnings = [i for i in found if i.severity == Severity.WARNING]
        print(f"Layer1 (field requirements): {len(blocking)} blocking issue(s), {len(warnings)} warning(s)\n")
        for issue in found:
            print(issue)