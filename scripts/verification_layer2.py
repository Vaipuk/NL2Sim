"""
scripts2/verification_layer2.py
---------------------------------
Layer2: referential integrity checks (cross-entity references).

Unlike verification_layer1.py (which only ever looks inside a single
entry), this layer checks references BETWEEN sections -- does a name
used in one place actually exist where it's supposed to be declared.

Conditions are added one at a time, by explicit instruction, and tested
before the next one is added. Do not add new checks here without
confirming the condition first.

Implemented so far:
    1. intermediate_materials[].bom keys must exist in raw_materials[].name
    2. products[].bom keys must exist in raw_materials[].name OR
       intermediate_materials[].name
    3. inventory[].name must exist in the material list corresponding to
       inventory[].type (raw_material -> raw_materials, etc.)
    4. supplier[].supply_material_name must exist in raw_materials[].name
    5. facility[].inventory_managed entries must exist in raw_materials[],
       intermediate_materials[], or products[] (any material category)
    6. (WARNING only) for manufacturing facilities: operation.input/output
       materials should appear in that facility's own inventory_managed,
       and vice versa -- every inventory_managed material should be
       referenced in operation.input or operation.output. Two-directional,
       local to a single facility entry (not cross-entity like 1-5).
    7. customer[].product must exist in products[].name
    8. every supplier[].name must be registered in nodes[0].supplier, and
       every facility[].name must be registered in nodes[0].facility
       (the reverse of checks 1-5: an entity that EXISTS but isn't
       referenced from nodes -- this is the original TechPartners bug
       from early in this project's debugging history)
    9. edges[].material_name must exist in the material list corresponding
       to edges[].material_type (raw_material -> raw_materials, etc.) --
       same pattern as check 3 (inventory), applied to edges.
    10. facility[].operation.resource_required, IF it is set to something
        other than blank/"missing" (it's a silent/optional field per
        verification_layer1.py), must match a real resource[].name entry.
    11. edges[].source and edges[].destination must be registered node
        names (nodes[0].supplier union nodes[0].facility).
    12. REVERSE of check 8: every name in nodes[0].supplier must exist as
        a real supplier[].name, and every name in nodes[0].facility must
        exist as a real facility[].name -- a name sitting in nodes with
        nothing behind it is a phantom node.
    13. no name may be claimed by more than one of raw_materials[],
        intermediate_materials[], products[] -- the original example that
        started this layer's design.
    14. for manufacturing facilities: operation.input/output entries must
        exist in SOME material list (raw/intermediate/product), checked
        directly -- not just internally consistent with inventory_managed
        (that's check 6, which is local and WARNING-only).
    15. no duplicate names WITHIN a single section (supplier,
        raw_materials, intermediate_materials, products, resource) --
        an ambiguous reference target. FACILITY and CUSTOMER are EXEMPT
        from this blanket rule -- see checks 21 and 22, which allow
        same-named entries specifically when disjoint on the relevant
        dimension (facility: operation.output; customer: product) --
        safe because every reference that matters carries that context
        alongside the name.
    16. no duplicate edges -- same (source, destination, material_name)
        triple appearing more than once.
    17. no name may be claimed by both supplier[].name and facility[].name
        -- would make edges/nodes references ambiguous about which entity
        is meant.
    18. every customer[].name must be registered in nodes[0].customer
        (mirrors check 8, now that nodes[0] carries a customer key).
        Checks 11 and 12 were also updated to include customer names as
        valid edge endpoints / phantom-node targets, respectively.
    19. for every edge whose destination is a FACILITY, the edge's
        material_name must appear in that facility's inventory_managed
        OR operation.input -- otherwise the edge delivers material the
        destination facility doesn't acknowledge managing or consuming
        at all (a "phantom delivery"). Edges into a customer are exempt.
    20. every raw_materials[]/intermediate_materials[]/products[] entry
        must have a corresponding inventory[] entry (matched by name) --
        the REVERSE of check 3, which only verifies an EXISTING inventory
        entry points at a real material.
    21. facility-specific counterpart to check 15: two facility[] entries
        MAY share a name, but only if their operation.output sets are
        fully disjoint. Flags the case where that condition fails --
        genuinely overlapping outputs (real ambiguity), or a same-named
        facility missing operation.output entirely (can't confirm
        disjointness, treated as unsafe).
    22. customer-specific counterpart to check 15/21: two customer[]
        entries MAY share a name, but only if they order different
        products. Flags the case where they order the same product, or
        either lacks a product (can't confirm safety).
"""

from issue_types import ValidationIssue, DefectType, Severity

LAYER = "Layer2"


def _collect_names(config: dict, section: str) -> set:
    """Collect the set of 'name' values from a list-section, ignoring
    malformed entries (Layer1/Layer0's job to catch those, not ours)."""
    names = set()
    for entry in config.get(section, []) or []:
        if isinstance(entry, dict):
            name = entry.get("name")
            if isinstance(name, str):
                names.add(name)
    return names


def check_intermediate_bom_references(config: dict) -> list[ValidationIssue]:
    """
    Every key in an intermediate_materials[].bom dict must correspond to
    a real raw_materials[].name entry.
    """
    issues: list[ValidationIssue] = []

    raw_material_names = _collect_names(config, "raw_materials")

    intermediates = config.get("intermediate_materials", []) or []
    for idx, entry in enumerate(intermediates):
        if not isinstance(entry, dict):
            continue  # malformed entry, not this layer's concern

        bom = entry.get("bom")
        if not isinstance(bom, dict):
            continue  # missing/malformed bom, not this layer's concern

        for material_name in bom.keys():
            if material_name not in raw_material_names:
                issues.append(ValidationIssue(
                    layer=LAYER,
                    location=f"intermediate_materials[{idx}].bom.{material_name}",
                    defect_type=DefectType.DANGLING_REFERENCE,
                    severity=Severity.BLOCKING,
                    detail=f"bom references '{material_name}', which is not declared "
                           f"in raw_materials.",
                    context={"referenced_name": material_name, "entry_index": idx},
                ))

    return issues


def check_product_bom_references(config: dict) -> list[ValidationIssue]:
    """
    Every key in a products[].bom dict must correspond to either a real
    raw_materials[].name or a real intermediate_materials[].name -- a
    finished product can be assembled from either raw inputs directly or
    from intermediate materials produced upstream.
    """
    issues: list[ValidationIssue] = []

    valid_names = _collect_names(config, "raw_materials") | _collect_names(config, "intermediate_materials")

    products = config.get("products", []) or []
    for idx, entry in enumerate(products):
        if not isinstance(entry, dict):
            continue  # malformed entry, not this layer's concern

        bom = entry.get("bom")
        if not isinstance(bom, dict):
            continue  # missing/malformed bom, not this layer's concern

        for material_name in bom.keys():
            if material_name not in valid_names:
                issues.append(ValidationIssue(
                    layer=LAYER,
                    location=f"products[{idx}].bom.{material_name}",
                    defect_type=DefectType.DANGLING_REFERENCE,
                    severity=Severity.BLOCKING,
                    detail=f"bom references '{material_name}', which is not declared "
                           f"in raw_materials or intermediate_materials.",
                    context={"referenced_name": material_name, "entry_index": idx},
                ))

    return issues


# Identity since type values are the section names themselves. Kept as a map
# because it is also the set of recognized type values, used by both layers
# and by repair.py.
INVENTORY_TYPE_TO_SECTION = {
    "raw_materials": "raw_materials",
    "intermediate_materials": "intermediate_materials",
    "products": "products",
}


def check_inventory_type_consistency(config: dict) -> list[ValidationIssue]:
    """
    Every inventory[] entry's name must exist in the material section that
    corresponds to its declared type. E.g. an inventory entry with
    type="raw_materials" and name="silicon wafer" requires a raw_materials[]
    entry named "silicon wafer".

    An inventory[].type value not in INVENTORY_TYPE_TO_SECTION is skipped
    here -- that's an invalid-value problem (wrong enum), not a referential
    one, and isn't this layer's concern.
    """
    issues: list[ValidationIssue] = []

    # Cache each section's name set once, rather than recomputing per entry.
    section_names = {
        section: _collect_names(config, section)
        for section in INVENTORY_TYPE_TO_SECTION.values()
    }

    inventory = config.get("inventory", []) or []
    for idx, entry in enumerate(inventory):
        if not isinstance(entry, dict):
            continue  # malformed entry, not this layer's concern

        inv_type = entry.get("type")
        inv_name = entry.get("name")

        if inv_type not in INVENTORY_TYPE_TO_SECTION:
            continue  # unrecognized/missing type -- not this layer's concern
        if not isinstance(inv_name, str):
            continue  # missing/malformed name -- not this layer's concern

        target_section = INVENTORY_TYPE_TO_SECTION[inv_type]
        if inv_name not in section_names[target_section]:
            issues.append(ValidationIssue(
                layer=LAYER,
                location=f"inventory[{idx}].name",
                defect_type=DefectType.DANGLING_REFERENCE,
                severity=Severity.BLOCKING,
                detail=f"inventory entry '{inv_name}' has type '{inv_type}', but "
                       f"'{inv_name}' is not declared in {target_section}.",
                context={"referenced_name": inv_name, "inventory_type": inv_type, "entry_index": idx},
            ))

    return issues


def check_supplier_material_references(config: dict) -> list[ValidationIssue]:
    """
    Every supplier[].supply_material_name must correspond to a real
    raw_materials[].name entry -- suppliers only source raw materials
    (intermediate materials and products are produced internally, not
    purchased from an external supplier).
    """
    issues: list[ValidationIssue] = []

    raw_material_names = _collect_names(config, "raw_materials")

    suppliers = config.get("supplier", []) or []
    for idx, entry in enumerate(suppliers):
        if not isinstance(entry, dict):
            continue  # malformed entry, not this layer's concern

        material_name = entry.get("supply_material_name")
        if not isinstance(material_name, str):
            continue  # missing/malformed name, not this layer's concern

        if material_name not in raw_material_names:
            issues.append(ValidationIssue(
                layer=LAYER,
                location=f"supplier[{idx}].supply_material_name",
                defect_type=DefectType.DANGLING_REFERENCE,
                severity=Severity.BLOCKING,
                detail=f"supply_material_name references '{material_name}', which is "
                       f"not declared in raw_materials.",
                context={"referenced_name": material_name, "entry_index": idx},
            ))

    return issues


def check_facility_inventory_managed_references(config: dict) -> list[ValidationIssue]:
    """
    Every entry in facility[].inventory_managed must correspond to a real
    material somewhere in the config -- raw, intermediate, or finished
    product. A facility can manage inventory of any material category
    (e.g. Wafer Fab manages a raw material; Warehouse manages a product),
    so this checks against the union of all three, unlike the
    inventory[] check which is type-specific.
    """
    issues: list[ValidationIssue] = []

    all_material_names = (
        _collect_names(config, "raw_materials")
        | _collect_names(config, "intermediate_materials")
        | _collect_names(config, "products")
    )

    facilities = config.get("facility", []) or []
    for idx, entry in enumerate(facilities):
        if not isinstance(entry, dict):
            continue  # malformed entry, not this layer's concern

        managed = entry.get("inventory_managed")
        if not isinstance(managed, list):
            continue  # missing/malformed field, not this layer's concern

        for i, material_name in enumerate(managed):
            if not isinstance(material_name, str):
                continue  # malformed list item, not this layer's concern

            if material_name not in all_material_names:
                issues.append(ValidationIssue(
                    layer=LAYER,
                    location=f"facility[{idx}].inventory_managed[{i}]",
                    defect_type=DefectType.DANGLING_REFERENCE,
                    severity=Severity.BLOCKING,
                    detail=f"inventory_managed references '{material_name}', which is not "
                           f"declared in raw_materials, intermediate_materials, or products.",
                    context={"referenced_name": material_name, "entry_index": idx},
                ))

    return issues


def check_facility_operation_inventory_consistency(config: dict) -> list[ValidationIssue]:
    """
    For manufacturing facilities only: cross-check operation.input/output
    against the facility's own inventory_managed list, in both directions.
    WARNING severity only, per instruction -- this flags a real modeling
    gap (e.g. an immediately-transferred output the facility never holds
    in its own inventory) without blocking anything, since it may be a
    deliberate design choice rather than an error.

    Direction A: every material in operation.input / operation.output
                 should appear in inventory_managed.
    Direction B: every material in inventory_managed should appear in
                 either operation.input or operation.output.

    Unlike checks 1-5, this is LOCAL to a single facility entry (input,
    output, and inventory_managed are all fields of the same entry) --
    included here anyway since it's still a reference-consistency check,
    just not a cross-SECTION one.
    """
    issues: list[ValidationIssue] = []

    facilities = config.get("facility", []) or []
    for idx, entry in enumerate(facilities):
        if not isinstance(entry, dict):
            continue  # malformed entry, not this layer's concern

        if entry.get("type") != "manufacturing":
            continue  # only manufacturing facilities have operation input/output

        operation = entry.get("operation")
        if not isinstance(operation, dict):
            continue  # missing/malformed operation, not this check's concern

        input_list = operation.get("input")
        output_list = operation.get("output")
        managed_list = entry.get("inventory_managed")

        managed_set = set(m for m in managed_list if isinstance(m, str)) if isinstance(managed_list, list) else set()
        io_set = set()
        if isinstance(input_list, list):
            io_set |= set(m for m in input_list if isinstance(m, str))
        if isinstance(output_list, list):
            io_set |= set(m for m in output_list if isinstance(m, str))

        # Direction A: input/output materials should be in inventory_managed.
        for list_name, material_list in (("input", input_list), ("output", output_list)):
            if not isinstance(material_list, list):
                continue
            for i, material_name in enumerate(material_list):
                if not isinstance(material_name, str):
                    continue
                if material_name not in managed_set:
                    issues.append(ValidationIssue(
                        layer=LAYER,
                        location=f"facility[{idx}].operation.{list_name}[{i}]",
                        defect_type=DefectType.INCONSISTENT_CROSS_FIELD,
                        severity=Severity.WARNING,
                        detail=f"operation.{list_name} references '{material_name}', which is "
                               f"not present in this facility's inventory_managed.",
                        context={"referenced_name": material_name, "entry_index": idx},
                    ))

        # Direction B: inventory_managed materials should be referenced in
        # either input or output.
        if isinstance(managed_list, list):
            for i, material_name in enumerate(managed_list):
                if not isinstance(material_name, str):
                    continue
                if material_name not in io_set:
                    issues.append(ValidationIssue(
                        layer=LAYER,
                        location=f"facility[{idx}].inventory_managed[{i}]",
                        defect_type=DefectType.INCONSISTENT_CROSS_FIELD,
                        severity=Severity.WARNING,
                        detail=f"inventory_managed includes '{material_name}', which is not "
                               f"referenced in operation.input or operation.output.",
                        context={"referenced_name": material_name, "entry_index": idx},
                    ))

    return issues


def check_customer_product_references(config: dict) -> list[ValidationIssue]:
    """
    Every customer[].product must correspond to a real products[].name
    entry -- customers only order finished products (not raw or
    intermediate materials, which aren't sold directly).
    """
    issues: list[ValidationIssue] = []

    product_names = _collect_names(config, "products")

    customers = config.get("customer", []) or []
    for idx, entry in enumerate(customers):
        if not isinstance(entry, dict):
            continue  # malformed entry, not this layer's concern

        product_name = entry.get("product")
        if not isinstance(product_name, str):
            continue  # missing/malformed field, not this layer's concern

        if product_name not in product_names:
            issues.append(ValidationIssue(
                layer=LAYER,
                location=f"customer[{idx}].product",
                defect_type=DefectType.DANGLING_REFERENCE,
                severity=Severity.BLOCKING,
                detail=f"product references '{product_name}', which is not declared "
                       f"in products.",
                context={"referenced_name": product_name, "entry_index": idx},
            ))

    return issues


def check_nodes_registration_completeness(config: dict) -> list[ValidationIssue]:
    """
    Every supplier[].name and facility[].name must be registered in
    nodes[0] under the corresponding key. This is the reverse direction
    of checks 1-5 (which check "does this reference point to something
    real") -- here we check "does this real thing have a reference
    pointing TO it from nodes." An entity that exists but was never
    registered in nodes is invisible to graph traversal even though it's
    a perfectly valid entity on its own -- this was the exact bug behind
    the original TechPartners supplier-creation crash earlier in this
    project (a what-if op created the supplier node but never appended
    it to nodes[0].supplier).
    """
    issues: list[ValidationIssue] = []

    nodes = config.get("nodes")
    if not isinstance(nodes, list) or len(nodes) == 0 or not isinstance(nodes[0], dict):
        return issues  # malformed/missing nodes, not this check's concern (Layer1's job)

    node_entry = nodes[0]
    registered_suppliers = set(n for n in (node_entry.get("supplier") or []) if isinstance(n, str))
    registered_facilities = set(n for n in (node_entry.get("facility") or []) if isinstance(n, str))

    for idx, entry in enumerate(config.get("supplier", []) or []):
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        if isinstance(name, str) and name not in registered_suppliers:
            issues.append(ValidationIssue(
                layer=LAYER,
                location="nodes[0].supplier",
                defect_type=DefectType.INCONSISTENT_CROSS_FIELD,
                severity=Severity.BLOCKING,
                detail=f"supplier '{name}' is defined in supplier[{idx}] but not "
                       f"registered in nodes[0].supplier.",
                context={"referenced_name": name, "entry_index": idx},
            ))

    for idx, entry in enumerate(config.get("facility", []) or []):
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        if isinstance(name, str) and name not in registered_facilities:
            issues.append(ValidationIssue(
                layer=LAYER,
                location="nodes[0].facility",
                defect_type=DefectType.INCONSISTENT_CROSS_FIELD,
                severity=Severity.BLOCKING,
                detail=f"facility '{name}' is defined in facility[{idx}] but not "
                       f"registered in nodes[0].facility.",
                context={"referenced_name": name, "entry_index": idx},
            ))

    return issues


def check_nodes_customer_registration(config: dict) -> list[ValidationIssue]:
    """
    Every customer[].name must be registered in nodes[0].customer.
    Mirrors check_nodes_registration_completeness's supplier/facility
    logic exactly. Activated now that nodes[0] is expected to carry a
    "customer" key going forward.
    """
    issues: list[ValidationIssue] = []

    nodes = config.get("nodes")
    if not isinstance(nodes, list) or len(nodes) == 0 or not isinstance(nodes[0], dict):
        return issues

    node_entry = nodes[0]
    registered_customers = set(n for n in (node_entry.get("customer") or []) if isinstance(n, str))

    for idx, entry in enumerate(config.get("customer", []) or []):
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        if isinstance(name, str) and name not in registered_customers:
            issues.append(ValidationIssue(
                layer=LAYER,
                location="nodes[0].customer",
                defect_type=DefectType.INCONSISTENT_CROSS_FIELD,
                severity=Severity.BLOCKING,
                detail=f"customer '{name}' is defined in customer[{idx}] but not "
                       f"registered in nodes[0].customer.",
                context={"referenced_name": name, "entry_index": idx},
            ))

    return issues


def check_material_has_inventory_entry(config: dict) -> list[ValidationIssue]:
    """
    Every raw_materials[]/intermediate_materials[]/products[] entry must
    have a corresponding inventory[] entry (matched by name). This is the
    REVERSE direction of check_inventory_type_consistency (check 3), which
    only verifies that an EXISTING inventory entry points at a real
    material -- it never checks that every real material has an inventory
    entry to begin with. Without this, a material can be fully valid,
    producible, and consumable, yet have no tracked stock level at all --
    the simulation engine has nothing to initialize for it.

    SAFETY NOTE: this fires at "raw_materials[idx]" / "intermediate_materials[idx]"
    / "products[idx]" -- shapes ALREADY heavily used by multiple Layer3
    checks (1, 5, 6, 7, 4, 11). Repair for this must be added to the
    EXISTING try-in-sequence dispatch lists for those locations, with its
    own live-condition guard, not a new standalone dispatch branch.
    """
    issues: list[ValidationIssue] = []

    inventory_names = _collect_names(config, "inventory")
    # type value == section name; listed explicitly to fix iteration order
    category_to_type = {s: s for s in INVENTORY_TYPE_TO_SECTION}

    for section, expected_type in category_to_type.items():
        for idx, entry in enumerate(config.get(section, []) or []):
            if not isinstance(entry, dict):
                continue
            name = entry.get("name")
            if not isinstance(name, str):
                continue
            if name not in inventory_names:
                issues.append(ValidationIssue(
                    layer=LAYER,
                    location=f"{section}[{idx}]",
                    defect_type=DefectType.INCONSISTENT_CROSS_FIELD,
                    severity=Severity.BLOCKING,
                    detail=f"'{name}' is declared in {section} but has no corresponding "
                           f"inventory[] entry -- its stock level is never tracked.",
                    context={"referenced_name": name, "expected_type": expected_type, "entry_index": idx},
                ))

    return issues


def check_edge_destination_material_consistency(config: dict) -> list[ValidationIssue]:
    """
    For every edge whose destination is a facility, the edge's
    material_name must appear in that facility's inventory_managed OR
    operation.input -- otherwise the edge is a "phantom delivery": the
    destination facility receives material it doesn't acknowledge
    managing or consuming at all. This is a real, concrete supply-chain
    error (found via an actual repair-flow bug: a supplier's outbound
    edge got created to a facility that didn't use the material at all,
    because at edge-creation time nothing referenced it yet -- nothing
    previously caught that the resulting edge was physically meaningless).

    Edges whose destination is a customer are exempt (customers have no
    inventory_managed/operation to check against).
    """
    issues: list[ValidationIssue] = []

    facility_names = _collect_names(config, "facility")
    facility_by_name = {
        f.get("name"): f for f in config.get("facility", []) or [] if isinstance(f, dict)
    }

    for idx, entry in enumerate(config.get("edges", []) or []):
        if not isinstance(entry, dict):
            continue
        destination = entry.get("destination")
        material_name = entry.get("material_name")
        if not isinstance(destination, str) or not isinstance(material_name, str):
            continue
        if destination not in facility_names:
            continue  # destination is a customer (or unresolved) -- not this check's concern

        facility = facility_by_name.get(destination)
        if facility is None:
            continue

        managed = facility.get("inventory_managed") or []
        operation = facility.get("operation") or {}
        inputs = operation.get("input") or []

        if material_name not in managed and material_name not in inputs:
            issues.append(ValidationIssue(
                layer=LAYER,
                location=f"edges[{idx}].material_name",
                defect_type=DefectType.DANGLING_REFERENCE,
                severity=Severity.BLOCKING,
                detail=f"edge delivers '{material_name}' to facility '{destination}', but that "
                       f"facility's inventory_managed and operation.input don't include it -- "
                       f"a phantom delivery the facility doesn't acknowledge.",
                context={
                    "referenced_name": material_name,
                    "destination_facility": destination,
                    "entry_index": idx,
                },
            ))

    return issues


def check_edge_material_references(config: dict) -> list[ValidationIssue]:
    """
    Every edges[].material_name must exist in the material list
    corresponding to edges[].material_type -- same mapping and same
    pattern as check_inventory_type_consistency, applied to edges instead
    of inventory entries.

    An edges[].material_type value not in INVENTORY_TYPE_TO_SECTION is
    skipped here -- that's an invalid-enum-value problem, not a
    referential one, and isn't this layer's concern (same scoping
    decision as check 3).
    """
    issues: list[ValidationIssue] = []

    section_names = {
        section: _collect_names(config, section)
        for section in INVENTORY_TYPE_TO_SECTION.values()
    }

    edges = config.get("edges", []) or []
    for idx, entry in enumerate(edges):
        if not isinstance(entry, dict):
            continue  # malformed entry, not this layer's concern

        material_type = entry.get("material_type")
        material_name = entry.get("material_name")

        if material_type not in INVENTORY_TYPE_TO_SECTION:
            continue  # unrecognized/missing type -- not this layer's concern
        if not isinstance(material_name, str):
            continue  # missing/malformed name -- not this layer's concern

        target_section = INVENTORY_TYPE_TO_SECTION[material_type]
        if material_name not in section_names[target_section]:
            issues.append(ValidationIssue(
                layer=LAYER,
                location=f"edges[{idx}].material_name",
                defect_type=DefectType.DANGLING_REFERENCE,
                severity=Severity.BLOCKING,
                detail=f"edge has material_type '{material_type}', but "
                       f"'{material_name}' is not declared in {target_section}.",
                context={"referenced_name": material_name, "material_type": material_type, "entry_index": idx},
            ))

    return issues


def check_operation_resource_references(config: dict) -> list[ValidationIssue]:
    """
    facility[].operation.resource_required is optional/silent in
    verification_layer1.py -- absence or the "missing" placeholder is not
    an error, since not every operation needs a named resource constraint.
    But IF a real value is given, it must correspond to an actual
    resource[].name entry.
    """
    issues: list[ValidationIssue] = []

    resource_names = _collect_names(config, "resource")

    facilities = config.get("facility", []) or []
    for idx, entry in enumerate(facilities):
        if not isinstance(entry, dict):
            continue  # malformed entry, not this layer's concern

        operation = entry.get("operation")
        if not isinstance(operation, dict):
            continue  # missing/malformed operation, not this check's concern

        resource_required = operation.get("resource_required")

        # Blank / missing / not-yet-filled-in -- not an error, skip.
        if resource_required is None:
            continue
        if resource_required == "missing":
            continue
        if isinstance(resource_required, str) and resource_required.strip() == "":
            continue

        if not isinstance(resource_required, str):
            continue  # wrong type entirely -- deferred to Step 2, not this layer's concern

        if resource_required not in resource_names:
            issues.append(ValidationIssue(
                layer=LAYER,
                location=f"facility[{idx}].operation.resource_required",
                defect_type=DefectType.DANGLING_REFERENCE,
                severity=Severity.BLOCKING,
                detail=f"resource_required references '{resource_required}', which is "
                       f"not declared in resource.",
                context={"referenced_name": resource_required, "entry_index": idx},
            ))

    return issues


def check_edge_node_references(config: dict) -> list[ValidationIssue]:
    """
    Every edges[].source and edges[].destination must be a name registered
    in nodes[0] (as a supplier, facility, or customer). nodes[0] is the
    canonical set of valid graph endpoints per the schema -- an edge
    pointing at a name not in nodes is a dangling connection, whether or
    not that name happens to exist as a real entity elsewhere (that's
    check 12's job to catch separately).
    """
    issues: list[ValidationIssue] = []

    nodes = config.get("nodes")
    if not isinstance(nodes, list) or len(nodes) == 0 or not isinstance(nodes[0], dict):
        return issues  # malformed/missing nodes, not this check's concern

    node_entry = nodes[0]
    valid_endpoints = set(n for n in (node_entry.get("supplier") or []) if isinstance(n, str))
    valid_endpoints |= set(n for n in (node_entry.get("facility") or []) if isinstance(n, str))
    valid_endpoints |= set(n for n in (node_entry.get("customer") or []) if isinstance(n, str))

    edges = config.get("edges", []) or []
    for idx, entry in enumerate(edges):
        if not isinstance(entry, dict):
            continue

        for field_name in ("source", "destination"):
            endpoint = entry.get(field_name)
            if not isinstance(endpoint, str):
                continue  # missing/malformed, not this layer's concern

            if endpoint not in valid_endpoints:
                issues.append(ValidationIssue(
                    layer=LAYER,
                    location=f"edges[{idx}].{field_name}",
                    defect_type=DefectType.DANGLING_REFERENCE,
                    severity=Severity.BLOCKING,
                    detail=f"{field_name} references '{endpoint}', which is not "
                           f"registered in nodes[0] as a supplier, facility, or customer.",
                    context={"referenced_name": endpoint, "entry_index": idx},
                ))

    return issues


def check_nodes_names_exist(config: dict) -> list[ValidationIssue]:
    """
    REVERSE of check_nodes_registration_completeness (check 8): every name
    listed in nodes[0].supplier must correspond to a real supplier[].name,
    and every name in nodes[0].facility must correspond to a real
    facility[].name. A name sitting in nodes with nothing behind it (e.g.
    left over after a rename or deletion) is a phantom node.
    """
    issues: list[ValidationIssue] = []

    nodes = config.get("nodes")
    if not isinstance(nodes, list) or len(nodes) == 0 or not isinstance(nodes[0], dict):
        return issues

    node_entry = nodes[0]
    real_supplier_names = _collect_names(config, "supplier")
    real_facility_names = _collect_names(config, "facility")
    real_customer_names = _collect_names(config, "customer")

    for i, name in enumerate(node_entry.get("supplier") or []):
        if isinstance(name, str) and name not in real_supplier_names:
            issues.append(ValidationIssue(
                layer=LAYER,
                location=f"nodes[0].supplier[{i}]",
                defect_type=DefectType.DANGLING_REFERENCE,
                severity=Severity.BLOCKING,
                detail=f"nodes[0].supplier lists '{name}', which does not correspond "
                       f"to any real supplier[].name.",
                context={"referenced_name": name},
            ))

    for i, name in enumerate(node_entry.get("facility") or []):
        if isinstance(name, str) and name not in real_facility_names:
            issues.append(ValidationIssue(
                layer=LAYER,
                location=f"nodes[0].facility[{i}]",
                defect_type=DefectType.DANGLING_REFERENCE,
                severity=Severity.BLOCKING,
                detail=f"nodes[0].facility lists '{name}', which does not correspond "
                       f"to any real facility[].name.",
                context={"referenced_name": name},
            ))

    for i, name in enumerate(node_entry.get("customer") or []):
        if isinstance(name, str) and name not in real_customer_names:
            issues.append(ValidationIssue(
                layer=LAYER,
                location=f"nodes[0].customer[{i}]",
                defect_type=DefectType.DANGLING_REFERENCE,
                severity=Severity.BLOCKING,
                detail=f"nodes[0].customer lists '{name}', which does not correspond "
                       f"to any real customer[].name.",
                context={"referenced_name": name},
            ))

    return issues


def check_material_name_uniqueness(config: dict) -> list[ValidationIssue]:
    """
    No single name may be claimed by more than one of raw_materials[],
    intermediate_materials[], products[]. A name in two categories at
    once is fundamentally ambiguous -- every check that resolves "which
    category is this material" (checks 1, 2, 3, 4, 9) implicitly assumes
    uniqueness across these three lists. This is the original example
    that motivated this entire layer.
    """
    issues: list[ValidationIssue] = []

    category_sections = ["raw_materials", "intermediate_materials", "products"]
    names_by_section = {section: _collect_names(config, section) for section in category_sections}

    # Compare every pair of sections for overlap.
    for i in range(len(category_sections)):
        for j in range(i + 1, len(category_sections)):
            section_a = category_sections[i]
            section_b = category_sections[j]
            overlap = names_by_section[section_a] & names_by_section[section_b]
            for name in sorted(overlap):
                issues.append(ValidationIssue(
                    layer=LAYER,
                    location=f"{section_a}/{section_b}",
                    defect_type=DefectType.DUPLICATE_ENTITY,
                    severity=Severity.BLOCKING,
                    detail=f"'{name}' is declared in both {section_a} and {section_b} -- "
                           f"a material name must belong to exactly one category.",
                    context={"referenced_name": name, "sections": [section_a, section_b]},
                ))

    return issues


def check_operation_io_material_references(config: dict) -> list[ValidationIssue]:
    """
    For manufacturing facilities: every entry in operation.input and
    operation.output must exist in SOME material list (raw, intermediate,
    or product), checked directly against the real material declarations.

    This is distinct from check 6 (check_facility_operation_inventory_consistency),
    which only checks LOCAL consistency between a facility's own
    input/output and its own inventory_managed -- two lists could agree
    with each other while both referencing a material that doesn't exist
    anywhere in the config, and check 6 would never catch that.
    """
    issues: list[ValidationIssue] = []

    all_material_names = (
        _collect_names(config, "raw_materials")
        | _collect_names(config, "intermediate_materials")
        | _collect_names(config, "products")
    )

    facilities = config.get("facility", []) or []
    for idx, entry in enumerate(facilities):
        if not isinstance(entry, dict):
            continue
        if entry.get("type") != "manufacturing":
            continue

        operation = entry.get("operation")
        if not isinstance(operation, dict):
            continue

        for list_name in ("input", "output"):
            material_list = operation.get(list_name)
            if not isinstance(material_list, list):
                continue
            for i, material_name in enumerate(material_list):
                if not isinstance(material_name, str):
                    continue
                if material_name not in all_material_names:
                    issues.append(ValidationIssue(
                        layer=LAYER,
                        location=f"facility[{idx}].operation.{list_name}[{i}]",
                        defect_type=DefectType.DANGLING_REFERENCE,
                        severity=Severity.BLOCKING,
                        detail=f"operation.{list_name} references '{material_name}', which "
                               f"is not declared in raw_materials, intermediate_materials, "
                               f"or products.",
                        context={"referenced_name": material_name, "entry_index": idx},
                    ))

    return issues


# Sections where a duplicate "name" within the same section is meaningful
# to check (every one of these is referenced BY NAME from somewhere else
# in the config, so a duplicate creates an ambiguous target).
DUPLICATE_NAME_SECTIONS = [
    "supplier", "raw_materials", "intermediate_materials",
    "products", "resource",
]


def check_duplicate_names_within_sections(config: dict) -> list[ValidationIssue]:
    """
    No two entries within the same section may share a "name" -- a
    duplicate makes it ambiguous which entry a reference elsewhere in the
    config (bom key, edge endpoint, inventory name, etc.) actually means.

    EXCEPTION -- facility is handled separately by
    check_facility_duplicate_names_disjoint_output below. A resource is
    scoped to a single operation, so two independent production lines
    needing different resources genuinely cannot be folded into one
    facility entry's multi-item operation.output -- two facility[]
    entries sharing a name, each with its own operation/resource, is the
    correct way to model that. This is safe specifically because every
    downstream reference that could matter (edges, phantom-delivery
    repair, etc.) already carries material context alongside the name --
    the ambiguity a bare duplicate name would otherwise create is
    resolved by which material is actually involved, PROVIDED the
    same-named facilities' outputs never overlap (see that check for the
    overlap condition that still gets flagged).
    """
    issues: list[ValidationIssue] = []

    for section in DUPLICATE_NAME_SECTIONS:
        entries = config.get(section, []) or []
        seen_at: dict[str, list[int]] = {}
        for idx, entry in enumerate(entries):
            if not isinstance(entry, dict):
                continue
            name = entry.get("name")
            if not isinstance(name, str):
                continue
            seen_at.setdefault(name, []).append(idx)

        for name, indices in seen_at.items():
            if len(indices) > 1:
                issues.append(ValidationIssue(
                    layer=LAYER,
                    location=f"{section}[{','.join(str(i) for i in indices)}]",
                    defect_type=DefectType.DUPLICATE_ENTITY,
                    severity=Severity.BLOCKING,
                    detail=f"'{name}' appears {len(indices)} times in {section} "
                           f"(indices {indices}) -- names must be unique within a section.",
                    context={"referenced_name": name, "section": section, "indices": indices},
                ))

    return issues


def check_facility_duplicate_names_disjoint_output(config: dict) -> list[ValidationIssue]:
    """
    Facility is exempt from check_duplicate_names_within_sections' blanket
    uniqueness rule, but NOT unconditionally -- two facility[] entries
    sharing a name are only safe if their operation.output sets never
    overlap (the material context is what resolves the reference
    ambiguity a bare duplicate name would otherwise create). This check
    catches the case where that condition FAILS:
      - two same-named facilities whose outputs actually overlap (genuine
        ambiguity: which one does an output-based reference mean?), or
      - a same-named facility missing an operation/output entirely
        (can't confirm disjointness at all, so treated as unsafe).
    """
    issues: list[ValidationIssue] = []

    entries = config.get("facility", []) or []
    seen_at: dict[str, list[int]] = {}
    for idx, entry in enumerate(entries):
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        if not isinstance(name, str):
            continue
        seen_at.setdefault(name, []).append(idx)

    for name, indices in seen_at.items():
        if len(indices) < 2:
            continue

        output_sets = []
        any_missing_output = False
        for idx in indices:
            entry = entries[idx]
            operation = entry.get("operation")
            outputs = (operation or {}).get("output") if isinstance(operation, dict) else None
            if not outputs:
                any_missing_output = True
                output_sets.append(set())
            else:
                output_sets.append(set(m for m in outputs if isinstance(m, str)))

        overlap = set()
        for i in range(len(output_sets)):
            for j in range(i + 1, len(output_sets)):
                overlap |= (output_sets[i] & output_sets[j])

        if any_missing_output or overlap:
            reason = (
                f"at least one lacks a defined operation.output" if any_missing_output
                else f"they overlap on: {sorted(overlap)}"
            )
            issues.append(ValidationIssue(
                layer=LAYER,
                location=f"facility[{','.join(str(i) for i in indices)}]",
                defect_type=DefectType.DUPLICATE_ENTITY,
                severity=Severity.BLOCKING,
                detail=f"'{name}' appears {len(indices)} times in facility (indices {indices}), "
                       f"but this is only safe when their outputs are fully disjoint -- {reason}.",
                context={"referenced_name": name, "section": "facility", "indices": indices},
            ))

    return issues


def check_customer_duplicate_names_disjoint_product(config: dict) -> list[ValidationIssue]:
    """
    Customer-specific counterpart to check_facility_duplicate_names_disjoint_output:
    two customer[] entries MAY share a name, but only if they order
    DIFFERENT products -- customer.product is a single string (not a set
    like facility.operation.output), so "disjoint" here just means the
    values differ. An edge/nodes reference to a shared customer name is
    resolved the same way a facility one is: by which material/product
    is actually involved.

    Flags the case where two same-named customers order the SAME product
    (genuine ambiguity -- which one does a reference mean?), or where
    either lacks a product at all (can't confirm safety).
    """
    issues: list[ValidationIssue] = []

    entries = config.get("customer", []) or []
    seen_at: dict[str, list[int]] = {}
    for idx, entry in enumerate(entries):
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        if not isinstance(name, str):
            continue
        seen_at.setdefault(name, []).append(idx)

    for name, indices in seen_at.items():
        if len(indices) < 2:
            continue

        products = []
        any_missing_product = False
        for idx in indices:
            product = entries[idx].get("product")
            if not isinstance(product, str) or not product:
                any_missing_product = True
                products.append(None)
            else:
                products.append(product)

        seen_products = [p for p in products if p is not None]
        has_duplicate_product = len(seen_products) != len(set(seen_products))

        if any_missing_product or has_duplicate_product:
            reason = (
                "at least one lacks a defined product" if any_missing_product
                else f"they both order the same product: '{seen_products[0]}'"
            )
            issues.append(ValidationIssue(
                layer=LAYER,
                location=f"customer[{','.join(str(i) for i in indices)}]",
                defect_type=DefectType.DUPLICATE_ENTITY,
                severity=Severity.BLOCKING,
                detail=f"'{name}' appears {len(indices)} times in customer (indices {indices}), "
                       f"but this is only safe when they order different products -- {reason}.",
                context={"referenced_name": name, "section": "customer", "indices": indices},
            ))

    return issues


def check_no_self_loop_edges(config: dict) -> list[ValidationIssue]:
    """
    An edge whose source and destination are the SAME node name is
    structurally meaningless in this domain -- material "moving" from a
    location to itself represents nothing real (unlike, say, a graph
    algorithm domain where self-loops can be legitimate). This is always
    a defect, never a valid modeling choice, so there is no ambiguity to
    resolve here -- the corresponding repair action deletes the edge
    outright rather than prompting, matching how transfer_time defaults
    are handled automatically elsewhere in this system.
    """
    issues: list[ValidationIssue] = []

    for idx, entry in enumerate(config.get("edges", []) or []):
        if not isinstance(entry, dict):
            continue
        source = entry.get("source")
        destination = entry.get("destination")
        if not isinstance(source, str) or not isinstance(destination, str):
            continue
        if source == destination:
            material = entry.get("material_name")
            material_note = f" (material: '{material}')" if isinstance(material, str) and material and material != "missing" else ""
            issues.append(ValidationIssue(
                layer=LAYER,
                location=f"edges[{idx}]",
                defect_type=DefectType.MALFORMED_ENTRY,
                severity=Severity.BLOCKING,
                detail=f"Edge {idx} is a self-loop: source and destination are both "
                       f"'{source}'{material_note} -- a node cannot deliver to itself. This "
                       f"edge will be removed automatically.",
                context={"source": source, "destination": destination},
            ))

    return issues


def check_duplicate_edges(config: dict) -> list[ValidationIssue]:
    """
    No two edges may share the same (source, destination, material_name)
    triple -- a duplicate is redundant at best and a data-entry error at
    worst (e.g. accidentally re-running a what-if that adds the same edge
    twice).
    """
    issues: list[ValidationIssue] = []

    edges = config.get("edges", []) or []
    seen_at: dict[tuple, list[int]] = {}
    for idx, entry in enumerate(edges):
        if not isinstance(entry, dict):
            continue
        source = entry.get("source")
        destination = entry.get("destination")
        material_name = entry.get("material_name")
        if not all(isinstance(v, str) for v in (source, destination, material_name)):
            continue  # malformed entry, not this check's concern

        key = (source, destination, material_name)
        seen_at.setdefault(key, []).append(idx)

    for (source, destination, material_name), indices in seen_at.items():
        if len(indices) > 1:
            issues.append(ValidationIssue(
                layer=LAYER,
                location=f"edges[{','.join(str(i) for i in indices)}]",
                defect_type=DefectType.DUPLICATE_ENTITY,
                severity=Severity.BLOCKING,
                detail=f"edge ({source} -> {destination}, material '{material_name}') "
                       f"appears {len(indices)} times (indices {indices}).",
                context={"source": source, "destination": destination,
                         "material_name": material_name, "indices": indices},
            ))

    return issues


def check_supplier_facility_name_collision(config: dict) -> list[ValidationIssue]:
    """
    No name may be claimed by both supplier[].name and facility[].name --
    if it were, an edges[] entry or nodes[] reference to that name would
    be genuinely ambiguous about which entity (the supplier or the
    facility) is meant.
    """
    issues: list[ValidationIssue] = []

    supplier_names = _collect_names(config, "supplier")
    facility_names = _collect_names(config, "facility")
    overlap = supplier_names & facility_names

    for name in sorted(overlap):
        issues.append(ValidationIssue(
            layer=LAYER,
            location="supplier/facility",
            defect_type=DefectType.DUPLICATE_ENTITY,
            severity=Severity.BLOCKING,
            detail=f"'{name}' is declared as both a supplier and a facility -- "
                   f"references to this name would be ambiguous.",
            context={"referenced_name": name},
        ))

    return issues


# ----------------------------------------------------------------------
# CLI entry point for quick manual testing
# ----------------------------------------------------------------------

if __name__ == "__main__":
    import json
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "test_config.json"
    with open(path) as f:
        cfg = json.load(f)

    found = (
        check_intermediate_bom_references(cfg)
        + check_product_bom_references(cfg)
        + check_inventory_type_consistency(cfg)
        + check_supplier_material_references(cfg)
        + check_facility_inventory_managed_references(cfg)
        + check_facility_operation_inventory_consistency(cfg)
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
        + check_edge_destination_material_consistency(cfg)
        + check_material_has_inventory_entry(cfg)
        + check_facility_duplicate_names_disjoint_output(cfg)
        + check_customer_duplicate_names_disjoint_product(cfg)
        + check_no_self_loop_edges(cfg)
    )

    if not found:
        print("Layer2: no issues found.")
    else:
        blocking = [i for i in found if i.severity == Severity.BLOCKING]
        warnings = [i for i in found if i.severity == Severity.WARNING]
        print(f"Layer2: {len(blocking)} blocking issue(s), {len(warnings)} warning(s)\n")
        for issue in found:
            print(issue)