"""Framework-owned boundary interfaces, with additive per-scenario/variant slots."""
import hashlib
import re

from validation.test_inspector import C_PARSER, _node_text, _walk, inspect_test_source
from scenario.condition_variants import planned_variants, variant_plan_errors

BEGIN = "/* RACA FIXED MOCK INTERFACES BEGIN */"
END = "/* RACA FIXED MOCK INTERFACES END */"


def framework_hook_specs(context):
    """Keep the compiler-owned type already exposed by each hook interface."""
    specifications = []
    for control in (context.get('target') or {}).get('export_interface_details', []):
        if control.get('source_kind') != 'boundary_hook' or control.get('boundary_control_role') != 'set_hook':
            continue
        original, boundary = control.get('boundary_expression', ''), control.get('boundary_id', '')
        if not boundary or not re.fullmatch(r'[A-Za-z_]\w*', original):
            raise ValueError('A direct hook must retain its source boundary and function identity')
        specifications.append({'boundary_id': boundary, 'original': original,
            'canonical_fake_name': 'fake_' + original, 'return_type': 'source_hook_type',
            'params': 'source_hook_type', 'signature_type': 'typeof(' + original + ')',
            'control_kind': 'hook'})
    return specifications


def allocation_errors(context):
    """An exported hook cannot be advertised with an empty allocation domain."""
    if 'mock_interfaces' not in context:
        return []
    errors = []
    controls = [c for c in (context.get('target') or {}).get('export_interface_details', [])
                if c.get('source_kind') == 'boundary_hook' and c.get('boundary_control_role') == 'set_hook']
    slots = {(v['scenario_id'], v['variant_id']) for v in planned_variants(context)}
    available = {(s['boundary_id'], s['scenario_id'], s['variant_id']) for s in context['mock_interfaces']
                 if s.get('name') and s.get('declaration')}
    for control in controls:
        boundary = control.get('boundary_id', '')
        if not slots:
            errors.append('Hook allocation has no active scenario slots: ' + boundary)
        for scenario, variant in sorted(slots):
            if (boundary, scenario, variant) not in available:
                errors.append('Hook has no typed allocated mock for ' + ':'.join((boundary, scenario, variant)))
    return errors


def callback_interface_specs(context):
    """Use source-declared member types; do not infer an unproven alias chain."""
    registry = context.get("scenario_registry", {})
    facts = {f["fact_id"]: f for f in registry.get("source_facts", [])}
    specs = []
    for boundary in registry.get("boundary_candidates", []):
        if boundary.get("source_fact_kind") != "MEMBER_CALL":
            continue
        expression = boundary.get("expression", "")
        match = re.fullmatch(r"([A-Za-z_]\w*)((?:\s*->\s*[A-Za-z_]\w*)+)", expression)
        if not match:
            continue
        types = {}
        for fid in boundary.get("fact_ids", []):
            types.update(facts.get(fid, {}).get("metadata", {}).get("function_parameter_types", {}))
        root_type = types.get(match.group(1), "")
        if not root_type.endswith("*") or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_\s*]*", root_type):
            continue
        specs.append({"boundary_id": boundary["candidate_id"], "canonical_fake_name": "fake_" + re.sub(r"\W", "_", expression),
                      "original": expression, "return_type": "source_callback_type", "params": "source_callback_type",
                      "signature_type": f"typeof(*((({root_type})0){match.group(2)}))", "control_kind": "callback"})
    return specs


def allocate_mock_interfaces(context, specifications):
    plan_errors = variant_plan_errors(context)
    if plan_errors:
        raise ValueError('; '.join(plan_errors))
    existing = context.setdefault("mock_interfaces", [])
    identities = {(s["boundary_id"], s["scenario_id"], s["variant_id"]) for s in existing}
    slots = [(v, v["variant_id"]) for v in planned_variants(context)]
    variants = context.setdefault("variant_bindings", [])
    for contract, variant in slots:
        binding = {"scenario_id": contract["scenario_id"], "variant_id": variant}
        if binding not in variants:
            variants.append(binding)
    for contract, variant in slots:
        boundaries = set(contract.get("dependent_boundaries", []))
        boundaries.update(c.get("boundary_id") for c in contract.get("hardware_environment_constraints", []))
        for spec in specifications:
            for boundary in spec.get("boundary_ids", [spec["boundary_id"]]):
                identity = (boundary, contract["scenario_id"], variant)
                # Specifications already come only from the target's file-local
                # dependency closure. Availability is not a claim of execution:
                # a conditional/helper call may need control on this variant even
                # when lexical analysis cannot establish a mandatory witness.
                if identity in identities:
                    continue
                suffix = hashlib.sha256(":".join(identity).encode()).hexdigest()[:12]
                name = f"{spec['canonical_fake_name']}_{suffix}"
                declaration = (f"static {spec['signature_type']} {name};" if spec.get("signature_type") else
                               f"static {spec['return_type']} {name}({spec['params']});")
                existing.append({"boundary_id": boundary, "scenario_id": identity[1], "variant_id": variant,
                                 "name": name, "return_type": spec["return_type"], "parameters": spec["params"],
                                 "declaration": declaration, "signature_type": spec.get("signature_type", ""),
                                 "control_kind": spec.get("control_kind", "hook"),
                                 "required_by_contract": boundary in boundaries,
                                 "original": spec["original"],
                                 "setter": f"raca_boundary_{boundary}_set_hook" if spec.get("control_kind", "hook") == "hook" else ""})
                identities.add(identity)
    errors = allocation_errors(context)
    if errors:
        raise ValueError('Incomplete framework mock allocation: ' + '; '.join(errors))
    return existing


def fixed_mock_block(context, used_names=None):
    declarations = [s["declaration"] for s in context.get("mock_interfaces", [])
                    if used_names is None or s["name"] in used_names]
    return "\n".join([BEGIN] + declarations + [END]) if declarations else ""


def install_mock_declarations(code, context, used_names=None):
    if used_names is None:
        used_names = set(inspect_test_source(code).function_definitions)
    block = fixed_mock_block(context, used_names)
    if BEGIN in code:
        start, end = code.index(BEGIN), code.index(END) + len(END)
        return code[:start] + block + code[end:]
    if not block:
        return code
    marker = "/* ===== Flexible Helpers BEGIN ===== */"
    if marker not in code:
        marker = "/* ===== Scenario Tests BEGIN ===== */"
    if marker in code:
        return code.replace(marker, block + "\n\n" + marker, 1)
    return code + "\n" + block + "\n"


def reconcile_mock_declarations(before, candidate, context):
    """Only the framework may append freshly allocated declarations to fixed code."""
    def block(code):
        if BEGIN not in code or END not in code:
            return ""
        return code[code.index(BEGIN):code.index(END) + len(END)]
    old, supplied = block(before), block(candidate)
    expected = fixed_mock_block(context, set(inspect_test_source(candidate).function_definitions))
    if supplied not in {old, expected}:
        return candidate, ["Protected mock declaration block was modified by the candidate."]
    return install_mock_declarations(candidate, context), []


def _signature_without_parameter_names(signature):
    data = signature.encode()
    edits = []
    for node in _walk(C_PARSER.parse(data).root_node):
        if node.type != "parameter_declaration":
            continue
        declarator = node.child_by_field_name("declarator")
        while declarator is not None and declarator.type != "identifier":
            child = declarator.child_by_field_name("declarator")
            if child is None and declarator.type == "parenthesized_declarator" and declarator.named_children:
                child = declarator.named_children[0]
            declarator = child
        if declarator is not None:
            edits.append((declarator.start_byte, declarator.end_byte))
    for start, end in sorted(set(edits), reverse=True):
        data = data[:start] + data[end:]
    return re.sub(r"\s+", "", data.decode())


def mock_interface_errors(code, context):
    if "mock_interfaces" not in context:
        return []
    errors = []
    specs = context["mock_interfaces"]
    by_boundary = {}
    for spec in specs:
        by_boundary.setdefault(spec["boundary_id"], set()).add(spec["name"])
    for control in (context.get("target") or {}).get("export_interface_details", []):
        if control.get("boundary_control_role") == "set_hook":
            by_boundary.setdefault(control.get("boundary_id", ""), set())
    from verification.scenario_static_verifier import mock_bindings_from_test
    for binding in mock_bindings_from_test(code, context):
        names = by_boundary.get(binding["boundary_id"])
        if names is not None and binding["replacement"] not in names:
            errors.append(f"Fixed mock interface: {binding['boundary_id']} uses unallocated mock {binding['replacement']}.")
    clean = re.sub(r"/\*.*?\*/|//[^\n]*", " ", code, flags=re.S)
    for boundary, names in by_boundary.items():
        setter = f"raca_boundary_{boundary}_set_hook"
        for call in re.finditer(r"\b" + setter + r"\s*\(\s*(\w+)\s*\)", clean):
            if call.group(1) not in names | {"NULL", "0"}:
                errors.append(f"Fixed mock interface: {setter} installs an unallocated mock {call.group(1)}.")
    if C_PARSER is None:
        errors.append("Fixed mock interface validation requires the C syntax parser.")
        return errors
    data = code.encode()
    tree = C_PARSER.parse(data)
    definitions = {}
    for node in _walk(tree.root_node):
        if node.type != "function_definition":
            continue
        body = node.child_by_field_name("body")
        signature = data[node.start_byte:body.start_byte].decode().strip() + ";"
        from validation.test_inspector import _function_name_from_definition
        definitions[_function_name_from_definition(node, data)] = signature
    for spec in specs:
        actual = definitions.get(spec["name"])
        if actual and not spec.get("signature_type") and _signature_without_parameter_names(actual) != _signature_without_parameter_names(spec["declaration"]):
            errors.append(f"Fixed mock interface changed: {spec['name']}; expected {spec['declaration']}")
    return errors
