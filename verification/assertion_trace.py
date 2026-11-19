"""Execution receipts for scenario assertions in generated tests, never in drivers."""
import re

from validation.test_inspector import C_PARSER, _node_text, _walk, inspect_test_source
from verification.assertion_quality import binding_is_effective_for_check, effective_check_ids
from scenario.condition_variants import contract_for_variant


def instrument_assertion_receipts(code, context):
    if C_PARSER is None:
        raise RuntimeError("Assertion receipts require the C syntax parser")
    registry = context.get("scenario_registry", {})
    contracts = {c["scenario_id"]: c for c in registry.get("scenario_contracts", [])}
    from verification.scenario_static_verifier import observation_bindings
    data = code.encode()
    tree = C_PARSER.parse(data)
    source_info = inspect_test_source(code)
    edits = []
    for fn in tree.root_node.named_children:
        if fn.type != "function_definition":
            continue
        text = _node_text(data, fn)
        info = inspect_test_source(text)
        if not info.test_functions:
            continue
        test = info.test_functions[0]
        if len(test.scenario_ids) != 1 or test.scenario_ids[0] not in contracts:
            continue
        contract = contract_for_variant(context, test.scenario_ids[0], test.variant_id)
        if contract is None:
            continue
        checks = {c["check_id"]: c for c in contract.get("scenario_checks", [])}
        bindings = observation_bindings(test, code, context, source_info)
        statements = {n.start_byte - fn.start_byte: n for n in _walk(fn) if n.type == "call_expression"}
        for binding in bindings:
            wrapper = contract.get("export_function", "")
            ids = effective_check_ids([binding], checks, wrapper)
            if binding_is_effective_for_check(binding, {}, wrapper):
                ids.add("__semantic__")
            if not ids:
                continue
            call = statements.get(binding.start_byte)
            if call is None or call.parent.type != "expression_statement":
                continue
            statement = call.parent
            parent = statement.parent
            # Framework wrappers are idempotent; candidate receipts outside one
            # are not used as a reason to skip instrumentation.
            parent_text = _node_text(data, parent)
            receipt = f"/* RACA ASSERT RECEIPT {test.name} */"
            if parent.type == "compound_statement" and parent_text.lstrip().startswith("{\n" + receipt):
                continue
            logs = [f'kunit_info(test, "RACA_ASSERT:{test.name}:{item}:HIT");' for item in sorted(ids)]
            replacement = "{\n" + receipt + "\n" + _node_text(data, statement) + "\n" + "\n".join(logs) + "\n}"
            # Move the immediately preceding check marker inside the block, so
            # the syntax-aware binding remains adjacent to its assertion.
            prefix = data[:statement.start_byte].decode()
            match = re.search(r"((?:/\*\s*RACA_CHECK:[^*]*\*/\s*)+)$", prefix)
            start = statement.start_byte
            if match:
                start = len(prefix[:match.start()].encode())
                replacement = replacement.replace(receipt + "\n", receipt + "\n" + match.group(1), 1)
            edits.append((start, statement.end_byte, replacement.encode()))
    for start, end, replacement in sorted(edits, reverse=True):
        data = data[:start] + replacement + data[end:]
    return data.decode()
