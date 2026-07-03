import hashlib
import re
from typing import Dict, Iterable, List, Optional, Set, Tuple

try:
    from tree_sitter_languages import get_parser
except ImportError:
    get_parser = None
try:
    from pycparser import c_ast, c_generator, c_parser
except ImportError:
    c_ast = None
    c_generator = None
    c_parser = None

from scenario.fact_model import SourceFact


C_PARSER = get_parser("c") if get_parser is not None else None


def _node_text(code: bytes, node) -> str:
    return code[node.start_byte:node.end_byte].decode("utf-8", errors="ignore")


def _walk(node):
    stack = [node]
    while stack:
        current = stack.pop()
        yield current
        stack.extend(reversed(current.children))


def _contains_node_type(node, node_type: str) -> bool:
    if node is None:
        return False
    return any(item.type == node_type for item in _walk(node))


def _stable_id(prefix: str, function: str, line: int, node_type: str, text: str) -> str:
    digest = hashlib.sha1(f"{function}:{line}:{node_type}:{text}".encode("utf-8")).hexdigest()[:10]
    return f"{prefix}_{digest}"


def _add_relation(source: SourceFact, target: SourceFact, relation: str, via: str = "", extra: Optional[Dict] = None) -> None:
    if source.fact_id == target.fact_id:
        return
    if target.fact_id not in source.related_fact_ids:
        source.related_fact_ids.append(target.fact_id)
    edges = source.metadata.setdefault("relation_edges", [])
    edge = {"relation": relation, "target_fact_id": target.fact_id}
    if via:
        edge["via"] = via
    if extra:
        edge.update(extra)
    if edge not in edges:
        edges.append(edge)


def _symbol_candidates(value: object) -> Set[str]:
    if not value:
        return set()
    if isinstance(value, list):
        result: Set[str] = set()
        for item in value:
            result.update(_symbol_candidates(item))
        return result
    text = str(value)
    parts = [part for part in text.replace("->", ".").split(".") if part]
    return {parts[0], parts[-1]} if parts else set()


def _stable_path_text(path: List[str]) -> str:
    return ".".join(item for item in path if item)


def _path_symbols(paths: Iterable[List[str]]) -> Set[str]:
    symbols: Set[str] = set()
    for path in paths or []:
        if not path:
            continue
        symbols.add(path[0])
        symbols.add(path[-1])
        symbols.add(_stable_path_text(path))
    return {item for item in symbols if item}


def _fact_definition_symbols(fact: SourceFact) -> Set[str]:
    metadata = fact.metadata or {}
    if fact.kind in {"CALL", "MEMBER_CALL"}:
        return _symbol_candidates(metadata.get("result_assignee")) | _symbol_candidates(metadata.get("output_arguments"))
    if fact.kind in {"ASSIGNMENT", "FIELD_WRITE"}:
        return _symbol_candidates(metadata.get("left")) | _symbol_candidates(metadata.get("left_path"))
    return set()


def _fact_use_symbols(fact: SourceFact) -> Set[str]:
    metadata = fact.metadata or {}
    if fact.kind in {"ASSIGNMENT", "FIELD_WRITE"}:
        return set(metadata.get("right_symbols", []) or [])
    if fact.kind == "GUARD":
        return set(metadata.get("condition_symbols", []) or fact.symbols)
    if fact.kind in {"CALL", "MEMBER_CALL"}:
        return set(fact.symbols) - _fact_definition_symbols(fact)
    return set(fact.symbols) - _fact_definition_symbols(fact)


def _definition_updates_for_fact(
    fact: SourceFact,
    inherited_origins: Iterable[Tuple[str, str]],
) -> Dict[str, Set[Tuple[str, str]]]:
    metadata = fact.metadata or {}
    updates: Dict[str, Set[Tuple[str, str]]] = {}
    if fact.kind in {"CALL", "MEMBER_CALL"}:
        for symbol in _symbol_candidates(metadata.get("result_assignee")):
            updates.setdefault(symbol, set()).add((fact.fact_id, "call_result"))
        for symbol in _symbol_candidates(metadata.get("output_arguments")):
            updates.setdefault(symbol, set()).add((fact.fact_id, "call_output"))
    elif fact.kind in {"ASSIGNMENT", "FIELD_WRITE"}:
        origins = set(inherited_origins)
        origins.add((fact.fact_id, "assignment"))
        for symbol in _symbol_candidates(metadata.get("left")) | _symbol_candidates(metadata.get("left_path")):
            updates.setdefault(symbol, set()).update(origins)
    return updates


def _argument_aliases(call_fact: SourceFact, callee_entry_fact: SourceFact) -> Dict[str, Dict]:
    call_metadata = call_fact.metadata or {}
    callee_params = (callee_entry_fact.metadata or {}).get("function_parameters", []) or []
    argument_map = call_metadata.get("argument_path_map", []) or []
    aliases: Dict[str, Dict] = {}
    for index, param in enumerate(callee_params):
        if index >= len(argument_map):
            continue
        argument = argument_map[index] or {}
        aliases[param] = {
            "text": argument.get("text", ""),
            "path": argument.get("path", []) or [],
        }
    return aliases


def _link_related_facts(facts: List[SourceFact]) -> List[SourceFact]:
    by_function: Dict[str, List[SourceFact]] = {}
    for fact in facts:
        by_function.setdefault(fact.function, []).append(fact)

    for function_facts in by_function.values():
        ordered = sorted(function_facts, key=lambda item: (item.start_line, item.end_line, item.fact_id))

        for guard in [item for item in ordered if item.kind == "GUARD"]:
            first_return_line = int((guard.metadata or {}).get("first_return_line") or 0)
            for candidate in ordered:
                if candidate.kind == "RETURN" and candidate.start_line == first_return_line:
                    _add_relation(guard, candidate, "guard_first_return")
                    _add_relation(candidate, guard, "returned_under_guard")
                    break
            for candidate in ordered:
                if candidate is guard:
                    continue
                if guard.start_line <= candidate.start_line and candidate.end_line <= guard.end_line:
                    _add_relation(guard, candidate, "control_contains")
                    _add_relation(candidate, guard, "control_dependent_on")

        fact_by_id = {fact.fact_id: fact for fact in ordered}
        active_defs: Dict[str, Set[Tuple[str, str]]] = {}
        for fact in ordered:
            use_symbols = _fact_use_symbols(fact)
            inherited_origins: Set[Tuple[str, str]] = set()
            for symbol in sorted(use_symbols):
                for origin_fact_id, relation_prefix in active_defs.get(symbol, set()):
                    origin = fact_by_id.get(origin_fact_id)
                    if origin is None:
                        continue
                    inherited_origins.add((origin_fact_id, relation_prefix))
                    relation = f"{relation_prefix}_influences_{fact.kind.lower()}"
                    _add_relation(origin, fact, relation, via=symbol)
                    _add_relation(fact, origin, f"uses_{relation_prefix}", via=symbol)

            for symbol, origins in _definition_updates_for_fact(fact, inherited_origins).items():
                active_defs[symbol] = set(origins)

    function_ordered_facts = {
        function: sorted(items, key=lambda item: (item.start_line, item.end_line, item.fact_id))
        for function, items in by_function.items()
    }
    for fact in facts:
        if fact.kind != "CALL":
            continue
        callee = (fact.metadata or {}).get("callee_name", "")
        callee_facts = function_ordered_facts.get(callee, [])
        if not callee_facts:
            continue
        aliases = _argument_aliases(fact, callee_facts[0])
        for callee_fact in callee_facts:
            extra = {"argument_aliases": aliases} if aliases else {}
            relation = "internal_call_enters" if callee_fact is callee_facts[0] else "internal_call_reaches"
            reverse = "entered_from_internal_call" if callee_fact is callee_facts[0] else "reached_from_internal_call"
            _add_relation(fact, callee_fact, relation, extra=extra)
            _add_relation(callee_fact, fact, reverse, extra=extra)

    for fact in facts:
        fact.related_fact_ids = sorted(set(fact.related_fact_ids))
        if "relation_edges" in fact.metadata:
            fact.metadata["relation_edges"] = sorted(
                fact.metadata["relation_edges"],
                key=lambda item: (item.get("target_fact_id", ""), item.get("relation", ""), item.get("via", "")),
            )
    return facts


def _identifier_symbols(node, code: bytes) -> List[str]:
    symbols: Set[str] = set()
    for item in _walk(node):
        if item.type in {"identifier", "field_identifier", "type_identifier"}:
            symbols.add(_node_text(code, item))
    return sorted(symbols)


def _declaration_name(node, code: bytes) -> str:
    if node is None:
        return ""
    identifiers: List[str] = []
    for item in _walk(node):
        if item.type == "identifier":
            identifiers.append(_node_text(code, item))
    return identifiers[-1] if identifiers else ""


def _function_parameters(tree, code: bytes) -> List[str]:
    for node in _walk(tree.root_node):
        if node.type != "function_definition":
            continue
        params: List[str] = []
        for item in _walk(node):
            if item.type != "parameter_declaration":
                continue
            name = _declaration_name(item, code)
            if name and name not in params:
                params.append(name)
        return params
    return []


def _function_parameter_type_map(tree, code: bytes) -> Dict[str, str]:
    for node in _walk(tree.root_node):
        if node.type != "function_definition":
            continue
        types: Dict[str, str] = {}
        for item in _walk(node):
            if item.type != "parameter_declaration":
                continue
            name = _declaration_name(item, code)
            if not name:
                continue
            text = _node_text(code, item).strip()
            type_text = re.sub(r"\b" + re.escape(name) + r"\b\s*$", "", text).strip()
            types[name] = type_text or text
        return types
    return {}


def _function_map(parse_result) -> Dict[str, object]:
    return {function.name: function for function in getattr(parse_result, "functions", []) or []}


def build_internal_call_closure(parse_result, function) -> List[str]:
    functions = _function_map(parse_result)
    graph = getattr(parse_result, "call_graph", {}) or {}
    closure: List[str] = []
    seen: Set[str] = set()
    stack = [function.name]
    while stack:
        name = stack.pop()
        if name in seen or name not in functions:
            continue
        seen.add(name)
        closure.append(name)
        for callee in reversed(graph.get(name, []) or []):
            if callee in functions and callee not in seen:
                stack.append(callee)
    return closure


def _field_path(node, code: bytes) -> List[str]:
    if node.type in {"identifier", "field_identifier"}:
        return [_node_text(code, node)]
    if node.type in {"parenthesized_expression", "pointer_expression"}:
        for child in node.children:
            if child.is_named:
                return _field_path(child, code)
        return []
    if node.type == "field_expression":
        argument = node.child_by_field_name("argument")
        field = node.child_by_field_name("field")
        path = _field_path(argument, code) if argument is not None else []
        if field is not None:
            path.append(_node_text(code, field))
        return path
    return []


def _callee_name(function_node, code: bytes) -> str:
    path = _field_path(function_node, code)
    if path:
        return path[-1]
    if function_node.type == "identifier":
        return _node_text(code, function_node)
    for child in function_node.children:
        if child.is_named:
            name = _callee_name(child, code)
            if name:
                return name
    return ""


def _argument_texts(argument_node, code: bytes) -> List[str]:
    if argument_node is None:
        return []
    args: List[str] = []
    for child in argument_node.children:
        if child.is_named:
            args.append(_node_text(code, child).strip())
    return args


def _output_argument_paths(argument_node, code: bytes) -> List[str]:
    if argument_node is None:
        return []
    outputs: List[str] = []
    for child in argument_node.children:
        if not child.is_named:
            continue
        if child.type == "pointer_expression":
            for inner in child.children:
                if inner.is_named:
                    if inner.type != "identifier":
                        break
                    path = _field_path(inner, code)
                    if path:
                        outputs.append(".".join(path))
                    else:
                        outputs.append(_node_text(code, inner).strip())
                    break
    return sorted(set(outputs))


def _argument_paths(argument_node, code: bytes) -> List[List[str]]:
    if argument_node is None:
        return []
    paths: List[List[str]] = []
    for child in argument_node.children:
        if not child.is_named:
            continue
        direct_path = _field_path(child, code)
        if direct_path and direct_path not in paths:
            paths.append(direct_path)
        for node in _walk(child):
            if node.type != "field_expression":
                continue
            path = _field_path(node, code)
            if path and path not in paths:
                paths.append(path)
    return paths


def _argument_path_map(argument_node, code: bytes) -> List[Dict]:
    if argument_node is None:
        return []
    items: List[Dict] = []
    for child in argument_node.children:
        if not child.is_named:
            continue
        path = _field_path(child, code)
        if not path:
            nested_paths = _argument_paths(child, code)
            path = nested_paths[0] if nested_paths else []
        items.append({"text": _node_text(code, child).strip(), "path": path})
    return items


def _argument_type_map(argument_path_map: List[Dict], parameter_types: Dict[str, str]) -> List[Dict]:
    items: List[Dict] = []
    for argument in argument_path_map or []:
        path = argument.get("path", []) or []
        if not path:
            continue
        root = path[0]
        type_text = parameter_types.get(root, "")
        if not type_text:
            continue
        items.append(
            {
                "argument": argument.get("text", ""),
                "path": path,
                "root": root,
                "type": type_text,
            }
        )
    return items


def _field_paths_in_node(node, code: bytes) -> List[List[str]]:
    if node is None:
        return []
    paths: List[List[str]] = []
    for item in _walk(node):
        if item.type == "field_expression":
            path = _field_path(item, code)
            if path and path not in paths:
                paths.append(path)
        elif item.type == "identifier":
            path = [_node_text(code, item)]
            if path not in paths:
                paths.append(path)
    return paths


def _is_nullish(node, code: bytes) -> bool:
    if node is None:
        return False
    text = _node_text(code, node).strip()
    return text in {"NULL", "0"}


def _node_operator_text(node, code: bytes) -> str:
    if node is None:
        return ""
    for child in node.children:
        if child.is_named:
            continue
        text = _node_text(code, child).strip()
        if text:
            return text
    return ""


def _unwrap_expression_node(node):
    current = node
    while current is not None and current.type == "parenthesized_expression":
        next_node = None
        for child in current.children:
            if child.is_named:
                next_node = child
                break
        if next_node is None:
            break
        current = next_node
    return current


def _condition_case_requirement(node, code: bytes, desired: bool, full_condition: str) -> Dict:
    expression = _node_text(code, node).strip() if node is not None else ""
    return {
        "expression": expression,
        "relation": "must_be_true" if desired else "must_be_false",
        "required_for": "branch" if desired else "fallthrough",
        "evidence": full_condition,
        "object_paths": _field_paths_in_node(node, code),
    }


def _merge_condition_case_parts(parts: List[Dict], outcome: str, full_condition: str) -> Dict:
    requirements: List[Dict] = []
    short_circuit = False
    decomposed = False
    for part in parts:
        requirements.extend(part.get("requirements", []) or [])
        short_circuit = short_circuit or bool(part.get("short_circuit"))
        decomposed = decomposed or bool(part.get("decomposed"))
    key_text = "|".join(
        f"{item.get('expression', '')}:{item.get('relation', '')}" for item in requirements
    )
    digest = hashlib.sha1(f"{full_condition}:{outcome}:{key_text}".encode("utf-8")).hexdigest()[:10]
    readable = " and ".join(
        f"`{item.get('expression', '')}` {item.get('relation', '').replace('_', ' ')}"
        for item in requirements
        if item.get("expression")
    )
    return {
        "case_id": f"{outcome}_{digest}",
        "condition": full_condition,
        "outcome": outcome,
        "requirements": requirements,
        "short_circuit": short_circuit,
        "decomposed": decomposed,
        "description": readable or f"drive condition `{full_condition}` to {outcome}",
    }


def _truth_condition_cases(node, code: bytes, desired: bool, full_condition: str) -> List[Dict]:
    node = _unwrap_expression_node(node)
    if node is None:
        return []
    if node.type == "unary_expression" and _node_operator_text(node, code) == "!":
        operand = None
        for child in node.children:
            if child.is_named:
                operand = child
                break
        return _truth_condition_cases(operand, code, not desired, full_condition)

    if node.type == "binary_expression":
        operator = _node_operator_text(node, code)
        left = node.child_by_field_name("left")
        right = node.child_by_field_name("right")
        if operator == "&&":
            if desired:
                cases: List[Dict] = []
                for left_case in _truth_condition_cases(left, code, True, full_condition):
                    for right_case in _truth_condition_cases(right, code, True, full_condition):
                        merged = _merge_condition_case_parts([left_case, right_case], "branch", full_condition)
                        merged["decomposed"] = True
                        cases.append(merged)
                return cases
            cases = []
            for left_case in _truth_condition_cases(left, code, False, full_condition):
                merged = _merge_condition_case_parts([left_case], "fallthrough", full_condition)
                merged["short_circuit"] = True
                merged["decomposed"] = True
                cases.append(merged)
            for left_case in _truth_condition_cases(left, code, True, full_condition):
                for right_case in _truth_condition_cases(right, code, False, full_condition):
                    merged = _merge_condition_case_parts([left_case, right_case], "fallthrough", full_condition)
                    merged["decomposed"] = True
                    cases.append(merged)
            return cases
        if operator == "||":
            if desired:
                cases = []
                for left_case in _truth_condition_cases(left, code, True, full_condition):
                    merged = _merge_condition_case_parts([left_case], "branch", full_condition)
                    merged["short_circuit"] = True
                    merged["decomposed"] = True
                    cases.append(merged)
                for left_case in _truth_condition_cases(left, code, False, full_condition):
                    for right_case in _truth_condition_cases(right, code, True, full_condition):
                        merged = _merge_condition_case_parts([left_case, right_case], "branch", full_condition)
                        merged["decomposed"] = True
                        cases.append(merged)
                return cases
            cases = []
            for left_case in _truth_condition_cases(left, code, False, full_condition):
                for right_case in _truth_condition_cases(right, code, False, full_condition):
                    merged = _merge_condition_case_parts([left_case, right_case], "fallthrough", full_condition)
                    merged["decomposed"] = True
                    cases.append(merged)
            return cases

    outcome = "branch" if desired else "fallthrough"
    return [
        {
            "case_id": hashlib.sha1(
                f"{full_condition}:{outcome}:{_node_text(code, node).strip()}:{desired}".encode("utf-8")
            ).hexdigest()[:10],
            "condition": full_condition,
            "outcome": outcome,
            "requirements": [_condition_case_requirement(node, code, desired, full_condition)],
            "short_circuit": False,
            "decomposed": False,
            "description": (
                f"`{_node_text(code, node).strip()}` "
                f"{'must be true' if desired else 'must be false'}"
            ),
        }
    ]


def _dedupe_condition_cases(cases: Iterable[Dict]) -> List[Dict]:
    deduped: List[Dict] = []
    seen: Set[Tuple] = set()
    for case in cases:
        requirements = tuple(
            sorted(
                (
                    item.get("expression", ""),
                    item.get("relation", ""),
                )
                for item in case.get("requirements", []) or []
            )
        )
        key = (case.get("outcome", ""), requirements)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(case)
    return deduped


def _condition_case_requirements(condition_node, code: bytes) -> List[Dict]:
    if condition_node is None:
        return []
    condition_node = _unwrap_expression_node(condition_node)
    condition_text = _node_text(code, condition_node).strip()
    return _dedupe_condition_cases(
        _truth_condition_cases(condition_node, code, True, condition_text)
        + _truth_condition_cases(condition_node, code, False, condition_text)
    )


def _condition_requirement(path: List[str], relation: str, required_for: str, evidence: str) -> Dict:
    return {
        "object_path": path,
        "relation": relation,
        "required_for": required_for,
        "evidence": evidence,
    }


def _condition_path_requirements(condition_node, code: bytes) -> Tuple[List[Dict], List[Dict]]:
    """Return branch and fallthrough requirements extracted from a guard condition."""
    if condition_node is None:
        return [], []
    condition_text = _node_text(code, condition_node).strip()
    branch: List[Dict] = []
    fallthrough: List[Dict] = []

    if condition_node.type == "unary_expression" and _node_operator_text(condition_node, code) == "!":
        operand = None
        for child in condition_node.children:
            if child.is_named:
                operand = child
                break
        for path in _field_paths_in_node(operand, code):
            branch.append(_condition_requirement(path, "is_null_or_false", "branch", condition_text))
            fallthrough.append(_condition_requirement(path, "is_non_null_or_true", "fallthrough", condition_text))
        return branch, fallthrough

    if condition_node.type == "binary_expression":
        left = condition_node.child_by_field_name("left")
        right = condition_node.child_by_field_name("right")
        operator = _node_operator_text(condition_node, code)
        if operator in {"==", "!="} and (_is_nullish(left, code) or _is_nullish(right, code)):
            target = right if _is_nullish(left, code) else left
            for path in _field_paths_in_node(target, code):
                if operator == "==":
                    branch.append(_condition_requirement(path, "is_null_or_zero", "branch", condition_text))
                    fallthrough.append(_condition_requirement(path, "is_non_null_or_nonzero", "fallthrough", condition_text))
                else:
                    branch.append(_condition_requirement(path, "is_non_null_or_nonzero", "branch", condition_text))
                    fallthrough.append(_condition_requirement(path, "is_null_or_zero", "fallthrough", condition_text))
            return branch, fallthrough
        for path in _field_paths_in_node(condition_node, code):
            branch.append(_condition_requirement(path, "makes_condition_true", "branch", condition_text))
            fallthrough.append(_condition_requirement(path, "makes_condition_false", "fallthrough", condition_text))
        return branch, fallthrough

    for path in _field_paths_in_node(condition_node, code):
        branch.append(_condition_requirement(path, "is_true_or_nonzero", "branch", condition_text))
        fallthrough.append(_condition_requirement(path, "is_false_or_zero", "fallthrough", condition_text))
    return branch, fallthrough


def _nearest_statement(node):
    current = node
    while current is not None:
        if current.type in {"expression_statement", "declaration", "return_statement", "if_statement"}:
            return current
        current = current.parent
    return node


def _assignment_left_for_call(call_node, code: bytes) -> str:
    current = call_node.parent
    while current is not None:
        if current.type == "assignment_expression":
            left = current.child_by_field_name("left")
            right = current.child_by_field_name("right")
            if left is not None and right is not None and call_node.start_byte >= right.start_byte and call_node.end_byte <= right.end_byte:
                return _node_text(code, left).strip()
            return ""
        if current.type == "init_declarator":
            value = current.child_by_field_name("value")
            declarator = current.child_by_field_name("declarator")
            if (
                value is not None
                and declarator is not None
                and call_node.start_byte >= value.start_byte
                and call_node.end_byte <= value.end_byte
            ):
                return _node_text(code, declarator).strip()
            return ""
        if current.type in {"expression_statement", "declaration", "return_statement"}:
            return ""
        current = current.parent
    return ""


def _return_expression(return_node, code: bytes) -> str:
    for child in return_node.children:
        if child.is_named:
            return _node_text(code, child).strip()
    return ""


def _condition_expression(if_node, code: bytes) -> str:
    condition = if_node.child_by_field_name("condition")
    return _node_text(code, condition).strip() if condition is not None else ""


def _loop_expression(loop_node, code: bytes, field_name: str) -> str:
    node = loop_node.child_by_field_name(field_name)
    return _node_text(code, node).strip() if node is not None else ""


def _loop_kind(loop_node) -> str:
    if loop_node is None:
        return ""
    if loop_node.type == "for_statement":
        return "for"
    if loop_node.type == "while_statement":
        return "while"
    if loop_node.type == "do_statement":
        return "do_while"
    return loop_node.type


def _loop_context(loop_node, code: bytes, function_base_line: int) -> Dict:
    if loop_node is None:
        return {}
    condition_node = loop_node.child_by_field_name("condition")
    initializer_node = loop_node.child_by_field_name("initializer")
    update_node = loop_node.child_by_field_name("update")
    condition = _loop_expression(loop_node, code, "condition")
    initializer = _loop_expression(loop_node, code, "initializer")
    update = _loop_expression(loop_node, code, "update")
    return {
        "kind": _loop_kind(loop_node),
        "start_line": function_base_line + loop_node.start_point[0] + 1,
        "condition": condition,
        "initializer": initializer,
        "update": update,
        "condition_symbols": _identifier_symbols(condition_node, code) if condition_node is not None else [],
        "initializer_symbols": _identifier_symbols(initializer_node, code) if initializer_node is not None else [],
        "update_symbols": _identifier_symbols(update_node, code) if update_node is not None else [],
        "symbols": _identifier_symbols(loop_node, code),
    }


def _enclosing_loop_context(node, code: bytes, function_base_line: int) -> Dict:
    current = node.parent if node is not None else None
    while current is not None:
        if current.type in {"for_statement", "while_statement", "do_statement"}:
            return _loop_context(current, code, function_base_line)
        if current.type == "function_definition":
            break
        current = current.parent
    return {}


def _loop_cases(loop_node, code: bytes) -> List[Dict]:
    if loop_node is None:
        return []
    kind = _loop_kind(loop_node)
    condition = _loop_expression(loop_node, code, "condition")
    cases = []
    if kind in {"for", "while"}:
        cases.append(
            {
                "case_id": "zero_iterations",
                "role": "loop_boundary",
                "description": (
                    "make the loop condition false before the first iteration"
                    + (f": `{condition}`" if condition else "")
                ),
            }
        )
    cases.append(
        {
            "case_id": "single_iteration",
            "role": "loop_boundary",
            "description": (
                "execute the loop body exactly once, then make the loop exit condition hold"
                + (f": `{condition}`" if condition else "")
            ),
        }
    )
    cases.append(
        {
            "case_id": "multiple_iterations",
            "role": "loop_boundary",
            "description": (
                "execute the loop body for more than one iteration"
                + (f" while evaluating `{condition}`" if condition else "")
            ),
        }
    )
    return cases


def _first_return_in_consequence(if_node, code: bytes) -> Tuple[str, int, str]:
    consequence = if_node.child_by_field_name("consequence")
    if consequence is None:
        return "", 0, ""
    for node in _walk(consequence):
        if node.type == "return_statement":
            return _return_expression(node, code), node.start_point[0] + 1, _node_text(code, node).strip()
    return "", 0, ""


def _assignment_parts(node, code: bytes) -> Tuple[str, str]:
    left = node.child_by_field_name("left")
    right = node.child_by_field_name("right")
    if left is None or right is None:
        return "", ""
    return _node_text(code, left).strip(), _node_text(code, right).strip()


def _fact(
    prefix: str,
    kind: str,
    file_path: str,
    function_name: str,
    function_base_line: int,
    code: bytes,
    node,
    metadata: Optional[Dict] = None,
    related_fact_ids: Optional[List[str]] = None,
) -> SourceFact:
    text = _node_text(code, node).strip()
    start_line = function_base_line + node.start_point[0] + 1
    end_line = function_base_line + node.end_point[0] + 1
    return SourceFact(
        fact_id=_stable_id(prefix, function_name, start_line, node.type, text),
        kind=kind,
        file=file_path,
        function=function_name,
        start_line=start_line,
        end_line=end_line,
        code=text,
        symbols=_identifier_symbols(node, code),
        related_fact_ids=related_fact_ids or [],
        metadata=metadata or {},
    )


def _branch_edge_fact(
    file_path: str,
    function_name: str,
    function_base_line: int,
    code: bytes,
    guard_fact: SourceFact,
    edge: str,
    requirements: List[Dict],
) -> SourceFact:
    condition = (guard_fact.metadata or {}).get("condition", "")
    text = f"{condition} [{edge}]"
    line = guard_fact.start_line
    return SourceFact(
        fact_id=_stable_id("F_EDGE", function_name, line, edge, text),
        kind="BRANCH_EDGE",
        file=file_path,
        function=function_name,
        start_line=guard_fact.start_line,
        end_line=guard_fact.end_line,
        code=text,
        symbols=list(guard_fact.symbols),
        related_fact_ids=[guard_fact.fact_id],
        metadata={
            "guard_fact_id": guard_fact.fact_id,
            "condition": condition,
            "edge": edge,
            "requirements": requirements,
            "function_parameters": (guard_fact.metadata or {}).get("function_parameters", []),
            "first_return_expression": (guard_fact.metadata or {}).get("first_return_expression", ""),
            "first_return_line": (guard_fact.metadata or {}).get("first_return_line", 0),
        },
    )


def _extract_facts_from_function(parse_result, function) -> List[SourceFact]:
    if C_PARSER is None:
        return _extract_facts_from_function_fallback(parse_result, function)
    file_path = getattr(parse_result, "path", "")
    function_code = getattr(function, "code", "") or ""
    code = function_code.encode("utf-8", errors="ignore")
    tree = C_PARSER.parse(code)
    function_parameters = _function_parameters(tree, code)
    function_parameter_types = _function_parameter_type_map(tree, code)
    base_line = max(0, getattr(function, "start_line", 1) - 1)
    facts: List[SourceFact] = []
    seen: Set[Tuple[str, int, str]] = set()

    for node in _walk(tree.root_node):
        if node.type == "call_expression":
            fn_node = node.child_by_field_name("function")
            args_node = node.child_by_field_name("arguments")
            if fn_node is None:
                continue
            is_member_call = fn_node.type == "field_expression" or _contains_node_type(fn_node, "field_expression")
            argument_path_map = _argument_path_map(args_node, code)
            metadata = {
                "callee_name": _callee_name(fn_node, code),
                "callee_expression": _node_text(code, fn_node).strip(),
                "callee_path": _field_path(fn_node, code),
                "arguments": _argument_texts(args_node, code),
                "argument_paths": _argument_paths(args_node, code),
                "argument_path_map": argument_path_map,
                "argument_type_map": _argument_type_map(argument_path_map, function_parameter_types),
                "output_arguments": _output_argument_paths(args_node, code),
                "result_assignee": _assignment_left_for_call(node, code),
                "statement": _node_text(code, _nearest_statement(node)).strip(),
                "statement_kind": getattr(_nearest_statement(node), "type", ""),
                "function_parameters": function_parameters,
                "function_parameter_types": function_parameter_types,
            }
            kind = "MEMBER_CALL" if is_member_call else "CALL"
            fact = _fact("F_CALL", kind, file_path, function.name, base_line, code, node, metadata)
        elif node.type == "if_statement":
            condition_node = node.child_by_field_name("condition")
            return_expr, return_line, return_text = _first_return_in_consequence(node, code)
            branch_requirements, fallthrough_requirements = _condition_path_requirements(condition_node, code)
            loop_context = _enclosing_loop_context(node, code, base_line)
            metadata = {
                "condition": _condition_expression(node, code),
                "condition_symbols": _identifier_symbols(condition_node, code) if condition_node is not None else [],
                "condition_paths": _field_paths_in_node(condition_node, code),
                "branch_requirements": branch_requirements,
                "fallthrough_requirements": fallthrough_requirements,
                "condition_cases": _condition_case_requirements(condition_node, code),
                "loop_context": loop_context,
                "function_parameters": function_parameters,
                "first_return_expression": return_expr,
                "first_return_line": base_line + return_line if return_line else 0,
                "first_return_code": return_text,
            }
            fact = _fact("F_GUARD", "GUARD", file_path, function.name, base_line, code, node, metadata)
            key = (fact.kind, fact.start_line, fact.code)
            if key not in seen:
                seen.add(key)
                facts.append(fact)
                facts.append(
                    _branch_edge_fact(
                        file_path,
                        function.name,
                        base_line,
                        code,
                        fact,
                        "branch",
                        branch_requirements,
                    )
                )
                facts.append(
                    _branch_edge_fact(
                        file_path,
                        function.name,
                        base_line,
                        code,
                        fact,
                        "fallthrough",
                        fallthrough_requirements,
                )
            )
            continue
        elif node.type in {"for_statement", "while_statement", "do_statement"}:
            metadata = _loop_context(node, code, base_line)
            metadata["loop_cases"] = _loop_cases(node, code)
            metadata["function_parameters"] = function_parameters
            fact = _fact("F_LOOP", "LOOP", file_path, function.name, base_line, code, node, metadata)
        elif node.type == "return_statement":
            fact = _fact(
                "F_RETURN",
                "RETURN",
                file_path,
                function.name,
                base_line,
                code,
                node,
                {"return_expression": _return_expression(node, code), "function_parameters": function_parameters},
            )
        elif node.type == "assignment_expression":
            left, right = _assignment_parts(node, code)
            right_node = node.child_by_field_name("right")
            kind = "FIELD_WRITE" if _contains_node_type(node.child_by_field_name("left"), "field_expression") else "ASSIGNMENT"
            fact = _fact(
                "F_ASSIGN",
                kind,
                file_path,
                function.name,
                base_line,
                code,
                node,
                {
                    "left": left,
                    "right": right,
                    "left_path": _field_path(node.child_by_field_name("left"), code),
                    "right_symbols": _identifier_symbols(right_node, code) if right_node is not None else [],
                    "function_parameters": function_parameters,
                },
            )
        elif node.type == "field_expression":
            fact = _fact(
                "F_FIELD",
                "FIELD_ACCESS",
                file_path,
                function.name,
                base_line,
                code,
                node,
                {"field_path": _field_path(node, code), "function_parameters": function_parameters},
            )
        else:
            continue

        key = (fact.kind, fact.start_line, fact.code)
        if key not in seen:
            seen.add(key)
            facts.append(fact)

    macros = getattr(parse_result, "macros", {}) or {}
    for macro_name in getattr(function, "macro_refs", []) or []:
        macro_info = macros.get(macro_name)
        code_text = getattr(macro_info, "code", macro_name)
        line = getattr(macro_info, "line", function.start_line)
        facts.append(
            SourceFact(
                fact_id=_stable_id("F_CONST", function.name, line, "macro", code_text),
                kind="CONSTANT",
                file=file_path,
                function=function.name,
                start_line=line,
                end_line=line,
                code=code_text,
                symbols=[macro_name],
                metadata={"name": macro_name},
            )
        )

    return facts


def _extract_facts_from_function_fallback(parse_result, function) -> List[SourceFact]:
    """AST fallback for tests when tree-sitter is unavailable."""
    if c_parser is None:
        return []
    file_path = getattr(parse_result, "path", "")
    function_code = getattr(function, "code", "") or ""
    base_line = max(0, getattr(function, "start_line", 1) - 1)
    parser = c_parser.CParser()
    generator = c_generator.CGenerator()
    try:
        tree = parser.parse(function_code)
    except Exception:
        return []
    function_parameters: List[str] = []
    for ext in getattr(tree, "ext", []) or []:
        if not isinstance(ext, c_ast.FuncDef):
            continue
        args = getattr(getattr(ext.decl, "type", None), "args", None)
        for param in getattr(args, "params", []) or []:
            name = getattr(param, "name", "")
            if name and name not in function_parameters:
                function_parameters.append(name)
    facts: List[SourceFact] = []
    seen: Set[Tuple[str, int, str]] = set()

    def line_for(node) -> int:
        coord = getattr(node, "coord", None)
        return base_line + (coord.line if coord and coord.line else 1)

    def text_for(node) -> str:
        try:
            return generator.visit(node)
        except Exception:
            return type(node).__name__

    def symbols_for(node) -> List[str]:
        symbols: Set[str] = set()

        class SymbolVisitor(c_ast.NodeVisitor):
            def visit_ID(self, item):
                symbols.add(item.name)

            def visit_StructRef(self, item):
                self.visit(item.name)
                if isinstance(item.field, c_ast.ID):
                    symbols.add(item.field.name)

        SymbolVisitor().visit(node)
        return sorted(symbols)

    def field_path(node) -> List[str]:
        if isinstance(node, c_ast.ID):
            return [node.name]
        if isinstance(node, c_ast.StructRef):
            return field_path(node.name) + ([node.field.name] if isinstance(node.field, c_ast.ID) else [])
        if isinstance(node, c_ast.UnaryOp):
            return field_path(node.expr)
        return []

    def call_path(node) -> List[str]:
        return field_path(node)

    def call_args(node) -> List[str]:
        if not isinstance(node, c_ast.FuncCall) or node.args is None:
            return []
        return [text_for(expr) for expr in node.args.exprs or []]

    def output_args(node) -> List[str]:
        outputs: List[str] = []
        if not isinstance(node, c_ast.FuncCall) or node.args is None:
            return outputs
        for expr in node.args.exprs or []:
            if isinstance(expr, c_ast.UnaryOp) and expr.op == "&":
                if not isinstance(expr.expr, c_ast.ID):
                    continue
                path = field_path(expr.expr)
                outputs.append(".".join(path) if path else text_for(expr.expr))
        return sorted(set(outputs))

    def argument_paths(node) -> List[List[str]]:
        paths: List[List[str]] = []
        if not isinstance(node, c_ast.FuncCall) or node.args is None:
            return paths
        for expr in node.args.exprs or []:
            path = field_path(expr)
            if path and path not in paths:
                paths.append(path)
        return paths

    def is_nullish(node) -> bool:
        if isinstance(node, c_ast.ID) and node.name == "NULL":
            return True
        if isinstance(node, c_ast.Constant) and node.value == "0":
            return True
        return False

    def condition_requirements(node) -> Tuple[List[Dict], List[Dict]]:
        branch: List[Dict] = []
        fallthrough: List[Dict] = []
        condition_text = text_for(node)
        if isinstance(node, c_ast.UnaryOp) and node.op == "!":
            for path in [field_path(node.expr)]:
                if not path:
                    continue
                branch.append(_condition_requirement(path, "is_null_or_false", "branch", condition_text))
                fallthrough.append(_condition_requirement(path, "is_non_null_or_true", "fallthrough", condition_text))
            return branch, fallthrough
        if isinstance(node, c_ast.BinaryOp) and node.op in {"==", "!="} and (is_nullish(node.left) or is_nullish(node.right)):
            target = node.right if is_nullish(node.left) else node.left
            path = field_path(target)
            if path:
                if node.op == "==":
                    branch.append(_condition_requirement(path, "is_null_or_zero", "branch", condition_text))
                    fallthrough.append(_condition_requirement(path, "is_non_null_or_nonzero", "fallthrough", condition_text))
                else:
                    branch.append(_condition_requirement(path, "is_non_null_or_nonzero", "branch", condition_text))
                    fallthrough.append(_condition_requirement(path, "is_null_or_zero", "fallthrough", condition_text))
            return branch, fallthrough
        if isinstance(node, c_ast.BinaryOp):
            for path in [field_path(node.left), field_path(node.right)]:
                if not path:
                    continue
                branch.append(_condition_requirement(path, "makes_condition_true", "branch", condition_text))
                fallthrough.append(_condition_requirement(path, "makes_condition_false", "fallthrough", condition_text))
            return branch, fallthrough
        for path in [field_path(node)]:
            if not path:
                continue
            branch.append(_condition_requirement(path, "is_true_or_nonzero", "branch", condition_text))
            fallthrough.append(_condition_requirement(path, "is_false_or_zero", "fallthrough", condition_text))
        return branch, fallthrough

    def condition_case_requirement(node, desired: bool, full_condition: str) -> Dict:
        return {
            "expression": text_for(node),
            "relation": "must_be_true" if desired else "must_be_false",
            "required_for": "branch" if desired else "fallthrough",
            "evidence": full_condition,
            "object_paths": [field_path(node)] if field_path(node) else [],
        }

    def merge_condition_case_parts(parts: List[Dict], outcome: str, full_condition: str) -> Dict:
        requirements: List[Dict] = []
        short_circuit = False
        decomposed = False
        for part in parts:
            requirements.extend(part.get("requirements", []) or [])
            short_circuit = short_circuit or bool(part.get("short_circuit"))
            decomposed = decomposed or bool(part.get("decomposed"))
        key_text = "|".join(
            f"{item.get('expression', '')}:{item.get('relation', '')}" for item in requirements
        )
        digest = hashlib.sha1(f"{full_condition}:{outcome}:{key_text}".encode("utf-8")).hexdigest()[:10]
        readable = " and ".join(
            f"`{item.get('expression', '')}` {item.get('relation', '').replace('_', ' ')}"
            for item in requirements
            if item.get("expression")
        )
        return {
            "case_id": f"{outcome}_{digest}",
            "condition": full_condition,
            "outcome": outcome,
            "requirements": requirements,
            "short_circuit": short_circuit,
            "decomposed": decomposed,
            "description": readable or f"drive condition `{full_condition}` to {outcome}",
        }

    def truth_condition_cases(node, desired: bool, full_condition: str) -> List[Dict]:
        if isinstance(node, c_ast.UnaryOp) and node.op == "!":
            return truth_condition_cases(node.expr, not desired, full_condition)
        if isinstance(node, c_ast.BinaryOp) and node.op == "&&":
            if desired:
                cases = []
                for left_case in truth_condition_cases(node.left, True, full_condition):
                    for right_case in truth_condition_cases(node.right, True, full_condition):
                        merged = merge_condition_case_parts([left_case, right_case], "branch", full_condition)
                        merged["decomposed"] = True
                        cases.append(merged)
                return cases
            cases = []
            for left_case in truth_condition_cases(node.left, False, full_condition):
                merged = merge_condition_case_parts([left_case], "fallthrough", full_condition)
                merged["short_circuit"] = True
                merged["decomposed"] = True
                cases.append(merged)
            for left_case in truth_condition_cases(node.left, True, full_condition):
                for right_case in truth_condition_cases(node.right, False, full_condition):
                    merged = merge_condition_case_parts([left_case, right_case], "fallthrough", full_condition)
                    merged["decomposed"] = True
                    cases.append(merged)
            return cases
        if isinstance(node, c_ast.BinaryOp) and node.op == "||":
            if desired:
                cases = []
                for left_case in truth_condition_cases(node.left, True, full_condition):
                    merged = merge_condition_case_parts([left_case], "branch", full_condition)
                    merged["short_circuit"] = True
                    merged["decomposed"] = True
                    cases.append(merged)
                for left_case in truth_condition_cases(node.left, False, full_condition):
                    for right_case in truth_condition_cases(node.right, True, full_condition):
                        merged = merge_condition_case_parts([left_case, right_case], "branch", full_condition)
                        merged["decomposed"] = True
                        cases.append(merged)
                return cases
            cases = []
            for left_case in truth_condition_cases(node.left, False, full_condition):
                for right_case in truth_condition_cases(node.right, False, full_condition):
                    merged = merge_condition_case_parts([left_case, right_case], "fallthrough", full_condition)
                    merged["decomposed"] = True
                    cases.append(merged)
            return cases
        outcome = "branch" if desired else "fallthrough"
        return [
            {
                "case_id": hashlib.sha1(
                    f"{full_condition}:{outcome}:{text_for(node)}:{desired}".encode("utf-8")
                ).hexdigest()[:10],
                "condition": full_condition,
                "outcome": outcome,
                "requirements": [condition_case_requirement(node, desired, full_condition)],
                "short_circuit": False,
                "decomposed": False,
                "description": f"`{text_for(node)}` {'must be true' if desired else 'must be false'}",
            }
        ]

    def condition_cases(node) -> List[Dict]:
        condition_text = text_for(node)
        deduped: List[Dict] = []
        seen: Set[Tuple] = set()
        for case in truth_condition_cases(node, True, condition_text) + truth_condition_cases(node, False, condition_text):
            requirements = tuple(
                sorted(
                    (
                        item.get("expression", ""),
                        item.get("relation", ""),
                    )
                    for item in case.get("requirements", []) or []
                )
            )
            key = (case.get("outcome", ""), requirements)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(case)
        return deduped

    def loop_metadata(node) -> Dict:
        kind = type(node).__name__.lower()
        condition = text_for(getattr(node, "cond", None)) if getattr(node, "cond", None) is not None else ""
        initializer = text_for(getattr(node, "init", None)) if getattr(node, "init", None) is not None else ""
        update = text_for(getattr(node, "next", None)) if getattr(node, "next", None) is not None else ""
        return {
            "kind": kind,
            "start_line": line_for(node),
            "condition": condition,
            "initializer": initializer,
            "update": update,
            "condition_symbols": symbols_for(getattr(node, "cond", None)) if getattr(node, "cond", None) is not None else [],
            "initializer_symbols": symbols_for(getattr(node, "init", None)) if getattr(node, "init", None) is not None else [],
            "update_symbols": symbols_for(getattr(node, "next", None)) if getattr(node, "next", None) is not None else [],
            "symbols": symbols_for(node),
            "loop_cases": [
                {
                    "case_id": "zero_iterations",
                    "role": "loop_boundary",
                    "description": "make the loop condition false before the first iteration"
                    + (f": `{condition}`" if condition else ""),
                },
                {
                    "case_id": "single_iteration",
                    "role": "loop_boundary",
                    "description": "execute the loop body exactly once, then make the loop exit condition hold"
                    + (f": `{condition}`" if condition else ""),
                },
                {
                    "case_id": "multiple_iterations",
                    "role": "loop_boundary",
                    "description": "execute the loop body for more than one iteration"
                    + (f" while evaluating `{condition}`" if condition else ""),
                },
            ],
            "function_parameters": function_parameters,
        }

    def add(kind: str, prefix: str, node, code_text: str, metadata: Optional[Dict] = None) -> Optional[SourceFact]:
        line = line_for(node)
        key = (kind, line, code_text.strip())
        if key in seen:
            return None
        seen.add(key)
        fact = SourceFact(
            fact_id=_stable_id(prefix, function.name, line, kind, code_text.strip()),
            kind=kind,
            file=file_path,
            function=function.name,
            start_line=line,
            end_line=line,
            code=code_text.strip(),
            symbols=symbols_for(node),
            metadata=metadata or {},
        )
        facts.append(fact)
        return fact

    def add_call(node, assignee: str = "", statement_kind: str = "") -> None:
        path = call_path(node.name)
        expression = "->".join(path) if len(path) > 1 else (path[0] if path else text_for(node.name))
        kind = "MEMBER_CALL" if len(path) > 1 else "CALL"
        add(
            kind,
            "F_CALL",
            node,
            text_for(node),
            {
                "callee_name": path[-1] if path else text_for(node.name),
                "callee_expression": expression,
                "callee_path": path,
                "arguments": call_args(node),
                "argument_paths": argument_paths(node),
                "argument_path_map": [
                    {"text": text_for(expr), "path": field_path(expr)}
                    for expr in ((node.args.exprs or []) if node.args is not None else [])
                ],
                "output_arguments": output_args(node),
                "result_assignee": assignee,
                "statement": text_for(node),
                "statement_kind": statement_kind or type(node).__name__,
                "function_parameters": function_parameters,
            },
        )

    class FactVisitor(c_ast.NodeVisitor):
        def visit_If(self, node):
            return_expr = ""
            return_line = 0
            return_code = ""
            branch_requirements, fallthrough_requirements = condition_requirements(node.cond)
            if isinstance(node.iftrue, c_ast.Return):
                return_expr = text_for(node.iftrue.expr) if node.iftrue.expr is not None else ""
                return_line = line_for(node.iftrue)
                return_code = text_for(node.iftrue)
            guard_fact = add(
                "GUARD",
                "F_GUARD",
                node,
                text_for(node),
                {
                    "condition": text_for(node.cond),
                    "condition_symbols": symbols_for(node.cond),
                    "condition_paths": [field_path(node.cond)] if field_path(node.cond) else [],
                    "branch_requirements": branch_requirements,
                    "fallthrough_requirements": fallthrough_requirements,
                    "condition_cases": condition_cases(node.cond),
                    "loop_context": {},
                    "function_parameters": function_parameters,
                    "first_return_expression": return_expr,
                    "first_return_line": return_line,
                    "first_return_code": return_code,
                },
            )
            if guard_fact is not None:
                facts.append(
                    _branch_edge_fact(file_path, function.name, base_line, b"", guard_fact, "branch", branch_requirements)
                )
                facts.append(
                    _branch_edge_fact(
                        file_path,
                        function.name,
                        base_line,
                        b"",
                        guard_fact,
                        "fallthrough",
                        fallthrough_requirements,
                    )
                )
            self.generic_visit(node)

        def visit_For(self, node):
            add("LOOP", "F_LOOP", node, text_for(node), loop_metadata(node))
            self.generic_visit(node)

        def visit_While(self, node):
            metadata = loop_metadata(node)
            metadata["loop_cases"] = [item for item in metadata.get("loop_cases", []) if item.get("case_id")]
            add("LOOP", "F_LOOP", node, text_for(node), metadata)
            self.generic_visit(node)

        def visit_DoWhile(self, node):
            metadata = loop_metadata(node)
            metadata["loop_cases"] = [
                item for item in metadata.get("loop_cases", []) if item.get("case_id") != "zero_iterations"
            ]
            add("LOOP", "F_LOOP", node, text_for(node), metadata)
            self.generic_visit(node)

        def visit_Return(self, node):
            add(
                "RETURN",
                "F_RETURN",
                node,
                text_for(node),
                {
                    "return_expression": text_for(node.expr) if node.expr is not None else "",
                    "function_parameters": function_parameters,
                },
            )
            if isinstance(node.expr, c_ast.FuncCall):
                add_call(node.expr, statement_kind="Return")
            self.generic_visit(node)

        def visit_Assignment(self, node):
            left_path = field_path(node.lvalue)
            kind = "FIELD_WRITE" if len(left_path) > 1 else "ASSIGNMENT"
            add(
                kind,
                "F_ASSIGN",
                node,
                text_for(node),
                {
                    "left": text_for(node.lvalue),
                    "right": text_for(node.rvalue),
                    "left_path": left_path,
                    "right_symbols": symbols_for(node.rvalue),
                    "function_parameters": function_parameters,
                },
            )
            if isinstance(node.rvalue, c_ast.FuncCall):
                add_call(node.rvalue, assignee=text_for(node.lvalue))
            self.generic_visit(node)

        def visit_Decl(self, node):
            if isinstance(node.init, c_ast.FuncCall):
                add_call(node.init, assignee=node.name or "")
            self.generic_visit(node)

        def visit_FuncCall(self, node):
            add_call(node)
            self.generic_visit(node)

    FactVisitor().visit(tree)

    return facts


def extract_source_facts(parse_result, function) -> Tuple[List[SourceFact], List[str]]:
    functions = _function_map(parse_result)
    closure = build_internal_call_closure(parse_result, function)
    if not closure:
        closure = [function.name]

    facts: List[SourceFact] = []
    seen: Set[str] = set()
    for name in closure:
        current = functions.get(name)
        if current is None:
            continue
        for fact in _extract_facts_from_function(parse_result, current):
            if fact.fact_id in seen:
                continue
            seen.add(fact.fact_id)
            facts.append(fact)
    return _link_related_facts(facts), closure
