import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

try:
    from tree_sitter_languages import get_parser
except ImportError:
    get_parser = None


C_PARSER = get_parser("c") if get_parser is not None else None
SCENARIO_MARKER_PATTERN = re.compile(r"RACA_SCENARIO\s*:\s*([^\n\r*]+)")
SCENARIO_ID_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_:\-]*$")
VARIANT_MARKER_PATTERN = re.compile(r"RACA_VARIANT\s*:\s*([A-Za-z_][A-Za-z0-9_:\-]*)")


@dataclass
class TestFunctionInfo:
    name: str
    body: str
    full_text: str
    scenario_ids: List[str] = field(default_factory=list)
    variant_id: str = ""
    call_names: List[str] = field(default_factory=list)


@dataclass
class TestSourceInfo:
    includes: List[str] = field(default_factory=list)
    externs: List[str] = field(default_factory=list)
    suite_names: List[str] = field(default_factory=list)
    test_functions: List[TestFunctionInfo] = field(default_factory=list)
    function_definitions: List[str] = field(default_factory=list)
    function_call_names: Dict[str, List[str]] = field(default_factory=dict)
    function_identifier_names: Dict[str, List[str]] = field(default_factory=dict)
    macro_definitions: List[str] = field(default_factory=list)
    registered_tests: List[str] = field(default_factory=list)


def _node_text(code: bytes, node) -> str:
    return code[node.start_byte:node.end_byte].decode("utf-8", errors="ignore")


def _walk(node):
    stack = [node]
    while stack:
        current = stack.pop()
        yield current
        stack.extend(reversed(current.children))


def _function_name_from_definition(func_node, code: bytes) -> Optional[str]:
    declarator = None
    for child in func_node.children:
        if "declarator" in child.type:
            declarator = child
            break
    if declarator is None:
        return None
    stack = [declarator]
    while stack:
        node = stack.pop()
        if node.type == "identifier":
            return _node_text(code, node)
        if node.type == "parameter_list":
            continue
        stack.extend(node.children)
    text = _node_text(code, func_node)
    signature = text.split("{", 1)[0]
    match = re.search(r"([A-Za-z_][A-Za-z0-9_]*)\s*\([^;{}]*\)\s*$", signature.strip())
    if match:
        return match.group(1)
    return None


def _is_kunit_test_definition(func_node, code: bytes, name: str) -> bool:
    full_text = _node_text(code, func_node)
    signature = full_text.split("{", 1)[0]
    pattern = (
        r"\bvoid\s+"
        + re.escape(name)
        + r"\s*\(\s*struct\s+kunit\s*\*\s*test\s*\)\s*$"
    )
    return bool(re.search(pattern, signature.strip(), re.MULTILINE))


def _callee_name(fn_node, code: bytes) -> str:
    text = re.sub(r"\s+", "", _node_text(code, fn_node).strip())
    parts = re.split(r"->|\.|\(|\)|\*", text)
    parts = [part for part in parts if part]
    return parts[-1] if parts else text


def _find_matching_brace(code: str, open_idx: int) -> Optional[int]:
    depth = 0
    i = open_idx
    in_string = False
    in_char = False
    escaped = False
    while i < len(code):
        ch = code[i]
        if escaped:
            escaped = False
            i += 1
            continue
        if ch == "\\":
            escaped = True
            i += 1
            continue
        if ch == '"' and not in_char:
            in_string = not in_string
        elif ch == "'" and not in_string:
            in_char = not in_char
        elif not in_string and not in_char:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return i
        i += 1
    return None


def _strip_c_comments(text: str) -> str:
    without_block = re.sub(r"/\*[\s\S]*?\*/", " ", text or "")
    return re.sub(r"//[^\n\r]*", " ", without_block)


def _extract_call_names_from_text(text: str) -> List[str]:
    names = []
    text = _strip_c_comments(text or "")
    for match in re.finditer(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(", text):
        name = match.group(1)
        if name not in names:
            names.append(name)
    return names


def _extract_identifier_names_from_text(text: str) -> List[str]:
    names: List[str] = []
    text = _strip_c_comments(text or "")
    for match in re.finditer(r"\b([A-Za-z_][A-Za-z0-9_]*)\b", text):
        name = match.group(1)
        if name not in names:
            names.append(name)
    return names


def _identifier_names_from_node(node, code: bytes) -> List[str]:
    names: List[str] = []
    if node is None:
        return names
    for inner in _walk(node):
        if inner.type != "identifier":
            continue
        name = _node_text(code, inner)
        if name not in names:
            names.append(name)
    return names


def _leading_raca_comment_block(code: str, start_idx: int) -> str:
    """Return contiguous RACA annotation comments immediately before a function."""
    prefix = code[:start_idx]
    lines = prefix.splitlines()
    collected: List[str] = []
    i = len(lines) - 1

    while i >= 0 and not lines[i].strip():
        i -= 1

    while i >= 0:
        stripped = lines[i].strip()
        if not stripped:
            break
        if (
            stripped.startswith("//")
            or stripped.startswith("/*")
            or stripped.startswith("*")
            or stripped.endswith("*/")
        ):
            collected.append(lines[i])
            i -= 1
            continue
        break

    block = "\n".join(reversed(collected))
    return block if "RACA_" in block else ""


def _scenario_ids_for_function(code: str, full_text: str, start_idx: int) -> List[str]:
    ids: List[str] = []
    marker_text = "\n".join(
        part for part in (_leading_raca_comment_block(code, start_idx), full_text) if part
    )
    for payload in SCENARIO_MARKER_PATTERN.findall(marker_text):
        normalized_payload = payload.split("*/", 1)[0].strip()
        for scenario_id in re.split(r"\s*,\s*", normalized_payload):
            scenario_id = scenario_id.strip()
            if not scenario_id or not SCENARIO_ID_PATTERN.match(scenario_id):
                continue
            if scenario_id not in ids:
                ids.append(scenario_id)
    return ids


def _variant_id_for_function(code: str, full_text: str, start_idx: int) -> str:
    marker_text = "\n".join(
        part for part in (_leading_raca_comment_block(code, start_idx), full_text) if part
    )
    match = VARIANT_MARKER_PATTERN.search(marker_text)
    return match.group(1).strip() if match else ""


def _inspect_with_tree_sitter(code_text: str) -> TestSourceInfo:
    code = code_text.encode("utf-8", errors="ignore")
    tree = C_PARSER.parse(code)
    info = TestSourceInfo(
        includes=_extract_includes_fallback(code_text),
        externs=_extract_externs_fallback(code_text),
        suite_names=_extract_suite_names_fallback(code_text),
        macro_definitions=_extract_macro_definitions_fallback(code_text),
    )
    for node in _walk(tree.root_node):
        if node.type == "call_expression":
            fn_node = node.child_by_field_name("function")
            if fn_node is not None and _callee_name(fn_node, code) == "KUNIT_CASE":
                args_node = node.child_by_field_name("arguments")
                if args_node is not None:
                    for child in args_node.children:
                        if child.is_named:
                            name = _node_text(code, child).strip()
                            if name not in info.registered_tests:
                                info.registered_tests.append(name)
                            break
            continue
        if node.type != "function_definition":
            continue
        name = _function_name_from_definition(node, code)
        if not name:
            continue
        info.function_definitions.append(name)
        full_text = _node_text(code, node)
        body_node = node.child_by_field_name("body")
        function_call_names: List[str] = []
        for inner in _walk(node):
            if inner.type != "call_expression":
                continue
            fn_node = inner.child_by_field_name("function")
            if fn_node is None:
                continue
            call_name = _callee_name(fn_node, code)
            if call_name not in function_call_names:
                function_call_names.append(call_name)
        info.function_call_names[name] = function_call_names
        info.function_identifier_names[name] = _identifier_names_from_node(body_node, code)
        if not _is_kunit_test_definition(node, code, name):
            continue
        body = _node_text(code, body_node) if body_node is not None else full_text
        info.test_functions.append(
            TestFunctionInfo(
                name=name,
                body=body,
                full_text=full_text,
                scenario_ids=_scenario_ids_for_function(code_text, full_text, node.start_byte),
                variant_id=_variant_id_for_function(code_text, full_text, node.start_byte),
                call_names=function_call_names,
            )
        )
    for name in _extract_function_definition_names_fallback(code_text):
        if name not in info.function_definitions:
            info.function_definitions.append(name)
    return info


def _extract_includes_fallback(code: str) -> List[str]:
    return [line.strip() for line in code.splitlines() if line.strip().startswith("#include")]


def _extract_externs_fallback(code: str) -> List[str]:
    return [line.strip() for line in code.splitlines() if line.strip().startswith("extern ")]


def _extract_suite_names_fallback(code: str) -> List[str]:
    names: List[str] = []
    suite_pattern = re.compile(
        r"struct\s+kunit_suite\s+[A-Za-z_][A-Za-z0-9_]*\s*=\s*\{(?P<body>.*?)\};",
        re.DOTALL,
    )
    for match in suite_pattern.finditer(code or ""):
        name_match = re.search(r"\.name\s*=\s*\"([^\"]+)\"", match.group("body"))
        if name_match:
            names.append(name_match.group(1))
    return names


def _extract_macro_definitions_fallback(code: str) -> List[str]:
    return re.findall(r"^\s*#\s*define\s+([A-Za-z_][A-Za-z0-9_]*)\b", code or "", re.MULTILINE)


def _extract_function_definition_names_fallback(code: str) -> List[str]:
    names: List[str] = []
    pattern = re.compile(
        r"(?m)^[\t ]*(?:static\s+)?(?:inline\s+)?(?:[A-Za-z_][A-Za-z0-9_]*\s+)+\*?\s*([A-Za-z_][A-Za-z0-9_]*)\s*\([^;{}]*\)\s*\{"
    )
    for match in pattern.finditer(code or ""):
        name = match.group(1)
        if name not in names:
            names.append(name)
    return names


def _extract_function_call_names_fallback(code: str) -> Dict[str, List[str]]:
    calls: Dict[str, List[str]] = {}
    pattern = re.compile(
        r"(?m)^[\t ]*(?:static\s+)?(?:inline\s+)?(?:[A-Za-z_][A-Za-z0-9_]*\s+)+\*?\s*([A-Za-z_][A-Za-z0-9_]*)\s*\([^;{}]*\)\s*\{"
    )
    for match in pattern.finditer(code or ""):
        name = match.group(1)
        brace_idx = code.find("{", match.end() - 1)
        if brace_idx < 0:
            continue
        end_idx = _find_matching_brace(code, brace_idx)
        if end_idx is None:
            continue
        body = code[brace_idx : end_idx + 1]
        calls[name] = _extract_call_names_from_text(body)
    return calls


def _extract_function_identifier_names_fallback(code: str) -> Dict[str, List[str]]:
    identifiers: Dict[str, List[str]] = {}
    pattern = re.compile(
        r"(?m)^[\t ]*(?:static\s+)?(?:inline\s+)?(?:[A-Za-z_][A-Za-z0-9_]*\s+)+\*?\s*([A-Za-z_][A-Za-z0-9_]*)\s*\([^;{}]*\)\s*\{"
    )
    for match in pattern.finditer(code or ""):
        name = match.group(1)
        brace_idx = code.find("{", match.end() - 1)
        if brace_idx < 0:
            continue
        end_idx = _find_matching_brace(code, brace_idx)
        if end_idx is None:
            continue
        body = code[brace_idx : end_idx + 1]
        identifiers[name] = _extract_identifier_names_from_text(body)
    return identifiers


def _inspect_with_fallback(code: str) -> TestSourceInfo:
    info = TestSourceInfo(
        includes=_extract_includes_fallback(code),
        externs=_extract_externs_fallback(code),
        suite_names=_extract_suite_names_fallback(code),
        macro_definitions=_extract_macro_definitions_fallback(code),
        function_call_names=_extract_function_call_names_fallback(code),
        function_identifier_names=_extract_function_identifier_names_fallback(code),
    )
    pattern = re.compile(
        r"static\s+void\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(\s*struct\s+kunit\s*\*\s*test\s*\)",
        re.MULTILINE,
    )
    for match in pattern.finditer(code or ""):
        name = match.group(1)
        brace_idx = code.find("{", match.end())
        if brace_idx < 0:
            continue
        end_idx = _find_matching_brace(code, brace_idx)
        if end_idx is None:
            continue
        full_text = code[match.start() : end_idx + 1]
        body = code[brace_idx : end_idx + 1]
        info.function_definitions.append(name)
        info.test_functions.append(
            TestFunctionInfo(
                name=name,
                body=body,
                full_text=full_text,
                scenario_ids=_scenario_ids_for_function(code, full_text, match.start()),
                variant_id=_variant_id_for_function(code, full_text, match.start()),
                call_names=_extract_call_names_from_text(body),
            )
        )
    for name in _extract_function_definition_names_fallback(code):
        if name not in info.function_definitions:
            info.function_definitions.append(name)
    for name in re.findall(r"\bKUNIT_CASE\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)", code or ""):
        if name not in info.registered_tests:
            info.registered_tests.append(name)
    return info


def inspect_test_source(code: str) -> TestSourceInfo:
    if C_PARSER is not None:
        return _inspect_with_tree_sitter(code or "")
    return _inspect_with_fallback(code or "")


def test_function_map(code: str) -> Dict[str, TestFunctionInfo]:
    return {item.name: item for item in inspect_test_source(code).test_functions}
