import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING, Set

if TYPE_CHECKING:
    from analysis.driver_parser import FileParseResult, FunctionInfo, MacroInfo

from tree_sitter_languages import get_parser


# -----------------------------
# Configuration: Section Markers
# -----------------------------

SECTION_BEGIN = (
    "/* ========================= */\n"
    "/* KUNIT TEST STUBS BEGIN    */\n"
    "/* ========================= */\n"
)
SECTION_END = (
    "/* ========================= */\n"
    "/* KUNIT TEST STUBS END      */\n"
    "/* ========================= */\n"
)

SECTION_GUARD_PATTERN = re.compile(r"#if\s+IS_ENABLED\(\s*(?P<symbol>[A-Za-z0-9_]+)\s*\)")

# Per-stub markers (for function-level replacement)
def stub_begin_marker(func_name: str) -> str:
    return f"/* KUNIT_STUB_BEGIN: {func_name} */"

def stub_end_marker(func_name: str) -> str:
    return f"/* KUNIT_STUB_END: {func_name} */"


BOUNDARY_DECLS_BEGIN = "/* RACA_BOUNDARY_HOOK_DECLS_BEGIN */"
BOUNDARY_DECLS_END = "/* RACA_BOUNDARY_HOOK_DECLS_END */"
BOUNDARY_DECL_PATTERN = re.compile(
    r"/\*\s*RACA_BOUNDARY_HOOK_DECL:\s*boundary=([A-Za-z0-9_]+)\s*;\s*original=([A-Za-z_][A-Za-z0-9_]*)\s*;\s*invoke=([A-Za-z_][A-Za-z0-9_]*)\s*\*/"
)


def _safe_boundary_suffix(boundary_id: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_]", "_", boundary_id or "boundary")
    if not text or text[0].isdigit():
        text = f"B_{text}"
    return text


def _boundary_symbol_names(boundary_id: str) -> Dict[str, str]:
    suffix = _safe_boundary_suffix(boundary_id)
    base = f"raca_boundary_{suffix}"
    return {
        "invoke": f"{base}_invoke",
        "hook": f"{base}_hook",
        "call_count": f"{base}_call_count",
        "set_hook": f"{base}_set_hook",
        "clear_hook": f"{base}_clear_hook",
        "get_call_count": f"{base}_get_call_count",
    }


# -----------------------------
# Tree-sitter signature extraction
# -----------------------------

C_PARSER = get_parser("c")

def _slice(code: bytes, node) -> str:
    return code[node.start_byte:node.end_byte].decode("utf-8", errors="ignore")

def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def _write_text(path: str, content: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

def extract_function_name_from_def(func_node, code: bytes) -> Optional[str]:
    """
    Extract function name from function_definition node.
    """
    target = None
    for c in func_node.children:
        if "declarator" in c.type:
            target = c
            break
    if target is None:
        return None

    stack = [target]
    while stack:
        n = stack.pop()
        if n.type == "identifier":
            return _slice(code, n)
        if n.type == "parameter_list":
            continue
        stack.extend(n.children)
    return None

def find_function_definition_node(tree, func_name: str, code: bytes):
    root = tree.root_node
    stack = [root]
    while stack:
        n = stack.pop()
        if n.type == "function_definition":
            name = extract_function_name_from_def(n, code)
            if name == func_name:
                return n
        stack.extend(n.children)
    return None

def extract_param_list_text_and_names(func_node, code: bytes) -> Tuple[str, List[str]]:
    """
    Return:
      - param_list_text: "(struct i2c_client *client, u8 reg, u8 *value)" or "(void)"
      - arg_names: ["client","reg","value"]
    """
    param_list = None
    stack = [func_node]
    while stack:
        n = stack.pop()
        if n.type == "parameter_list":
            param_list = n
            break
        stack.extend(n.children)

    if param_list is None:
        return "(void)", []

    param_list_text = _slice(code, param_list).strip()
    if re.match(r"^\(\s*void\s*\)$", param_list_text):
        return "(void)", []

    arg_names: List[str] = []

    #  关键：按 children 顺序遍历
    for child in param_list.children:
        if child.type == "parameter_declaration":
            name = None
            stack = [child]
            while stack:
                n = stack.pop()
                if n.type == "identifier":
                    name = _slice(code, n)
                stack.extend(n.children)
            if name:
                arg_names.append(name)

    return param_list_text, arg_names

def extract_return_type_text(func_node, code: bytes, func_name: str) -> str:
    """
    Best-effort return type extraction.
    Strategy:
      - Take the text from function start to the first occurrence of func_name
      - Remove 'static' and 'inline'
      - Return the last line of that prefix (handles attributes/macros above)
    """
    full_text = _slice(code, func_node).strip()
    idx = full_text.find(func_name)
    if idx <= 0:
        return "int"

    prefix = full_text[:idx].strip()
    last_line = prefix.splitlines()[-1].strip()

    # Remove storage-class / inline keywords from export signature
    last_line = re.sub(r"\bstatic\b", "", last_line).strip()
    last_line = re.sub(r"\binline\b", "", last_line).strip()

    # If it becomes empty, fallback
    return last_line if last_line else "int"


# -----------------------------
# Stub block generation (your format)
# -----------------------------

@dataclass
class StubSpec:
    driver_c_path: str
    target_func_name: str
    config_symbol: str              # e.g., "CONFIG_LEDS_LP3944_KUNIT_TEST"
    export_suffix: str = "_test_export"  # export function name: <orig><suffix>
    export_gpl: bool = True         # EXPORT_SYMBOL_GPL vs EXPORT_SYMBOL
    mode: str = "incremental"       # "incremental" or "single"
    parse_result: Optional["FileParseResult"] = None  # optional parser output for context stubs
    enable_boundary_hooks: bool = True  # direct hardware hook controls are part of the full method


@dataclass
class ExportInterface:
    prototype: str
    source_symbol: str
    source_kind: str
    description: str = ""
    boundary_id: str = ""
    boundary_expression: str = ""
    boundary_control_role: str = ""


@dataclass
class StubGenerationResult:
    file_content: str
    export_interfaces: List[ExportInterface]


def _extract_macro_expression(macro_info: "MacroInfo") -> str:
    parts = macro_info.code.split(None, 2)
    if len(parts) < 3:
        return ""
    expr = parts[2].strip()
    expr = expr.split("//", 1)[0].strip()
    expr = expr.split("/*", 1)[0].strip()
    return expr


def _strip_enclosing_parentheses(expr: str) -> str:
    expr = expr.strip()
    while expr.startswith("(") and expr.endswith(")"):
        depth = 0
        balanced = True
        for i, ch in enumerate(expr):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0 and i != len(expr) - 1:
                    balanced = False
                    break
                if depth < 0:
                    balanced = False
                    break
        if depth != 0 or not balanced:
            break
        expr = expr[1:-1].strip()
    return expr


def _parse_integer_literal(expr: str) -> Tuple[Optional[int], str]:
    expr = _strip_enclosing_parentheses(expr)
    cleaned = re.sub(r"\s+", "", expr)
    match = re.match(r"^([+-]?)(0[xX][0-9A-Fa-f]+|0[0-7]*|[0-9]+)([uUlL]*)$", cleaned)
    if not match:
        return None, ""
    sign, literal, suffix = match.groups()
    try:
        value = int(literal, 0)
    except ValueError:
        return None, ""
    if sign == "-":
        value = -value
    return value, suffix


def _literal_type_from_suffix(suffix: str) -> Optional[str]:
    if not suffix:
        return None
    s = suffix.lower()
    if "u" in s and "ll" in s:
        return "unsigned long long"
    if "ll" in s:
        return "long long"
    if "u" in s and "l" in s:
        return "unsigned long"
    if "l" in s:
        return "long"
    if "u" in s:
        return "unsigned int"
    return None


def _literal_type_from_value(value: int) -> str:
    if value < 0:
        return "int"
    if value <= 0xFF:
        return "u8"
    if value <= 0xFFFF:
        return "u16"
    if value <= 0xFFFFFFFF:
        return "u32"
    return "unsigned long long"


def _infer_macro_return_type(macro_info: "MacroInfo") -> str:
    expr = _extract_macro_expression(macro_info)
    if not expr:
        return "int"
    cast_match = re.match(r"^\(\s*([^)]+)\s*\)\s*(.+)$", expr)
    if cast_match:
        cast_type = cast_match.group(1).strip()
        if cast_type:
            return cast_type
    value, suffix = _parse_integer_literal(expr)
    if value is not None:
        literal_type = _literal_type_from_suffix(suffix)
        if literal_type:
            return literal_type
        return _literal_type_from_value(value)
    return "int"

def build_stub_block(
    orig_func_name: str,
    export_func_name: str,
    return_type: str,
    param_list_text: str,
    arg_names: List[str],
    export_gpl: bool,
    source_kind: str,
    context_block: Optional[Tuple[str, List[ExportInterface]]] = None,
) -> Tuple[str, List[ExportInterface]]:
    """
    Produce block like:
    /* prototype ... */
    int foo_test_export(...);
    int foo_test_export(...) { return foo(...); }
    EXPORT_SYMBOL_GPL(foo_test_export);
    """
    base_block, base_interface = _build_call_wrapper_block(
        orig_func_name=orig_func_name,
        export_func_name=export_func_name,
        return_type=return_type,
        param_list_text=param_list_text,
        arg_names=arg_names,
        export_gpl=export_gpl,
        source_kind=source_kind,
    )
    if not base_block.endswith("\n"):
        base_block += "\n"
    block = f"{stub_begin_marker(orig_func_name)}\n"
    interfaces: List[ExportInterface] = []
    if context_block:
        context_text, context_interfaces = context_block
        block += context_text
        if not context_text.endswith("\n"):
            block += "\n"
        interfaces.extend(context_interfaces)
    block += base_block
    interfaces.append(base_interface)
    block += f"{stub_end_marker(orig_func_name)}\n"
    return block, interfaces


def _build_call_wrapper_block(
    orig_func_name: str,
    export_func_name: str,
    return_type: str,
    param_list_text: str,
    arg_names: List[str],
    export_gpl: bool,
    source_kind: str,
) -> Tuple[str, ExportInterface]:
    call_args = ", ".join(arg_names)
    call_expr = f"{orig_func_name}({call_args})" if call_args else f"{orig_func_name}()"
    if return_type.strip() == "void":
        body_line = f"    {call_expr};\n"
    else:
        body_line = f"    return {call_expr};\n"
    return _build_export_wrapper(
        export_func_name=export_func_name,
        return_type=return_type,
        param_list_text=param_list_text,
        body_lines=[body_line],
        export_gpl=export_gpl,
        source_symbol=orig_func_name,
        source_kind=source_kind,
        description=f"Wrapper around {orig_func_name}()",
    )


def _build_export_wrapper(
    export_func_name: str,
    return_type: str,
    param_list_text: str,
    body_lines: List[str],
    export_gpl: bool,
    source_symbol: str,
    source_kind: str,
    description: str,
    prototype_comment: str = "/* Prototype first to avoid -Wmissing-prototypes */",
) -> Tuple[str, ExportInterface]:
    export_macro = "EXPORT_SYMBOL_GPL" if export_gpl else "EXPORT_SYMBOL"
    block = ""
    if prototype_comment:
        block += f"{prototype_comment}\n"
    block += f"/* Original {source_kind}: {source_symbol} */\n"
    block += (
        f"{return_type} {export_func_name}{param_list_text};\n\n"
        f"{return_type} {export_func_name}{param_list_text}\n"
        "{\n"
    )
    body = "".join(body_lines)
    block += body
    block += (
        "}\n"
        f"{export_macro}({export_func_name});\n"
    )
    prototype = f"extern {return_type} {export_func_name}{param_list_text};"
    interface = ExportInterface(
        prototype=prototype,
        source_symbol=source_symbol,
        source_kind=source_kind,
        description=description,
    )
    return block, interface


def _macro_is_function_like(macro_info: "MacroInfo") -> bool:
    # Function-like macros have parentheses immediately after the name.
    # Example: #define FOO(x) ...
    pattern = re.compile(r"#\s*define\s+" + re.escape(macro_info.name) + r"\s*\(")
    return bool(pattern.search(macro_info.code))


def _sanitize_export_basename(name: str, prefix: str = "") -> str:
    base = re.sub(r"[^0-9A-Za-z_]", "_", name)
    if prefix:
        base = f"{prefix}_{base}"
    if not base:
        base = "kunit_stub"
    if base[0].isdigit():
        base = f"stub_{base}"
    return base.lower()


def _build_macro_wrapper_blocks(
    target_info: "FunctionInfo",
    spec: StubSpec,
    parse_result: "FileParseResult",
) -> List[Tuple[str, ExportInterface]]:
    wrappers: List[Tuple[str, ExportInterface]] = []
    macro_names = target_info.file_internal_symbols.get("macros", [])
    file_func_names = parse_result.file_level_defs.get("functions", set())
    for name in macro_names:
        macro_info = parse_result.macros.get(name)
        if macro_info is None or _macro_is_function_like(macro_info):
            continue
        base_name = _sanitize_export_basename(name)
        if base_name in file_func_names:
            base_name = _sanitize_export_basename(name, "macro")
        export_name = f"{base_name}{spec.export_suffix}"
        return_type = _infer_macro_return_type(macro_info)
        body_line = f"    return ({return_type})({name});\n"
        block, interface = _build_export_wrapper(
                export_func_name=export_name,
                return_type=return_type,
                param_list_text="(void)",
                body_lines=[body_line],
                export_gpl=spec.export_gpl,
                source_symbol=name,
                source_kind="macro",
                description=f"Accessor for macro {name}",
                prototype_comment=f"/* Accessor for macro {name} */",
            )
        wrappers.append((block, interface))
    return wrappers


def _build_struct_wrapper_blocks(
    target_info: "FunctionInfo",
    spec: StubSpec,
    parse_result: "FileParseResult",
) -> List[Tuple[str, ExportInterface]]:
    wrappers: List[Tuple[str, ExportInterface]] = []
    seen = set()
    struct_candidates = list(target_info.type_refs)
    struct_candidates.extend(target_info.file_internal_symbols.get("types", []))

    declared_forward: Set[str] = set()

    for name in struct_candidates:
        if name in seen or name not in parse_result.structs:
            continue
        seen.add(name)
        export_name = f"{_sanitize_export_basename(name, 'struct')}{spec.export_suffix}"
        return_type = f"struct {name} *"
        body_lines = [
            f"    static struct {name} dummy;\n",
            "    return &dummy;\n",
        ]
        forward_decl = ""
        if name not in declared_forward:
            declared_forward.add(name)
            forward_decl = f"struct {name};\n"
        body, interface = _build_export_wrapper(
                export_func_name=export_name,
                return_type=return_type,
                param_list_text="(void)",
                body_lines=body_lines,
                export_gpl=spec.export_gpl,
                source_symbol=name,
                source_kind="struct",
                description=f"Accessor for struct {name}",
                prototype_comment=f"/* Accessor for struct {name} */",
            )
        accessor_var = f"struct {name} *{name}_test_handle = {export_name}();\n"
        block = forward_decl + body + accessor_var
        wrappers.append((block, interface))
    return wrappers


def _build_called_function_wrapper_blocks(
    target_info: "FunctionInfo",
    spec: StubSpec,
    tree,
    code_bytes: bytes,
) -> List[Tuple[str, ExportInterface]]:
    wrappers: List[Tuple[str, ExportInterface]] = []
    seen = set()
    for callee in target_info.calls:
        if callee == target_info.name or callee in seen:
            continue
        seen.add(callee)
        func_node = find_function_definition_node(tree, callee, code_bytes)
        if func_node is None:
            continue
        param_list_text, arg_names = extract_param_list_text_and_names(func_node, code_bytes)
        return_type = extract_return_type_text(func_node, code_bytes, callee)
        export_name = f"{callee}{spec.export_suffix}"
        block, interface = _build_call_wrapper_block(
                orig_func_name=callee,
                export_func_name=export_name,
                return_type=return_type,
                param_list_text=param_list_text,
                arg_names=arg_names,
                export_gpl=spec.export_gpl,
                source_kind="callee",
            )
        wrappers.append((block, interface))
    return wrappers


def _build_related_context_block(
    func_name: str,
    spec: StubSpec,
    parse_result: Optional["FileParseResult"],
    tree,
    code_bytes: bytes,
) -> Tuple[str, List[ExportInterface]]:
    if parse_result is None:
        return "", []

    target_info: Optional["FunctionInfo"] = None
    for f in parse_result.functions:
        if f.name == func_name:
            target_info = f
            break

    if target_info is None:
        return "", []

    sections: List[str] = []
    interfaces: List[ExportInterface] = []
    for block, interface in _build_called_function_wrapper_blocks(target_info, spec, tree, code_bytes):
        sections.append(block)
        interfaces.append(interface)

    if not sections:
        return "", []

    return "\n".join(sections), interfaces


def _remove_boundary_decl_blocks(content: str) -> str:
    pattern = re.compile(
        re.escape(BOUNDARY_DECLS_BEGIN) + r".*?" + re.escape(BOUNDARY_DECLS_END) + r"\s*\n?",
        re.DOTALL,
    )
    return pattern.sub("", content)


def remove_generated_boundary_instrumentation(content: str) -> str:
    """Remove RACA direct-boundary callsite instrumentation from driver source text."""
    mapping: Dict[str, str] = {}
    for match in BOUNDARY_DECL_PATTERN.finditer(content or ""):
        _, original, invoke = match.groups()
        mapping[invoke] = original
    restored = content or ""
    for invoke, original in mapping.items():
        restored = re.sub(r"\b" + re.escape(invoke) + r"\s*\(", f"{original}(", restored)
    return _remove_boundary_decl_blocks(restored)


def _file_local_macro_names(parse_result) -> Set[str]:
    defs = getattr(parse_result, "file_level_defs", {}) or {}
    macros = defs.get("macros", set()) or set()
    return {str(item) for item in macros if item}


def _statement_is_return_like(metadata: Dict) -> bool:
    statement_kind = str(metadata.get("statement_kind", "") or "")
    return statement_kind in {"return_statement", "Return"}


def _call_result_controls_observable_behavior(metadata: Dict) -> bool:
    if metadata.get("result_assignee"):
        return True
    if metadata.get("output_arguments"):
        return True
    if _statement_is_return_like(metadata):
        return True
    for edge in metadata.get("relation_edges", []) or []:
        if not isinstance(edge, dict):
            continue
        relation = str(edge.get("relation", "") or "")
        if relation.startswith("call_result_") or relation.startswith("call_output_"):
            return True
    return False


def _fact_for_boundary(boundary, facts_by_id: Dict[str, object]):
    for fact_id in getattr(boundary, "fact_ids", []) or []:
        fact = facts_by_id.get(fact_id)
        if fact is not None:
            return fact
    return None


def _direct_call_boundary_is_safely_hookable(boundary, fact, parse_result) -> bool:
    """Return whether the current C hook mechanism can safely control a call.

    Boundary discovery is intentionally broader than hook generation.  A source
    call may still be a useful scenario fact, but the generated hook block relies
    on `typeof(callee)` and therefore must only be emitted for direct external
    function symbols that are not file-local macros and whose result/output is
    actually needed to drive a test path.
    """
    if getattr(boundary, "source_fact_kind", "") != "CALL":
        return False
    expression = str(getattr(boundary, "expression", "") or "").strip()
    if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", expression):
        return False
    if expression in _file_local_macro_names(parse_result):
        return False
    if getattr(boundary, "semantic_role", "unknown") != "hardware_boundary":
        return False
    metadata = getattr(fact, "metadata", {}) or {}
    if not _call_result_controls_observable_behavior(metadata):
        return False
    return True


def _direct_boundary_candidates_for_stub(spec: StubSpec, target_info) -> List[object]:
    if spec.parse_result is None:
        return []
    try:
        from scenario.hardware_boundary_analyzer import analyze_boundary_candidates
        from scenario.source_fact_extractor import extract_source_facts
    except Exception:
        return []
    try:
        facts, closure = extract_source_facts(spec.parse_result, target_info)
        boundaries = analyze_boundary_candidates(spec.parse_result, facts)
    except Exception:
        return []
    facts_by_id = {getattr(fact, "fact_id", ""): fact for fact in facts}
    closure_set = set(closure or [])
    selected = []
    seen = set()
    for boundary in boundaries:
        fact = _fact_for_boundary(boundary, facts_by_id)
        if fact is None:
            continue
        if closure_set and getattr(boundary, "source_function", "") not in closure_set:
            continue
        if not _direct_call_boundary_is_safely_hookable(boundary, fact, spec.parse_result):
            continue
        if boundary.candidate_id in seen:
            continue
        seen.add(boundary.candidate_id)
        selected.append(boundary)
    return selected


def _callee_expression_from_call(call_node, code: bytes) -> str:
    fn_node = call_node.child_by_field_name("function")
    return _slice(code, fn_node).strip() if fn_node is not None else ""


def _call_arguments_text(call_node, code: bytes) -> str:
    args_node = call_node.child_by_field_name("arguments")
    if args_node is None:
        return ""
    text = _slice(code, args_node).strip()
    if text.startswith("(") and text.endswith(")"):
        return text[1:-1].strip()
    return text


def _find_boundary_callsite_replacements(content: str, boundaries: List[object]) -> List[Tuple[int, int, str]]:
    if not boundaries:
        return []
    code = content.encode("utf-8", errors="ignore")
    tree = C_PARSER.parse(code)
    by_line_expr: Dict[Tuple[int, str], object] = {}
    for boundary in boundaries:
        by_line_expr[(int(getattr(boundary, "source_line", 0) or 0), getattr(boundary, "expression", ""))] = boundary

    replacements: List[Tuple[int, int, str]] = []
    for node in _walk_tree(tree.root_node):
        if node.type != "call_expression":
            continue
        line = node.start_point[0] + 1
        expression = _callee_expression_from_call(node, code)
        boundary = by_line_expr.get((line, expression))
        if boundary is None:
            continue
        names = _boundary_symbol_names(boundary.candidate_id)
        args = _call_arguments_text(node, code)
        replacement = f"{names['invoke']}({args})"
        replacements.append((node.start_byte, node.end_byte, replacement))
    return replacements


def _walk_tree(node):
    stack = [node]
    while stack:
        current = stack.pop()
        yield current
        stack.extend(reversed(current.children))


def _apply_byte_replacements(content: str, replacements: List[Tuple[int, int, str]]) -> str:
    if not replacements:
        return content
    data = content.encode("utf-8", errors="ignore")
    for start, end, replacement in sorted(replacements, key=lambda item: item[0], reverse=True):
        data = data[:start] + replacement.encode("utf-8") + data[end:]
    return data.decode("utf-8", errors="ignore")


def _boundary_decl_block(boundaries: List[object], config_symbol: str) -> Tuple[str, List[ExportInterface]]:
    if not boundaries:
        return "", []
    lines = [BOUNDARY_DECLS_BEGIN + "\n", f"#if IS_ENABLED({config_symbol})\n"]
    interfaces: List[ExportInterface] = []
    for boundary in boundaries:
        original = getattr(boundary, "expression", "")
        names = _boundary_symbol_names(boundary.candidate_id)
        lines.append(
            f"/* RACA_BOUNDARY_HOOK_DECL: boundary={boundary.candidate_id}; original={original}; invoke={names['invoke']} */\n"
        )
        lines.append(f"extern typeof({original}) *{names['hook']};\n")
        lines.append(f"extern unsigned long {names['call_count']};\n")
        lines.append(
            f"#define {names['invoke']}(...) \\\n"
            "({ \\\n"
            f"    {names['call_count']}++; \\\n"
            f"    {names['hook']} ? {names['hook']}(__VA_ARGS__) : {original}(__VA_ARGS__); \\\n"
            "})\n"
        )
    lines.append(f"#endif /* IS_ENABLED({config_symbol}) */\n")
    lines.append(BOUNDARY_DECLS_END + "\n")
    return "".join(lines), interfaces


def _boundary_hook_export_block(boundaries: List[object], export_gpl: bool) -> Tuple[str, List[ExportInterface]]:
    if not boundaries:
        return "", []
    export_macro = "EXPORT_SYMBOL_GPL" if export_gpl else "EXPORT_SYMBOL"
    blocks: List[str] = ["/* RACA direct-boundary hook controls. */\n"]
    interfaces: List[ExportInterface] = []
    for boundary in boundaries:
        original = getattr(boundary, "expression", "")
        names = _boundary_symbol_names(boundary.candidate_id)
        set_proto = f"extern void {names['set_hook']}(typeof({original}) *hook);"
        clear_proto = f"extern void {names['clear_hook']}(void);"
        count_proto = f"extern unsigned long {names['get_call_count']}(void);"
        blocks.extend(
            [
                f"typeof({original}) *{names['hook']};\n",
                f"unsigned long {names['call_count']};\n",
                f"void {names['set_hook']}(typeof({original}) *hook);\n",
                f"void {names['clear_hook']}(void);\n",
                f"unsigned long {names['get_call_count']}(void);\n\n",
                f"void {names['set_hook']}(typeof({original}) *hook)\n",
                "{\n",
                f"    {names['hook']} = hook;\n",
                "}\n",
                f"{export_macro}({names['set_hook']});\n\n",
                f"void {names['clear_hook']}(void)\n",
                "{\n",
                f"    {names['hook']} = NULL;\n",
                f"    {names['call_count']} = 0;\n",
                "}\n",
                f"{export_macro}({names['clear_hook']});\n\n",
                f"unsigned long {names['get_call_count']}(void)\n",
                "{\n",
                f"    return {names['call_count']};\n",
                "}\n",
                f"{export_macro}({names['get_call_count']});\n\n",
            ]
        )
        for role, proto, description in (
            ("set_hook", set_proto, f"Install fake implementation for direct boundary {boundary.candidate_id} ({original})"),
            ("clear_hook", clear_proto, f"Clear fake implementation and reset call count for {boundary.candidate_id}"),
            ("get_call_count", count_proto, f"Read production-path call count for {boundary.candidate_id}"),
        ):
            interfaces.append(
                ExportInterface(
                    prototype=proto,
                    source_symbol=boundary.candidate_id,
                    source_kind="boundary_hook",
                    description=description,
                    boundary_id=boundary.candidate_id,
                    boundary_expression=original,
                    boundary_control_role=role,
                )
            )
    return "".join(blocks), interfaces


def _insert_boundary_decl_block(content: str, decl_block: str, boundaries: List[object], tree) -> str:
    if not decl_block or not boundaries:
        return content
    function_names = {getattr(boundary, "source_function", "") for boundary in boundaries}
    insert_byte = None
    code = content.encode("utf-8", errors="ignore")
    for node in _walk_tree(tree.root_node):
        if node.type != "function_definition":
            continue
        name = extract_function_name_from_def(node, code)
        if name in function_names:
            insert_byte = node.start_byte if insert_byte is None else min(insert_byte, node.start_byte)
    if insert_byte is None:
        return content
    data = content.encode("utf-8", errors="ignore")
    data = data[:insert_byte] + (decl_block + "\n").encode("utf-8") + data[insert_byte:]
    return data.decode("utf-8", errors="ignore")


# -----------------------------
# Section management
# -----------------------------

def _find_section_range(content: str) -> Optional[Tuple[int, int]]:
    """
    Return (start_idx, end_idx) for the entire stub section (including markers),
    or None if not found.
    """
    start = content.find(SECTION_BEGIN)
    if start < 0:
        return None
    end = content.find(SECTION_END, start)
    if end < 0:
        return None
    end += len(SECTION_END)
    return (start, end)

def _ensure_section_exists(content: str, config_symbol: str) -> str:
    """
    Ensure KUnit stub section exists. If not, append it at EOF (safe, style-agnostic).
    """
    rng = _find_section_range(content)
    if rng is None:
        guard_body = _ensure_guard_wrapped_body("", config_symbol)
        section_text = _rebuild_section(guard_body)
        if not content.endswith("\n"):
            content += "\n"
        content += "\n" + section_text
        return content

    sec_start, sec_end = rng
    section_text = content[sec_start:sec_end]
    body = _extract_section_body(section_text)
    ensured_body = _ensure_guard_wrapped_body(body, config_symbol)
    if ensured_body == body:
        return content

    new_section_text = _rebuild_section(ensured_body)
    return content[:sec_start] + new_section_text + content[sec_end:]


def _extract_section_body(section_text: str) -> str:
    """
    section_text includes SECTION_BEGIN + body + SECTION_END
    Return body (text between them).
    """
    if not section_text.startswith(SECTION_BEGIN) or not section_text.endswith(SECTION_END):
        raise ValueError("Invalid section text boundaries.")
    body = section_text[len(SECTION_BEGIN):]
    body = body[: -len(SECTION_END)]
    return body

def _rebuild_section(body: str) -> str:
    # Normalize: ensure exactly one leading/trailing newline around body
    body = body.strip("\n")
    if body:
        body = "\n" + body + "\n"
    else:
        body = "\n"
    return SECTION_BEGIN + body + SECTION_END


def _ensure_guard_wrapped_body(body: str, config_symbol: str) -> str:
    """
    Ensure the section body contains a single #if/#endif guard around the stubs.
    """
    match = SECTION_GUARD_PATTERN.search(body)
    if match:
        existing = match.group("symbol")
        if existing != config_symbol:
            raise ValueError(
                f"Stub section already guarded by {existing}, expected {config_symbol}"
            )
        return body

    inner = body.strip("\n")
    lines = [f"#if IS_ENABLED({config_symbol})\n"]
    if inner:
        lines.append("\n")
        lines.append(inner)
        if not inner.endswith("\n"):
            lines.append("\n")
    lines.append(f"#endif /* IS_ENABLED({config_symbol}) */\n")
    return "".join(lines)


def _split_guard_content(body: str) -> Tuple[str, str, str]:
    """
    Split guard-wrapped body into (prefix, payload, suffix).
    Prefix contains the #if line (including trailing newline), payload contains
    the stub text, and suffix contains the trailing #endif comment.
    """
    match = SECTION_GUARD_PATTERN.search(body)
    if not match:
        raise ValueError("Missing guard in stub section.")

    guard_line_end = body.find("\n", match.end())
    if guard_line_end < 0:
        raise ValueError("Malformed guard line in stub section.")
    prefix = body[: guard_line_end + 1]

    suffix_start = body.rfind("#endif")
    if suffix_start < 0:
        raise ValueError("Missing #endif for stub section guard.")
    suffix = body[suffix_start:]

    payload = body[guard_line_end + 1 : suffix_start]
    return prefix, payload, suffix


def _normalize_payload(payload: str) -> str:
    payload = payload.strip("\n")
    if payload:
        return "\n" + payload + "\n"
    return "\n"

def _replace_or_append_stub(body: str, func_name: str, stub_block: str) -> str:
    """
    Replace existing stub block for func_name if present, else append.
    """
    prefix, payload, suffix = _split_guard_content(body)
    begin = stub_begin_marker(func_name)
    end = stub_end_marker(func_name)

    if begin in payload and end in payload:
        # Replace that block (non-greedy across any content)
        pattern = re.compile(re.escape(begin) + r".*?" + re.escape(end) + r"\n?", re.DOTALL)
        payload = pattern.sub(stub_block, payload)
        return prefix + payload + suffix

    # Append at end of body
    payload = payload.rstrip()
    insertion = stub_block.rstrip() + "\n"
    if payload:
        payload = payload + "\n\n" + insertion
    else:
        payload = "\n" + insertion
    return prefix + payload + suffix

def _remove_all_stubs(body: str) -> str:
    """
    Remove all stub blocks within the section body.
    """
    prefix, payload, suffix = _split_guard_content(body)
    pattern = re.compile(r"/\*\s*KUNIT_STUB_BEGIN:.*?\*/.*?/\*\s*KUNIT_STUB_END:.*?\*/\s*\n?",
                         re.DOTALL)
    payload = pattern.sub("", payload)
    payload = _normalize_payload(payload)
    return prefix + payload + suffix

def _remove_stub(body: str, func_name: str) -> str:
    begin = stub_begin_marker(func_name)
    end = stub_end_marker(func_name)
    prefix, payload, suffix = _split_guard_content(body)
    if begin not in payload or end not in payload:
        return body
    pattern = re.compile(re.escape(begin) + r".*?" + re.escape(end) + r"\n?", re.DOTALL)
    payload = pattern.sub("", payload)
    payload = _normalize_payload(payload)
    return prefix + payload + suffix


# -----------------------------
# Public API
# -----------------------------

def apply_stub(spec: StubSpec) -> StubGenerationResult:
    """
    Generate + insert/update stub for spec.target_func_name in spec.driver_c_path.

    Returns a StubGenerationResult containing the updated file content and
    metadata about exported interfaces for tests.
    """
    content = remove_generated_boundary_instrumentation(_read_text(spec.driver_c_path))
    content = _ensure_section_exists(content, spec.config_symbol)

    code_bytes = content.encode("utf-8", errors="ignore")
    tree = C_PARSER.parse(code_bytes)

    func_node = find_function_definition_node(tree, spec.target_func_name, code_bytes)
    if func_node is None:
        raise ValueError(f"Function definition not found: {spec.target_func_name}")

    param_list_text, arg_names = extract_param_list_text_and_names(func_node, code_bytes)
    return_type = extract_return_type_text(func_node, code_bytes, spec.target_func_name)

    export_name = f"{spec.target_func_name}{spec.export_suffix}"
    boundary_candidates = []
    if spec.enable_boundary_hooks:
        boundary_candidates = _direct_boundary_candidates_for_stub(spec, target_info=next(
            (f for f in getattr(spec.parse_result, "functions", []) or [] if f.name == spec.target_func_name),
            None,
        ) or type("_Target", (), {"name": spec.target_func_name, "code": _slice(code_bytes, func_node), "start_line": func_node.start_point[0] + 1, "end_line": func_node.end_point[0] + 1, "calls": []})())
    replacements = _find_boundary_callsite_replacements(content, boundary_candidates)
    if replacements:
        content = _apply_byte_replacements(content, replacements)
        decl_block, _ = _boundary_decl_block(boundary_candidates, spec.config_symbol)
        content = _insert_boundary_decl_block(content, decl_block, boundary_candidates, tree)

    context_text, context_interfaces = _build_related_context_block(
        func_name=spec.target_func_name,
        spec=spec,
        parse_result=spec.parse_result,
        tree=tree,
        code_bytes=code_bytes,
    )
    boundary_hook_text, boundary_hook_interfaces = _boundary_hook_export_block(
        boundary_candidates if replacements else [],
        spec.export_gpl,
    )
    if boundary_hook_text:
        if context_text and not context_text.endswith("\n"):
            context_text += "\n"
        context_text = boundary_hook_text + ("\n" + context_text if context_text else "")
        context_interfaces = boundary_hook_interfaces + context_interfaces
    context_block = None
    if context_text:
        context_block = (context_text, context_interfaces)
    stub_block, interfaces = build_stub_block(
        orig_func_name=spec.target_func_name,
        export_func_name=export_name,
        return_type=return_type,
        param_list_text=param_list_text,
        arg_names=arg_names,
        export_gpl=spec.export_gpl,
        source_kind="function",
        context_block=context_block,
    )

    # Update section content
    rng = _find_section_range(content)
    assert rng is not None
    sec_start, sec_end = rng
    section_text = content[sec_start:sec_end]
    body = _extract_section_body(section_text)

    if spec.mode not in ("incremental", "single"):
        raise ValueError("spec.mode must be 'incremental' or 'single'")

    if spec.mode == "single":
        # keep only this one stub in the section
        body = _remove_all_stubs(body)
        body = _replace_or_append_stub(body, spec.target_func_name, stub_block)
    else:
        # incremental: replace/append just this one
        body = _replace_or_append_stub(body, spec.target_func_name, stub_block)

    new_section_text = _rebuild_section(body)
    new_content = content[:sec_start] + new_section_text + content[sec_end:]

    _write_text(spec.driver_c_path, new_content)
    return StubGenerationResult(file_content=new_content, export_interfaces=interfaces)

def remove_stub(driver_c_path: str, func_name: str) -> None:
    """
    Remove a specific function stub from the stub section (if exists).
    """
    content = _read_text(driver_c_path)
    rng = _find_section_range(content)
    if rng is None:
        return

    sec_start, sec_end = rng
    section_text = content[sec_start:sec_end]
    body = _extract_section_body(section_text)
    body = _remove_stub(body, func_name)
    new_section_text = _rebuild_section(body)
    new_content = content[:sec_start] + new_section_text + content[sec_end:]
    _write_text(driver_c_path, new_content)

def ensure_stub_section(driver_c_path: str, config_symbol: str) -> None:
    """
    Ensure the stub section exists (no-op if already exists).
    """
    content = _read_text(driver_c_path)
    new_content = _ensure_section_exists(content, config_symbol)
    if new_content != content:
        _write_text(driver_c_path, new_content)
