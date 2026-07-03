import re
from dataclasses import dataclass, field
from typing import List

try:
    from tree_sitter_languages import get_parser
except ImportError:
    get_parser = None


C_PARSER = get_parser("c") if get_parser is not None else None
CHECK_MARKER_PATTERN = re.compile(r"RACA_CHECK\s*:\s*([A-Za-z0-9_]+)")
WITNESS_MARKER_PATTERN = re.compile(r"RACA_WITNESS\s*:\s*([A-Za-z0-9_]+)")
EFFECT_MARKER_PATTERN = re.compile(r"RACA_EFFECT\s*:\s*([A-Za-z0-9_]+)")
KUNIT_STATEMENT_PATTERN = re.compile(r"\b(?:KUNIT_(?:EXPECT|ASSERT)_[A-Z0-9_]+|kunit_info)\s*\(")


@dataclass
class KunitBinding:
    macro: str
    start_line: int
    end_line: int
    statement_text: str = ""
    check_ids: List[str] = field(default_factory=list)
    witness_ids: List[str] = field(default_factory=list)
    effect_ids: List[str] = field(default_factory=list)


def _node_text(code: bytes, node) -> str:
    return code[node.start_byte:node.end_byte].decode("utf-8", errors="ignore")


def _walk(node):
    stack = [node]
    while stack:
        current = stack.pop()
        yield current
        stack.extend(reversed(current.children))


def _callee_name(function_node, code: bytes) -> str:
    if function_node.type == "identifier":
        return _node_text(code, function_node)
    for child in function_node.children:
        if child.is_named:
            name = _callee_name(child, code)
            if name:
                return name
    return ""


def _comment_markers(code_text: str):
    if C_PARSER is None:
        comments = []
        for idx, line in enumerate(code_text.splitlines(), start=1):
            comments.append(
                {
                    "line": idx,
                    "checks": CHECK_MARKER_PATTERN.findall(line),
                    "witnesses": WITNESS_MARKER_PATTERN.findall(line),
                    "effects": EFFECT_MARKER_PATTERN.findall(line),
                }
            )
        return comments
    code = code_text.encode("utf-8", errors="ignore")
    tree = C_PARSER.parse(code)
    comments = []
    for node in _walk(tree.root_node):
        if node.type != "comment":
            continue
        text = _node_text(code, node)
        comments.append(
            {
                "line": node.end_point[0] + 1,
                "checks": CHECK_MARKER_PATTERN.findall(text),
                "witnesses": WITNESS_MARKER_PATTERN.findall(text),
                "effects": EFFECT_MARKER_PATTERN.findall(text),
            }
        )
    return comments


def _enclosing_function_start_line(node) -> int:
    current = node
    while current is not None:
        if current.type == "function_definition":
            return current.start_point[0] + 1
        current = getattr(current, "parent", None)
    return 1


def _nearby_markers(comments, statement_line: int, min_line: int = 1, max_distance: int = 12):
    checks: List[str] = []
    witnesses: List[str] = []
    effects: List[str] = []
    for comment in comments:
        if comment["line"] < min_line:
            continue
        distance = statement_line - comment["line"]
        if distance < 0 or distance > max_distance:
            continue
        checks.extend(comment["checks"])
        witnesses.extend(comment["witnesses"])
        effects.extend(comment["effects"])
    return sorted(set(checks)), sorted(set(witnesses)), sorted(set(effects))


def _merge_marker_ids(existing: List[str], text: str, pattern: re.Pattern) -> List[str]:
    ids = set(existing)
    ids.update(pattern.findall(text or ""))
    return sorted(ids)


def collect_kunit_bindings(code_text: str) -> List[KunitBinding]:
    if C_PARSER is None:
        return _collect_kunit_bindings_fallback(code_text or "")
    code = (code_text or "").encode("utf-8", errors="ignore")
    tree = C_PARSER.parse(code)
    comments = _comment_markers(code_text or "")
    bindings: List[KunitBinding] = []
    for node in _walk(tree.root_node):
        if node.type != "call_expression":
            continue
        function_node = node.child_by_field_name("function")
        if function_node is None:
            continue
        macro = _callee_name(function_node, code)
        if (
            not macro.startswith("KUNIT_EXPECT_")
            and not macro.startswith("KUNIT_ASSERT_")
            and macro != "kunit_info"
        ):
            continue
        start_line = node.start_point[0] + 1
        min_line = _enclosing_function_start_line(node)
        check_ids, witness_ids, effect_ids = _nearby_markers(comments, start_line, min_line=min_line)
        statement_text = _node_text(code, node)
        check_ids = _merge_marker_ids(check_ids, statement_text, CHECK_MARKER_PATTERN)
        witness_ids = _merge_marker_ids(witness_ids, statement_text, WITNESS_MARKER_PATTERN)
        effect_ids = _merge_marker_ids(effect_ids, statement_text, EFFECT_MARKER_PATTERN)
        bindings.append(
            KunitBinding(
                macro=macro,
                start_line=start_line,
                end_line=node.end_point[0] + 1,
                statement_text=statement_text,
                check_ids=check_ids,
                witness_ids=witness_ids,
                effect_ids=effect_ids,
            )
        )
    return bindings


def _collect_kunit_bindings_fallback(code_text: str) -> List[KunitBinding]:
    comments = _comment_markers(code_text)
    bindings: List[KunitBinding] = []
    for idx, line in enumerate(code_text.splitlines(), start=1):
        match = KUNIT_STATEMENT_PATTERN.search(line)
        if not match:
            continue
        macro = line[match.start() :].split("(", 1)[0].strip()
        check_ids, witness_ids, effect_ids = _nearby_markers(comments, idx)
        check_ids = _merge_marker_ids(check_ids, line, CHECK_MARKER_PATTERN)
        witness_ids = _merge_marker_ids(witness_ids, line, WITNESS_MARKER_PATTERN)
        effect_ids = _merge_marker_ids(effect_ids, line, EFFECT_MARKER_PATTERN)
        bindings.append(
            KunitBinding(
                macro=macro,
                start_line=idx,
                end_line=idx,
                statement_text=line.strip(),
                check_ids=check_ids,
                witness_ids=witness_ids,
                effect_ids=effect_ids,
            )
        )
    return bindings
