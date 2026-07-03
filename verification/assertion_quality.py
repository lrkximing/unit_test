import re
from typing import Dict, Iterable, List, Optional, Set

from validation.test_inspector import inspect_test_source
from verification.kunit_binding_extractor import KunitBinding, collect_kunit_bindings


WEAK_ASSERTION_SUFFIXES = {
    "ASSERT_NOT_NULL",
    "EXPECT_NOT_NULL",
    "ASSERT_NOT_ERR_OR_NULL",
    "EXPECT_NOT_ERR_OR_NULL",
}


def _macro_suffix(macro: str) -> str:
    return macro[len("KUNIT_") :] if macro.startswith("KUNIT_") else macro


def _split_call_args(statement: str) -> List[str]:
    start = statement.find("(")
    end = statement.rfind(")")
    if start < 0 or end <= start:
        return []
    text = statement[start + 1 : end]
    args: List[str] = []
    current: List[str] = []
    depth = 0
    in_string = False
    in_char = False
    escaped = False
    for ch in text:
        if escaped:
            current.append(ch)
            escaped = False
            continue
        if ch == "\\":
            current.append(ch)
            escaped = True
            continue
        if ch == '"' and not in_char:
            in_string = not in_string
        elif ch == "'" and not in_string:
            in_char = not in_char
        elif not in_string and not in_char:
            if ch in "([{":
                depth += 1
            elif ch in ")]}" and depth > 0:
                depth -= 1
            elif ch == "," and depth == 0:
                args.append("".join(current).strip())
                current = []
                continue
        current.append(ch)
    tail = "".join(current).strip()
    if tail:
        args.append(tail)
    return args


def _normalize_expr(expr: str) -> str:
    expr = re.sub(r"/\*[\s\S]*?\*/", " ", expr or "")
    expr = re.sub(r"//[^\n\r]*", " ", expr)
    expr = re.sub(r"\s+", "", expr)
    while expr.startswith("(") and expr.endswith(")"):
        depth = 0
        outer = True
        for idx, ch in enumerate(expr):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0 and idx != len(expr) - 1:
                    outer = False
                    break
        if not outer or depth != 0:
            break
        expr = expr[1:-1]
    return expr


def _literal_true(expr: str) -> bool:
    return _normalize_expr(expr).lower() in {"1", "true", "!!1"}


def _literal_false(expr: str) -> bool:
    return _normalize_expr(expr).lower() in {"0", "false", "null", "!!0"}


def binding_is_nontrivial_assertion(binding: KunitBinding) -> bool:
    macro = binding.macro or ""
    if not (macro.startswith("KUNIT_EXPECT_") or macro.startswith("KUNIT_ASSERT_")):
        return False
    suffix = _macro_suffix(macro)
    if suffix in WEAK_ASSERTION_SUFFIXES:
        return False
    args = _split_call_args(binding.statement_text or "")
    value_args = args[1:] if args and args[0] == "test" else args
    if not value_args:
        return False
    if suffix.endswith("_TRUE") and _literal_true(value_args[0]):
        return False
    if suffix.endswith("_FALSE") and _literal_false(value_args[0]):
        return False
    if len(value_args) >= 2 and _normalize_expr(value_args[0]) == _normalize_expr(value_args[1]):
        return False
    return True


def binding_is_effective_for_check(binding: KunitBinding, check: Dict) -> bool:
    if not binding_is_nontrivial_assertion(binding):
        return False
    expected = str((check or {}).get("expected_relation", "") or "").lower()
    if re.fullmatch(r"(non[- ]null\s+)?allocation(\s+check)?", expected.strip()):
        return False
    return True


def effective_check_ids(bindings: Iterable[KunitBinding], checks_by_id: Dict[str, Dict]) -> Set[str]:
    ids: Set[str] = set()
    for binding in bindings:
        for check_id in binding.check_ids:
            if binding_is_effective_for_check(binding, checks_by_id.get(check_id, {})):
                ids.add(check_id)
    return ids


def nontrivial_assertion_tests(test_code: str, selected_tests: Optional[Iterable[str]] = None) -> List[str]:
    selected = {item for item in (selected_tests or []) if isinstance(item, str) and item}
    result: List[str] = []
    for test_function in inspect_test_source(test_code or "").test_functions:
        if selected and test_function.name not in selected:
            continue
        bindings = collect_kunit_bindings(test_function.full_text)
        if any(binding_is_nontrivial_assertion(binding) for binding in bindings):
            result.append(test_function.name)
    return result
