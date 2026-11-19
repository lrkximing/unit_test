import re
from typing import Dict, Iterable, List, Optional, Set

from validation.test_inspector import inspect_test_source
from verification.kunit_binding_extractor import KunitBinding, collect_kunit_bindings
from verification.observation_provenance import local_observations, observed_target


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


def _assertion_value_args(binding):
    args = _split_call_args(binding.statement_text or "")[1:]
    suffix = _macro_suffix(binding.macro or "").removesuffix("_MSG")
    if suffix.endswith(("_TRUE", "_FALSE", "_NULL", "_ERR_OR_NULL")):
        return args[:1]
    # Message/format arguments are diagnostics, not oracle operands.
    return args[:2]


def binding_is_nontrivial_assertion(binding: KunitBinding) -> bool:
    macro = binding.macro or ""
    if not (macro.startswith("KUNIT_EXPECT_") or macro.startswith("KUNIT_ASSERT_")):
        return False
    suffix = _macro_suffix(macro).removesuffix("_MSG")
    if suffix in WEAK_ASSERTION_SUFFIXES:
        return False
    value_args = _assertion_value_args(binding)
    if not value_args:
        return False
    if suffix.endswith("_TRUE") and _literal_true(value_args[0]):
        return False
    if suffix.endswith("_FALSE") and _literal_false(value_args[0]):
        return False
    if len(value_args) >= 2 and _normalize_expr(value_args[0]) == _normalize_expr(value_args[1]):
        return False
    return True


def binding_is_effective_for_check(binding: KunitBinding, check: Dict, wrapper: str = "") -> bool:
    if not binding_is_nontrivial_assertion(binding):
        return False
    expected = str((check or {}).get("expected_relation", "") or "").lower()
    if re.fullmatch(r"(non[- ]null\s+)?allocation(\s+check)?", expected.strip()):
        return False
    args = _assertion_value_args(binding)
    evidence = local_observations(binding, wrapper) if wrapper else None
    if evidence and len(args) == 2:
        identities = evidence.get("value_identity", {})
        left, right = (_normalize_expr(a) for a in args)
        from verification.object_relations import resolve_path
        if resolve_path(left, identities) == resolve_path(right, identities):
            return False
    if wrapper and not observed_target(binding, check, wrapper, args):
        return False
    if check.get("kind") in {"ReturnComputation", "BoundaryArgument"}:
        if not re.fullmatch(r"KUNIT_(?:EXPECT|ASSERT)_(?:EQ|PTR_EQ|MEMEQ|STREQ)(?:_MSG)?", binding.macro):
            return False
        if "MEMEQ" in binding.macro:
            operands = _split_call_args(binding.statement_text or "")[1:]
            if len(operands) < 3 or _literal_false(operands[2]):
                return False
    if str(check.get("kind", "")).startswith("Return") and evidence and len(args) == 2:
        # A second target execution is not an independent exact-value oracle.
        def from_target(expression):
            identifiers = set(re.findall(r"\b[A-Za-z_]\w*\b", expression))
            return wrapper in identifiers or bool(identifiers & evidence["returns"])
        if all(from_target(argument) for argument in args):
            return False
    if str(check.get("kind", "")).startswith("ReturnEquals"):
        relation = str(check.get("expected_relation", ""))
        boolean_match = (relation == "equals true" and re.fullmatch(r"KUNIT_(?:EXPECT|ASSERT)_TRUE(?:_MSG)?", binding.macro) or
                         relation == "equals false" and re.fullmatch(r"KUNIT_(?:EXPECT|ASSERT)_FALSE(?:_MSG)?", binding.macro))
        if not boolean_match and not re.fullmatch(r"KUNIT_(?:EXPECT|ASSERT)_EQ(?:_MSG)?", binding.macro):
            return False
    if str(check.get("kind", "")).startswith("Return") and str(check.get("expected_relation", "")).startswith("equals "):
        literal = str(check["expected_relation"])[7:].strip()
        if re.fullmatch(r"-?\d+", literal):
            # An exact, statically supplied return value must not become a broad inequality.
            evidence = local_observations(binding, wrapper)
            actual_literals = [str((evidence or {}).get("literals", {}).get(a, a)) for a in args[:2]]
            numeric = [int(a) for a in actual_literals if re.fullmatch(r"-?\d+", a)]
            if numeric and int(literal) not in numeric:
                return False
    return True


def effective_check_ids(bindings: Iterable[KunitBinding], checks_by_id: Dict[str, Dict], wrapper: str = "") -> Set[str]:
    ids: Set[str] = set()
    for binding in bindings:
        for check_id in binding.check_ids:
            if binding_is_effective_for_check(binding, checks_by_id.get(check_id, {}), wrapper):
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
