"""Conservative, source-based checks of KUnit assertions against scenario checks.

Only relations that can be established from the test body and the scenario
contract are accepted as proven. Unresolved C expressions remain inconclusive;
they are never treated as evidence that an oracle is correct.
"""

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from validation.test_inspector import inspect_test_source
from verification.assertion_quality import (
    _normalize_expr,
    _split_call_args,
    binding_is_nontrivial_assertion,
)
from verification.kunit_binding_extractor import (
    CHECK_MARKER_PATTERN,
    KunitBinding,
    collect_kunit_bindings,
)


_TARGET_RETURN = "<target-wrapper-return>"
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_FIELD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:(?:->|\.)[A-Za-z_][A-Za-z0-9_]*)+\Z")
_INTEGER = re.compile(r"([+-]?(?:0[xX][0-9a-fA-F]+|[0-9]+))[uUlL]*\Z")
_ASSIGNMENT = re.compile(r"(?<![=!<>])=(?!=)")
_LEFT_NAME = re.compile(
    r"([A-Za-z_][A-Za-z0-9_]*(?:\s*(?:->|\.)\s*[A-Za-z_][A-Za-z0-9_]*)*)\s*\Z"
)
_COMPARISON = re.compile(r"^(.+?)\s*(==|!=|>=|<=|>|<)\s*(.+)$", re.DOTALL)
_REVERSED = {"==": "==", "!=": "!=", ">": "<", "<": ">", ">=": "<=", "<=": ">="}
_NEGATED = {"==": "!=", "!=": "==", ">": "<=", "<": ">=", ">=": "<", "<=": ">="}
_MACRO_OPERATORS = {
    "EQ": "==",
    "NE": "!=",
    "GT": ">",
    "LT": "<",
    "GE": ">=",
    "LE": "<=",
    "STREQ": "==",
    "STRNEQ": "!=",
}


@dataclass(frozen=True)
class OracleAssessment:
    status: str  # proven, mismatch, or unresolved
    reason: str = ""


@dataclass(frozen=True)
class AssertionRelation:
    left: str
    operator: str
    right: str


def _without_comments(text: str) -> str:
    def blank(match):
        return "".join("\n" if char == "\n" else " " for char in match.group(0))

    text = re.sub(r"/\*[\s\S]*?\*/", blank, text or "")
    return re.sub(r"//[^\n]*", blank, text)


def _assignments(test_body: str) -> List[Tuple[int, str, str]]:
    """Record simple local assignments in source order, without assuming branches run."""
    code = _without_comments(test_body)
    result: List[Tuple[int, str, str]] = []
    for statement in re.finditer(r"[^;{}]+;", code, re.DOTALL):
        source = statement.group(0)[:-1].strip()
        if not source or re.match(r"^(?:if|for|while|switch|return)\b", source):
            continue
        assign = _ASSIGNMENT.search(source)
        if assign is None:
            continue
        left = _LEFT_NAME.search(source[: assign.start()].strip())
        if left is None:
            continue
        line = code.count("\n", 0, statement.start()) + 1
        result.append((line, _normalize_expr(left.group(1)), source[assign.end() :].strip()))
    return result


def _resolve(
    expression: str,
    assignments: List[Tuple[int, str, str]],
    before_line: int,
    wrapper: str,
    seen: Optional[set] = None,
) -> str:
    value = _normalize_expr(expression)
    if not value:
        return value
    if wrapper and re.fullmatch(r"\b" + re.escape(wrapper) + r"\(.*\)", value):
        return _TARGET_RETURN
    if value.startswith("-") and _IDENTIFIER.fullmatch(value[1:]):
        resolved = _resolve(value[1:], assignments, before_line, wrapper, seen)
        return "-" + resolved if resolved != value[1:] else value
    if _FIELD.fullmatch(value):
        # The target may write a field after setup. A prior setup assignment is
        # not the value observed by a post-call assertion.
        return value
    if not _IDENTIFIER.fullmatch(value):
        return value
    seen = seen or set()
    if value in seen:
        return value
    for line, target, source in reversed(assignments):
        if target == value and line < before_line:
            if _FIELD.fullmatch(_normalize_expr(source)) and wrapper:
                wrapper_lines = [
                    call_line for call_line, _, rhs in assignments
                    if re.search(r"\b" + re.escape(wrapper) + r"\s*\(", rhs)
                ]
                if wrapper_lines and line < min(wrapper_lines):
                    return "<pre-call-state>"
            return _resolve(source, assignments, line, wrapper, seen | {value})
    return value


def _known_value(expression: str) -> Optional[Tuple[str, object]]:
    value = _normalize_expr(expression)
    match = _INTEGER.fullmatch(value)
    if match:
        try:
            return ("integer", int(match.group(1), 0))
        except ValueError:
            return None
    if re.fullmatch(r"-?[A-Z][A-Z0-9_]*", value):
        return ("symbol", value)
    return None


def _is_negative(expression: str) -> Optional[bool]:
    known = _known_value(expression)
    if known is None:
        return None
    if known[0] == "integer":
        return known[1] < 0
    if known[0] == "symbol" and str(known[1]).startswith("-E"):
        return True
    return None


def _assertion_relation(binding: KunitBinding) -> Optional[AssertionRelation]:
    macro = binding.macro or ""
    if not (macro.startswith("KUNIT_EXPECT_") or macro.startswith("KUNIT_ASSERT_")):
        return None
    suffix = macro[len("KUNIT_") :]
    if suffix not in _MACRO_OPERATORS and suffix.rsplit("_", 1)[-1] in _MACRO_OPERATORS:
        suffix = suffix.rsplit("_", 1)[-1]
    args = _split_call_args(binding.statement_text)
    values = args[1:] if args and args[0].strip() == "test" else args
    if suffix in _MACRO_OPERATORS and len(values) >= 2:
        return AssertionRelation(values[0], _MACRO_OPERATORS[suffix], values[1])
    if suffix.endswith("_TRUE") or suffix.endswith("_FALSE"):
        if not values:
            return None
        condition = _normalize_expr(values[0])
        comparison = _COMPARISON.fullmatch(condition)
        if comparison:
            operator = comparison.group(2)
            if suffix.endswith("_FALSE"):
                operator = _NEGATED[operator]
            return AssertionRelation(comparison.group(1), operator, comparison.group(3))
        return AssertionRelation(values[0], "!=" if suffix.endswith("_TRUE") else "==", "0")
    return None


def direct_check_ids(binding: KunitBinding, test_text: str) -> List[str]:
    """Read the marker next to this assertion, not every marker in a 12-line window."""
    inline = CHECK_MARKER_PATTERN.findall(binding.statement_text)
    if inline:
        return sorted(set(inline))
    lines = test_text.splitlines()
    if 0 < binding.start_line <= len(lines):
        same_line = lines[binding.start_line - 1]
        prefix = same_line.split(binding.macro + "(", 1)[0]
        inline = CHECK_MARKER_PATTERN.findall(prefix)
        if inline:
            return sorted(set(inline))
    found = set()
    index = binding.start_line - 2
    while index >= 0:
        line = lines[index].strip()
        if not line:
            index -= 1
            continue
        if line.startswith(("/*", "*", "//")) or line.endswith("*/"):
            found.update(CHECK_MARKER_PATTERN.findall(line))
            index -= 1
            continue
        break
    return sorted(found)


def _expected_from_contract(check: Dict) -> Optional[str]:
    relation = str(check.get("expected_relation", "") or "").strip()
    match = re.match(r"^equals\s+(.+)$", relation, flags=re.IGNORECASE)
    if not match:
        return None
    value = re.split(r"\s+when\s+|;", match.group(1), maxsplit=1)[0].strip()
    return value or None


def _configured_mock_return(
    check: Dict, full_code: str, assignments: List[Tuple[int, str, str]],
    before_line: int, wrapper: str,
) -> Optional[str]:
    """Follow an explicit RACA_MOCK binding to a simple configured fake return."""
    relation = str(check.get("expected_relation", "") or "")
    boundary = re.search(r"\bproduced by\s+([A-Za-z0-9_]+)", relation)
    if boundary is None or not full_code:
        return None
    marker = re.search(
        r"RACA_MOCK\s*:\s*boundary=" + re.escape(boundary.group(1))
        + r"\s*;\s*original=[^;]+\s*;\s*replacement=([A-Za-z_][A-Za-z0-9_]*)",
        full_code,
    )
    if marker is None:
        return None
    fake = marker.group(1)
    definition = re.search(
        r"\b" + re.escape(fake) + r"\s*\([^;{}]*\)\s*\{", _without_comments(full_code)
    )
    if definition is None:
        return None
    code = _without_comments(full_code)
    depth = 1
    end = definition.end()
    while end < len(code) and depth:
        depth += (code[end] == "{") - (code[end] == "}")
        end += 1
    if depth:
        return None
    returns = re.findall(r"\breturn\s+([^;{}]+);", code[definition.end():end])
    if len(returns) != 1:
        return None
    return _resolve(returns[0], assignments, before_line, wrapper)


def _target_side(
    relation: AssertionRelation,
    assignments: List[Tuple[int, str, str]],
    line: int,
    wrapper: str,
    target: str,
) -> Tuple[Optional[str], str, str]:
    left = _resolve(relation.left, assignments, line, wrapper)
    right = _resolve(relation.right, assignments, line, wrapper)
    if target == "return_value":
        wanted = _TARGET_RETURN
    else:
        wanted = _normalize_expr(target)
    if left == wanted:
        return left, relation.operator, right
    if right == wanted:
        return right, _REVERSED[relation.operator], left
    return None, relation.operator, right


def assess_oracle_binding(
    binding: KunitBinding, check: Dict, test_body: str, wrapper: str,
    full_code: str = "",
) -> OracleAssessment:
    if not binding_is_nontrivial_assertion(binding):
        return OracleAssessment("mismatch", "assertion is vacuous or checks setup only")
    relation = _assertion_relation(binding)
    if relation is None:
        return OracleAssessment("unresolved", "assertion relation is not statically supported")
    kind = str(check.get("kind", "") or "")
    target = str(check.get("target", "") or "")
    if wrapper and kind.startswith(("Return", "Field", "State")):
        source = _without_comments(test_body)
        prior = source.splitlines()[:binding.start_line]
        if not re.search(r"\b" + re.escape(wrapper) + r"\s*\(", "\n".join(prior)):
            return OracleAssessment("mismatch", "assertion occurs before the target wrapper call")
    assignments = _assignments(test_body)
    if kind in {"BoundaryCalled", "BoundaryNotCalled"}:
        count = re.search(r"call_count\s*(==|>=|<=)\s*(\d+)", str(check.get("expected_relation", "")))
        if count:
            left = _resolve(relation.left, assignments, binding.start_line, wrapper)
            right = _resolve(relation.right, assignments, binding.start_line, wrapper)
            expected_count = ("integer", int(count.group(2)))
            if _known_value(relation.left) == expected_count:
                checked, operator, expected = relation.right, _REVERSED[relation.operator], left
            else:
                checked, operator, expected = relation.left, relation.operator, right
            if _resolve(checked, assignments, binding.start_line, wrapper) == _TARGET_RETURN \
                    or _known_value(checked) is not None:
                return OracleAssessment("mismatch", "boundary check does not inspect a call counter")
            observed = _known_value(expected)
            if operator != count.group(1):
                return OracleAssessment("mismatch", "boundary call-count comparison differs from scenario")
            if observed and observed != expected_count:
                return OracleAssessment("mismatch", "boundary call-count expected value differs from scenario")
        return OracleAssessment("unresolved", "boundary counter provenance is not established")

    if kind not in {
        "ReturnEquals", "ReturnEqualsBoundaryEffect", "ReturnEqualsBoundaryOutput",
        "ReturnRelation", "StateRelation", "FieldEquals",
    }:
        return OracleAssessment("unresolved", "scenario relation has no exact static rule")

    wanted = "return_value" if kind.startswith("Return") else target
    actual, operator, expected = _target_side(
        relation, assignments, binding.start_line, wrapper, wanted
    )
    if actual is None:
        left = _resolve(relation.left, assignments, binding.start_line, wrapper)
        right = _resolve(relation.right, assignments, binding.start_line, wrapper)
        if "<pre-call-state>" in {left, right}:
            return OracleAssessment("mismatch", "assertion reads state captured before the target call")
        if wanted == "return_value" and (
            left == _TARGET_RETURN or right == _TARGET_RETURN
        ):
            return OracleAssessment("mismatch", "assertion does not check the target return")
        if wanted != "return_value" and (left == _TARGET_RETURN or right == _TARGET_RETURN):
            return OracleAssessment("mismatch", "assertion checks a return instead of required driver state")
        if _known_value(left) is not None and _known_value(right) is not None:
            return OracleAssessment("mismatch", "assertion does not inspect the required target behavior")
        return OracleAssessment("unresolved", "checked value cannot be traced to the required target behavior")

    if kind in {"ReturnEquals", "ReturnEqualsBoundaryEffect", "ReturnEqualsBoundaryOutput", "FieldEquals"}:
        if operator != "==":
            return OracleAssessment("mismatch", "scenario requires equality but assertion uses " + operator)
        required = _expected_from_contract(check)
        resolved_expected = _resolve(expected, assignments, binding.start_line, wrapper)
        if kind == "ReturnEqualsBoundaryEffect":
            negative = _is_negative(resolved_expected)
            if negative is False:
                return OracleAssessment("mismatch", "expected value is not a negative boundary errno")
            configured = _configured_mock_return(
                check, full_code, assignments, binding.start_line, wrapper
            )
            configured_value = _known_value(configured or "")
            asserted_value = _known_value(resolved_expected)
            if configured_value is not None and asserted_value is not None:
                if configured_value != asserted_value:
                    return OracleAssessment(
                        "mismatch", "expected value differs from the configured boundary return"
                    )
                return OracleAssessment("proven")
            return OracleAssessment(
                "unresolved", "configured boundary errno cannot be tied to the asserted value",
            )
        if required:
            contract_value = _known_value(required)
            asserted_value = _known_value(resolved_expected)
            if contract_value is not None and asserted_value is not None:
                if contract_value != asserted_value:
                    return OracleAssessment("mismatch", "expected value differs from scenario: " + required)
                return OracleAssessment("proven")
            if _normalize_expr(required) == resolved_expected:
                return OracleAssessment("unresolved", "matching names do not prove the test value has the source value")
            return OracleAssessment("unresolved", "expected value relation cannot be resolved statically")
    if kind in {"ReturnRelation", "StateRelation"} and operator in {"!=", ">=", "<="}:
        return OracleAssessment("unresolved", "broad relation may not validate the scenario outcome")
    return OracleAssessment("proven" if actual == _TARGET_RETURN else "unresolved")


def oracle_repair_errors(before_code: str, after_code: str, registry: Dict) -> List[str]:
    """Reject removal or broadening of a previously substantive scenario check.

    An incorrect assertion may be corrected: exact expected values may change
    when the new assertion is checked against the unchanged scenario contract.
    """
    before = {item.name: item for item in inspect_test_source(before_code).test_functions}
    after = {item.name: item for item in inspect_test_source(after_code).test_functions}
    contracts = {
        item.get("scenario_id"): item
        for item in registry.get("scenario_contracts", []) or []
        if isinstance(item, dict) and item.get("scenario_id")
    }
    active = set(registry.get("active_scenario_ids", contracts.keys()) or [])
    errors: List[str] = []
    for name, original in before.items():
        revised = after.get(name)
        if revised is None:
            continue  # Test removal is handled by the existing repair gate.
        required_scenarios = set(original.scenario_ids) & active & set(contracts)
        if not required_scenarios.issubset(set(revised.scenario_ids)):
            errors.append(
                f"Oracle weakened: test {name} removed its active scenario binding "
                f"{sorted(required_scenarios - set(revised.scenario_ids))}."
            )
        if required_scenarios and original.variant_id and revised.variant_id != original.variant_id:
            errors.append(f"Oracle weakened: test {name} changed its condition variant binding.")
        old_bindings = collect_kunit_bindings(original.full_text)
        new_bindings = collect_kunit_bindings(revised.full_text)
        for scenario_id in original.scenario_ids:
            contract = contracts.get(scenario_id, {})
            wrapper = str(contract.get("export_function", "") or "")
            for check in contract.get("scenario_checks", []) or []:
                check_id = check.get("check_id")
                if not check_id:
                    continue
                old = [
                    binding for binding in old_bindings
                    if check_id in direct_check_ids(binding, original.full_text)
                    and binding_is_nontrivial_assertion(binding)
                    and assess_oracle_binding(
                        binding, check, original.full_text, wrapper, before_code
                    ).status != "mismatch"
                ]
                if not old:
                    continue
                new = [
                    binding for binding in new_bindings
                    if check_id in direct_check_ids(binding, revised.full_text)
                    and binding_is_nontrivial_assertion(binding)
                ]
                if not new:
                    errors.append(
                        f"Oracle weakened: test {name} removed the substantive assertion for {scenario_id}/{check_id}."
                    )
                    continue
                old_exact = any(
                    (relation := _assertion_relation(binding)) is not None
                    and relation.operator == "==" for binding in old
                )
                new_exact = any(
                    (relation := _assertion_relation(binding)) is not None
                    and relation.operator == "==" for binding in new
                )
                if (
                    old_exact and not new_exact
                    and check.get("kind") not in {"BoundaryCalled", "BoundaryNotCalled"}
                ):
                    errors.append(
                        f"Oracle weakened: test {name} replaced an equality check for {scenario_id}/{check_id} "
                        "with a weaker comparison; restore a scenario-grounded exact assertion."
                    )
    return errors
