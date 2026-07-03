import re
from typing import Dict, List, Optional, Set, Tuple


RESULT_PATTERN = re.compile(r"(?:\[[^\]]+\]\s+)?\s*(ok|not ok)\s+\d+\s+(\S+)")
SUITE_SUMMARY_PATTERN = re.compile(
    r"(?:\[[^\]]+\]\s+)?\s*#\s+(\S+):\s+pass:\d+\s+fail:\d+\s+skip:\d+\s+total:\d+"
)


def empty_kunit_summary() -> Dict[str, object]:
    return {
        "tests": [],
        "passed": [],
        "failed": [],
        "tests_total": 0,
        "tests_passed_count": 0,
        "tests_failed_count": 0,
        "overall_passed": False,
        "pass_rate": 0.0,
        "per_test_pass_rate": [],
    }


def test_metrics_from_kunit(kunit_summary: Dict) -> Dict[str, object]:
    tests = [item for item in (kunit_summary or {}).get("tests", []) or [] if isinstance(item, dict)]
    passed = [item for item in tests if item.get("status") == "passed"]
    failed = [item for item in tests if item.get("status") == "failed"]
    total = len(tests)
    return {
        "tests_total": total,
        "tests_passed": len(passed),
        "tests_failed": len(failed),
        "pass_rate": (len(passed) / total * 100.0) if total else 0.0,
        "per_test": [
            {
                "name": item.get("name", ""),
                "status": item.get("status", "unknown"),
                "pass_rate": 100.0 if item.get("status") == "passed" else 0.0,
            }
            for item in tests
        ],
    }


def filter_kunit_cases_to_tests(test_code: str, selected_tests: set[str]) -> str:
    if not selected_tests:
        return test_code
    all_registered_tests = {
        match.group(1)
        for match in re.finditer(
            r"(?m)^[ \t]*KUNIT_CASE\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)\s*,?[ \t]*\n",
            test_code or "",
        )
    }

    def replace_case(match: re.Match) -> str:
        test_name = match.group(1)
        return match.group(0) if test_name in selected_tests else ""

    filtered = re.sub(
        r"(?m)^[ \t]*KUNIT_CASE\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)\s*,?[ \t]*\n",
        replace_case,
        test_code or "",
    )
    filtered = _remove_unselected_kunit_test_functions(
        filtered,
        selected_tests,
        all_registered_tests,
    )
    filtered = _prune_unreachable_static_functions(filtered, selected_tests)
    return _remove_orphan_static_prototypes(filtered)


def _find_matching_brace(code: str, open_idx: int) -> Optional[int]:
    depth = 0
    in_string = False
    in_char = False
    escaped = False
    i = open_idx
    while i < len(code):
        ch = code[i]
        if escaped:
            escaped = False
            i += 1
            continue
        if not in_string and not in_char and code.startswith("/*", i):
            end = code.find("*/", i + 2)
            i = len(code) if end < 0 else end + 2
            continue
        if not in_string and not in_char and code.startswith("//", i):
            end = code.find("\n", i + 2)
            i = len(code) if end < 0 else end + 1
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


def _remove_unselected_kunit_test_functions(
    test_code: str,
    selected_tests: set[str],
    all_registered_tests: set[str],
) -> str:
    if not all_registered_tests:
        return test_code or ""
    removals = []
    for name, start_idx, end_idx in _static_function_spans(test_code or ""):
        looks_like_test_function = name in all_registered_tests or name.startswith("test_")
        if not looks_like_test_function or name in selected_tests:
            continue
        adjusted_start = start_idx
        while start_idx > 0:
            line_start = (test_code or "").rfind("\n", 0, start_idx - 1) + 1
            line = (test_code or "")[line_start:start_idx]
            stripped = line.strip()
            if not stripped:
                adjusted_start = line_start
                start_idx = line_start
                continue
            if stripped.startswith("/* RACA_") or stripped.startswith("// RACA_"):
                adjusted_start = line_start
                start_idx = line_start
                continue
            break
        removals.append((adjusted_start, end_idx))
    if not removals:
        return test_code or ""
    updated = test_code or ""
    for start, end in reversed(removals):
        updated = updated[:start] + updated[end:]
    return re.sub(r"\n{3,}", "\n\n", updated)


def _static_function_spans(test_code: str) -> List[Tuple[str, int, int]]:
    pattern = re.compile(
        r"(?ms)^[ \t]*static\b"
        r"(?P<header>[^;{}]*?\b(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*"
        r"\([^;{}]*\)\s*(?:__[A-Za-z_][A-Za-z0-9_]*\s*)*)\{"
    )
    spans: List[Tuple[str, int, int]] = []
    for match in pattern.finditer(test_code or ""):
        open_idx = (test_code or "").find("{", match.end() - 1)
        if open_idx < 0:
            continue
        close_idx = _find_matching_brace(test_code or "", open_idx)
        if close_idx is None:
            continue
        spans.append((match.group("name"), match.start(), close_idx + 1))
    return spans


def _suite_function_roots(test_code: str) -> Set[str]:
    roots: Set[str] = set()
    for match in re.finditer(
        r"(?m)^[ \t]*\.(?:init|exit)\s*=\s*([A-Za-z_][A-Za-z0-9_]*)\s*,?",
        test_code or "",
    ):
        roots.add(match.group(1))
    return roots


def _called_static_functions(body: str, static_names: Set[str]) -> Set[str]:
    called: Set[str] = set()
    for name in static_names:
        if re.search(r"\b" + re.escape(name) + r"\b", body or ""):
            called.add(name)
    return called


def _prune_unreachable_static_functions(test_code: str, selected_tests: set[str]) -> str:
    spans = _static_function_spans(test_code or "")
    if not spans:
        return test_code or ""
    static_names = {name for name, _, _ in spans}
    roots = (set(selected_tests) & static_names) | (_suite_function_roots(test_code or "") & static_names)
    if not roots:
        return test_code or ""

    bodies: Dict[str, str] = {
        name: (test_code or "")[start:end]
        for name, start, end in spans
    }
    reachable = set(roots)
    changed = True
    while changed:
        changed = False
        for name in list(reachable):
            for callee in _called_static_functions(bodies.get(name, ""), static_names):
                if callee not in reachable:
                    reachable.add(callee)
                    changed = True

    removals = [(start, end) for name, start, end in spans if name not in reachable]
    if not removals:
        return test_code or ""
    updated = test_code or ""
    for start, end in reversed(removals):
        updated = updated[:start] + updated[end:]
    return re.sub(r"\n{3,}", "\n\n", updated)


def _remove_orphan_static_prototypes(test_code: str) -> str:
    code = test_code or ""
    defined_names = {name for name, _, _ in _static_function_spans(code)}
    prototype_pattern = re.compile(
        r"(?ms)^[ \t]*static\b[^;{}]*?\b(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*"
        r"\([^;{}]*\)\s*;[ \t]*(?:\n|$)"
    )
    removals = []
    for match in prototype_pattern.finditer(code):
        if match.group("name") not in defined_names:
            removals.append((match.start(), match.end()))
    if not removals:
        return code
    updated = code
    for start, end in reversed(removals):
        updated = updated[:start] + updated[end:]
    return re.sub(r"\n{3,}", "\n\n", updated)


def parse_kunit_results_text(text: str, suite_name: Optional[str] = None) -> Dict[str, object]:
    results: List[Dict[str, str]] = []
    suite_names = {suite_name} if suite_name else set()
    for raw_line in (text or "").splitlines():
        stripped = raw_line.strip()
        summary_match = SUITE_SUMMARY_PATTERN.match(stripped)
        if summary_match:
            suite_names.add(summary_match.group(1))
            continue
        match = RESULT_PATTERN.match(stripped)
        if not match:
            continue
        status_word, name = match.groups()
        if name in suite_names or "-" in name:
            continue
        status = "passed" if status_word == "ok" else "failed"
        results.append(
            {
                "name": name,
                "status": status,
                "pass_rate": 100.0 if status == "passed" else 0.0,
            }
        )
    passed = [r["name"] for r in results if r["status"] == "passed"]
    failed = [r["name"] for r in results if r["status"] == "failed"]
    total = len(results)
    pass_rate = (len(passed) / total * 100.0) if total else 0.0
    return {
        "tests": results,
        "passed": passed,
        "failed": failed,
        "tests_total": total,
        "tests_passed_count": len(passed),
        "tests_failed_count": len(failed),
        "overall_passed": len(failed) == 0 and bool(results),
        "pass_rate": pass_rate,
        "per_test_pass_rate": [
            {
                "name": item["name"],
                "status": item["status"],
                "pass_rate": item["pass_rate"],
            }
            for item in results
        ],
    }


def suite_log_block(text: str, suite_name: str) -> str:
    if not text or not suite_name:
        return text or ""
    lines = text.splitlines()
    start = None
    end = None
    subtest_pattern = re.compile(r"#\s+Subtest:\s+" + re.escape(suite_name) + r"\b")
    suite_done_pattern = re.compile(r"(?:ok|not ok)\s+\d+\s+" + re.escape(suite_name) + r"\b")
    for idx, line in enumerate(lines):
        normalized = re.sub(r"^\[[^\]]+\]\s*", "", line.strip())
        if start is None and subtest_pattern.search(normalized):
            start = idx
            continue
        if start is not None and idx > start and suite_done_pattern.search(normalized):
            end = idx + 1
            break
    if start is None:
        return ""
    return "\n".join(lines[start:end])


def parse_kunit_results_file(log_path: str, suite_name: Optional[str] = None) -> Dict[str, object]:
    try:
        with open(log_path, "r") as f:
            text = f.read()
    except FileNotFoundError:
        return empty_kunit_summary()
    if suite_name:
        text = suite_log_block(text, suite_name)
        if not text:
            return empty_kunit_summary()
    return parse_kunit_results_text(text, suite_name=suite_name)


def extract_failure_reason_from_text(text: str) -> str:
    if not text:
        return ""
    normalized_lines = [line.strip() for line in text.splitlines()]
    filtered = []
    for line in normalized_lines:
        if not line:
            continue
        stripped = line.lstrip()
        if stripped.startswith("KTAP") or stripped.startswith("1.."):
            continue
        if stripped.startswith("ok "):
            continue
        if stripped.startswith("# Totals"):
            continue
        if stripped.startswith("# ") and "pass:" in stripped and "fail:" in stripped:
            continue
        filtered.append(stripped)
    return "\n".join(filtered)


def extract_failure_reason_from_log(text: str, suite_name: Optional[str] = None) -> str:
    if not text:
        return ""
    if suite_name:
        text = suite_log_block(text, suite_name)
        if not text:
            return ""
    normalized_lines = [re.sub(r"^\[[^\]]+\]\s*", "", line.strip()) for line in text.splitlines()]
    bug_rip_lines = [
        line
        for line in normalized_lines
        if line.startswith("BUG:")
        or line.startswith("RIP:")
        or line.startswith("not ok ")
        or "internal error occurred" in line
        or "try faulted" in line
    ]
    if bug_rip_lines:
        return "\n".join(bug_rip_lines)
    return ""


def extract_failure_reason(log_path: str) -> str:
    try:
        with open(log_path, "r") as f:
            return extract_failure_reason_from_log(f.read())
    except FileNotFoundError:
        return ""
