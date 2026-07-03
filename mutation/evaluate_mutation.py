import argparse
import bisect
import collections
import json
import os
import re
import shlex
import subprocess
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple


OPERATOR_FAMILIES: List[Tuple[str, List[str]]] = [
    ("relational", ["==", "!=", ">=", "<=", ">", "<"]),
    ("logical", ["&&", "||"]),
    ("arithmetic", ["+", "-", "*", "/", "%"]),
    ("bitwise", ["<<", ">>", "&", "|", "^"]),
    ("compound_assignment", ["<<=", ">>=", "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^="]),
    ("inc_dec", ["++", "--"]),
]
OPERATOR_GROUP_ORDER = ["rep_const", "rep_op", "negate", "del_stmt"]
REPLACEMENT_POLICIES = {"representative", "exhaustive"}

OPERATOR_NAMES = {
    "==": "eq",
    "!=": "ne",
    ">=": "ge",
    "<=": "le",
    ">": "gt",
    "<": "lt",
    "&&": "and",
    "||": "or",
    "+": "add",
    "-": "sub",
    "*": "mul",
    "/": "div",
    "%": "mod",
    "<<": "shl",
    ">>": "shr",
    "&": "bitand",
    "|": "bitor",
    "^": "bitxor",
    "+=": "add_assign",
    "-=": "sub_assign",
    "*=": "mul_assign",
    "/=": "div_assign",
    "%=": "mod_assign",
    "&=": "and_assign",
    "|=": "or_assign",
    "^=": "xor_assign",
    "<<=": "shl_assign",
    ">>=": "shr_assign",
    "++": "inc",
    "--": "dec",
}

ALL_OPERATOR_TOKENS = sorted(
    {operator for _, operators in OPERATOR_FAMILIES for operator in operators},
    key=len,
    reverse=True,
)
OPERATOR_PATTERN = re.compile("|".join(re.escape(operator) for operator in ALL_OPERATOR_TOKENS))
INTEGER_LITERAL_PATTERN = re.compile(
    r"(?<![\w.])(?:0[xX][0-9a-fA-F]+|0[bB][01]+|0[0-7]+|[1-9][0-9]*|0)(?:[uUlL]*)\b"
)


@dataclass
class Mutant:
    mutant_id: str
    operator_group: str
    operator: str
    line: int
    end_line: int
    column: int
    original: str
    replacement: str
    content: str

    def metadata(self) -> Dict:
        return {
            "mutant_id": self.mutant_id,
            "operator_group": self.operator_group,
            "operator": self.operator,
            "line": self.line,
            "end_line": self.end_line,
            "column": self.column,
            "original": self.original,
            "replacement": self.replacement,
        }


def _line_offsets(text: str) -> List[int]:
    offsets = [0]
    for match in re.finditer("\n", text):
        offsets.append(match.end())
    return offsets


def _line_for_offset(offsets: List[int], pos: int) -> int:
    return bisect.bisect_right(offsets, pos)


def _line_start_for_offset(offsets: List[int], pos: int) -> int:
    line = _line_for_offset(offsets, pos)
    return offsets[line - 1] if 1 <= line <= len(offsets) else 0


def _within_line_range(line: int, start_line: Optional[int], end_line: Optional[int]) -> bool:
    if start_line is not None and line < start_line:
        return False
    if end_line is not None and line > end_line:
        return False
    return True


def _mask_comments_and_literals(source: str) -> str:
    chars = list(source)
    i = 0
    while i < len(chars):
        if source.startswith("//", i):
            end = source.find("\n", i + 2)
            end = len(source) if end < 0 else end
            for idx in range(i, end):
                chars[idx] = " "
            i = end
            continue
        if source.startswith("/*", i):
            end = source.find("*/", i + 2)
            end = len(source) - 2 if end < 0 else end
            for idx in range(i, min(end + 2, len(chars))):
                if chars[idx] != "\n":
                    chars[idx] = " "
            i = end + 2
            continue
        if source[i] in {'"', "'"}:
            quote = source[i]
            chars[i] = " "
            i += 1
            escaped = False
            while i < len(chars):
                ch = source[i]
                if ch != "\n":
                    chars[i] = " "
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == quote:
                    i += 1
                    break
                i += 1
            continue
        i += 1
    return "".join(chars)


def _line_text(source: str, line: int) -> str:
    lines = source.splitlines()
    if 1 <= line <= len(lines):
        return lines[line - 1]
    return ""


def _is_preprocessor_line(source: str, line: int) -> bool:
    return _line_text(source, line).lstrip().startswith("#")


def _line_span(start_line: int, end_line: int) -> Set[int]:
    return set(range(start_line, end_line + 1))


def _location_allowed(
    start_line: int,
    end_line: int,
    start_range: Optional[int],
    end_range: Optional[int],
    allowed_lines: Optional[Set[int]],
) -> bool:
    if not _within_line_range(start_line, start_range, end_range):
        return False
    if not _within_line_range(end_line, start_range, end_range):
        return False
    if allowed_lines is not None and not (_line_span(start_line, end_line) & allowed_lines):
        return False
    return True


def _operator_family(token: str) -> Optional[Tuple[str, List[str]]]:
    for family, operators in OPERATOR_FAMILIES:
        if token in operators:
            return family, operators
    return None


def _operator_name(original: str, replacement: str) -> str:
    return f"{OPERATOR_NAMES.get(original, original)}_to_{OPERATOR_NAMES.get(replacement, replacement)}"


def _replacement_candidates_for_constant(value: int) -> List[str]:
    candidates = ["0", "1", "-1", str(value + 1), str(value - 1)]
    deduped: List[str] = []
    seen = set()
    for candidate in candidates:
        if candidate not in seen:
            deduped.append(candidate)
            seen.add(candidate)
    return deduped


def _select_replacements(candidates: List[str], original: str, replacement_policy: str) -> List[str]:
    filtered = [candidate for candidate in candidates if candidate != original]
    if replacement_policy == "exhaustive":
        return filtered
    return filtered[:1]


def _representative_operator_replacement(token: str) -> Optional[str]:
    replacements = {
        "==": "!=",
        "!=": "==",
        ">": "<=",
        "<": ">=",
        ">=": "<",
        "<=": ">",
        "&&": "||",
        "||": "&&",
        "+": "-",
        "-": "+",
        "*": "/",
        "/": "*",
        "%": "/",
        "&": "|",
        "|": "&",
        "^": "&",
        "<<": ">>",
        ">>": "<<",
        "+=": "-=",
        "-=": "+=",
        "*=": "/=",
        "/=": "*=",
        "%=": "/=",
        "&=": "|=",
        "|=": "&=",
        "^=": "&=",
        "<<=": ">>=",
        ">>=": "<<=",
        "++": "--",
        "--": "++",
    }
    return replacements.get(token)


def _parse_integer_literal(literal: str) -> Optional[int]:
    cleaned = re.sub(r"[uUlL]+$", "", literal)
    if cleaned.lower().startswith("0b"):
        return int(cleaned[2:], 2)
    try:
        return int(cleaned, 0)
    except ValueError:
        return None


def _append_mutant(
    mutants: List[Mutant],
    source: str,
    offsets: List[int],
    edit_start: int,
    edit_end: int,
    replacement: str,
    operator_group: str,
    operator: str,
    max_mutants: Optional[int],
) -> bool:
    original = source[edit_start:edit_end]
    if original == replacement:
        return False
    start_line = _line_for_offset(offsets, edit_start)
    end_line = _line_for_offset(offsets, max(edit_start, edit_end - 1))
    line_start = _line_start_for_offset(offsets, edit_start)
    mutant_id = f"M{len(mutants) + 1:04d}"
    mutants.append(
        Mutant(
            mutant_id=mutant_id,
            operator_group=operator_group,
            operator=operator,
            line=start_line,
            end_line=end_line,
            column=edit_start - line_start + 1,
            original=original,
            replacement=replacement,
            content=source[:edit_start] + replacement + source[edit_end:],
        )
    )
    return max_mutants is not None and len(mutants) >= max_mutants


def _looks_like_operator_token(masked: str, start: int, end: int, token: str) -> bool:
    prev_ch = masked[start - 1] if start > 0 else ""
    next_ch = masked[end] if end < len(masked) else ""
    if token == "-" and next_ch == ">":
        return False
    if token == ">" and prev_ch == "-":
        return False
    if token in {"&", "|", "+", "-"} and next_ch == token:
        return False
    if token in {"&", "|", "+", "-"} and prev_ch == token:
        return False
    if token in {">", "<"} and next_ch == "=":
        return False
    if token in {">", "<"} and prev_ch in {"<", ">", "!"}:
        return False
    if token == "=":
        return False
    if token in {"*", "&", "+", "-"}:
        prev_sig = _previous_significant_char(masked, start)
        if prev_sig is None or not (prev_sig.isalnum() or prev_sig == "_" or prev_sig in {")", "]", "}"}):
            return False
    return True


def _previous_significant_char(masked: str, start: int) -> Optional[str]:
    idx = start - 1
    while idx >= 0:
        ch = masked[idx]
        if not ch.isspace():
            return ch
        idx -= 1
    return None


def _matching_paren(masked: str, open_idx: int) -> Optional[int]:
    depth = 0
    for idx in range(open_idx, len(masked)):
        ch = masked[idx]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return idx
    return None


def _negatable_conditions(masked: str) -> Iterable[Tuple[int, int, int]]:
    for match in re.finditer(r"\b(if|while)\s*\(", masked):
        open_idx = masked.find("(", match.start(), match.end())
        if open_idx < 0:
            continue
        close_idx = _matching_paren(masked, open_idx)
        if close_idx is None:
            continue
        yield match.start(), open_idx + 1, close_idx


def _statement_spans(masked: str, offsets: List[int]) -> Iterable[Tuple[int, int]]:
    paren_depth = 0
    bracket_depth = 0
    statement_start = 0
    for idx, ch in enumerate(masked):
        if ch == "\n":
            line = _line_for_offset(offsets, idx)
            if _is_preprocessor_line(masked, line + 1):
                statement_start = idx + 1
            continue
        if ch == "(":
            paren_depth += 1
        elif ch == ")" and paren_depth > 0:
            paren_depth -= 1
        elif ch == "[":
            bracket_depth += 1
        elif ch == "]" and bracket_depth > 0:
            bracket_depth -= 1
        elif ch in "{}":
            statement_start = idx + 1
        elif ch == ";" and paren_depth == 0 and bracket_depth == 0:
            raw_start = statement_start
            while raw_start < idx and masked[raw_start].isspace():
                raw_start += 1
            if raw_start < idx:
                yield raw_start, idx + 1
            statement_start = idx + 1


def _generate_constant_mutants(
    source: str,
    masked: str,
    offsets: List[int],
    start_line: Optional[int],
    end_line: Optional[int],
    allowed_lines: Optional[Set[int]],
    max_mutants: Optional[int],
    replacement_policy: str,
    mutants: List[Mutant],
) -> bool:
    for match in INTEGER_LITERAL_PATTERN.finditer(masked):
        line = _line_for_offset(offsets, match.start())
        if _is_preprocessor_line(source, line):
            continue
        if match.start() > 0 and source[match.start() - 1] == "-":
            continue
        if not _location_allowed(line, line, start_line, end_line, allowed_lines):
            continue
        literal = source[match.start() : match.end()]
        value = _parse_integer_literal(literal)
        if value is None:
            continue
        for replacement in _select_replacements(
            _replacement_candidates_for_constant(value),
            literal,
            replacement_policy,
        ):
            if _append_mutant(
                mutants,
                source,
                offsets,
                match.start(),
                match.end(),
                replacement,
                "rep_const",
                f"const_to_{replacement}",
                max_mutants,
            ):
                return True
    return False


def _generate_operator_mutants(
    source: str,
    masked: str,
    offsets: List[int],
    start_line: Optional[int],
    end_line: Optional[int],
    allowed_lines: Optional[Set[int]],
    max_mutants: Optional[int],
    replacement_policy: str,
    mutants: List[Mutant],
) -> bool:
    for match in OPERATOR_PATTERN.finditer(masked):
        token = match.group(0)
        if not _looks_like_operator_token(masked, match.start(), match.end(), token):
            continue
        line = _line_for_offset(offsets, match.start())
        if _is_preprocessor_line(source, line):
            continue
        if not _location_allowed(line, line, start_line, end_line, allowed_lines):
            continue
        family = _operator_family(token)
        if family is None:
            continue
        _, replacements = family
        if replacement_policy == "representative":
            representative = _representative_operator_replacement(token)
            replacements = [representative] if representative else []
        for replacement in replacements:
            if replacement == token:
                continue
            if _append_mutant(
                mutants,
                source,
                offsets,
                match.start(),
                match.end(),
                replacement,
                "rep_op",
                _operator_name(token, replacement),
                max_mutants,
            ):
                return True
    return False


def _generate_negation_mutants(
    source: str,
    masked: str,
    offsets: List[int],
    start_line: Optional[int],
    end_line: Optional[int],
    allowed_lines: Optional[Set[int]],
    max_mutants: Optional[int],
    replacement_policy: str,
    mutants: List[Mutant],
) -> bool:
    for keyword_start, condition_start, condition_end in _negatable_conditions(masked):
        first_line = _line_for_offset(offsets, keyword_start)
        last_line = _line_for_offset(offsets, max(condition_start, condition_end - 1))
        if _is_preprocessor_line(source, first_line):
            continue
        if not _location_allowed(first_line, last_line, start_line, end_line, allowed_lines):
            continue
        condition = source[condition_start:condition_end].strip()
        if not condition:
            continue
        replacement = f"!({condition})"
        if _append_mutant(
            mutants,
            source,
            offsets,
            condition_start,
            condition_end,
            replacement,
            "negate",
            "negate_condition",
            max_mutants,
        ):
            return True
    return False


def _generate_statement_deletion_mutants(
    source: str,
    masked: str,
    offsets: List[int],
    start_line: Optional[int],
    end_line: Optional[int],
    allowed_lines: Optional[Set[int]],
    max_mutants: Optional[int],
    replacement_policy: str,
    mutants: List[Mutant],
) -> bool:
    for stmt_start, stmt_end in _statement_spans(masked, offsets):
        first_line = _line_for_offset(offsets, stmt_start)
        last_line = _line_for_offset(offsets, max(stmt_start, stmt_end - 1))
        if _is_preprocessor_line(source, first_line):
            continue
        statement = source[stmt_start:stmt_end].strip()
        if not statement or statement.startswith(("case ", "default:")):
            continue
        if not _location_allowed(first_line, last_line, start_line, end_line, allowed_lines):
            continue
        if _append_mutant(
            mutants,
            source,
            offsets,
            stmt_start,
            stmt_end,
            ";",
            "del_stmt",
            "delete_statement",
            max_mutants,
        ):
            return True
    return False


def generate_mutants(
    source: str,
    start_line: Optional[int] = None,
    end_line: Optional[int] = None,
    max_mutants: Optional[int] = None,
    allowed_lines: Optional[Set[int]] = None,
    replacement_policy: str = "representative",
    max_mutants_per_group: Optional[int] = None,
) -> List[Mutant]:
    if replacement_policy not in REPLACEMENT_POLICIES:
        raise ValueError(f"Unsupported replacement_policy: {replacement_policy}")
    if max_mutants_per_group is not None and max_mutants_per_group <= 0:
        raise ValueError("max_mutants_per_group must be positive when provided")
    offsets = _line_offsets(source)
    masked = _mask_comments_and_literals(source)
    mutants: List[Mutant] = []
    generation_limit = None if max_mutants_per_group is not None else max_mutants
    generators = [
        _generate_constant_mutants,
        _generate_operator_mutants,
        _generate_negation_mutants,
        _generate_statement_deletion_mutants,
    ]
    for generator in generators:
        reached_limit = generator(
            source,
            masked,
            offsets,
            start_line,
            end_line,
            allowed_lines,
            generation_limit,
            replacement_policy,
            mutants,
        )
        if reached_limit:
            return mutants
    if max_mutants_per_group is not None:
        mutants = _limit_mutants_per_group(mutants, max_mutants_per_group)
    if max_mutants is not None:
        mutants = mutants[:max_mutants]
    return mutants


def _limit_mutants_per_group(mutants: List[Mutant], max_per_group: int) -> List[Mutant]:
    grouped: Dict[str, List[Mutant]] = collections.defaultdict(list)
    for mutant in mutants:
        grouped[mutant.operator_group].append(mutant)

    selected: List[Mutant] = []
    for group in OPERATOR_GROUP_ORDER:
        selected.extend(_source_stratified_sample(grouped.get(group, []), max_per_group))
    for group in sorted(item for item in grouped if item not in OPERATOR_GROUP_ORDER):
        selected.extend(_source_stratified_sample(grouped[group], max_per_group))
    return sorted(selected, key=lambda item: (item.line, item.column, item.operator_group, item.operator))


def _source_stratified_sample(mutants: List[Mutant], max_items: int) -> List[Mutant]:
    """Select a deterministic early/middle/late spread within one operator group."""
    ordered = sorted(mutants, key=lambda item: (item.line, item.column, item.operator, item.replacement))
    if len(ordered) <= max_items:
        return ordered
    if max_items == 1:
        return [ordered[len(ordered) // 2]]
    indices = []
    for i in range(max_items):
        idx = round(i * (len(ordered) - 1) / (max_items - 1))
        if idx not in indices:
            indices.append(idx)
    # Rounding can theoretically collapse adjacent picks for very small lists.
    cursor = 0
    while len(indices) < max_items and cursor < len(ordered):
        if cursor not in indices:
            indices.append(cursor)
        cursor += 1
    return [ordered[idx] for idx in sorted(indices)]


def _dedupe_anchor_facts(anchor_facts: Iterable[Dict]) -> List[Dict]:
    deduped: List[Dict] = []
    seen = set()
    for fact in anchor_facts:
        key = (
            fact.get("test_function"),
            fact.get("scenario_id"),
            fact.get("fact_id"),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(fact)
    return deduped


def _attach_line_annotations(entry: Dict, mutant: Mutant, line_annotations: Optional[Dict[int, List[Dict]]]) -> None:
    if not line_annotations:
        entry["passed_scenario_related"] = False
        return
    anchors: List[Dict] = []
    for line in range(mutant.line, mutant.end_line + 1):
        anchors.extend(line_annotations.get(line, []))
    anchors = _dedupe_anchor_facts(anchors)
    entry["passed_scenario_related"] = bool(anchors)
    if anchors:
        entry["passed_scenario_anchor_facts"] = anchors


def _result_metrics(results: List[Dict]) -> Dict:
    valid = [item for item in results if item.get("status") in {"killed", "survived"}]
    killed = [item for item in valid if item.get("status") == "killed"]
    invalid = [item for item in results if item.get("status") == "invalid"]
    survived = [item for item in valid if item.get("status") == "survived"]
    by_group = collections.Counter(item.get("operator_group", "unknown") for item in results)
    by_operator = collections.Counter(item.get("operator", "unknown") for item in results)
    return {
        "total_mutants": len(results),
        "valid_mutants": len(valid),
        "invalid_mutants": len(invalid),
        "killed_mutants": len(killed),
        "survived_mutants": len(survived),
        "mutation_score": (len(killed) / len(valid) * 100.0) if valid else 0.0,
        "operator_group_counts": dict(sorted(by_group.items())),
        "operator_counts": dict(sorted(by_operator.items())),
        "operator_group_summary": _metrics_by_key(results, "operator_group", OPERATOR_GROUP_ORDER),
        "operator_summary": _metrics_by_key(results, "operator", None),
    }


def _subset_coverage_metrics(target_metrics: Dict, subset_metrics: Dict) -> Dict:
    target_total = int(target_metrics.get("total_mutants", 0) or 0)
    subset_total = int(subset_metrics.get("total_mutants", 0) or 0)
    group_coverage: Dict[str, Dict] = {}
    target_groups = target_metrics.get("operator_group_summary", {}) or {}
    subset_groups = subset_metrics.get("operator_group_summary", {}) or {}
    for group in OPERATOR_GROUP_ORDER:
        if group not in target_groups and group not in subset_groups:
            continue
        group_target = int((target_groups.get(group, {}) or {}).get("total_mutants", 0) or 0)
        group_subset = int((subset_groups.get(group, {}) or {}).get("total_mutants", 0) or 0)
        group_coverage[group] = {
            "target_mutants": group_target,
            "passed_scenario_related_mutants": group_subset,
            "coverage_percent": (group_subset / group_target * 100.0) if group_target else 0.0,
        }
    return {
        "target_mutants": target_total,
        "passed_scenario_related_mutants": subset_total,
        "coverage_percent": (subset_total / target_total * 100.0) if target_total else 0.0,
        "operator_group_coverage": group_coverage,
    }


def _metrics_by_key(results: List[Dict], key: str, preferred_order: Optional[List[str]]) -> Dict[str, Dict]:
    grouped: Dict[str, List[Dict]] = collections.defaultdict(list)
    for item in results:
        grouped[str(item.get(key, "unknown"))].append(item)
    if preferred_order:
        ordered_keys = [item for item in preferred_order if item in grouped]
        ordered_keys.extend(sorted(item for item in grouped if item not in preferred_order))
    else:
        ordered_keys = sorted(grouped)
    summary: Dict[str, Dict] = {}
    for group_key in ordered_keys:
        items = grouped[group_key]
        valid = [item for item in items if item.get("status") in {"killed", "survived"}]
        killed = [item for item in valid if item.get("status") == "killed"]
        summary[group_key] = {
            "total_mutants": len(items),
            "valid_mutants": len(valid),
            "invalid_mutants": len([item for item in items if item.get("status") == "invalid"]),
            "killed_mutants": len(killed),
            "survived_mutants": len([item for item in valid if item.get("status") == "survived"]),
            "mutation_score": (len(killed) / len(valid) * 100.0) if valid else 0.0,
        }
    return summary


def _run_command(command: str, cwd: Optional[str], timeout: Optional[int]) -> Dict:
    started = time.time()
    if not command:
        return {"returncode": 0, "stdout": "", "stderr": "", "elapsed_seconds": 0.0}
    result = subprocess.run(
        shlex.split(command),
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return {
        "returncode": result.returncode,
        "stdout": result.stdout[-4000:],
        "stderr": result.stderr[-4000:],
        "elapsed_seconds": time.time() - started,
    }


def evaluate_mutants(
    driver_c_path: str,
    build_command: str,
    run_command: str,
    output_path: str,
    command_cwd: Optional[str] = None,
    start_line: Optional[int] = None,
    end_line: Optional[int] = None,
    max_mutants: Optional[int] = None,
    timeout: Optional[int] = None,
    allowed_lines: Optional[Set[int]] = None,
    line_annotations: Optional[Dict[int, List[Dict]]] = None,
    replacement_policy: str = "representative",
    max_mutants_per_group: Optional[int] = None,
) -> Dict:
    with open(driver_c_path, "r", encoding="utf-8", errors="ignore") as f:
        original_source = f.read()

    mutants = generate_mutants(
        original_source,
        start_line=start_line,
        end_line=end_line,
        max_mutants=max_mutants,
        allowed_lines=allowed_lines,
        replacement_policy=replacement_policy,
        max_mutants_per_group=max_mutants_per_group,
    )
    results: List[Dict] = []

    try:
        for mutant in mutants:
            with open(driver_c_path, "w", encoding="utf-8") as f:
                f.write(mutant.content)

            entry = mutant.metadata()
            _attach_line_annotations(entry, mutant, line_annotations)
            try:
                build = _run_command(build_command, command_cwd, timeout)
            except subprocess.TimeoutExpired:
                entry.update({"status": "invalid", "reason": "build_timeout"})
                results.append(entry)
                continue

            entry["build"] = build
            if build["returncode"] != 0:
                entry.update({"status": "invalid", "reason": "build_failed"})
                results.append(entry)
                continue

            try:
                run = _run_command(run_command, command_cwd, timeout)
            except subprocess.TimeoutExpired:
                entry.update({"status": "killed", "reason": "run_timeout"})
                results.append(entry)
                continue

            entry["run"] = run
            if run["returncode"] == 0:
                entry.update({"status": "survived", "reason": "tests_passed"})
            else:
                entry.update({"status": "killed", "reason": "tests_failed_or_crashed"})
            results.append(entry)
    finally:
        with open(driver_c_path, "w", encoding="utf-8") as f:
            f.write(original_source)

    target_metrics = _result_metrics(results)
    passed_related_results = [item for item in results if item.get("passed_scenario_related")]
    passed_related_metrics = _result_metrics(passed_related_results)
    passed_scenario_site_coverage = _subset_coverage_metrics(target_metrics, passed_related_metrics)
    summary = {
        "driver_c_path": driver_c_path,
        "replacement_policy": replacement_policy,
        "max_mutants_per_group": max_mutants_per_group,
        "mutant_selection_policy": "source_stratified_by_operator_group" if max_mutants_per_group else "all_generated",
        **target_metrics,
        "target_mutation": target_metrics,
        "passed_scenarios_subset": passed_related_metrics,
        "passed_scenario_mutation_site_coverage": passed_scenario_site_coverage,
        "allowed_lines": sorted(allowed_lines) if allowed_lines is not None else None,
        "annotated_lines": sorted(line_annotations) if line_annotations else [],
        "results": results,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate fixed generated KUnit tests with mutation testing.")
    parser.add_argument("--driver-c-path", required=True)
    parser.add_argument("--build-command", required=True, help="Command that builds the fixed test target.")
    parser.add_argument("--run-command", required=True, help="Command that runs the fixed KUnit tests.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--cwd", default=None)
    parser.add_argument("--start-line", type=int, default=None)
    parser.add_argument("--end-line", type=int, default=None)
    parser.add_argument("--max-mutants", type=int, default=None)
    parser.add_argument("--max-mutants-per-group", type=int, default=None)
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument("--replacement-policy", choices=sorted(REPLACEMENT_POLICIES), default="representative")
    args = parser.parse_args()

    summary = evaluate_mutants(
        driver_c_path=args.driver_c_path,
        build_command=args.build_command,
        run_command=args.run_command,
        output_path=args.output,
        command_cwd=args.cwd,
        start_line=args.start_line,
        end_line=args.end_line,
        max_mutants=args.max_mutants,
        max_mutants_per_group=args.max_mutants_per_group,
        timeout=args.timeout,
        replacement_policy=args.replacement_policy,
    )
    print(
        "[MUTATION] valid={valid} killed={killed} survived={survived} score={score:.2f}%".format(
            valid=summary["valid_mutants"],
            killed=summary["killed_mutants"],
            survived=summary["survived_mutants"],
            score=summary["mutation_score"],
        )
    )


if __name__ == "__main__":
    main()
