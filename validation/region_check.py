import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

from validation.test_inspector import inspect_test_source, test_function_map


DRIVER_LOCAL_BEGIN = "/* ===== Driver Local Definitions BEGIN ===== */"
DRIVER_LOCAL_END = "/* ===== Driver Local Definitions END ===== */"
TEST_EXPORT_INTERFACES = "/* TEST EXPORT INTERFACES - DO NOT MODIFY */"


@dataclass
class RegionValidationResult:
    ok: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {"ok": self.ok, "errors": self.errors, "warnings": self.warnings}


def _include_lines(code: str) -> List[str]:
    return inspect_test_source(code).includes


def _extern_lines(code: str) -> List[str]:
    return inspect_test_source(code).externs


def _protected_extern_lines(code: str) -> List[str]:
    start = (code or "").find(TEST_EXPORT_INTERFACES)
    if start < 0:
        return _extern_lines(code)
    end = (code or "").find(DRIVER_LOCAL_BEGIN, start)
    block = (code or "")[start:end if end >= 0 else len(code)]
    return _extern_lines(block)


def _driver_local_block(code: str) -> str:
    start = code.find(DRIVER_LOCAL_BEGIN)
    if start < 0:
        return ""
    end = code.find(DRIVER_LOCAL_END, start)
    if end < 0:
        return code[start:].strip()
    end += len(DRIVER_LOCAL_END)
    return code[start:end].strip()


def _suite_names(code: str) -> List[str]:
    return inspect_test_source(code).suite_names


def _find_matching_brace(code: str, open_idx: int) -> Optional[int]:
    depth = 0
    i = open_idx
    while i < len(code):
        ch = code[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return None


def _function_blocks(code: str) -> Dict[str, str]:
    blocks: Dict[str, str] = {}
    pattern = re.compile(
        r"(?m)^[A-Za-z_][A-Za-z0-9_\s\*]*\s+([A-Za-z_][A-Za-z0-9_]*)\s*\([^;{}]*\)\s*\{"
    )
    for match in pattern.finditer(code or ""):
        name = match.group(1)
        open_idx = code.find("{", match.end() - 1)
        if open_idx < 0:
            continue
        end_idx = _find_matching_brace(code, open_idx)
        if end_idx is None:
            continue
        blocks[name] = code[match.start() : end_idx + 1]
    return blocks


def extract_test_function(code: str, test_name: str) -> Optional[str]:
    test_function = test_function_map(code).get(test_name)
    return test_function.full_text if test_function else None


def extract_test_names(code: str) -> List[str]:
    return [item.name for item in inspect_test_source(code).test_functions]


def validate_protected_regions(
    before: str,
    after: str,
    frozen_tests: Optional[Iterable[str]] = None,
) -> RegionValidationResult:
    errors: List[str] = []
    warnings: List[str] = []

    if _include_lines(before) != _include_lines(after):
        errors.append("Protected include region changed.")

    if _protected_extern_lines(before) != _protected_extern_lines(after):
        errors.append("Protected extern declarations changed.")

    if _suite_names(before) != _suite_names(after):
        errors.append("KUnit suite name changed; result collection path would become stale.")

    before_local = _driver_local_block(before)
    after_local = _driver_local_block(after)
    if before_local or after_local:
        if before_local != after_local:
            errors.append("Protected driver-local definitions block changed.")

    frozen = sorted({name for name in (frozen_tests or []) if name})
    before_info = inspect_test_source(before or "")
    after_blocks = _function_blocks(after or "")
    before_blocks = _function_blocks(before or "")
    for test_name in frozen:
        before_body = extract_test_function(before, test_name)
        after_body = extract_test_function(after, test_name)
        if before_body is None:
            warnings.append(f"Frozen test not found before patch: {test_name}")
            continue
        if after_body is None:
            errors.append(f"Frozen test was removed: {test_name}")
            continue
        if before_body != after_body:
            errors.append(f"Frozen test changed: {test_name}")
            continue
        frozen_test_info = next(
            (item for item in before_info.test_functions if item.name == test_name),
            None,
        )
        if frozen_test_info is None:
            continue
        for helper_name in frozen_test_info.call_names:
            if helper_name == test_name:
                continue
            before_helper = before_blocks.get(helper_name)
            if before_helper is None:
                continue
            after_helper = after_blocks.get(helper_name)
            if after_helper != before_helper:
                errors.append(f"Helper used by frozen test changed: {helper_name} used by {test_name}")

    return RegionValidationResult(ok=not errors, errors=errors, warnings=warnings)
