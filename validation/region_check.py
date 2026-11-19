import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

from validation.test_inspector import (C_PARSER, _node_text, _walk, _find_matching_brace as matching_brace,
                                       inspect_test_source, test_function_map)


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
    if C_PARSER is not None:
        data = code.encode()
        result = {}
        for node in _walk(C_PARSER.parse(data).root_node):
            if node.type != "function_definition":
                continue
            from validation.test_inspector import _function_name_from_definition
            name = _function_name_from_definition(node, data)
            if name:
                result[name] = _node_text(data, node)
        return result
    blocks: Dict[str, str] = {}
    pattern = re.compile(
        r"(?m)^[A-Za-z_][A-Za-z0-9_\s\*]*\s+([A-Za-z_][A-Za-z0-9_]*)\s*\([^;{}]*\)\s*\{"
    )
    for match in pattern.finditer(code or ""):
        name = match.group(1)
        open_idx = code.find("{", match.end() - 1)
        if open_idx < 0:
            continue
        end_idx = matching_brace(code, open_idx)
        if end_idx is None:
            continue
        blocks[name] = code[match.start() : end_idx + 1]
    return blocks


def _file_declarations(code):
    """Map referenced globals/types/macros to their complete source declarations."""
    if C_PARSER is None:
        return {}
    data = code.encode()
    result = {}
    for node in C_PARSER.parse(data).root_node.named_children:
        if node.type in {"comment", "function_definition", "preproc_include"}:
            continue
        owners = []
        for declarator in node.children_by_field_name("declarator"):
            while declarator.child_by_field_name("declarator") is not None:
                declarator = declarator.child_by_field_name("declarator")
            if declarator.type in {"identifier", "type_identifier"}:
                owners.append(declarator)
        if node.type in {"struct_specifier", "union_specifier", "enum_specifier", "preproc_def", "preproc_function_def"}:
            name = node.child_by_field_name("name")
            if name is not None:
                owners.append(name)
        for owner in owners:
            result.setdefault(_node_text(data, owner), []).append(_node_text(data, node))
    return {name: sorted(set(values)) for name, values in result.items()}


def _suite_contract(code):
    suites = re.findall(r"(?:static\s+)?struct\s+kunit_suite\s+\w+\s*=\s*\{.*?\};", code, re.S)
    registrations = re.findall(r"\bkunit_test_suites?\s*\([^;]*\);", code)
    return suites, registrations


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
    if _suite_contract(before) != _suite_contract(after):
        errors.append("Protected KUnit suite registration/setup changed.")
    lifecycle = r"/\* ===== Framework Hook Lifecycle BEGIN ===== \*/.*?/\* ===== Framework Hook Lifecycle END ===== \*/"
    if re.findall(lifecycle, before, re.S) != re.findall(lifecycle, after, re.S):
        errors.append("Protected framework hook lifecycle changed.")
    from generation.mock_interfaces import BEGIN, END
    def mock_block(code):
        return code[code.index(BEGIN):code.index(END) + len(END)] if BEGIN in code and END in code else ""
    if mock_block(before) != mock_block(after):
        errors.append("Protected mock declarations changed.")

    before_local = _driver_local_block(before)
    after_local = _driver_local_block(after)
    if before_local or after_local:
        if before_local != after_local:
            errors.append("Protected driver-local definitions block changed.")

    frozen = sorted({name for name in (frozen_tests or []) if name})
    before_info = inspect_test_source(before or "")
    after_blocks = _function_blocks(after or "")
    before_blocks = _function_blocks(before or "")
    after_info = inspect_test_source(after or "")
    before_decls, after_decls = _file_declarations(before), _file_declarations(after)
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
        after_test = next((test for test in after_info.test_functions if test.name == test_name), None)
        if after_test and (after_test.scenario_ids, after_test.variant_id) != (frozen_test_info.scenario_ids, frozen_test_info.variant_id):
            errors.append(f"Frozen test binding changed: {test_name}")
        pending, seen = [test_name], set()
        while pending:
            name = pending.pop()
            if name in seen:
                continue
            seen.add(name)
            if name in before_blocks:
                if before_blocks[name] != after_blocks.get(name):
                    errors.append(f"Helper used by frozen test changed: {name} used by {test_name}")
                pending.extend(before_info.function_identifier_names.get(name, []))
            elif name in before_decls:
                # Case-array growth is framework registration, not a mock-state change.
                declarations = [d for d in before_decls[name] if 'struct kunit_case' not in d and 'struct kunit_suite' not in d]
                corresponding = [d for d in after_decls.get(name, []) if 'struct kunit_case' not in d and 'struct kunit_suite' not in d]
                if declarations != corresponding:
                    errors.append(f"State/type/macro used by frozen test changed: {name} used by {test_name}")
                for declaration in declarations:
                    pending.extend(re.findall(r"\b[A-Za-z_]\w*\b", declaration))

    return RegionValidationResult(ok=not errors, errors=errors, warnings=warnings)
