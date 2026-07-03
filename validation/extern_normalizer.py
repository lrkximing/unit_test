import re
from typing import Dict, List, Optional, Set, Tuple


def extern_declared_function_name(declaration: str) -> str:
    match = re.search(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(", declaration or "")
    return match.group(1) if match else ""


def known_fixed_extern_names(scenario_context: Optional[Dict]) -> Set[str]:
    names: Set[str] = set()
    if not isinstance(scenario_context, dict):
        return names

    registry = scenario_context.get("scenario_registry") or {}
    export_function = registry.get("export_function")
    if isinstance(export_function, str) and export_function.strip():
        names.add(export_function.strip())
    for contract in registry.get("scenario_contracts", []) or []:
        if not isinstance(contract, dict):
            continue
        export_function = contract.get("export_function")
        if isinstance(export_function, str) and export_function.strip():
            names.add(export_function.strip())

    target = scenario_context.get("target") or {}
    if isinstance(target, dict):
        wrapper = target.get("wrapper")
        if isinstance(wrapper, str) and wrapper.strip():
            names.add(wrapper.strip())
        for item in target.get("export_interface_details", []) or []:
            if not isinstance(item, dict):
                continue
            prototype = str(item.get("prototype", "") or "")
            function_name = extern_declared_function_name(prototype)
            if function_name:
                names.add(function_name)
    return names


def strip_flexible_extern_declarations(
    test_code: str,
    scenario_context: Optional[Dict] = None,
) -> Tuple[str, List[str]]:
    """Remove repeated fixed extern declarations outside the fixed export block.

    Unknown extern declarations remain in the candidate and are rejected by the
    scenario verifier. Known test_export or boundary-control externs are already
    emitted by the framework, so repeated declarations in flexible code are safe
    to normalize away before build.
    """
    if not test_code:
        return test_code or "", []
    lines = test_code.splitlines(keepends=True)
    output: List[str] = []
    removed: List[str] = []
    known_fixed_externs = known_fixed_extern_names(scenario_context)
    in_test_export_interfaces = False
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if stripped.startswith("/* TEST EXPORT INTERFACES"):
            in_test_export_interfaces = True
            output.append(line)
            i += 1
            continue
        if stripped.startswith("/* ===== Driver Local Definitions BEGIN"):
            in_test_export_interfaces = False

        if not in_test_export_interfaces and re.match(r"^\s*extern\b", line):
            decl_lines = [line]
            j = i + 1
            while ";" not in "".join(decl_lines) and j < len(lines):
                decl_lines.append(lines[j])
                j += 1
            declaration = "".join(decl_lines)
            function_name = extern_declared_function_name(declaration)
            if function_name in known_fixed_externs:
                removed.append(re.sub(r"\s+", " ", declaration.strip()))
                i = j
                continue

        output.append(line)
        i += 1
    return "".join(output), removed
