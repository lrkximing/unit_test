from dataclasses import dataclass, field
import re
from typing import Dict, List, Set

from validation.test_inspector import inspect_test_source
from verification.kunit_binding_extractor import collect_kunit_bindings
from verification.assertion_quality import (
    binding_is_nontrivial_assertion,
    effective_check_ids,
)
from scenario.harness_feasibility import (
    active_contracts,
    active_scenario_ids,
)


TEST_EXPORT_INTERFACES = "/* TEST EXPORT INTERFACES - DO NOT MODIFY */"
DRIVER_LOCAL_BEGIN = "/* ===== Driver Local Definitions BEGIN ===== */"
MOCK_MARKER_PATTERN = re.compile(
    r"RACA_MOCK\s*:\s*boundary=([A-Za-z0-9_]+)\s*;\s*original=([^;]+)\s*;\s*replacement=([A-Za-z_][A-Za-z0-9_]*)"
)


@dataclass
class ScenarioStaticResult:
    ok: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    covered_scenarios: List[str] = field(default_factory=list)
    covered_checks: List[str] = field(default_factory=list)
    covered_witnesses: List[str] = field(default_factory=list)
    covered_effects: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "ok": self.ok,
            "errors": self.errors,
            "warnings": self.warnings,
            "covered_scenarios": self.covered_scenarios,
            "covered_checks": self.covered_checks,
            "covered_witnesses": self.covered_witnesses,
            "covered_effects": self.covered_effects,
        }


def is_blocking_scenario_static_error(error: str) -> bool:
    """Return True for static findings that should stop a candidate before build.

    Keep this set deliberately small.  Static scenario validation is an audit
    and repair signal; it should not prevent build/KUnit from running unless the
    candidate damages the harness contract or tries to replace production code.
    Otherwise we lose the normal iteration path where build/runtime failures are
    fed back to the LLM for repair.
    """
    message = error or ""
    blocking_prefixes = (
        "Generated test code redefines original driver function",
        "Scenario contract without scenario_id.",
    )
    if message.startswith(blocking_prefixes):
        return True
    if message.startswith("Boundary ") and " is a direct external call, but the test redefines " in message:
        return True
    return False


def blocking_scenario_static_errors(errors: List[str]) -> List[str]:
    return [error for error in errors or [] if is_blocking_scenario_static_error(error)]


def nonblocking_scenario_static_findings(errors: List[str]) -> List[str]:
    return [error for error in errors or [] if not is_blocking_scenario_static_error(error)]


def _registry(plan_or_registry: Dict) -> Dict:
    if not isinstance(plan_or_registry, dict):
        return {}
    if "scenario_registry" in plan_or_registry:
        return plan_or_registry.get("scenario_registry") or {}
    if "scenario_contracts" in plan_or_registry:
        return plan_or_registry
    return {}


def _target_export_interface_details(plan_or_registry: Dict) -> List[Dict]:
    if not isinstance(plan_or_registry, dict):
        return []
    target = plan_or_registry.get("target") or {}
    if isinstance(target, dict):
        details = target.get("export_interface_details") or []
        return [item for item in details if isinstance(item, dict)]
    return []


def _function_name_from_prototype(prototype: str) -> str:
    match = re.search(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(", prototype or "")
    return match.group(1) if match else ""


def _boundary_hook_controls(plan_or_registry: Dict) -> Dict[str, Dict[str, Dict[str, str]]]:
    controls: Dict[str, Dict[str, Dict[str, str]]] = {}
    for item in _target_export_interface_details(plan_or_registry):
        if item.get("source_kind") != "boundary_hook":
            continue
        boundary_id = str(item.get("boundary_id", "") or "").strip()
        role = str(item.get("boundary_control_role", "") or "").strip()
        if not boundary_id or not role:
            continue
        controls.setdefault(boundary_id, {})[role] = {
            "name": _function_name_from_prototype(str(item.get("prototype", "") or "")),
            "prototype": str(item.get("prototype", "") or ""),
            "boundary_expression": str(item.get("boundary_expression", "") or ""),
        }
    return controls


def _contracts(registry: Dict) -> List[Dict]:
    return active_contracts(registry)


def _test_functions_for_scenario(info, scenario_id: str):
    return [test_function for test_function in info.test_functions if scenario_id in test_function.scenario_ids]


def _check_ids_from_bindings(bindings) -> Set[str]:
    ids: Set[str] = set()
    for binding in bindings:
        ids.update(binding.check_ids)
    return ids


def _witness_ids_from_bindings(bindings) -> Set[str]:
    ids: Set[str] = set()
    for binding in bindings:
        ids.update(binding.witness_ids)
    return ids


def _effect_ids_from_bindings(bindings) -> Set[str]:
    ids: Set[str] = set()
    for binding in bindings:
        ids.update(getattr(binding, "effect_ids", []))
    return ids


def _direct_boundary_redefinition_errors(info, registry: Dict) -> List[str]:
    errors: List[str] = []
    defined_functions = set(info.function_definitions)
    defined_macros = set(info.macro_definitions)
    for candidate in registry.get("boundary_candidates", []) or []:
        if not isinstance(candidate, dict):
            continue
        if candidate.get("source_fact_kind") != "CALL":
            continue
        expression = str(candidate.get("expression", "")).strip()
        if not expression:
            continue
        if expression in defined_functions or expression in defined_macros:
            errors.append(
                f"Boundary {candidate.get('candidate_id')} is a direct external call, but the test redefines {expression}; this does not prove the boundary was intercepted."
            )
    return errors


def _driver_function_redefinition_errors(info, registry: Dict) -> List[str]:
    errors: List[str] = []
    defined_functions = set(info.function_definitions)
    driver_functions: Set[str] = set()

    target = str(registry.get("target_function", "") or "").strip()
    if target:
        driver_functions.add(target)

    for function_name in registry.get("internal_call_closure", []) or []:
        if isinstance(function_name, str) and function_name.strip():
            driver_functions.add(function_name.strip())

    for function_name in sorted(driver_functions & defined_functions):
        errors.append(
            f"Generated test code redefines original driver function `{function_name}`. "
            "Call the target only through test_export wrappers and do not replace driver-internal helpers in the test file."
        )
    return errors


def _fixed_external_symbol_names(registry: Dict, plan_or_registry: Dict) -> Set[str]:
    names: Set[str] = set()
    export_function = registry.get("export_function")
    if isinstance(export_function, str) and export_function.strip():
        names.add(export_function.strip())
    for contract in registry.get("scenario_contracts", []) or []:
        if not isinstance(contract, dict):
            continue
        export_function = contract.get("export_function")
        if isinstance(export_function, str) and export_function.strip():
            names.add(export_function.strip())
    for item in _target_export_interface_details(plan_or_registry):
        name = _function_name_from_prototype(str(item.get("prototype", "") or ""))
        if name:
            names.add(name)
    target = plan_or_registry.get("target") if isinstance(plan_or_registry, dict) else {}
    if isinstance(target, dict):
        wrapper = target.get("wrapper")
        if isinstance(wrapper, str) and wrapper.strip():
            names.add(wrapper.strip())
    return names


def _extern_declaration_errors(test_code: str, registry: Dict, plan_or_registry: Dict) -> List[str]:
    errors: List[str] = []
    code = test_code or ""
    protected_start = code.find(TEST_EXPORT_INTERFACES)
    protected_end = code.find(DRIVER_LOCAL_BEGIN, protected_start) if protected_start >= 0 else -1
    fixed_extern_names = _fixed_external_symbol_names(registry, plan_or_registry)

    for match in re.finditer(r"(?m)^\s*extern\s+[^;]+;", code):
        pos = match.start()
        in_protected_export_block = (
            protected_start >= 0
            and protected_end >= 0
            and protected_start <= pos < protected_end
        )
        if in_protected_export_block:
            continue
        declaration = re.sub(r"\s+", " ", match.group(0).strip())
        declared_name = _function_name_from_prototype(declaration)
        if declared_name in fixed_extern_names:
            continue
        errors.append(
            "Generated flexible test code declares an external symbol outside the fixed "
            f"test_export interface block: `{declaration}`. Define the helper/fake in this "
            "test file or use an existing test_export interface; do not invent harness APIs."
        )
    return errors


def _dead_helper_findings(test_code: str, info) -> tuple[List[str], List[str]]:
    errors: List[str] = []
    warnings: List[str] = []
    registered = set(info.registered_tests)
    test_names = {item.name for item in info.test_functions}
    defined = set(info.function_definitions)
    mock_replacements = {binding["replacement"] for binding in _explicit_mock_bindings(test_code or "")}
    reachable: Set[str] = set()
    worklist = list(registered & defined)
    while worklist:
        name = worklist.pop()
        if name in reachable:
            continue
        reachable.add(name)
        referenced = set((getattr(info, "function_call_names", {}) or {}).get(name, []))
        referenced.update((getattr(info, "function_identifier_names", {}) or {}).get(name, []))
        for callee in referenced:
            if callee in defined and callee not in reachable:
                worklist.append(callee)
    for name in info.function_definitions:
        if name in registered or name in test_names:
            continue
        if name not in reachable:
            looks_like_boundary_fake = (
                name in mock_replacements
                or name.startswith("__wrap_")
                or name.startswith("fake_")
                or name.startswith("mock_")
            )
            message = f"Helper/fake function {name} is defined but never referenced by the generated tests."
            if looks_like_boundary_fake:
                errors.append(message)
            else:
                warnings.append(message)
    return errors, warnings


def _constraint_effect_ids(contract: Dict) -> List[str]:
    ids: List[str] = []
    for constraint in contract.get("hardware_environment_constraints", []) or []:
        for effect in constraint.get("required_effects", []) or []:
            effect_id = effect.get("effect_id", "")
            if effect_id:
                ids.append(effect_id)
    return ids


def _constraint_boundary_ids_requiring_control(contract: Dict) -> Set[str]:
    ids: Set[str] = set()
    for constraint in contract.get("hardware_environment_constraints", []) or []:
        if constraint.get("required_effects"):
            boundary_id = constraint.get("boundary_id", "")
            if boundary_id:
                ids.add(boundary_id)
    return ids


def _constraint_witness_ids(contract: Dict) -> List[str]:
    ids: List[str] = []
    for constraint in contract.get("hardware_environment_constraints", []) or []:
        for witness in constraint.get("runtime_witnesses", []) or []:
            witness_id = witness.get("witness_id", "")
            if witness_id:
                ids.append(witness_id)
    return ids


def _check_satisfied_by_runtime_witness(check: Dict, contract: Dict, local_witness_ids: Set[str]) -> bool:
    if check.get("kind") != "BoundaryNotCalled":
        return False
    target = check.get("target", "")
    if not target:
        return False
    for witness in contract.get("runtime_witnesses", []) or []:
        if witness.get("target") != target:
            continue
        if witness.get("kind") not in {"BOUNDARY_NOT_REACHED", "BOUNDARY_NOT_CALLED"}:
            continue
        witness_id = witness.get("witness_id", "")
        if witness_id and witness_id in local_witness_ids:
            return True
    return False


def _check_satisfied_by_guard_return(check: Dict, contract: Dict, local_check_ids: Set[str]) -> bool:
    if check.get("kind") != "BoundaryNotCalled":
        return False
    if contract.get("derivation") != "source_fact_graph:guard_return":
        return False
    target = check.get("target", "")
    if not target:
        return False
    has_not_reached_witness = any(
        witness.get("target") == target
        and witness.get("kind") in {"BOUNDARY_NOT_REACHED", "BOUNDARY_NOT_CALLED"}
        for witness in contract.get("runtime_witnesses", []) or []
    )
    if not has_not_reached_witness:
        return False
    return any(
        scenario_check.get("check_id") in local_check_ids
        and str(scenario_check.get("kind", "")).startswith("Return")
        for scenario_check in contract.get("scenario_checks", []) or []
    )


def _witness_satisfied_by_boundary_check(witness: Dict, contract: Dict, local_check_ids: Set[str]) -> bool:
    if witness.get("kind") not in {"BOUNDARY_NOT_REACHED", "BOUNDARY_NOT_CALLED"}:
        return False
    target = witness.get("target", "")
    if not target:
        return False
    for check in contract.get("scenario_checks", []) or []:
        if check.get("kind") != "BoundaryNotCalled":
            continue
        if check.get("target") != target:
            continue
        check_id = check.get("check_id", "")
        if check_id and check_id in local_check_ids:
            return True
    return False


def _boundary_by_id(registry: Dict) -> Dict[str, Dict]:
    return {
        candidate.get("candidate_id", ""): candidate
        for candidate in registry.get("boundary_candidates", []) or []
        if isinstance(candidate, dict) and candidate.get("candidate_id")
    }


def _is_direct_external_call_boundary(boundary: Dict) -> bool:
    return isinstance(boundary, dict) and boundary.get("source_fact_kind") == "CALL"


def _explicit_mock_bindings(test_code: str) -> List[Dict[str, str]]:
    bindings: List[Dict[str, str]] = []
    for match in MOCK_MARKER_PATTERN.finditer(test_code or ""):
        bindings.append(
            {
                "boundary_id": match.group(1).strip(),
                "original": match.group(2).strip(),
                "replacement": match.group(3).strip(),
                "source": "RACA_MOCK",
            }
        )
    return bindings


def _hook_setter_mock_bindings(test_code: str, plan_or_registry: Dict) -> List[Dict[str, str]]:
    bindings: List[Dict[str, str]] = []
    for boundary_id, controls in _boundary_hook_controls(plan_or_registry).items():
        setter = controls.get("set_hook", {})
        setter_name = setter.get("name", "")
        if not setter_name:
            continue
        pattern = re.compile(
            r"\b" + re.escape(setter_name) + r"\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)"
        )
        for match in pattern.finditer(test_code or ""):
            replacement = match.group(1).strip()
            if replacement in {"NULL", "null"}:
                continue
            bindings.append(
                {
                    "boundary_id": boundary_id,
                    "original": setter.get("boundary_expression", "").strip(),
                    "replacement": replacement,
                    "source": "boundary_hook_setter",
                    "setter": setter_name,
                }
            )
    return bindings


def mock_bindings_from_test(test_code: str, plan_or_registry: Dict) -> List[Dict[str, str]]:
    """Return explicit RACA_MOCK bindings plus bindings inferred from exported hook setters."""
    bindings = _explicit_mock_bindings(test_code)
    seen = {
        (item.get("boundary_id", ""), item.get("replacement", ""), item.get("source", ""))
        for item in bindings
    }
    for item in _hook_setter_mock_bindings(test_code, plan_or_registry):
        key = (item.get("boundary_id", ""), item.get("replacement", ""), item.get("source", ""))
        if key in seen:
            continue
        seen.add(key)
        bindings.append(item)
    return bindings


def _last_identifier(expression: str) -> str:
    identifiers = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", expression or "")
    return identifiers[-1] if identifiers else ""


def _mock_replacement_is_connected(test_code: str, replacement: str) -> bool:
    if not replacement:
        return False
    code_without_markers = MOCK_MARKER_PATTERN.sub("", test_code or "")
    occurrences = re.findall(r"\b" + re.escape(replacement) + r"\b", code_without_markers)
    return len(occurrences) > 1


def _mock_binding_errors(test_code: str, info, registry: Dict, plan_or_registry: Dict) -> List[str]:
    errors: List[str] = []
    bindings = mock_bindings_from_test(test_code, plan_or_registry)
    bindings_by_boundary: Dict[str, List[Dict[str, str]]] = {}
    for binding in bindings:
        bindings_by_boundary.setdefault(binding["boundary_id"], []).append(binding)

    boundaries = _boundary_by_id(registry)
    defined_replacements = set(info.function_definitions) | set(info.macro_definitions)
    hook_controls = _boundary_hook_controls(plan_or_registry)

    for contract in _contracts(registry):
        scenario_id = contract.get("scenario_id", "")
        for boundary_id in sorted(_constraint_boundary_ids_requiring_control(contract)):
            boundary = boundaries.get(boundary_id)
            if boundary_id in bindings_by_boundary:
                continue
            if _is_direct_external_call_boundary(boundary) and boundary_id in hook_controls:
                errors.append(
                    f"Scenario {scenario_id} controls direct boundary {boundary_id}, but no exported hook setter installs a fake implementation."
                )
                continue
            if _is_direct_external_call_boundary(boundary):
                continue
            errors.append(
                f"Scenario {scenario_id} controls boundary {boundary_id} but has no RACA_MOCK binding."
            )

    for binding in bindings:
        boundary_id = binding["boundary_id"]
        boundary = boundaries.get(boundary_id)
        if boundary is None:
            errors.append(f"RACA_MOCK references unknown boundary: {boundary_id}.")
            continue
        expected_original = str(boundary.get("expression", "")).strip()
        if expected_original and binding["original"] != expected_original:
            errors.append(
                f"RACA_MOCK for {boundary_id} declares original `{binding['original']}`, expected `{expected_original}`."
            )
        if binding["replacement"] not in defined_replacements:
            errors.append(
                f"RACA_MOCK replacement `{binding['replacement']}` for {boundary_id} is not defined in the test file."
            )
        elif not _mock_replacement_is_connected(test_code or "", binding["replacement"]):
            errors.append(
                f"RACA_MOCK replacement `{binding['replacement']}` for {boundary_id} is defined but not connected to any setup, callback, hook, or test path."
            )
        original_callee = _last_identifier(binding["original"])
        if boundary.get("source_fact_kind") == "CALL" and original_callee:
            for test_function in info.test_functions:
                if original_callee in test_function.call_names:
                    errors.append(
                        f"Test {test_function.name} directly calls original boundary `{original_callee}`; use RACA_MOCK replacement `{binding['replacement']}` instead."
                    )
    return errors


def _registered_test_binding_errors(info, registry: Dict) -> List[str]:
    errors: List[str] = []
    active = active_scenario_ids(registry)
    known_active = active
    if known_active is None:
        known_active = {
            contract.get("scenario_id", "")
            for contract in registry.get("scenario_contracts", []) or []
            if isinstance(contract, dict) and contract.get("scenario_id")
        }
    tests_by_name = {test.name: test for test in info.test_functions}

    for registered_name in sorted(info.registered_tests):
        test_function = tests_by_name.get(registered_name)
        if test_function is None:
            continue
        scenario_ids = list(test_function.scenario_ids or [])
        if not scenario_ids:
            errors.append(
                f"Registered KUnit test {registered_name} has no RACA_SCENARIO binding; generated tests must map to scenario contracts."
            )
            continue
        if len(scenario_ids) > 1:
            errors.append(
                f"Registered KUnit test {registered_name} binds multiple scenarios: {scenario_ids}."
            )
        for scenario_id in scenario_ids:
            if scenario_id not in known_active:
                errors.append(
                    f"Registered KUnit test {registered_name} binds inactive or unknown scenario: {scenario_id}."
                )
    return errors


def verify_scenario_contracts(test_code: str, plan_or_registry: Dict) -> ScenarioStaticResult:
    registry = _registry(plan_or_registry)
    if not registry:
        return ScenarioStaticResult(ok=True, warnings=["No scenario registry found; falling back to legacy checks."])

    errors: List[str] = []
    warnings: List[str] = []
    covered_scenarios: List[str] = []
    covered_checks: Set[str] = set()
    covered_witnesses: Set[str] = set()
    covered_effects: Set[str] = set()
    info = inspect_test_source(test_code or "")
    dead_helper_errors, dead_helper_warnings = _dead_helper_findings(test_code or "", info)
    errors.extend(_extern_declaration_errors(test_code or "", registry, plan_or_registry))
    errors.extend(_direct_boundary_redefinition_errors(info, registry))
    errors.extend(_driver_function_redefinition_errors(info, registry))
    errors.extend(dead_helper_errors)
    warnings.extend(dead_helper_warnings)
    errors.extend(_mock_binding_errors(test_code or "", info, registry, plan_or_registry))
    errors.extend(_registered_test_binding_errors(info, registry))

    for contract in _contracts(registry):
        scenario_id = contract.get("scenario_id", "")
        if not scenario_id:
            errors.append("Scenario contract without scenario_id.")
            continue
        tests = _test_functions_for_scenario(info, scenario_id)
        if not tests:
            errors.append(f"Scenario test not found: {scenario_id}")
            continue
        if len(tests) > 1:
            warnings.append(
                f"Scenario {scenario_id} is bound to multiple test variants: {[test.name for test in tests]}."
            )
        for test_function in tests:
            if test_function.name not in info.registered_tests:
                errors.append(
                    f"Scenario {scenario_id} test function is not registered with KUNIT_CASE: {test_function.name}."
                )
            wrapper = contract.get("export_function", "")
            if wrapper and wrapper not in test_function.call_names:
                errors.append(
                    f"Scenario {scenario_id} test function does not call target wrapper {wrapper}: {test_function.name}."
                )
        covered_scenarios.append(scenario_id)

        scenario_bindings = []
        for test_function in tests:
            test_bindings = collect_kunit_bindings(test_function.full_text)
            scenario_bindings.extend(test_bindings)
            if not any(binding_is_nontrivial_assertion(binding) for binding in test_bindings):
                errors.append(
                    f"Scenario {scenario_id} test {test_function.name} has no non-vacuous target-behavior assertion."
                )

        local_check_ids = _check_ids_from_bindings(scenario_bindings)
        checks_by_id = {
            check.get("check_id", ""): check
            for check in contract.get("scenario_checks", []) or []
            if check.get("check_id")
        }
        effective_local_check_ids = effective_check_ids(scenario_bindings, checks_by_id)
        local_witness_ids = _witness_ids_from_bindings(scenario_bindings)
        local_effect_ids = _effect_ids_from_bindings(scenario_bindings)
        covered_checks.update(local_check_ids)
        covered_witnesses.update(local_witness_ids)
        covered_effects.update(local_effect_ids)

        scenario_checks = contract.get("scenario_checks", []) or []
        if not scenario_checks:
            errors.append(f"Scenario {scenario_id} has no scenario checks.")

        for check in scenario_checks:
            check_id = check.get("check_id", "")
            if not check_id:
                errors.append(f"Scenario {scenario_id} has a scenario check without check_id.")
                continue
            if check_id not in local_check_ids and _check_satisfied_by_guard_return(check, contract, local_check_ids):
                local_check_ids.add(check_id)
                covered_checks.add(check_id)
                effective_local_check_ids.add(check_id)
            if check_id not in local_check_ids and not _check_satisfied_by_runtime_witness(check, contract, local_witness_ids):
                errors.append(f"Scenario {scenario_id} missing scenario check binding: {check_id}.")
            elif check_id not in local_check_ids:
                covered_checks.add(check_id)
            elif check_id not in effective_local_check_ids and not _check_satisfied_by_runtime_witness(
                check, contract, local_witness_ids
            ):
                errors.append(
                    f"Scenario {scenario_id} check {check_id} is bound only to a weak or non-observational assertion."
                )

        for witness in contract.get("runtime_witnesses", []) or []:
            witness_id = witness.get("witness_id", "")
            if not witness_id:
                errors.append(f"Scenario {scenario_id} has a runtime witness without witness_id.")
                continue
            if witness_id not in local_witness_ids and not _witness_satisfied_by_boundary_check(
                witness, contract, local_check_ids
            ):
                errors.append(f"Scenario {scenario_id} missing runtime witness binding: {witness_id}.")
            elif witness_id not in local_witness_ids:
                covered_witnesses.add(witness_id)

        for witness_id in _constraint_witness_ids(contract):
            if witness_id not in local_witness_ids:
                errors.append(f"Scenario {scenario_id} missing hardware-constraint witness binding: {witness_id}.")

        for effect_id in _constraint_effect_ids(contract):
            if effect_id not in local_effect_ids:
                errors.append(f"Scenario {scenario_id} missing required-effect binding: {effect_id}.")

    return ScenarioStaticResult(
        ok=not errors,
        errors=errors,
        warnings=warnings,
        covered_scenarios=sorted(set(covered_scenarios)),
        covered_checks=sorted(covered_checks),
        covered_witnesses=sorted(covered_witnesses),
        covered_effects=sorted(covered_effects),
    )
