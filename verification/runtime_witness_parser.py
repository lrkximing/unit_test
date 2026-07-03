from dataclasses import dataclass, field
import re
from typing import Dict, List, Set

from validation.test_inspector import inspect_test_source
from verification.scenario_static_verifier import verify_scenario_contracts
from scenario.harness_feasibility import active_contracts


WITNESS_MARKER_PATTERN = re.compile(r"RACA_WITNESS\s*:\s*([A-Za-z0-9_]+)\s*:\s*(?:HIT|REACHED|OK|PASS)")
EFFECT_MARKER_PATTERN = re.compile(r"RACA_EFFECT\s*:\s*([A-Za-z0-9_]+)\s*:\s*(?:HIT|APPLIED|OK|PASS)")


@dataclass
class ScenarioRuntimeStatus:
    scenario_id: str
    test_functions: List[str] = field(default_factory=list)
    test_variants: List[Dict[str, str]] = field(default_factory=list)
    static_valid: bool = False
    buildable: bool = False
    reached: bool = False
    checks_passed: bool = False
    runtime_witness_hits: List[str] = field(default_factory=list)
    runtime_effect_hits: List[str] = field(default_factory=list)
    status: str = "PLANNED"
    reason: str = ""

    def to_dict(self) -> Dict:
        return {
            "scenario_id": self.scenario_id,
            "test_functions": self.test_functions,
            "test_variants": self.test_variants,
            "static_valid": self.static_valid,
            "buildable": self.buildable,
            "reached": self.reached,
            "checks_passed": self.checks_passed,
            "runtime_witness_hits": self.runtime_witness_hits,
            "runtime_effect_hits": self.runtime_effect_hits,
            "status": self.status,
            "reason": self.reason,
        }


def _registry(context: Dict) -> Dict:
    return context.get("scenario_registry") or context


def _contracts(registry: Dict) -> List[Dict]:
    return active_contracts(registry)


def _scenario_test_map(test_code: str) -> Dict[str, List[str]]:
    mapping: Dict[str, List[str]] = {}
    info = inspect_test_source(test_code or "")
    for test_function in info.test_functions:
        for scenario_id in test_function.scenario_ids:
            mapping.setdefault(scenario_id, []).append(test_function.name)
    return mapping


def _scenario_variant_map(test_code: str) -> Dict[str, List[Dict[str, str]]]:
    mapping: Dict[str, List[Dict[str, str]]] = {}
    info = inspect_test_source(test_code or "")
    for test_function in info.test_functions:
        for scenario_id in test_function.scenario_ids:
            mapping.setdefault(scenario_id, []).append(
                {
                    "test_function": test_function.name,
                    "variant_id": test_function.variant_id,
                }
            )
    return mapping


def _test_status_sets(kunit_summary: Dict) -> tuple[Set[str], Set[str]]:
    passed = set(kunit_summary.get("passed", []) or [])
    failed = set(kunit_summary.get("failed", []) or [])
    return passed, failed


def extract_runtime_markers(log_text: str) -> Dict[str, Set[str]]:
    return {
        "witnesses": set(WITNESS_MARKER_PATTERN.findall(log_text or "")),
        "effects": set(EFFECT_MARKER_PATTERN.findall(log_text or "")),
    }


def _expected_witness_ids(contract: Dict) -> Set[str]:
    ids = {item.get("witness_id") for item in contract.get("runtime_witnesses", []) or []}
    for constraint in contract.get("hardware_environment_constraints", []) or []:
        ids.update(item.get("witness_id") for item in constraint.get("runtime_witnesses", []) or [])
    return {item for item in ids if item}


def _expected_effect_ids(contract: Dict) -> Set[str]:
    ids: Set[str] = set()
    for constraint in contract.get("hardware_environment_constraints", []) or []:
        ids.update(item.get("effect_id") for item in constraint.get("required_effects", []) or [])
    return {item for item in ids if item}


def evaluate_scenario_runtime_status(
    test_code: str,
    scenario_context: Dict,
    kunit_summary: Dict,
    buildable: bool = True,
    iteration: int = 0,
    max_attempts: int = 0,
    runtime_log: str = "",
) -> Dict:
    registry = _registry(scenario_context)
    static_result = verify_scenario_contracts(test_code, scenario_context)
    scenario_to_tests = _scenario_test_map(test_code)
    scenario_to_variants = _scenario_variant_map(test_code)
    passed, failed = _test_status_sets(kunit_summary or {})
    covered_scenarios = set(static_result.covered_scenarios)
    statically_covered_witnesses = set(static_result.covered_witnesses)
    runtime_markers = extract_runtime_markers(runtime_log)
    statuses: List[ScenarioRuntimeStatus] = []

    for contract in _contracts(registry):
        scenario_id = contract.get("scenario_id", "")
        tests = scenario_to_tests.get(scenario_id, [])
        item = ScenarioRuntimeStatus(
            scenario_id=scenario_id,
            test_functions=tests,
            test_variants=scenario_to_variants.get(scenario_id, []),
            static_valid=scenario_id in covered_scenarios and not any(
                scenario_id in error for error in static_result.errors
            ),
            buildable=buildable,
        )
        expected_witnesses = _expected_witness_ids(contract)
        expected_effects = _expected_effect_ids(contract)
        witness_hits = sorted(
            expected_witnesses
            & (runtime_markers["witnesses"] | statically_covered_witnesses)
        )
        effect_hits = sorted(expected_effects & runtime_markers["effects"])
        item.runtime_witness_hits = witness_hits
        item.runtime_effect_hits = effect_hits
        runtime_markers_required = bool(runtime_log and (expected_witnesses or expected_effects))
        runtime_markers_ok = (
            not runtime_markers_required
            or (
                expected_witnesses.issubset(runtime_markers["witnesses"] | statically_covered_witnesses)
                and expected_effects.issubset(runtime_markers["effects"])
            )
        )
        exhausted = bool(max_attempts and iteration >= max_attempts)
        if not tests:
            item.status = "PLANNED"
            item.reason = "no test function bound to scenario"
        elif any(test in failed for test in tests):
            item.status = "FAILED"
            item.reason = "one or more bound KUnit tests failed"
        elif tests and all(test in passed for test in tests) and item.static_valid and runtime_markers_ok:
            item.reached = True
            item.checks_passed = True
            item.status = "PASSED"
            item.reason = "static scenario contract valid, runtime markers satisfied, and all bound KUnit tests passed"
        elif tests and all(test in passed for test in tests) and item.static_valid:
            item.status = "NOT_REACHED"
            missing_witnesses = sorted(expected_witnesses - (runtime_markers["witnesses"] | statically_covered_witnesses))
            missing_effects = sorted(expected_effects - runtime_markers["effects"])
            item.reason = f"runtime markers missing; witnesses={missing_witnesses}, effects={missing_effects}"
        elif tests and all(test in passed for test in tests):
            item.status = "NOT_REACHED"
            item.reason = "KUnit tests passed but required scenario effects or witnesses were not validated"
        else:
            item.status = "NOT_REACHED"
            item.reason = "scenario tests are not present in passed or failed KUnit results"
        if exhausted and item.status == "NOT_REACHED":
            item.status = "UNREALIZABLE_IN_CURRENT_HARNESS"
            item.reason = "hardware environment constraints remained unsatisfied after the configured repair attempts"
        statuses.append(item)

    summary = {
        "total_scenarios": len(statuses),
        "generated_scenarios": len([item for item in statuses if item.test_functions]),
        "static_valid_scenarios": len([item for item in statuses if item.static_valid]),
        "buildable_scenarios": len([item for item in statuses if item.buildable and item.test_functions]),
        "reached_scenarios": len([item for item in statuses if item.reached]),
        "passed_scenarios": len([item for item in statuses if item.status == "PASSED"]),
        "failed_scenarios": len([item for item in statuses if item.status == "FAILED"]),
        "not_reached_scenarios": len([item for item in statuses if item.status == "NOT_REACHED"]),
        "unrealizable_scenarios": len([item for item in statuses if item.status == "UNREALIZABLE_IN_CURRENT_HARNESS"]),
        "unresolved_scenarios": len([item for item in statuses if item.status in {"NOT_REACHED", "UNREALIZABLE_IN_CURRENT_HARNESS"}]),
        "scenarios": [item.to_dict() for item in statuses],
        "static_errors": static_result.errors,
        "static_warnings": static_result.warnings,
    }
    return summary
