from dataclasses import dataclass, field
import re
from typing import Dict, List, Set

from validation.test_inspector import inspect_test_source
from verification.scenario_static_verifier import verify_scenario_contracts
from scenario.harness_feasibility import active_contracts
from scenario.condition_variants import contract_for_variant, planned_variants


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
        if constraint.get("conditional"):
            continue
        ids.update(item.get("witness_id") for item in constraint.get("runtime_witnesses", []) or [])
    return {item for item in ids if item}


def _expected_effect_ids(contract: Dict) -> Set[str]:
    ids: Set[str] = set()
    for constraint in contract.get("hardware_environment_constraints", []) or []:
        if constraint.get("conditional"):
            continue
        ids.update(item.get("effect_id") for item in constraint.get("required_effects", []) or [])
    return {item for item in ids if item}

def runtime_markers_by_test(log_text: str, test_names) -> Dict:
    """Attribute KUnit diagnostics to their named case, or the following TAP case."""
    by_test = {name: {"witnesses": set(), "effects": set(), "assertions": set()} for name in test_names}
    pending = {"witnesses": set(), "effects": set()}
    for line in (log_text or "").splitlines():
        assertion = re.search(r"RACA_ASSERT:([A-Za-z0-9_]+):([A-Za-z0-9_]+):HIT", line)
        if assertion and assertion.group(1) in by_test:
            by_test[assertion.group(1)]["assertions"].add(assertion.group(2))
        markers = extract_runtime_markers(line)
        if markers["witnesses"] or markers["effects"]:
            prefix = line.split("RACA_", 1)[0]
            owner = next((name for name in test_names if re.search(r"\b" + re.escape(name) + r"\s*:", prefix)), None)
            destination = by_test[owner] if owner else pending
            for kind in pending:
                destination[kind].update(markers[kind])
        result = re.search(r"\b(?:not ok|ok)\s+\d+\s+(\S+)", line)
        if result:
            name = result.group(1)
            if name in by_test:
                for kind in pending:
                    by_test[name][kind].update(pending[kind])
            pending = {"witnesses": set(), "effects": set()}
    return by_test


def evaluate_scenario_runtime_status(
    test_code: str, scenario_context: Dict, kunit_summary: Dict,
    buildable: bool = True, iteration: int = 0, max_attempts: int = 0,
    runtime_log: str = "",
) -> Dict:
    registry = _registry(scenario_context)
    static_result = verify_scenario_contracts(test_code, scenario_context)
    scenario_to_tests = _scenario_test_map(test_code)
    passed, failed = _test_status_sets(kunit_summary or {})
    names = [test.name for test in inspect_test_source(test_code or "").test_functions]
    markers = runtime_markers_by_test(runtime_log, names)
    scenarios = []
    for contract in _contracts(registry):
        sid = contract.get("scenario_id", "")
        tests = scenario_to_tests.get(sid, [])
        variants = []
        for name in tests:
            report = static_result.test_reports.get(name, {})
            variant = contract_for_variant(scenario_context, sid, report.get("variant_id", "initial")) or contract
            expected_witnesses = _expected_witness_ids(variant)
            expected_effects = _expected_effect_ids(variant)
            hits = markers[name]
            missing = {"witnesses": sorted(expected_witnesses - hits["witnesses"]),
                       "effects": sorted(expected_effects - hits["effects"]),
                       "assertions": sorted((set(report.get("checks", [])) | {"__semantic__"}) - hits["assertions"])}
            runtime_ok = not any(missing.values())
            valid = bool(report.get("valid"))
            if not buildable:
                status, reason = "NOT_BUILT", "candidate was not built"
            elif name in failed:
                status, reason = "FAILED", "KUnit test failed"
            elif name not in passed or not kunit_summary.get("log_complete", True):
                status, reason = "NOT_REACHED", "no complete passing KUnit result for this test"
            elif not valid:
                status, reason = "NOT_REACHED", "this variant's structural/observation checks are incomplete"
            elif not runtime_ok:
                status, reason = "NOT_REACHED", f"no executed marker evidence for this test: {missing}"
            else:
                status, reason = "PASSED", "per-test KUnit result, observation bindings and required runtime markers verified"
            if max_attempts and iteration >= max_attempts and status == "NOT_REACHED":
                status = "BUDGET_EXHAUSTED"
            variants.append({"test_function": name, "variant_id": report.get("variant_id", "initial"),
                             "status": status, "reason": reason, "static_valid": valid,
                             "runtime_witness_hits": sorted(hits["witnesses"] & expected_witnesses),
                             "runtime_effect_hits": sorted(hits["effects"] & expected_effects),
                             "runtime_assertion_hits": sorted(hits["assertions"]),
                             "missing_runtime_evidence": missing})
        actual_variants = {v["variant_id"] for v in variants}
        required = (planned_variants(scenario_context) if "scenario_variants" in scenario_context
                    else scenario_context.get("coverage_targets", []))
        for target in required:
            if target.get("scenario_id") != sid or target.get("variant_id") in actual_variants:
                continue
            variants.append({"test_function": "", "variant_id": target["variant_id"], "status": "NOT_REACHED",
                             "reason": "planned condition variant has no bound test", "static_valid": False,
                             "runtime_witness_hits": [], "runtime_effect_hits": [], "runtime_assertion_hits": []})
        status = ("PLANNED" if not variants else "PASSED" if all(v["status"] == "PASSED" for v in variants)
                  else "FAILED" if any(v["status"] == "FAILED" for v in variants)
                  else "BUDGET_EXHAUSTED" if any(v["status"] == "BUDGET_EXHAUSTED" for v in variants)
                  else "NOT_REACHED")
        scenarios.append({"scenario_id": sid, "test_functions": tests, "test_variants": variants,
                          "kunit_passed_tests": sorted(set(tests) & passed) if kunit_summary.get("log_complete", True) else [],
                          "static_valid": bool(variants) and all(v["static_valid"] for v in variants),
                          "buildable": buildable, "reached": status == "PASSED", "checks_passed": status == "PASSED",
                          "status": status, "reason": "; ".join(v["reason"] for v in variants) or "no test bound",
                          "runtime_witness_hits": sorted({x for v in variants for x in v["runtime_witness_hits"]}),
                          "runtime_effect_hits": sorted({x for v in variants for x in v["runtime_effect_hits"]})})
    return {
        "total_scenarios": len(scenarios),
        "generated_scenarios": sum(bool(s["test_functions"]) for s in scenarios),
        "static_valid_scenarios": sum(s["static_valid"] for s in scenarios),
        "buildable_scenarios": sum(s["buildable"] and bool(s["test_functions"]) for s in scenarios),
        "reached_scenarios": sum(s["reached"] for s in scenarios),
        "passed_scenarios": sum(s["status"] == "PASSED" for s in scenarios),
        "failed_scenarios": sum(s["status"] == "FAILED" for s in scenarios),
        "not_reached_scenarios": sum(s["status"] == "NOT_REACHED" for s in scenarios),
        "unrealizable_scenarios": 0,
        "unresolved_scenarios": sum(s["status"] != "PASSED" for s in scenarios),
        "scenarios": scenarios, "static_errors": static_result.errors, "static_warnings": static_result.warnings,
    }
