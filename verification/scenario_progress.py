from typing import Dict, List, Set

from validation.test_inspector import inspect_test_source


def scenario_status_complete(scenario_runtime_status: Dict) -> bool:
    scenarios = scenario_runtime_status.get("scenarios", []) or []
    if not scenarios:
        return True
    return all(item.get("status") in {"PASSED", "UNREALIZABLE_IN_CURRENT_HARNESS"} for item in scenarios)


def scenario_repair_targets(scenario_runtime_status: Dict) -> List[Dict]:
    return [
        item
        for item in scenario_runtime_status.get("scenarios", []) or []
        if item.get("status") not in {"PASSED", "UNREALIZABLE_IN_CURRENT_HARNESS"}
    ]


def stable_tests_from_scenario_status(scenario_runtime_status: Dict) -> Set[str]:
    stable: Set[str] = set()
    for item in scenario_runtime_status.get("scenarios", []) or []:
        if item.get("status") != "PASSED":
            continue
        stable.update(item.get("test_functions", []) or [])
    return stable


def scenario_status_repair_log(scenario_runtime_status: Dict) -> str:
    targets = scenario_repair_targets(scenario_runtime_status)
    if not targets:
        return "All scenario contracts are satisfied."
    lines = ["Scenario contracts that still need repair:"]
    for item in targets:
        tests = ", ".join(item.get("test_functions", []) or []) or "<no bound test>"
        lines.append(
            "- scenario_id={scenario}; status={status}; tests={tests}; reason={reason}; "
            "witness_hits={witnesses}; effect_hits={effects}".format(
                scenario=item.get("scenario_id", ""),
                status=item.get("status", ""),
                tests=tests,
                reason=item.get("reason", ""),
                witnesses=item.get("runtime_witness_hits", []),
                effects=item.get("runtime_effect_hits", []),
            )
        )
    return "\n".join(lines)


def scenario_test_bindings(
    test_code: str,
    scenario_context: Dict,
    scenario_runtime_status: Dict,
    kunit_summary: Dict,
) -> List[Dict]:
    registry = (scenario_context or {}).get("scenario_registry") or scenario_context or {}
    active = None
    if "active_scenario_ids" in registry:
        active_ids = registry.get("active_scenario_ids", []) or []
        active = {item for item in active_ids if isinstance(item, str)}
    contracts = {
        contract.get("scenario_id", ""): contract
        for contract in registry.get("scenario_contracts", []) or []
        if isinstance(contract, dict)
        and (active is None or contract.get("scenario_id") in active)
    }
    test_status = {
        test.get("name", ""): test.get("status", "")
        for test in kunit_summary.get("tests", []) or []
        if isinstance(test, dict)
    }
    scenario_status = {
        item.get("scenario_id", ""): item
        for item in scenario_runtime_status.get("scenarios", []) or []
        if isinstance(item, dict)
    }
    bindings: List[Dict] = []
    for test_function in inspect_test_source(test_code or "").test_functions:
        for scenario_id in test_function.scenario_ids:
            contract = contracts.get(scenario_id, {})
            status = scenario_status.get(scenario_id, {})
            bindings.append(
                {
                    "test_function": test_function.name,
                    "variant_id": test_function.variant_id,
                    "kunit_status": test_status.get(test_function.name, "unknown"),
                    "scenario_id": scenario_id,
                    "scenario_status": status.get("status", "UNKNOWN"),
                    "scenario_reason": status.get("reason", ""),
                    "runtime_witness_hits": status.get("runtime_witness_hits", []),
                    "runtime_effect_hits": status.get("runtime_effect_hits", []),
                    "source_anchors": contract.get("source_anchors", []),
                    "dependent_boundaries": contract.get("dependent_boundaries", []),
                    "derivation": contract.get("derivation", ""),
                }
            )
    return bindings
