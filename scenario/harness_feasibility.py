from typing import Dict, List, Optional, Set


def registry_from_context(context_or_registry: Dict) -> Dict:
    if not isinstance(context_or_registry, dict):
        return {}
    return context_or_registry.get("scenario_registry") or context_or_registry


def active_scenario_ids(registry: Dict) -> Optional[Set[str]]:
    if not isinstance(registry, dict) or "active_scenario_ids" not in registry:
        return None
    return {item for item in (registry.get("active_scenario_ids") or []) if isinstance(item, str)}


def active_contracts(registry: Dict) -> List[Dict]:
    contracts = [item for item in registry.get("scenario_contracts", []) or [] if isinstance(item, dict)]
    active = active_scenario_ids(registry)
    if active is None:
        return contracts
    return [contract for contract in contracts if contract.get("scenario_id") in active]


def boundary_by_id(registry: Dict) -> Dict[str, Dict]:
    return {
        candidate.get("candidate_id", ""): candidate
        for candidate in registry.get("boundary_candidates", []) or []
        if isinstance(candidate, dict) and candidate.get("candidate_id")
    }


def direct_uncontrollable_scenario_ids(registry: Dict) -> Set[str]:
    """Scenarios that require controlling direct external CALL boundaries.

    A test-file fake cannot intercept a direct call already compiled into the
    production driver.  These scenarios are still generation targets: the repair
    loop should try real object setup, existing wrappers/hooks, or harness-level
    instrumentation before marking them unrealizable after attempts are exhausted.
    This helper is therefore diagnostic/risk metadata, not a generation filter.
    """
    boundaries = boundary_by_id(registry)
    scenario_ids: Set[str] = set()
    for contract in active_contracts(registry):
        scenario_id = contract.get("scenario_id", "")
        if not scenario_id:
            continue
        for constraint in contract.get("hardware_environment_constraints", []) or []:
            if not constraint.get("required_effects"):
                continue
            boundary = boundaries.get(constraint.get("boundary_id", ""))
            if isinstance(boundary, dict) and boundary.get("source_fact_kind") == "CALL":
                scenario_ids.add(scenario_id)
                break
    return scenario_ids


def allowed_generated_scenario_ids(registry: Dict) -> Set[str]:
    active = active_scenario_ids(registry)
    if active is None:
        active = {
            contract.get("scenario_id", "")
            for contract in registry.get("scenario_contracts", []) or []
            if isinstance(contract, dict) and contract.get("scenario_id")
        }
    return active
