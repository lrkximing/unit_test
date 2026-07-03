import time
from typing import Dict, Iterable, List


def append_revision(
    scenario_context: Dict,
    reason: str,
    affected_scenarios: Iterable[str] = (),
    source_fact_ids: Iterable[str] = (),
) -> Dict:
    registry = scenario_context.get("scenario_registry") or scenario_context
    history: List[Dict] = registry.setdefault("revision_history", [])
    current_version = int(registry.get("version", 1))
    next_version = current_version + 1
    registry["version"] = next_version
    scenario_context["version"] = next_version
    history.append(
        {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "from_version": current_version,
            "to_version": next_version,
            "reason": reason,
            "affected_scenarios": sorted(set(affected_scenarios)),
            "source_fact_ids": sorted(set(source_fact_ids)),
        }
    )
    return scenario_context


def set_scenario_status(scenario_context: Dict, scenario_id: str, status: str) -> None:
    registry = scenario_context.get("scenario_registry") or scenario_context
    for contract in registry.get("scenario_contracts", []) or []:
        if contract.get("scenario_id") == scenario_id:
            contract["status"] = status
            return
