from typing import Dict, List, Set

from scenario.fact_model import ScenarioRegistry
from scenario.hardware_boundary_analyzer import analyze_boundary_candidates
from scenario.scenario_candidate_graph import (
    build_scenario_candidates,
    scenario_contracts_from_candidates,
)
from scenario.source_fact_extractor import extract_source_facts


REGISTRY_VERSION = 2


def _wrapper_name(function_name: str) -> str:
    return f"{function_name}_test_export"


def rebuild_scenarios_for_registry(registry: ScenarioRegistry) -> None:
    registry.scenario_candidates = build_scenario_candidates(
        target_function=registry.target_function,
        wrapper=registry.export_function,
        facts=registry.source_facts,
        boundaries=registry.boundary_candidates,
    )
    registry.scenario_contracts = scenario_contracts_from_candidates(registry.scenario_candidates)


def build_scenario_registry(parse_result, function, export_interfaces=None) -> ScenarioRegistry:
    facts, closure = extract_source_facts(parse_result, function)
    boundaries = analyze_boundary_candidates(parse_result, facts)
    wrapper = _wrapper_name(function.name)
    registry = ScenarioRegistry(
        version=REGISTRY_VERSION,
        target_function=function.name,
        export_function=wrapper,
        source_facts=facts,
        boundary_candidates=boundaries,
        scenario_candidates=[],
        scenario_contracts=[],
        internal_call_closure=closure,
    )
    rebuild_scenarios_for_registry(registry)
    return registry


def _contract_evidence_ids(registry_dict: Dict) -> Set[str]:
    ids: Set[str] = set()

    def visit(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key in {"fact_id", "source_fact_id", "target_fact_id", "guard_fact_id"} and isinstance(item, str):
                    ids.add(item)
                elif key in {"fact_ids", "source_fact_ids", "evidence_ids", "source_anchors", "classification_evidence_ids"}:
                    if isinstance(item, list):
                        ids.update(entry for entry in item if isinstance(entry, str))
                else:
                    visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    visit(registry_dict.get("scenario_contracts", []))
    visit(registry_dict.get("boundary_candidates", []))
    return ids


def _generation_source_fact_projection(registry_dict: Dict) -> List[Dict]:
    target_function = registry_dict.get("target_function", "")
    evidence_ids = _contract_evidence_ids(registry_dict)
    projected: List[Dict] = []
    seen: Set[str] = set()
    for fact in registry_dict.get("source_facts", []) or []:
        fact_id = fact.get("fact_id", "")
        if fact.get("function") != target_function and fact_id not in evidence_ids:
            continue
        if fact_id in seen:
            continue
        seen.add(fact_id)
        projected.append(fact)
    return projected


def _export_interface_detail(interface) -> Dict:
    return {
        "prototype": getattr(interface, "prototype", str(interface)),
        "source_symbol": getattr(interface, "source_symbol", ""),
        "source_kind": getattr(interface, "source_kind", ""),
        "description": getattr(interface, "description", ""),
        "boundary_id": getattr(interface, "boundary_id", ""),
        "boundary_expression": getattr(interface, "boundary_expression", ""),
        "boundary_control_role": getattr(interface, "boundary_control_role", ""),
    }


def registry_to_generation_context(registry: ScenarioRegistry, export_interfaces=None) -> Dict:
    registry_dict = registry.to_dict()
    projected_source_facts = _generation_source_fact_projection(registry_dict)
    return {
        "version": registry.version,
        "target": {
            "function": registry.target_function,
            "wrapper": registry.export_function,
            "export_interfaces": [
                getattr(interface, "prototype", str(interface)) for interface in export_interfaces or []
            ],
            "export_interface_details": [
                _export_interface_detail(interface) for interface in export_interfaces or []
            ],
        },
        "scenario_registry": registry_dict,
        "source_facts": projected_source_facts,
        "boundary_candidates": registry_dict["boundary_candidates"],
        "scenario_contracts": registry_dict["scenario_contracts"],
        "analysis_scope": {
            "generation_source_facts": (
                "target-function facts plus helper facts directly cited by boundary/scenario evidence"
            ),
            "full_static_evidence_graph": "scenario_registry.source_facts",
            "internal_call_closure": registry.internal_call_closure,
        },
        "scenario_policy": {
            "scenario_candidate_rule": "Scenario candidates are derived from SourceFact nodes and relation_edges before they are materialized as contracts; they are not a closed taxonomy.",
            "hardware_environment_rule": "Hardware/environment behavior is expressed as scenario-level constraints: boundary, preconditions, required effects, and runtime witnesses. The context does not prescribe a mock technique.",
            "scenario_check_rule": "Scenario checks guide test construction and help preserve scenario identity during repair; they are not used as a standalone assertion-quality metric.",
            "runtime_witness_rule": "Runtime witnesses help decide whether the generated test reached the intended scenario.",
        },
        "source": "scenario_context_static_evidence_graph",
        "analysis_backend": "tree_sitter",
    }
