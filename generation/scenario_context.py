import json
from typing import Dict, Iterable, List, Optional, Set

from scenario.boundary_classifier import classify_boundary_candidates_with_optional_llm
from scenario.harness_feasibility import allowed_generated_scenario_ids, direct_uncontrollable_scenario_ids
from scenario.scenario_builder import (
    build_scenario_registry,
    rebuild_scenarios_for_registry,
    registry_to_generation_context,
)


def build_scenario_context(parse_result, function, export_interfaces) -> Dict:
    registry = build_scenario_registry(parse_result, function, export_interfaces)
    context = registry_to_generation_context(registry, export_interfaces)
    active = select_initial_active_scenario_ids(context.get("scenario_registry", {}))
    context["active_scenario_ids"] = active
    context["scenario_registry"]["active_scenario_ids"] = active
    return context


def generate_scenario_context(
    parse_result,
    function,
    local_or_api: str,
    prompt_path: str,
    model,
    tokenizer,
    export_interfaces,
) -> Dict:
    registry = build_scenario_registry(parse_result, function, export_interfaces)
    classify_boundary_candidates_with_optional_llm(
        registry=registry,
        parse_result=parse_result,
        function=function,
        local_or_api=local_or_api,
        prompt_path=prompt_path,
        model=model,
        tokenizer=tokenizer,
    )
    rebuild_scenarios_for_registry(registry)
    context = registry_to_generation_context(registry, export_interfaces)
    active = select_initial_active_scenario_ids(context.get("scenario_registry", {}))
    context["active_scenario_ids"] = active
    context["scenario_registry"]["active_scenario_ids"] = active
    return context


def _contract_fact_ids(contract: Dict) -> Set[str]:
    fact_ids: Set[str] = set()

    def visit(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key in {"fact_id", "source_fact_id", "target_fact_id", "guard_fact_id"} and isinstance(item, str):
                    if item.startswith("F_"):
                        fact_ids.add(item)
                elif key in {"fact_ids", "source_fact_ids", "evidence_ids", "source_anchors", "classification_evidence_ids"}:
                    if isinstance(item, list):
                        fact_ids.update(entry for entry in item if isinstance(entry, str) and entry.startswith("F_"))
                else:
                    visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    visit(contract)
    return fact_ids


def _contract_selection_fact_ids(contract: Dict) -> Set[str]:
    """Facts that define what this scenario is meant to exercise.

    This intentionally avoids recursively consuming every evidence edge in the
    contract.  Environment constraints and relation metadata may mention facts
    that explain dependencies, but a single test scenario should not claim those
    as direct coverage goals unless the scenario anchors/checks name them.
    """
    fact_ids: Set[str] = set()
    for fact_id in contract.get("source_anchors", []) or []:
        if isinstance(fact_id, str) and fact_id.startswith("F_"):
            fact_ids.add(fact_id)

    for key in ("trigger_conditions", "observations", "scenario_checks", "runtime_witnesses"):
        for item in contract.get(key, []) or []:
            if not isinstance(item, dict):
                continue
            for fact_id in item.get("evidence_ids", []) or []:
                if isinstance(fact_id, str) and fact_id.startswith("F_"):
                    fact_ids.add(fact_id)

    if fact_ids:
        return fact_ids
    return _contract_fact_ids(contract)


def _contract_coverage_atoms(contract: Dict, facts_by_id: Dict[str, Dict], target_function: str) -> Set[tuple]:
    atoms: Set[tuple] = set()
    derivation = contract.get("derivation", "")
    if derivation in {"source_fact_graph:condition_case", "source_fact_graph:loop_boundary"}:
        atoms.add(("scenario", contract.get("scenario_id", "")))

    for fact_id in _contract_selection_fact_ids(contract):
        fact = facts_by_id.get(fact_id)
        if not fact:
            continue
        if target_function and fact.get("function") != target_function:
            continue
        line = fact.get("start_line")
        if not line:
            continue
        if fact.get("kind") == "BRANCH_EDGE":
            edge = (fact.get("metadata", {}) or {}).get("edge", "")
            atoms.add(("branch", line, edge))
        else:
            atoms.add(("line", line))

    if atoms:
        return atoms

    for check in contract.get("scenario_checks", []) or []:
        if not isinstance(check, dict):
            continue
        atoms.add(
            (
                "check",
                check.get("kind", ""),
                check.get("target", ""),
                check.get("expected_relation", ""),
            )
        )
    return atoms


def _contract_first_line(contract: Dict, facts_by_id: Dict[str, Dict], target_function: str) -> int:
    lines = []
    for fact_id in _contract_selection_fact_ids(contract):
        fact = facts_by_id.get(fact_id)
        if not fact:
            continue
        if target_function and fact.get("function") != target_function:
            continue
        line = fact.get("start_line")
        if isinstance(line, int) and line > 0:
            lines.append(line)
    return min(lines) if lines else 10**9


def select_initial_active_scenario_ids(registry: Dict) -> List[str]:
    """Pick a source-coverage frontier without discarding the full registry.

    The full scenario registry is still saved.  Initial generation activates
    contracts that add new target-function source-fact coverage.  Hardware
    feasibility is not used as a hard filter here; later verification decides
    whether a generated test actually reaches and controls the boundary.
    """
    contracts = registry.get("scenario_contracts", []) or []
    facts_by_id = {
        item.get("fact_id", ""): item
        for item in registry.get("source_facts", []) or []
        if isinstance(item, dict) and item.get("fact_id")
    }
    target_function = registry.get("target_function", "")

    candidates: List[Dict] = []

    for contract in contracts:
        scenario_id = contract.get("scenario_id", "")
        if not scenario_id:
            continue
        atoms = _contract_coverage_atoms(contract, facts_by_id, target_function)
        if not atoms:
            continue
        candidates.append(
            {
                "scenario_id": scenario_id,
                "atoms": atoms,
                "first_line": _contract_first_line(contract, facts_by_id, target_function),
            }
        )

    selected: List[str] = []
    covered: Set[tuple] = set()
    remaining = list(candidates)
    while remaining:
        ranked = sorted(
            remaining,
            key=lambda item: (-len(item["atoms"] - covered), item["first_line"], item["scenario_id"]),
        )
        best = ranked[0]
        new_atoms = best["atoms"] - covered
        if not new_atoms:
            break
        selected.append(best["scenario_id"])
        covered.update(best["atoms"])
        remaining = [item for item in remaining if item["scenario_id"] != best["scenario_id"]]

    return selected


MAX_TEXT_FIELD = 160


def _short_text(value, limit: int = MAX_TEXT_FIELD):
    if not isinstance(value, str):
        return value
    value = value.strip()
    if len(value) <= limit:
        return value
    return value[:limit] + "...<truncated>"


def _compact_list(value: Iterable, limit: int = 8) -> List:
    if not isinstance(value, list):
        return []
    compacted = []
    for item in value[:limit]:
        if isinstance(item, str):
            compacted.append(_short_text(item))
        elif isinstance(item, list):
            compacted.append(item[:limit])
        elif isinstance(item, dict):
            compacted.append(_compact_small_dict(item))
        else:
            compacted.append(item)
    return compacted


def _compact_small_dict(value: Dict) -> Dict:
    if not isinstance(value, dict):
        return {}
    compacted = {}
    for key, item in value.items():
        if isinstance(item, str):
            compacted[key] = _short_text(item)
        elif isinstance(item, list):
            compacted[key] = _compact_list(item)
        elif isinstance(item, dict):
            compacted[key] = _compact_small_dict(item)
        else:
            compacted[key] = item
    return compacted


def _compact_metadata(metadata: Dict) -> Dict:
    if not isinstance(metadata, dict):
        return {}
    keep = {}
    for key in (
        "callee_name",
        "callee_expression",
        "callee_path",
        "arguments",
        "argument_paths",
        "argument_type_map",
        "output_arguments",
        "result_assignee",
        "condition",
        "return_expression",
        "first_return_expression",
        "first_return_line",
        "left",
        "right",
        "left_path",
        "field_path",
        "name",
        "edge",
        "requirements",
        "argument_aliases",
        "initializer",
        "update",
    ):
        if key not in metadata:
            continue
        value = metadata[key]
        if isinstance(value, str):
            value = _short_text(value)
        elif isinstance(value, list):
            value = _compact_list(value)
        elif isinstance(value, dict):
            value = _compact_small_dict(value)
        keep[key] = value
    return keep


def _compact_fact(fact: Dict) -> Dict:
    compact = {
        "fact_id": fact.get("fact_id", ""),
        "kind": fact.get("kind", ""),
        "function": fact.get("function", ""),
        "line": fact.get("start_line", 0),
        "code": _short_text(fact.get("code", "")),
    }
    metadata = _compact_metadata(fact.get("metadata", {}))
    if metadata:
        compact["detail"] = metadata
    return compact


def _compact_boundary(boundary: Dict) -> Dict:
    return {
        "boundary_id": boundary.get("candidate_id", ""),
        "expression": _short_text(boundary.get("expression", "")),
        "fact_ids": boundary.get("fact_ids", []),
        "location": {
            "function": boundary.get("source_function", ""),
            "line": boundary.get("source_line", 0),
            "fact_kind": boundary.get("source_fact_kind", ""),
        },
        "access_path": boundary.get("access_path", []),
        "arguments": boundary.get("arguments", []),
        "argument_paths": boundary.get("argument_paths", []),
        "argument_type_map": boundary.get("argument_type_map", []),
        "output_arguments": boundary.get("output_arguments", []),
        "result_assignee": boundary.get("result_assignee", ""),
        "semantic_role": boundary.get("semantic_role", ""),
        "environment_notes": boundary.get("environment_notes", []),
    }


def _boundary_controls_from_context(scenario_context: Dict) -> List[Dict]:
    target = (scenario_context or {}).get("target", {}) or {}
    controls = []
    for item in target.get("export_interface_details", []) or []:
        if not isinstance(item, dict):
            continue
        if item.get("source_kind") != "boundary_hook":
            continue
        controls.append(
            {
                "boundary_id": item.get("boundary_id", ""),
                "boundary_expression": item.get("boundary_expression", ""),
                "role": item.get("boundary_control_role", ""),
                "prototype": item.get("prototype", ""),
                "description": item.get("description", ""),
            }
        )
    return controls


def _ids_referenced_by_contracts(registry: Dict, contracts: Optional[List[Dict]] = None) -> tuple[set, set]:
    fact_ids = set()
    boundary_ids = set()

    def maybe_boundary(value: str) -> None:
        if isinstance(value, str) and value.startswith("B_"):
            boundary_ids.add(value)

    def visit(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key == "relation_edges":
                    continue
                if key in {"fact_id", "source_fact_id", "target_fact_id", "guard_fact_id"} and isinstance(item, str):
                    fact_ids.add(item)
                    maybe_boundary(item)
                elif key in {"boundary_id", "target"} and isinstance(item, str):
                    maybe_boundary(item)
                elif key in {"fact_ids", "source_fact_ids", "evidence_ids", "source_anchors", "classification_evidence_ids"}:
                    if isinstance(item, list):
                        for entry in item:
                            if isinstance(entry, str):
                                fact_ids.add(entry)
                                maybe_boundary(entry)
                elif key == "dependent_boundaries" and isinstance(item, list):
                    for entry in item:
                        maybe_boundary(entry)
                else:
                    visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    visit(_active_contracts(registry) if contracts is None else contracts)
    return fact_ids, boundary_ids


def _active_contracts(registry: Dict) -> List[Dict]:
    contracts = registry.get("scenario_contracts", []) or []
    if "active_scenario_ids" not in registry:
        return contracts
    active_ids = registry.get("active_scenario_ids", []) or []
    active = {item for item in active_ids if isinstance(item, str)}
    return [contract for contract in contracts if contract.get("scenario_id") in active]


def _fact_ids_from_boundaries(boundaries: Iterable[Dict]) -> Set[str]:
    ids: Set[str] = set()
    for boundary in boundaries or []:
        for fact_id in boundary.get("fact_ids", []) or []:
            if isinstance(fact_id, str):
                ids.add(fact_id)
    return ids


def _fact_lookup(facts: Iterable[Dict]) -> Dict[str, Dict]:
    return {
        fact.get("fact_id", ""): fact
        for fact in facts or []
        if isinstance(fact, dict) and fact.get("fact_id")
    }


def _compact_preconditions(preconditions: Iterable[Dict]) -> List[Dict]:
    compacted = []
    for item in preconditions or []:
        compacted.append(
            {
                "precondition_id": item.get("precondition_id", ""),
                "description": _short_text(item.get("description", "")),
                "evidence": item.get("evidence_ids", []),
                "object_path": item.get("object_path", []),
                "callsite_object_path": item.get("callsite_object_path", []),
            }
        )
    return compacted


def _compact_effects(effects: Iterable[Dict]) -> List[Dict]:
    compacted = []
    for item in effects or []:
        compacted.append(
            {
                "effect_id": item.get("effect_id", ""),
                "description": _short_text(item.get("description", "")),
                "relation": _short_text(item.get("relation", "")),
                "evidence": item.get("evidence_ids", []),
            }
        )
    return compacted


def _compact_witnesses(witnesses: Iterable[Dict]) -> List[Dict]:
    compacted = []
    for item in witnesses or []:
        compacted.append(
            {
                "witness_id": item.get("witness_id", ""),
                "kind": item.get("kind", ""),
                "target": item.get("target", ""),
                "relation": _short_text(item.get("relation", "")),
                "evidence": item.get("evidence_ids", []),
            }
        )
    return compacted


def _compact_checks(checks: Iterable[Dict]) -> List[Dict]:
    compacted = []
    for item in checks or []:
        compacted.append(
            {
                "check_id": item.get("check_id", ""),
                "kind": item.get("kind", ""),
                "target": item.get("target", ""),
                "expected": _short_text(item.get("expected_relation", "")),
                "evidence": item.get("evidence_ids", []),
            }
        )
    return compacted


def _compact_observations(observations: Iterable[Dict]) -> List[Dict]:
    compacted = []
    for item in observations or []:
        if not isinstance(item, dict):
            continue
        compacted.append(
            {
                "observation_id": item.get("observation_id", ""),
                "kind": item.get("kind", ""),
                "target": _short_text(item.get("target", "")),
                "evidence": item.get("evidence_ids", []),
            }
        )
    return compacted


def _semantic_observation_guidance(contract: Dict) -> List[Dict]:
    """Summarize what a test should assert, without inventing expected values."""
    guidance: List[Dict] = []

    for item in contract.get("observations", []) or []:
        if not isinstance(item, dict):
            continue
        kind = item.get("kind", "")
        target = item.get("target", "")
        if kind == "return_value":
            observation = "wrapper return value"
            assertion_hint = "prefer an exact KUnit assertion on the wrapper return value when the value can be derived from source evidence, scenario input, or mock configuration"
        elif kind == "driver_state":
            observation = f"driver state `{target}`"
            assertion_hint = "assert the concrete state change or state relation caused by this scenario"
        elif kind in {"branch_edge", "condition_case", "loop_execution"}:
            observation = f"{kind} `{target}`"
            assertion_hint = "assert the return value, output argument, state change, or boundary argument that distinguishes this path from sibling paths"
        else:
            observation = f"{kind} `{target}`" if target else kind
            assertion_hint = "assert a concrete target-visible behavior for this observation"
        guidance.append(
            {
                "observation": _short_text(observation),
                "assertion_hint": _short_text(assertion_hint),
                "evidence": item.get("evidence_ids", []),
            }
        )

    for check in contract.get("scenario_checks", []) or []:
        if not isinstance(check, dict):
            continue
        kind = str(check.get("kind", "") or "")
        expected = str(check.get("expected_relation", "") or "")
        target = str(check.get("target", "") or "")
        if kind.startswith("Return"):
            guidance.append(
                {
                    "observation": "wrapper return value",
                    "assertion_hint": _short_text(f"assert return value relation: {expected}"),
                    "evidence": check.get("evidence_ids", []),
                }
            )
        elif kind.startswith("State"):
            guidance.append(
                {
                    "observation": _short_text(f"driver state `{target}`"),
                    "assertion_hint": _short_text(f"assert state relation: {expected}"),
                    "evidence": check.get("evidence_ids", []),
                }
            )
        elif kind.startswith("Path"):
            guidance.append(
                {
                    "observation": _short_text(f"path outcome `{target}`"),
                    "assertion_hint": "assert a target-visible behavior that differs from sibling paths; do not use witness markers alone",
                    "evidence": check.get("evidence_ids", []),
                }
            )

    for constraint in contract.get("hardware_environment_constraints", []) or []:
        if not isinstance(constraint, dict):
            continue
        boundary = constraint.get("boundary_expression", "") or constraint.get("boundary_id", "")
        for effect in constraint.get("required_effects", []) or []:
            if not isinstance(effect, dict):
                continue
            guidance.append(
                {
                    "observation": _short_text(f"semantic effect of boundary `{boundary}`"),
                    "assertion_hint": _short_text(
                        "assert how the configured boundary effect is visible through the target wrapper "
                        f"or driver state: {effect.get('relation', '') or effect.get('description', '')}"
                    ),
                    "evidence": effect.get("evidence_ids", []) or constraint.get("source_fact_ids", []),
                }
            )

    seen = set()
    unique: List[Dict] = []
    for item in guidance:
        key = (item.get("observation", ""), item.get("assertion_hint", ""))
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique[:8]


def _compact_coverage_targets(targets: Iterable[Dict]) -> List[Dict]:
    compacted = []
    for item in targets or []:
        if not isinstance(item, dict):
            continue
        compacted.append(
            {
                "target_id": item.get("target_id", ""),
                "scenario_id": item.get("scenario_id", ""),
                "variant_id": item.get("variant_id", ""),
                "action": item.get("action", ""),
                "relation": item.get("relation", ""),
                "coverage_kind": item.get("coverage_kind", ""),
                "line": item.get("line", 0),
                "branch_index": item.get("branch_index"),
                "source_fact_id": item.get("source_fact_id", ""),
                "source_fact_kind": item.get("source_fact_kind", ""),
                "source_fact_code": _short_text(item.get("source_fact_code", "")),
                "scenario_derivation": item.get("scenario_derivation", ""),
                "instruction": _short_text(item.get("instruction", ""), 240),
            }
        )
    return compacted


def _compact_contract(contract: Dict) -> Dict:
    compact = {
        "scenario_id": contract.get("scenario_id", ""),
        "source_anchors": contract.get("source_anchors", []),
        "trigger_conditions": [
            {
                "condition_id": item.get("condition_id", ""),
                "expression": _short_text(item.get("expression", "")),
                "evidence": item.get("evidence_ids", []),
            }
            for item in contract.get("trigger_conditions", []) or []
        ],
        "environment_constraints": [],
        "observations": _compact_observations(contract.get("observations", []) or []),
        "semantic_observations": _semantic_observation_guidance(contract),
        "checks": _compact_checks(contract.get("scenario_checks", []) or []),
        "runtime_witnesses": _compact_witnesses(contract.get("runtime_witnesses", []) or []),
        "dependent_boundaries": contract.get("dependent_boundaries", []),
        "derivation": contract.get("derivation", ""),
    }
    for constraint in contract.get("hardware_environment_constraints", []) or []:
        compact["environment_constraints"].append(
            {
                "constraint_id": constraint.get("constraint_id", ""),
                "boundary_id": constraint.get("boundary_id", ""),
                "boundary_expression": _short_text(constraint.get("boundary_expression", "")),
                "location": {
                    "function": constraint.get("source_function", ""),
                    "line": constraint.get("source_line", 0),
                },
                "evidence": constraint.get("source_fact_ids", []),
                "preconditions": _compact_preconditions(constraint.get("preconditions", []) or []),
                "required_effects": _compact_effects(constraint.get("required_effects", []) or []),
                "runtime_witnesses": _compact_witnesses(constraint.get("runtime_witnesses", []) or []),
            }
        )
    return compact


def _scenario_evidence_ids(contract: Dict) -> Set[str]:
    fact_ids: Set[str] = set()

    def visit(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key == "relation_edges":
                    continue
                if key in {"fact_id", "source_fact_id", "target_fact_id", "guard_fact_id"} and isinstance(item, str):
                    fact_ids.add(item)
                elif key in {"fact_ids", "source_fact_ids", "evidence_ids", "source_anchors", "classification_evidence_ids"}:
                    if isinstance(item, list):
                        fact_ids.update(entry for entry in item if isinstance(entry, str) and entry.startswith("F_"))
                else:
                    visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    visit(contract)
    return fact_ids


def _attach_contract_evidence(contract: Dict, facts_by_id: Dict[str, Dict]) -> Dict:
    compact = _compact_contract(contract)
    evidence = []
    for fact_id in sorted(_scenario_evidence_ids(contract)):
        fact = facts_by_id.get(fact_id)
        if fact:
            evidence.append(_compact_fact(fact))
    if evidence:
        compact["evidence"] = evidence
    return compact


def compact_scenario_context_for_prompt(scenario_context: Dict) -> Dict:
    registry = (scenario_context or {}).get("scenario_registry") or {}
    allowed_ids = allowed_generated_scenario_ids(registry)
    direct_external_ids = sorted(direct_uncontrollable_scenario_ids(registry))
    prompt_contracts = [
        contract
        for contract in _active_contracts(registry)
        if contract.get("scenario_id") in allowed_ids
    ]
    registry_source_facts = registry.get("source_facts", []) or []
    prompt_source_facts = scenario_context.get("source_facts") or []
    source_facts = prompt_source_facts if scenario_context.get("analysis_scope") else registry_source_facts
    referenced_fact_ids, referenced_boundary_ids = _ids_referenced_by_contracts(registry, prompt_contracts)
    selected_facts = [
        fact for fact in source_facts if fact.get("fact_id") in referenced_fact_ids
    ]
    selected_boundaries = [
        boundary
        for boundary in registry.get("boundary_candidates", []) or []
        if boundary.get("candidate_id") in referenced_boundary_ids
        or any(fact_id in referenced_fact_ids for fact_id in boundary.get("fact_ids", []) or [])
    ]
    selected_fact_ids = {fact.get("fact_id") for fact in selected_facts}
    missing_boundary_fact_ids = _fact_ids_from_boundaries(selected_boundaries) - selected_fact_ids
    if missing_boundary_fact_ids:
        for fact in source_facts:
            if fact.get("fact_id") in missing_boundary_fact_ids:
                selected_facts.append(fact)
    facts_by_id = _fact_lookup(selected_facts)

    return {
        "format": "raca_scenario_generation_view_v1",
        "target": {
            "function": registry.get("target_function") or (scenario_context.get("target", {}) or {}).get("function", ""),
            "wrapper": registry.get("export_function") or (scenario_context.get("target", {}) or {}).get("wrapper", ""),
        },
        "internal_call_closure": registry.get("internal_call_closure", []),
        "boundaries": [_compact_boundary(boundary) for boundary in selected_boundaries],
        "boundary_controls": _boundary_controls_from_context(scenario_context),
        "scenarios": [
            _attach_contract_evidence(contract, facts_by_id)
            for contract in prompt_contracts
        ],
        "coverage_targets": _compact_coverage_targets(scenario_context.get("coverage_targets", [])),
        "active_scenario_ids": [scenario_id for scenario_id in registry.get("active_scenario_ids", []) if scenario_id in allowed_ids],
        "all_active_scenario_ids": registry.get("active_scenario_ids", []),
        "direct_external_boundary_scenario_ids": direct_external_ids,
        "oracle_policy": {
            "semantic_assertion_required": "Each generated test must assert at least one target-visible semantic observation for its scenario.",
            "witness_only_is_insufficient": "Runtime witnesses, effect markers, boundary call counts, allocation checks, and non-NULL checks prove reachability/setup only; they cannot be the sole assertion.",
            "llm_expected_value_rule": "The generator may derive exact expected values from scenario input, mock configuration, source evidence, constants/macros, output arguments, or visible state updates. If no exact value is grounded, use a narrower relation over a semantic observation rather than a broad EXPECT_NE/range check.",
        },
        "note": "Full scenario_context.json keeps the complete SourceFact graph and inactive scenario contracts; this prompt view keeps active scenario evidence, including scenarios that require hardware-environment modeling.",
    }


def format_scenario_context_for_prompt(scenario_context: Optional[Dict]) -> str:
    if not scenario_context:
        return "No scenario-contract context is available."
    compact = compact_scenario_context_for_prompt(scenario_context)
    return json.dumps(compact, separators=(",", ":"), sort_keys=True)


def save_scenario_context(scenario_context: Dict, out_path: str) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(scenario_context, f, indent=2, sort_keys=True)
