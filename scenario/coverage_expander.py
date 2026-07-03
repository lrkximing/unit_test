import hashlib
from typing import Dict, Iterable, List, Set, Tuple

from scenario.contract_revision import append_revision


def _stable_id(prefix: str, *parts: object) -> str:
    text = ":".join(str(part) for part in parts)
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]
    return f"{prefix}_{digest}"


def _registry(context: Dict) -> Dict:
    return context.get("scenario_registry") or context


def _fact_for_line(facts: List[Dict], line: int):
    for fact in facts:
        start = int(fact.get("start_line") or 0)
        end = int(fact.get("end_line") or start)
        if start <= line <= end:
            return fact
    return None


def _fact_for_branch(facts: List[Dict], branch: Dict):
    if not isinstance(branch, dict) or not branch.get("line"):
        return None
    line = int(branch.get("line") or 0)
    branch_index = branch.get("branch_index")
    expected_edge = None
    if branch_index == 0:
        expected_edge = "fallthrough"
    elif branch_index == 1:
        expected_edge = "branch"
    candidates = []
    for fact in facts:
        if fact.get("kind") != "BRANCH_EDGE":
            continue
        start = int(fact.get("start_line") or 0)
        end = int(fact.get("end_line") or start)
        if not (start <= line <= end):
            continue
        if expected_edge and (fact.get("metadata", {}) or {}).get("edge") != expected_edge:
            continue
        candidates.append(fact)
    if candidates:
        return candidates[0]
    return _fact_for_line(facts, line)


def _anchored_fact_ids(registry: Dict) -> Set[str]:
    ids: Set[str] = set()
    for contract in registry.get("scenario_contracts", []) or []:
        ids.update(contract.get("source_anchors", []) or [])
    return ids


def _activate_contract(registry: Dict, contract: Dict) -> bool:
    scenario_id = contract.get("scenario_id", "")
    if not scenario_id:
        return False
    active = registry.setdefault("active_scenario_ids", [])
    if scenario_id in active:
        return False
    active.append(scenario_id)
    return True


def _existing_contract_for_fact(registry: Dict, fact_id: str):
    for contract in registry.get("scenario_contracts", []) or []:
        if fact_id in (contract.get("source_anchors", []) or []):
            return contract
    return None


def _active_scenario_set(registry: Dict) -> Set[str]:
    return {
        item
        for item in registry.get("active_scenario_ids", []) or []
        if isinstance(item, str)
    }


def _contract_for_existing_fact(registry: Dict, fact: Dict):
    fact_id = fact.get("fact_id", "")
    direct = _existing_contract_for_fact(registry, fact_id)
    if direct is not None:
        return direct

    active = _active_scenario_set(registry)
    candidates = []
    for contract in registry.get("scenario_contracts", []) or []:
        anchors = set(contract.get("source_anchors", []) or [])
        if fact_id not in anchors:
            continue
        candidates.append(contract)
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item.get("scenario_id", "") not in active, item.get("scenario_id", "")))
    return candidates[0]


def _coverage_target(
    *,
    registry: Dict,
    fact: Dict,
    contract: Dict,
    coverage_item: Dict,
    relation: str,
) -> Dict:
    scenario_id = contract.get("scenario_id", "")
    fact_id = fact.get("fact_id", "")
    kind = coverage_item.get("kind", "line")
    line = coverage_item.get("line", fact.get("start_line", 0))
    branch_index = coverage_item.get("branch_index")
    variant_parts = [kind, line]
    if branch_index is not None:
        variant_parts.append(branch_index)
    variant_id = f"cov_{_stable_id('V', scenario_id, fact_id, *variant_parts)}"
    return {
        "target_id": _stable_id("COVT", scenario_id, fact_id, *variant_parts),
        "relation": relation,
        "scenario_id": scenario_id,
        "variant_id": variant_id,
        "action": "add_test_variant",
        "coverage_kind": kind,
        "line": line,
        "branch_index": branch_index,
        "source_fact_id": fact_id,
        "source_fact_kind": fact.get("kind", ""),
        "source_fact_code": fact.get("code", ""),
        "scenario_derivation": contract.get("derivation", ""),
        "instruction": (
            f"Add a new KUnit test variant for scenario {scenario_id}; bind it with "
            f"RACA_SCENARIO and RACA_VARIANT {variant_id}; keep existing tests unchanged."
        ),
        "target_function": registry.get("target_function", ""),
    }


def _boundary_for_fact(registry: Dict, fact_id: str):
    for candidate in registry.get("boundary_candidates", []) or []:
        if fact_id in (candidate.get("fact_ids", []) or []):
            return candidate
    return None


def _relation_edges_for_fact(fact: Dict) -> List[Dict]:
    edges = []
    for edge in (fact.get("metadata", {}) or {}).get("relation_edges", []) or []:
        if not isinstance(edge, dict):
            continue
        copied = dict(edge)
        copied["source_fact_id"] = fact.get("fact_id")
        edges.append(copied)
    return edges


def _fact_by_id(registry: Dict) -> Dict[str, Dict]:
    return {
        fact.get("fact_id", ""): fact
        for fact in registry.get("source_facts", []) or []
        if isinstance(fact, dict) and fact.get("fact_id")
    }


def _relation_adjacency(facts: Iterable[Dict]) -> Dict[str, Set[str]]:
    adjacency: Dict[str, Set[str]] = {}
    for fact in facts or []:
        source_id = fact.get("fact_id")
        if not source_id:
            continue
        adjacency.setdefault(source_id, set())
        for edge in _relation_edges_for_fact(fact):
            target_id = edge.get("target_fact_id")
            if not target_id:
                continue
            adjacency.setdefault(source_id, set()).add(target_id)
            adjacency.setdefault(target_id, set()).add(source_id)
    return adjacency


def _reachable_fact_ids(registry: Dict, fact_id: str) -> Set[str]:
    if not fact_id:
        return set()
    adjacency = _relation_adjacency(registry.get("source_facts", []) or [])
    reached: Set[str] = set()
    stack = [fact_id]
    while stack:
        current = stack.pop()
        if current in reached:
            continue
        reached.add(current)
        for neighbor in adjacency.get(current, set()):
            if neighbor not in reached:
                stack.append(neighbor)
    return reached


def _boundary_internal_call_fact_ids(boundary: Dict, facts_by_id: Dict[str, Dict]) -> Set[str]:
    call_fact_ids: Set[str] = set()
    for fact_id in boundary.get("fact_ids", []) or []:
        fact = facts_by_id.get(fact_id)
        if not fact:
            continue
        for edge in _relation_edges_for_fact(fact):
            if edge.get("relation") not in {"entered_from_internal_call", "reached_from_internal_call"}:
                continue
            target_id = edge.get("target_fact_id")
            if target_id:
                call_fact_ids.add(target_id)
    return call_fact_ids


def _fact_may_continue_to_later_code(fact: Dict) -> bool:
    if fact.get("kind") == "RETURN":
        return False
    metadata = fact.get("metadata", {}) or {}
    if (
        fact.get("kind") == "BRANCH_EDGE"
        and metadata.get("edge") == "branch"
        and metadata.get("first_return_expression")
    ):
        return False
    return True


def _hardware_boundaries_related_to_fact(registry: Dict, fact: Dict) -> List[Dict]:
    fact_id = fact.get("fact_id", "")
    facts_by_id = _fact_by_id(registry)
    reachable = _reachable_fact_ids(registry, fact_id)
    target_function = registry.get("target_function", "")
    fact_line = int(fact.get("start_line") or 0)
    may_continue = _fact_may_continue_to_later_code(fact)
    related: List[Dict] = []

    for boundary in registry.get("boundary_candidates", []) or []:
        if not _is_hardware_boundary(boundary):
            continue
        boundary_fact_ids = {
            item for item in (boundary.get("fact_ids", []) or []) if isinstance(item, str)
        }
        internal_call_fact_ids = _boundary_internal_call_fact_ids(boundary, facts_by_id)

        if reachable & (boundary_fact_ids | internal_call_fact_ids):
            related.append(boundary)
            continue

        if not may_continue:
            continue
        for call_fact_id in internal_call_fact_ids:
            call_fact = facts_by_id.get(call_fact_id)
            if not call_fact:
                continue
            if target_function and call_fact.get("function") != target_function:
                continue
            call_line = int(call_fact.get("start_line") or 0)
            if fact_line and call_line and call_line >= fact_line:
                related.append(boundary)
                break
    return related


def _candidate_for_contract(contract: Dict) -> Dict:
    return {
        "candidate_id": contract["source_candidate_id"],
        "target_function": contract.get("target_function", ""),
        "export_function": contract.get("export_function", ""),
        "source_anchors": contract.get("source_anchors", []),
        "trigger_conditions": contract.get("trigger_conditions", []),
        "hardware_environment_constraints": contract.get("hardware_environment_constraints", []),
        "observations": contract.get("observations", []),
        "scenario_checks": contract.get("scenario_checks", []),
        "runtime_witnesses": contract.get("runtime_witnesses", []),
        "dependent_boundaries": contract.get("dependent_boundaries", []),
        "derivation": contract.get("derivation", ""),
        "relation_edges": contract.get("relation_edges", []),
    }


def _is_hardware_boundary(boundary: Dict) -> bool:
    if not boundary:
        return False
    if boundary.get("source_fact_kind") not in {"CALL", "MEMBER_CALL"}:
        return False
    if boundary.get("semantic_role", "unknown") == "ordinary_helper":
        return False
    if boundary.get("semantic_role", "unknown") == "hardware_boundary":
        return True
    return boundary.get("source_fact_kind") == "CALL" and boundary.get("semantic_role", "unknown") == "unknown"


def _path_matches_boundary(requirement_path: List[str], boundary_paths: Iterable[List[str]]) -> bool:
    if not requirement_path:
        return False
    for boundary_path in boundary_paths:
        if not boundary_path:
            continue
        if boundary_path[: len(requirement_path)] == requirement_path:
            return True
        if requirement_path[: len(boundary_path)] == boundary_path:
            return True
    return False


def _guard_preconditions_for_boundary(registry: Dict, boundary: Dict) -> List[Dict]:
    boundary_paths = []
    if boundary.get("access_path"):
        boundary_paths.append(boundary.get("access_path"))
    boundary_paths.extend(boundary.get("argument_paths", []) or [])
    preconditions: List[Dict] = []
    for fact in registry.get("source_facts", []) or []:
        if fact.get("kind") != "GUARD":
            continue
        if fact.get("function") != boundary.get("source_function"):
            continue
        if int(fact.get("start_line") or 0) >= int(boundary.get("source_line") or 0):
            continue
        for requirement in (fact.get("metadata", {}) or {}).get("fallthrough_requirements", []) or []:
            path = requirement.get("object_path", []) or []
            if not _path_matches_boundary(path, boundary_paths):
                continue
            callsite_path: List[str] = []
            callsite_evidence_ids: List[str] = []
            for edge in (fact.get("metadata", {}) or {}).get("relation_edges", []) or []:
                if edge.get("relation") not in {"entered_from_internal_call", "reached_from_internal_call"}:
                    continue
                aliases = edge.get("argument_aliases", {}) or {}
                if path and path[0] in aliases:
                    alias = aliases[path[0]] or {}
                    alias_path = alias.get("path", []) or []
                    if alias_path:
                        callsite_path = list(alias_path) + path[1:]
                        target_fact_id = edge.get("target_fact_id")
                        if target_fact_id:
                            callsite_evidence_ids.append(target_fact_id)
                        break
            preconditions.append(
                {
                    "precondition_id": _stable_id(
                        "PRE",
                        boundary.get("candidate_id", ""),
                        fact.get("fact_id", ""),
                        ".".join(path),
                        requirement.get("relation", ""),
                    ),
                    "boundary_id": boundary.get("candidate_id", ""),
                    "description": (
                        f"guard `{(fact.get('metadata', {}) or {}).get('condition', '')}` must fall through: "
                        f"{'.'.join(callsite_path or path)} {requirement.get('relation', '')}"
                    ),
                    "evidence_ids": [fact.get("fact_id")],
                    "object_path": path,
                    "callsite_object_path": callsite_path,
                    "callsite_evidence_ids": callsite_evidence_ids,
                }
            )
    return preconditions


def _boundary_preconditions(registry: Dict, boundary: Dict) -> List[Dict]:
    path = boundary.get("access_path", []) or []
    boundary_id = boundary.get("candidate_id", "")
    evidence_ids = boundary.get("fact_ids", []) or []
    preconditions: List[Dict] = _guard_preconditions_for_boundary(registry, boundary)
    if boundary.get("source_fact_kind") == "MEMBER_CALL" and len(path) >= 2:
        for index in range(1, len(path)):
            object_path = path[:index]
            preconditions.append(
                {
                    "precondition_id": _stable_id("PRE", boundary_id, ".".join(object_path)),
                    "boundary_id": boundary_id,
                    "description": f"object path {'.'.join(object_path)} must be initialized before reaching {boundary.get('expression', boundary_id)}",
                    "evidence_ids": evidence_ids,
                    "object_path": object_path,
                }
            )
        preconditions.append(
            {
                "precondition_id": _stable_id("PRE", boundary_id, ".".join(path), "callable"),
                "boundary_id": boundary_id,
                "description": f"boundary expression {boundary.get('expression', boundary_id)} must be callable when the scenario reaches it",
                "evidence_ids": evidence_ids,
                "object_path": path,
            }
        )
    elif boundary.get("source_fact_kind") == "CALL":
        preconditions.append(
            {
                "precondition_id": _stable_id("PRE", boundary_id, "reachable_call"),
                "boundary_id": boundary_id,
                "description": f"test harness must make boundary call {boundary.get('expression', boundary_id)} executable for this scenario",
                "evidence_ids": evidence_ids,
                "object_path": [],
            }
        )
    return preconditions


def _boundary_reach_constraint(registry: Dict, scenario_id: str, boundary: Dict, fact: Dict) -> Tuple[Dict, Dict]:
    boundary_id = boundary.get("candidate_id", "")
    fact_id = fact.get("fact_id")
    witness = {
        "witness_id": _stable_id("WIT", scenario_id, boundary_id, "reached"),
        "kind": "BOUNDARY_INTERACTION",
        "target": boundary_id,
        "relation": "boundary call_count >= 1",
        "evidence_ids": boundary.get("fact_ids", []) or [fact_id],
    }
    constraint = {
        "constraint_id": _stable_id("HEC", scenario_id, boundary_id),
        "boundary_id": boundary_id,
        "boundary_expression": boundary.get("expression", ""),
        "source_function": boundary.get("source_function", ""),
        "source_line": boundary.get("source_line", 0),
        "source_fact_ids": boundary.get("fact_ids", []) or [fact_id],
        "preconditions": _boundary_preconditions(registry, boundary),
        "required_effects": [
            {
                "effect_id": _stable_id("EFF", scenario_id, boundary_id, "covered"),
                "boundary_id": boundary_id,
                "description": f"boundary {boundary.get('expression', boundary_id)} executes with an effect that allows the uncovered source fact to be reached",
                "evidence_ids": boundary.get("fact_ids", []) or [fact_id],
                "relation": "boundary effect enables source-fact coverage",
            }
        ],
        "runtime_witnesses": [witness],
        "relation_edges": _relation_edges_for_fact(fact),
    }
    return constraint, witness


def _boundary_completion_constraint(registry: Dict, scenario_id: str, boundary: Dict, fact: Dict) -> Tuple[Dict, Dict]:
    boundary_id = boundary.get("candidate_id", "")
    fact_id = fact.get("fact_id")
    witness = {
        "witness_id": _stable_id("WIT", scenario_id, boundary_id, "completion"),
        "kind": "BOUNDARY_INTERACTION",
        "target": boundary_id,
        "relation": "boundary call_count >= 1 when this coverage path continues past the anchored source fact",
        "evidence_ids": boundary.get("fact_ids", []) or [fact_id],
    }
    constraint = {
        "constraint_id": _stable_id("HEC", scenario_id, boundary_id, "completion"),
        "boundary_id": boundary_id,
        "boundary_expression": boundary.get("expression", ""),
        "source_function": boundary.get("source_function", ""),
        "source_line": boundary.get("source_line", 0),
        "source_fact_ids": boundary.get("fact_ids", []) or [fact_id],
        "preconditions": _boundary_preconditions(registry, boundary),
        "required_effects": [
            {
                "effect_id": _stable_id("EFF", scenario_id, boundary_id, "success_compatible"),
                "boundary_id": boundary_id,
                "description": (
                    f"boundary {boundary.get('expression', boundary_id)} must be executable with a "
                    "success-compatible effect so the coverage scenario can finish after reaching the anchored source fact"
                ),
                "evidence_ids": list((boundary.get("fact_ids", []) or [])) + [fact_id],
                "relation": "boundary effect permits the coverage path to complete without a harness fault",
            }
        ],
        "runtime_witnesses": [witness],
        "relation_edges": _relation_edges_for_fact(fact),
    }
    return constraint, witness


def _contract_for_fact(registry: Dict, fact: Dict) -> Dict:
    fact_id = fact["fact_id"]
    target = registry.get("target_function", "")
    wrapper = registry.get("export_function", f"{target}_test_export")
    scenario_id = f"{target}:coverage:{_stable_id('A', fact_id)}"
    kind = fact.get("kind", "")
    metadata = fact.get("metadata", {}) or {}
    checks: List[Dict] = []
    witnesses: List[Dict] = []
    observations: List[Dict] = []
    boundary = _boundary_for_fact(registry, fact_id)
    hardware_constraints: List[Dict] = []
    dependent_boundaries: List[str] = []

    if _is_hardware_boundary(boundary):
        constraint, witness = _boundary_reach_constraint(registry, scenario_id, boundary, fact)
        hardware_constraints.append(constraint)
        witnesses.append(witness)
        dependent_boundaries.append(boundary.get("candidate_id"))
        checks.append(
            {
                "check_id": _stable_id("CHK", scenario_id, boundary.get("candidate_id"), "reached"),
                "kind": "BoundaryReached",
                "target": boundary.get("candidate_id"),
                "expected_relation": "boundary call_count >= 1",
                "evidence_ids": boundary.get("fact_ids", []) or [fact_id],
            }
        )

    observations.append(
        {
            "observation_id": _stable_id("OBS", scenario_id, "source_fact"),
            "kind": "source_fact_reachability",
            "target": fact_id,
            "evidence_ids": [fact_id],
        }
    )
    existing_boundary_ids = set(dependent_boundaries)
    for downstream_boundary in _hardware_boundaries_related_to_fact(registry, fact):
        boundary_id = downstream_boundary.get("candidate_id")
        if not boundary_id or boundary_id in existing_boundary_ids:
            continue
        constraint, witness = _boundary_completion_constraint(registry, scenario_id, downstream_boundary, fact)
        hardware_constraints.append(constraint)
        witnesses.append(witness)
        dependent_boundaries.append(boundary_id)
        existing_boundary_ids.add(boundary_id)
        checks.append(
            {
                "check_id": _stable_id("CHK", scenario_id, boundary_id, "completion"),
                "kind": "BoundaryCalledForCoverageCompletion",
                "target": boundary_id,
                "expected_relation": "boundary call_count >= 1 if the coverage path reaches code after the anchored source fact",
                "evidence_ids": downstream_boundary.get("fact_ids", []) or [fact_id],
            }
        )

    return {
        "scenario_id": scenario_id,
        "target_function": target,
        "export_function": wrapper,
        "source_anchors": [fact_id],
        "trigger_conditions": [
            {
                "condition_id": _stable_id("COND", scenario_id, "coverage"),
                "expression": f"exercise uncovered {kind} fact at line {fact.get('start_line')}",
                "evidence_ids": [fact_id],
            }
        ],
        "hardware_environment_constraints": hardware_constraints,
        "observations": observations,
        "scenario_checks": checks,
        "runtime_witnesses": witnesses,
        "test_function": None,
        "dependent_boundaries": dependent_boundaries,
        "source_candidate_id": scenario_id,
        "derivation": "coverage_feedback:uncovered_source_fact",
        "relation_edges": _relation_edges_for_fact(fact),
        "status": "PLANNED",
        "version": 1,
    }


def expand_scenario_context_for_coverage(
    scenario_context: Dict,
    missed_lines: Iterable[int],
    missed_branches: Iterable[Dict],
) -> Tuple[Dict, List[Dict]]:
    registry = _registry(scenario_context)
    facts = registry.get("source_facts", []) or []
    anchored = _anchored_fact_ids(registry)
    added: List[Dict] = []
    candidate_facts: List[Dict] = []
    coverage_items_by_fact_id: Dict[str, List[Dict]] = {}
    seen_candidate_fact_ids: Set[str] = set()
    for line in sorted({int(line) for line in missed_lines or [] if line}):
        fact = _fact_for_line(facts, line)
        if fact:
            coverage_items_by_fact_id.setdefault(fact.get("fact_id"), []).append(
                {"kind": "line", "line": line}
            )
        if fact and fact.get("fact_id") not in seen_candidate_fact_ids:
            candidate_facts.append(fact)
            seen_candidate_fact_ids.add(fact.get("fact_id"))
    for branch in missed_branches or []:
        fact = _fact_for_branch(facts, branch)
        if fact:
            coverage_items_by_fact_id.setdefault(fact.get("fact_id"), []).append(
                {
                    "kind": "branch",
                    "line": branch.get("line"),
                    "branch_index": branch.get("branch_index"),
                }
            )
        if fact and fact.get("fact_id") not in seen_candidate_fact_ids:
            candidate_facts.append(fact)
            seen_candidate_fact_ids.add(fact.get("fact_id"))

    existing_ids = {contract.get("scenario_id") for contract in registry.get("scenario_contracts", []) or []}
    existing_candidate_ids = {
        candidate.get("candidate_id") for candidate in registry.get("scenario_candidates", []) or []
    }
    coverage_targets: List[Dict] = []
    for fact in candidate_facts:
        fact_id = fact.get("fact_id")
        existing_contract = _contract_for_existing_fact(registry, fact)
        if existing_contract is not None:
            if _activate_contract(registry, existing_contract):
                added.append(existing_contract)
            for coverage_item in coverage_items_by_fact_id.get(fact_id, []):
                coverage_targets.append(
                    _coverage_target(
                        registry=registry,
                        fact=fact,
                        contract=existing_contract,
                        coverage_item=coverage_item,
                        relation="existing_scenario_variant",
                    )
                )
            anchored.add(fact["fact_id"])
            continue
        if fact.get("fact_id") in anchored:
            continue
        contract = _contract_for_fact(registry, fact)
        if contract["scenario_id"] in existing_ids:
            continue
        registry.setdefault("scenario_contracts", []).append(contract)
        _activate_contract(registry, contract)
        if contract["source_candidate_id"] not in existing_candidate_ids:
            registry.setdefault("scenario_candidates", []).append(_candidate_for_contract(contract))
            existing_candidate_ids.add(contract["source_candidate_id"])
        existing_ids.add(contract["scenario_id"])
        anchored.add(fact["fact_id"])
        added.append(contract)
        for coverage_item in coverage_items_by_fact_id.get(fact_id, []):
            coverage_targets.append(
                _coverage_target(
                    registry=registry,
                    fact=fact,
                    contract=contract,
                    coverage_item=coverage_item,
                    relation="new_coverage_scenario_variant",
                )
            )

    scenario_context["coverage_targets"] = coverage_targets

    if added or coverage_targets:
        scenario_context["active_scenario_ids"] = list(registry.get("active_scenario_ids", []) or [])
        append_revision(
            scenario_context,
            reason="coverage expansion mapped missed coverage to scenario contracts and test variants",
            affected_scenarios=sorted({target["scenario_id"] for target in coverage_targets}),
            source_fact_ids=sorted({target["source_fact_id"] for target in coverage_targets}),
        )
    return scenario_context, added
