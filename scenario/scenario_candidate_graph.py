import hashlib
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from scenario.fact_model import (
    BoundaryCandidate,
    BoundaryPrecondition,
    BoundaryRequiredEffect,
    Condition,
    HardwareEnvironmentConstraint,
    Observation,
    ResultCheck,
    RuntimeWitness,
    ScenarioCandidate,
    ScenarioContract,
    SourceFact,
)


def _stable_id(prefix: str, *parts: object) -> str:
    text = ":".join(str(part) for part in parts)
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]
    return f"{prefix}_{digest}"


def _facts_by_kind(facts: Sequence[SourceFact], kind: str) -> List[SourceFact]:
    return [fact for fact in facts if fact.kind == kind]


def _fact_map(facts: Sequence[SourceFact]) -> Dict[str, SourceFact]:
    return {fact.fact_id: fact for fact in facts}


def _relation_edges(fact: SourceFact) -> List[Dict]:
    return [edge for edge in (fact.metadata or {}).get("relation_edges", []) or [] if isinstance(edge, dict)]


def _collect_relation_edges(facts: Sequence[SourceFact], fact_ids: Iterable[str]) -> List[Dict]:
    fact_by_id = _fact_map(facts)
    selected: List[Dict] = []
    seen: Set[Tuple[str, str, str, str]] = set()
    for fact_id in fact_ids:
        fact = fact_by_id.get(fact_id)
        if fact is None:
            continue
        for edge in _relation_edges(fact):
            key = (
                fact.fact_id,
                str(edge.get("relation", "")),
                str(edge.get("target_fact_id", "")),
                str(edge.get("via", "")),
            )
            if key in seen:
                continue
            seen.add(key)
            copied = dict(edge)
            copied["source_fact_id"] = fact.fact_id
            selected.append(copied)
    return selected


def _targets_for_relation(
    fact: SourceFact,
    fact_by_id: Dict[str, SourceFact],
    relation_prefix: str,
    target_kind: str = "",
) -> List[SourceFact]:
    targets: List[SourceFact] = []
    for edge in _relation_edges(fact):
        relation = str(edge.get("relation", ""))
        if not relation.startswith(relation_prefix):
            continue
        target = fact_by_id.get(str(edge.get("target_fact_id", "")))
        if target is None:
            continue
        if target_kind and target.kind != target_kind:
            continue
        targets.append(target)
    return targets


def _return_after(facts: Sequence[SourceFact], function: str, line: int, expression: str) -> Optional[SourceFact]:
    for fact in sorted(_facts_by_kind(facts, "RETURN"), key=lambda item: item.start_line):
        if fact.function != function or fact.start_line <= line:
            continue
        if (fact.metadata or {}).get("return_expression") == expression:
            return fact
    return None


def _return_at_line(facts: Sequence[SourceFact], line: int) -> Optional[SourceFact]:
    for fact in facts:
        if fact.kind == "RETURN" and fact.start_line == line:
            return fact
    return None


def _branch_edge_for_guard(
    facts: Sequence[SourceFact],
    guard: SourceFact,
    edge: str,
) -> Optional[SourceFact]:
    for fact in facts:
        if fact.kind != "BRANCH_EDGE":
            continue
        metadata = fact.metadata or {}
        if metadata.get("guard_fact_id") == guard.fact_id and metadata.get("edge") == edge:
            return fact
    return None


def _guard_for_branch_edge(
    facts: Sequence[SourceFact],
    edge_fact: SourceFact,
) -> Optional[SourceFact]:
    guard_id = (edge_fact.metadata or {}).get("guard_fact_id", "")
    for fact in facts:
        if fact.fact_id == guard_id and fact.kind == "GUARD":
            return fact
    return None


def _branch_edge_label(edge_fact: SourceFact) -> str:
    edge = (edge_fact.metadata or {}).get("edge", "")
    if edge == "branch":
        return "condition is true and the guarded branch is taken"
    if edge == "fallthrough":
        return "condition is false and execution continues past the guard"
    return "selected condition edge is reached"


def _condition_for_branch_edge(candidate_id: str, edge_fact: SourceFact) -> Condition:
    metadata = edge_fact.metadata or {}
    condition = metadata.get("condition", "") or edge_fact.code
    return Condition(
        condition_id=_stable_id("COND", candidate_id, edge_fact.fact_id, "edge"),
        expression=f"{condition}: {_branch_edge_label(edge_fact)}",
        evidence_ids=[edge_fact.fact_id],
    )


def _condition_case_text(case: Dict) -> str:
    condition = case.get("condition", "")
    outcome = case.get("outcome", "")
    outcome_text = "take guarded branch" if outcome == "branch" else "fall through"
    requirements = _case_requirements_text(case)
    prefix = f"condition outcome: {outcome_text}"
    if requirements:
        prefix += f"; subconditions: {requirements}"
    if case.get("short_circuit"):
        prefix += "; preserve C short-circuit"
    if condition:
        prefix += f"; condition: `{condition}`"
    return prefix


def _case_requirements_text(case: Dict) -> str:
    pieces = []
    for requirement in case.get("requirements", []) or []:
        expression = requirement.get("expression", "")
        relation = requirement.get("relation", "")
        if not expression or not relation:
            continue
        pieces.append(f"`{expression}` {relation.replace('_', ' ')}")
    return "; ".join(pieces)


def _is_grounded_exact_return_expression(expression: str) -> bool:
    expression = (expression or "").strip()
    if not expression:
        return False
    if expression in {"NULL", "true", "false"}:
        return True
    if expression.startswith("-"):
        stripped = expression[1:].strip()
        return bool(stripped) and (stripped.isdigit() or stripped.isupper())
    if expression.isdigit():
        return True
    return expression.isupper()


def _fallthrough_edges_before_success(
    facts: Sequence[SourceFact],
    target_function: str,
    success_line: int,
) -> List[SourceFact]:
    edges: List[SourceFact] = []
    for fact in sorted(_facts_by_kind(facts, "BRANCH_EDGE"), key=lambda item: item.start_line):
        if fact.function != target_function:
            continue
        metadata = fact.metadata or {}
        if metadata.get("edge") != "fallthrough":
            continue
        first_return_line = int(metadata.get("first_return_line") or 0)
        if first_return_line and first_return_line < success_line:
            edges.append(fact)
    return edges


def _first_success_return(facts: Sequence[SourceFact], target_function: str) -> Optional[SourceFact]:
    for fact in sorted(_facts_by_kind(facts, "RETURN"), key=lambda item: item.start_line):
        if fact.function == target_function and (fact.metadata or {}).get("return_expression") == "0":
            return fact
    return None


def _first_boundary_after(
    boundaries: Sequence[BoundaryCandidate],
    function: str,
    line: int,
) -> Optional[BoundaryCandidate]:
    for boundary in sorted(boundaries, key=lambda item: item.source_line):
        if _scenario_boundary(boundary) and boundary.source_function == function and boundary.source_line > line:
            return boundary
    return None


def _scenario_boundary(boundary: BoundaryCandidate) -> bool:
    if getattr(boundary, "source_fact_kind", "") not in {"CALL", "MEMBER_CALL"}:
        return False
    return getattr(boundary, "semantic_role", "unknown") == "hardware_boundary"


def _runtime_dependency_boundary(boundary: BoundaryCandidate) -> bool:
    if getattr(boundary, "semantic_role", "unknown") == "ordinary_helper":
        return False
    if _scenario_boundary(boundary):
        return True
    return (
        getattr(boundary, "source_fact_kind", "") == "CALL"
        and getattr(boundary, "semantic_role", "unknown") == "unknown"
    )


def _boundary_source_fact(boundary: BoundaryCandidate, fact_by_id: Dict[str, SourceFact]) -> Optional[SourceFact]:
    for fact_id in boundary.fact_ids:
        fact = fact_by_id.get(fact_id)
        if fact is not None:
            return fact
    return None


def _return_targets_from_fact(
    fact: SourceFact,
    fact_by_id: Dict[str, SourceFact],
    target_function: str,
    seen: Optional[Set[str]] = None,
) -> List[SourceFact]:
    if seen is None:
        seen = set()
    if fact.fact_id in seen:
        return []
    seen.add(fact.fact_id)

    results: List[SourceFact] = []
    for edge in _relation_edges(fact):
        relation = str(edge.get("relation", ""))
        target = fact_by_id.get(str(edge.get("target_fact_id", "")))
        if target is None:
            continue
        if relation.startswith("call_result_influences_"):
            if target.kind == "RETURN" and target.function == target_function:
                results.append(target)
            elif target.kind in {"CALL", "MEMBER_CALL", "RETURN"}:
                results.extend(_return_targets_from_fact(target, fact_by_id, target_function, seen))
        elif relation in {"entered_from_internal_call", "reached_from_internal_call"}:
            results.extend(_return_targets_from_fact(target, fact_by_id, target_function, seen))

    deduped: List[SourceFact] = []
    seen_returns: Set[str] = set()
    for item in results:
        if item.fact_id in seen_returns:
            continue
        seen_returns.add(item.fact_id)
        deduped.append(item)
    return deduped


def _boundary_return_targets(
    boundary: BoundaryCandidate,
    fact_by_id: Dict[str, SourceFact],
    target_function: str,
) -> List[SourceFact]:
    source_fact = _boundary_source_fact(boundary, fact_by_id)
    if source_fact is None:
        return []
    return _return_targets_from_fact(source_fact, fact_by_id, target_function)


def _boundaries_reaching_return(
    return_fact: SourceFact,
    boundaries: Sequence[BoundaryCandidate],
    fact_by_id: Dict[str, SourceFact],
    target_function: str,
) -> List[BoundaryCandidate]:
    matched: List[BoundaryCandidate] = []
    for boundary in boundaries:
        if not _runtime_dependency_boundary(boundary):
            continue
        if any(item.fact_id == return_fact.fact_id for item in _boundary_return_targets(boundary, fact_by_id, target_function)):
            matched.append(boundary)
    return matched


def _unique_constraints(constraints: Iterable[HardwareEnvironmentConstraint]) -> List[HardwareEnvironmentConstraint]:
    seen: Set[str] = set()
    unique: List[HardwareEnvironmentConstraint] = []
    for item in constraints:
        if item.constraint_id in seen:
            continue
        seen.add(item.constraint_id)
        unique.append(item)
    return unique


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


def _guard_preconditions_for_boundary(
    boundary: BoundaryCandidate,
    facts: Sequence[SourceFact],
) -> List[BoundaryPrecondition]:
    boundary_paths: List[List[str]] = []
    if boundary.access_path:
        boundary_paths.append(list(boundary.access_path))
    boundary_paths.extend([list(path) for path in getattr(boundary, "argument_paths", []) or [] if path])
    preconditions: List[BoundaryPrecondition] = []
    for fact in facts:
        if fact.kind != "GUARD" or fact.function != boundary.source_function or fact.start_line >= boundary.source_line:
            continue
        for requirement in (fact.metadata or {}).get("fallthrough_requirements", []) or []:
            path = requirement.get("object_path", []) or []
            if not _path_matches_boundary(path, boundary_paths):
                continue
            callsite_path: List[str] = []
            callsite_evidence_ids: List[str] = []
            for edge in _relation_edges(fact):
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
                BoundaryPrecondition(
                    precondition_id=_stable_id(
                        "PRE",
                        boundary.candidate_id,
                        fact.fact_id,
                        ".".join(path),
                        requirement.get("relation", ""),
                    ),
                    boundary_id=boundary.candidate_id,
                    description=(
                        f"guard `{(fact.metadata or {}).get('condition', '')}` must fall through: "
                        f"{'.'.join(callsite_path or path)} {requirement.get('relation', '')}"
                    ),
                    evidence_ids=[fact.fact_id],
                    object_path=path,
                    callsite_object_path=callsite_path,
                    callsite_evidence_ids=callsite_evidence_ids,
                )
            )
    return preconditions


def _boundary_preconditions(boundary: BoundaryCandidate, facts: Sequence[SourceFact]) -> List[BoundaryPrecondition]:
    path = list(boundary.access_path or [])
    preconditions: List[BoundaryPrecondition] = []
    preconditions.extend(_guard_preconditions_for_boundary(boundary, facts))
    if boundary.source_fact_kind == "MEMBER_CALL" and len(path) >= 2:
        for index in range(1, len(path)):
            object_path = path[:index]
            preconditions.append(
                BoundaryPrecondition(
                    precondition_id=_stable_id("PRE", boundary.candidate_id, ".".join(object_path)),
                    boundary_id=boundary.candidate_id,
                    description=f"object path {'.'.join(object_path)} must be initialized before reaching {boundary.expression}",
                    evidence_ids=list(boundary.fact_ids),
                    object_path=object_path,
                )
            )
        callable_path = path
        preconditions.append(
            BoundaryPrecondition(
                precondition_id=_stable_id("PRE", boundary.candidate_id, ".".join(callable_path), "callable"),
                boundary_id=boundary.candidate_id,
                description=f"boundary expression {boundary.expression} must be callable when the scenario reaches it",
                evidence_ids=list(boundary.fact_ids),
                object_path=callable_path,
            )
        )
    elif boundary.source_fact_kind == "CALL":
        preconditions.append(
            BoundaryPrecondition(
                precondition_id=_stable_id("PRE", boundary.candidate_id, "reachable_call"),
                boundary_id=boundary.candidate_id,
                description=f"test harness must make boundary call {boundary.expression} executable for this scenario",
                evidence_ids=list(boundary.fact_ids),
            )
        )
    return preconditions


def _boundary_witness(
    scenario_id: str,
    boundary: BoundaryCandidate,
    key: str,
    relation: str,
    kind: str = "BOUNDARY_INTERACTION",
) -> RuntimeWitness:
    return RuntimeWitness(
        witness_id=_stable_id("WIT", scenario_id, boundary.candidate_id, key),
        kind=kind,
        target=boundary.candidate_id,
        relation=relation,
        evidence_ids=list(boundary.fact_ids),
    )


def _required_effect(
    scenario_id: str,
    boundary: BoundaryCandidate,
    key: str,
    description: str,
    relation: str,
    evidence_ids: Optional[List[str]] = None,
) -> BoundaryRequiredEffect:
    return BoundaryRequiredEffect(
        effect_id=_stable_id("EFF", scenario_id, boundary.candidate_id, key),
        boundary_id=boundary.candidate_id,
        description=description,
        relation=relation,
        evidence_ids=evidence_ids or list(boundary.fact_ids),
    )


def _constraint_for_boundary(
    scenario_id: str,
    boundary: BoundaryCandidate,
    facts: Sequence[SourceFact],
    required_effects: Optional[List[BoundaryRequiredEffect]] = None,
    runtime_witnesses: Optional[List[RuntimeWitness]] = None,
) -> HardwareEnvironmentConstraint:
    return HardwareEnvironmentConstraint(
        constraint_id=_stable_id("HEC", scenario_id, boundary.candidate_id),
        boundary_id=boundary.candidate_id,
        boundary_expression=boundary.expression,
        source_function=boundary.source_function,
        source_line=boundary.source_line,
        source_fact_ids=list(boundary.fact_ids),
        preconditions=_boundary_preconditions(boundary, facts),
        required_effects=required_effects or [],
        runtime_witnesses=runtime_witnesses or [],
        relation_edges=_collect_relation_edges(facts, boundary.fact_ids),
    )


def _candidate(
    candidate_id: str,
    target_function: str,
    wrapper: str,
    source_anchors: List[str],
    facts: Sequence[SourceFact],
    derivation: str,
    trigger_conditions: Optional[List[Condition]] = None,
    observations: Optional[List[Observation]] = None,
    scenario_checks: Optional[List[ResultCheck]] = None,
    runtime_witnesses: Optional[List[RuntimeWitness]] = None,
    hardware_environment_constraints: Optional[List[HardwareEnvironmentConstraint]] = None,
    dependent_boundaries: Optional[List[str]] = None,
) -> ScenarioCandidate:
    observations = observations or []
    checks = scenario_checks or _checks_from_observations(candidate_id, observations, source_anchors)
    return ScenarioCandidate(
        candidate_id=candidate_id,
        target_function=target_function,
        export_function=wrapper,
        source_anchors=source_anchors,
        trigger_conditions=trigger_conditions or [],
        hardware_environment_constraints=_unique_constraints(hardware_environment_constraints or []),
        observations=observations,
        scenario_checks=checks,
        runtime_witnesses=runtime_witnesses or [],
        dependent_boundaries=dependent_boundaries or [],
        derivation=derivation,
        relation_edges=_collect_relation_edges(facts, source_anchors),
    )


def _checks_from_observations(
    candidate_id: str,
    observations: Sequence[Observation],
    source_anchors: Sequence[str],
) -> List[ResultCheck]:
    checks: List[ResultCheck] = []
    for observation in observations:
        evidence_ids = list(observation.evidence_ids or source_anchors)
        target = observation.target or observation.kind or "target_behavior"
        if observation.kind == "return_value":
            kind = "ReturnRelation"
            target = "return_value"
            relation = (
                "assert a non-vacuous relation on the target wrapper return value "
                "using scenario input, pre-state, mock configuration, source evidence, "
                "or sibling-input comparison"
            )
        elif observation.kind == "driver_state":
            kind = "StateRelation"
            relation = (
                f"assert a non-vacuous relation on driver state `{target}` using "
                "pre-state or scenario source evidence"
            )
        elif observation.kind in {"branch_edge", "condition_case", "loop_execution"}:
            kind = "PathOutcomeRelation"
            relation = (
                f"assert an observable result that distinguishes this {observation.kind} "
                "from sibling paths, such as return value, output parameter, driver state, "
                "or boundary interaction"
            )
        else:
            kind = "ObservationRelation"
            relation = f"assert a non-vacuous observable relation for `{observation.kind}`"
        checks.append(
            ResultCheck(
                check_id=_stable_id("CHK", candidate_id, observation.observation_id, "observation"),
                kind=kind,
                target=target,
                expected_relation=(
                    relation
                    + "; allocation-only, tautological, and wrapper-call-only assertions do not satisfy this check"
                ),
                evidence_ids=evidence_ids,
            )
        )
    return checks


def _candidate_from_guard(
    target_function: str,
    wrapper: str,
    guard: SourceFact,
    facts: Sequence[SourceFact],
    boundaries: Sequence[BoundaryCandidate],
    fact_by_id: Dict[str, SourceFact],
) -> Optional[ScenarioCandidate]:
    metadata = guard.metadata or {}
    return_expr = metadata.get("first_return_expression")
    if not return_expr:
        return None

    candidate_id = f"{target_function}:guard:{_stable_id('A', guard.fact_id)}"
    source_anchors = [guard.fact_id]
    branch_edge = _branch_edge_for_guard(facts, guard, "branch")
    if branch_edge is not None:
        source_anchors.append(branch_edge.fact_id)
    return_fact = _return_at_line(facts, int(metadata.get("first_return_line") or 0))
    if return_fact is not None:
        source_anchors.append(return_fact.fact_id)

    checks = [
        ResultCheck(
            check_id=_stable_id("CHK", candidate_id, "return"),
            kind="ReturnEquals",
            target="return_value",
            expected_relation=f"equals {return_expr}",
            evidence_ids=list(source_anchors),
        )
    ]
    witnesses: List[RuntimeWitness] = []
    constraints: List[HardwareEnvironmentConstraint] = []
    dependent_boundaries: List[str] = []
    if return_fact is not None:
        upstream_boundaries = _boundaries_reaching_return(return_fact, boundaries, fact_by_id, target_function)
        for boundary in upstream_boundaries:
            if boundary.candidate_id in dependent_boundaries:
                continue
            dependent_boundaries.append(boundary.candidate_id)
            witness = _boundary_witness(candidate_id, boundary, "called", "boundary call_count == 1")
            effect = _required_effect(
                candidate_id,
                boundary,
                "guard_return_compatible",
                (
                    f"boundary {boundary.expression} produces the value that satisfies guard "
                    f"`{metadata.get('condition', '')}` and reaches return `{return_expr}`"
                ),
                f"target return `{return_expr}` is determined by this boundary interaction",
                list(boundary.fact_ids) + source_anchors,
            )
            witnesses.append(witness)
            constraints.append(_constraint_for_boundary(candidate_id, boundary, facts, [effect], [witness]))
            checks.append(
                ResultCheck(
                    check_id=_stable_id("CHK", candidate_id, boundary.candidate_id, "boundary_called"),
                    kind="BoundaryCalled",
                    target=boundary.candidate_id,
                    expected_relation="boundary call_count == 1",
                    evidence_ids=list(boundary.fact_ids),
                )
            )
    later_boundary = _first_boundary_after(boundaries, guard.function, guard.start_line)
    if later_boundary is not None:
        witness = _boundary_witness(
            candidate_id,
            later_boundary,
            "not_called",
            "boundary call_count == 0 after guard exits",
            kind="BOUNDARY_NOT_REACHED",
        )
        witnesses.append(witness)
        checks.append(
            ResultCheck(
                check_id=_stable_id("CHK", candidate_id, later_boundary.candidate_id, "not_called"),
                kind="BoundaryNotCalled",
                target=later_boundary.candidate_id,
                expected_relation="boundary call_count == 0",
                evidence_ids=[guard.fact_id, later_boundary.candidate_id],
            )
        )

    return _candidate(
        candidate_id=candidate_id,
        target_function=target_function,
        wrapper=wrapper,
        source_anchors=source_anchors,
        facts=facts,
        derivation="source_fact_graph:guard_return",
        trigger_conditions=[
            Condition(
                condition_id=_stable_id("COND", candidate_id, "guard"),
                expression=metadata.get("condition", ""),
                evidence_ids=[guard.fact_id] + ([branch_edge.fact_id] if branch_edge is not None else []),
            )
        ]
        + ([_condition_for_branch_edge(candidate_id, branch_edge)] if branch_edge is not None else []),
        observations=[
            Observation(
                observation_id=_stable_id("OBS", candidate_id, "return"),
                kind="return_value",
                target="return_value",
                evidence_ids=source_anchors,
            )
        ],
        scenario_checks=checks,
        runtime_witnesses=witnesses,
        hardware_environment_constraints=constraints,
        dependent_boundaries=dependent_boundaries,
    )


def _candidate_from_boundary_result(
    target_function: str,
    wrapper: str,
    boundary: BoundaryCandidate,
    facts: Sequence[SourceFact],
    fact_by_id: Dict[str, SourceFact],
) -> Optional[ScenarioCandidate]:
    if not _runtime_dependency_boundary(boundary):
        return None
    source_fact = _boundary_source_fact(boundary, fact_by_id)
    if source_fact is None:
        return None
    related_returns = _boundary_return_targets(boundary, fact_by_id, target_function)
    return_fact = sorted(related_returns, key=lambda item: item.start_line)[0] if related_returns else None
    if return_fact is None and boundary.result_assignee:
        return_fact = _return_after(facts, boundary.source_function, boundary.source_line, boundary.result_assignee)
    if return_fact is None:
        return None

    candidate_id = f"{target_function}:boundary_error:{_stable_id('A', boundary.candidate_id)}"
    source_anchors = list(boundary.fact_ids) + [return_fact.fact_id]
    witness = _boundary_witness(candidate_id, boundary, "called", "boundary call_count == 1")
    flow_target = boundary.result_assignee or "the boundary return value"
    effect = _required_effect(
        candidate_id,
        boundary,
        "negative_errno",
        f"boundary {boundary.expression} produces a negative errno through {flow_target}",
        f"target return equals the boundary-produced errno through {flow_target}",
        source_anchors,
    )
    return _candidate(
        candidate_id=candidate_id,
        target_function=target_function,
        wrapper=wrapper,
        source_anchors=source_anchors,
        facts=facts,
        derivation="source_fact_graph:boundary_result_propagation",
        trigger_conditions=[
            Condition(
                condition_id=_stable_id("COND", candidate_id, "boundary_effect"),
                expression=f"boundary {boundary.candidate_id} produces a negative errno",
                evidence_ids=boundary.fact_ids,
            )
        ],
        hardware_environment_constraints=[
            _constraint_for_boundary(candidate_id, boundary, facts, [effect], [witness])
        ],
        observations=[
            Observation(
                observation_id=_stable_id("OBS", candidate_id, "return"),
                kind="return_value",
                target="return_value",
                evidence_ids=source_anchors,
            ),
            Observation(
                observation_id=_stable_id("OBS", candidate_id, "boundary_reached"),
                kind="boundary_interaction",
                target=boundary.candidate_id,
                evidence_ids=boundary.fact_ids,
            ),
        ],
        scenario_checks=[
            ResultCheck(
                check_id=_stable_id("CHK", candidate_id, "return"),
                kind="ReturnEqualsBoundaryEffect",
                target="return_value",
                expected_relation=f"equals negative errno produced by {boundary.candidate_id}",
                evidence_ids=source_anchors,
            ),
            ResultCheck(
                check_id=_stable_id("CHK", candidate_id, "boundary_called"),
                kind="BoundaryCalled",
                target=boundary.candidate_id,
                expected_relation="boundary call_count == 1",
                evidence_ids=boundary.fact_ids,
            ),
        ],
        runtime_witnesses=[witness],
        dependent_boundaries=[boundary.candidate_id],
    )


def _candidates_from_boundary_outputs(
    target_function: str,
    wrapper: str,
    boundary: BoundaryCandidate,
    facts: Sequence[SourceFact],
    fact_by_id: Dict[str, SourceFact],
) -> List[ScenarioCandidate]:
    if not _scenario_boundary(boundary) or not boundary.output_arguments:
        return []
    source_fact = _boundary_source_fact(boundary, fact_by_id)
    if source_fact is None:
        return []

    guards = _targets_for_relation(source_fact, fact_by_id, "call_output_influences_", "GUARD")
    if not guards:
        output_symbols = set(boundary.output_arguments)
        guards = [
            guard
            for guard in _facts_by_kind(facts, "GUARD")
            if guard.function == boundary.source_function
            and guard.start_line > boundary.source_line
            and output_symbols & set(guard.symbols)
        ]

    candidates: List[ScenarioCandidate] = []
    for guard in sorted(guards, key=lambda item: item.start_line):
        return_expr = (guard.metadata or {}).get("first_return_expression")
        candidate_id = f"{target_function}:boundary_output:{_stable_id('A', boundary.candidate_id, guard.fact_id)}"
        witness = _boundary_witness(candidate_id, boundary, "called", "boundary call_count == 1")
        effect = _required_effect(
            candidate_id,
            boundary,
            "output_controls_guard",
            f"boundary {boundary.expression} writes output argument(s) {boundary.output_arguments} so that guard `{(guard.metadata or {}).get('condition', '')}` is satisfied",
            "boundary output controls the subsequent guard branch",
            boundary.fact_ids + [guard.fact_id],
        )
        checks = [
            ResultCheck(
                check_id=_stable_id("CHK", candidate_id, "boundary_called"),
                kind="BoundaryCalled",
                target=boundary.candidate_id,
                expected_relation="boundary call_count == 1",
                evidence_ids=boundary.fact_ids,
            )
        ]
        if return_expr:
            checks.append(
                ResultCheck(
                    check_id=_stable_id("CHK", candidate_id, "return"),
                    kind="ReturnEqualsBoundaryOutput",
                    target="return_value",
                    expected_relation=f"equals {return_expr} when boundary output satisfies the guard",
                    evidence_ids=boundary.fact_ids + [guard.fact_id],
                )
            )
        candidates.append(
            _candidate(
                candidate_id=candidate_id,
                target_function=target_function,
                wrapper=wrapper,
                source_anchors=boundary.fact_ids + [guard.fact_id],
                facts=facts,
                derivation="source_fact_graph:boundary_output_controls_guard",
                trigger_conditions=[
                    Condition(
                        condition_id=_stable_id("COND", candidate_id, "boundary_output"),
                        expression="boundary output parameter satisfies the guarded branch condition",
                        evidence_ids=boundary.fact_ids + [guard.fact_id],
                    )
                ],
                hardware_environment_constraints=[
                    _constraint_for_boundary(candidate_id, boundary, facts, [effect], [witness])
                ],
                observations=[
                    Observation(
                        observation_id=_stable_id("OBS", candidate_id, "boundary_output"),
                        kind="boundary_output",
                        target=boundary.candidate_id,
                        evidence_ids=boundary.fact_ids + [guard.fact_id],
                    )
                ],
                scenario_checks=checks,
                runtime_witnesses=[witness],
                dependent_boundaries=[boundary.candidate_id],
            )
        )
    return candidates


def _candidate_from_terminal_effects(
    target_function: str,
    wrapper: str,
    facts: Sequence[SourceFact],
    boundaries: Sequence[BoundaryCandidate],
) -> Optional[ScenarioCandidate]:
    success_return = _first_success_return(facts, target_function)
    field_writes = [
        fact
        for fact in _facts_by_kind(facts, "FIELD_WRITE")
        if fact.function == target_function and (success_return is None or fact.start_line <= success_return.start_line)
    ]
    if success_return is None and not field_writes:
        return None

    candidate_id = f"{target_function}:success:{_stable_id('A', target_function, 'success')}"
    source_anchors = []
    if success_return is not None:
        source_anchors.append(success_return.fact_id)
        source_anchors.extend(
            edge.fact_id
            for edge in _fallthrough_edges_before_success(facts, target_function, success_return.start_line)
        )
    source_anchors.extend(fact.fact_id for fact in field_writes)

    checks: List[ResultCheck] = []
    observations: List[Observation] = []
    if success_return is not None:
        checks.append(
            ResultCheck(
                check_id=_stable_id("CHK", candidate_id, "return"),
                kind="ReturnEquals",
                target="return_value",
                expected_relation="equals 0",
                evidence_ids=[success_return.fact_id],
            )
        )
        observations.append(
            Observation(
                observation_id=_stable_id("OBS", candidate_id, "return"),
                kind="return_value",
                target="return_value",
                evidence_ids=[success_return.fact_id],
            )
        )
    for fact in field_writes:
        metadata = fact.metadata or {}
        checks.append(
            ResultCheck(
                check_id=_stable_id("CHK", candidate_id, fact.fact_id, "field"),
                kind="FieldEquals",
                target=metadata.get("left", ""),
                expected_relation=f"equals {metadata.get('right', '')}",
                evidence_ids=[fact.fact_id],
            )
        )
        observations.append(
            Observation(
                observation_id=_stable_id("OBS", candidate_id, fact.fact_id, "field"),
                kind="driver_state",
                target=metadata.get("left", ""),
                evidence_ids=[fact.fact_id],
            )
        )

    fact_by_id = _fact_map(facts)
    active_boundaries = []
    seen_boundaries: Set[str] = set()
    for boundary in boundaries:
        include = _scenario_boundary(boundary)
        if not include and _runtime_dependency_boundary(boundary):
            include = bool(_boundary_return_targets(boundary, fact_by_id, target_function))
        if not include or boundary.candidate_id in seen_boundaries:
            continue
        seen_boundaries.add(boundary.candidate_id)
        active_boundaries.append(boundary)
    witnesses = [
        _boundary_witness(candidate_id, boundary, "called", "boundary call_count >= 1")
        for boundary in active_boundaries
    ]
    constraints = [
        _constraint_for_boundary(
            candidate_id,
            boundary,
            facts,
            [
                _required_effect(
                    candidate_id,
                    boundary,
                    "success_compatible",
                    f"boundary {boundary.expression} produces a success-compatible effect for this scenario",
                    "target follows the success path after the boundary interaction",
                    boundary.fact_ids,
                )
            ],
            [witness],
        )
        for boundary, witness in zip(active_boundaries, witnesses)
    ]

    return _candidate(
        candidate_id=candidate_id,
        target_function=target_function,
        wrapper=wrapper,
        source_anchors=source_anchors,
        facts=facts,
        derivation="source_fact_graph:terminal_effects",
        trigger_conditions=[
            Condition(
                condition_id=_stable_id("COND", candidate_id, "valid_env"),
                expression="valid initialized environment with success-compatible boundary effects",
                evidence_ids=source_anchors,
            )
        ]
        + (
            [
                _condition_for_branch_edge(candidate_id, edge)
                for edge in _fallthrough_edges_before_success(facts, target_function, success_return.start_line)
            ]
            if success_return is not None
            else []
        ),
        hardware_environment_constraints=constraints,
        observations=observations,
        scenario_checks=checks,
        runtime_witnesses=witnesses,
        dependent_boundaries=[boundary.candidate_id for boundary in active_boundaries],
    )


def _candidate_from_unanchored_return(
    target_function: str,
    wrapper: str,
    return_fact: SourceFact,
    facts: Sequence[SourceFact],
    anchored_return_ids: Set[str],
    boundaries: Sequence[BoundaryCandidate],
    fact_by_id: Dict[str, SourceFact],
) -> Optional[ScenarioCandidate]:
    if return_fact.function != target_function or return_fact.fact_id in anchored_return_ids:
        return None
    expression = (return_fact.metadata or {}).get("return_expression", "")
    if expression == "0":
        return None
    candidate_id = f"{target_function}:return:{_stable_id('A', return_fact.fact_id)}"
    dependent_boundaries = _boundaries_reaching_return(return_fact, boundaries, fact_by_id, target_function)
    witnesses = [
        _boundary_witness(candidate_id, boundary, "called", "boundary call_count == 1")
        for boundary in dependent_boundaries
    ]
    constraints = [
        _constraint_for_boundary(
            candidate_id,
            boundary,
            facts,
            [
                _required_effect(
                    candidate_id,
                    boundary,
                    "return_compatible",
                    f"boundary {boundary.expression} produces the value that reaches target return `{expression}`",
                    f"target return `{expression}` is determined by this boundary interaction",
                    list(boundary.fact_ids) + [return_fact.fact_id],
                )
            ],
            [witness],
        )
        for boundary, witness in zip(dependent_boundaries, witnesses)
    ]
    checks: List[ResultCheck] = []
    if _is_grounded_exact_return_expression(expression):
        checks.append(
            ResultCheck(
                check_id=_stable_id("CHK", candidate_id, "return"),
                kind="ReturnEquals",
                target="return_value",
                expected_relation=f"equals {expression}",
                evidence_ids=[return_fact.fact_id],
            )
        )
    for boundary in dependent_boundaries:
        checks.append(
            ResultCheck(
                check_id=_stable_id("CHK", candidate_id, boundary.candidate_id, "boundary_called"),
                kind="BoundaryCalled",
                target=boundary.candidate_id,
                expected_relation="boundary call_count == 1",
                evidence_ids=list(boundary.fact_ids),
            )
        )
    return _candidate(
        candidate_id=candidate_id,
        target_function=target_function,
        wrapper=wrapper,
        source_anchors=[return_fact.fact_id],
        facts=facts,
        derivation="source_fact_graph:unanchored_return",
        trigger_conditions=[
            Condition(
                condition_id=_stable_id("COND", candidate_id, "reach_return"),
                expression=f"drive execution to return statement at line {return_fact.start_line}",
                evidence_ids=[return_fact.fact_id],
            )
        ],
        observations=[
            Observation(
                observation_id=_stable_id("OBS", candidate_id, "return"),
                kind="return_value",
                target="return_value",
                evidence_ids=[return_fact.fact_id],
            )
        ],
        scenario_checks=checks,
        runtime_witnesses=witnesses,
        hardware_environment_constraints=constraints,
        dependent_boundaries=[boundary.candidate_id for boundary in dependent_boundaries],
    )


def _candidate_from_branch_edge(
    target_function: str,
    wrapper: str,
    edge_fact: SourceFact,
    facts: Sequence[SourceFact],
    anchored_fact_ids: Set[str],
) -> Optional[ScenarioCandidate]:
    if edge_fact.function != target_function or edge_fact.fact_id in anchored_fact_ids:
        return None
    guard = _guard_for_branch_edge(facts, edge_fact)
    candidate_id = f"{target_function}:condition_edge:{_stable_id('A', edge_fact.fact_id)}"
    source_anchors = [edge_fact.fact_id]
    if guard is not None:
        source_anchors.insert(0, guard.fact_id)

    checks: List[ResultCheck] = []
    metadata = edge_fact.metadata or {}
    return_fact = _return_at_line(facts, int(metadata.get("first_return_line") or 0))
    if metadata.get("edge") == "branch" and return_fact is not None:
        source_anchors.append(return_fact.fact_id)

    return _candidate(
        candidate_id=candidate_id,
        target_function=target_function,
        wrapper=wrapper,
        source_anchors=source_anchors,
        facts=facts,
        derivation="source_fact_graph:condition_edge",
        trigger_conditions=[_condition_for_branch_edge(candidate_id, edge_fact)],
        observations=[
            Observation(
                observation_id=_stable_id("OBS", candidate_id, edge_fact.fact_id, "edge"),
                kind="branch_edge",
                target=edge_fact.fact_id,
                evidence_ids=[edge_fact.fact_id],
            )
        ],
        scenario_checks=checks,
    )


def _candidate_from_condition_case(
    target_function: str,
    wrapper: str,
    guard: SourceFact,
    case: Dict,
    facts: Sequence[SourceFact],
) -> Optional[ScenarioCandidate]:
    if guard.function != target_function or not case.get("decomposed"):
        return None
    edge = case.get("outcome", "")
    if edge not in {"branch", "fallthrough"}:
        return None
    edge_fact = _branch_edge_for_guard(facts, guard, edge)
    candidate_id = f"{target_function}:condition_case:{_stable_id('A', guard.fact_id, case.get('case_id', ''), edge)}"
    source_anchors = [guard.fact_id]
    if edge_fact is not None:
        source_anchors.append(edge_fact.fact_id)
    return_line = int((guard.metadata or {}).get("first_return_line") or 0)
    return_fact = _return_at_line(facts, return_line)
    if edge == "branch" and return_fact is not None:
        source_anchors.append(return_fact.fact_id)
    trigger_text = _condition_case_text(case)
    return _candidate(
        candidate_id=candidate_id,
        target_function=target_function,
        wrapper=wrapper,
        source_anchors=source_anchors,
        facts=facts,
        derivation="source_fact_graph:condition_case",
        trigger_conditions=[
            Condition(
                condition_id=_stable_id("COND", candidate_id, "condition_case"),
                expression=trigger_text,
                evidence_ids=source_anchors,
            )
        ],
        observations=[
            Observation(
                observation_id=_stable_id("OBS", candidate_id, "condition_case"),
                kind="condition_case",
                target=str(case.get("case_id", "")),
                evidence_ids=source_anchors,
            )
        ],
        scenario_checks=[],
    )


def _candidate_from_loop_boundary(
    target_function: str,
    wrapper: str,
    loop_fact: SourceFact,
    loop_case: Dict,
    facts: Sequence[SourceFact],
) -> Optional[ScenarioCandidate]:
    if loop_fact.function != target_function:
        return None
    case_id = str(loop_case.get("case_id", ""))
    description = str(loop_case.get("description", ""))
    if not case_id or not description:
        return None
    metadata = loop_fact.metadata or {}
    loop_condition = metadata.get("condition", "")
    candidate_id = f"{target_function}:loop_boundary:{_stable_id('A', loop_fact.fact_id, case_id)}"
    trigger = description
    if loop_condition:
        trigger += f"; loop condition: `{loop_condition}`"
    initializer = metadata.get("initializer", "")
    update = metadata.get("update", "")
    if initializer:
        trigger += f"; initializer: `{initializer}`"
    if update:
        trigger += f"; update: `{update}`"
    return _candidate(
        candidate_id=candidate_id,
        target_function=target_function,
        wrapper=wrapper,
        source_anchors=[loop_fact.fact_id],
        facts=facts,
        derivation="source_fact_graph:loop_boundary",
        trigger_conditions=[
            Condition(
                condition_id=_stable_id("COND", candidate_id, "loop_boundary"),
                expression=trigger,
                evidence_ids=[loop_fact.fact_id],
            )
        ],
        observations=[
            Observation(
                observation_id=_stable_id("OBS", candidate_id, "loop_boundary"),
                kind="loop_execution",
                target=case_id,
                evidence_ids=[loop_fact.fact_id],
            )
        ],
        scenario_checks=[],
    )


def _loop_iteration_count_is_input_controlled(loop_fact: SourceFact) -> bool:
    metadata = loop_fact.metadata or {}
    condition_symbols = set(metadata.get("condition_symbols", []) or [])
    function_parameters = set(metadata.get("function_parameters", []) or [])
    if condition_symbols & function_parameters:
        return True
    parameter_roots = {
        symbol
        for symbol in condition_symbols
        if symbol.split(".")[0] in function_parameters
    }
    return bool(parameter_roots)


def _candidate_fingerprint(candidate: ScenarioCandidate) -> Tuple:
    checks = tuple(
        sorted((check.kind, check.target, check.expected_relation) for check in candidate.scenario_checks)
    )
    triggers = tuple(sorted(condition.expression for condition in candidate.trigger_conditions))
    witnesses = tuple(
        sorted((witness.kind, witness.target, witness.relation) for witness in candidate.runtime_witnesses)
    )
    constraints = tuple(
        sorted(
            (
                constraint.boundary_id,
                tuple(effect.relation for effect in constraint.required_effects),
            )
            for constraint in candidate.hardware_environment_constraints
        )
    )
    return (
        tuple(sorted(candidate.source_anchors)),
        tuple(sorted(candidate.dependent_boundaries)),
        triggers,
        checks,
        witnesses,
        constraints,
    )


def _dedupe_candidates(candidates: Sequence[ScenarioCandidate]) -> List[ScenarioCandidate]:
    seen_ids: Set[str] = set()
    seen_fingerprints: Set[Tuple] = set()
    unique: List[ScenarioCandidate] = []
    for candidate in candidates:
        if candidate.candidate_id in seen_ids:
            continue
        fingerprint = _candidate_fingerprint(candidate)
        if fingerprint in seen_fingerprints:
            continue
        seen_ids.add(candidate.candidate_id)
        seen_fingerprints.add(fingerprint)
        unique.append(candidate)
    return unique


def build_scenario_candidates(
    target_function: str,
    wrapper: str,
    facts: Sequence[SourceFact],
    boundaries: Sequence[BoundaryCandidate],
) -> List[ScenarioCandidate]:
    fact_by_id = _fact_map(facts)
    candidates: List[ScenarioCandidate] = []

    for guard in sorted(_facts_by_kind(facts, "GUARD"), key=lambda item: item.start_line):
        if guard.function != target_function:
            continue
        candidate = _candidate_from_guard(target_function, wrapper, guard, facts, boundaries, fact_by_id)
        if candidate is not None:
            candidates.append(candidate)
        for case in (guard.metadata or {}).get("condition_cases", []) or []:
            candidate = _candidate_from_condition_case(target_function, wrapper, guard, case, facts)
            if candidate is not None:
                candidates.append(candidate)

    for loop_fact in sorted(_facts_by_kind(facts, "LOOP"), key=lambda item: item.start_line):
        if loop_fact.function != target_function:
            continue
        if not _loop_iteration_count_is_input_controlled(loop_fact):
            continue
        for loop_case in (loop_fact.metadata or {}).get("loop_cases", []) or []:
            candidate = _candidate_from_loop_boundary(target_function, wrapper, loop_fact, loop_case, facts)
            if candidate is not None:
                candidates.append(candidate)

    for boundary in sorted(boundaries, key=lambda item: (item.source_line, item.candidate_id)):
        candidate = _candidate_from_boundary_result(target_function, wrapper, boundary, facts, fact_by_id)
        if candidate is not None:
            candidates.append(candidate)
        candidates.extend(_candidates_from_boundary_outputs(target_function, wrapper, boundary, facts, fact_by_id))

    success = _candidate_from_terminal_effects(target_function, wrapper, facts, boundaries)
    if success is not None:
        candidates.append(success)

    anchored_return_ids = {
        anchor
        for candidate in candidates
        for anchor in candidate.source_anchors
        if fact_by_id.get(anchor) is not None and fact_by_id[anchor].kind == "RETURN"
    }
    for return_fact in sorted(_facts_by_kind(facts, "RETURN"), key=lambda item: item.start_line):
        candidate = _candidate_from_unanchored_return(
            target_function,
            wrapper,
            return_fact,
            facts,
            anchored_return_ids,
            boundaries,
            fact_by_id,
        )
        if candidate is not None:
            candidates.append(candidate)

    anchored_fact_ids = {
        anchor
        for candidate in candidates
        for anchor in candidate.source_anchors
    }
    for edge_fact in sorted(_facts_by_kind(facts, "BRANCH_EDGE"), key=lambda item: item.start_line):
        candidate = _candidate_from_branch_edge(
            target_function,
            wrapper,
            edge_fact,
            facts,
            anchored_fact_ids,
        )
        if candidate is not None:
            candidates.append(candidate)
            anchored_fact_ids.update(candidate.source_anchors)

    return _dedupe_candidates(candidates)


def scenario_contract_from_candidate(candidate: ScenarioCandidate) -> ScenarioContract:
    return ScenarioContract(
        scenario_id=candidate.candidate_id,
        target_function=candidate.target_function,
        export_function=candidate.export_function,
        source_anchors=list(candidate.source_anchors),
        trigger_conditions=list(candidate.trigger_conditions),
        hardware_environment_constraints=list(candidate.hardware_environment_constraints),
        observations=list(candidate.observations),
        scenario_checks=list(candidate.scenario_checks),
        runtime_witnesses=list(candidate.runtime_witnesses),
        dependent_boundaries=list(candidate.dependent_boundaries),
        source_candidate_id=candidate.candidate_id,
        derivation=candidate.derivation,
        relation_edges=list(candidate.relation_edges),
    )


def scenario_contracts_from_candidates(
    candidates: Sequence[ScenarioCandidate],
) -> List[ScenarioContract]:
    return [scenario_contract_from_candidate(candidate) for candidate in candidates]
