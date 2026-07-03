import hashlib
import re
from typing import Dict, Iterable, List, Set

from scenario.fact_model import BoundaryCandidate, SourceFact


def _stable_id(prefix: str, *parts: object) -> str:
    text = ":".join(str(part) for part in parts)
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]
    return f"{prefix}_{digest}"


def _internal_function_names(parse_result) -> Set[str]:
    return set(((getattr(parse_result, "file_level_defs", {}) or {}).get("functions", set())) or set())


def _file_local_macro_names(parse_result) -> Set[str]:
    return set(((getattr(parse_result, "file_level_defs", {}) or {}).get("macros", set())) or set())


def _has_object_like_argument(metadata: Dict) -> bool:
    for path in metadata.get("argument_paths", []) or []:
        if len(path) >= 2:
            return True
    for item in metadata.get("argument_path_map", []) or []:
        text = str((item or {}).get("text", ""))
        if "->" in text or "." in text:
            return True
    for item in metadata.get("argument_type_map", []) or []:
        type_text = str((item or {}).get("type", ""))
        if "*" in type_text or re.search(r"\bstruct\s+[A-Za-z_][A-Za-z0-9_]*\b", type_text):
            return True
    return False


def _call_has_controllable_effect(fact: SourceFact) -> bool:
    metadata = fact.metadata or {}
    if metadata.get("result_assignee"):
        return True
    if metadata.get("output_arguments"):
        return True
    if metadata.get("statement_kind") in {"return_statement", "Return"}:
        return True
    for edge in metadata.get("relation_edges", []) or []:
        if not isinstance(edge, dict):
            continue
        relation = str(edge.get("relation", "") or "")
        if relation.startswith("call_result_") or relation.startswith("call_output_"):
            return True
    return False


def _call_result_is_pointer_setup(fact: SourceFact) -> bool:
    metadata = fact.metadata or {}
    if metadata.get("output_arguments"):
        return False
    result_assignee = str(metadata.get("result_assignee", "") or "")
    statement = str(metadata.get("statement", "") or "")
    if "*" in result_assignee:
        return True
    if re.search(r"\b(?:const\s+)?(?:struct\s+)?[A-Za-z_][A-Za-z0-9_\s]*\s*\*\s*[A-Za-z_][A-Za-z0-9_]*\s*=", statement):
        return True
    return False


def _call_looks_like_pure_software_helper(fact: SourceFact) -> bool:
    metadata = fact.metadata or {}
    if fact.kind != "CALL":
        return False
    if _call_result_is_pointer_setup(fact):
        return True
    if not _call_has_controllable_effect(fact):
        return True
    relations = {
        str(edge.get("relation", ""))
        for edge in metadata.get("relation_edges", []) or []
        if isinstance(edge, dict)
    }
    if not relations:
        return False
    return all(
        relation.startswith("call_result_influences_assignment")
        or relation.startswith("call_result_influences_return")
        for relation in relations
    )


def _boundary_from_call(fact: SourceFact, internal_functions: Set[str]) -> BoundaryCandidate:
    metadata = fact.metadata or {}
    callee = metadata.get("callee_name", "")
    expression = metadata.get("callee_expression", callee or fact.code)
    path = metadata.get("callee_path", []) or []
    is_direct_external = fact.kind == "CALL" and callee not in internal_functions
    has_controllable_effect = _call_has_controllable_effect(fact)
    is_pure_software_helper = _call_looks_like_pure_software_helper(fact)
    is_pointer_setup = _call_result_is_pointer_setup(fact)
    is_object_backed_external = is_direct_external and _has_object_like_argument(metadata) and has_controllable_effect
    notes = []
    if is_direct_external:
        notes = [
            "This is a direct external call from driver code. A same-name fake function or macro in the test file will not intercept it.",
            "The test must configure a real object/hook/wrapper/instrumentation path that affects this production call; if repeated repair attempts cannot do so, the scenario is marked unrealizable by the iteration loop.",
        ]
    elif fact.kind == "MEMBER_CALL":
        notes = [
            "This is an indirect member/function-pointer call. The test environment must initialize the object path and install a callable implementation before the target reaches it."
        ]
    candidate_id = _stable_id("B", fact.function, fact.start_line, expression, fact.kind)
    return BoundaryCandidate(
        candidate_id=candidate_id,
        expression=expression,
        fact_ids=[fact.fact_id],
        source_function=fact.function,
        source_line=fact.start_line,
        source_fact_kind=fact.kind,
        access_path=path,
        arguments=metadata.get("arguments", []) or [],
        argument_paths=metadata.get("argument_paths", []) or [],
        argument_type_map=metadata.get("argument_type_map", []) or [],
        output_arguments=metadata.get("output_arguments", []) or [],
        result_assignee=metadata.get("result_assignee", "") or "",
        semantic_role=(
            "hardware_boundary"
            if fact.kind == "MEMBER_CALL" or (is_object_backed_external and not is_pointer_setup)
            else "environment_prerequisite"
            if is_direct_external and is_pointer_setup
            else "ordinary_helper"
            if is_pure_software_helper
            else "unknown"
        ),
        classification_reason=(
            "direct external pointer-return setup call treated as environment prerequisite"
            if is_direct_external and is_pointer_setup
            else
            "pure software helper candidate excluded from hardware/environment boundary scenarios"
            if is_pure_software_helper
            else "direct external call with object-backed argument path"
            if is_object_backed_external
            else "direct external call candidate" if fact.kind == "CALL" and callee not in internal_functions
            else "member-call boundary candidate"
        ),
        classification_evidence_ids=[fact.fact_id],
        environment_notes=notes,
    )


def _boundary_from_field_access(fact: SourceFact) -> BoundaryCandidate:
    metadata = fact.metadata or {}
    path = metadata.get("field_path", []) or metadata.get("left_path", []) or []
    expression = ".".join(path) if path else fact.code
    candidate_id = _stable_id("B", fact.function, fact.start_line, expression, fact.kind)
    return BoundaryCandidate(
        candidate_id=candidate_id,
        expression=expression,
        fact_ids=[fact.fact_id],
        source_function=fact.function,
        source_line=fact.start_line,
        source_fact_kind=fact.kind,
        access_path=path,
        semantic_role="environment_prerequisite",
        classification_reason="object field access required by the scenario environment",
        classification_evidence_ids=[fact.fact_id],
        environment_notes=[
            "This field access is an environment prerequisite. It should be initialized as part of the scenario setup, not mocked as a separate interaction."
        ],
    )


def analyze_boundary_candidates(parse_result, facts: Iterable[SourceFact]) -> List[BoundaryCandidate]:
    """Extract hardware/environment boundary candidates without prescribing mock technique.

    The result describes where the target or its helper closure crosses a hardware
    or execution-environment boundary.  Whether a test realizes the boundary via a
    fake object, callback, hook, wrapper, or another technique is left to the LLM
    and verified through scenario-level constraints.
    """
    internal_functions = _internal_function_names(parse_result)
    file_local_macros = _file_local_macro_names(parse_result)
    candidates: Dict[str, BoundaryCandidate] = {}

    for fact in facts:
        candidate = None
        metadata = fact.metadata or {}
        if fact.kind == "CALL":
            callee = metadata.get("callee_name", "")
            if not callee or callee in internal_functions:
                continue
            if callee in file_local_macros:
                continue
            candidate = _boundary_from_call(fact, internal_functions)
        elif fact.kind == "MEMBER_CALL":
            candidate = _boundary_from_call(fact, internal_functions)
        elif fact.kind in {"FIELD_ACCESS", "FIELD_WRITE"}:
            path = metadata.get("field_path", []) or metadata.get("left_path", []) or []
            if len(path) < 2:
                continue
            candidate = _boundary_from_field_access(fact)

        if candidate is None:
            continue
        if candidate.candidate_id in candidates:
            candidates[candidate.candidate_id].fact_ids.append(fact.fact_id)
            continue
        candidates[candidate.candidate_id] = candidate

    return sorted(candidates.values(), key=lambda item: (item.source_line, item.expression))
