from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class SourceFact:
    fact_id: str
    kind: str
    file: str
    function: str
    start_line: int
    end_line: int
    code: str
    symbols: List[str] = field(default_factory=list)
    related_fact_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BoundaryCandidate:
    candidate_id: str
    expression: str
    fact_ids: List[str]
    source_function: str
    source_line: int
    source_fact_kind: str = ""
    access_path: List[str] = field(default_factory=list)
    arguments: List[str] = field(default_factory=list)
    argument_paths: List[List[str]] = field(default_factory=list)
    argument_type_map: List[Dict[str, Any]] = field(default_factory=list)
    output_arguments: List[str] = field(default_factory=list)
    result_assignee: str = ""
    semantic_role: str = "unknown"
    classification_reason: str = ""
    classification_evidence_ids: List[str] = field(default_factory=list)
    environment_notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BoundaryPrecondition:
    precondition_id: str
    boundary_id: str
    description: str
    evidence_ids: List[str] = field(default_factory=list)
    object_path: List[str] = field(default_factory=list)
    callsite_object_path: List[str] = field(default_factory=list)
    callsite_evidence_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BoundaryRequiredEffect:
    effect_id: str
    boundary_id: str
    description: str
    evidence_ids: List[str] = field(default_factory=list)
    relation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Condition:
    condition_id: str
    expression: str
    evidence_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Observation:
    observation_id: str
    kind: str
    target: str
    evidence_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ResultCheck:
    check_id: str
    kind: str
    target: str
    expected_relation: str
    evidence_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RuntimeWitness:
    witness_id: str
    kind: str
    target: str
    relation: str
    evidence_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class HardwareEnvironmentConstraint:
    constraint_id: str
    boundary_id: str
    boundary_expression: str
    source_function: str
    source_line: int
    source_fact_ids: List[str] = field(default_factory=list)
    preconditions: List[BoundaryPrecondition] = field(default_factory=list)
    required_effects: List[BoundaryRequiredEffect] = field(default_factory=list)
    runtime_witnesses: List[RuntimeWitness] = field(default_factory=list)
    relation_edges: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["preconditions"] = [item.to_dict() for item in self.preconditions]
        data["required_effects"] = [item.to_dict() for item in self.required_effects]
        data["runtime_witnesses"] = [item.to_dict() for item in self.runtime_witnesses]
        return data


@dataclass
class ScenarioCandidate:
    candidate_id: str
    target_function: str
    export_function: str
    source_anchors: List[str] = field(default_factory=list)
    trigger_conditions: List[Condition] = field(default_factory=list)
    hardware_environment_constraints: List[HardwareEnvironmentConstraint] = field(default_factory=list)
    observations: List[Observation] = field(default_factory=list)
    scenario_checks: List[ResultCheck] = field(default_factory=list)
    runtime_witnesses: List[RuntimeWitness] = field(default_factory=list)
    dependent_boundaries: List[str] = field(default_factory=list)
    derivation: str = ""
    relation_edges: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["trigger_conditions"] = [item.to_dict() for item in self.trigger_conditions]
        data["hardware_environment_constraints"] = [
            item.to_dict() for item in self.hardware_environment_constraints
        ]
        data["observations"] = [item.to_dict() for item in self.observations]
        data["scenario_checks"] = [item.to_dict() for item in self.scenario_checks]
        data["runtime_witnesses"] = [item.to_dict() for item in self.runtime_witnesses]
        return data


@dataclass
class ScenarioContract:
    scenario_id: str
    target_function: str
    export_function: str
    source_anchors: List[str] = field(default_factory=list)
    trigger_conditions: List[Condition] = field(default_factory=list)
    hardware_environment_constraints: List[HardwareEnvironmentConstraint] = field(default_factory=list)
    observations: List[Observation] = field(default_factory=list)
    scenario_checks: List[ResultCheck] = field(default_factory=list)
    runtime_witnesses: List[RuntimeWitness] = field(default_factory=list)
    test_function: Optional[str] = None
    dependent_boundaries: List[str] = field(default_factory=list)
    source_candidate_id: str = ""
    derivation: str = ""
    relation_edges: List[Dict[str, Any]] = field(default_factory=list)
    status: str = "PLANNED"
    version: int = 1

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["trigger_conditions"] = [item.to_dict() for item in self.trigger_conditions]
        data["hardware_environment_constraints"] = [
            item.to_dict() for item in self.hardware_environment_constraints
        ]
        data["observations"] = [item.to_dict() for item in self.observations]
        data["scenario_checks"] = [item.to_dict() for item in self.scenario_checks]
        data["runtime_witnesses"] = [item.to_dict() for item in self.runtime_witnesses]
        return data


@dataclass
class ScenarioRegistry:
    version: int
    target_function: str
    export_function: str
    source_facts: List[SourceFact] = field(default_factory=list)
    boundary_candidates: List[BoundaryCandidate] = field(default_factory=list)
    scenario_candidates: List[ScenarioCandidate] = field(default_factory=list)
    scenario_contracts: List[ScenarioContract] = field(default_factory=list)
    internal_call_closure: List[str] = field(default_factory=list)
    classification_warnings: List[str] = field(default_factory=list)
    revision_history: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "target_function": self.target_function,
            "export_function": self.export_function,
            "internal_call_closure": self.internal_call_closure,
            "source_facts": [item.to_dict() for item in self.source_facts],
            "boundary_candidates": [item.to_dict() for item in self.boundary_candidates],
            "scenario_candidates": [item.to_dict() for item in self.scenario_candidates],
            "scenario_contracts": [item.to_dict() for item in self.scenario_contracts],
            "classification_warnings": self.classification_warnings,
            "revision_history": self.revision_history,
        }


def source_fact_map(registry: ScenarioRegistry) -> Dict[str, SourceFact]:
    return {fact.fact_id: fact for fact in registry.source_facts}
