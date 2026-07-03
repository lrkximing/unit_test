from dataclasses import dataclass, field
from typing import Dict, List

from verification.scenario_static_verifier import verify_scenario_contracts


@dataclass
class ScenarioContextValidationResult:
    ok: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    covered_scenarios: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "ok": self.ok,
            "errors": self.errors,
            "warnings": self.warnings,
            "covered_scenarios": self.covered_scenarios,
        }


def _registry(scenario_context: Dict) -> Dict:
    if not isinstance(scenario_context, dict):
        return {}
    return scenario_context.get("scenario_registry") or {}


def validate_scenario_context_structure(scenario_context: Dict) -> ScenarioContextValidationResult:
    errors: List[str] = []
    warnings: List[str] = []
    registry = _registry(scenario_context)
    if not registry:
        return ScenarioContextValidationResult(
            ok=False,
            errors=["Scenario context missing scenario_registry."],
        )
    if not registry.get("target_function"):
        errors.append("Scenario registry missing target_function.")
    if not registry.get("export_function"):
        errors.append("Scenario registry missing export_function.")
    if not registry.get("source_facts"):
        warnings.append("Scenario registry contains no source facts.")
    if not isinstance(registry.get("boundary_candidates", []), list):
        errors.append("Scenario registry boundary_candidates must be a list.")
    if not isinstance(registry.get("scenario_candidates", []), list):
        errors.append("Scenario registry scenario_candidates must be a list.")
    contracts = registry.get("scenario_contracts", [])
    if not isinstance(contracts, list) or not contracts:
        errors.append("Scenario registry must include at least one scenario contract.")
    for idx, contract in enumerate(contracts if isinstance(contracts, list) else []):
        scenario_id = contract.get("scenario_id")
        if not scenario_id:
            errors.append(f"Scenario contract at index {idx} missing scenario_id.")
        if not contract.get("source_candidate_id"):
            warnings.append(f"Scenario contract {scenario_id or idx} has no source_candidate_id.")
        if not contract.get("source_anchors"):
            warnings.append(f"Scenario contract {scenario_id or idx} has no source anchors.")
        if not contract.get("scenario_checks") and contract.get("derivation") != "source_fact_graph:condition_edge":
            warnings.append(f"Scenario contract {scenario_id or idx} has no scenario checks.")
        for check in contract.get("scenario_checks", []) or []:
            if not check.get("check_id"):
                errors.append(f"Scenario contract {scenario_id or idx} has scenario check without check_id.")
            if not check.get("evidence_ids"):
                errors.append(f"Scenario check {check.get('check_id', '<unknown>')} has no evidence_ids.")
        for witness in contract.get("runtime_witnesses", []) or []:
            if not witness.get("witness_id"):
                errors.append(f"Scenario contract {scenario_id or idx} has runtime witness without witness_id.")
            if not witness.get("evidence_ids"):
                errors.append(f"Runtime witness {witness.get('witness_id', '<unknown>')} has no evidence_ids.")
        for constraint in contract.get("hardware_environment_constraints", []) or []:
            boundary_id = constraint.get("boundary_id", "")
            if not boundary_id:
                errors.append(f"Scenario contract {scenario_id or idx} has hardware constraint without boundary_id.")
            if not constraint.get("source_fact_ids"):
                errors.append(f"Hardware constraint {constraint.get('constraint_id', '<unknown>')} has no source_fact_ids.")
            for precondition in constraint.get("preconditions", []) or []:
                if not precondition.get("precondition_id"):
                    errors.append(f"Hardware constraint {constraint.get('constraint_id', '<unknown>')} has precondition without precondition_id.")
                if not precondition.get("evidence_ids"):
                    errors.append(f"Precondition {precondition.get('precondition_id', '<unknown>')} has no evidence_ids.")
            for effect in constraint.get("required_effects", []) or []:
                if not effect.get("effect_id"):
                    errors.append(f"Hardware constraint {constraint.get('constraint_id', '<unknown>')} has required effect without effect_id.")
                if not effect.get("evidence_ids"):
                    errors.append(f"Required effect {effect.get('effect_id', '<unknown>')} has no evidence_ids.")
            for witness in constraint.get("runtime_witnesses", []) or []:
                if not witness.get("witness_id"):
                    errors.append(f"Hardware constraint {constraint.get('constraint_id', '<unknown>')} has runtime witness without witness_id.")
                if not witness.get("evidence_ids"):
                    errors.append(f"Hardware constraint witness {witness.get('witness_id', '<unknown>')} has no evidence_ids.")
    return ScenarioContextValidationResult(ok=not errors, errors=errors, warnings=warnings)


def validate_test_against_scenario_context(
    test_code: str,
    scenario_context: Dict,
) -> ScenarioContextValidationResult:
    structure = validate_scenario_context_structure(scenario_context)
    if not structure.ok:
        return structure
    scenario_result = verify_scenario_contracts(test_code, scenario_context)
    return ScenarioContextValidationResult(
        ok=scenario_result.ok,
        errors=scenario_result.errors,
        warnings=structure.warnings + scenario_result.warnings,
        covered_scenarios=scenario_result.covered_scenarios,
    )
