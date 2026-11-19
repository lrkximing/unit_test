from dataclasses import dataclass, field
from typing import Dict, List

from validation.region_check import validate_protected_regions
from validation.repair_audit import build_repair_audit
from verification.oracle_contract_verifier import oracle_repair_errors
from verification.scenario_static_verifier import (
    blocking_scenario_static_errors,
    nonblocking_scenario_static_findings,
    verify_scenario_contracts,
)


@dataclass
class ScenarioPatchGateResult:
    ok: bool
    hard_errors: List[str] = field(default_factory=list)
    audit_warnings: List[str] = field(default_factory=list)
    report: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "ok": self.ok,
            "hard_errors": self.hard_errors,
            "audit_warnings": self.audit_warnings,
            "report": self.report,
        }


def evaluate_scenario_patch(
    before_code: str,
    after_code: str,
    scenario_context: Dict,
    frozen_tests=None,
) -> ScenarioPatchGateResult:
    from generation.mock_interfaces import install_mock_declarations
    from validation.test_inspector import inspect_test_source
    # Extend only the fixed declaration baseline with this framework-owned proposal.
    before_code = install_mock_declarations(before_code, scenario_context,
                                            set(inspect_test_source(after_code).function_definitions))
    region_result = validate_protected_regions(before_code, after_code, frozen_tests=frozen_tests or [])
    scenario_result = verify_scenario_contracts(after_code, scenario_context)
    audit = build_repair_audit(before_code, after_code, scenario_context)
    hard_errors: List[str] = []
    audit_warnings: List[str] = []

    if not region_result.ok:
        hard_errors.extend(region_result.errors)
    if not scenario_result.ok:
        hard_errors.extend(blocking_scenario_static_errors(scenario_result.errors))
        audit_warnings.extend(nonblocking_scenario_static_findings(scenario_result.errors))
    if audit.target_wrapper_missing:
        hard_errors.append("Patch removed or bypassed the target test_export wrapper call.")
    registry = scenario_context.get("scenario_registry") or scenario_context
    hard_errors.extend(oracle_repair_errors(before_code, after_code, registry, context=scenario_context))

    def bindings(code):
        return {t.name: (tuple(t.scenario_ids), t.variant_id or "initial")
                for t in inspect_test_source(code).test_functions}
    before_bindings, after_bindings = bindings(before_code), bindings(after_code)
    from scenario.condition_variants import contract_for_variant
    def established(identity):
        scenario_ids, variant_id = identity
        return len(scenario_ids) == 1 and contract_for_variant(scenario_context, scenario_ids[0], variant_id) is not None
    removed_planned = [name for name in audit.removed_tests if established(before_bindings[name])]
    if removed_planned:
        hard_errors.append(f"Patch removed planned tests: {removed_planned}")
    for name in before_bindings.keys() & after_bindings.keys():
        if established(before_bindings[name]) and before_bindings[name] != after_bindings[name]:
            hard_errors.append(f"Patch changed the scenario/condition binding of existing test {name}.")

    report = {
        "region": region_result.to_dict(),
        "scenario_static": scenario_result.to_dict(),
        "repair_audit": audit.to_dict(),
    }
    return ScenarioPatchGateResult(
        ok=not hard_errors,
        hard_errors=hard_errors,
        audit_warnings=audit_warnings,
        report=report,
    )
