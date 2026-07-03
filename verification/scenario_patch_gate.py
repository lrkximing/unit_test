from dataclasses import dataclass, field
from typing import Dict, List

from validation.region_check import validate_protected_regions
from validation.repair_audit import build_repair_audit
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

    if audit.removed_tests:
        audit_warnings.append(f"Patch removed tests: {audit.removed_tests}")

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
