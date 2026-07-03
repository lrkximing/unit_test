from dataclasses import dataclass, field
from typing import Dict, List, Optional

from validation.region_check import extract_test_names


@dataclass
class RepairAudit:
    test_count_before: int
    test_count_after: int
    removed_tests: List[str] = field(default_factory=list)
    added_tests: List[str] = field(default_factory=list)
    target_wrapper_missing: bool = False

    def to_dict(self) -> Dict:
        return {
            "test_count_before": self.test_count_before,
            "test_count_after": self.test_count_after,
            "removed_tests": self.removed_tests,
            "added_tests": self.added_tests,
            "target_wrapper_missing": self.target_wrapper_missing,
        }


def build_repair_audit(before: str, after: str, scenario_context: Optional[Dict] = None) -> RepairAudit:
    before_tests = set(extract_test_names(before or ""))
    after_tests = set(extract_test_names(after or ""))

    wrapper_missing = False
    wrapper = ((scenario_context or {}).get("target") or {}).get("wrapper")
    if wrapper and wrapper not in (after or ""):
        wrapper_missing = True

    return RepairAudit(
        test_count_before=len(before_tests),
        test_count_after=len(after_tests),
        removed_tests=sorted(before_tests - after_tests),
        added_tests=sorted(after_tests - before_tests),
        target_wrapper_missing=wrapper_missing,
    )
