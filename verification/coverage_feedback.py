"""Interpret measurement applicability without inventing edge semantics."""


def coverage_target_met(coverage, target):
    measurement = coverage.get("measurement", {})
    if not measurement.get("valid", True):
        return False
    if coverage.get("line_percent", 0.0) < target:
        return False
    branchless = (measurement.get("branch_applicable") is False or
                  (not measurement and coverage.get("branch_total") == 0 and
                   coverage.get("line_total", 0) > 0 and coverage.get("line_hit", 0) > 0))
    return branchless or coverage.get("branch_percent", 0.0) >= target


def evaluation_action(kunit, coverage, target, require_branch=True, scenarios_complete=True):
    """Keep execution repair, measurement failure and test expansion separate."""
    if not kunit.get("tests_total") or not kunit.get("log_complete", True):
        return "repair_results"
    if not kunit.get("overall_passed"):
        return "repair_tests"
    measurement = coverage.get("measurement", {})
    valid = measurement.get("valid", True) if require_branch else measurement.get("line_valid", measurement.get("valid", True))
    if not valid:
        return "measurement_review"
    if not scenarios_complete:
        return "repair_scenarios"
    met = coverage_target_met(coverage, target) if require_branch else coverage.get("line_percent", 0.0) >= target
    return "complete" if met else "extend_coverage"


def missed_branch_feedback(branch, source_line=""):
    text = f"Uncovered GCOV edge {branch.get('branch_index')} at line {branch.get('line')} (edge index does not establish a true/false outcome)."
    if source_line:
        text += " Source: " + source_line
    return text
