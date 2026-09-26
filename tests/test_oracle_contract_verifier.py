import unittest

from verification.scenario_patch_gate import evaluate_scenario_patch
from verification.scenario_static_verifier import (
    blocking_scenario_static_errors,
    verify_scenario_contracts,
)


def source(assertion, setup="int rc = target_export(dev);", marker="C_RET"):
    return (
        "static void sample_test(struct kunit *test)\n"
        "{\n"
        " /* RACA_SCENARIO: S1 */\n"
        f" {setup}\n"
        f" /* RACA_CHECK: {marker} */\n"
        f" {assertion}\n"
        "}\n"
        "static struct kunit_case sample_cases[] = { KUNIT_CASE(sample_test), {} };\n"
    )


def context(kind="ReturnEquals", expected="equals -EIO", target="return_value"):
    return {
        "scenario_registry": {
            "target_function": "driver_target",
            "export_function": "target_export",
            "scenario_contracts": [{
                "scenario_id": "S1",
                "export_function": "target_export",
                "scenario_checks": [{
                    "check_id": "C_RET",
                    "kind": kind,
                    "target": target,
                    "expected_relation": expected,
                }],
            }],
        }
    }


class OracleContractTests(unittest.TestCase):
    def test_return_alias_and_mock_configuration_are_checked(self):
        code = source(
            "KUNIT_EXPECT_EQ(test, result, expected);",
            "int mock_errno = -EIO;\n"
            " int expected = mock_errno;\n"
            " int rc = target_export(dev);\n"
            " int result = rc;",
        )
        result = verify_scenario_contracts(code, context())
        self.assertFalse(blocking_scenario_static_errors(result.errors), result.errors)

    def test_multiline_kunit_assertion_is_parsed(self):
        code = source(
            "KUNIT_EXPECT_EQ(\n"
            "     test,\n"
            "     rc,\n"
            "     -EIO);"
        )
        result = verify_scenario_contracts(code, context())
        self.assertFalse(blocking_scenario_static_errors(result.errors), result.errors)

    def test_wrong_expected_value_is_blocked(self):
        code = source("KUNIT_EXPECT_EQ(test, rc, 0);")
        errors = blocking_scenario_static_errors(verify_scenario_contracts(code, context()).errors)
        self.assertTrue(any("expected value differs" in item for item in errors), errors)

    def test_boolean_assertion_does_not_replace_exact_errno(self):
        code = source("KUNIT_EXPECT_TRUE(test, rc < 0);")
        errors = blocking_scenario_static_errors(verify_scenario_contracts(code, context()).errors)
        self.assertTrue(any("requires equality" in item for item in errors), errors)

    def test_equal_relation_accepts_reversed_assertion_operands(self):
        code = source("KUNIT_EXPECT_EQ(test, -EIO, rc);")
        result = verify_scenario_contracts(code, context())
        self.assertFalse(blocking_scenario_static_errors(result.errors), result.errors)

    def test_same_line_check_marker_is_bound(self):
        code = source("/* RACA_CHECK: C_RET */ KUNIT_EXPECT_EQ(test, rc, -EIO);", marker="C_OTHER")
        result = verify_scenario_contracts(code, context())
        self.assertFalse(blocking_scenario_static_errors(result.errors), result.errors)

    def test_marker_on_unrelated_assertion_is_not_an_oracle(self):
        code = source(
            "KUNIT_EXPECT_EQ(test, mock_calls, -EIO);",
            "int mock_calls = 1;\n int rc = target_export(dev);",
        )
        errors = blocking_scenario_static_errors(verify_scenario_contracts(code, context()).errors)
        self.assertTrue(any("does not inspect" in item for item in errors), errors)

    def test_distant_marker_does_not_bind_a_later_assertion(self):
        code = source(
            "KUNIT_EXPECT_EQ(test, rc, -EIO);\n"
            " KUNIT_EXPECT_EQ(test, rc, -EIO);",
            marker="C_OTHER",
        )
        errors = blocking_scenario_static_errors(verify_scenario_contracts(code, context()).errors)
        self.assertTrue(any("no assertion immediately bound" in item for item in errors), errors)

    def test_state_check_uses_post_call_value_and_selected_input(self):
        code = source(
            "KUNIT_EXPECT_EQ(test, dev->last, sample);",
            "int sample = 42;\n int rc = target_export(dev);",
        )
        result = verify_scenario_contracts(
            code, context(kind="FieldEquals", expected="equals 42", target="dev->last")
        )
        self.assertFalse(blocking_scenario_static_errors(result.errors), result.errors)

    def test_pre_call_state_snapshot_is_rejected(self):
        code = source(
            "KUNIT_EXPECT_EQ(test, snapshot, 42);",
            "int snapshot = dev->last;\n int rc = target_export(dev);",
        )
        errors = blocking_scenario_static_errors(
            verify_scenario_contracts(
                code, context(kind="FieldEquals", expected="equals 42", target="dev->last")
            ).errors
        )
        self.assertTrue(any("captured before" in item for item in errors), errors)

    def test_boundary_assertion_must_match_configured_mock_return(self):
        fake = (
            "/* RACA_MOCK: boundary=B1;original=dev->ops->read;replacement=fake_read */\n"
            "static int fake_read(void) { return mock_errno; }\n"
        )
        code = fake + source(
            "KUNIT_EXPECT_EQ(test, rc, -EINVAL);",
            "mock_errno = -EIO;\n"
            " dev->ops->read = fake_read;\n"
            " int rc = target_export(dev);",
        )
        errors = blocking_scenario_static_errors(
            verify_scenario_contracts(
                code, context(
                    kind="ReturnEqualsBoundaryEffect",
                    expected="equals negative errno produced by B1",
                )
            ).errors
        )
        self.assertTrue(any("configured boundary return" in item for item in errors), errors)

    def test_repair_rejects_equality_weakening(self):
        baseline = source("KUNIT_EXPECT_EQ(test, rc, 0);")
        revised = source("KUNIT_EXPECT_NE(test, rc, 0);")
        gate = evaluate_scenario_patch(
            baseline, revised,
            context(kind="ReturnRelation", expected="assert a non-vacuous return relation"),
        )
        self.assertFalse(gate.ok)
        self.assertTrue(any("Oracle weakened:" in item for item in gate.hard_errors))

    def test_repair_may_follow_a_boundary_count_contract(self):
        baseline = source("KUNIT_EXPECT_EQ(test, calls, 1);", "int calls = 1;\n int rc = target_export(dev);")
        revised = source("KUNIT_EXPECT_GE(test, calls, 1);", "int calls = 1;\n int rc = target_export(dev);")
        gate = evaluate_scenario_patch(
            baseline, revised,
            context(kind="BoundaryCalled", expected="boundary call_count >= 1", target="B1"),
        )
        self.assertTrue(gate.ok, gate.hard_errors)

    def test_boundary_count_cannot_be_replaced_with_target_return(self):
        code = source("KUNIT_EXPECT_EQ(test, rc, 1);")
        errors = blocking_scenario_static_errors(
            verify_scenario_contracts(
                code, context(kind="BoundaryCalled", expected="boundary call_count == 1", target="B1")
            ).errors
        )
        self.assertTrue(any("does not inspect a call counter" in item for item in errors), errors)

    def test_repair_can_correct_a_wrong_exact_value(self):
        baseline = source("KUNIT_EXPECT_EQ(test, rc, -EINVAL);")
        revised = source("KUNIT_EXPECT_EQ(test, rc, -EIO);")
        gate = evaluate_scenario_patch(baseline, revised, context())
        self.assertTrue(gate.ok, gate.hard_errors)

    def test_repair_cannot_escape_an_oracle_by_changing_scenario_binding(self):
        baseline = source("KUNIT_EXPECT_EQ(test, rc, -EIO);")
        revised = baseline.replace("RACA_SCENARIO: S1", "RACA_SCENARIO: S2")
        gate = evaluate_scenario_patch(baseline, revised, context())
        self.assertFalse(gate.ok)
        self.assertTrue(any("removed its active scenario binding" in item for item in gate.hard_errors))


if __name__ == "__main__":
    unittest.main()
