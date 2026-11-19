"""Persist each local variant's condition, environment and checks together.

Coverage counters locate new work; they are not themselves a variant contract.
Old contexts remain readable without rewriting their saved contracts.
"""
from copy import deepcopy
import json

PAYLOAD_FIELDS = (
    "target_function", "export_function", "source_anchors", "trigger_conditions",
    "hardware_environment_constraints", "observations", "scenario_checks",
    "runtime_witnesses", "dependent_boundaries", "relation_edges", "derivation",
)


def behavior_fingerprint(contract):
    """Ignore generated labels, never conflate distinct conditions or oracles."""
    labels = {"condition_id", "check_id", "observation_id", "constraint_id",
              "effect_id", "witness_id"}
    def semantic(value):
        if isinstance(value, dict):
            return {k: semantic(v) for k, v in value.items() if k not in labels}
        if isinstance(value, list):
            # Order can encode a callback/effect sequence, not just a set.
            return [semantic(v) for v in value]
        return value
    fields = ("source_anchors", "trigger_conditions", "hardware_environment_constraints",
              "scenario_checks", "observations", "runtime_witnesses", "dependent_boundaries")
    return json.dumps(semantic({k: contract.get(k, []) for k in fields}), sort_keys=True)


def make_variant(contract, variant_id="initial"):
    return {"scenario_id": contract["scenario_id"], "variant_id": variant_id,
            **{k: deepcopy(contract[k]) for k in PAYLOAD_FIELDS if k in contract}}


def active_contracts(context):
    registry = context.get("scenario_registry") or context
    active = registry.get("active_scenario_ids")
    return [c for c in registry.get("scenario_contracts", [])
            if active is None or c["scenario_id"] in active]


def planned_variants(context):
    contracts = {c["scenario_id"]: c for c in active_contracts(context)}
    if "scenario_variants" in context:
        return [v for v in context["scenario_variants"] if v.get("scenario_id") in contracts]
    # Compatibility projection for old, immutable checkpoints.
    result = [make_variant(c) for c in contracts.values()]
    for target in context.get("coverage_targets", []):
        if target.get("scenario_id") in contracts:
            variant = make_variant(contracts[target["scenario_id"]], target["variant_id"])
            if variant not in result:
                result.append(variant)
    return result


def initialize_variants(context):
    if "scenario_variants" not in context:
        context["scenario_variants"] = planned_variants(context)
    return context


def append_variant(context, variant):
    initialize_variants(context)
    identity = (variant["scenario_id"], variant["variant_id"])
    for old in context["scenario_variants"]:
        if (old["scenario_id"], old["variant_id"]) == identity:
            if ({k: v for k, v in old.items() if k != "coverage_evidence"}
                    != {k: v for k, v in variant.items() if k != "coverage_evidence"}):
                raise ValueError("Existing scenario variant contract changed: " + repr(identity))
            return
    context["scenario_variants"].append(deepcopy(variant))


def contract_for_variant(context, scenario_id, variant_id):
    for variant in planned_variants(context):
        if (variant["scenario_id"], variant["variant_id"]) == (scenario_id, variant_id or "initial"):
            return variant
    # Legacy contexts without allocation used implicit user-named variants.
    if "scenario_variants" not in context and "variant_bindings" not in context:
        return next((c for c in active_contracts(context) if c["scenario_id"] == scenario_id), None)
    return None


def variant_plan_errors(context):
    if "scenario_variants" not in context:
        return []
    contracts = {c["scenario_id"]: c for c in active_contracts(context)}
    errors, seen = [], set()
    for variant in context["scenario_variants"]:
        identity = (variant.get("scenario_id"), variant.get("variant_id"))
        parent = contracts.get(identity[0])
        if not parent or not identity[1] or identity in seen:
            errors.append("Invalid or duplicate scenario variant: " + repr(identity))
        elif (variant.get("export_function") != parent.get("export_function")
              or not variant.get("scenario_checks")
              or variant.get("source_anchors") != parent.get("source_anchors")):
            errors.append("Incomplete scenario variant contract: " + repr(identity))
        witnesses = list(variant.get("runtime_witnesses", []))
        for constraint in variant.get("hardware_environment_constraints", []):
            if not constraint.get("conditional"):
                witnesses.extend(constraint.get("runtime_witnesses", []))
        reached = {w.get("target") for w in witnesses if w.get("kind") == "BOUNDARY_INTERACTION"}
        absent = {w.get("target") for w in witnesses if w.get("kind") == "BOUNDARY_NOT_REACHED"}
        if (reached & absent) - {None, ""}:
            errors.append("Contradictory boundary reachability in variant: " + repr(identity))
        seen.add(identity)
    for sid in contracts:
        if not any(identity[0] == sid for identity in seen):
            errors.append("Active scenario has no planned condition variant: " + sid)
    return errors
