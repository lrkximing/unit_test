import json
import os
from typing import Dict, Iterable, List, Set


ROLE_HARDWARE_BOUNDARY = "hardware_boundary"
ROLE_ENVIRONMENT_PREREQUISITE = "environment_prerequisite"
ROLE_ORDINARY_HELPER = "ordinary_helper"
ROLE_UNKNOWN = "unknown"
ALLOWED_ROLES = {
    ROLE_HARDWARE_BOUNDARY,
    ROLE_ENVIRONMENT_PREREQUISITE,
    ROLE_ORDINARY_HELPER,
    ROLE_UNKNOWN,
}


def _use_llm_classifier() -> bool:
    return os.getenv("RACA_ENABLE_LLM_BOUNDARY_CLASSIFICATION", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }


def _default_structural_role(candidate) -> str:
    source_fact_kind = getattr(candidate, "source_fact_kind", "")
    if source_fact_kind == "MEMBER_CALL":
        return ROLE_HARDWARE_BOUNDARY
    if source_fact_kind == "CALL":
        return ROLE_UNKNOWN
    if source_fact_kind in {"FIELD_ACCESS", "FIELD_WRITE"}:
        return ROLE_ENVIRONMENT_PREREQUISITE
    return ROLE_UNKNOWN


def _apply_default_roles(registry) -> None:
    for candidate in registry.boundary_candidates:
        if getattr(candidate, "semantic_role", "unknown") == "unknown":
            candidate.semantic_role = _default_structural_role(candidate)
            candidate.classification_evidence_ids = list(candidate.fact_ids)
            candidate.classification_reason = "structural default from boundary candidate mechanism"


def _load_prompt(prompt_path: str) -> str:
    from model_utils import load_prompt_from_yaml

    return load_prompt_from_yaml(prompt_path, "boundary_classification_prompt")


def _invoke_classifier(prompt: str, local_or_api: str, model, tokenizer) -> str:
    if local_or_api == "local":
        import torch

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=2048,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=False,
            )
        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        return tokenizer.decode(generated_ids, skip_special_tokens=True)

    from model_utils import gpt_api

    return gpt_api(prompt)


def _extract_json_object(text: str) -> Dict:
    stripped = (text or "").strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.lower().startswith("json"):
            stripped = stripped[4:].strip()
    decoder = json.JSONDecoder()
    start = stripped.find("{")
    while start >= 0:
        try:
            obj, _ = decoder.raw_decode(stripped[start:])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            start = stripped.find("{", start + 1)
            continue
        break
    raise ValueError("Boundary classifier output does not contain a JSON object.")


def _candidate_payload(registry) -> List[Dict]:
    return [
        {
            "candidate_id": candidate.candidate_id,
            "expression": candidate.expression,
            "fact_ids": candidate.fact_ids,
            "source_function": candidate.source_function,
            "source_line": candidate.source_line,
            "source_fact_kind": candidate.source_fact_kind,
            "access_path": candidate.access_path,
            "arguments": candidate.arguments,
            "output_arguments": candidate.output_arguments,
            "result_assignee": candidate.result_assignee,
        }
        for candidate in registry.boundary_candidates
    ]


def _fact_payload(registry) -> List[Dict]:
    return [
        {
            "fact_id": fact.fact_id,
            "kind": fact.kind,
            "function": fact.function,
            "start_line": fact.start_line,
            "code": fact.code,
            "symbols": fact.symbols,
            "metadata": fact.metadata,
        }
        for fact in registry.source_facts
    ]


def _validate_and_apply_decisions(registry, decisions: Iterable[Dict]) -> List[str]:
    warnings: List[str] = []
    by_id = {candidate.candidate_id: candidate for candidate in registry.boundary_candidates}
    fact_ids: Set[str] = {fact.fact_id for fact in registry.source_facts}

    for decision in decisions:
        if not isinstance(decision, dict):
            warnings.append("Ignored non-object boundary decision.")
            continue
        candidate_id = decision.get("candidate_id", "")
        candidate = by_id.get(candidate_id)
        if candidate is None:
            warnings.append(f"Ignored decision for unknown candidate_id: {candidate_id}")
            continue
        role = decision.get("semantic_role", ROLE_UNKNOWN)
        if role not in ALLOWED_ROLES:
            warnings.append(f"Ignored invalid semantic role for {candidate_id}: {role}")
            role = ROLE_UNKNOWN
        evidence_ids = [item for item in decision.get("evidence_ids", []) or [] if item in fact_ids]
        if not evidence_ids:
            warnings.append(f"Decision for {candidate_id} has no valid evidence_ids; using candidate facts.")
            evidence_ids = list(candidate.fact_ids)
        candidate.semantic_role = role
        candidate.classification_evidence_ids = evidence_ids
        candidate.classification_reason = str(decision.get("reason", "")).strip()

    return warnings


def classify_boundary_candidates_with_optional_llm(
    registry,
    parse_result,
    function,
    local_or_api: str,
    prompt_path: str,
    model,
    tokenizer,
) -> None:
    _apply_default_roles(registry)
    registry.classification_warnings = []
    if not _use_llm_classifier():
        return
    prompt_text = _load_prompt(prompt_path)
    if not prompt_text:
        registry.classification_warnings.append("No boundary_classification_prompt found.")
        return

    from model_utils import PromptTemplate

    prompt = PromptTemplate.from_template(prompt_text).format(
        function_name=function.name,
        function_code=function.code,
        candidates=json.dumps(_candidate_payload(registry), indent=2, sort_keys=True),
        source_facts=json.dumps(_fact_payload(registry), indent=2, sort_keys=True),
    )
    try:
        output = _invoke_classifier(prompt, local_or_api, model, tokenizer)
        payload = _extract_json_object(output)
        decisions = payload.get("decisions", [])
        registry.classification_warnings.extend(_validate_and_apply_decisions(registry, decisions))
    except Exception as err:
        registry.classification_warnings.append(f"LLM boundary classification failed: {err}")
