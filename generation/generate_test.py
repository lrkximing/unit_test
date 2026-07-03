try:
    from model_utils import *
except ModuleNotFoundError as _MODEL_IMPORT_ERROR:
    PromptTemplate = None

    def load_prompt_from_yaml(*args, **kwargs):
        raise _MODEL_IMPORT_ERROR

    def gpt_api(*args, **kwargs):
        raise _MODEL_IMPORT_ERROR
import difflib
import json
import os
import re
from typing import Dict, Iterable, List, Tuple, Optional
from generation.scenario_context import format_scenario_context_for_prompt
from validation.test_inspector import inspect_test_source
from scenario.harness_feasibility import (
    allowed_generated_scenario_ids,
    registry_from_context,
)


BASE_REQUIRED_HEADERS = [
    "#include <kunit/test.h>",
    "#include <linux/kernel.h>",
    "#include <linux/types.h>",
    "#include <linux/string.h>",
    "#include <linux/module.h>",
]


def strip_markdown_code_fence(text: str) -> str:
    if not text:
        return "\n"
    stripped = text.strip()
    pattern = re.compile(r"```(?:[^\n`]*)?\n([\s\S]*?)```", re.IGNORECASE)
    match = pattern.search(stripped)
    if match:
        code = match.group(1).strip("\n")
        return (code + "\n") if code else "\n"
    lines = stripped.splitlines()
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    cleaned = "\n".join(lines).strip("\n")
    return (cleaned + "\n") if cleaned else "\n"


def _looks_like_unified_diff(text: str) -> bool:
    return bool(
        text
        and "@@" in text
        and re.search(r"(?m)^---\s+", text)
        and re.search(r"(?m)^\+\+\+\s+", text)
    )


def _rewrite_output_to_patch(model_output: str, before_code: str, test_file_name: str) -> str:
    rewritten = strip_markdown_code_fence(model_output or "")
    if _looks_like_unified_diff(rewritten):
        return rewritten
    if rewritten and not rewritten.endswith("\n"):
        rewritten += "\n"
    before_lines = (before_code or "").splitlines(keepends=True)
    after_lines = rewritten.splitlines(keepends=True)
    return "".join(
        difflib.unified_diff(
            before_lines,
            after_lines,
            fromfile=test_file_name,
            tofile=test_file_name,
        )
    )


def _write_generation_trace(trace_path: Optional[str], content: str) -> None:
    if not trace_path:
        return
    os.makedirs(os.path.dirname(os.path.abspath(trace_path)), exist_ok=True)
    with open(trace_path, "w", encoding="utf-8") as f:
        f.write(content or "")


def _format_external_symbols(function) -> str:
    if getattr(function, "external_symbols", None):
        return ", ".join(function.external_symbols)
    return ""


def _collect_driver_local_definitions(parse_result, function) -> Tuple[List[str], List[str]]:
    if not parse_result or not function:
        return [], []

    macros: List[str] = []
    structs: List[str] = []

    macro_names = set(getattr(function, "macro_refs", []) or [])
    internal_macros = function.file_internal_symbols.get("macros", []) if getattr(function, "file_internal_symbols", None) else []
    macro_names.update(internal_macros or [])
    macro_infos = []
    for name in macro_names:
        info = parse_result.macros.get(name) if getattr(parse_result, "macros", None) else None
        if info:
            macro_infos.append(info)

    struct_names = set(getattr(function, "type_refs", []) or [])
    internal_types = function.file_internal_symbols.get("types", []) if getattr(function, "file_internal_symbols", None) else []
    struct_names.update(internal_types or [])
    struct_infos = []
    structs_map = getattr(parse_result, "structs", {}) or {}
    for name in struct_names:
        info = structs_map.get(name)
        if info:
            struct_infos.append(info)
            for identifier in re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", info.code or ""):
                macro_info = parse_result.macros.get(identifier) if getattr(parse_result, "macros", None) else None
                if macro_info:
                    macro_infos.append(macro_info)
    seen_macro_names = set()
    unique_macro_infos = []
    for info in sorted(macro_infos, key=lambda m: getattr(m, "line", 0)):
        if info.name in seen_macro_names:
            continue
        seen_macro_names.add(info.name)
        unique_macro_infos.append(info)
    for info in unique_macro_infos:
        macros.append(info.code.strip())

    struct_infos.sort(key=lambda s: getattr(s, "start_line", 0))
    for info in struct_infos:
        structs.append(info.code.strip("\n"))

    return macros, structs


def _format_driver_local_definitions_for_prompt(macros: List[str], structs: List[str]) -> str:
    if not macros and not structs:
        return "None (target function does not require driver-local macros or structs)."
    lines: List[str] = []
    if macros:
        lines.append("Driver Macros:")
        lines.extend(macros)
    if structs:
        if lines:
            lines.append("")
        lines.append("Driver Structs:")
        lines.extend(structs)
    return "\n".join(lines)


def _build_driver_local_definitions_block(macros: List[str], structs: List[str]) -> str:
    sections: List[str] = []
    if macros:
        sections.append("/* Driver Macros (copied from original driver) */")
        sections.extend(macros)
    if structs:
        if sections:
            sections.append("")
        sections.append("/* Driver Structs (copied from original driver) */")
        sections.extend(structs)
    return "\n".join(sections).strip()


def _inject_driver_local_definitions(test_code: str, macros: List[str], structs: List[str]) -> str:
    block = _build_driver_local_definitions_block(macros, structs)
    if not block:
        return test_code

    lines = test_code.splitlines()
    last_include_idx = -1
    for idx, line in enumerate(lines):
        if line.strip().startswith("#include"):
            last_include_idx = idx
    insert_idx = last_include_idx + 1 if last_include_idx >= 0 else 0
    while insert_idx < len(lines) and not lines[insert_idx].strip():
        insert_idx += 1

    insertion = [""]
    insertion.append("/* ===== Driver Local Definitions BEGIN ===== */")
    insertion.extend(block.splitlines())
    insertion.append("/* ===== Driver Local Definitions END ===== */")
    insertion.append("")

    new_lines = lines[:insert_idx] + insertion + lines[insert_idx:]
    return "\n".join(new_lines)


def _prune_driver_local_definitions(test_code: str, macros: List[str], structs: List[str]) -> str:
    text = test_code
    for macro in macros:
        macro_line = macro.strip()
        if not macro_line:
            continue
        pattern = re.compile(r"^\s*" + re.escape(macro_line) + r"\s*$\n?", re.MULTILINE)
        text = pattern.sub("", text)
    for struct in structs:
        struct_block = struct.strip()
        if not struct_block:
            continue
        text = text.replace(struct_block + "\n", "")
        text = text.replace(struct_block, "")
    return text


def _format_frozen_tests_for_prompt(frozen_tests: Optional[List[str]]) -> str:
    if not frozen_tests:
        return "None (all test cases failed; no frozen tests)."
    unique = sorted({name.strip() for name in frozen_tests if name.strip()})
    if not unique:
        return "None (no frozen tests)."
    return "\n".join(unique)


def _invoke_generation_model(prompt: str, local_or_api: str, model, tokenizer) -> str:
    if local_or_api == "local":
        import torch

        test_case_inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(model.device)
        with torch.no_grad():
            test_case_outputs = model.generate(
                **test_case_inputs,
                max_new_tokens=4096,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=False
            )
        generated_ids = test_case_outputs[0][test_case_inputs["input_ids"].shape[1]:]
        return tokenizer.decode(generated_ids, skip_special_tokens=True)
    return strip_markdown_code_fence(gpt_api(prompt))


def _compute_required_headers(export_interfaces, driver_includes: Iterable[str]) -> List[str]:
    headers: List[str] = []
    seen = set()

    def add(header: str) -> None:
        if not header or header in seen:
            return
        headers.append(header)
        seen.add(header)

    for base in BASE_REQUIRED_HEADERS:
        add(base)

    for inc in driver_includes or []:
        inc = inc.strip()
        if not inc:
            continue
        if not inc.startswith("#include"):
            inc = f"#include {inc}"
        add(inc)
    return headers


def _ensure_required_includes(test_code: str, required_headers: List[str]) -> str:
    if not required_headers:
        return test_code
    lines = test_code.splitlines()
    existing = set(line.strip() for line in lines if line.strip().startswith("#include"))
    missing = [hdr for hdr in required_headers if hdr not in existing]
    if not missing:
        return test_code

    insert_idx = 0
    if lines and lines[0].startswith("// SPDX"):
        insert_idx = 1
        while insert_idx < len(lines) and not lines[insert_idx].strip():
            insert_idx += 1

    new_lines = lines[:insert_idx] + missing + [""] + lines[insert_idx:]
    return "\n".join(new_lines)


def _insert_after_includes(test_code: str, block: str) -> str:
    if not block.strip():
        return test_code
    lines = (test_code or "").splitlines()
    last_include_idx = -1
    for idx, line in enumerate(lines):
        if line.strip().startswith("#include"):
            last_include_idx = idx
    insert_idx = last_include_idx + 1 if last_include_idx >= 0 else 0
    insertion = ["", block.strip(), ""]
    return "\n".join(lines[:insert_idx] + insertion + lines[insert_idx:])


def _normalize_unconstrained_complete_file(
    model_output: str,
    *,
    required_headers: List[str],
    export_interface_block: str,
    macros: List[str],
    structs: List[str],
) -> str:
    code = strip_markdown_code_fence(model_output)
    if not code.lstrip().startswith("/* SPDX-License-Identifier:"):
        code = "/* SPDX-License-Identifier: GPL-2.0 */\n" + code.lstrip()
    code = _ensure_required_includes(code, required_headers)
    forward_decls = _forward_struct_declarations_for_exports(export_interface_block)
    support_parts = []
    if forward_decls:
        support_parts.append("/* TEST EXPORT TYPE DECLARATIONS */\n" + forward_decls)
    if export_interface_block.strip() and export_interface_block.strip() not in code:
        support_parts.append("/* TEST EXPORT INTERFACES */\n" + export_interface_block.strip())
    if support_parts:
        code = _insert_after_includes(code, "\n\n".join(support_parts))
    code = _prune_driver_local_definitions(code, macros, structs)
    code = _inject_driver_local_definitions(code, macros, structs)
    if "MODULE_LICENSE" not in code:
        code = code.rstrip() + '\n\nMODULE_LICENSE("GPL");\n'
    return code.rstrip() + "\n"


def _format_export_interfaces(export_interfaces) -> str:
    if not export_interfaces:
        return "(no test exports provided)"
    lines = []
    for interface in export_interfaces:
        prototype = getattr(interface, "prototype", str(interface))
        source_symbol = getattr(interface, "source_symbol", "")
        source_kind = getattr(interface, "source_kind", "")
        description = getattr(interface, "description", "")
        note_parts = []
        if source_kind or source_symbol:
            note_parts.append(f"source={source_kind}:{source_symbol}")
        if description:
            note_parts.append(description)
        note = ""
        if note_parts:
            note = "  // " + "; ".join(note_parts)
        lines.append(f"{prototype}{note}")
    return "\n".join(lines)


def _forward_struct_declarations_for_exports(export_interface_block: str) -> str:
    declarations: List[str] = []
    seen = set()
    for line in (export_interface_block or "").splitlines():
        prototype = line.split("//", 1)[0]
        for struct_name in re.findall(r"\bstruct\s+([A-Za-z_][A-Za-z0-9_]*)\b", prototype):
            if struct_name in seen:
                continue
            seen.add(struct_name)
            declarations.append(f"struct {struct_name};")
    return "\n".join(declarations)


def _sanitize_c_identifier(value: str, suffix: str = "") -> str:
    text = re.sub(r"[^A-Za-z0-9_]", "_", value or "raca")
    if not text or text[0].isdigit():
        text = f"raca_{text}"
    return text + suffix


def _find_matching_brace(code: str, open_idx: int) -> Optional[int]:
    depth = 0
    in_string = False
    in_char = False
    escaped = False
    i = open_idx
    while i < len(code):
        ch = code[i]
        if escaped:
            escaped = False
            i += 1
            continue
        if ch == "\\":
            escaped = True
            i += 1
            continue
        if ch == '"' and not in_char:
            in_string = not in_string
        elif ch == "'" and not in_string:
            in_char = not in_char
        elif not in_string and not in_char:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return i
        i += 1
    return None


def _leading_raca_comments(code: str, start_idx: int) -> str:
    lines = code[:start_idx].splitlines()
    collected: List[str] = []
    i = len(lines) - 1
    while i >= 0 and not lines[i].strip():
        i -= 1
    while i >= 0:
        stripped = lines[i].strip()
        if not stripped:
            break
        if (
            stripped.startswith("//")
            or stripped.startswith("/*")
            or stripped.startswith("*")
            or stripped.endswith("*/")
        ):
            collected.append(lines[i])
            i -= 1
            continue
        break
    text = "\n".join(reversed(collected))
    return text if "RACA_" in text else ""


def _extract_function_blocks(code: str) -> List[Tuple[str, str]]:
    blocks: List[Tuple[str, str]] = []
    pattern = re.compile(
        r"(?m)^[\t ]*(?:static\s+)?(?:inline\s+)?(?:[A-Za-z_][A-Za-z0-9_]*\s+)+\*?\s*([A-Za-z_][A-Za-z0-9_]*)\s*\([^;{}]*\)\s*\{"
    )
    for match in pattern.finditer(code or ""):
        open_idx = code.find("{", match.start())
        if open_idx < 0:
            continue
        end_idx = _find_matching_brace(code, open_idx)
        if end_idx is None:
            continue
        block = code[match.start() : end_idx + 1].strip()
        leading = _leading_raca_comments(code, match.start())
        if leading:
            block = leading.rstrip() + "\n" + block
        blocks.append((match.group(1), block))
    return blocks


def _ensure_static_function_block(block: str) -> str:
    lines = block.splitlines()
    for idx, line in enumerate(lines):
        stripped = line.lstrip()
        if not stripped or stripped.startswith("/*") or stripped.startswith("*") or stripped.startswith("//"):
            continue
        if not stripped.startswith("static "):
            indent = line[: len(line) - len(stripped)]
            lines[idx] = indent + "static " + stripped
        break
    return "\n".join(lines)


def _strip_leading_mock_markers(block: str) -> str:
    lines = block.splitlines()
    while lines and "RACA_MOCK:" in lines[0]:
        lines = lines[1:]
    return "\n".join(lines)


def _ensure_scenario_marker_in_block(block: str, scenario_ids: Iterable[str]) -> str:
    if "RACA_SCENARIO:" in (block or ""):
        return block
    ids = [item for item in scenario_ids or [] if item]
    if len(ids) != 1:
        return block
    open_idx = (block or "").find("{")
    if open_idx < 0:
        return block
    return block[: open_idx + 1] + f"\n\t/* RACA_SCENARIO: {ids[0]} */" + block[open_idx + 1 :]


def _function_references(name: str, text: str) -> bool:
    return bool(re.search(r"\b" + re.escape(name) + r"\s*\(", text or ""))


def _select_referenced_helpers(helper_blocks: List[Tuple[str, str]], test_blocks: List[str]) -> List[str]:
    kept: List[Tuple[str, str]] = []
    selected_names = set()
    search_text = "\n\n".join(test_blocks)
    changed = True
    while changed:
        changed = False
        for name, block in helper_blocks:
            if name in selected_names:
                continue
            if _function_references(name, search_text):
                selected_names.add(name)
                kept.append((name, block))
                search_text += "\n" + block
                changed = True
    return [_ensure_static_function_block(block) for _, block in kept]


def _looks_like_generated_test_block(name: str, block: str) -> bool:
    if "RACA_SCENARIO:" in (block or ""):
        return True
    return bool(name.startswith("test_") and re.search(r"\bstruct\s+kunit\s*\*\s*test\b", block or ""))


def _looks_like_llm_fake_helper(name: str, block: str) -> bool:
    if "RACA_MOCK:" in (block or ""):
        return True
    return bool(re.search(r"(?:^|_)(?:fake|fakes|mock|mocks|local_fake)(?:_|$)|\braca_fake_", name or ""))


def _function_prototype_from_block(block: str) -> str:
    text = (block or "").strip()
    open_idx = text.find("{")
    if open_idx < 0:
        return ""
    header = text[:open_idx].strip()
    lines = [
        line
        for line in header.splitlines()
        if line.strip()
        and not line.strip().startswith("/*")
        and not line.strip().startswith("*")
        and not line.strip().startswith("//")
    ]
    prototype = "\n".join(lines).strip()
    if not prototype or "(" not in prototype or ")" not in prototype:
        return ""
    return prototype + ";"


def _function_prototypes_from_blocks(blocks: List[str]) -> List[str]:
    prototypes: List[str] = []
    seen = set()
    for block in blocks:
        prototype = _function_prototype_from_block(block)
        if not prototype or prototype in seen:
            continue
        seen.add(prototype)
        prototypes.append(prototype)
    return prototypes


def _rebuild_initial_test_file(
    model_output: str,
    *,
    function_name: str,
    required_headers: List[str],
    export_interface_block: str,
    macros: List[str],
    structs: List[str],
    scenario_context=None,
) -> str:
    raw = strip_markdown_code_fence(model_output)
    info = inspect_test_source(raw)
    registry = registry_from_context(scenario_context or {})
    allowed_scenarios = allowed_generated_scenario_ids(registry) if registry else None
    kept_tests = []
    for test in info.test_functions:
        scenario_ids = set(test.scenario_ids or [])
        if len(scenario_ids) != 1:
            continue
        if allowed_scenarios is not None and not scenario_ids <= allowed_scenarios:
            continue
        kept_tests.append(test)
    kept_test_name_set = {test.name for test in kept_tests}
    blocks = _extract_function_blocks(raw)
    block_by_name = {name: block for name, block in blocks}
    extracted_test_blocks: List[Tuple[str, str]] = []
    for test in kept_tests:
        block = block_by_name.get(test.name)
        if block is None:
            block = _ensure_scenario_marker_in_block(test.full_text, test.scenario_ids)
        extracted_test_blocks.append((test.name, block))
    test_names = [name for name, _ in extracted_test_blocks]
    test_name_set = set(test_names)
    test_blocks = [
        _strip_leading_mock_markers(_ensure_static_function_block(block))
        for name, block in extracted_test_blocks
    ]
    helper_blocks = [
        (name, block)
        for name, block in blocks
        if name not in test_name_set
        and not name.startswith("KUNIT_CASE")
        and not _looks_like_generated_test_block(name, block)
        and not _looks_like_llm_fake_helper(name, block)
    ]
    selected_helpers = _select_referenced_helpers(helper_blocks, test_blocks)
    helper_prototypes = _function_prototypes_from_blocks(selected_helpers)

    suite_id = _sanitize_c_identifier(function_name.lower(), "_raca")
    suite_name = _sanitize_c_identifier(function_name)
    local_defs = _build_driver_local_definitions_block(macros, structs)
    export_forward_decls = _forward_struct_declarations_for_exports(export_interface_block)

    sections: List[str] = [
        "/* SPDX-License-Identifier: GPL-2.0 */",
        "/* Auto-generated KUnit test scaffold. Fixed sections are assembled by RACA. */",
        "",
        "\n".join(required_headers),
        "",
    ]
    if export_forward_decls:
        sections.extend(
            [
                "/* TEST EXPORT TYPE DECLARATIONS - DO NOT MODIFY */",
                export_forward_decls,
                "",
            ]
        )
    sections.extend(
        [
        "/* TEST EXPORT INTERFACES - DO NOT MODIFY */",
        export_interface_block,
        ]
    )
    if local_defs:
        sections.extend(
            [
                "",
                "/* ===== Driver Local Definitions BEGIN ===== */",
                local_defs,
                "/* ===== Driver Local Definitions END ===== */",
            ]
        )
    if selected_helpers:
        helper_parts = []
        if helper_prototypes:
            helper_parts.extend(helper_prototypes)
            helper_parts.append("")
        helper_parts.extend(selected_helpers)
        sections.extend(["", "/* ===== Flexible Helpers BEGIN ===== */", "\n\n".join(helper_parts), "/* ===== Flexible Helpers END ===== */"])
    if test_blocks:
        sections.extend(["", "/* ===== Scenario Tests BEGIN ===== */", "\n\n".join(test_blocks), "/* ===== Scenario Tests END ===== */"])
    else:
        sections.extend(["", "/* No KUnit test functions were generated. */"])
    cases = [f"\tKUNIT_CASE({name})," for name in test_names]
    suite_lines = [
        f"static struct kunit_suite {suite_id}_suite = {{",
        f"\t.name = \"{suite_name}\",",
        f"\t.test_cases = {suite_id}_cases,",
    ]
    suite_lines.append("};")
    sections.extend(
        [
            "",
            f"static struct kunit_case {suite_id}_cases[] = {{",
            "\n".join(cases),
            "\t{}",
            "};",
            "",
            "\n".join(suite_lines),
            "",
            f"kunit_test_suite({suite_id}_suite);",
            "",
            'MODULE_LICENSE("GPL");',
            "",
        ]
    )
    return "\n".join(part for part in sections if part is not None).rstrip() + "\n"


def generate_test_case(
    parse_result,
    function,
    local_or_api: str,
    prompt_path: str,
    model,
    tokenizer,
    export_interfaces,
    scenario_context=None,
) -> str:
    function_code = function.code
    export_interface_block = _format_export_interfaces(export_interfaces)
    driver_includes = getattr(parse_result, "includes", []) if parse_result else []
    required_headers = _compute_required_headers(export_interfaces, driver_includes)
    required_headers_block = "\n".join(required_headers)
    macro_defs, struct_defs = _collect_driver_local_definitions(parse_result, function)
    # print(f"macro_defs: {macro_defs}")
    # print(f"struct_defs: {struct_defs}")
    driver_local_defs_prompt = _format_driver_local_definitions_for_prompt(macro_defs, struct_defs)
    # print(f"driver_local_defs_prompt: {driver_local_defs_prompt}")

    prompt = PromptTemplate.from_template(
        load_prompt_from_yaml(prompt_path, "test_case_generate_prompt")
    )
    test_case_prompt = prompt.format(
        function_code=function_code,
        stub=export_interface_block,
        required_headers=required_headers_block,
        driver_local_definitions=driver_local_defs_prompt,
        scenario_context=format_scenario_context_for_prompt(scenario_context),
    )
    model_output = _invoke_generation_model(test_case_prompt, local_or_api, model, tokenizer).strip()
    return _rebuild_initial_test_file(
        model_output,
        function_name=function.name,
        required_headers=required_headers,
        export_interface_block=export_interface_block,
        macros=macro_defs,
        structs=struct_defs,
        scenario_context=scenario_context,
    )


def refine_test_case(function,
                     current_test_case: str,
                     error_log: str,
                     failure_stage: str,
                     local_or_api: str,
                     prompt_path: str,
                     model,
                     tokenizer,
                     frozen_tests: Optional[List[str]] = None,
                     scenario_context=None,
                     test_file_name: str = "test_case.c",
                     trace_path: Optional[str] = None) -> str:

    prompt = PromptTemplate.from_template(
        load_prompt_from_yaml(prompt_path, "test_case_fix_prompt")
    )
    frozen_tests_block = _format_frozen_tests_for_prompt(frozen_tests)
    test_case_prompt = prompt.format(
        function_code=function.code,
        current_test_case=current_test_case,
        error_log=error_log,
        failure_stage=failure_stage,
        frozen_tests=frozen_tests_block,
        scenario_context=format_scenario_context_for_prompt(scenario_context),
        test_file_name=test_file_name,
    )
    rewritten = _invoke_generation_model(test_case_prompt, local_or_api, model, tokenizer).strip()
    _write_generation_trace(trace_path, rewritten)
    return rewritten


def fix_failed_tests(function,
                     current_test_case: str,
                     failing_tests: List[str],
                     failure_reasons: str,
                     local_or_api: str,
                     prompt_path: str,
                     model,
                     tokenizer,
                     frozen_tests: Optional[List[str]] = None,
                     scenario_context=None,
                     test_file_name: str = "test_case.c",
                     trace_path: Optional[str] = None) -> str:
    prompt = PromptTemplate.from_template(
        load_prompt_from_yaml(prompt_path, "test_case_fix_failed_prompt")
    )
    frozen_tests_block = _format_frozen_tests_for_prompt(frozen_tests)
    test_case_prompt = prompt.format(
        function_code=function.code,
        current_test_case=current_test_case,
        frozen_tests=frozen_tests_block,
        # failing_tests="\n".join(failing_tests) if failing_tests else "none",
        failing_tests="\n".join(failing_tests) if failing_tests else "none",
        failing_reasons=failure_reasons or "Not available",
        scenario_context=format_scenario_context_for_prompt(scenario_context),
        test_file_name=test_file_name,
    )
    rewritten = _invoke_generation_model(test_case_prompt, local_or_api, model, tokenizer).strip()
    _write_generation_trace(trace_path, rewritten)
    return rewritten


def extend_coverage_tests(function,
                          current_test_case: str,
                          missed_line_details: Optional[List[str]],
                          missed_branch_details: Optional[List[str]],
                          local_or_api: str,
                          prompt_path: str,
                          model,
                          tokenizer,
                          scenario_context=None,
                          test_file_name: str = "test_case.c",
                          trace_path: Optional[str] = None) -> str:
    
    prompt = PromptTemplate.from_template(
        load_prompt_from_yaml(prompt_path, "test_case_extend_coverage_prompt")
    )
    line_detail_block = "\n".join(missed_line_details or []) or "None (no specific line details)."
    branch_detail_block = "\n".join(missed_branch_details or []) or "None (no specific branch details)."
    test_case_prompt = prompt.format(
        function_code=function.code,
        current_test_case=current_test_case,
        missed_line_details=line_detail_block,
        missed_branch_details=branch_detail_block,
        scenario_context=format_scenario_context_for_prompt(scenario_context),
        test_file_name=test_file_name,
    )
    rewritten = _invoke_generation_model(test_case_prompt, local_or_api, model, tokenizer).strip()
    _write_generation_trace(trace_path, rewritten)
    return rewritten


def repair_rejected_patch(function,
                          current_test_case: str,
                          rejected_candidate_code: str,
                          rejected_patch: str,
                          rejection_reason: str,
                          failure_stage: str,
                          original_task_context: str,
                          local_or_api: str,
                          prompt_path: str,
                          model,
                          tokenizer,
                          scenario_context=None,
                          test_file_name: str = "test_case.c",
                          trace_path: Optional[str] = None) -> str:
    prompt = PromptTemplate.from_template(
        load_prompt_from_yaml(prompt_path, "test_case_patch_retry_prompt")
    )
    test_case_prompt = prompt.format(
        function_code=function.code,
        current_test_case=current_test_case,
        rejected_candidate_code=rejected_candidate_code or "",
        rejected_patch=rejected_patch or "",
        rejection_reason=rejection_reason or "Patch was rejected without a structured reason.",
        failure_stage=failure_stage,
        original_task_context=original_task_context or "Not available",
        scenario_context=format_scenario_context_for_prompt(scenario_context),
        test_file_name=test_file_name,
    )
    rewritten = _invoke_generation_model(test_case_prompt, local_or_api, model, tokenizer).strip()
    _write_generation_trace(trace_path, rewritten)
    return rewritten
