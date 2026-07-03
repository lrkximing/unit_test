from analysis.driver_parser import *
from generation.generate_stub import *
from generation.generate_test import *
from generation.scenario_context import generate_scenario_context, save_scenario_context
from run.config_change import *
from run.build import *
from run.gcov import ensure_gcov_profile, collect_gcov_results, summarize_function_coverage
from run.test_isolation import isolate_current_driver_test_case
from scenario.coverage_expander import expand_scenario_context_for_coverage
from scenario.harness_feasibility import active_scenario_ids, registry_from_context
from validation.scenario_context_check import validate_scenario_context_structure, validate_test_against_scenario_context
from validation.extern_normalizer import strip_flexible_extern_declarations
from verification.scenario_patch_gate import evaluate_scenario_patch
from verification.runtime_witness_parser import evaluate_scenario_runtime_status
from verification.kunit_result_parser import extract_failure_reason_from_log, extract_failure_reason_from_text, filter_kunit_cases_to_tests, parse_kunit_results_file, parse_kunit_results_text, suite_log_block, test_metrics_from_kunit
from verification.scenario_progress import scenario_status_complete, scenario_status_repair_log, scenario_test_bindings, stable_tests_from_scenario_status
from verification.scenario_static_verifier import blocking_scenario_static_errors, mock_bindings_from_test, nonblocking_scenario_static_findings
from verification.assertion_quality import nontrivial_assertion_tests
from validation.test_inspector import inspect_test_source
import argparse
import difflib
import os
import json
import re
import subprocess
from typing import Optional, List, Dict, Set, Tuple
import shutil
import time
import traceback
from datetime import datetime, timezone
MAX_FIX_ATTEMPTS = 5
TARGET_COVERAGE_PERCENT = 90.0
PATCH_RETRY_LIMIT = 1

def _parse_args():
    parser = argparse.ArgumentParser(description='Run RACA Linux driver KUnit generation experiments.')
    parser.add_argument('--result-root', default=None, help='Output directory. Defaults to unit_test_v2/output_all or RACA_RESULT_ROOT.')
    parser.add_argument('--limit-drivers', type=int, default=0, help='Pilot mode: process at most this many drivers after filtering. 0 means no limit.')
    parser.add_argument('--limit-functions', type=int, default=0, help='Pilot mode: process at most this many target functions per driver. 0 means no limit.')
    parser.add_argument('--driver', action='append', default=[], help='Only process this driver relative path. Can be passed multiple times.')
    parser.add_argument('--function', action='append', default=[], help='Only process this target function name. Can be passed multiple times.')
    parser.add_argument('--local-or-api', choices=('api', 'local'), default=os.getenv('RACA_LLM_MODE', 'api'), help='LLM backend to use.')
    parser.add_argument('--model-path', default=None, help='Local model path when --local-or-api local is used.')
    parser.add_argument('--max-fix-attempts', type=int, default=MAX_FIX_ATTEMPTS, help='Maximum compile/run/repair iterations per function.')
    parser.add_argument('--target-coverage', type=float, default=TARGET_COVERAGE_PERCENT, help='Line and branch coverage target percentage.')
    parser.add_argument('--dry-run-targets', action='store_true', help='Print selected drivers/functions and exit without modifying kernel files.')
    parser.add_argument('--cleanup-only', action='store_true', help='Remove RACA-generated kernel-tree artifacts for the selected drivers/functions and exit.')
    parser.add_argument('--linux-kernel-path', default=os.getenv('RACA_LINUX_KERNEL_PATH'), help='Path to the Linux kernel source tree. Can also be set with RACA_LINUX_KERNEL_PATH.')
    parser.add_argument('--buildroot-dir', default=os.getenv('RACA_BUILDROOT_DIR'), help='Path to the Buildroot tree used to rebuild and run the kernel. Can also be set with RACA_BUILDROOT_DIR.')
    return parser.parse_args()

def _read_text(path: str) -> str:
    with open(path, 'r') as f:
        return f.read()

def _write_text(path: str, content: str) -> None:
    with open(path, 'w') as f:
        f.write(content)

def _write_json(path: str, data: object) -> None:
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def _timing_payload(started_at: Optional[str], start_monotonic: Optional[float]) -> Dict[str, object]:
    elapsed = None
    if start_monotonic is not None:
        elapsed = time.monotonic() - start_monotonic
    return {'started_at': started_at, 'ended_at': _utc_now_iso(), 'elapsed_seconds': elapsed}

def _collect_process_error(exc: subprocess.CalledProcessError) -> str:
    parts = []
    for attr in ('stdout', 'stderr', 'output'):
        value = getattr(exc, attr, None)
        if value:
            parts.append(value)
    text = '\n'.join(parts).strip()
    return text or str(exc)

def _condense_log(log: str, limit: int=1000) -> str:
    if not log:
        return 'No log captured.'
    log = log.strip()
    if len(log) <= limit:
        return log
    return log[-limit:]

def _runtime_environment_repair_hint(log_text: str) -> str:
    """Return generic repair guidance for runtime environment crashes."""
    if not log_text:
        return ''
    lower = log_text.lower()
    hints: List[str] = []
    if 'null pointer dereference' in lower or 'null pointer' in lower:
        hints.append('- A NULL dereference means the fake environment is incomplete. Do not return NULL or an arbitrary pointer from a fake when the production path later dereferences it. Allocate the real containing driver object type, wire it into the object path used by the target call, and initialize every field that the downstream code dereferences.')
    if 'deadbeef' in lower:
        hints.append('- A poison/sentinel pointer such as 0xdeadbeef is not a valid fake return object. If the production path dereferences the fake result or locks a field inside it, allocate the correct driver-private object type and initialize its locks and nested fields instead of returning a cast integer address.')
    if re.search('\\bmutex_(?:lock|unlock|trylock)\\b', log_text):
        hints.append('- If the failing path reaches mutex_lock/mutex_unlock, initialize the containing object and call mutex_init(&object->lock) before invoking the test_export wrapper. Do not fake mutex helpers or bypass the target path.')
    if hints:
        hints.append('- For any direct external CALL boundary on the failing path, use the scenario_context boundary_controls metadata. Install a type-compatible test-local fake through the exported hook setter before calling the target wrapper, and clear it afterwards. A same-name fake function in the test file does not intercept a production direct call.')
        hints.append('- If one direct boundary on the executed path is faked, also install safe default fakes for the other direct boundaries on the same target/helper path. Otherwise the tested path may pass the first fake and then fall through into the original external API later in the function or in a helper.')
    if 'uninitialized' in lower or 'may be used uninitialized' in lower or '未经初始化' in log_text:
        hints.append('- If a fake return value or setup field is assigned from a local object, create and initialize that object before assigning it to fake state or installing the hook.')
    if 'discarded-qualifiers' in lower or ('discards' in lower and 'const' in lower):
        hints.append('- If the compiler reports that a call discards a const qualifier, do not write through that field. For `const char *` fields such as names, assign a string literal or keep a separate writable buffer pointer and assign the field to that buffer after formatting.')
    if 'incompatible-pointer-types' in lower or '不兼容的指针类型' in log_text:
        hints.append("- For hook setter pointer-type errors, rewrite the fake function signature to exactly match the setter's `typeof(original_api)` type shown in the build log, including const qualifiers and integer typedef names.")
    if not hints:
        return ''
    return '\n'.join(hints)

def _augment_repair_log(log_text: str) -> str:
    hint = _runtime_environment_repair_hint(log_text)
    if not hint:
        return log_text or 'No log captured.'
    return '\n\n'.join([log_text or 'No log captured.', 'RACA runtime environment repair guidance:\n' + hint])

def _save_error_log(base_dir: str, func_name: str, stage: str, iteration: int, log_text: str) -> str:
    os.makedirs(base_dir, exist_ok=True)
    log_text = log_text if log_text else 'No log captured.'
    log_path = os.path.join(base_dir, f'{func_name}_{stage}_iteration{iteration}.log')
    with open(log_path, 'w') as f:
        f.write(log_text)
    return log_path

def _write_function_failure_summary(function_output_dir: str, function_name: str, status: str, stage: str, iteration: int, reason: str, history: List[Dict[str, object]], **details) -> Dict[str, object]:
    os.makedirs(function_output_dir, exist_ok=True)
    summary_path = os.path.join(function_output_dir, 'summary.json')
    summary_data: Dict[str, object] = {'function': function_name, 'status': status, 'stage': stage, 'iteration': iteration, 'reason': reason, 'kunit': empty_kunit_summary(), 'coverage': empty_coverage_summary(), 'failing_tests': [], 'failure_reasons': {}, 'history': history}
    summary_data.update(details)
    _write_json(summary_path, summary_data)
    print(f'[RESULT] {stage} 阶段失败，跳过该函数。详情见 {summary_path}')
    return summary_data
SUITE_NAME_PATTERN = re.compile('struct\\s+kunit_suite\\s+[A-Za-z_][A-Za-z0-9_]*\\s*=\\s*\\{(?P<body>.*?)\\};', re.DOTALL)

def _is_missing_scenario_binding_error(error: str) -> bool:
    return not blocking_scenario_static_errors([error or ''])

def _removable_generated_tests(test_code: str, scenario_context: Dict) -> Set[str]:
    registry = registry_from_context(scenario_context or {})
    active = active_scenario_ids(registry)
    if active is None:
        active = {contract.get('scenario_id', '') for contract in registry.get('scenario_contracts', []) or [] if isinstance(contract, dict) and contract.get('scenario_id')}
    info = inspect_test_source(test_code or '')
    tests_by_name = {test.name: test for test in info.test_functions}
    removable: Set[str] = set()
    for registered_name in info.registered_tests:
        test_function = tests_by_name.get(registered_name)
        if test_function is None:
            continue
        scenario_ids = set(test_function.scenario_ids or [])
        if not scenario_ids:
            removable.add(registered_name)
            continue
        if not scenario_ids <= active:
            removable.add(registered_name)
            continue
    return removable

def _read_optional_text(path: str) -> str:
    if not path or not os.path.exists(path):
        return ''
    try:
        return _read_text(path)
    except OSError:
        return ''

def _remove_if_exists(path: str) -> None:
    if not path:
        return
    try:
        os.remove(path)
    except FileNotFoundError:
        return
    except OSError:
        return

def _extract_suite_name(test_code: str, default: str) -> str:
    for match in SUITE_NAME_PATTERN.finditer(test_code or ''):
        name_match = re.search('\\.name\\s*=\\s*\\"([^\\"]+)\\"', match.group('body'))
        if name_match:
            return name_match.group(1)
    return default

def empty_kunit_summary() -> Dict[str, object]:
    return {'tests': [], 'passed': [], 'failed': [], 'tests_total': 0, 'tests_passed_count': 0, 'tests_failed_count': 0, 'overall_passed': False, 'pass_rate': 0.0, 'per_test_pass_rate': []}

def empty_coverage_summary() -> Dict[str, object]:
    return {'blocks_percent': 0.0, 'line_percent': 0.0, 'branch_percent': 0.0, 'line_total': 0, 'line_hit': 0, 'branch_total': 0, 'branch_hit': 0, 'covered_lines': [], 'missed_lines': [], 'covered_branches': [], 'missed_branches': []}

def _current_function_line_range(parser: 'CDriverParser', driver_file_path: str, function_name: str, fallback_start_line: int, fallback_end_line: int) -> Tuple[int, int, str]:
    try:
        current_parse = parser.parse_file(driver_file_path)
    except Exception:
        return (fallback_start_line, fallback_end_line, 'original_parse_fallback')
    for current_function in current_parse.functions:
        if current_function.name == function_name:
            return (current_function.start_line, current_function.end_line, 'current_instrumented_parse')
    return (fallback_start_line, fallback_end_line, 'original_parse_fallback')

def _write_passing_test_subset_snapshot(*, source_test_path: str, output_path: str, passed_tests: List[str]) -> Optional[str]:
    selected = {item for item in passed_tests if isinstance(item, str) and item}
    if not selected:
        return None
    source = _read_text(source_test_path)
    filtered = filter_kunit_cases_to_tests(source, selected)
    _write_text(output_path, filtered)
    return output_path

def _mutation_ready_passed_tests(test_code: str, passed_tests: List[str]) -> Tuple[List[str], List[str]]:
    selected = [item for item in passed_tests if isinstance(item, str) and item]
    effective = set(nontrivial_assertion_tests(test_code or '', selected))
    weak = [item for item in selected if item not in effective]
    return (selected, weak)

def _mutation_candidate_tests_from_bindings(passed_tests: List[str], scenario_bindings: List[Dict], scenario_total: int) -> Tuple[List[str], List[str]]:
    """Select mutation candidates at test-function granularity.

    Scenario status is an aggregate over every test bound to that scenario.  A
    later coverage variant may pass and carry a useful semantic assertion even
    when an earlier sibling test for the same scenario failed.  Mutation-ready
    selection must therefore use each KUnit test's own pass status, not the
    scenario's aggregate status.
    """
    selected = [item for item in passed_tests if isinstance(item, str) and item]
    if scenario_total == 0 or not scenario_bindings:
        return (selected, [])
    passed_set = set(selected)
    candidates: List[str] = []
    blocked: List[str] = []
    blocking_statuses = {'UNREALIZABLE_IN_CURRENT_HARNESS', 'NOT_REACHED', 'PLANNED'}
    for binding in scenario_bindings or []:
        test_name = binding.get('test_function')
        if not test_name or test_name not in passed_set:
            continue
        if binding.get('kunit_status') != 'passed':
            continue
        if not binding.get('scenario_id'):
            continue
        scenario_status_value = binding.get('scenario_status')
        if scenario_status_value in blocking_statuses:
            if test_name not in blocked:
                blocked.append(test_name)
            continue
        if test_name not in candidates:
            candidates.append(test_name)
    return (candidates, blocked)

def _write_fixed_tests_manifest(manifest_path: str, *, function_name: str, driver_path: str, test_file_path: str, test_snapshot_path: Optional[str], suite_name: str, iteration: int, scenario_context_path: str, qemu_log_path: str, suite_results_path: str, kunit_summary: Dict, coverage_summary: Dict, scenario_runtime_status: Dict, scenario_bindings: List[Dict], mutation_ready: bool, mock_bindings: Optional[List[Dict]]=None, mutation_ready_tests: Optional[List[str]]=None, runnable_passed_tests: Optional[List[str]]=None, unrealizable_but_kunit_passed_tests: Optional[List[str]]=None, excluded_failed_tests: Optional[List[str]]=None, weak_oracle_tests: Optional[List[str]]=None, mutation_test_snapshot_path: Optional[str]=None) -> Dict:
    data = {'function': function_name, 'driver_path': driver_path, 'test_file_path': test_file_path, 'test_snapshot_path': test_snapshot_path, 'suite_name': suite_name, 'iteration': iteration, 'scenario_context_path': scenario_context_path, 'qemu_log_path': qemu_log_path, 'suite_results_path': suite_results_path, 'mutation_ready': mutation_ready, 'mutation_scope': 'passing_tests_only' if mutation_ready else 'none', 'mutation_ready_tests': mutation_ready_tests or [], 'runnable_passed_tests': runnable_passed_tests or [], 'unrealizable_but_kunit_passed_tests': unrealizable_but_kunit_passed_tests or [], 'excluded_failed_tests': excluded_failed_tests or [], 'weak_oracle_tests': weak_oracle_tests or [], 'mutation_test_snapshot_path': mutation_test_snapshot_path, 'kunit': kunit_summary, 'test_metrics': test_metrics_from_kunit(kunit_summary), 'coverage': coverage_summary, 'scenario_complete': scenario_status_complete(scenario_runtime_status), 'scenario_status': scenario_runtime_status, 'scenario_test_bindings': scenario_bindings, 'mock_bindings': mock_bindings or []}
    _write_json(manifest_path, data)
    return data

def _load_function_dataset(script_dir: str) -> Tuple[Dict[str, List[str]], str]:
    path = os.path.join(script_dir, 'data', 'ut_targets.json')
    if not os.path.exists(path):
        raise FileNotFoundError(f'未找到固定实验函数数据集: {path}')
    with open(path, 'r', encoding='utf-8') as f:
        return (json.load(f), path)

def _load_driver_selection_list(function_dataset: Dict[str, List[str]]) -> List[str]:
    """Build the driver list directly from the target-function dataset."""
    return list(function_dataset.keys())
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ.pop('HF_ENDPOINT', None)

def _snapshot_file(path: str) -> Optional[str]:
    if os.path.exists(path):
        return _read_text(path)
    return None

def _restore_file(path: str, content: Optional[str]) -> None:
    if content is None:
        return
    _write_text(path, content)

def _rewrite_file_if_changed(path: str, transform) -> bool:
    if not os.path.exists(path):
        return False
    before = _read_text(path)
    after, changed = transform(before)
    if changed:
        _write_text(path, after)
    return bool(changed)

def _remove_generated_stub_section(content: str) -> Tuple[str, bool]:
    cleaned = remove_generated_boundary_instrumentation(content)
    boundary_changed = cleaned != content
    content = cleaned
    start = content.find(SECTION_BEGIN)
    if start < 0:
        return (content, boundary_changed)
    end = content.find(SECTION_END, start)
    if end < 0:
        return (content, boundary_changed)
    end += len(SECTION_END)
    new_content = content[:start].rstrip() + '\n' + content[end:].lstrip('\n')
    return (new_content, boundary_changed or new_content != content)

def _remove_kconfig_block(content: str, config_name: str) -> Tuple[str, bool]:
    lines = content.splitlines(keepends=True)
    out: List[str] = []
    removed = False
    idx = 0
    block_start = re.compile(f'^\\s*config\\s+{re.escape(config_name)}\\b')
    next_top_level = re.compile('^\\s*(config|menuconfig|menu|endmenu|source|comment|if|endif|choice|endchoice)\\b')
    while idx < len(lines):
        if not block_start.match(lines[idx]):
            out.append(lines[idx])
            idx += 1
            continue
        removed = True
        idx += 1
        while idx < len(lines):
            if next_top_level.match(lines[idx]):
                break
            idx += 1
    return (''.join(out), removed)

def _remove_makefile_artifacts(content: str, config_name: str, driver_object: str) -> Tuple[str, bool]:
    removed = False
    out = []
    gcov_directive = f'GCOV_PROFILE_{driver_object} := y'
    config_ref = f'CONFIG_{config_name}'
    for line in content.splitlines(keepends=True):
        if config_ref in line or line.strip() == gcov_directive:
            removed = True
            continue
        out.append(line)
    return (''.join(out), removed)

def _remove_config_entries(content: str, config_name: str) -> Tuple[str, bool]:
    removed = False
    out = []
    config_re = re.compile(f'^\\s*#?\\s*CONFIG_{re.escape(config_name)}(?:=|\\s+is\\s+not\\s+set)')
    for line in content.splitlines(keepends=True):
        if config_re.match(line):
            removed = True
            continue
        out.append(line)
    return (''.join(out), removed)

def _cleanup_driver_artifacts(linux_kernel_path: str, linux_config_path: str, driver_rel_path: str, target_function_names: List[str]) -> Dict[str, object]:
    driver_file_path = os.path.join(linux_kernel_path, driver_rel_path)
    driver_dir_path = os.path.dirname(driver_file_path)
    file_name = os.path.splitext(os.path.basename(driver_file_path))[0]
    config_name = f"{file_name.upper().replace('-', '_')}_KUNIT_TEST"
    driver_object = f'{file_name}.o'
    kconfig_path = os.path.join(driver_dir_path, 'Kconfig')
    makefile_path = os.path.join(driver_dir_path, 'Makefile')
    changed: List[str] = []
    removed_files: List[str] = []

    def rewrite_if_changed(path: str, transform) -> None:
        if not os.path.exists(path):
            return
        before = _read_text(path)
        after, did_change = transform(before)
        if did_change:
            _write_text(path, after)
            changed.append(path)
    rewrite_if_changed(driver_file_path, _remove_generated_stub_section)
    rewrite_if_changed(kconfig_path, lambda text: _remove_kconfig_block(text, config_name))
    rewrite_if_changed(makefile_path, lambda text: _remove_makefile_artifacts(text, config_name, driver_object))
    rewrite_if_changed(linux_config_path, lambda text: _remove_config_entries(text, config_name))
    candidates = set(target_function_names or [])
    for name in candidates:
        path = os.path.join(driver_dir_path, f'{name}_test_case.c')
        if os.path.exists(path):
            os.remove(path)
            removed_files.append(path)
    return {'driver': driver_rel_path, 'config': config_name, 'changed_files': changed, 'removed_files': removed_files}

def _extract_error_summary(log_text: str) -> str:
    if not log_text:
        return 'No log captured.'
    lines = log_text.splitlines()
    path_pattern = re.compile('^(?:\\.{1,2}/|/|[A-Za-z0-9_.-]+/).+?:\\d+(?::\\d+)?')
    make_pattern = re.compile('^make\\[\\d+\\]:')
    snippets: List[str] = []
    i = 0
    total = len(lines)
    while i < total:
        stripped = lines[i].lstrip()
        if path_pattern.match(stripped):
            segment: List[str] = [lines[i]]
            j = i + 1
            while j < total:
                next_stripped = lines[j].lstrip()
                if not lines[j].strip():
                    segment.append(lines[j])
                    j += 1
                    break
                if path_pattern.match(next_stripped) or make_pattern.match(next_stripped):
                    break
                segment.append(lines[j])
                j += 1
            snippets.append('\n'.join(segment).strip())
            i = j
        else:
            i += 1
    if snippets:
        return '\n'.join(snippets)
    return _condense_log(log_text)

def _format_line_snippets(file_path: Optional[str], lines: List[int]) -> List[str]:
    if not file_path or not os.path.exists(file_path) or (not lines):
        return []
    snippets: List[str] = []
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            file_lines = f.readlines()
    except OSError:
        return []
    total = len(file_lines)
    for line_no in lines:
        if line_no <= 0 or line_no > total:
            continue
        code = file_lines[line_no - 1].rstrip('\n')
        snippets.append(f'{line_no}: {code}')
    return snippets

def _split_c_parameters(param_text: str) -> List[str]:
    param_text = (param_text or '').strip()
    if not param_text or param_text == 'void':
        return []
    params: List[str] = []
    depth = 0
    start = 0
    for idx, ch in enumerate(param_text):
        if ch in '([{':
            depth += 1
        elif ch in ')]}' and depth:
            depth -= 1
        elif ch == ',' and depth == 0:
            params.append(param_text[start:idx].strip())
            start = idx + 1
    tail = param_text[start:].strip()
    if tail:
        params.append(tail)
    return params

def _extract_c_param_name(param: str, fallback: str) -> str:
    text = (param or '').strip()
    text = re.sub('/\\*.*?\\*/', '', text)
    text = text.split('=')[0].strip()
    match = re.search('([A-Za-z_][A-Za-z0-9_]*)\\s*(?:\\[[^\\]]*\\])?\\s*$', text)
    if match:
        name = match.group(1)
        if name not in {'const', 'struct', 'union', 'enum', 'void'}:
            return name
    return fallback

def _format_c_param(param_type: str, name: str) -> str:
    param_type = re.sub('\\s+', ' ', (param_type or '').strip())
    param_type = param_type.replace(' *', ' *').replace('* ', '*')
    if param_type.endswith('*'):
        return f'{param_type}{name}'
    return f'{param_type} {name}'

def _parse_expected_function_pointer_type(type_text: str) -> Optional[Tuple[str, List[str]]]:
    text = re.sub('\\s+', ' ', (type_text or '').strip())
    match = re.match('(?P<ret>.+?)\\s*\\(\\s*\\*\\s*\\)\\s*\\((?P<params>.*)\\)\\s*$', text)
    if not match:
        return None
    ret_type = match.group('ret').strip()
    param_types = _split_c_parameters(match.group('params'))
    return (ret_type, param_types)

def _expected_hook_types_from_build_log(log_text: str) -> Dict[str, Tuple[str, List[str]]]:
    if not log_text:
        return {}
    results: Dict[str, Tuple[str, List[str]]] = {}
    lines = log_text.splitlines()
    call_pattern = re.compile('(?P<setter>raca_boundary_[A-Za-z0-9_]+_set_hook)\\s*\\(\\s*(?P<fake>[A-Za-z_][A-Za-z0-9_]*)\\s*\\)')
    expected_patterns = [re.compile("需要类型[‘'`](?P<type>.+?\\(\\s*\\*\\s*\\)\\s*\\([^’'`]+\\))[’'`]"), re.compile("expected\\s+[‘'`](?P<type>.+?\\(\\s*\\*\\s*\\)\\s*\\([^’'`]+\\))[’'`]")]
    for idx, line in enumerate(lines):
        call = call_pattern.search(line)
        if not call:
            continue
        fake_name = call.group('fake')
        window = '\n'.join(lines[idx:min(len(lines), idx + 8)])
        for pattern in expected_patterns:
            type_match = pattern.search(window)
            if not type_match:
                continue
            parsed = _parse_expected_function_pointer_type(type_match.group('type'))
            if parsed:
                results[fake_name] = parsed
                break
    return results

def _rewrite_function_signature(test_code: str, function_name: str, return_type: str, param_types: List[str]) -> Tuple[str, bool]:
    if not test_code or not function_name:
        return (test_code, False)
    name_re = re.escape(function_name)
    pattern = re.compile('(?P<prefix>\\bstatic\\s+)(?P<ret>[A-Za-z_][A-Za-z0-9_\\s\\*\\d]*?)\\s+' + name_re + '\\s*\\((?P<params>[^;{}]*)\\)\\s*(?=\\{)', re.MULTILINE)
    match = pattern.search(test_code)
    if not match:
        return (test_code, False)
    old_params = _split_c_parameters(match.group('params'))
    names = [_extract_c_param_name(param, f'arg{idx}') for idx, param in enumerate(old_params)]
    if len(names) < len(param_types):
        names.extend((f'arg{idx}' for idx in range(len(names), len(param_types))))
    new_params = ', '.join((_format_c_param(param_type, names[idx]) for idx, param_type in enumerate(param_types))) or 'void'
    new_signature = f"{match.group('prefix')}{return_type} {function_name}({new_params})"
    return (test_code[:match.start()] + new_signature + test_code[match.end():], True)

def _auto_repair_hook_fake_signatures(test_code: str, build_log: str) -> Tuple[str, List[Dict[str, object]]]:
    repairs: List[Dict[str, object]] = []
    updated = test_code or ''
    for fake_name, (return_type, param_types) in _expected_hook_types_from_build_log(build_log).items():
        updated_next, changed = _rewrite_function_signature(updated, fake_name, return_type, param_types)
        if changed:
            updated = updated_next
            repairs.append({'fake': fake_name, 'return_type': return_type, 'param_types': param_types})
    return (updated, repairs)

def _missing_fake_symbols_from_build_log(log_text: str) -> Set[str]:
    if not log_text:
        return set()
    symbols: Set[str] = set()
    for match in re.finditer("[‘'`]((?:fake|mock)_[A-Za-z0-9_]+)[’'`]\\s+(?:undeclared|未声明)", log_text):
        symbols.add(match.group(1))
    return symbols

def _function_prototype_from_kernel(linux_kernel_path: str, function_name: str) -> Optional[Tuple[str, str]]:
    if not linux_kernel_path or not function_name:
        return None
    search_roots = [os.path.join(linux_kernel_path, 'include'), os.path.join(linux_kernel_path, 'drivers')]
    existing_roots = [path for path in search_roots if os.path.exists(path)]
    if not existing_roots:
        return None
    try:
        proc = subprocess.run(['rg', '-l', '--glob', '*.h', '--glob', '*.c', f'\\b{re.escape(function_name)}\\s*\\(', *existing_roots], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, check=False, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return None

    def _strip_c_comments(text: str) -> str:
        text = re.sub('/\\*.*?\\*/', ' ', text, flags=re.DOTALL)
        return re.sub('//.*', ' ', text)

    def _valid_type_fragment(type_text: str) -> bool:
        if not type_text:
            return False
        if any((token in type_text for token in ('(', ')', '{', '}', ';', ','))):
            return False
        if re.search('\\b(?:return|if|for|while|switch|case|do|else)\\b', type_text):
            return False
        return bool(re.fullmatch('[A-Za-z_][A-Za-z0-9_\\s\\*]*', type_text.strip()))

    def _valid_params_fragment(params_text: str) -> bool:
        if params_text is None:
            return False
        if any((token in params_text for token in ('{', '}', ';'))):
            return False
        if re.search('\\b(?:return|if|for|while|switch|case|do|else)\\b', params_text):
            return False
        return bool(params_text.strip())
    pattern = re.compile('^[ \\t]*(?:(?:static|extern|inline|__always_inline|__maybe_unused|__must_check|__printf\\s*\\([^)]*\\))\\s+)*(?P<ret>[A-Za-z_][A-Za-z0-9_\\s\\*]*?)' + '\\b' + re.escape(function_name) + '\\s*\\((?P<params>[^;{}]*)\\)\\s*(?:;|\\{)', re.MULTILINE)
    paths = proc.stdout.splitlines()
    paths.sort(key=lambda path: (0 if f'/include/' in path else 1, len(path)))
    for path in paths[:80]:
        try:
            text = _strip_c_comments(_read_text(path))
        except OSError:
            continue
        for match in pattern.finditer(text):
            return_type = re.sub('\\s+', ' ', match.group('ret')).strip()
            params = re.sub('\\s+', ' ', match.group('params')).strip()
            if _valid_type_fragment(return_type) and _valid_params_fragment(params):
                return (return_type, params)
    return None

def _fake_name_for_boundary_expression(expression: str) -> str:
    safe = re.sub('[^A-Za-z0-9_]', '_', expression or 'boundary')
    return f'fake_{safe}'

def _boundary_hook_fake_specs(scenario_context: Dict, linux_kernel_path: str, missing_symbols: Set[str], test_code: str) -> List[Dict[str, object]]:
    target = (scenario_context or {}).get('target', {}) or {}
    controls = [item for item in target.get('export_interface_details', []) or [] if isinstance(item, dict) and item.get('source_kind') == 'boundary_hook' and (item.get('boundary_control_role') == 'set_hook')]
    specs: List[Dict[str, object]] = []
    code_without_auto_blocks = _remove_auto_boundary_fake_blocks(test_code or '')

    def _hook_targets_for_boundary(boundary_id: str) -> Set[str]:
        return {match.group(1) for match in re.finditer(f'\\braca_boundary_{re.escape(boundary_id)}_set_hook\\s*\\(\\s*([A-Za-z_][A-Za-z0-9_]*)\\s*\\)', test_code or '') if match.group(1) not in {'NULL', 'null'}}

    def _has_static_function_definition(function_name: str) -> bool:
        return bool(re.search(f'^\\s*static\\s+[\\w\\s\\*]+\\b{re.escape(function_name)}\\s*\\(', code_without_auto_blocks, re.MULTILINE))

    def _state_symbol_pool(canonical_fake_name: str, hook_fake_name: str, related: Set[str]) -> Set[str]:
        prefixes = {canonical_fake_name, hook_fake_name}
        pool = set(related)
        for prefix in prefixes:
            pool.update(re.findall(f'\\b{re.escape(prefix)}_[A-Za-z0-9_]+\\b', test_code or ''))
        return pool

    def _preferred_state_name(pool: Set[str], canonical_fake_name: str, hook_fake_name: str, suffix: str) -> Optional[str]:
        preferred = [f'{canonical_fake_name}{suffix}', f'{hook_fake_name}{suffix}']
        for name in preferred:
            if name in pool:
                return name
        candidates = sorted((symbol for symbol in pool if symbol.endswith(suffix)))
        return candidates[0] if candidates else None
    for control in controls:
        original = str(control.get('boundary_expression', '') or '').strip()
        boundary_id = str(control.get('boundary_id', '') or '').strip()
        canonical_fake_name = _fake_name_for_boundary_expression(original)
        hook_targets = _hook_targets_for_boundary(boundary_id)
        fake_name = canonical_fake_name
        referenced = bool(re.search(f'\\b{re.escape(canonical_fake_name)}\\b', test_code or '') or any((re.search(f'\\b{re.escape(target)}\\b', test_code or '') for target in hook_targets)))
        related_missing = {symbol for symbol in missing_symbols if symbol == canonical_fake_name or symbol.startswith(canonical_fake_name + '_') or any((symbol == target or symbol.startswith(target + '_') for target in hook_targets))}
        if not related_missing and (not referenced):
            continue
        if _has_static_function_definition(canonical_fake_name):
            continue
        prototype = _function_prototype_from_kernel(linux_kernel_path, original)
        if not prototype:
            continue
        return_type, params = prototype
        state_pool = _state_symbol_pool(canonical_fake_name, fake_name, related_missing)
        return_state_name = None
        if return_type != 'void':
            return_state_name = _preferred_state_name(state_pool, canonical_fake_name, fake_name, '_return') or f'{canonical_fake_name}_return'
        call_count_name = _preferred_state_name(state_pool, canonical_fake_name, fake_name, '_call_count')
        specs.append({'boundary_id': boundary_id, 'original': original, 'canonical_fake_name': canonical_fake_name, 'fake_name': fake_name, 'return_type': return_type, 'params': params, 'return_state_name': return_state_name, 'call_count_name': call_count_name, 'related_missing': sorted(related_missing), 'hook_targets': sorted(hook_targets)})
    return specs

def _default_boundary_fake_return(alias_name: str) -> Optional[str]:
    if re.search('(?:^|_)(?:success|ok)(?:_|$)', alias_name):
        return '0'
    if re.search('(?:^|_)(?:positive|valid)(?:_|$)', alias_name):
        return '1'
    if re.search('(?:^|_)(?:negative|error|err|fail|failure|errval)(?:_|$)', alias_name):
        return '-EIO'
    return None

def _normalize_boundary_hook_fake_contract(test_code: str, scenario_context: Dict, linux_kernel_path: str, missing_symbols: Optional[Set[str]]=None) -> Tuple[str, List[Dict[str, object]], List[Dict[str, object]]]:
    """Force direct-boundary hooks to use framework-owned canonical fake names.

    The LLM may edit the canonical fake implementation, but it must not invent
    independent hook fake declarations such as fake_foo_success or
    fake_foo_negative. Those aliases are normalized to fake_foo plus the
    generated fake_foo_return state.
    """
    original_code = test_code or ''
    target = (scenario_context or {}).get('target', {}) or {}
    controls = [item for item in target.get('export_interface_details', []) or [] if isinstance(item, dict) and item.get('source_kind') == 'boundary_hook' and (item.get('boundary_control_role') == 'set_hook')]
    specs = _boundary_hook_fake_specs(scenario_context, linux_kernel_path, missing_symbols or set(), original_code)
    updated = original_code
    actions: List[Dict[str, object]] = []
    marker_updates: List[Tuple[str, str, str]] = []
    for control in controls:
        original = str(control.get('boundary_expression', '') or '').strip()
        boundary_id = str(control.get('boundary_id', '') or '').strip()
        if not original or not boundary_id:
            continue
        canonical = _fake_name_for_boundary_expression(original)
        return_state = f'{canonical}_return'
        state_aliases: List[str] = []
        for suffix in ('_errval', '_retval', '_ret', '_result', '_value'):
            state_alias = f'{canonical}{suffix}'
            updated_next = re.sub(f'\\b{re.escape(state_alias)}\\b', return_state, updated)
            if updated_next != updated:
                state_aliases.append(state_alias)
                updated = updated_next
        if state_aliases:
            actions.append({'boundary_id': boundary_id, 'original': original, 'canonical_fake_name': canonical, 'return_state_name': return_state, 'normalized_state_aliases': state_aliases})
    if not specs:
        if updated == original_code:
            return (original_code, [], [])
        return (updated, [], actions)
    for spec in specs:
        boundary_id = str(spec['boundary_id'])
        canonical = str(spec['canonical_fake_name'])
        return_state = str(spec.get('return_state_name') or '')
        aliases = [name for name in spec.get('hook_targets', []) or [] if name and name != canonical]
        if not aliases:
            continue
        setter = f'raca_boundary_{boundary_id}_set_hook'
        alias_actions: List[Dict[str, str]] = []
        for alias in aliases:
            default_return = _default_boundary_fake_return(alias)
            statement_pattern = re.compile(f'(?m)^(?P<indent>[ \\t]*)(?P<setter>{re.escape(setter)}\\s*\\(\\s*){re.escape(alias)}(?P<suffix>\\s*\\)\\s*;)')

            def _replace_setter(match, value=default_return):
                call = f"{match.group('indent')}{match.group('setter')}{canonical}{match.group('suffix')}"
                if return_state and value is not None:
                    return f"{match.group('indent')}{return_state} = {value};\n{call}"
                return call
            updated, count = statement_pattern.subn(_replace_setter, updated)
            if count == 0:
                updated = re.sub(f'\\b{re.escape(setter)}\\s*\\(\\s*{re.escape(alias)}\\s*\\)', f'{setter}({canonical})', updated)
            if return_state:
                for suffix in ('_errval', '_retval', '_ret', '_return', '_result', '_value'):
                    updated = re.sub(f'\\b{re.escape(alias)}{suffix}\\b', return_state, updated)
                for suffix in ('_errval', '_retval', '_ret', '_result', '_value'):
                    updated = re.sub(f'\\b{re.escape(canonical)}{suffix}\\b', return_state, updated)
            marker_updates.append((boundary_id, alias, canonical))
            alias_actions.append({'alias': alias, 'canonical': canonical, 'default_return': default_return or ''})
        if alias_actions:
            actions.append({'boundary_id': boundary_id, 'original': spec.get('original', ''), 'canonical_fake_name': canonical, 'return_state_name': return_state, 'normalized_aliases': alias_actions})
        if return_state:
            state_aliases: List[str] = []
            for suffix in ('_errval', '_retval', '_ret', '_result', '_value'):
                state_alias = f'{canonical}{suffix}'
                updated_next = re.sub(f'\\b{re.escape(state_alias)}\\b', return_state, updated)
                if updated_next != updated:
                    state_aliases.append(state_alias)
                    updated = updated_next
            if state_aliases:
                actions.append({'boundary_id': boundary_id, 'original': spec.get('original', ''), 'canonical_fake_name': canonical, 'return_state_name': return_state, 'normalized_state_aliases': state_aliases})
    for boundary_id, alias, canonical in marker_updates:
        updated = re.sub(f'(RACA_MOCK\\s*:\\s*boundary={re.escape(boundary_id)}\\s*;\\s*original=[^;]+;\\s*replacement=){re.escape(alias)}\\b', f'\\1{canonical}', updated)
    block = _build_fake_skeleton_block(specs)
    if block:
        updated = _insert_auto_boundary_fakes(_remove_auto_boundary_fake_blocks(updated), block)
    if updated == original_code:
        return (original_code, [], [])
    return (updated, specs, actions)

def _build_fake_skeleton_block(specs: List[Dict[str, object]]) -> str:
    if not specs:
        return ''
    lines = ['/* RACA auto boundary fake definitions BEGIN */\n', '/* Generated from compiler-reported missing fake symbols and boundary hook metadata. */\n']
    for spec in specs:
        return_type = str(spec['return_type'])
        params = str(spec['params'])
        fake_name = str(spec['fake_name'])
        return_state_name = spec.get('return_state_name')
        call_count_name = spec.get('call_count_name')
        original = str(spec['original'])
        boundary_id = str(spec['boundary_id'])
        lines.append(f'/* RACA_MOCK: boundary={boundary_id}; original={original}; replacement={fake_name} */\n')
        if call_count_name:
            lines.append(f'static unsigned long {call_count_name};\n')
        if return_type != 'void' and return_state_name:
            lines.append(f'static {return_type} {return_state_name};\n')
        lines.append(f'static {return_type} {fake_name}({params})\n')
        lines.append('{\n')
        if call_count_name:
            lines.append(f'\t{call_count_name}++;\n')
        if return_type != 'void' and return_state_name:
            lines.append(f'\treturn {return_state_name};\n')
        lines.append('}\n\n')
    lines.append('/* RACA auto boundary fake definitions END */\n')
    return ''.join(lines)

def _remove_auto_boundary_fake_blocks(test_code: str) -> str:
    return re.sub('\\n?/\\* RACA auto boundary fake definitions BEGIN \\*/.*?/\\* RACA auto boundary fake definitions END \\*/\\n?', '\n', test_code or '', flags=re.DOTALL)

def _insert_auto_boundary_fakes(test_code: str, block: str) -> str:
    marker = '/* ===== Flexible Helpers BEGIN ===== */'
    if marker in test_code:
        return test_code.replace(marker, marker + '\n' + block + '\n', 1)
    marker = '/* ===== Scenario Tests BEGIN ===== */'
    if marker in test_code:
        return test_code.replace(marker, block + '\n' + marker, 1)
    return test_code.rstrip() + '\n\n' + block

def _auto_insert_missing_boundary_fakes(test_code: str, build_log: str, scenario_context: Dict, linux_kernel_path: str) -> Tuple[str, List[Dict[str, object]]]:
    missing = _missing_fake_symbols_from_build_log(build_log)
    if not missing:
        return (test_code or '', [])
    normalized, specs, actions = _normalize_boundary_hook_fake_contract(test_code or '', scenario_context, linux_kernel_path, missing)
    for spec, action in zip(specs, actions):
        spec['normalization'] = action
    return (normalized, specs)

def _nested_function_diagnostics(test_code: str) -> List[str]:
    diagnostics: List[str] = []
    inspected = inspect_test_source(test_code or '')
    for test_func in inspected.test_functions:
        body = test_func.body or ''
        if not body:
            continue
        for match in re.finditer('(?m)^[ \\t]*(?:static\\s+)?(?:[A-Za-z_][A-Za-z0-9_]*[\\w\\s\\*]*\\s+)+(?P<name>[A-Za-z_][A-Za-z0-9_]*)\\s*\\([^;{}]*\\)\\s*\\{', body):
            name = match.group('name')
            if name in {'if', 'for', 'while', 'switch'}:
                continue
            diagnostics.append(f'Test function {test_func.name} contains nested function definition {name}; helper/fake functions must be file-scope.')
    return diagnostics

def _poison_fake_pointer_diagnostics(test_code: str) -> List[str]:
    diagnostics: List[str] = []
    poison_pattern = re.compile('\\b(?P<state>fake_[A-Za-z0-9_]*_return)\\s*=\\s*(?:\\([^;=]*\\)\\s*)?(?P<value>0x(?:deadbeef|DEADBEEF|[0-9A-Fa-f]*dead[0-9A-Fa-f]*))\\s*;', re.MULTILINE)
    for match in poison_pattern.finditer(test_code or ''):
        diagnostics.append(f"Fake return state {match.group('state')} is assigned poison pointer {match.group('value')}; allocate the correctly typed object and initialize downstream fields/locks instead.")
    return diagnostics

def summarize_driver_iterations(driver_rel_path: str, function_summaries: List[Dict[str, object]]) -> Dict[str, object]:
    total_functions = len(function_summaries)
    iteration_stats: Dict[int, Dict[str, float]] = {}
    all_iterations = set()
    for summary in function_summaries:
        for entry in summary.get('history', []):
            iteration = entry.get('iteration')
            if iteration is None:
                continue
            all_iterations.add(iteration)
            if entry.get('stage') != 'results':
                continue
            stats = iteration_stats.setdefault(iteration, {'compile_pass_count': 0, 'test_pass_rate_sum': 0.0, 'tests_total': 0, 'tests_passed': 0, 'tests_failed': 0, 'line_cov_sum': 0.0, 'branch_cov_sum': 0.0, 'block_cov_sum': 0.0})
            stats['compile_pass_count'] += 1
            kunit_summary = entry.get('kunit', {}) or {}
            stats['test_pass_rate_sum'] += float(kunit_summary.get('pass_rate', 0.0))
            stats['tests_total'] += int(kunit_summary.get('tests_total', len(kunit_summary.get('tests', []) or [])) or 0)
            stats['tests_passed'] += int(kunit_summary.get('tests_passed_count', len(kunit_summary.get('passed', []) or [])) or 0)
            stats['tests_failed'] += int(kunit_summary.get('tests_failed_count', len(kunit_summary.get('failed', []) or [])) or 0)
            coverage_summary = entry.get('coverage', {}) or {}
            stats['line_cov_sum'] += float(coverage_summary.get('line_percent', 0.0))
            stats['branch_cov_sum'] += float(coverage_summary.get('branch_percent', 0.0))
            stats['block_cov_sum'] += float(coverage_summary.get('blocks_percent', 0.0))
    iterations = []
    for iteration in sorted(all_iterations):
        stats = iteration_stats.get(iteration, {'compile_pass_count': 0, 'test_pass_rate_sum': 0.0, 'tests_total': 0, 'tests_passed': 0, 'tests_failed': 0, 'line_cov_sum': 0.0, 'branch_cov_sum': 0.0, 'block_cov_sum': 0.0})
        compile_pass_count = stats['compile_pass_count']
        compile_pass_rate = compile_pass_count / total_functions * 100.0 if total_functions else 0.0
        avg_test_pass_rate = stats['test_pass_rate_sum'] / compile_pass_count if compile_pass_count else 0.0
        avg_line_cov = stats['line_cov_sum'] / compile_pass_count if compile_pass_count else 0.0
        avg_branch_cov = stats['branch_cov_sum'] / compile_pass_count if compile_pass_count else 0.0
        avg_block_cov = stats['block_cov_sum'] / compile_pass_count if compile_pass_count else 0.0
        iterations.append({'iteration': iteration, 'compile_pass_count': compile_pass_count, 'compile_pass_rate': compile_pass_rate, 'tests_total': int(stats['tests_total']), 'tests_passed': int(stats['tests_passed']), 'tests_failed': int(stats['tests_failed']), 'overall_test_pass_rate': stats['tests_passed'] / stats['tests_total'] * 100.0 if stats['tests_total'] else 0.0, 'average_test_pass_rate': avg_test_pass_rate, 'average_line_coverage': avg_line_cov, 'average_branch_coverage': avg_branch_cov, 'average_block_coverage': avg_block_cov})
    return {'driver': driver_rel_path, 'total_functions': total_functions, 'iterations': iterations}

def main():
    global MAX_FIX_ATTEMPTS
    global TARGET_COVERAGE_PERCENT
    args = _parse_args()
    MAX_FIX_ATTEMPTS = args.max_fix_attempts
    TARGET_COVERAGE_PERCENT = args.target_coverage
    script_dir = os.path.dirname(os.path.abspath(__file__))
    result_root = args.result_root or os.getenv('RACA_RESULT_ROOT') or os.path.join(script_dir, 'output_all')
    if not args.linux_kernel_path:
        raise SystemExit('Missing Linux kernel path. Pass --linux-kernel-path or set RACA_LINUX_KERNEL_PATH.')
    if not args.buildroot_dir:
        raise SystemExit('Missing Buildroot path. Pass --buildroot-dir or set RACA_BUILDROOT_DIR.')
    linux_kernel_path = os.path.abspath(args.linux_kernel_path)
    buildroot_dir_path = os.path.abspath(args.buildroot_dir)
    linux_config_path = os.path.join(linux_kernel_path, '.config')
    model_path = args.model_path or os.getenv('RACA_LOCAL_MODEL_PATH')
    prompt_path = os.path.join(script_dir, 'prompt.yaml')
    local_or_api = args.local_or_api
    try:
        function_dataset, function_dataset_path = _load_function_dataset(script_dir)
        print(f'[INFO] 使用函数数据集: {function_dataset_path}')
    except FileNotFoundError as exc:
        print(f'[ERROR] {exc}')
        return
    driver_rel_paths = _load_driver_selection_list(function_dataset)
    if not driver_rel_paths:
        print('[WARN] 驱动列表为空，且函数数据集没有可运行驱动。')
        return
    selected_drivers = set(args.driver or [])
    if selected_drivers:
        driver_rel_paths = [path for path in driver_rel_paths if path in selected_drivers]
    if args.limit_drivers and args.limit_drivers > 0:
        driver_rel_paths = driver_rel_paths[:args.limit_drivers]
    if not driver_rel_paths:
        print('[WARN] 没有驱动满足当前 --driver/--limit-drivers 过滤条件。')
        return
    selected_functions = set(args.function or [])
    if args.cleanup_only:
        for driver_entry in driver_rel_paths:
            driver_rel_path = driver_entry.split()[0]
            target_function_names = list(function_dataset.get(driver_rel_path, []))
            if selected_functions:
                target_function_names = [name for name in target_function_names if name in selected_functions]
            if args.limit_functions and args.limit_functions > 0:
                target_function_names = target_function_names[:args.limit_functions]
            report = _cleanup_driver_artifacts(linux_kernel_path=linux_kernel_path, linux_config_path=linux_config_path, driver_rel_path=driver_rel_path, target_function_names=target_function_names)
            print('[CLEANUP] {driver}: changed={changed}, removed={removed}'.format(driver=report['driver'], changed=len(report['changed_files']), removed=len(report['removed_files'])))
            for path in report['changed_files']:
                print(f'  changed: {path}')
            for path in report['removed_files']:
                print(f'  removed: {path}')
        return
    if local_or_api == 'local':
        model, tokenizer = load_local_model(model_path)
    else:
        model = None
        tokenizer = None
    host_share_path = os.path.join(buildroot_dir_path, 'qemu-share')
    os.makedirs(host_share_path, exist_ok=True)
    for driver_rel_path in driver_rel_paths:
        driver_file_path = os.path.join(linux_kernel_path, driver_rel_path)
        if not os.path.isfile(driver_file_path):
            print(f'[WARN] 驱动文件不存在，跳过: {driver_rel_path}')
            continue
        driver_start = time.time()
        driver_dir_path = os.path.dirname(driver_file_path)
        file_name = os.path.splitext(os.path.basename(driver_file_path))[0]
        driver_config_symbol = file_name.upper().replace('-', '_')
        driver_result_dir = os.path.join(result_root, file_name)
        os.makedirs(driver_result_dir, exist_ok=True)
        kconfig_path = os.path.join(driver_dir_path, 'Kconfig')
        makefile_path = os.path.join(driver_dir_path, 'Makefile')
        driver_object = f'{file_name}.o'
        kunit_config_name = f'{driver_config_symbol}_KUNIT_TEST'
        print(f'[DRIVER] 开始处理 {driver_rel_path}')
        cleanup_changed = []
        if _rewrite_file_if_changed(driver_file_path, _remove_generated_stub_section):
            cleanup_changed.append(driver_file_path)
        if _rewrite_file_if_changed(kconfig_path, lambda text: _remove_kconfig_block(text, kunit_config_name)):
            cleanup_changed.append(kconfig_path)
        if _rewrite_file_if_changed(makefile_path, lambda text: _remove_makefile_artifacts(text, kunit_config_name, driver_object)):
            cleanup_changed.append(makefile_path)
        if _rewrite_file_if_changed(linux_config_path, lambda text: _remove_config_entries(text, kunit_config_name)):
            cleanup_changed.append(linux_config_path)
        if cleanup_changed:
            _write_json(os.path.join(driver_result_dir, f'{file_name}_cleanup.json'), {'changed_files': cleanup_changed})
        original_driver = _snapshot_file(driver_file_path)
        original_kconfig = _snapshot_file(kconfig_path)
        original_makefile = _snapshot_file(makefile_path)
        original_linux_config = _snapshot_file(linux_config_path)

        def reset_driver_environment():
            _restore_file(driver_file_path, original_driver)
            _restore_file(kconfig_path, original_kconfig)
            _restore_file(makefile_path, original_makefile)
            _restore_file(linux_config_path, original_linux_config)
        parser = CDriverParser()
        parse_result = parser.parse_file(driver_file_path)
        save_result(parse_result, os.path.join(driver_result_dir, f'{file_name}_code_parse_result.json'))
        driver_function_summaries: List[Dict[str, object]] = []
        target_function_names = function_dataset.get(driver_rel_path, [])
        if selected_functions:
            target_function_names = [name for name in target_function_names if name in selected_functions]
        if args.limit_functions and args.limit_functions > 0:
            target_function_names = target_function_names[:args.limit_functions]
        if not target_function_names:
            print(f'[INFO] 数据集未选择任何函数，跳过: {driver_rel_path}')
            reset_driver_environment()
            continue
        if args.dry_run_targets:
            print(f'[DRY-RUN] {driver_rel_path}: {target_function_names}')
            reset_driver_environment()
            continue
        available_function_names = {func.name for func in parse_result.functions}
        missing_function_names = [func_name for func_name in target_function_names if func_name not in available_function_names]
        for func_name in missing_function_names:
            print(f'[WARN] 数据集中的函数在当前驱动中不存在: {driver_rel_path}::{func_name}')
        target_function_set = set(target_function_names)
        print(f'[INFO] 数据集为该驱动选择了 {len(target_function_set)} 个函数')
        function_list = parse_result.functions
        for function in function_list:
            decision = function.name in target_function_set
            print(f'function {function.name} \n decision(from_dataset): {decision}')
            if decision:
                function_output_dir = os.path.join(driver_result_dir, function.name)
                os.makedirs(function_output_dir, exist_ok=True)
                reset_driver_environment()
                iteration_count = 1
                function_summary_data: Optional[Dict[str, object]] = None
                function_started_at = _utc_now_iso()
                function_start_monotonic = time.monotonic()
                generation_history: List[Dict[str, object]] = []
                try:
                    try:
                        stub_result = apply_stub(StubSpec(driver_c_path=driver_file_path, target_func_name=function.name, config_symbol=f"CONFIG_{file_name.upper().replace('-', '_')}_KUNIT_TEST", mode='single', parse_result=parse_result, enable_boundary_hooks=True))
                        generation_history.append({'iteration': iteration_count, 'stage': 'apply_stub', 'status': 'passed'})
                    except Exception as stub_err:
                        error_text = traceback.format_exc()
                        log_file = _save_error_log(function_output_dir, function.name, 'apply_stub', iteration_count, error_text)
                        generation_history.append({'iteration': iteration_count, 'stage': 'apply_stub', 'status': 'failed', 'log_file': log_file})
                        function_summary_data = _write_function_failure_summary(function_output_dir=function_output_dir, function_name=function.name, status='stub_failed', stage='apply_stub', iteration=iteration_count, reason=str(stub_err), history=generation_history, **_timing_payload(function_started_at, function_start_monotonic), error_log=log_file)
                        driver_function_summaries.append(function_summary_data)
                        continue
                    print('插桩完毕')
                    instrumented_driver_path = driver_file_path
                    try:
                        instrumented_copy = os.path.join(function_output_dir, f'{file_name}_instrumented.c')
                        shutil.copyfile(driver_file_path, instrumented_copy)
                        instrumented_driver_path = instrumented_copy
                        print(f'[INFO] 插桩后的驱动已保存至 {instrumented_copy}')
                    except OSError as copy_err:
                        print(f'[WARN] 无法保存插桩驱动副本: {copy_err}')
                    scenario_context = None
                    scenario_context_path = ''
                    scenario_context_path = os.path.join(function_output_dir, 'scenario_context.json')
                    try:
                        scenario_context = generate_scenario_context(parse_result=parse_result, function=function, local_or_api=local_or_api, prompt_path=prompt_path, model=model, tokenizer=tokenizer, export_interfaces=stub_result.export_interfaces)
                        save_scenario_context(scenario_context, scenario_context_path)
                        if isinstance(scenario_context, dict) and isinstance(scenario_context.get('scenario_registry'), dict):
                            _write_json(os.path.join(function_output_dir, 'scenario_registry.json'), scenario_context['scenario_registry'])
                        _write_json(os.path.join(function_output_dir, 'scenario_context_structure_validation.json'), validate_scenario_context_structure(scenario_context).to_dict())
                        generation_history.append({'iteration': iteration_count, 'stage': 'scenario_context', 'status': 'passed', 'scenario_context_path': scenario_context_path})
                    except Exception as context_err:
                        error_text = traceback.format_exc()
                        log_file = _save_error_log(function_output_dir, function.name, 'scenario_context', iteration_count, error_text)
                        generation_history.append({'iteration': iteration_count, 'stage': 'scenario_context', 'status': 'failed', 'log_file': log_file})
                        function_summary_data = _write_function_failure_summary(function_output_dir=function_output_dir, function_name=function.name, status='scenario_context_failed', stage='scenario_context', iteration=iteration_count, reason=str(context_err), history=generation_history, **_timing_payload(function_started_at, function_start_monotonic), error_log=log_file)
                        driver_function_summaries.append(function_summary_data)
                        continue
                    print(f'[INFO] Scenario context 已保存至 {scenario_context_path}')
                    try:
                        test_case = generate_test_case(parse_result, function, local_or_api, prompt_path, model, tokenizer, stub_result.export_interfaces, scenario_context=scenario_context)
                        generation_history.append({'iteration': iteration_count, 'stage': 'generate_test_case', 'status': 'passed'})
                    except Exception as gen_err:
                        error_text = traceback.format_exc()
                        log_file = _save_error_log(function_output_dir, function.name, 'generate_test_case', iteration_count, error_text)
                        generation_history.append({'iteration': iteration_count, 'stage': 'generate_test_case', 'status': 'failed', 'log_file': log_file})
                        function_summary_data = _write_function_failure_summary(function_output_dir=function_output_dir, function_name=function.name, status='generation_failed', stage='generate_test_case', iteration=iteration_count, reason=str(gen_err), history=generation_history, scenario_context_path=scenario_context_path, **_timing_payload(function_started_at, function_start_monotonic), error_log=log_file)
                        driver_function_summaries.append(function_summary_data)
                        continue
                    test_case, removed_initial_externs = strip_flexible_extern_declarations(test_case, scenario_context)
                    if removed_initial_externs:
                        _write_json(os.path.join(function_output_dir, f'{function.name}_initial_candidate_normalization.json'), {'removed_flexible_extern_declarations': removed_initial_externs, 'reason': 'generated extern declarations outside the fixed TEST EXPORT block are forbidden'})
                    suite_name = _extract_suite_name(test_case, f'{function.name}_test')
                    test_case_save_path = os.path.join(driver_dir_path, f'{function.name}_test_case.c')
                    kunit_config_name = f"{file_name.upper().replace('-', '_')}_KUNIT_TEST"
                    test_obj_name = f'{function.name}_test_case.o'
                    isolation_report = isolate_current_driver_test_case(driver_dir_path=driver_dir_path, makefile_path=makefile_path, config_name=kunit_config_name, current_test_file_name=os.path.basename(test_case_save_path), current_obj_name=test_obj_name)
                    if isolation_report['removed_files'] or isolation_report['changed_files']:
                        _write_json(os.path.join(function_output_dir, f'{function.name}_build_isolation.json'), isolation_report)
                    _write_text(test_case_save_path, test_case)
                    _write_json(os.path.join(function_output_dir, f'{function.name}_initial_scenario_context_validation.json'), validate_test_against_scenario_context(test_case, scenario_context).to_dict())
                    kconfig_driver_name = ensure_kunit_kconfig(kconfig_path=kconfig_path, driver_c_path=driver_file_path, config_name=kunit_config_name)
                    print('Kconfig 配置完毕')
                    ensure_makefile(makefile_path=makefile_path, config_name=kunit_config_name, obj_name=test_obj_name)
                    print('Makefile 配置完毕')
                    ensure_gcov_profile(makefile_path=makefile_path, driver_object=driver_object)
                    print('GCOV 已启用')
                    iteration_history = list(generation_history)
                    frozen_tests: Set[str] = set()
                    test_file_name = os.path.basename(test_case_save_path)
                    last_patch_rejection: Dict[str, object] = {}

                    def _validation_path(label: str, iteration: int) -> str:
                        return os.path.join(function_output_dir, f'{function.name}_{label}_iteration{iteration}.json')

                    def _candidate_diff(before_code: str, candidate_code: str) -> str:
                        return ''.join(difflib.unified_diff((before_code or '').splitlines(keepends=True), (candidate_code or '').splitlines(keepends=True), fromfile=test_file_name, tofile=test_file_name))

                    def _apply_llm_candidate(stage: str, iteration: int, candidate_code: str, before_code: str) -> bool:
                        nonlocal last_patch_rejection
                        last_patch_rejection = {}
                        candidate_path = os.path.join(function_output_dir, f'{function.name}_{stage}_iteration{iteration}_candidate.c')
                        diff_path = os.path.join(function_output_dir, f'{function.name}_{stage}_iteration{iteration}.patch')
                        candidate_code = candidate_code or ''
                        if candidate_code and (not candidate_code.endswith('\n')):
                            candidate_code += '\n'
                        removed_candidate_externs = []
                        candidate_code, removed_candidate_externs = strip_flexible_extern_declarations(candidate_code, scenario_context)
                        _write_text(candidate_path, candidate_code)
                        audit_diff = _candidate_diff(before_code, candidate_code)
                        _write_text(diff_path, audit_diff)
                        if removed_candidate_externs:
                            _write_json(_validation_path(f'{stage}_candidate_normalization', iteration), {'removed_flexible_extern_declarations': removed_candidate_externs, 'reason': 'generated extern declarations outside the fixed TEST EXPORT block are forbidden'})
                        if not candidate_code.strip():
                            _write_json(_validation_path(f'{stage}_patch_rejected', iteration), {'ok': False, 'reason': 'LLM returned an empty candidate test file.', 'candidate': candidate_path, 'audit_diff': diff_path})
                            last_patch_rejection = {'stage': stage, 'iteration': iteration, 'reason': 'LLM returned an empty candidate test file.', 'candidate': candidate_path, 'audit_diff': diff_path, 'phase': 'candidate_empty'}
                            print(f'[WARN] {stage} 候选文件为空，已拒绝。')
                            return False
                        if not audit_diff.strip():
                            _write_json(_validation_path(f'{stage}_patch_rejected', iteration), {'ok': False, 'reason': 'LLM candidate is identical to the current test file; no source-code changes were produced.', 'candidate': candidate_path, 'audit_diff': diff_path})
                            last_patch_rejection = {'stage': stage, 'iteration': iteration, 'reason': 'LLM candidate is identical to the current test file; no source-code changes were produced.', 'candidate': candidate_path, 'audit_diff': diff_path, 'phase': 'candidate_no_change'}
                            print(f'[WARN] {stage} 候选文件没有产生源码变化，已拒绝。')
                            return False
                        gate_frozen_tests = set(frozen_tests)
                        if stage.startswith('coverage'):
                            gate_frozen_tests.update((item.name for item in inspect_test_source(before_code or '').test_functions))
                        gate_result = evaluate_scenario_patch(before_code, candidate_code, scenario_context, frozen_tests=sorted(gate_frozen_tests))
                        context_result = validate_test_against_scenario_context(candidate_code, scenario_context)
                        nested_function_errors = _nested_function_diagnostics(candidate_code)
                        poison_fake_pointer_errors = _poison_fake_pointer_diagnostics(candidate_code)
                        validation_report = {'candidate': {'candidate_file': candidate_path, 'audit_diff': diff_path, 'merge_mode': 'validated_full_file_overwrite', 'removed_flexible_extern_declarations': removed_candidate_externs}, 'gate': gate_result.to_dict(), 'scenario_context': context_result.to_dict(), 'nested_function_errors': nested_function_errors, 'poison_fake_pointer_errors': poison_fake_pointer_errors}
                        _write_json(_validation_path(f'{stage}_validation', iteration), validation_report)
                        hard_errors = [error for error in gate_result.hard_errors if not _is_missing_scenario_binding_error(error)]
                        if context_result.errors:
                            hard_errors.extend((error for error in context_result.errors if not _is_missing_scenario_binding_error(error)))
                        hard_errors.extend(nested_function_errors)
                        hard_errors.extend(poison_fake_pointer_errors)
                        audit = (gate_result.report or {}).get('repair_audit') or {}
                        removed_tests = audit.get('removed_tests', []) or []
                        added_tests = audit.get('added_tests', []) or []
                        is_coverage_patch = stage.startswith('coverage')
                        is_scenario_patch = stage.startswith('scenario')
                        removable_before = _removable_generated_tests(before_code, scenario_context)
                        if is_coverage_patch and removed_tests:
                            hard_errors.append(f'Coverage candidate removed existing test functions: {removed_tests}.')
                        if is_scenario_patch:
                            illegal_removed = sorted(set(removed_tests) - removable_before)
                            if illegal_removed:
                                hard_errors.append(f'Scenario candidate removed tests that were not unbound or inactive generated tests: {illegal_removed}.')
                        elif not is_coverage_patch and (removed_tests or added_tests):
                            hard_errors.append(f'Repair candidate changed the test function set; keep test names stable and modify only the implicated test bodies. removed={removed_tests}, added={added_tests}')
                        if hard_errors:
                            _write_json(_validation_path(f'{stage}_patch_rejected', iteration), {'ok': False, 'hard_errors': hard_errors, 'validation': validation_report})
                            last_patch_rejection = {'stage': stage, 'iteration': iteration, 'hard_errors': hard_errors, 'validation': validation_report, 'candidate': candidate_path, 'audit_diff': diff_path, 'phase': 'validation'}
                            print(f'[WARN] {stage} 候选文件破坏约束，已拒绝。')
                            return False
                        if gate_result.audit_warnings:
                            print(f'[AUDIT] {stage} 候选文件审计警告: {gate_result.audit_warnings}')
                        _write_text(test_case_save_path, candidate_code)
                        print(f'[INFO] {stage} 候选文件已通过硬约束检查并覆盖当前测试文件。')
                        return True

                    def _apply_llm_candidate_with_retry(stage: str, iteration: int, candidate_code: str, before_code: str, original_task_context: str) -> bool:
                        if _apply_llm_candidate(stage, iteration, candidate_code, before_code):
                            return True
                        rejected_candidate_code = candidate_code or ''
                        rejected_candidate = _candidate_diff(before_code, rejected_candidate_code)
                        for retry_index in range(1, PATCH_RETRY_LIMIT + 1):
                            rejection = json.dumps(last_patch_rejection or {}, indent=2, ensure_ascii=False)
                            retry_candidate = repair_rejected_patch(function=function, current_test_case=before_code, rejected_candidate_code=rejected_candidate_code, rejected_patch=rejected_candidate, rejection_reason=rejection, failure_stage=stage, original_task_context=original_task_context, local_or_api=local_or_api, prompt_path=prompt_path, model=model, tokenizer=tokenizer, scenario_context=scenario_context, test_file_name=test_file_name, trace_path=os.path.join(function_output_dir, f'{function.name}_{stage}_retry{retry_index}_iteration{iteration}_llm_rewrite.c'))
                            if _apply_llm_candidate(f'{stage}_retry{retry_index}', iteration, retry_candidate, before_code):
                                return True
                            rejected_candidate_code = retry_candidate or ''
                            rejected_candidate = _candidate_diff(before_code, rejected_candidate_code)
                        return False

                    def apply_llm_fix(stage: str, error_log: str):
                        print(f'{stage} 阶段失败，正在请求 LLM 修复测试用例 ...')
                        current_test_case = _read_text(test_case_save_path)
                        augmented_error_log = _augment_repair_log(error_log)
                        candidate_code = refine_test_case(function=function, current_test_case=current_test_case, error_log=_condense_log(augmented_error_log), failure_stage=stage, local_or_api=local_or_api, prompt_path=prompt_path, model=model, tokenizer=tokenizer, frozen_tests=sorted(frozen_tests), scenario_context=scenario_context, test_file_name=test_file_name, trace_path=os.path.join(function_output_dir, f'{function.name}_{stage}_iteration{iteration_count}_llm_rewrite.c'))
                        if not _apply_llm_candidate_with_retry(stage, iteration_count, candidate_code, current_test_case, augmented_error_log):
                            return False
                        print('测试用例候选文件已写入，准备重试。')
                        return True

                    def snapshot_test_case(iteration: int) -> Optional[str]:
                        snapshot_path = os.path.join(function_output_dir, f'{function.name}_iteration{iteration}.c')
                        try:
                            shutil.copyfile(test_case_save_path, snapshot_path)
                            return snapshot_path
                        except OSError as err:
                            print(f'[WARN] 复制测试文件失败: {err}')
                            return None

                    def record_history_event(stage: str, iteration: int, status: str, **details) -> None:
                        entry = {'iteration': iteration, 'stage': stage, 'status': status}
                        if details:
                            entry.update(details)
                        iteration_history.append(entry)

                    def qemu_log_path_for(iteration: int) -> str:
                        return os.path.join(function_output_dir, f'{function.name}_qemu_iteration{iteration}.log')
                    debugfs_base = os.path.splitext(os.path.join('/sys/kernel/debug/gcov', driver_file_path.lstrip('/')))[0]
                    debugfs_gcda = f'{debugfs_base}.gcda'
                    suite_results_filename = f'{suite_name}.results'
                    gcda_export_filename = f'{file_name}.gcda'
                    gcov_export_cmds = ['cd /', 'mount -t debugfs debugfs /sys/kernel/debug || true', 'mount -t 9p -o trans=virtio hostshare /mnt', 'mkdir -p /mnt/gcov_export', f'rm -f /mnt/gcov_export/{gcda_export_filename} /mnt/gcov_export/{suite_results_filename}', f'cat {debugfs_gcda} > /mnt/gcov_export/{gcda_export_filename}', 'sync']
                    suite_results_host_path = os.path.join(host_share_path, 'gcov_export', suite_results_filename)
                    gcda_host_path = os.path.join(host_share_path, 'gcov_export', gcda_export_filename)
                    latest_fixed_manifest_path: Optional[str] = None
                    latest_mutation_ready_manifest_path: Optional[str] = None
                    best_metrics_checkpoint: Optional[Dict[str, object]] = None
                    best_metrics_score: Optional[Tuple[float, float, int, int]] = None

                    def _checkpoint_score(kunit_summary: Dict, coverage_summary: Dict, iteration: int) -> Tuple[float, float, int, int]:
                        pass_rate = float(kunit_summary.get('pass_rate', 0.0) or 0.0)
                        coverage_score = float(coverage_summary.get('line_percent', 0.0) or 0.0) + float(coverage_summary.get('branch_percent', 0.0) or 0.0)
                        passed_count = int(kunit_summary.get('tests_passed_count', len(kunit_summary.get('passed', []) or [])) or 0)
                        return (pass_rate, coverage_score, passed_count, iteration)

                    def remember_metrics_checkpoint(kunit_summary: Dict, coverage_summary: Dict, iteration: int, failure_reasons, fixed_manifest_path: Optional[str], mutation_manifest_path: Optional[str]) -> None:
                        nonlocal best_metrics_checkpoint
                        nonlocal best_metrics_score
                        score = _checkpoint_score(kunit_summary, coverage_summary, iteration)
                        if best_metrics_score is not None and score < best_metrics_score:
                            return
                        best_metrics_score = score
                        best_metrics_checkpoint = {'iteration': iteration, 'kunit': kunit_summary, 'test_metrics': test_metrics_from_kunit(kunit_summary), 'coverage': coverage_summary, 'failing_tests': kunit_summary.get('failed', []), 'failure_reasons': failure_reasons, 'latest_fixed_tests_manifest': fixed_manifest_path, 'mutation_ready_manifest': mutation_manifest_path, 'score': {'pass_rate': score[0], 'line_plus_branch_coverage': score[1], 'passed_tests': score[2]}}
                    prior_fixed_manifest_path = os.path.join(function_output_dir, 'latest_fixed_tests_manifest.json')
                    if os.path.exists(prior_fixed_manifest_path):
                        try:
                            with open(prior_fixed_manifest_path, 'r', encoding='utf-8') as f:
                                prior_fixed_manifest = json.load(f)
                            prior_kunit = prior_fixed_manifest.get('kunit', {}) or {}
                            prior_coverage = prior_fixed_manifest.get('coverage', {}) or {}
                            if prior_kunit.get('tests') or prior_coverage.get('blocks_percent', 0.0):
                                prior_mutation_manifest_path = os.path.join(function_output_dir, 'mutation_ready.json') if os.path.exists(os.path.join(function_output_dir, 'mutation_ready.json')) else None
                                remember_metrics_checkpoint(prior_kunit, prior_coverage, int(prior_fixed_manifest.get('iteration', 0) or 0), 'prior latest_fixed_tests_manifest from an earlier run', prior_fixed_manifest_path, prior_mutation_manifest_path)
                                if best_metrics_checkpoint is not None:
                                    best_metrics_checkpoint['source'] = 'prior_latest_fixed_tests_manifest'
                        except (OSError, json.JSONDecodeError, TypeError, ValueError):
                            pass

                    def _summary_payload(status: str, kunit_summary, coverage_summary, iteration, failure_reasons) -> Dict[str, object]:
                        summary_data = {'function': function.name, 'status': status, 'iteration': iteration, 'kunit': kunit_summary, 'test_metrics': test_metrics_from_kunit(kunit_summary), 'coverage': coverage_summary, 'failing_tests': kunit_summary.get('failed', []), 'failure_reasons': failure_reasons, 'scenario_registry_version': ((scenario_context or {}).get('scenario_registry') or {}).get('version') if isinstance(scenario_context, dict) else None, 'latest_fixed_tests_manifest': latest_fixed_manifest_path, 'mutation_ready_manifest': latest_mutation_ready_manifest_path, 'history': iteration_history}
                        summary_data.update(_timing_payload(function_started_at, function_start_monotonic))
                        if best_metrics_checkpoint:
                            summary_data['best_metrics_checkpoint'] = best_metrics_checkpoint
                        return summary_data

                    def write_summary(status: str, kunit_summary, coverage_summary, iteration, failure_reasons):
                        nonlocal function_summary_data
                        nonlocal latest_fixed_manifest_path
                        nonlocal latest_mutation_ready_manifest_path
                        summary_path = os.path.join(function_output_dir, 'summary.json')
                        summary_data = _summary_payload(status, kunit_summary, coverage_summary, iteration, failure_reasons)
                        with open(summary_path, 'w') as f:
                            json.dump(summary_data, f, indent=2)
                        function_summary_data = summary_data
                        print(f'[RESULT] 汇总信息保存至 {summary_path}')
                        return summary_data

                    def terminal_failure_summary(status: str, iteration: int, stage: str, failure_reasons):
                        nonlocal function_summary_data
                        summary_path = os.path.join(function_output_dir, 'summary.json')
                        if best_metrics_checkpoint:
                            summary_data = _summary_payload(f'{status}_after_metrics', best_metrics_checkpoint.get('kunit', empty_kunit_summary()), best_metrics_checkpoint.get('coverage', empty_coverage_summary()), iteration, best_metrics_checkpoint.get('failure_reasons', ''))
                            summary_data['metrics_iteration'] = best_metrics_checkpoint.get('iteration')
                            summary_data['terminal_status'] = status
                            summary_data['terminal_stage'] = stage
                            summary_data['terminal_iteration'] = iteration
                            summary_data['terminal_failure_reasons'] = failure_reasons
                            summary_data['latest_fixed_tests_manifest'] = best_metrics_checkpoint.get('latest_fixed_tests_manifest')
                            summary_data['mutation_ready_manifest'] = best_metrics_checkpoint.get('mutation_ready_manifest')
                        else:
                            summary_data = {'function': function.name, 'status': status, 'stage': stage, 'iteration': iteration, 'kunit': empty_kunit_summary(), 'test_metrics': test_metrics_from_kunit(empty_kunit_summary()), 'coverage': empty_coverage_summary(), 'failing_tests': [], 'failure_reasons': failure_reasons or {}, 'history': iteration_history}
                            summary_data.update(_timing_payload(function_started_at, function_start_monotonic))
                        with open(summary_path, 'w') as f:
                            json.dump(summary_data, f, indent=2)
                        function_summary_data = summary_data
                        if best_metrics_checkpoint:
                            print(f"[RESULT] {stage} 阶段最终失败，但保留第 {best_metrics_checkpoint.get('iteration')} 轮可用指标。详情见 {summary_path}")
                        else:
                            print(f'[RESULT] {stage} 阶段失败，跳过该函数。详情见 {summary_path}')
                        return summary_data

                    def build_failed_summary(iteration, failure_reasons=''):
                        return terminal_failure_summary('build_failed', iteration, 'build', failure_reasons)

                    def repair_rejected_summary(iteration, stage, reason):
                        nonlocal function_summary_data
                        summary_path = os.path.join(function_output_dir, 'summary.json')
                        if best_metrics_checkpoint:
                            summary_data = _summary_payload('repair_rejected_after_metrics', best_metrics_checkpoint.get('kunit', empty_kunit_summary()), best_metrics_checkpoint.get('coverage', empty_coverage_summary()), iteration, best_metrics_checkpoint.get('failure_reasons', ''))
                            summary_data['metrics_iteration'] = best_metrics_checkpoint.get('iteration')
                            summary_data['terminal_status'] = 'repair_rejected'
                            summary_data['terminal_stage'] = stage
                            summary_data['terminal_iteration'] = iteration
                            summary_data['terminal_failure_reasons'] = reason
                            summary_data['latest_fixed_tests_manifest'] = best_metrics_checkpoint.get('latest_fixed_tests_manifest')
                            summary_data['mutation_ready_manifest'] = best_metrics_checkpoint.get('mutation_ready_manifest')
                        else:
                            summary_data = {'function': function.name, 'status': 'repair_rejected', 'stage': stage, 'iteration': iteration, 'reason': reason, 'kunit': empty_kunit_summary(), 'test_metrics': test_metrics_from_kunit(empty_kunit_summary()), 'coverage': empty_coverage_summary(), 'failing_tests': [], 'failure_reasons': {}, 'history': iteration_history}
                            summary_data.update(_timing_payload(function_started_at, function_start_monotonic))
                        with open(summary_path, 'w') as f:
                            json.dump(summary_data, f, indent=2)
                        function_summary_data = summary_data
                        print(f'[RESULT] 修复候选文件被拒绝，跳过该函数。详情见 {summary_path}')
                        return summary_data
                    driver_configured = False

                    def run_build_step():
                        nonlocal driver_configured
                        if not driver_configured:
                            enable_driver_and_kunit_test(buildroot_dir=buildroot_dir_path, linux_dir=linux_kernel_path, driver_config=kconfig_driver_name, driver_kunit_config=f"{file_name.upper().replace('-', '_')}_KUNIT_TEST", driver_kconfig_path=kconfig_path)
                            driver_configured = True
                        else:
                            rebuild_kernel_buildroot(buildroot_dir_path)
                    while iteration_count <= MAX_FIX_ATTEMPTS:
                        current_prebuild_source = _read_text(test_case_save_path)
                        normalized_prebuild_source, removed_prebuild_externs = strip_flexible_extern_declarations(current_prebuild_source, scenario_context)
                        if removed_prebuild_externs and normalized_prebuild_source != current_prebuild_source:
                            _write_text(test_case_save_path, normalized_prebuild_source)
                            _write_json(_validation_path('scenario_prebuild_normalization', iteration_count), {'removed_flexible_extern_declarations': removed_prebuild_externs, 'reason': 'removed repeated fixed extern declarations before scenario validation'})
                            current_prebuild_source = normalized_prebuild_source
                        boundary_fake_normalized_source, boundary_fake_specs, boundary_fake_actions = _normalize_boundary_hook_fake_contract(current_prebuild_source, scenario_context, linux_kernel_path)
                        if boundary_fake_actions and boundary_fake_normalized_source != current_prebuild_source:
                            _write_text(test_case_save_path, boundary_fake_normalized_source)
                            _write_json(_validation_path('boundary_fake_contract_normalization', iteration_count), {'actions': boundary_fake_actions, 'specs': boundary_fake_specs, 'reason': 'normalized hook-installed direct-boundary fakes to framework-owned canonical fake names before scenario validation/build'})
                            record_history_event('scenario', iteration_count, 'boundary_fake_contract_normalized', actions=boundary_fake_actions)
                            current_prebuild_source = boundary_fake_normalized_source
                        prebuild_scenario_validation = validate_test_against_scenario_context(current_prebuild_source, scenario_context)
                        prebuild_blocking_errors = blocking_scenario_static_errors([error for error in prebuild_scenario_validation.errors if error])
                        prebuild_audit_findings = nonblocking_scenario_static_findings([error for error in prebuild_scenario_validation.errors if error])
                        if prebuild_scenario_validation.errors or prebuild_scenario_validation.warnings:
                            _write_json(_validation_path('scenario_prebuild_validation', iteration_count), prebuild_scenario_validation.to_dict())
                        if prebuild_audit_findings:
                            record_history_event('scenario', iteration_count, 'audit_before_build', findings=prebuild_audit_findings)
                        if prebuild_blocking_errors:
                            record_history_event('scenario', iteration_count, 'blocking_static_issue_before_build', errors=prebuild_blocking_errors)
                            snapshot_test_case(iteration_count)
                            if iteration_count >= MAX_FIX_ATTEMPTS:
                                write_summary('scenario_static_blocked', empty_kunit_summary(), empty_coverage_summary(), iteration_count, '\n'.join(prebuild_blocking_errors))
                                break
                            if not apply_llm_fix('scenario', 'Blocking static scenario issues before build:\n' + '\n'.join(prebuild_blocking_errors)):
                                record_history_event('repair', iteration_count, 'patch_rejected', repair_stage='scenario', rejection=last_patch_rejection)
                                iteration_count += 1
                                continue
                            iteration_count += 1
                            continue
                        try:
                            run_build_step()
                            record_history_event('build', iteration_count, 'passed')
                        except subprocess.CalledProcessError as exc:
                            error_log = _collect_process_error(exc)
                            log_file = _save_error_log(function_output_dir, function.name, 'build', iteration_count, error_log)
                            summary_text = _extract_error_summary(error_log)
                            summary_path = os.path.join(function_output_dir, f'{function.name}_build_iteration{iteration_count}_summary.log')
                            _write_text(summary_path, summary_text)
                            record_history_event('build', iteration_count, 'failed', log_file=log_file, summary_log=summary_path)
                            snapshot_test_case(iteration_count)
                            if iteration_count < MAX_FIX_ATTEMPTS and True:
                                current_source_for_auto_repair = _read_text(test_case_save_path)
                                auto_repaired_source, inserted_fake_specs = _auto_insert_missing_boundary_fakes(current_source_for_auto_repair, summary_text, scenario_context, linux_kernel_path)
                                auto_repaired_source, auto_repairs = _auto_repair_hook_fake_signatures(auto_repaired_source, summary_text)
                                auto_actions = []
                                if inserted_fake_specs:
                                    auto_actions.append({'kind': 'insert_missing_boundary_fakes', 'specs': inserted_fake_specs})
                                if auto_repairs:
                                    auto_actions.append({'kind': 'repair_hook_fake_signatures', 'repairs': auto_repairs})
                                if auto_actions and auto_repaired_source != current_source_for_auto_repair:
                                    auto_diff = _candidate_diff(current_source_for_auto_repair, auto_repaired_source)
                                    auto_repair_path = os.path.join(function_output_dir, f'{function.name}_build_auto_repair_iteration{iteration_count}.json')
                                    _write_json(auto_repair_path, {'stage': 'build', 'iteration': iteration_count, 'actions': auto_actions, 'diff': auto_diff, 'reason': 'compiler-reported missing boundary fake symbols or hook setter function pointer type'})
                                    _write_text(test_case_save_path, auto_repaired_source)
                                    record_history_event('build', iteration_count, 'auto_repaired_boundary_fakes', actions=auto_actions, repair_log=auto_repair_path)
                                    iteration_count += 1
                                    continue
                            if iteration_count >= MAX_FIX_ATTEMPTS:
                                build_failed_summary(iteration_count, summary_text)
                                break
                            if not apply_llm_fix('build', summary_text):
                                record_history_event('repair', iteration_count, 'patch_rejected', repair_stage='build', rejection=last_patch_rejection)
                                iteration_count += 1
                                continue
                            iteration_count += 1
                            continue
                        print('Qemu 配置完毕')
                        coverage_summary = empty_coverage_summary()
                        kunit_summary = empty_kunit_summary()
                        failure_reason_text = ''
                        current_qemu_log = qemu_log_path_for(iteration_count)
                        try:
                            _remove_if_exists(suite_results_host_path)
                            _remove_if_exists(gcda_host_path)
                            run_qemu_direct(buildroot_dir=buildroot_dir_path, log_path=current_qemu_log, commands=list(gcov_export_cmds), extra_boot_args=[f'kunit.filter_glob={suite_name}.*'])
                            print('Qemu 测试完毕')
                            record_history_event('qemu', iteration_count, 'passed', log_file=current_qemu_log)
                        except QemuRunError as qemu_error:
                            log_text = qemu_error.log or _read_text(current_qemu_log)
                            log_file = _save_error_log(function_output_dir, function.name, 'qemu', iteration_count, log_text)
                            record_history_event('qemu', iteration_count, 'failed', log_file=log_file)
                            snapshot_test_case(iteration_count)
                            if iteration_count >= MAX_FIX_ATTEMPTS:
                                terminal_failure_summary('qemu_failed', iteration_count, 'qemu', _condense_log(log_text))
                                break
                            if not apply_llm_fix('qemu', _condense_log(log_text)):
                                record_history_event('repair', iteration_count, 'patch_rejected', repair_stage='qemu', rejection=last_patch_rejection)
                                iteration_count += 1
                                continue
                            iteration_count += 1
                            continue
                        coverage_report_path: Optional[str] = None
                        try:
                            coverage_report_path = os.path.join(function_output_dir, f'{function.name}_coverage_iteration{iteration_count}.json')
                            collect_gcov_results(driver_c_path=driver_file_path, driver_dir_path=driver_dir_path, host_share_path=host_share_path, linux_dir=linux_kernel_path, buildroot_dir=buildroot_dir_path, coverage_output_path=coverage_report_path)
                            driver_rel_path_cov = os.path.relpath(driver_file_path, linux_kernel_path)
                            coverage_start_line, coverage_end_line, line_range_source = _current_function_line_range(parser, driver_file_path, function.name, function.start_line, function.end_line)
                            coverage_summary = summarize_function_coverage(coverage_json_path=coverage_report_path, driver_rel_path=driver_rel_path_cov, function_name=function.name, start_line=coverage_start_line, end_line=coverage_end_line)
                            coverage_summary['line_range_source'] = line_range_source
                            coverage_summary['start_line'] = coverage_start_line
                            coverage_summary['end_line'] = coverage_end_line
                            coverage_summary['coverage_report'] = coverage_report_path
                            print('[GCOV] {name} - blocks: {blocks:.1f}%, lines: {lines:.1f}% ({hit}/{total}), branches: {branches:.1f}% ({bhit}/{btotal})'.format(name=function.name, blocks=coverage_summary['blocks_percent'], lines=coverage_summary['line_percent'], hit=coverage_summary['line_hit'], total=coverage_summary['line_total'], branches=coverage_summary['branch_percent'], bhit=coverage_summary['branch_hit'], btotal=coverage_summary['branch_total']))
                        except (FileNotFoundError, subprocess.CalledProcessError, ValueError) as err:
                            print(f'[GCOV] 覆盖率收集失败: {err}')
                        suite_results_text = ''
                        if os.path.exists(suite_results_host_path):
                            try:
                                suite_results_text = _read_text(suite_results_host_path)
                            except OSError as err:
                                print(f'[WARN] 读取 KUnit 结果失败: {err}')
                        qemu_log_text = _read_optional_text(current_qemu_log)
                        target_qemu_log_text = suite_log_block(qemu_log_text, suite_name)
                        if suite_results_text:
                            kunit_summary = parse_kunit_results_text(suite_results_text, suite_name=suite_name)
                        else:
                            kunit_summary = parse_kunit_results_file(current_qemu_log, suite_name=suite_name)
                        failing_tests = kunit_summary.get('failed', [])
                        kunit_pass_ok = bool(kunit_summary.get('overall_passed'))
                        failure_reason_text = ''
                        if not kunit_pass_ok:
                            if suite_results_text:
                                failure_reason_text = extract_failure_reason_from_text(suite_results_text)
                            if not failure_reason_text:
                                failure_reason_text = extract_failure_reason_from_log(qemu_log_text, suite_name)
                        current_test_source = _read_text(test_case_save_path)
                        scenario_runtime_status = evaluate_scenario_runtime_status(current_test_source, scenario_context, kunit_summary, buildable=True, iteration=iteration_count, max_attempts=MAX_FIX_ATTEMPTS, runtime_log='\n'.join([target_qemu_log_text or '', suite_results_text or '']))
                        _write_json(os.path.join(function_output_dir, f'{function.name}_scenario_status_iteration{iteration_count}.json'), scenario_runtime_status)
                        snapshot_path = snapshot_test_case(iteration_count)
                        scenario_bindings = []
                        scenario_complete = True
                        scenario_bindings = scenario_test_bindings(current_test_source, scenario_context, scenario_runtime_status, kunit_summary)
                        scenario_complete = scenario_status_complete(scenario_runtime_status)
                        scenario_total = int(scenario_runtime_status.get('total_scenarios', 0) or 0)
                        kunit_results_available = bool(kunit_summary.get('tests'))
                        metrics_ready_now = bool(kunit_results_available)
                        passed_tests = [item for item in kunit_summary.get('passed', []) or [] if isinstance(item, str)]
                        failed_tests = [item for item in kunit_summary.get('failed', []) or [] if isinstance(item, str)]
                        mutation_candidate_tests, unrealizable_but_kunit_passed_tests = _mutation_candidate_tests_from_bindings(passed_tests, scenario_bindings, scenario_total)
                        mutation_candidate_tests = passed_tests
                        unrealizable_but_kunit_passed_tests = []
                        mutation_ready_tests, weak_oracle_tests = _mutation_ready_passed_tests(current_test_source, mutation_candidate_tests)
                        mutation_test_snapshot_path = None
                        if mutation_ready_tests:
                            mutation_test_snapshot_path = os.path.join(function_output_dir, f'{function.name}_mutation_tests_iteration{iteration_count}.c')
                            _write_passing_test_subset_snapshot(source_test_path=test_case_save_path, output_path=mutation_test_snapshot_path, passed_tests=mutation_ready_tests)
                        mutation_ready_now = bool(mutation_ready_tests)
                        fixed_manifest_path = os.path.join(function_output_dir, f'{function.name}_fixed_tests_iteration{iteration_count}.json')
                        fixed_manifest = _write_fixed_tests_manifest(fixed_manifest_path, function_name=function.name, driver_path=driver_file_path, test_file_path=test_case_save_path, test_snapshot_path=snapshot_path, suite_name=suite_name, iteration=iteration_count, scenario_context_path=scenario_context_path, qemu_log_path=current_qemu_log, suite_results_path=suite_results_host_path, kunit_summary=kunit_summary, coverage_summary=coverage_summary, scenario_runtime_status=scenario_runtime_status, scenario_bindings=scenario_bindings, mock_bindings=mock_bindings_from_test(current_test_source, scenario_context), mutation_ready=mutation_ready_now, mutation_ready_tests=mutation_ready_tests, runnable_passed_tests=passed_tests, unrealizable_but_kunit_passed_tests=unrealizable_but_kunit_passed_tests, excluded_failed_tests=failed_tests, weak_oracle_tests=weak_oracle_tests, mutation_test_snapshot_path=mutation_test_snapshot_path)
                        latest_fixed_manifest_path = fixed_manifest_path
                        _write_json(os.path.join(function_output_dir, 'latest_fixed_tests_manifest.json'), fixed_manifest)
                        current_mutation_ready_manifest_path = None
                        if mutation_ready_now:
                            latest_mutation_ready_manifest_path = os.path.join(function_output_dir, 'mutation_ready.json')
                            current_mutation_ready_manifest_path = latest_mutation_ready_manifest_path
                            _write_json(latest_mutation_ready_manifest_path, fixed_manifest)
                        remember_metrics_checkpoint(kunit_summary, coverage_summary, iteration_count, failure_reason_text, latest_fixed_manifest_path, current_mutation_ready_manifest_path)
                        record_history_event('results', iteration_count, 'recorded', kunit=kunit_summary, coverage=coverage_summary, failing_tests=failing_tests, failure_reasons=failure_reason_text, scenario_status=scenario_runtime_status, scenario_test_bindings=scenario_bindings, metrics_ready=metrics_ready_now, fixed_tests_manifest=fixed_manifest_path, mutation_ready=mutation_ready_now, test_case=snapshot_path, qemu_log=current_qemu_log)
                        frozen_tests.update(stable_tests_from_scenario_status(scenario_runtime_status))
                        coverage_ok = coverage_summary.get('line_percent', 0.0) >= TARGET_COVERAGE_PERCENT and coverage_summary.get('branch_percent', 0.0) >= TARGET_COVERAGE_PERCENT
                        coverage_feedback_available = bool(kunit_results_available and (coverage_summary.get('line_total', 0) or coverage_summary.get('branch_total', 0) or coverage_summary.get('missed_lines') or coverage_summary.get('missed_branches')))
                        if coverage_ok and metrics_ready_now and kunit_pass_ok:
                            write_summary('completed', kunit_summary, coverage_summary, iteration_count, failure_reason_text)
                            break
                        if iteration_count >= MAX_FIX_ATTEMPTS:
                            if metrics_ready_now and coverage_ok:
                                status = 'completed_with_test_failures' if not kunit_pass_ok else 'completed'
                            else:
                                status = 'iteration_exhausted_with_metrics' if metrics_ready_now else 'iteration_exhausted'
                            write_summary(status, kunit_summary, coverage_summary, iteration_count, failure_reason_text)
                            break
                        current_test_case = _read_text(test_case_save_path)
                        updated = False
                        if not kunit_results_available:
                            repair_log = '\n'.join([failure_reason_text or 'KUnit did not report any test result.', _condense_log(suite_results_text or qemu_log_text)])
                            if apply_llm_fix('kunit_results', repair_log):
                                updated = True
                                iteration_count += 1
                                continue
                            else:
                                record_history_event('repair', iteration_count, 'patch_rejected', repair_stage='kunit_results', rejection=last_patch_rejection)
                                iteration_count += 1
                                continue
                        if kunit_results_available and failing_tests:
                            augmented_failure_reasons = _augment_repair_log(failure_reason_text or '(no details)')
                            candidate_code = fix_failed_tests(function=function, current_test_case=current_test_case, failing_tests=failing_tests, failure_reasons=augmented_failure_reasons, frozen_tests=sorted(frozen_tests) if frozen_tests else [], local_or_api=local_or_api, prompt_path=prompt_path, model=model, tokenizer=tokenizer, scenario_context=scenario_context, test_file_name=test_file_name, trace_path=os.path.join(function_output_dir, f'{function.name}_failed_tests_iteration{iteration_count}_llm_rewrite.c'))
                            if _apply_llm_candidate_with_retry('failed_tests', iteration_count, candidate_code, current_test_case, augmented_failure_reasons):
                                updated = True
                                iteration_count += 1
                                continue
                            record_history_event('repair', iteration_count, 'pass_rate_patch_rejected_continue_to_coverage', repair_stage='failed_tests', rejection=last_patch_rejection)
                            if coverage_ok:
                                write_summary('completed_with_test_failures', kunit_summary, coverage_summary, iteration_count, failure_reason_text)
                                break
                        if not failing_tests and (not scenario_complete):
                            scenario_repair_log = scenario_status_repair_log(scenario_runtime_status)
                            if apply_llm_fix('scenario', scenario_repair_log):
                                updated = True
                                iteration_count += 1
                                continue
                            else:
                                record_history_event('repair', iteration_count, 'patch_rejected', repair_stage='scenario', rejection=last_patch_rejection)
                                iteration_count += 1
                                continue
                        if not coverage_ok and coverage_feedback_available:
                            missed_lines = coverage_summary.get('missed_lines', [])
                            missed_branches = coverage_summary.get('missed_branches', [])
                            added_contracts = []
                            added_ids = set()
                            scenario_context, added_contracts = expand_scenario_context_for_coverage(scenario_context, missed_lines=missed_lines, missed_branches=missed_branches)
                            coverage_targets = scenario_context.get('coverage_targets', []) or []
                            if added_contracts or coverage_targets:
                                _write_json(os.path.join(function_output_dir, f'{function.name}_coverage_added_scenarios_iteration{iteration_count}.json'), {'added_contracts': added_contracts, 'coverage_targets': coverage_targets, 'scenario_registry': scenario_context.get('scenario_registry', {})})
                                save_scenario_context(scenario_context, scenario_context_path)
                            added_ids = {contract.get('scenario_id') for contract in added_contracts if isinstance(contract, dict) and contract.get('scenario_id')}
                            missed_line_details = _format_line_snippets(instrumented_driver_path, missed_lines)
                            branch_lines = sorted({b.get('line') for b in missed_branches if isinstance(b, dict) and b.get('line')})
                            branch_snippet_map = {}
                            if branch_lines:
                                for snippet in _format_line_snippets(instrumented_driver_path, branch_lines):
                                    parts = snippet.split(': ', 1)
                                    line_no = None
                                    try:
                                        line_no = int(parts[0]) if parts else None
                                    except ValueError:
                                        line_no = None
                                    if line_no is not None:
                                        branch_snippet_map[line_no] = parts[1] if len(parts) > 1 else snippet
                            missed_branch_details = []
                            for b in missed_branches:
                                if not isinstance(b, dict):
                                    continue
                                line_no = b.get('line')
                                branch_idx = b.get('branch_index')
                                branch_desc = ''
                                if branch_idx == 0:
                                    branch_desc = 'false-path not covered (branch_index 0)'
                                elif branch_idx == 1:
                                    branch_desc = 'true-path not covered (branch_index 1)'
                                desc = f'{branch_desc}'
                                code = branch_snippet_map.get(line_no)
                                if code:
                                    desc += f' => {code}\n'
                                missed_branch_details.append(desc)
                            coverage_before_code = _read_text(test_case_save_path)
                            candidate_code = extend_coverage_tests(function=function, current_test_case=coverage_before_code, missed_line_details=missed_line_details, missed_branch_details=missed_branch_details, local_or_api=local_or_api, prompt_path=prompt_path, model=model, tokenizer=tokenizer, scenario_context=scenario_context, test_file_name=test_file_name, trace_path=os.path.join(function_output_dir, f'{function.name}_coverage_iteration{iteration_count}_llm_rewrite.c'))
                            if _apply_llm_candidate_with_retry('coverage', iteration_count, candidate_code, coverage_before_code, '\n'.join((missed_line_details or []) + (missed_branch_details or []))):
                                updated = True
                            elif updated:
                                record_history_event('coverage', iteration_count, 'patch_rejected_after_prior_update', rejection=last_patch_rejection)
                            else:
                                record_history_event('repair', iteration_count, 'patch_rejected', repair_stage='coverage', rejection=last_patch_rejection)
                                iteration_count += 1
                                continue
                        if updated:
                            iteration_count += 1
                            continue
                        write_summary('iteration_stalled', kunit_summary, coverage_summary, iteration_count, failure_reason_text or 'No applicable repair or coverage feedback was available.')
                        break
                finally:
                    reset_driver_environment()
                if function_summary_data is not None:
                    driver_function_summaries.append(function_summary_data)
            else:
                continue
        reset_driver_environment()
        driver_summary = summarize_driver_iterations(driver_rel_path, driver_function_summaries)
        driver_summary['elapsed_seconds'] = time.time() - driver_start
        driver_summary_path = os.path.join(driver_result_dir, 'summary.json')
        with open(driver_summary_path, 'w') as f:
            json.dump(driver_summary, f, indent=2)
        print(f'[DRIVER] 汇总信息保存至 {driver_summary_path}')
if __name__ == '__main__':
    main()
