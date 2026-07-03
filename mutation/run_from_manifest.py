import argparse
import json
import os
import re
import shlex
import shutil
import sys
import time
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


UNIT_TEST_ROOT = Path(__file__).resolve().parents[1]
if str(UNIT_TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(UNIT_TEST_ROOT))

from mutation.evaluate_mutation import evaluate_mutants
from run.build import QemuRunError, enable_driver_and_kunit_test, rebuild_kernel_buildroot, run_qemu_direct
from run.config_change import derive_driver_config, ensure_kunit_kconfig, ensure_makefile
from run.gcov import ensure_gcov_profile
from verification.kunit_result_parser import filter_kunit_cases_to_tests, parse_kunit_results_text


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _resolve(base_dir: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base_dir / path)


def _load_manifest(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_manifest_artifact(manifest_path: Path, value: str, fallback_names: Optional[List[str]] = None) -> Path:
    manifest_dir = _function_dir(manifest_path)
    candidates = [
        _resolve(Path.cwd(), value),
    ]
    try:
        candidates.append(_resolve(manifest_path.resolve().parents[3], value))
    except IndexError:
        pass
    value_path = Path(value)
    candidates.append(manifest_dir / value_path.name)
    for name in fallback_names or []:
        candidates.append(manifest_dir / name)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _find_linux_dir(driver_path: Path) -> Path:
    for parent in [driver_path.parent, *driver_path.parents]:
        if (parent / ".config").exists() and (parent / "scripts" / "config").exists():
            return parent
    raise RuntimeError(f"Cannot infer Linux tree from {driver_path}")


def _infer_buildroot_dir(linux_dir: Path) -> Path:
    # Typical Buildroot path: <buildroot>/output/build/linux-custom
    if len(linux_dir.parents) >= 3:
        candidate = linux_dir.parents[2]
        if (candidate / "output").exists():
            return candidate
    raise RuntimeError(f"Cannot infer Buildroot dir from {linux_dir}")


def _function_dir(manifest_path: Path) -> Path:
    return manifest_path.resolve().parent


def _find_instrumented_driver(function_dir: Path) -> Path:
    candidates = sorted(function_dir.glob("*_instrumented.c"))
    if not candidates:
        raise FileNotFoundError(f"No *_instrumented.c found in {function_dir}")
    return candidates[0]


def _line_range_from_scenario_context(manifest: Dict, manifest_dir: Path) -> Tuple[Optional[int], Optional[int]]:
    context_path_value = manifest.get("scenario_context_path")
    if not context_path_value:
        return None, None
    context_path = _resolve(manifest_dir.parents[2] if len(manifest_dir.parents) > 2 else Path.cwd(), context_path_value)
    if not context_path.exists():
        context_path = _resolve(Path.cwd(), context_path_value)
    if not context_path.exists():
        return None, None
    try:
        context = _load_manifest(context_path)
    except (OSError, json.JSONDecodeError):
        return None, None
    target_function = manifest.get("function")
    lines = []
    for fact in context.get("source_facts", []) or []:
        if target_function and fact.get("function") != target_function:
            continue
        for key in ("start_line", "end_line"):
            value = fact.get(key)
            if isinstance(value, int) and value > 0:
                lines.append(value)
    if not lines:
        return None, None
    return min(lines), max(lines)


def _find_matching_brace_in_text(code: str, open_idx: int) -> Optional[int]:
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
        if not in_string and not in_char and code.startswith("/*", i):
            end = code.find("*/", i + 2)
            i = len(code) if end < 0 else end + 2
            continue
        if not in_string and not in_char and code.startswith("//", i):
            end = code.find("\n", i + 2)
            i = len(code) if end < 0 else end + 1
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


def _line_for_offset(code: str, offset: int) -> int:
    return code.count("\n", 0, offset) + 1


def _function_body_line_range(source_path: Path, function_name: str) -> Tuple[Optional[int], Optional[int]]:
    if not function_name or not source_path.exists():
        return None, None
    source = _read_text(source_path)
    name_pattern = re.compile(r"\b" + re.escape(function_name) + r"\s*\(")
    open_idx = -1
    for match in name_pattern.finditer(source):
        line_start = source.rfind("\n", 0, match.start()) + 1
        prefix = source[line_start:match.start()]
        if "=" in prefix or prefix.strip().startswith(("return", "if", "for", "while", "switch")):
            continue
        close_paren = source.find(")", match.end())
        if close_paren < 0:
            continue
        next_semicolon = source.find(";", close_paren)
        next_open = source.find("{", close_paren)
        if next_open < 0 or (next_semicolon >= 0 and next_semicolon < next_open):
            continue
        open_idx = next_open
        break
    if open_idx < 0:
        return None, None
    close_idx = _find_matching_brace_in_text(source, open_idx)
    if close_idx is None:
        return None, None
    return _line_for_offset(source, open_idx), _line_for_offset(source, close_idx)


def _snippet_regex(snippet: str) -> Optional[re.Pattern]:
    text = (snippet or "").strip()
    if not text:
        return None
    text = re.sub(r"\s+\[(?:branch|fallthrough)\]\s*$", "", text)
    text = text.strip()
    if not text:
        return None
    parts = re.split(r"\s+", text)
    return re.compile(r"\s+".join(re.escape(part) for part in parts), re.DOTALL)


def _lines_for_snippet(source: str, snippet: str, search_start_line: Optional[int], search_end_line: Optional[int]) -> Set[int]:
    pattern = _snippet_regex(snippet)
    if pattern is None:
        return set()
    start_offset = 0
    end_offset = len(source)
    if search_start_line is not None:
        line_offsets = [0]
        for match in re.finditer("\n", source):
            line_offsets.append(match.end())
        if 1 <= search_start_line <= len(line_offsets):
            start_offset = line_offsets[search_start_line - 1]
        if search_end_line is not None and 1 <= search_end_line < len(line_offsets):
            end_offset = line_offsets[search_end_line]
    segment = source[start_offset:end_offset]
    match = pattern.search(segment)
    if not match:
        return set()
    abs_start = start_offset + match.start()
    abs_end = start_offset + match.end()
    return set(range(_line_for_offset(source, abs_start), _line_for_offset(source, abs_end) + 1))


def _passed_scenario_mutation_lines(
    manifest: Dict,
    manifest_path: Path,
    prepared_driver_path: Path,
    target_start_line: Optional[int],
    target_end_line: Optional[int],
) -> Tuple[Set[int], List[Dict]]:
    context_value = manifest.get("scenario_context_path")
    if not context_value:
        return set(), []
    context_path = _resolve_manifest_artifact(
        manifest_path,
        context_value,
        fallback_names=["scenario_context.json"],
    )
    if not context_path.exists():
        return set(), []
    context = _load_manifest(context_path)
    target_function = manifest.get("function")
    facts = {}
    for fact in (context.get("source_facts", []) or []):
        if not isinstance(fact, dict):
            continue
        fact_id = fact.get("fact_id")
        if isinstance(fact_id, str):
            facts[fact_id] = fact
    selected_tests = {
        item for item in manifest.get("mutation_ready_tests", []) or [] if isinstance(item, str) and item
    }
    source = _read_text(prepared_driver_path)
    allowed_lines: Set[int] = set()
    used_anchors: List[Dict] = []
    for binding in manifest.get("scenario_test_bindings", []) or []:
        if not isinstance(binding, dict):
            continue
        test_function = binding.get("test_function")
        if test_function not in selected_tests:
            continue
        for fact_id in binding.get("source_anchors", []) or []:
            fact = facts.get(fact_id)
            if not fact or fact.get("function") != target_function:
                continue
            lines = _lines_for_snippet(
                source,
                str(fact.get("code", "") or ""),
                target_start_line,
                target_end_line,
            )
            if target_start_line is not None:
                lines = {line for line in lines if line >= target_start_line}
            if target_end_line is not None:
                lines = {line for line in lines if line <= target_end_line}
            if not lines:
                continue
            allowed_lines.update(lines)
            used_anchors.append(
                {
                    "test_function": test_function,
                    "scenario_id": binding.get("scenario_id"),
                    "fact_id": fact_id,
                    "fact_kind": fact.get("kind"),
                    "fact_code": fact.get("code"),
                    "instrumented_lines": sorted(lines),
                }
            )
    return allowed_lines, used_anchors


def _anchor_line_annotations(anchors: List[Dict]) -> Dict[int, List[Dict]]:
    annotations: Dict[int, List[Dict]] = {}
    for anchor in anchors:
        if not isinstance(anchor, dict):
            continue
        for line in anchor.get("instrumented_lines", []) or []:
            if isinstance(line, int):
                annotations.setdefault(line, []).append(anchor)
    return annotations


def _snapshot(paths) -> Dict[Path, Optional[str]]:
    snap: Dict[Path, Optional[str]] = {}
    for path in paths:
        snap[path] = _read_text(path) if path.exists() else None
    return snap


def _restore(snapshot: Dict[Path, Optional[str]]) -> None:
    for path, text in snapshot.items():
        if text is None:
            try:
                path.unlink()
            except FileNotFoundError:
                pass
            continue
        _write_text(path, text)


def _run_eval_command(command: str, cwd: Optional[str], timeout: Optional[int]) -> Dict:
    started = time.time()
    try:
        result = subprocess.run(
            shlex.split(command),
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "returncode": 124,
            "stdout": (exc.stdout or "")[-8000:] if isinstance(exc.stdout, str) else "",
            "stderr": (exc.stderr or "")[-8000:] if isinstance(exc.stderr, str) else "",
            "elapsed_seconds": time.time() - started,
            "timeout": True,
        }
    return {
        "returncode": result.returncode,
        "stdout": (result.stdout or "")[-8000:],
        "stderr": (result.stderr or "")[-8000:],
        "elapsed_seconds": time.time() - started,
        "timeout": False,
    }


def _ensure_kunit_kconfig_once(kconfig_path: Path, driver_path: Path, config_name: str) -> str:
    if kconfig_path.exists() and f"config {config_name}" in _read_text(kconfig_path):
        return derive_driver_config(str(driver_path))
    return ensure_kunit_kconfig(
        kconfig_path=str(kconfig_path),
        driver_c_path=str(driver_path),
        config_name=config_name,
    )


def _remove_generated_test_makefile_entries(makefile_path: Path) -> None:
    if not makefile_path.exists():
        return
    lines = _read_text(makefile_path).splitlines()
    kept = [line for line in lines if "_test_case.o" not in line]
    _write_text(makefile_path, "\n".join(kept) + ("\n" if kept else ""))


def _remove_stale_generated_test_files(driver_dir: Path, keep_test_file_path: Path) -> None:
    keep = keep_test_file_path.resolve()
    for path in driver_dir.glob("*_test_case.c"):
        if path.resolve() == keep:
            continue
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def prepare_from_manifest(manifest_path: Path, buildroot_dir_arg: Optional[str] = None) -> Dict:
    manifest = _load_manifest(manifest_path)
    manifest_dir = _function_dir(manifest_path)
    driver_path = Path(manifest["driver_path"])
    test_file_path = Path(manifest["test_file_path"])
    test_snapshot_value = manifest.get("test_snapshot_path") or manifest.get("mutation_test_snapshot_path")
    mutation_snapshot = _resolve_manifest_artifact(
        manifest_path,
        test_snapshot_value,
        fallback_names=[
            Path(str(manifest.get("mutation_test_snapshot_path", ""))).name,
            Path(str(manifest.get("test_snapshot_path", ""))).name,
        ],
    )
    instrumented_driver = _find_instrumented_driver(manifest_dir)
    linux_dir = _find_linux_dir(driver_path)
    buildroot_dir = Path(buildroot_dir_arg) if buildroot_dir_arg else _infer_buildroot_dir(linux_dir)
    driver_dir = driver_path.parent
    file_name = driver_path.stem
    config_name = f"{file_name.upper().replace('-', '_')}_KUNIT_TEST"
    kconfig_path = driver_dir / "Kconfig"
    makefile_path = driver_dir / "Makefile"
    driver_object = f"{file_name}.o"

    _remove_stale_generated_test_files(driver_dir, test_file_path)
    _remove_generated_test_makefile_entries(makefile_path)
    shutil.copyfile(instrumented_driver, driver_path)
    mutation_test_source = _read_text(mutation_snapshot)
    selected_tests = {
        item for item in manifest.get("mutation_ready_tests", []) if isinstance(item, str) and item
    }
    if selected_tests:
        mutation_test_source = filter_kunit_cases_to_tests(mutation_test_source, selected_tests)
    _write_text(test_file_path, mutation_test_source)
    driver_config = _ensure_kunit_kconfig_once(kconfig_path, driver_path, config_name)
    ensure_makefile(
        makefile_path=str(makefile_path),
        config_name=config_name,
        obj_name=test_file_path.with_suffix(".o").name,
    )
    ensure_gcov_profile(str(makefile_path), driver_object)
    enable_driver_and_kunit_test(
        buildroot_dir=str(buildroot_dir),
        linux_dir=str(linux_dir),
        driver_config=driver_config,
        driver_kunit_config=config_name,
        driver_kconfig_path=str(kconfig_path),
    )
    return {
        "manifest": manifest,
        "manifest_path": str(manifest_path),
        "driver_path": str(driver_path),
        "test_file_path": str(test_file_path),
        "linux_dir": str(linux_dir),
        "buildroot_dir": str(buildroot_dir),
        "suite_name": manifest["suite_name"],
        "driver_base": file_name,
    }


def build_only(manifest_path: Path, buildroot_dir_arg: Optional[str] = None) -> int:
    manifest = _load_manifest(manifest_path)
    linux_dir = _find_linux_dir(Path(manifest["driver_path"]))
    buildroot_dir = Path(buildroot_dir_arg) if buildroot_dir_arg else _infer_buildroot_dir(linux_dir)
    rebuild_kernel_buildroot(str(buildroot_dir))
    return 0


def run_only(manifest_path: Path, output_dir: Path, buildroot_dir_arg: Optional[str] = None) -> int:
    manifest = _load_manifest(manifest_path)
    driver_path = Path(manifest["driver_path"])
    linux_dir = _find_linux_dir(driver_path)
    buildroot_dir = Path(buildroot_dir_arg) if buildroot_dir_arg else _infer_buildroot_dir(linux_dir)
    suite_name = manifest["suite_name"]
    host_share = buildroot_dir / "qemu-share"
    suite_results_filename = f"{suite_name}.results"
    suite_results_host = host_share / "gcov_export" / suite_results_filename
    gcda_host = host_share / "gcov_export" / f"{driver_path.stem}.gcda"
    for path in (suite_results_host, gcda_host):
        try:
            path.unlink()
        except FileNotFoundError:
            pass

    debugfs_base = os.path.splitext(os.path.join("/sys/kernel/debug/gcov", str(driver_path).lstrip("/")))[0]
    commands = [
        "cd /",
        "mount -t debugfs debugfs /sys/kernel/debug || true",
        "mount -t 9p -o trans=virtio hostshare /mnt",
        "mkdir -p /mnt/gcov_export",
        f"rm -f /mnt/gcov_export/{driver_path.stem}.gcda /mnt/gcov_export/{suite_results_filename}",
        f"cat {debugfs_base}.gcda > /mnt/gcov_export/{driver_path.stem}.gcda",
        f"cat /sys/kernel/debug/kunit/{suite_name}/results > /mnt/gcov_export/{suite_results_filename}",
        "sync",
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = int(time.time())
    log_path = output_dir / f"mutation_qemu_{run_id}.log"
    try:
        run_qemu_direct(
            buildroot_dir=str(buildroot_dir),
            log_path=str(log_path),
            commands=commands,
            extra_boot_args=[f"kunit.filter_glob={suite_name}.*"],
        )
    except QemuRunError:
        return 1
    qemu_log_text = _read_text(log_path) if log_path.exists() else ""
    results_text = _read_text(suite_results_host) if suite_results_host.exists() else ""
    if not results_text.strip():
        results_text = qemu_log_text
    summary = parse_kunit_results_text(results_text, suite_name=suite_name)
    run_summary_path = output_dir / f"mutation_run_{run_id}.json"
    _write_text(
        run_summary_path,
        json.dumps(
            {
                "suite_name": suite_name,
                "qemu_log": str(log_path),
                "suite_results_host": str(suite_results_host),
                "used_qemu_log_fallback": not (
                    suite_results_host.exists() and _read_text(suite_results_host).strip()
                ),
                "kunit": summary,
            },
            indent=2,
        ),
    )
    return 0 if summary.get("overall_passed") else 1


def evaluate_from_manifest(args) -> Dict:
    manifest_path = Path(args.manifest).resolve()
    manifest = _load_manifest(manifest_path)
    driver_path = Path(manifest["driver_path"])
    linux_dir = _find_linux_dir(driver_path)
    driver_dir = driver_path.parent
    snapshots = _snapshot(
        [
            driver_path,
            Path(manifest["test_file_path"]),
            *sorted(driver_dir.glob("*_test_case.c")),
            driver_dir / "Kconfig",
            driver_dir / "Makefile",
            linux_dir / ".config",
        ]
    )
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        try:
            prepare_from_manifest(manifest_path, args.buildroot_dir)
        except subprocess.CalledProcessError as exc:
            summary = {
                "manifest_path": str(manifest_path),
                "status": "preparation_failed",
                "stage": "prepare",
                "returncode": exc.returncode,
                "cmd": exc.cmd,
                "stdout": (exc.stdout or "")[-8000:],
                "stderr": (exc.stderr or "")[-8000:],
            }
            _write_text(output_path, json.dumps(summary, indent=2))
            raise
        start_line, end_line = (args.start_line, args.end_line)
        allowed_lines = None
        passed_anchor_lines: Set[int] = set()
        used_anchor_facts: List[Dict] = []
        mutation_scope = args.mutation_scope
        if start_line is None and end_line is None and mutation_scope in {
            "target",
            "target-with-passed",
            "passed-scenarios",
        }:
            start_line, end_line = _function_body_line_range(driver_path, manifest.get("function", ""))
        if mutation_scope in {"target", "target-with-passed", "passed-scenarios"}:
            passed_anchor_lines, used_anchor_facts = _passed_scenario_mutation_lines(
                manifest,
                manifest_path,
                driver_path,
                start_line,
                end_line,
            )
        if mutation_scope == "passed-scenarios":
            allowed_lines = passed_anchor_lines
        if start_line is None and end_line is None:
            start_line, end_line = _line_range_from_scenario_context(manifest, manifest_path.resolve().parent)
        build_command = (
            f"{sys.executable} {Path(__file__).resolve()} "
            f"--manifest {manifest_path} --build-only"
        )
        run_command = (
            f"{sys.executable} {Path(__file__).resolve()} "
            f"--manifest {manifest_path} --run-only --output-dir {output_path.parent}"
        )
        if args.buildroot_dir:
            build_command += f" --buildroot-dir {args.buildroot_dir}"
            run_command += f" --buildroot-dir {args.buildroot_dir}"

        original_build = _run_eval_command(
            build_command,
            cwd=str(UNIT_TEST_ROOT.parent),
            timeout=args.timeout,
        )
        if original_build["returncode"] != 0:
            summary = {
                "manifest_path": str(manifest_path),
                "status": "original_build_failed",
                "stage": "original_build",
                "original_build": original_build,
                "mutation_ready_tests": manifest.get("mutation_ready_tests", []),
                "mutation_scope": mutation_scope,
                "line_range": {"start_line": start_line, "end_line": end_line},
            }
            _write_text(output_path, json.dumps(summary, indent=2))
            return summary

        original_run = _run_eval_command(
            run_command,
            cwd=str(UNIT_TEST_ROOT.parent),
            timeout=args.timeout,
        )
        if original_run["returncode"] != 0:
            summary = {
                "manifest_path": str(manifest_path),
                "status": "original_test_failed",
                "stage": "original_run",
                "original_build": original_build,
                "original_run": original_run,
                "mutation_ready_tests": manifest.get("mutation_ready_tests", []),
                "mutation_scope": mutation_scope,
                "line_range": {"start_line": start_line, "end_line": end_line},
            }
            _write_text(output_path, json.dumps(summary, indent=2))
            return summary

        summary = evaluate_mutants(
            driver_c_path=str(driver_path),
            build_command=build_command,
            run_command=run_command,
            output_path=str(output_path),
            command_cwd=str(UNIT_TEST_ROOT.parent),
            start_line=start_line,
            end_line=end_line,
            max_mutants=args.max_mutants,
            max_mutants_per_group=args.max_mutants_per_group,
            timeout=args.timeout,
            allowed_lines=allowed_lines,
            line_annotations=_anchor_line_annotations(used_anchor_facts),
            replacement_policy=args.replacement_policy,
        )
        summary["manifest_path"] = str(manifest_path)
        summary["original_build"] = original_build
        summary["original_run"] = original_run
        summary["mutation_ready_tests"] = manifest.get("mutation_ready_tests", [])
        summary["mutation_scope"] = mutation_scope
        summary["passed_scenario_anchor_lines"] = sorted(passed_anchor_lines)
        summary["passed_scenario_anchor_facts"] = used_anchor_facts
        summary["line_range"] = {"start_line": start_line, "end_line": end_line}
        _write_text(output_path, json.dumps(summary, indent=2))
        return summary
    finally:
        if not args.keep_prepared_tree:
            _restore(snapshots)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run mutation evaluation from a RACA mutation_ready.json manifest.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", default="mutation_summary.json")
    parser.add_argument("--output-dir", default="mutation_eval")
    parser.add_argument("--buildroot-dir", default=None)
    parser.add_argument("--max-mutants", type=int, default=None)
    parser.add_argument(
        "--max-mutants-per-group",
        type=int,
        default=None,
        help="Deterministically keep at most this many mutants per operator group for each target function.",
    )
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument(
        "--replacement-policy",
        choices=("representative", "exhaustive"),
        default="representative",
        help=(
            "representative creates one deterministic replacement per mutation site and operator group; "
            "exhaustive expands every legal replacement in the operator family."
        ),
    )
    parser.add_argument("--start-line", type=int, default=None)
    parser.add_argument("--end-line", type=int, default=None)
    parser.add_argument(
        "--mutation-scope",
        choices=("target-with-passed", "target", "passed-scenarios", "source-facts"),
        default="target-with-passed",
        help=(
            "target-with-passed mutates the whole target function and also reports the passed-scenario "
            "subset from the same mutant set; passed-scenarios restricts mutation to passing-test anchors; "
            "source-facts uses scenario source fact line range."
        ),
    )
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--run-only", action="store_true")
    parser.add_argument("--keep-prepared-tree", action="store_true")
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    if args.build_only:
        raise SystemExit(build_only(manifest_path, args.buildroot_dir))
    if args.run_only:
        raise SystemExit(run_only(manifest_path, Path(args.output_dir), args.buildroot_dir))

    summary = evaluate_from_manifest(args)
    subset = summary.get("passed_scenarios_subset", {})
    site_coverage = summary.get("passed_scenario_mutation_site_coverage", {}) or {}
    print(f"[MUTATION] replacement_policy={summary.get('replacement_policy', args.replacement_policy)}")
    print("[MUTATION] target by operator group:")
    for group, metrics in (summary.get("operator_group_summary") or {}).items():
        print(
            "  - {group}: total={total} valid={valid} killed={killed} survived={survived} score={score:.2f}%".format(
                group=group,
                total=metrics.get("total_mutants", 0),
                valid=metrics.get("valid_mutants", 0),
                killed=metrics.get("killed_mutants", 0),
                survived=metrics.get("survived_mutants", 0),
                score=metrics.get("mutation_score", 0.0),
            )
        )
    print("[MUTATION] passed-scenario subset by operator group:")
    for group, metrics in (subset.get("operator_group_summary") or {}).items():
        print(
            "  - {group}: total={total} valid={valid} killed={killed} survived={survived} score={score:.2f}%".format(
                group=group,
                total=metrics.get("total_mutants", 0),
                valid=metrics.get("valid_mutants", 0),
                killed=metrics.get("killed_mutants", 0),
                survived=metrics.get("survived_mutants", 0),
                score=metrics.get("mutation_score", 0.0),
            )
        )
    print(
        "[MUTATION] passed-scenario mutation-site coverage: {covered}/{total} ({percent:.2f}%)".format(
            covered=site_coverage.get("passed_scenario_related_mutants", 0),
            total=site_coverage.get("target_mutants", 0),
            percent=site_coverage.get("coverage_percent", 0.0),
        )
    )
    print(
        "[MUTATION] target total={total} valid={valid} killed={killed} survived={survived} score={score:.2f}% | "
        "passed-scenario subset valid={subset_valid} killed={subset_killed} survived={subset_survived} score={subset_score:.2f}%".format(
            total=summary["total_mutants"],
            valid=summary["valid_mutants"],
            killed=summary["killed_mutants"],
            survived=summary["survived_mutants"],
            score=summary["mutation_score"],
            subset_valid=subset.get("valid_mutants", 0),
            subset_killed=subset.get("killed_mutants", 0),
            subset_survived=subset.get("survived_mutants", 0),
            subset_score=subset.get("mutation_score", 0.0),
        )
    )


if __name__ == "__main__":
    main()
