import os
import json
import shutil
import subprocess
from typing import Dict, Optional, List


def ensure_gcov_profile(makefile_path: str, driver_object: str) -> None:
    """
    Ensure the Makefile enables GCOV profiling for the given driver object.
    Adds a line like `GCOV_PROFILE_<driver_object> := y` if missing.
    """
    directive = f"GCOV_PROFILE_{driver_object} := y"

    if not os.path.exists(makefile_path):
        raise FileNotFoundError(f"Makefile not found: {makefile_path}")

    with open(makefile_path, "r", encoding="utf-8", errors="ignore") as f:
        contents = f.read()
        if directive in contents:
            return

    with open(makefile_path, "a", encoding="utf-8") as f:
        f.write("\n" + directive + "\n")


def move_gcov_data_from_share(
    driver_base: str,
    driver_dir_path: str,
    host_share_path: str,
    export_subdir: str = "gcov_export",
) -> str:
    """
    Move the gcda file exported from QEMU (via the shared folder) into the driver directory.
    Returns the destination path.
    """
    src = os.path.join(host_share_path, export_subdir, f"{driver_base}.gcda")
    if not os.path.exists(src):
        raise FileNotFoundError(
            f"GCOV data file not found at {src}. "
            "Ensure you mounted hostshare inside QEMU and copied the gcda file."
        )

    dst = os.path.join(driver_dir_path, f"{driver_base}.gcda")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.move(src, dst)
    return dst


def run_gcovr_report(
    linux_dir: str,
    buildroot_dir: str,
    output_path: str,
    gcov_tool: Optional[str] = None,
) -> None:
    """
    Run gcovr to produce a JSON coverage report rooted at linux_dir.
    """
    output_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if gcov_tool is None:
        gcov_tool = os.path.join(
            buildroot_dir,
            "output",
            "host",
            "bin",
            "x86_64-linux-gcov",
        )

    cmd = [
        "gcovr",
        ".",
        "--gcov-executable",
        gcov_tool,
        "--gcov-ignore-errors",
        "all",
        "--root",
        ".",
        "--json",
        output_path,
    ]
    subprocess.run(cmd, cwd=linux_dir, check=True)


def collect_gcov_results(
    driver_c_path: str,
    driver_dir_path: str,
    host_share_path: str,
    linux_dir: str,
    buildroot_dir: str,
    coverage_output_path: str,
    export_subdir: str = "gcov_export",
) -> str:
    """
    Fetch gcov data from the shared folder and run gcovr.
    Returns the path of the moved gcda file.
    """
    driver_base = os.path.splitext(os.path.basename(driver_c_path))[0]
    gcda_path = move_gcov_data_from_share(
        driver_base=driver_base,
        driver_dir_path=driver_dir_path,
        host_share_path=host_share_path,
        export_subdir=export_subdir,
    )
    run_gcovr_report(
        linux_dir=linux_dir,
        buildroot_dir=buildroot_dir,
        output_path=coverage_output_path,
    )
    return gcda_path


def summarize_function_coverage(
    coverage_json_path: str,
    driver_rel_path: str,
    function_name: str,
    start_line: int,
    end_line: int,
) -> Dict[str, float]:
    """
    Compute block/line/branch coverage for a specific function using the gcovr JSON report.
    """
    with open(coverage_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    files = data.get("files", [])
    file_entry = next(
        (
            entry
            for entry in files
            if os.path.normpath(entry.get("file", "")) == os.path.normpath(driver_rel_path)
        ),
        None,
    )
    if not file_entry:
        raise ValueError(f"Coverage data for {driver_rel_path} not found in {coverage_json_path}")

    func_entry = next(
        (
            fn
            for fn in file_entry.get("functions", [])
            if fn.get("name") == function_name
        ),
        {},
    )
    blocks_percent = float(func_entry.get("blocks_percent", 0.0))
    function_entry_found = bool(func_entry)

    lines = file_entry.get("lines", [])
    line_entries = [
        ln
        for ln in lines
        if start_line <= ln.get("line_number", 0) <= end_line
    ]

    covered_lines: List[int] = []
    missed_lines: List[int] = []
    for ln in line_entries:
        line_no = ln.get("line_number")
        if ln.get("count", 0) > 0:
            covered_lines.append(line_no)
        else:
            missed_lines.append(line_no)

    line_total = len(line_entries)
    line_hit = len(covered_lines)
    line_percent = (line_hit / line_total * 100.0) if line_total else 0.0

    covered_branches: List[Dict[str, int]] = []
    missed_branches: List[Dict[str, int]] = []
    for ln in line_entries:
        line_no = ln.get("line_number")
        for idx, br in enumerate(ln.get("branches", [])):
            entry = {"line": line_no, "branch_index": idx}
            if br.get("count", 0) > 0:
                covered_branches.append(entry)
            else:
                missed_branches.append(entry)

    branch_total = len(covered_branches) + len(missed_branches)
    branch_hit = len(covered_branches)
    branch_percent = (branch_hit / branch_total * 100.0) if branch_total else 100.0

    return {
        "blocks_percent": blocks_percent,
        "line_percent": line_percent,
        "branch_percent": branch_percent,
        "line_total": line_total,
        "line_hit": line_hit,
        "branch_total": branch_total,
        "branch_hit": branch_hit,
        "function_entry_found": function_entry_found,
        "coverage_line_mapping_failed": bool(function_entry_found and blocks_percent > 0.0 and line_total == 0),
        "covered_lines": covered_lines,
        "missed_lines": missed_lines,
        "covered_branches": covered_branches,
        "missed_branches": missed_branches,
    }
