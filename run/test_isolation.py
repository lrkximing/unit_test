import os
import re
from typing import Dict, List, Optional, Tuple


def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def remove_generated_test_objects_from_makefile(
    content: str,
    config_name: str,
    keep_obj_name: Optional[str] = None,
) -> Tuple[str, bool]:
    removed = False
    out: List[str] = []
    config_ref = f"CONFIG_{config_name}"
    keep = (keep_obj_name or "").strip()
    generated_test_re = re.compile(r"\b[A-Za-z0-9_.-]+_test_case\.o\b")
    for line in content.splitlines(keepends=True):
        if config_ref in line and generated_test_re.search(line):
            if keep and re.search(rf"\b{re.escape(keep)}\b", line):
                out.append(line)
            else:
                removed = True
            continue
        out.append(line)
    return "".join(out), removed


def is_raca_generated_test_file(path: str) -> bool:
    try:
        text = _read_text(path)
    except OSError:
        return False
    markers = (
        "Auto-generated KUnit test scaffold",
        "RACA_SCENARIO:",
        "RACA Environment Skeleton",
        "TEST EXPORT INTERFACES - DO NOT MODIFY",
    )
    return any(marker in text for marker in markers)


def isolate_current_driver_test_case(
    *,
    driver_dir_path: str,
    makefile_path: str,
    config_name: str,
    current_test_file_name: str,
    current_obj_name: str,
) -> Dict[str, object]:
    removed_files: List[str] = []
    changed_files: List[str] = []
    current_abs = os.path.abspath(os.path.join(driver_dir_path, current_test_file_name))
    if os.path.isdir(driver_dir_path):
        for name in os.listdir(driver_dir_path):
            if not name.endswith("_test_case.c"):
                continue
            path = os.path.join(driver_dir_path, name)
            if os.path.abspath(path) == current_abs:
                continue
            if not os.path.isfile(path) or not is_raca_generated_test_file(path):
                continue
            try:
                os.remove(path)
                removed_files.append(path)
            except OSError:
                pass
    if os.path.exists(makefile_path):
        before = _read_text(makefile_path)
        after, changed = remove_generated_test_objects_from_makefile(
            before,
            config_name,
            keep_obj_name=current_obj_name,
        )
        if changed:
            with open(makefile_path, "w", encoding="utf-8") as f:
                f.write(after)
            changed_files.append(makefile_path)
    return {
        "removed_files": removed_files,
        "changed_files": changed_files,
    }
