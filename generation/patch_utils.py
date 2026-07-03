import os
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple


class PatchApplicationError(RuntimeError):
    pass


@dataclass
class PatchResult:
    patch_text: str
    strip_level: int
    stdout: str
    stderr: str
    attempts: List[Dict[str, str]] = field(default_factory=list)


def extract_unified_diff(model_output: str) -> str:
    text = (model_output or "").strip()
    if not text:
        raise PatchApplicationError(
            "No patch was produced. The model rewrite was empty or produced no source-code "
            "changes after conversion to a local diff."
        )

    fenced = re.search(r"```(?:diff|patch)?\s*([\s\S]*?)```", text, re.IGNORECASE)
    if fenced:
        text = fenced.group(1).strip()

    starts = []
    for marker in ("diff --git ", "--- "):
        idx = text.find(marker)
        if idx >= 0:
            starts.append(idx)
    if starts:
        text = text[min(starts):].strip()

    if "@@" not in text or not re.search(r"(?m)^---\s+", text) or not re.search(r"(?m)^\+\+\+\s+", text):
        raise PatchApplicationError("Model output does not look like a unified diff.")
    return text + ("\n" if not text.endswith("\n") else "")


def changed_files_from_patch(patch_text: str) -> List[str]:
    patch_text = extract_unified_diff(patch_text)
    files: List[str] = []
    current_old = ""
    for line in patch_text.splitlines():
        if line.startswith("--- "):
            current_old = line[4:].strip().split("\t", 1)[0]
            continue
        if not line.startswith("+++ "):
            continue
        path = line[4:].strip().split("\t", 1)[0]
        if path == "/dev/null":
            path = current_old
        if path.startswith("a/") or path.startswith("b/"):
            path = path[2:]
        if path and path != "/dev/null" and path not in files:
            files.append(path)
    return files


def patch_path_targets_file(patch_path: str, test_file_name: str) -> bool:
    normalized = os.path.normpath(patch_path)
    if os.path.isabs(normalized):
        return False
    if normalized.startswith("a/") or normalized.startswith("b/"):
        normalized = normalized[2:]
    if any(part == ".." for part in normalized.split(os.sep)):
        return False
    return normalized in {test_file_name, os.path.join(".", test_file_name)} or (
        os.path.basename(normalized) == test_file_name
    )


def normalize_patch_to_target_file(patch_text: str, test_file_name: str) -> str:
    patch_text = extract_unified_diff(patch_text)
    if not test_file_name:
        return patch_text

    changed_files = changed_files_from_patch(patch_text)
    if not changed_files:
        return patch_text
    if any(not patch_path_targets_file(path, test_file_name) for path in changed_files):
        return patch_text

    normalized_lines: List[str] = []
    for line in patch_text.splitlines():
        if line.startswith("--- ") and line[4:].strip().split("\t", 1)[0] != "/dev/null":
            suffix = ""
            if "\t" in line[4:]:
                suffix = "\t" + line[4:].split("\t", 1)[1]
            normalized_lines.append(f"--- {test_file_name}{suffix}")
            continue
        if line.startswith("+++ ") and line[4:].strip().split("\t", 1)[0] != "/dev/null":
            suffix = ""
            if "\t" in line[4:]:
                suffix = "\t" + line[4:].split("\t", 1)[1]
            normalized_lines.append(f"+++ {test_file_name}{suffix}")
            continue
        if line.startswith("diff --git "):
            normalized_lines.append(f"diff --git {test_file_name} {test_file_name}")
            continue
        normalized_lines.append(line)

    normalized = "\n".join(normalized_lines)
    return normalized + ("\n" if patch_text.endswith("\n") else "")


def _run_patch(cwd: str, patch_path: str, strip_level: int, dry_run: bool) -> subprocess.CompletedProcess:
    cmd = ["patch", "--batch", "--forward", "--fuzz=0", f"-p{strip_level}", "-i", patch_path]
    if dry_run:
        cmd.insert(1, "--dry-run")
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)


def apply_unified_patch(
    patch_text: str,
    cwd: str,
    strip_levels: Iterable[int] = (0, 1, 2, 3),
    target_file_name: Optional[str] = None,
) -> PatchResult:
    if not os.path.isdir(cwd):
        raise PatchApplicationError(f"Patch cwd does not exist: {cwd}")

    patch_text = extract_unified_diff(patch_text)
    if target_file_name:
        patch_text = normalize_patch_to_target_file(patch_text, target_file_name)
        strip_levels = (0,)
    attempts: List[Dict[str, str]] = []
    last_error: Optional[Tuple[int, str, str]] = None
    with tempfile.NamedTemporaryFile("w", suffix=".patch", delete=False) as tmp:
        tmp.write(patch_text)
        patch_path = tmp.name

    try:
        for strip_level in strip_levels:
            dry = _run_patch(cwd, patch_path, strip_level, dry_run=True)
            attempts.append(
                {
                    "strip_level": str(strip_level),
                    "phase": "dry-run",
                    "returncode": str(dry.returncode),
                    "stdout": dry.stdout,
                    "stderr": dry.stderr,
                }
            )
            if dry.returncode != 0:
                last_error = (strip_level, dry.stdout, dry.stderr)
                continue
            applied = _run_patch(cwd, patch_path, strip_level, dry_run=False)
            attempts.append(
                {
                    "strip_level": str(strip_level),
                    "phase": "apply",
                    "returncode": str(applied.returncode),
                    "stdout": applied.stdout,
                    "stderr": applied.stderr,
                }
            )
            if applied.returncode != 0:
                last_error = (strip_level, applied.stdout, applied.stderr)
                continue
            return PatchResult(
                patch_text=patch_text,
                strip_level=strip_level,
                stdout=applied.stdout,
                stderr=applied.stderr,
                attempts=attempts,
            )
    finally:
        try:
            os.remove(patch_path)
        except OSError:
            pass

    if last_error:
        strip_level, stdout, stderr = last_error
        attempt_text = "\n".join(
            "p{strip} {phase} rc={rc}\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}".format(
                strip=item["strip_level"],
                phase=item["phase"],
                rc=item["returncode"],
                stdout=item["stdout"],
                stderr=item["stderr"],
            )
            for item in attempts
        )
        raise PatchApplicationError(
            f"Patch failed with -p{strip_level}.\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}\nALL ATTEMPTS:\n{attempt_text}"
        )
    raise PatchApplicationError("Patch failed for all strip levels.")
