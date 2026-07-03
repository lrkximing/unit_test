import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


UNIT_TEST_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = UNIT_TEST_ROOT.parent


def _load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, data: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _append_jsonl(path: Path, data: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")


def _manifest_key(root: Path, manifest_path: Path) -> str:
    parent = manifest_path.resolve().parent
    try:
        return str(parent.relative_to(root.resolve()))
    except ValueError:
        return str(parent)


def _iter_ready_manifests(root: Path) -> Iterable[Tuple[str, Path, Dict]]:
    for manifest_path in sorted(root.rglob("mutation_ready.json")):
        try:
            manifest = _load_json(manifest_path)
        except (OSError, json.JSONDecodeError):
            continue
        ready_tests = manifest.get("mutation_ready_tests") or []
        if not isinstance(ready_tests, list) or not ready_tests:
            continue
        yield _manifest_key(root, manifest_path), manifest_path, manifest


def _looks_complete(summary_path: Path) -> bool:
    if not summary_path.exists():
        return False
    try:
        data = _load_json(summary_path)
    except (OSError, json.JSONDecodeError):
        return False
    if data.get("status") in {"batch_timeout", "batch_failed", "preparation_failed"}:
        return False
    return "valid_mutants" in data and "operator_group_summary" in data


def _run_one(
    manifest_path: Path,
    summary_path: Path,
    replacement_policy: str,
    mutation_scope: str,
    timeout: int,
    function_timeout: Optional[int],
    max_mutants: Optional[int],
    max_mutants_per_group: Optional[int],
    buildroot_dir: Optional[str],
) -> Dict:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    cmd: List[str] = [
        sys.executable,
        str(UNIT_TEST_ROOT / "mutation" / "run_from_manifest.py"),
        "--manifest",
        str(manifest_path),
        "--output",
        str(summary_path),
        "--output-dir",
        str(summary_path.parent),
        "--replacement-policy",
        replacement_policy,
        "--mutation-scope",
        mutation_scope,
        "--timeout",
        str(timeout),
    ]
    if max_mutants is not None:
        cmd.extend(["--max-mutants", str(max_mutants)])
    if max_mutants_per_group is not None:
        cmd.extend(["--max-mutants-per-group", str(max_mutants_per_group)])
    if buildroot_dir:
        cmd.extend(["--buildroot-dir", buildroot_dir])

    started = time.time()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            text=True,
            capture_output=True,
            timeout=function_timeout,
        )
    except subprocess.TimeoutExpired as exc:
        elapsed = time.time() - started
        timeout_summary = {
            "status": "batch_timeout",
            "manifest_path": str(manifest_path),
            "elapsed_seconds": elapsed,
            "function_timeout": function_timeout,
            "stdout": (exc.stdout or "")[-8000:] if isinstance(exc.stdout, str) else "",
            "stderr": (exc.stderr or "")[-8000:] if isinstance(exc.stderr, str) else "",
        }
        _write_json(summary_path, timeout_summary)
        return timeout_summary

    elapsed = time.time() - started
    record = {
        "status": "completed" if proc.returncode == 0 else "batch_failed",
        "manifest_path": str(manifest_path),
        "summary_path": str(summary_path),
        "returncode": proc.returncode,
        "elapsed_seconds": elapsed,
        "stdout_tail": proc.stdout[-8000:],
        "stderr_tail": proc.stderr[-8000:],
    }
    if proc.returncode != 0 and not summary_path.exists():
        _write_json(summary_path, record)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run mutation evaluation for every mutation_ready.json under a result root. "
            "Each manifest is evaluated independently and can be resumed."
        )
    )
    parser.add_argument("--root", required=True, help="RACA result root containing mutation_ready.json files.")
    parser.add_argument("--method", required=True, help="Method label stored in batch records, e.g. raca.")
    parser.add_argument("--output-root", required=True, help="Directory where per-function mutation summaries are written.")
    parser.add_argument("--replacement-policy", choices=("representative", "exhaustive"), default="representative")
    parser.add_argument(
        "--mutation-scope",
        choices=("target-with-passed", "target", "passed-scenarios", "source-facts"),
        default="target-with-passed",
    )
    parser.add_argument("--timeout", type=int, default=900, help="Per build/run command timeout passed to mutation runner.")
    parser.add_argument("--function-timeout", type=int, default=7200, help="Whole manifest timeout. Use 0 to disable.")
    parser.add_argument("--max-mutants", type=int, default=None)
    parser.add_argument("--max-mutants-per-group", type=int, default=None)
    parser.add_argument("--buildroot-dir", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument(
        "--include-keys",
        default=None,
        help="Optional newline file of relative function keys to evaluate, e.g. leds-lp3944/lp3944_dim_set_period.",
    )
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    output_root = Path(args.output_root).resolve()
    progress_path = output_root / "batch_progress.jsonl"
    manifest_items = list(_iter_ready_manifests(root))
    if args.include_keys:
        include_keys = {
            line.strip()
            for line in Path(args.include_keys).read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
        manifest_items = [item for item in manifest_items if item[0] in include_keys]
    selected = manifest_items[args.offset :]
    if args.limit is not None:
        selected = selected[: args.limit]

    batch_summary = {
        "method": args.method,
        "root": str(root),
        "output_root": str(output_root),
        "replacement_policy": args.replacement_policy,
        "mutation_scope": args.mutation_scope,
        "max_mutants": args.max_mutants,
        "max_mutants_per_group": args.max_mutants_per_group,
        "total_ready_manifests": len(manifest_items),
        "selected_manifests": len(selected),
        "offset": args.offset,
        "limit": args.limit,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    _write_json(output_root / "batch_summary.json", batch_summary)

    completed = skipped = failed = 0
    for index, (key, manifest_path, manifest) in enumerate(selected, start=args.offset + 1):
        summary_path = output_root / key / "mutation_summary.json"
        if args.skip_existing and _looks_complete(summary_path):
            skipped += 1
            record = {
                "method": args.method,
                "index": index,
                "total_ready_manifests": len(manifest_items),
                "key": key,
                "manifest_path": str(manifest_path),
                "summary_path": str(summary_path),
                "status": "skipped_existing",
            }
            _append_jsonl(progress_path, record)
            print(f"[SKIP] {index}/{len(manifest_items)} {key}", flush=True)
            continue
        if args.skip_existing and not args.retry_failed and summary_path.exists() and not _looks_complete(summary_path):
            skipped += 1
            record = {
                "method": args.method,
                "index": index,
                "total_ready_manifests": len(manifest_items),
                "key": key,
                "manifest_path": str(manifest_path),
                "summary_path": str(summary_path),
                "status": "skipped_failed_existing",
            }
            _append_jsonl(progress_path, record)
            print(f"[SKIP-FAILED] {index}/{len(manifest_items)} {key}", flush=True)
            continue

        print(f"[RUN] {index}/{len(manifest_items)} {args.method} {key}", flush=True)
        record = _run_one(
            manifest_path=manifest_path,
            summary_path=summary_path,
            replacement_policy=args.replacement_policy,
            mutation_scope=args.mutation_scope,
            timeout=args.timeout,
            function_timeout=None if args.function_timeout == 0 else args.function_timeout,
            max_mutants=args.max_mutants,
            max_mutants_per_group=args.max_mutants_per_group,
            buildroot_dir=args.buildroot_dir,
        )
        record.update(
            {
                "method": args.method,
                "index": index,
                "total_ready_manifests": len(manifest_items),
                "key": key,
                "function": manifest.get("function"),
                "ready_tests": manifest.get("mutation_ready_tests", []),
            }
        )
        _append_jsonl(progress_path, record)
        if record.get("status") == "completed":
            completed += 1
        elif record.get("status") != "skipped_existing":
            failed += 1
        print(f"[DONE] {key} status={record.get('status')} elapsed={record.get('elapsed_seconds', 0):.1f}s", flush=True)

    batch_summary.update(
        {
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "completed": completed,
            "skipped": skipped,
            "failed": failed,
        }
    )
    _write_json(output_root / "batch_summary.json", batch_summary)
    print(
        f"[BATCH] method={args.method} completed={completed} skipped={skipped} failed={failed} output={output_root}",
        flush=True,
    )


if __name__ == "__main__":
    main()
