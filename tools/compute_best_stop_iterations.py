#!/usr/bin/env python3
"""Compute the iteration where each function reaches its best checkpoint.

The checkpoint ranking matches the current same-checkpoint plotting logic:
Pass is primary, then Line+Branch coverage, then passed test count, then the
later iteration.  Failed/no-runnable checkpoints contribute zero branch; valid
source-level no-branch checkpoints are treated as branch N/A and add no branch
score.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


CATEGORY_ORDER = [
    ("leds", "LED"),
    ("hwmon", "Hwmon"),
    ("iio/temperature", "Temperature"),
    ("iio/light", "Light"),
    ("iio/gyro", "Gyro"),
    ("iio/humidity", "Humidity"),
    ("power/supply", "Power Supply"),
]


def _load_json(path: Path) -> Dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _driver_slug(driver_path: str) -> str:
    return Path(driver_path).stem


def _category_key(driver_path: str) -> str:
    parts = driver_path.split("/")
    if len(parts) >= 4 and parts[0] == "drivers" and parts[1] == "iio":
        return "iio/" + parts[2]
    if len(parts) >= 4 and parts[:3] == ["drivers", "power", "supply"]:
        return "power/supply"
    if len(parts) >= 2 and parts[0] == "drivers":
        return parts[1]
    return "other"


def _float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _int(value, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _kunit_counts(kunit: Dict) -> Tuple[int, int, int, float]:
    tests = kunit.get("tests") or []
    total = _int(kunit.get("tests_total"), len(tests))
    passed = _int(kunit.get("tests_passed_count"), len(kunit.get("passed") or []))
    failed = _int(kunit.get("tests_failed_count"), len(kunit.get("failed") or []))
    pass_rate = kunit.get("pass_rate")
    if pass_rate is None and total:
        pass_rate = 100.0 * passed / total
    return total, passed, failed, _float(pass_rate)


def _adjusted_line_branch(item: Dict) -> Tuple[float, float]:
    coverage = item.get("coverage") or {}
    line_total = _int(coverage.get("line_total"))
    line_hit = _int(coverage.get("line_hit"))
    branch_total = _int(coverage.get("branch_total"))
    branch_hit = _int(coverage.get("branch_hit"))
    line = _float(coverage.get("line_percent"))

    total, passed, _failed, _pass_rate = _kunit_counts(item.get("kunit") or {})
    if total == 0 or passed == 0:
        line = 0.0
        branch = 0.0
    elif branch_total > 0:
        branch = _float(coverage.get("branch_percent"), 100.0 * branch_hit / branch_total)
    elif line_total > 0 and line_hit > 0:
        branch = 0.0
    else:
        branch = 0.0

    return line, branch


def _score(item: Dict) -> Tuple[float, float, int, int]:
    total, passed, _failed, pass_rate = _kunit_counts(item.get("kunit") or {})
    line, branch = _adjusted_line_branch(item)
    if total == 0:
        pass_rate = 0.0
    return (pass_rate, line + branch, passed, _int(item.get("iteration")))


def _result_candidates(summary: Dict) -> Iterable[Dict]:
    for entry in summary.get("history") or []:
        if entry.get("stage") == "results":
            yield entry

    best = summary.get("best_metrics_checkpoint")
    if isinstance(best, dict):
        yield best

    if summary.get("kunit") or summary.get("coverage"):
        synthetic = dict(summary)
        synthetic["iteration"] = _int(summary.get("metrics_iteration") or summary.get("iteration"))
        yield synthetic


def _best_iteration(summary: Dict) -> int:
    candidates = list(_result_candidates(summary))
    if not candidates:
        return 0
    return _int(max(candidates, key=_score).get("iteration"))


def _empty_acc() -> Dict:
    return {"function_count": 0, "iteration_sum": 0.0, "missing": 0, "distribution": Counter()}


def _add(acc: Dict, iteration: int, missing: bool) -> None:
    acc["function_count"] += 1
    acc["iteration_sum"] += iteration
    acc["missing"] += int(missing)
    acc["distribution"][iteration] += 1


def _finalize(label: str, acc: Dict) -> Dict:
    count = int(acc["function_count"])
    return {
        "Category": label,
        "Functions": count,
        "AvgBestIteration": acc["iteration_sum"] / count if count else 0.0,
        "Missing": int(acc["missing"]),
        "IterationDistribution": dict(sorted(acc["distribution"].items())),
    }


def compute(root: Path, targets: Dict[str, List[str]]) -> List[Dict]:
    by_category = defaultdict(_empty_acc)
    total = _empty_acc()

    for driver, functions in targets.items():
        category = _category_key(driver)
        slug = _driver_slug(driver)
        for function in functions:
            summary_path = root / slug / function / "summary.json"
            missing = not summary_path.exists()
            summary = _load_json(summary_path) if not missing else {}
            iteration = _best_iteration(summary)
            _add(by_category[category], iteration, missing)
            _add(total, iteration, missing)

    rows = [_finalize(display, by_category[key]) for key, display in CATEGORY_ORDER]
    rows.append(_finalize("Total", total))
    return rows


def _write_csv(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "Category",
                "Functions",
                "AvgBestIteration",
                "Missing",
                "IterationDistribution",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "IterationDistribution": json.dumps(row["IterationDistribution"])})


def _write_tex(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Average best-effect iteration across driver categories.}",
        r"\label{tab:best_stop_iteration}",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"\textbf{Category} & \textbf{Functions} & \textbf{Avg Best Iterations} \\",
        r"\midrule",
    ]
    for row in rows:
        category = row["Category"]
        functions = int(row["Functions"])
        avg_iter = f"{row['AvgBestIteration']:.2f}"
        if category == "Total":
            lines.extend(
                [
                    r"\midrule",
                    rf"\textbf{{Total}} & \textbf{{{functions}}} & \textbf{{{avg_iter}}} \\",
                ]
            )
        else:
            lines.append(rf"{category} & {functions} & {avg_iter} \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", default="unit_test_v2/data/ut_targets.json")
    parser.add_argument("--root", default="extracted_results/output_full_iter5_combined_20260623")
    parser.add_argument("--output-prefix", default="paper_figures/best_stop_iterations_current")
    args = parser.parse_args()

    targets = json.loads(Path(args.targets).read_text(encoding="utf-8"))
    rows = compute(Path(args.root), targets)
    output_prefix = Path(args.output_prefix)

    output_prefix.with_suffix(".json").write_text(
        json.dumps({"root": args.root, "targets": args.targets, "rows": rows}, indent=2),
        encoding="utf-8",
    )
    _write_csv(output_prefix.with_suffix(".csv"), rows)
    _write_tex(output_prefix.with_suffix(".tex"), rows)

    for row in rows:
        print(
            f"{row['Category']}: functions={row['Functions']}, "
            f"avg_best_iter={row['AvgBestIteration']:.2f}, missing={row['Missing']}"
        )


if __name__ == "__main__":
    main()
