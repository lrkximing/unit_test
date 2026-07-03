import json
import os
from pathlib import Path
from typing import Dict, List

MAX_ITERATIONS = 5

def _load_function_iterations(func_dir: Path) -> Dict[int, Dict[str, float]]:
    summary_path = func_dir / "summary.json"
    if not summary_path.exists():
        return {}, 0, {}
    with summary_path.open("r") as f:
        data = json.load(f)
    history = data.get("history", [])
    results: Dict[int, Dict[str, float]] = {}
    for entry in history:
        if entry.get("stage") != "results":
            continue
        iteration = entry.get("iteration")
        if iteration is None:
            continue
        kunit = entry.get("kunit", {})
        coverage = entry.get("coverage", {})
        branch_total = int(coverage.get("branch_total") or 0)
        branch_hit = int(coverage.get("branch_hit") or 0)
        line_total = int(coverage.get("line_total") or 0)
        line_hit = int(coverage.get("line_hit") or 0)
        passed = int(kunit.get("tests_passed_count") or len(kunit.get("passed") or []))
        if passed == 0:
            branch_percent = 0.0
            branch_valid = True
        elif branch_total > 0:
            branch_percent = coverage.get("branch_percent", branch_hit / branch_total * 100.0)
            branch_valid = True
        elif line_total > 0 and line_hit > 0:
            branch_percent = None
            branch_valid = False
        else:
            branch_percent = 0.0
            branch_valid = True
        results[iteration] = {
            "pass_rate": kunit.get("pass_rate", 0.0),
            "line_percent": coverage.get("line_percent", 0.0),
            "branch_percent": branch_percent,
            "branch_valid": branch_valid,
        }
    stop_iteration = data.get("iteration") or (max(results.keys()) if results else 0)
    return results, stop_iteration, {"status": data.get("status", "")}

def _best_metrics_up_to(results: Dict[int, Dict[str, float]], iteration: int) -> Dict[str, float]:
    best_iter = None
    best_line = -1.0
    best_pass = -1.0
    for iter_idx, metrics in results.items():
        if iter_idx > iteration:
            continue
        line_percent = metrics.get("line_percent", 0.0)
        pass_rate = metrics.get("pass_rate", 0.0)
        if (
            best_iter is None
            or line_percent > best_line
            or (line_percent == best_line and pass_rate > best_pass)
            or (
                line_percent == best_line
                and pass_rate == best_pass
                and iter_idx > best_iter
            )
        ):
            best_line = line_percent
            best_pass = pass_rate
            best_iter = iter_idx
    if best_iter is None:
        return {"pass_rate": 0.0, "line_percent": 0.0, "branch_percent": 0.0}
    return results[best_iter]

def evaluate_driver(driver_dir: Path):
    function_dirs: List[Path] = []
    for item in sorted(driver_dir.iterdir()):
        if item.is_dir() and (item / "summary.json").exists():
            function_dirs.append(item)
    total_functions = len(function_dirs)
    if total_functions == 0:
        return
    function_data = {}
    for func_dir in function_dirs:
        results, stop_iter, meta = _load_function_iterations(func_dir)
        function_data[func_dir.name] = {
            "results": results,
            "stop_iteration": stop_iter,
            "meta": meta,
        }
    iterations_summary = []
    for iteration in range(1, MAX_ITERATIONS + 1):
        compile_pass = 0
        pass_sum = 0.0
        line_sum = 0.0
        branch_sum = 0.0
        branch_den = 0
        for data in function_data.values():
            results = data["results"]
            if any(iter_idx <= iteration for iter_idx in results.keys()):
                compile_pass += 1
            metrics = _best_metrics_up_to(results, iteration)
            pass_sum += metrics["pass_rate"]
            line_sum += metrics["line_percent"]
            if metrics.get("branch_valid", True):
                branch_sum += metrics["branch_percent"] or 0.0
                branch_den += 1
        compile_rate = compile_pass / total_functions if total_functions else 0.0
        iterations_summary.append(
            {
                "iteration": iteration,
                "compile_pass_rate": compile_rate,
                "average_test_pass_rate": pass_sum / total_functions if total_functions else 0.0,
                "average_line_coverage": line_sum / total_functions if total_functions else 0.0,
                "average_branch_coverage": branch_sum / branch_den if branch_den else 0.0,
                "branch_denominator": branch_den,
            }
        )
    functions_summary = []
    for name, data in function_data.items():
        per_iteration = []
        for iteration in range(1, MAX_ITERATIONS + 1):
            metrics = _best_metrics_up_to(data["results"], iteration)
            per_iteration.append({
                "iteration": iteration,
                "pass_rate": metrics["pass_rate"],
                "line_percent": metrics["line_percent"],
                "branch_percent": metrics["branch_percent"],
                "branch_valid": metrics.get("branch_valid", True),
            })
        functions_summary.append(
            {
                "function": name,
                "stop_iteration": data["stop_iteration"],
                "per_iteration": per_iteration,
            }
        )
    output = {
        "driver": driver_dir.name,
        "total_functions": total_functions,
        "iterations": iterations_summary,
        "functions": functions_summary,
    }
    summary_path = driver_dir / "summary_res.json"
    with summary_path.open("w") as f:
        json.dump(output, f, indent=2)
    print(f"[EVAL] {driver_dir.name} -> {summary_path}")

def main():
    base_dir = Path(__file__).resolve().parent
    output_root = base_dir / "output_all"
    if not output_root.exists():
        print(f"[WARN] output directory not found: {output_root}")
        return
    for driver_dir in sorted(output_root.iterdir()):
        if not driver_dir.is_dir():
            continue
        evaluate_driver(driver_dir)

if __name__ == "__main__":
    main()
