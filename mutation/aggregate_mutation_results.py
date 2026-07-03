import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


OPERATOR_GROUP_ORDER = ["rep_const", "rep_op", "negate", "del_stmt"]


def _load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, data: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _category_from_driver_path(driver_path: str) -> str:
    path = driver_path or ""
    if "/drivers/leds/" in path:
        return "LED"
    if "/drivers/hwmon/" in path:
        return "Hwmon"
    if "/drivers/iio/temperature/" in path:
        return "Temperature"
    if "/drivers/iio/light/" in path:
        return "Light"
    if "/drivers/iio/gyro/" in path:
        return "Gyro"
    if "/drivers/iio/humidity/" in path:
        return "Humidity"
    if "/drivers/power/supply/" in path:
        return "Power Supply"
    if "/drivers/input/" in path:
        return "Input"
    return "Other"


def _summary_files(root: Path) -> Iterable[Path]:
    yield from sorted(root.rglob("mutation_summary.json"))


def _is_complete(summary: Dict) -> bool:
    return "valid_mutants" in summary and "operator_group_summary" in summary


def _empty_accumulator() -> Dict:
    return {
        "functions": 0,
        "complete_functions": 0,
        "generated_mutants": 0,
        "total_mutants": 0,
        "valid_mutants": 0,
        "invalid_mutants": 0,
        "killed_mutants": 0,
        "survived_mutants": 0,
    }


def _add_counts(acc: Dict, metrics: Dict, count_function: bool = False, complete_function: bool = False) -> None:
    if count_function:
        acc["functions"] += 1
    if complete_function:
        acc["complete_functions"] += 1
    generated = int(metrics.get("generated_mutants", metrics.get("total_mutants", 0)) or 0)
    acc["generated_mutants"] += generated
    acc["total_mutants"] += int(metrics.get("total_mutants", generated) or 0)
    acc["valid_mutants"] += int(metrics.get("valid_mutants", 0) or 0)
    acc["invalid_mutants"] += int(metrics.get("invalid_mutants", 0) or 0)
    acc["killed_mutants"] += int(metrics.get("killed_mutants", 0) or 0)
    acc["survived_mutants"] += int(metrics.get("survived_mutants", max(generated - int(metrics.get("killed_mutants", 0) or 0), 0)) or 0)


def _finalize(acc: Dict) -> Dict:
    generated = int(acc.get("generated_mutants", acc.get("total_mutants", 0)) or 0)
    valid = int(acc.get("valid_mutants", 0) or 0)
    killed = int(acc.get("killed_mutants", 0) or 0)
    out = dict(acc)
    out["mutation_score"] = (killed / generated * 100.0) if generated else 0.0
    out["valid_mutation_score"] = (killed / valid * 100.0) if valid else 0.0
    return out


def _method_label(spec: str) -> Tuple[str, Path]:
    if "=" not in spec:
        path = Path(spec)
        return path.name, path
    label, path = spec.split("=", 1)
    return label, Path(path)


def _manifest_key(root: Path, summary_path: Path) -> str:
    function_dir = summary_path.parent
    try:
        return str(function_dir.relative_to(root.resolve()))
    except ValueError:
        return str(function_dir)


def _summary_scope(summary: Dict, scope: str) -> Dict:
    if scope == "target":
        return summary
    if scope == "passed-scenarios":
        return summary.get("passed_scenarios_subset") or {}
    raise ValueError(f"unsupported scope: {scope}")


def _summary_map(root: Path) -> Tuple[Dict[str, Dict], List[Dict]]:
    root = root.resolve()
    summaries: Dict[str, Dict] = {}
    incomplete: List[Dict] = []
    for summary_path in _summary_files(root):
        try:
            summary = _load_json(summary_path)
        except (OSError, json.JSONDecodeError):
            incomplete.append({"summary_path": str(summary_path), "status": "unreadable"})
            continue
        key = _manifest_key(root, summary_path)
        if not _is_complete(summary):
            incomplete.append(
                {
                    "key": key,
                    "summary_path": str(summary_path),
                    "status": summary.get("status", "incomplete"),
                }
            )
            continue
        summaries[key] = summary
    return summaries, incomplete


def _catalog_group_metrics(function_entry: Dict, method_metrics: Dict, group: str) -> Dict:
    generated = int((function_entry.get("operator_group_counts") or {}).get(group, 0) or 0)
    method_group = (method_metrics.get("operator_group_summary") or {}).get(group, {}) or {}
    killed = int(method_group.get("killed_mutants", 0) or 0)
    valid = int(method_group.get("valid_mutants", 0) or 0)
    invalid = int(method_group.get("invalid_mutants", 0) or 0)
    return {
        "generated_mutants": generated,
        "total_mutants": generated,
        "valid_mutants": valid,
        "invalid_mutants": invalid,
        "killed_mutants": killed,
        "survived_mutants": max(generated - killed, 0),
    }


def _catalog_function_metrics(function_entry: Dict, method_metrics: Optional[Dict]) -> Dict:
    generated = int(function_entry.get("total_mutants", 0) or 0)
    method_metrics = method_metrics or {}
    killed = int(method_metrics.get("killed_mutants", 0) or 0)
    valid = int(method_metrics.get("valid_mutants", 0) or 0)
    invalid = int(method_metrics.get("invalid_mutants", 0) or 0)
    return {
        "generated_mutants": generated,
        "total_mutants": generated,
        "valid_mutants": valid,
        "invalid_mutants": invalid,
        "killed_mutants": killed,
        "survived_mutants": max(generated - killed, 0),
    }


def aggregate_method(label: str, root: Path, scope: str, catalog: Optional[Dict] = None) -> Dict:
    root = root.resolve()
    overall = _empty_accumulator()
    by_category = defaultdict(_empty_accumulator)
    by_operator_group = defaultdict(_empty_accumulator)
    by_operator = defaultdict(_empty_accumulator)
    functions: List[Dict] = []
    summaries, incomplete = _summary_map(root)

    if catalog is not None:
        for function_entry in catalog.get("functions", []) or []:
            key = function_entry.get("key")
            category = function_entry.get("category") or "Other"
            summary = summaries.get(key)
            scoped_metrics = _summary_scope(summary, scope) if summary else {}
            metrics = _catalog_function_metrics(function_entry, scoped_metrics)
            _add_counts(overall, metrics, count_function=True, complete_function=bool(summary))
            _add_counts(by_category[category], metrics, count_function=True, complete_function=bool(summary))

            for group in OPERATOR_GROUP_ORDER:
                group_metrics = _catalog_group_metrics(function_entry, scoped_metrics, group)
                _add_counts(by_operator_group[group], group_metrics)

            functions.append(
                {
                    "method": label,
                    "key": key,
                    "category": category,
                    "function": function_entry.get("function"),
                    "summary_path": str(root / key / "mutation_summary.json") if key else "",
                    "evaluation_status": "evaluated" if summary else "zero_filled_no_ready_tests",
                    **_finalize(_empty_accumulator() | {
                        "functions": 1,
                        "complete_functions": 1 if summary else 0,
                        **metrics,
                    }),
                }
            )

        ordered_group_keys = [item for item in OPERATOR_GROUP_ORDER if item in by_operator_group]
        return {
            "method": label,
            "root": str(root),
            "scope": scope,
            "score_denominator": "catalog_generated_mutants",
            "catalog_path": catalog.get("path"),
            "overall": _finalize(overall),
            "by_category": {key: _finalize(value) for key, value in sorted(by_category.items())},
            "by_operator_group": {key: _finalize(by_operator_group[key]) for key in ordered_group_keys},
            "by_operator": {key: _finalize(value) for key, value in sorted(by_operator.items())},
            "functions": functions,
            "incomplete": incomplete,
        }

    for key, summary in summaries.items():

        metrics = _summary_scope(summary, scope)
        driver_path = summary.get("driver_c_path") or summary.get("driver_path") or ""
        category = _category_from_driver_path(driver_path)
        _add_counts(overall, metrics, count_function=True, complete_function=True)
        _add_counts(by_category[category], metrics, count_function=True, complete_function=True)

        for group, group_metrics in (metrics.get("operator_group_summary") or {}).items():
            _add_counts(by_operator_group[group], group_metrics)
        for operator, operator_metrics in (metrics.get("operator_summary") or {}).items():
            _add_counts(by_operator[operator], operator_metrics)

        functions.append(
            {
                "method": label,
                "key": key,
                "category": category,
                "function": summary.get("function"),
                "summary_path": str(root / key / "mutation_summary.json"),
                **_finalize(_empty_accumulator() | {
                    "functions": 1,
                    "complete_functions": 1,
                    "total_mutants": int(metrics.get("total_mutants", 0) or 0),
                    "valid_mutants": int(metrics.get("valid_mutants", 0) or 0),
                    "invalid_mutants": int(metrics.get("invalid_mutants", 0) or 0),
                    "killed_mutants": int(metrics.get("killed_mutants", 0) or 0),
                    "survived_mutants": int(metrics.get("survived_mutants", 0) or 0),
                }),
            }
        )

    ordered_group_keys = [item for item in OPERATOR_GROUP_ORDER if item in by_operator_group]
    ordered_group_keys.extend(sorted(item for item in by_operator_group if item not in ordered_group_keys))
    return {
        "method": label,
        "root": str(root),
        "scope": scope,
        "overall": _finalize(overall),
        "by_category": {key: _finalize(value) for key, value in sorted(by_category.items())},
        "by_operator_group": {key: _finalize(by_operator_group[key]) for key in ordered_group_keys},
        "by_operator": {key: _finalize(value) for key, value in sorted(by_operator.items())},
        "functions": functions,
        "incomplete": incomplete,
    }


def _write_table_csv(path: Path, rows: List[Dict], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _flatten_method_tables(method_result: Dict) -> Tuple[List[Dict], List[Dict], List[Dict], List[Dict]]:
    method = method_result["method"]
    scope = method_result["scope"]
    overall_rows = [{"method": method, "scope": scope, "bucket": "overall", **method_result["overall"]}]
    category_rows = [
        {"method": method, "scope": scope, "category": category, **metrics}
        for category, metrics in method_result["by_category"].items()
    ]
    group_rows = [
        {"method": method, "scope": scope, "operator_group": group, **metrics}
        for group, metrics in method_result["by_operator_group"].items()
    ]
    operator_rows = [
        {"method": method, "scope": scope, "operator": operator, **metrics}
        for operator, metrics in method_result["by_operator"].items()
    ]
    return overall_rows, category_rows, group_rows, operator_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate mutation_summary.json files by method and mutation operator.")
    parser.add_argument("--result", action="append", required=True, help="METHOD=mutation_output_root. Can be repeated.")
    parser.add_argument("--scope", choices=("target", "passed-scenarios"), default="target")
    parser.add_argument("--catalog", default=None, help="Shared mutant catalog. Missing method summaries are zero-filled.")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-prefix", required=True, help="CSV prefix. Writes *_overall.csv, *_by_operator_group.csv, etc.")
    args = parser.parse_args()

    catalog = None
    if args.catalog:
        catalog = _load_json(Path(args.catalog))
        catalog["path"] = str(Path(args.catalog).resolve())

    results = [aggregate_method(label, path, args.scope, catalog=catalog) for label, path in map(_method_label, args.result)]
    _write_json(Path(args.output_json), {"scope": args.scope, "methods": results})

    all_overall: List[Dict] = []
    all_categories: List[Dict] = []
    all_groups: List[Dict] = []
    all_operators: List[Dict] = []
    all_functions: List[Dict] = []
    for result in results:
        overall, categories, groups, operators = _flatten_method_tables(result)
        all_overall.extend(overall)
        all_categories.extend(categories)
        all_groups.extend(groups)
        all_operators.extend(operators)
        all_functions.extend(result["functions"])

    fields_common = [
        "functions",
        "complete_functions",
        "generated_mutants",
        "total_mutants",
        "valid_mutants",
        "invalid_mutants",
        "killed_mutants",
        "survived_mutants",
        "mutation_score",
        "valid_mutation_score",
    ]
    prefix = Path(args.output_prefix)
    _write_table_csv(prefix.with_name(prefix.name + "_overall.csv"), all_overall, ["method", "scope", "bucket", *fields_common])
    _write_table_csv(prefix.with_name(prefix.name + "_by_category.csv"), all_categories, ["method", "scope", "category", *fields_common])
    _write_table_csv(prefix.with_name(prefix.name + "_by_operator_group.csv"), all_groups, ["method", "scope", "operator_group", *fields_common])
    _write_table_csv(prefix.with_name(prefix.name + "_by_operator.csv"), all_operators, ["method", "scope", "operator", *fields_common])
    _write_table_csv(prefix.with_name(prefix.name + "_by_function.csv"), all_functions, ["method", "key", "category", "function", "summary_path", *fields_common])

    for result in results:
        overall = result["overall"]
        print(
            "{method} {scope}: functions={functions} valid={valid} killed={killed} score={score:.2f}% incomplete={incomplete}".format(
                method=result["method"],
                scope=args.scope,
                functions=overall.get("complete_functions", 0),
                valid=overall.get("valid_mutants", 0),
                killed=overall.get("killed_mutants", 0),
                score=overall.get("mutation_score", 0.0),
                incomplete=len(result.get("incomplete") or []),
            )
        )


if __name__ == "__main__":
    main()
