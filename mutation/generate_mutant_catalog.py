import argparse
import collections
import json
import sys
import time
from pathlib import Path
from typing import Dict, List


UNIT_TEST_ROOT = Path(__file__).resolve().parents[1]
if str(UNIT_TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(UNIT_TEST_ROOT))

from mutation.evaluate_mutation import generate_mutants
from mutation.run_from_manifest import _function_body_line_range


OPERATOR_GROUP_ORDER = ["rep_const", "rep_op", "negate", "del_stmt"]


def _category_from_driver_path(driver_path: str) -> str:
    path = "/" + driver_path.lstrip("/")
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
    return "Other"


def _function_key(driver_path: str, function_name: str) -> str:
    return f"{Path(driver_path).stem}/{function_name}"


def _write_json(path: Path, data: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _write_csv(path: Path, functions: List[Dict]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "key",
        "category",
        "driver_path",
        "function",
        "start_line",
        "end_line",
        "total_mutants",
        *OPERATOR_GROUP_ORDER,
        "status",
        "reason",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for item in functions:
            row = {field: item.get(field, "") for field in fields}
            groups = item.get("operator_group_counts") or {}
            for group in OPERATOR_GROUP_ORDER:
                row[group] = groups.get(group, 0)
            writer.writerow(row)


def build_catalog(targets_path: Path, linux_dir: Path, replacement_policy: str) -> Dict:
    targets = json.loads(targets_path.read_text(encoding="utf-8"))
    functions: List[Dict] = []
    totals = collections.Counter()
    by_category: Dict[str, Dict] = collections.defaultdict(lambda: {"functions": 0, "total_mutants": 0, "operator_group_counts": collections.Counter()})
    missing: List[Dict] = []

    for driver_path, function_names in sorted(targets.items()):
        absolute_driver_path = linux_dir / driver_path
        category = _category_from_driver_path(driver_path)
        for function_name in function_names:
            key = _function_key(driver_path, function_name)
            entry = {
                "key": key,
                "category": category,
                "driver_path": driver_path,
                "absolute_driver_path": str(absolute_driver_path),
                "function": function_name,
                "replacement_policy": replacement_policy,
            }
            by_category[category]["functions"] += 1
            if not absolute_driver_path.exists():
                entry.update({"status": "missing_source", "reason": "driver source file not found", "total_mutants": 0})
                functions.append(entry)
                missing.append(entry)
                continue
            start_line, end_line = _function_body_line_range(absolute_driver_path, function_name)
            entry["start_line"] = start_line
            entry["end_line"] = end_line
            if start_line is None or end_line is None:
                entry.update({"status": "missing_function_range", "reason": "target function body was not located", "total_mutants": 0})
                functions.append(entry)
                missing.append(entry)
                continue
            source = absolute_driver_path.read_text(encoding="utf-8", errors="ignore")
            mutants = generate_mutants(
                source,
                start_line=start_line,
                end_line=end_line,
                replacement_policy=replacement_policy,
            )
            group_counts = collections.Counter(mutant.operator_group for mutant in mutants)
            entry.update(
                {
                    "status": "ok",
                    "total_mutants": len(mutants),
                    "operator_group_counts": dict(group_counts),
                    "mutants": [mutant.metadata() for mutant in mutants],
                }
            )
            functions.append(entry)
            totals.update(group_counts)
            by_category[category]["total_mutants"] += len(mutants)
            by_category[category]["operator_group_counts"].update(group_counts)

    by_category_out = {}
    for category, item in sorted(by_category.items()):
        by_category_out[category] = {
            "functions": item["functions"],
            "total_mutants": item["total_mutants"],
            "operator_group_counts": dict(item["operator_group_counts"]),
        }
    return {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "targets_path": str(targets_path),
        "linux_dir": str(linux_dir),
        "replacement_policy": replacement_policy,
        "target_count": sum(len(items) for items in targets.values()),
        "function_range_found": len([item for item in functions if item.get("status") == "ok"]),
        "missing_count": len(missing),
        "total_mutants": sum(totals.values()),
        "operator_group_counts": dict(totals),
        "by_category": by_category_out,
        "missing": missing,
        "functions": functions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a shared representative mutant catalog for a target dataset.")
    parser.add_argument("--targets", required=True)
    parser.add_argument("--linux-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--csv", default=None)
    parser.add_argument("--replacement-policy", choices=("representative", "exhaustive"), default="representative")
    args = parser.parse_args()

    catalog = build_catalog(
        targets_path=Path(args.targets),
        linux_dir=Path(args.linux_dir),
        replacement_policy=args.replacement_policy,
    )
    _write_json(Path(args.output), catalog)
    if args.csv:
        _write_csv(Path(args.csv), catalog["functions"])
    print(
        "catalog targets={targets} found={found} mutants={mutants} missing={missing}".format(
            targets=catalog["target_count"],
            found=catalog["function_range_found"],
            mutants=catalog["total_mutants"],
            missing=catalog["missing_count"],
        )
    )


if __name__ == "__main__":
    main()
