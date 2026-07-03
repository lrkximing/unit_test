import json
from pathlib import Path

def _type_key_from_driver_path(driver_path: str):
    parts = driver_path.split("/")
    if len(parts) >= 4 and parts[0] == "drivers" and parts[1] == "iio":
        return "iio/" + parts[2]
    if len(parts) >= 4 and parts[:3] == ["drivers", "power", "supply"]:
        return "power/supply"
    if len(parts) >= 2 and parts[0] == "drivers":
        return parts[1]
    return "unknown"

def load_driver_types(targets_path: Path):
    mapping = {}
    if not targets_path.exists():
        return mapping
    with targets_path.open("r") as f:
        targets = json.load(f)
    for rel in targets:
        mapping[Path(rel).stem] = _type_key_from_driver_path(rel)
    return mapping

def accumulate_type(data_store, type_key, iterations):
    entry = data_store.setdefault(type_key, {"driver_count": 0, "sums": []})
    entry["driver_count"] += 1
    if not entry["sums"]:
        entry["sums"] = [
            {
                "compile_pass_rate": 0.0,
                "average_test_pass_rate": 0.0,
                "average_line_coverage": 0.0,
                "average_branch_coverage": 0.0,
            }
            for _ in iterations
        ]
    for idx, iter_entry in enumerate(iterations):
        sums = entry["sums"][idx]
        sums["compile_pass_rate"] += iter_entry.get("compile_pass_rate", 0.0)
        sums["average_test_pass_rate"] += iter_entry.get("average_test_pass_rate", 0.0)
        sums["average_line_coverage"] += iter_entry.get("average_line_coverage", 0.0)
        sums["average_branch_coverage"] += iter_entry.get("average_branch_coverage", 0.0)


def finalize_types(data_store):
    result = {}
    for type_key, info in data_store.items():
        count = info["driver_count"]
        averages = []
        for idx, sums in enumerate(info["sums"]):
            averages.append(
                {
                    "iteration": idx + 1,
                    "compile_pass_rate": sums["compile_pass_rate"] / count if count else 0.0,
                    "average_test_pass_rate": sums["average_test_pass_rate"] / count if count else 0.0,
                    "average_line_coverage": sums["average_line_coverage"] / count if count else 0.0,
                    "average_branch_coverage": sums["average_branch_coverage"] / count if count else 0.0,
                }
            )
        result[type_key] = {
            "driver_count": count,
            "iterations": averages,
        }
    return result

def main():
    base_dir = Path(__file__).resolve().parent
    output_root = base_dir / "output_all"
    if not output_root.exists():
        print(f"[WARN] output directory not found: {output_root}")
        return
    targets_path = base_dir / "data" / "ut_targets.json"
    type_mapping = load_driver_types(targets_path)
    type_accumulator = {}
    overall_accumulator = {}
    for driver_dir in sorted(output_root.iterdir()):
        if not driver_dir.is_dir():
            continue
        summary_path = driver_dir / "summary_res.json"
        if not summary_path.exists():
            continue
        with summary_path.open("r") as f:
            summary = json.load(f)
        iterations = summary.get("iterations", [])
        driver_name = driver_dir.name
        type_key = type_mapping.get(driver_name, "unknown")
        accumulate_type(type_accumulator, type_key, iterations)
        accumulate_type(overall_accumulator, "overall", iterations)
    types_summary = finalize_types(type_accumulator)
    overall_summary = finalize_types(overall_accumulator).get("overall", {"driver_count": 0, "iterations": []})
    output = {
        "types": types_summary,
        "overall": overall_summary,
    }
    out_path = output_root / "summary_types.json"
    with out_path.open("w") as f:
        json.dump(output, f, indent=2)
    print(f"[EVAL-TYPES] results written to {out_path}")

if __name__ == "__main__":
    main()
