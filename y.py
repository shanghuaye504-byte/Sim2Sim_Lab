#!/usr/bin/env python3
"""
aggregate_eval_results.py

Traverse all eval_results.json files under the directory and output aggregated metrics as CSV.

Usage:
    python aggregate_eval_results.py --dir /path/to/libero/logs --outputdir ./summary
"""

import argparse
import csv
import json
import pathlib
import sys


def find_logs_root(base_dir: pathlib.Path) -> pathlib.Path:
    """If base_dir itself is named 'logs', return it directly; otherwise check for a 'logs' subdirectory."""
    if base_dir.name == "logs":
        return base_dir
    logs_subdir = base_dir / "logs"
    if logs_subdir.is_dir():
        print(f"Note: Automatically using subdirectory {logs_subdir} as the log root")
        return logs_subdir
    return base_dir


def main():
    parser = argparse.ArgumentParser(description="Aggregate eval_results.json into a summary table")
    parser.add_argument("--dir", required=True, help="Log root directory, e.g. /app/data/libero/logs")
    parser.add_argument("--outputdir", required=True, help="Output directory; aggregated_results.csv will be generated here")
    args = parser.parse_args()

    raw_root = pathlib.Path(args.dir)
    if not raw_root.is_dir():
        print(f"Error: Directory does not exist - {raw_root}")
        sys.exit(1)

    log_root = find_logs_root(raw_root)
    output_dir = pathlib.Path(args.outputdir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(log_root.rglob("eval_results.json"))
    if not json_files:
        print(f"No eval_results.json found (search path: {log_root})")
        sys.exit(0)

    print(f"Found {len(json_files)} result files")

    rows = []
    for jf in json_files:
        # Get the path relative to log_root and strip any redundant "logs" prefix
        rel = jf.relative_to(log_root)
        parts = list(rel.parts)

        # If the first component is "logs" (when user passed a parent directory), strip it
        if parts and parts[0] == "logs":
            parts = parts[1:]

        # Expected structure: model / task_suite / domain_name / eval_results.json
        if len(parts) != 4:
            print(f"  Skipping file with incorrect path depth: {rel} (actual depth: {len(parts)})")
            continue

        model, task_suite, domain_name, filename = parts
        if filename != "eval_results.json":
            continue

        with open(jf, "r", encoding="utf-8") as f:
            data = json.load(f)

        total_ep = data.get("total_episodes", 0)
        total_succ = data.get("total_successes", 0)
        total_rate = data.get("total_success_rate", 0.0)

        rows.append({
            "model": model,
            "task_suite": task_suite,
            "domain": domain_name,
            "total_episodes": total_ep,
            "total_successes": total_succ,
            "success_rate": total_rate,
        })

    # Write CSV
    output_path = output_dir / "aggregated_results.csv"
    fieldnames = ["model", "task_suite", "domain", "total_episodes", "total_successes", "success_rate"]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Aggregation complete, results saved to: {output_path}")


if __name__ == "__main__":
    main()
