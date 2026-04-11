#!/usr/bin/env python3
"""
aggregate_results.py
遍历 LOG_ROOT 下所有 eval_results.json，汇总为 CSV 表格。

目录约定:
    {LOG_ROOT}/{model}/{task_suite}/{domain}/eval_results.json

用法:
    python aggregate_results.py --log-root /app/data/libero/logs \
                                --output /app/data/libero/results_table.csv
"""
import argparse
import csv
import json
import pathlib


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-root", type=str, required=True)
    parser.add_argument("--output", type=str, default="results_table.csv")
    args = parser.parse_args()

    log_root = pathlib.Path(args.log_root)
    if not log_root.exists():
        print(f"[aggregate] 日志根目录不存在: {log_root}")
        return

    # 收集所有结果
    rows = []
    for json_file in sorted(log_root.rglob("eval_results.json")):
        # 路径: {log_root}/{model}/{task_suite}/{domain}/eval_results.json
        parts = json_file.relative_to(log_root).parts
        if len(parts) != 4:
            print(f"[aggregate] 跳过路径层级不符的文件: {json_file}")
            continue

        model, task_suite, domain, _ = parts

        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        total_sr = data.get("total_success_rate", 0.0)
        total_ep = data.get("total_episodes", 0)
        total_succ = data.get("total_successes", 0)

        rows.append({
            "model": model,
            "task_suite": task_suite,
            "domain": domain,
            "success_rate": total_sr,
            "successes": total_succ,
            "episodes": total_ep,
        })

        # 每个 task 的细粒度结果
        for task_desc, task_data in data.get("tasks", {}).items():
            rows_detail_key = f"{model}/{task_suite}/{domain}/{task_desc}"

    if not rows:
        print("[aggregate] 未找到任何 eval_results.json")
        return

    # 写 CSV
    output_path = pathlib.Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["model", "task_suite", "domain", "success_rate", "successes", "episodes"]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"[aggregate] CSV 已保存: {output_path}")
    print(f"[aggregate] 共 {len(rows)} 条记录")
    print()

    # 终端打印可读表格
    print(f"{'Model':<20} {'Task Suite':<18} {'Domain':<25} {'SR':>8} {'Succ':>6} {'Ep':>6}")
    print("─" * 90)
    for row in rows:
        sr_pct = f"{row['success_rate'] * 100:.1f}%"
        print(
            f"{row['model']:<20} {row['task_suite']:<18} {row['domain']:<25} "
            f"{sr_pct:>8} {row['successes']:>6} {row['episodes']:>6}"
        )


if __name__ == "__main__":
    main()