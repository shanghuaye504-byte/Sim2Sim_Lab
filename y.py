#!/usr/bin/env python3
"""
aggregate_eval_results.py

遍历目录下的所有 eval_results.json，汇总总体指标并输出 CSV。

用法:
    python aggregate_eval_results.py --dir /path/to/libero/logs --outputdir ./summary
"""

import argparse
import csv
import json
import pathlib
import sys


def find_logs_root(base_dir: pathlib.Path) -> pathlib.Path:
    """如果 base_dir 本身就叫 logs，直接返回；否则检查其下是否有 logs 子目录。"""
    if base_dir.name == "logs":
        return base_dir
    logs_subdir = base_dir / "logs"
    if logs_subdir.is_dir():
        print(f"注意: 自动使用子目录 {logs_subdir} 作为日志根目录")
        return logs_subdir
    return base_dir


def main():
    parser = argparse.ArgumentParser(description="汇总 eval_results.json 生成统计表")
    parser.add_argument("--dir", required=True, help="日志根目录，例如 /app/data/libero/logs")
    parser.add_argument("--outputdir", required=True, help="输出目录，将在此目录下生成 aggregated_results.csv")
    args = parser.parse_args()

    raw_root = pathlib.Path(args.dir)
    if not raw_root.is_dir():
        print(f"错误: 目录不存在 - {raw_root}")
        sys.exit(1)

    log_root = find_logs_root(raw_root)
    output_dir = pathlib.Path(args.outputdir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(log_root.rglob("eval_results.json"))
    if not json_files:
        print(f"未找到任何 eval_results.json（搜索路径: {log_root}）")
        sys.exit(0)

    print(f"找到 {len(json_files)} 个结果文件")

    rows = []
    for jf in json_files:
        # 获取相对于 log_root 的路径，并去掉可能多余的 "logs" 前缀（如果 log_root 本身就是 logs，则不会有多余）
        rel = jf.relative_to(log_root)
        parts = list(rel.parts)

        # 如果第一个部分是 "logs"（当用户传了父目录时），则去掉它
        if parts and parts[0] == "logs":
            parts = parts[1:]

        # 期望结构: model / task_suite / domain_name / eval_results.json
        if len(parts) != 4:
            print(f"  跳过路径层级不符的文件: {rel} (实际层级: {len(parts)})")
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

    # 写入 CSV
    output_path = output_dir / "aggregated_results.csv"
    fieldnames = ["model", "task_suite", "domain", "total_episodes", "total_successes", "success_rate"]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"汇总完成，结果已保存至: {output_path}")


if __name__ == "__main__":
    main()
