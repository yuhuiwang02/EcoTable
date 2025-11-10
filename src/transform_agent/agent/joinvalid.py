#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd
import argparse


def compare_overlap(table_dir: str, table_name1: str, table_name2: str, col_name1: str, col_name2: str) -> float:
    """
    计算两个 CSV 文件中指定列的最大重合度 (overlap_coefficient)。
    overlap_coefficient = |A∩B| / min(|A|, |B|)

    参数:
        table_dir : CSV 文件所在目录
        table_name1, table_name2 : CSV 文件名（可不带 .csv）
        col_name1, col_name2 : 各文件的列名

    返回:
        float : 最大重合度
    """
    table_dir = Path(table_dir)

    def ensure_csv(name: str) -> Path:
        if not name.lower().endswith(".csv"):
            name += ".csv"
        return table_dir / name

    f1, f2 = ensure_csv(table_name1), ensure_csv(table_name2)

    if not f1.exists():
        raise FileNotFoundError(f"未找到文件: {f1}")
    if not f2.exists():
        raise FileNotFoundError(f"未找到文件: {f2}")

    # 只读取目标列
    df1 = pd.read_csv(f1, usecols=[col_name1])
    df2 = pd.read_csv(f2, usecols=[col_name2])

    A = set(df1[col_name1].dropna())
    B = set(df2[col_name2].dropna())

    if not A or not B:
        return 0.0

    intersection = len(A & B)
    overlap = intersection / min(len(A), len(B))
    return overlap


# —— 命令行执行接口 ——
def main():
    parser = argparse.ArgumentParser(description="计算两个 CSV 指定列的最大重合度（overlap_coefficient）")
    parser.add_argument("table_dir", help="CSV 文件目录")
    parser.add_argument("table_name1", help="CSV 文件1 名称（可不带 .csv）")
    parser.add_argument("table_name2", help="CSV 文件2 名称（可不带 .csv）")
    parser.add_argument("col_name1", help="文件1中的列名")
    parser.add_argument("col_name2", help="文件2中的列名")
    args = parser.parse_args()

    coef = compare_overlap(args.table_dir, args.table_name1, args.table_name2, args.col_name1, args.col_name2)
    print(f"overlap_coefficient = {coef:.6f}")


if __name__ == "__main__":
    main()
