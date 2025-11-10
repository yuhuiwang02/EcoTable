from pathlib import Path
import argparse
import duckdb


def _quote_ident(ident: str) -> str:
    """
    给标识符加上 DuckDB 的双引号引用，支持 schema.table 形式。
    例如: page -> "page"; analytics.page -> "analytics"."page"
    """
    parts = [p for p in ident.split(".") if p]
    return ".".join(f'"{p}"' for p in parts) if parts else '""'


def _distinct_non_null_set(con: duckdb.DuckDBPyConnection, table: str, col: str) -> set:
    """
    从 DuckDB 中取某表某列的非空去重集合。
    """
    t = _quote_ident(table)
    c = _quote_ident(col)
    sql = f"SELECT DISTINCT {c} AS v FROM {t} WHERE {c} IS NOT NULL"
    rows = con.execute(sql).fetchall()
    return {r[0] for r in rows}


def compare_overlap_duckdb(db_path: str, table_name1: str, table_name2: str,
                           col_name1: str, col_name2: str) -> float:
    """
    在 DuckDB 中计算两个表两列的最大重合度 (overlap_coefficient)：
        overlap = |A∩B| / min(|A|, |B|)
    不做大小写/空白/空值等额外处理，仅排除 NULL。
    """
    db_path = Path(db_path)
    if not db_path.is_file():
        raise FileNotFoundError(f"未找到 DuckDB 文件: {db_path}")

    con = duckdb.connect(str(db_path), read_only=True)
    try:
        A = _distinct_non_null_set(con, table_name1, col_name1)
        B = _distinct_non_null_set(con, table_name2, col_name2)

        if not A or not B:
            return 0.0

        inter = len(A & B)
        overlap = inter / min(len(A), len(B))
        return float(overlap)
    finally:
        con.close()


# —— 命令行执行接口 ——
def main():
    parser = argparse.ArgumentParser(description="在 DuckDB 中计算两个表两列的最大重合度（overlap_coefficient）")
    parser.add_argument("duckdb_path", help="DuckDB 文件路径，例如 /path/to/db.duckdb")
    parser.add_argument("table_name1", help="表1名称（可带 schema，如 analytics.page）")
    parser.add_argument("table_name2", help="表2名称（可带 schema，如 analytics.post_history）")
    parser.add_argument("col_name1", help="表1中的列名（可带点号，如 schema.列名 中只需列名本身即可）")
    parser.add_argument("col_name2", help="表2中的列名")
    args = parser.parse_args()

    coef = compare_overlap_duckdb(
        args.duckdb_path,
        args.table_name1,
        args.table_name2,
        args.col_name1,
        args.col_name2,
    )
    print(f"overlap_coefficient = {coef:.6f}")


if __name__ == "__main__":
    main()
