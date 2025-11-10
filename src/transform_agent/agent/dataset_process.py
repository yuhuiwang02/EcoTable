import argparse
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import json
import re
import shutil
import duckdb

query_json=""
def id_to_table(table_names: List[str], node_id: int) -> str:
    """根据节点 id 获取表名（假设 id 为 1-based）"""
    idx = node_id
    if idx < 0 or idx >= len(table_names):
        raise IndexError(f"节点 id={node_id} 超出范围（len={len(table_names)}）")
    return table_names[idx]


def ensure_duckdb_with_tables(
    db_path: Path, csv_root: Path, s: str, table_names_needed: List[str], copy_to: Optional[Path] = None
):
    """创建/更新 duckdb 并加载需要的 csv 表"""
    csv_dir = csv_root / s
    if not csv_dir.is_dir():
        raise FileNotFoundError(f"CSV 目录不存在: {csv_dir}")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    try:
        for t in table_names_needed:
            csv_file = csv_dir / f"{t}.csv"
            if not csv_file.is_file():
                raise FileNotFoundError(f"缺失 CSV 文件: {csv_file}")
            if copy_to:
                dst = copy_to / "csv" / s
                dst.mkdir(parents=True, exist_ok=True)
                shutil.copy2(csv_file, dst / csv_file.name)
            con.execute(
                f"""CREATE OR REPLACE TABLE "{t}" AS 
                SELECT * FROM read_csv_auto(?, IGNORE_ERRORS=true);""",
                [str(csv_file)],
            )
    finally:
        con.close()


def parse_true_candidates(candidates: Dict[str, object],change:bool) -> List[Tuple[str, str]]:
    """提取 candidates 中值为 True 的列对"""
    pairs = []
    for k, v in candidates.items():
        if v is True or (isinstance(v, str) and v.lower() == "true"):
            sep = "↔" if "↔" in k else "<->" if "<->" in k else "->" if "->" in k else "-"
            if sep in k:
                left, right = k.split(sep, 1)
                if change:
                    left,right=right,left
                pairs.append((left.strip(), right.strip()))
    return pairs


def load_edge_candidates(edge_root: Path, s: str, t1: str, t2: str, copy_to: Optional[Path] = None) -> List[Tuple[str, str]]:
    """读取 {edge_root}/{s}/{t1}-{t2}.json 并解析为列对"""
    f = edge_root / s / f"{t1}-{t2}.json"
    if not f.exists():
        raise FileNotFoundError(f"缺少边文件: {f}")
    if copy_to:
        dst = copy_to / "edge_judge" / s
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, dst / f.name)
    data = json.load(open(f, "r", encoding="utf-8"))
    if not isinstance(data, list) or not data:
        return []
    if t1==data[0].get("table2", {}):
        return parse_true_candidates(data[0].get("candidates", {}),True)
    return parse_true_candidates(data[0].get("candidates", {}),False)


def process_join_graph(
    input_dir: str,
    db_out_dir: str,
    csv_root: str = "./csv",
    edge_root: str = "./LLMjudge",
    out_json: Optional[str] = None,
    copy_to: Optional[str] = None,
) -> List[Dict]:
    """
    解析 {s}-{num}.json，创建 DuckDB 并抽取连接列关系。
    """
    input_dir = Path(input_dir).resolve()
    db_out_dir = Path(db_out_dir).resolve()
    csv_root = Path(csv_root).resolve()
    edge_root = Path(edge_root).resolve()
    copy_to = Path(copy_to).resolve() if copy_to else None

    s = input_dir.name
    pattern = re.compile(rf"^{re.escape(s)}-(\d+)\.json$")
    files = sorted(
        [p for p in input_dir.iterdir() if pattern.match(p.name)],
        key=lambda x: int(pattern.match(x.name).group(1)),
    )
    if not files:
        raise FileNotFoundError(f"未找到 {s}-{{num}}.json 文件于 {input_dir}")
    json_of_query = json.load(open(query_json, "r", encoding="utf-8"))
    project=json_of_query.get(f"{s}",[])
    results = []
    idx=0
    for f in files:
        obj = json.load(open(f, "r", encoding="utf-8"))
        tables = obj.get("table_name", [])
        edges = obj.get("gt", {}).get("edges", [])
        nodes = obj.get("gt", {}).get("nodes", [])

        node_tables = {id_to_table(tables, n["id"]) for n in nodes if "id" in n}
        ensure_duckdb_with_tables(db_out_dir / f"{s}.duckdb", csv_root, s, list(node_tables), copy_to)

        edge_map = {}
        for e in edges:
            if "id1" in e and "id2" in e:
                id1, id2 = e["id1"], e["id2"]
                # 按 id 升序决定“表1-表2”的顺序
                if id1 > id2:
                    id1, id2 = id2, id1
                t1, t2 = id_to_table(tables, id1), id_to_table(tables, id2)

                pairs = load_edge_candidates(edge_root, s, t1, t2, copy_to)
                if pairs:
                    edge_map[f"{t1}-{t2}"] = pairs
                
        results.append({"query": project[idx].get("query"), "tables": tables, "edges": edge_map})
        print(f"[OK] {f.name} processed.")
        idx+=1

    if out_json is None:
        out_json = str((input_dir / f"{s}_edges.json").resolve())
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[DONE] 汇总文件已生成: {out_json}")
    return results


def main():
    parser = argparse.ArgumentParser(description="解析 join 结构并创建 DuckDB")
    parser.add_argument("input_dir", help="输入目录（包含 {s}-{num}.json）")
    parser.add_argument("--db-out-dir", required=True, help="DuckDB 输出目录")
    parser.add_argument("--csv-root", default="/data/new_dbt_dataset/111test/csv", help="CSV 根目录")
    parser.add_argument("--edge-root", default="/data/new_dbt_dataset/111test/LLMjudgeair", help="边文件根目录")
    parser.add_argument("--out-json", default=None, help="输出汇总 JSON 路径")
    parser.add_argument("--copy-to", default=None, help="可选：复制用到的文件到该目录")

    args = parser.parse_args()

    process_join_graph(
        input_dir=args.input_dir,
        db_out_dir=args.db_out_dir,
        csv_root=args.csv_root,
        edge_root=args.edge_root,
        out_json=args.out_json,
        copy_to=args.copy_to,
    )


if __name__ == "__main__":
    main()
