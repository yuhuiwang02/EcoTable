import json
import sys
from typing import Dict, Tuple, List, Set


def _norm_pair(a, b) -> Tuple[int, int]:
    """将边标准化为 (min_id, max_id)，用于无向比较"""
    a, b = int(a), int(b)
    return (a, b) if a <= b else (b, a)


def _parse_join_path_edge(s: str) -> Tuple[int, int]:
    """解析 join_path 元素 '5-4' → (4,5)"""
    s = s.strip()
    left, right = s.split("-", 1)
    return _norm_pair(left.strip(), right.strip())


def evaluate_edges(data: Dict) -> Dict:
    """计算匹配情况与正确率"""
    edge_set: Set[Tuple[int, int]] = set()
    for e in data.get("gt", {}).get("edges", []):
        edge_set.add(_norm_pair(e["id1"], e["id2"]))

    jp_edges: List[Tuple[int, int]] = []
    for s in data.get("join_path", []):
        jp_edges.append(_parse_join_path_edge(s))

    matches = []
    misses = []
    for e in jp_edges:
        (matches if e in edge_set else misses).append(e)

    total = len(jp_edges)
    correct = len(matches)
    accuracy = correct / total if total > 0 else 0.0

    return {
        "total_join_path_edges": total,
        "matched_count": correct,
        "missed_count": len(misses),
        "accuracy": accuracy,
        "matched_edges": matches,
        "missed_edges": misses,
    }


def check_edges_from_file(json_path: str) -> Dict:
    """
    读取 JSON 文件并检测匹配结果。

    参数:
        json_path: str - JSON 文件路径

    返回:
        dict，包含匹配详情与正确率。
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return evaluate_edges(data)


def main():
    if len(sys.argv) < 2:
        print("用法: python check_edges.py /path/to/input.json")
        sys.exit(1)

    json_path = sys.argv[1]
    result = check_edges_from_file(json_path)
    percent = f"{result['accuracy']*100:.2f}%"

    print(json.dumps({
        "total_join_path_edges": result["total_join_path_edges"],
        "matched_count": result["matched_count"],
        "missed_count": result["missed_count"],
        "accuracy_percent": percent,
        "matched_edges": result["matched_edges"],
        "missed_edges": result["missed_edges"]
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
