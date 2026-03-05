import json
from collections import Counter

# ---- 加载数据 ----
with open('F:\workspace\join_pipeline\dbt\gt_paths.json', 'r', encoding='utf-8') as f:
    gt_paths = json.load(f)

with open('F:\workspace\join_pipeline\evaluation_results_dbt.json', 'r', encoding='utf-8') as f:
    eval_results = json.load(f)

eval_by_idx = {item['query_idx']: item for item in eval_results}

# ---- 评测 ----
def normalize_edge(e):
    return tuple(sorted(e))

total = 0
precision_sum = 0.0
recall_sum = 0.0
f1_sum = 0.0
status_counter = Counter()
perfect = 0

for gt_item in gt_paths:
    oi = gt_item['orig_index']
    total += 1

    ev = eval_by_idx.get(oi)
    if ev is None:
        status_counter['missing'] += 1
        continue

    gt_edges = set(normalize_edge(e) for e in gt_item['gt_edges'])
    steiner_edges = set(normalize_edge(e) for e in ev.get('steiner_edges', []))

    status = ev.get('status', '')
    status_counter[status] += 1

    if not gt_edges:
        continue

    if not steiner_edges:
        continue

    tp = len(gt_edges & steiner_edges)
    prec = tp / len(steiner_edges) if steiner_edges else 0
    rec = tp / len(gt_edges) if gt_edges else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

    precision_sum += prec
    recall_sum += rec
    f1_sum += f1

    if prec == 1.0 and rec == 1.0:
        perfect += 1

# ---- 输出 ----
print(f"评测结果 (共 {total} 条 query)")
print(f"=" * 50)
print(f"状态分布:")
for s, c in status_counter.most_common():
    print(f"  {s}: {c} ({c/total*100:.1f}%)")
print()
print(f"平均 Edge Precision: {precision_sum/total:.4f}")
print(f"平均 Edge Recall:    {recall_sum/total:.4f}")
print(f"平均 Edge F1:        {f1_sum/total:.4f}")
print(f"完美匹配 (P=1,R=1): {perfect}/{total} ({perfect/total*100:.1f}%)")
