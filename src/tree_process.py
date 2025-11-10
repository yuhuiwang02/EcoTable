import heapq

# ---------- 最短路（含前驱，用于路径恢复） ----------
def dijkstra_with_prev(G, s):
    dist = {u: float('inf') for u in G}
    prev = {u: None for u in G}
    dist[s] = 0
    pq = [(0, s)]
    while pq:
        d, u = heapq.heappop(pq)
        if d != dist[u]:
            continue
        for v, w in G[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))
    return dist, prev

def restore_path(prev, s, t):
    path = []
    u = t
    while u is not None:
        path.append(u)
        if u == s: break
        u = prev[u]
    path.reverse()
    # 若无法到达，返回空
    return path if path and path[0] == s else []

# ---------- Kruskal MST（无向、可能是子图） ----------
def kruskal_mst(nodes, edges):
    # edges: [(w,u,v)], 无向
    parent = {u: u for u in nodes}
    rank   = {u: 0 for u in nodes}

    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u

    def union(a,b):
        ra, rb = find(a), find(b)
        if ra == rb: return False
        if rank[ra] < rank[rb]:
            parent[ra] = rb
        elif rank[ra] > rank[rb]:
            parent[rb] = ra
        else:
            parent[rb] = ra
            rank[ra] += 1
        return True

    mst = []
    for w,u,v in sorted(edges):
        if union(u,v):
            mst.append((w,u,v))
    return mst

# ---------- 2-近似斯坦纳树主体 ----------
def steiner_tree_2approx(G, gt):
    """
    G: {node: [(nbr, weight), ...]} (无向图需双方都给出)
    gt: 终端(必需)节点列表/集合
    返回:
    {
      "nodes": [{"id": x, "table": "required"/"steiner"}, ...],
      "edges": [{"id1": u, "id2": v, "we": w}, ...],
      "total_weight": W
    }
    """
    T = list(gt)
    Tset = set(T)

    # 1) 在原图上，计算每个终端到所有点的最短路
    all_dist = {}
    all_prev = {}
    for s in T:
        dist, prev = dijkstra_with_prev(G, s)
        all_dist[s] = dist
        all_prev[s] = prev

    # 2) 构造度量闭包（终端完全图）的边集（u,v,最短路长度）
    metric_edges = []
    for i in range(len(T)):
        for j in range(i+1, len(T)):
            u, v = T[i], T[j]
            d = all_dist[u].get(v, float('inf'))
            if d < float('inf'):
                metric_edges.append((d, u, v))

    # 3) 在闭包上求 MST（连通假设：终端间可互达）
    mt = kruskal_mst(T, metric_edges)  # [(w,u,v)]

    # 4) 将闭包 MST 的每条边替换回原图中的最短路径，取这些路径的并集形成子图 H
    H_nodes = set()
    H_edges = {}  # 用无序对作为键，记录最小权重  {(min(u,v),max(u,v)): w}
    for w,u,v in mt:
        # 用从 u 出发的前驱来还原 u->v 的路径
        path = restore_path(all_prev[u], u, v)
        if not path:
            # 安全兜底：若因输入问题无法恢复，跳过该边
            continue
        H_nodes.update(path)
        for a, b in zip(path, path[1:]):
            # 查原图中 (a,b) 的权重
            w_ab = None
            for nb, ww in G[a]:
                if nb == b:
                    w_ab = ww
                    break
            if w_ab is None:  # 若输入不是双向对称，这里再从 b 找一次
                for nb, ww in G[b]:
                    if nb == a:
                        w_ab = ww
                        break
            key = (a,b) if a < b else (b,a)
            if w_ab is not None:
                if key not in H_edges or w_ab < H_edges[key]:
                    H_edges[key] = w_ab

    # 5) 去环：在子图 H 上再做一次 MST，得到无环的树 T_H
    H_edge_list = [(w,u,v) for (u,v), w in H_edges.items()]
    TH = kruskal_mst(H_nodes, H_edge_list)  # [(w,u,v)]

    # 6) 斯坦纳修剪：反复删除度为1且不在 T 的叶子
    #    先转为邻接计数结构
    deg = {u:0 for u in H_nodes}
    adj = {u:set() for u in H_nodes}
    for w,u,v in TH:
        deg[u]+=1; deg[v]+=1
        adj[u].add((v,w)); adj[v].add((u,w))

    from collections import deque
    q = deque([u for u in list(H_nodes) if deg[u]==1 and u not in Tset])
    removed = set()
    while q:
        u = q.popleft()
        if deg[u] != 1 or u in Tset or u in removed:
            continue
        removed.add(u)
        # 找到唯一相邻点并“删边”
        (v,w), = adj[u]  # 只有一个元素
        adj[u].clear()
        if (u,w) in adj[v]:
            adj[v].remove((u,w))  # 但这里元素是 (neighbor,weight)，不能这样查
        # 正确删除：
        for item in list(adj[v]):
            if item[0]==u:
                adj[v].remove(item)
                break
        deg[u]-=1
        deg[v]-=1
        if deg[v]==1 and v not in Tset:
            q.append(v)

    # 7) 汇总剩余的节点与边
    #    节点：度>0 或者是终端（单个终端时会是0度）
    kept_nodes = {u for u in H_nodes if deg.get(u,0)>0 or u in Tset}
    kept_edges = []
    seen = set()
    for u in kept_nodes:
        for v,w in adj[u]:
            if u < v and (u,v) not in seen:
                kept_edges.append((u,v,w))
                seen.add((u,v))

    # 8) 组织输出
    nodes_out = [{"id": int(u), "table": ("required" if u in Tset else "steiner")} for u in sorted(kept_nodes)]
    edges_out = [{"id1": int(u), "id2": int(v), "we": float(w)} for (u,v,w) in kept_edges]
    total_weight = sum(w for _,_,w in kept_edges)

    return {
        "nodes": nodes_out,
        "edges": edges_out,
        "total_weight": total_weight
    }

# graph = {
#     0: [(1, 1), (2, 4), (3, 2)],
#     1: [(0, 1), (2, 0.5)],
#     2: [(0, 4), (1, 0.5), (3, 2)],
#     3: [(0, 2), (1, 3), (2, 2)]
# }
# gt = {0,2,3}

graph = {
    0: [(1, 2), (2, 8)],  # 节点0连接1权重2，连接2权重8
    1: [(0, 2), (2, 9), (3, 2), (6, 3)],  # 节点1连接0权重2，2权重9，3权重2, 7权重3
    2: [(0, 8), (1, 9), (4, 4), (5, 8)],  # 节点2连接0权重8，1权重9，4权重4，5权重8
    3: [(1, 2), (4, 1), (6, 2)],  # 节点3连接1权重2，4权重1，6权重2
    4: [(2, 4), (3, 1), (5, 3), (7, 5)],  # 节点4连接3权重1，5权重3，6权重5
    5: [(2, 8), (4, 3), (7, 7)],  # 节点5连接2权重8，4权重3，7权重7
    6: [(1, 3), (3, 2), (7, 5)],  # 节点6连接1权重3，3权重2，7权重5
    7: [(4, 5), (5, 7), (6, 5), (8, 8)],  # 节点7连接4权重5，5权重7，6权重5，8权重8
    8: [(6, 8), (7, 8)]  # 节点8连接6权重8，7权重8
}
required_nodes = {1, 2, 5, 6, 7}  # 必需节点集合


gt = steiner_tree_2approx(graph, required_nodes)
print(gt)
# 期望（此图）：
# nodes 大致包含 {0,2,3}（可能还包含作为斯坦纳点的中间节点）
# edges 形成一棵树；total_weight 为这些边权之和