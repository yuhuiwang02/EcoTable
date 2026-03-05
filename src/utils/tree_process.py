import heapq

# ---------- Shortest path (with predecessor, for path recovery) ----------
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
    # If unreachable, return empty
    return path if path and path[0] == s else []

# ---------- Kruskal MST (undirected, possibly a subgraph) ----------
def kruskal_mst(nodes, edges):
    # edges: [(w,u,v)], undirected
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

# ---------- Validate if Steiner tree matches GT ----------
def validate_steiner_tree(steiner_result, gt_edges):
    """
    Validate if Steiner tree edges are all in ground truth
    Parameters:
    - steiner_result: return result from steiner_tree_2approx
    - gt_edges: ground truth edge set {(u, v), ...}
    Returns:
    - is_valid: bool, whether all edges are in GT
    - matched_edges: number of matched edges
    - total_edges: total number of Steiner tree edges
    """
    if not steiner_result or not steiner_result.get("edges"):
        return True, 0, 0  # Empty tree is considered valid

    steiner_edges = set()
    for edge in steiner_result["edges"]:
        u, v = edge["id1"], edge["id2"]
        # Normalize to unordered pair
        normalized_edge = tuple(sorted([u, v]))
        steiner_edges.add(normalized_edge)

    # Check if each edge is in GT
    matched = steiner_edges & gt_edges
    is_valid = steiner_edges.issubset(gt_edges)

    return is_valid, len(matched), len(steiner_edges)

# ---------- Build graph from edges ----------
def build_graph_from_edges(edges):
    """
    Build adjacency list graph from edge set
    edges: [(u, v, w), ...] or {(u, v): w, ...}
    Returns: {node: [(nbr, weight), ...]}
    """
    G = {}

    # Support two input formats
    if isinstance(edges, dict):
        edge_list = [(u, v, w) for (u, v), w in edges.items()]
    else:
        edge_list = edges

    for edge in edge_list:
        if len(edge) == 3:
            u, v, w = edge
        else:
            # Assume it's (u, v), weight is 1
            u, v = edge
            w = 1.0

        if u not in G:
            G[u] = []
        if v not in G:
            G[v] = []

        # Undirected graph, add bidirectionally
        G[u].append((v, w))
        G[v].append((u, w))

    return G

# ---------- 2-approximation Steiner tree main body ----------
def steiner_tree_2approx(G, gt):
    """
    G: {node: [(nbr, weight), ...]} (undirected graph needs both directions)
       or edge list [(u, v, w), ...]
       or edge dict {(u, v): w, ...}
    gt: terminal (required) node list/set
    Returns:
    {
      "nodes": [{"id": x, "table": "required"/"steiner"}, ...],
      "edges": [{"id1": u, "id2": v, "we": w}, ...],
      "total_weight": W
    }
    """
    # If input is edge set rather than adjacency list, build graph first
    if isinstance(G, (list, dict)):
        # Check if it's edge set format
        if isinstance(G, list) and len(G) > 0 and isinstance(G[0], (tuple, list)) and len(G[0]) >= 2:
            G = build_graph_from_edges(G)
        elif isinstance(G, dict) and len(G) > 0:
            # Check if it's edge set dict {(u,v): w}
            first_key = next(iter(G))
            if isinstance(first_key, tuple) and len(first_key) == 2:
                G = build_graph_from_edges(G)

    # If no nodes or terminal nodes are empty, return empty result
    if not G or not gt:
        return {"nodes": [], "edges": [], "total_weight": 0}

    T = list(gt)
    Tset = set(T)

    # 1) On the original graph, compute shortest paths from each terminal to all nodes
    all_dist = {}
    all_prev = {}
    for s in T:
        dist, prev = dijkstra_with_prev(G, s)
        all_dist[s] = dist
        all_prev[s] = prev

    # 2) Construct metric closure (terminal complete graph) edge set (u,v,shortest path length)
    metric_edges = []
    for i in range(len(T)):
        for j in range(i+1, len(T)):
            u, v = T[i], T[j]
            d = all_dist[u].get(v, float('inf'))
            if d < float('inf'):
                metric_edges.append((d, u, v))

    # 3) Find MST on the closure (connectivity assumption: terminals are mutually reachable)
    mt = kruskal_mst(T, metric_edges)  # [(w,u,v)]

    # 4) Replace each edge in closure MST with shortest path in original graph, take union of these paths to form subgraph H
    H_nodes = set()
    H_edges = {}  # Use unordered pair as key, record minimum weight  {(min(u,v),max(u,v)): w}
    for w,u,v in mt:
        # Use predecessor from u to restore u->v path
        path = restore_path(all_prev[u], u, v)
        if not path:
            # Safety fallback: if unable to recover due to input issues, skip this edge
            continue
        H_nodes.update(path)
        for a, b in zip(path, path[1:]):
            # Look up weight of (a,b) in original graph
            w_ab = None
            for nb, ww in G[a]:
                if nb == b:
                    w_ab = ww
                    break
            if w_ab is None:  # If input is not bidirectional symmetric, search from b again
                for nb, ww in G[b]:
                    if nb == a:
                        w_ab = ww
                        break
            key = (a,b) if a < b else (b,a)
            if w_ab is not None:
                if key not in H_edges or w_ab < H_edges[key]:
                    H_edges[key] = w_ab

    # 5) Remove cycles: do MST again on subgraph H to get acyclic tree T_H
    H_edge_list = [(w,u,v) for (u,v), w in H_edges.items()]
    TH = kruskal_mst(H_nodes, H_edge_list)  # [(w,u,v)]

    # 6) Steiner pruning: repeatedly delete leaves with degree 1 that are not in T
    #    First convert to adjacency count structure
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
        # Find the only adjacent node and "remove edge"
        (v,w), = adj[u]  # Only one element
        adj[u].clear()
        if (u,w) in adj[v]:
            adj[v].remove((u,w))  # But element here is (neighbor,weight), can't search like this
        # Correct deletion:
        for item in list(adj[v]):
            if item[0]==u:
                adj[v].remove(item)
                break
        deg[u]-=1
        deg[v]-=1
        if deg[v]==1 and v not in Tset:
            q.append(v)

    # 7) Aggregate remaining nodes and edges
    #    Nodes: degree > 0 or is terminal (single terminal may have degree 0)
    kept_nodes = {u for u in H_nodes if deg.get(u,0)>0 or u in Tset}
    kept_edges = []
    seen = set()
    for u in kept_nodes:
        for v,w in adj[u]:
            if u < v and (u,v) not in seen:
                kept_edges.append((u,v,w))
                seen.add((u,v))

    # 8) Organize output
    # Smart node ID conversion: convert to int if possible, otherwise keep as string
    def convert_node_id(node):
        try:
            return int(node)
        except (ValueError, TypeError):
            return node

    # Use string form when sorting to ensure consistency
    def sort_key(node):
        return str(node)

    nodes_out = [{"id": convert_node_id(u), "table": ("required" if u in Tset else "steiner")} for u in sorted(kept_nodes, key=sort_key)]
    edges_out = [{"id1": convert_node_id(u), "id2": convert_node_id(v), "we": float(w)} for (u,v,w) in kept_edges]
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

# Expected (for this graph):
# nodes roughly contain {0,2,3} (may also include intermediate nodes as Steiner points)
# edges form a tree; total_weight is the sum of these edge weights