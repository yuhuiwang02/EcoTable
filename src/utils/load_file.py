import json
import sys
import os
import math
import csv
from collections import defaultdict
import networkx as nx
from networkx.readwrite import json_graph
from itertools import combinations
from tqdm import tqdm

def normalize_str(s):
    """Normalize string: strip whitespace and unify path separators"""
    if not isinstance(s, str):
        return str(s)
    return s.strip().replace('\\', '/')

def extract_table_name(path):
    """Extract table name from path (remove .csv extension)"""
    table_name = normalize_str(path)
    if table_name.endswith('.csv'):
        table_name = table_name[:-4]
    return table_name


def load_json_data(json_path):
    """Load JSON file"""
    if not os.path.exists(json_path):
        print(f"[Error] JSON file does not exist: {json_path}")
        sys.exit(1)

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            if 'table1' in data and 'prob' in data:
                return [data]
            return list(data.values())
        else:
            print(f"[Error] Unrecognized JSON format: {type(data)}")
            return []
    except Exception as e:
        print(f"[Error] Failed to read JSON file: {e}")
        sys.exit(1)

def build_global_graph(json_data, verbose=False):
    """
    Build global graph from JSON data
    - Nodes: table names
    - Edge weights: negative log of probability of highest-prob column pair between two tables (-log(prob))
      Lower weight indicates higher probability, suitable for shortest path algorithms
    Returns: NetworkX graph object and edge details dictionary
    """
    G = nx.Graph()
    edge_details = defaultdict(lambda: {'prob': 0.0, 'column1': '', 'column2': ''})

    for item in json_data:
        t1 = extract_table_name(item.get('table1', ''))
        t2 = extract_table_name(item.get('table2', ''))
        c1 = normalize_str(item.get('column1', ''))
        c2 = normalize_str(item.get('column2', ''))
        prob = item.get('prob', 0.0)

        if not t1 or not t2:
            continue

        edge = tuple(sorted([t1, t2]))

        if prob > edge_details[edge]['prob']:
            edge_details[edge] = {
                'prob': prob,
                'column1': c1,
                'column2': c2,
                'table1': t1,
                'table2': t2
            }

    for (t1, t2), details in edge_details.items():
        prob = details['prob']
        if prob <= 0:
            prob = 1e-10
        weight = -math.log(prob)

        G.add_edge(t1, t2, weight=weight)
        if verbose:
            print(f"[Graph] {t1} <-> {t2}, prob={prob:.4f}, weight={weight:.4f} ({details['column1']} - {details['column2']})")

    print(f"Global graph constructed: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G, dict(edge_details)



def load_split_statistics(split_stats_path, verbose=False):
    """
    Load split_statistics.json
    Returns:
    - split_stats: {base_table_name: node_mapping_dict}
    """
    if not os.path.exists(split_stats_path):
        print(f"[Error] split_statistics.json file does not exist: {split_stats_path}")
        sys.exit(1)

    try:
        with open(split_stats_path, 'r', encoding='utf-8') as f:
            stats_data = json.load(f)

        if not isinstance(stats_data, dict) or 'normal' not in stats_data:
            print(f"[Error] split_statistics.json should be a dict structure with 'normal' key")
            sys.exit(1)

        split_stats = {}

        for table_info in stats_data['normal']:
            table_name = table_info.get('table_name', '')
            node_mapping = table_info.get('node_mapping', {})

            if table_name and node_mapping:
                split_stats[table_name] = node_mapping

                if verbose:
                    print(f"[Split Stats] {table_name}: {len(node_mapping)} nodes")

        print(f"Split Statistics loaded: {len(split_stats)} base tables")
        return split_stats

    except Exception as e:
        print(f"[Error] Failed to read split_statistics.json: {e}")
        sys.exit(1)

def convert_nodes_to_subtables(base_table_name, user_selected_nodes, split_stats, verbose=False):
    """
    Convert user_selected_nodes to subtable paths
    Args:
    - base_table_name: base table name
    - user_selected_nodes: list of numeric strings, e.g. ["2", "3"]
    - split_stats: split_statistics data
    Returns:
    - subtable_paths: set of subtable paths
    """
    subtable_paths = []

    if base_table_name not in split_stats:
        if verbose:
            print(f"[Warning] Base table {base_table_name} not in split_statistics")
        return subtable_paths

    node_mapping = split_stats[base_table_name]

    for node_id in user_selected_nodes:
        if node_id in node_mapping:
            subtable_file = node_mapping[node_id].get('subtable_file', '')
            if subtable_file:
                subtable_path = f"{base_table_name}_normal/{subtable_file}"
                subtable_paths.append(subtable_path)

                if verbose:
                    print(f"[Convert] Node {node_id} -> {subtable_path}")
        else:
            if verbose:
                print(f"[Warning] Node {node_id} not in {base_table_name}'s node_mapping")

    return subtable_paths

def find_k_shortest_paths(G, source, target, k=10, max_path_length=5):
    """
    Find k shortest paths using NetworkX (sorted by weight)
    Weight is -log(prob), so shortest path corresponds to highest probability path

    Optimization strategies:
    1. Limit path length to avoid searching overly long paths in large graphs
    2. Use generator to compute next path only when needed
    3. Set search limit to avoid infinite search

    Returns list of paths, each path is a list of nodes
    """
    try:
        if source not in G or target not in G:
            return []

        paths = []
        path_generator = nx.shortest_simple_paths(G, source, target, weight='weight')

        valid_path_count = 0
        total_checked = 0
        max_search_limit = max(100, k * 20)

        for path in path_generator:
            total_checked += 1

            path_length = len(path) - 1

            if path_length <= max_path_length:
                paths.append(path)
                valid_path_count += 1

                if valid_path_count >= k:
                    break

            if total_checked >= max_search_limit:
                if valid_path_count > 0:
                    break
                else:
                    break

        return paths

    except nx.NetworkXNoPath:
        return []
    except nx.NodeNotFound:
        return []
    except StopIteration:
        return paths
    except Exception as e:
        print(f"[Warning] Error finding paths ({source} -> {target}): {e}")
        return []

def path_to_edges(path):
    """Convert path (list of nodes) to set of edges"""
    edges = set()
    for i in range(len(path) - 1):
        edge = tuple(sorted([path[i], path[i+1]]))
        edges.add(edge)
    return edges

def calculate_path_weight(G, path):
    """
    Calculate total weight of a path
    Args:
    - G: global graph
    - path: path (list of nodes)
    Returns:
    - total weight of path
    """
    total_weight = 0.0
    for i in range(len(path) - 1):
        if G.has_edge(path[i], path[i+1]):
            total_weight += G[path[i]][path[i+1]]['weight']
        else:
            return float('inf')
    return total_weight


def count_paths_between_nodes(G, source, target, max_paths=100, cutoff=6):
    """
    Count simple paths between source and target in graph G
    For efficiency, count at most max_paths and only consider paths with length <= cutoff
    """
    try:
        if source not in G or target not in G:
            return 0

        count = 0
        for _ in nx.all_simple_paths(G, source, target, cutoff=cutoff):
            count += 1
            if count >= max_paths:
                break
        return count
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return 0

def prune_paths_global(G, all_paths_with_info, k, t=1.0, verbose=False):
    """
    Global pruning: sort all paths by weight, gradually add to new graph

    Args:
    - G: original global graph
    - all_paths_with_info: list of all path info [{'path': [...], 'weight': ..., 'pair': (t1, t2)}, ...]
    - k: maximum k paths to retain per table pair
    - t: t-spanner parameter
    - verbose: whether to print detailed output

    Returns:
    - pruned_graph: pruned graph (NetworkX graph object)
    - pruned_paths_by_pair: {(t1, t2): [path1, path2, ...]} pruned path set
    """
    if verbose:
        print(f"\n[Global Pruning] Starting global pruning, k={k}, t={t}")
        print(f"[Global Pruning] Total input paths: {len(all_paths_with_info)}")

    sorted_paths = sorted(all_paths_with_info, key=lambda x: x['weight'])

    Gs = nx.Graph()

    pruned_paths_by_pair = defaultdict(list)

    for idx, path_info in enumerate(tqdm(sorted_paths,total=len(sorted_paths))):
        path = path_info['path']
        weight = path_info['weight']

        if len(path) < 2:
            continue

        source = path[0]
        target = path[-1]
        pair = tuple(sorted([source, target]))

        try:
            if source in Gs and target in Gs and nx.has_path(Gs, source, target):
                k_path =find_k_shortest_paths(Gs, source, target, k=k, max_path_length=100)
                if calculate_path_weight(Gs, k_path[-1]) <= weight * t and len(k_path) >= k :
                    if verbose:
                        print(f"[Global Pruning] Skipping path {' -> '.join(path)}: weight*t={weight*t:.4f}")
                    continue
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            pass

        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            if G.has_edge(u, v):
                edge_weight = G[u][v]['weight']
                if Gs.has_edge(u, v):
                    Gs[u][v]['weight'] = min(Gs[u][v]['weight'], edge_weight)
                else:
                    Gs.add_edge(u, v, weight=edge_weight)

        pruned_paths_by_pair[pair].append(path)

        if verbose:
            print(f"[Global Pruning] Added path {' -> '.join(path)} (weight={weight:.4f})")
            print(f"  Endpoint table pair: {source} <-> {target}")

    if verbose:
        print(f"\n[Global Pruning] Pruning complete")
        print(f"  New graph nodes: {Gs.number_of_nodes()}")
        print(f"  New graph edges: {Gs.number_of_edges()}")
        print(f"  Covered table pairs: {len(pruned_paths_by_pair)}")

    return Gs, dict(pruned_paths_by_pair)

def collect_paths_for_queries(G, base_table_names, table_queries, split_stats, k=10, max_path_length=5, verbose=False):
    """
    Collect top-k paths for all queries

    Args:
    - G: global graph
    - base_table_names: set of base table names
    - table_queries: {base_table_name: [list of user_selected_nodes lists]}
    - split_stats: split_statistics data
    - k: number of shortest paths to retain per table pair
    - max_path_length: maximum path length (edge count)
    - verbose: whether to print detailed output

    Returns:
    - all_paths_with_info: [{'path': [...], 'weight': ..., 'pair': (t1, t2)}, ...]
    """
    print(f"\n[Path Collection] Collecting Top-{k} paths...")

    all_paths_with_info = []
    seen_paths = set()
    seen_pairs = set()

    for base_table_name in tqdm(base_table_names, total=len(base_table_names), desc=f"Collecting paths k={k}"):
        if base_table_name not in table_queries:
            continue

        queries = table_queries[base_table_name]

        for query_idx, user_selected_nodes in enumerate(queries):
            subtable_paths = convert_nodes_to_subtables(base_table_name, user_selected_nodes, split_stats, verbose=False)

            if len(subtable_paths) < 2:
                continue

            for st1, st2 in combinations(subtable_paths, 2):
                t1 = extract_table_name(st1)
                t2 = extract_table_name(st2)
                pair = tuple(sorted([t1, t2]))

                if pair in seen_pairs:
                    continue

                seen_pairs.add(pair)

                paths = find_k_shortest_paths(G, t1, t2, k=k, max_path_length=max_path_length)

                for path in paths:
                    path_tuple = tuple(path)
                    if path_tuple not in seen_paths:
                        seen_paths.add(path_tuple)
                        weight = calculate_path_weight(G, path)
                        all_paths_with_info.append({
                            'path': path,
                            'weight': weight,
                            'pair': pair
                        })

    print(f"[Path Collection] Collected {len(all_paths_with_info)} unique paths, processed {len(seen_pairs)} unique table pairs")
    return all_paths_with_info

def build_pruned_graph(G, base_table_names, table_queries, split_stats, k=10, max_path_length=5, t_spanner=1.0, verbose=False):
    """
    Build pruned global graph

    Args:
    - G: original global graph
    - base_table_names: set of base table names
    - table_queries: {base_table_name: [list of user_selected_nodes lists]}
    - split_stats: split_statistics data
    - k: number of shortest paths to retain per table pair
    - max_path_length: maximum path length (edge count)
    - t_spanner: t-spanner parameter
    - verbose: whether to print detailed output

    Returns:
    - pruned_graph: pruned NetworkX graph object
    """
    all_paths_with_info = collect_paths_for_queries(
        G, base_table_names, table_queries, split_stats, k, max_path_length, verbose
    )

    print(f"\n[Global Pruning] Starting pruning, k={k}, t={t_spanner}...")

    edges_before = set()
    for path_info in all_paths_with_info:
        edges_before.update(path_to_edges(path_info['path']))

    pruned_graph, pruned_paths = prune_paths_global(
        G, all_paths_with_info, k, t=t_spanner, verbose=verbose
    )

    edges_after = set()
    for pair, paths in pruned_paths.items():
        for path in paths:
            edges_after.update(path_to_edges(path))

    print(f"\n[Pruning Statistics]:")
    print(f"  Before pruning: {len(all_paths_with_info)} paths, {len(edges_before)} edges")
    print(f"  After pruning: {sum(len(paths) for paths in pruned_paths.values())} paths, {len(edges_after)} edges")
    if len(edges_before) > 0:
        reduction = (1 - len(edges_after)/len(edges_before))*100
        print(f"  Edge reduction: {len(edges_before) - len(edges_after)} ({reduction:.2f}%)")

    return pruned_graph

def save_pruned_graph(pruned_graph, output_path):
    """
    Save pruned graph to JSON file

    Args:
    - pruned_graph: NetworkX graph object
    - output_path: output file path
    """
    graph_data = json_graph.node_link_data(pruned_graph)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(graph_data, f, indent=2, ensure_ascii=False)

    print(f"\n[Save] Pruned graph saved to: {output_path}")
    print(f"  Nodes: {pruned_graph.number_of_nodes()}")
    print(f"  Edges: {pruned_graph.number_of_edges()}")

def load_ground_truth(csv_path, verbose=False):
    """
    Load ground truth from CSV, extract folder names and remove _normal suffix to get base table names
    Returns:
    - base_table_names: set of base table names (folder names with _normal suffix removed)
    - gt_edges: {(table1, table2)} ground truth edge set (unordered pairs) - retained for possible future use
    - gt_edges_by_basetable: {base_table_name: {(table1, table2)}} ground truth edges organized by base table
    """
    if not os.path.exists(csv_path):
        print(f"[Error] CSV file does not exist: {csv_path}")
        sys.exit(1)

    base_table_names = set()
    gt_edges = set()
    gt_edges_by_basetable = defaultdict(set)

    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)

            for row in reader:
                if not row or len(row) < 4:
                    continue

                t1_path = normalize_str(row[0])
                t2_path = normalize_str(row[1])

                t1 = extract_table_name(t1_path)
                t2 = extract_table_name(t2_path)

                folder1 = extract_folder_name(t1_path)
                folder2 = extract_folder_name(t2_path)

                if folder1:
                    base_name = folder1.replace('_normal', '')
                    base_table_names.add(base_name)

                if folder2:
                    base_name = folder2.replace('_normal', '')
                    base_table_names.add(base_name)

                if folder1 and folder1 == folder2:
                    edge = tuple(sorted([t1, t2]))
                    gt_edges.add(edge)

                    base_name = folder1.replace('_normal', '')
                    gt_edges_by_basetable[base_name].add(edge)

                    if verbose:
                        print(f"[GT] {t1} <-> {t2} (folder: {folder1}, base: {base_name})")

    except Exception as e:
        print(f"[Error] Error processing CSV file: {e}")
        sys.exit(1)

    print(f"Ground Truth loaded: {len(base_table_names)} base tables, {len(gt_edges)} edges")

    if verbose:
        for base_name in sorted(gt_edges_by_basetable.keys()):
            print(f"  [GT Stats] {base_name}: {len(gt_edges_by_basetable[base_name])} edges")

    return base_table_names, gt_edges, dict(gt_edges_by_basetable)
def extract_folder_name(path):
    """Extract folder name from path"""
    normalized = normalize_str(path)
    if '/' in normalized:
        return normalized.split('/')[0]
    return None
