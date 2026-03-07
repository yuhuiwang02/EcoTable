"""
Join Evaluation Script with LLM-based Verification

This script evaluates join relationships using:
1. Merging user_selected_nodes with containment relationships
2. Building subgraphs for merged sub_tables
3. LLM-based edge verification (coarse + fine)
4. Voting mechanism for critical edges
5. Steiner tree generation for queries
"""

import json
import sys
import os
import argparse
from collections import defaultdict, Counter
from typing import Dict, List, Set, Tuple, Optional
import networkx as nx
from networkx.readwrite import json_graph
from tqdm import tqdm
from itertools import combinations


from utils.load_file import (
    normalize_str, extract_table_name,
    load_json_data, build_global_graph, load_ground_truth,
    find_k_shortest_paths, calculate_path_weight,
    prune_paths_global
)

from utils.llm_verifier import LLMVerifier
from utils.edge_validator import EdgeValidator
from utils.tree_process import steiner_tree_2approx


def convert_networkx_to_adjacency_list(graph: nx.Graph) -> Dict:
    """
    Convert NetworkX graph to adjacency list format for steiner_tree_2approx

    Args:
        graph: NetworkX graph object

    Returns:
        Adjacency list format: {node: [(neighbor, weight), ...]}
    """
    adj_list = {}
    for node in graph.nodes():
        adj_list[node] = []
        for neighbor in graph.neighbors(node):
            weight = graph[node][neighbor].get('weight', 1.0)
            adj_list[node].append((neighbor, weight))
    return adj_list


def save_pruned_graph(pruned_graph: nx.Graph, cache_file: str,
                      metadata: Dict):
    """
    Save pruned graph to cache file

    Args:
        pruned_graph: Pruned graph
        cache_file: Cache file path
        metadata: Metadata (including k, t_spanner, max_path_length, etc.)
    """
    try:
        graph_data = json_graph.node_link_data(pruned_graph, edges="edges")

        cache_data = {
            'metadata': metadata,
            'graph': graph_data,
            'node_count': pruned_graph.number_of_nodes(),
            'edge_count': pruned_graph.number_of_edges()
        }

        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, indent=2, ensure_ascii=False)

        print(f"[Cache] Pruned graph saved to: {cache_file}")
        print(f"  Nodes: {cache_data['node_count']}, Edges: {cache_data['edge_count']}")
    except Exception as e:
        print(f"[Warning] Failed to save pruned graph cache: {e}")


def save_pruned_graph_with_join_cols(pruned_graph: nx.Graph, verified_edges: Dict,
                                      cache_file: str):
    """
    Save pruned graph with join column information

    Args:
        pruned_graph: Pruned graph
        verified_edges: Verified edges dictionary, key is "table1<->table2", value contains join_pairs
        cache_file: Save path
    """
    try:
        graph_copy = pruned_graph.copy()

        edge_order = {}
        for u, v in graph_copy.edges():
            edge_order[frozenset([u, v])] = (u, v)

        for edge_key, result in verified_edges.items():
            if not result.get('is_valid', False):
                continue
            parts = edge_key.split("<->")
            t1, t2 = parts[0], parts[1]
            if graph_copy.has_edge(t1, t2):
                join_pairs = result.get('join_pairs', [])
                source, target = edge_order[frozenset([t1, t2])]
                if source == t1:
                    col_pairs = [[p['column1'], p['column2']] for p in join_pairs]
                else:
                    col_pairs = [[p['column2'], p['column1']] for p in join_pairs]
                graph_copy[t1][t2]['join_columns'] = col_pairs

        graph_data = json_graph.node_link_data(graph_copy, edges="edges")

        cache_data = {
            'metadata': {
                'description': 'Pruned graph with join column annotations',
                'verified_edges_count': sum(1 for r in verified_edges.values() if r.get('is_valid', False)),
                'total_edges': graph_copy.number_of_edges()
            },
            'graph': graph_data,
            'node_count': graph_copy.number_of_nodes(),
            'edge_count': graph_copy.number_of_edges()
        }

        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, indent=2, ensure_ascii=False)

        print(f"[Cache] Pruned graph with join columns saved to: {cache_file}")
        print(f"  Nodes: {cache_data['node_count']}, Edges: {cache_data['edge_count']}")
        print(f"  Edges annotated with join columns: {cache_data['metadata']['verified_edges_count']}")
    except Exception as e:
        print(f"[Warning] Failed to save pruned graph with join columns: {e}")


def load_pruned_graph(cache_file: str, expected_metadata: Dict) -> Optional[nx.Graph]:
    """
    Load pruned graph from cache file

    Args:
        cache_file: Cache file path
        expected_metadata: Expected metadata (for verifying cache validity)

    Returns:
        Pruned graph, or None if cache is invalid or doesn't exist
    """
    if not os.path.exists(cache_file):
        print(f"[Cache] Cache file does not exist: {cache_file}")
        return None

    try:
        with open(cache_file, 'r', encoding='utf-8') as f:
            cache_data = json.load(f)

        cached_metadata = cache_data.get('metadata', {})

        key_params = ['k', 't_spanner', 'max_path_length']
        for param in key_params:
            if cached_metadata.get(param) != expected_metadata.get(param):
                print(f"[Cache] Parameter mismatch: {param} (cached={cached_metadata.get(param)}, expected={expected_metadata.get(param)})")
                return None

        graph_data = cache_data.get('graph')
        if not graph_data:
            print(f"[Cache] No graph data in cache file")
            return None

        pruned_graph = json_graph.node_link_graph(graph_data, edges="edges")

        print(f"[Cache] Loaded pruned graph from cache: {cache_file}")
        print(f"  Nodes: {cache_data['node_count']}, Edges: {cache_data['edge_count']}")

        return pruned_graph

    except Exception as e:
        print(f"[Warning] Failed to load pruned graph cache: {e}")
        return None


def load_queries_from_llm_filter(query_file: str, verbose: bool = False) -> List[List[str]]:
    """
    Load queries from LLM filter result JSON file

    Reads detailed_results in each element's llm_interaction.parsed_result.selected_tables,
    converts paths to table names.

    Args:
        query_file: LLM filter result JSON file path
        verbose: Whether to print detailed output

    Returns:
        List of queries, each query is a list of table names
    """
    try:
        with open(query_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"[Error] Failed to load query file: {e}")
        return []

    queries = []
    detailed_results = data.get('detailed_results', [])

    for item in detailed_results:
        llm_interaction = item.get('llm_interaction', {})
        parsed_result = llm_interaction.get('parsed_result') if llm_interaction else None
        if not parsed_result:
            queries.append([])
            continue
        selected_tables_paths = item.get('selected_tables', [])

        if not selected_tables_paths:
            queries.append([])
            continue

        table_names = [extract_table_name(p) for p in selected_tables_paths]
        table_names = [t for t in table_names if t]

        queries.append(table_names)

    if verbose:
        valid = sum(1 for q in queries if len(q) >= 2)
        print(f"Loaded {valid} valid queries from {len(detailed_results)} samples ({len(detailed_results) - valid} skipped)")

    return queries


def extract_candidate_pairs_from_json(json_data: List[Dict],
                                     edge_details: Dict) -> Dict[Tuple[str, str], List[Dict]]:
    """
    Extract candidate column pairs for each edge from JSON data

    Args:
        json_data: List of prediction records
        edge_details: Edge details from global graph construction

    Returns:
        Dictionary mapping edge tuple to list of candidate column pairs
    """
    candidate_pairs_dict = defaultdict(list)

    for item in json_data:
        t1 = extract_table_name(item.get('table1', ''))
        t2 = extract_table_name(item.get('table2', ''))
        c1 = normalize_str(item.get('column1', ''))
        c2 = normalize_str(item.get('column2', ''))
        prob = item.get('prob', 0.0)

        if not t1 or not t2:
            continue

        edge = tuple(sorted([t1, t2]))

        overlap = item.get('overlap', 'N/A')

        if edge[0] == t1:
            col1, col2 = c1, c2
        else:
            col1, col2 = c2, c1

        candidate_pairs_dict[edge].append({
            'column1': col1,
            'column2': col2,
            'prob': prob,
            'overlap': overlap
        })

    return dict(candidate_pairs_dict)


def build_sparse_graph_from_queries(queries: List[List[str]],
                                    global_graph: nx.Graph,
                                    k: int = 10,
                                    max_path_length: int = 5,
                                    t_spanner: float = 1.1,
                                    cache_file: str = None,
                                    overwrite: bool = False,
                                    verbose: bool = False) -> nx.Graph:
    """
    Build sparse graph from top-k paths between table pairs in all queries

    Args:
        queries: List of queries, each query is a list of table names
        global_graph: Global graph
        k: Number of shortest paths to collect per table pair
        max_path_length: Maximum path length
        t_spanner: t-spanner pruning parameter
        cache_file: Cache file path
        overwrite: Overwrite mode, ignore cache and force rebuild
        verbose: Whether to print detailed output

    Returns:
        Sparse graph (NetworkX graph object)
    """
    metadata = {
        'k': k,
        't_spanner': t_spanner,
        'max_path_length': max_path_length
    }

    if cache_file and not overwrite:
        pruned_graph = load_pruned_graph(cache_file, metadata)
        if pruned_graph is not None:
            return pruned_graph

    print("\n[Phase 1] Collecting paths for all queries...")
    print(f"  Total {len(queries)} queries")

    all_paths_with_info = []
    seen_pairs = set()
    seen_paths = set()

    for query_idx, required_tables in enumerate(tqdm(queries, desc="Collecting query paths"), 1):
        if len(required_tables) < 2:
            if verbose:
                print(f"  [Query {query_idx}/{len(queries)}] Skipped (table count < 2)")
            continue

        all_tables_present = all(t in global_graph for t in required_tables)
        if not all_tables_present:
            if verbose:
                missing = [t for t in required_tables if t not in global_graph]
                print(f"  [Query {query_idx}/{len(queries)}] Skipped (tables not in graph: {missing})")
            continue

        for t1, t2 in combinations(required_tables, 2):
            pair = tuple(sorted([t1, t2]))

            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)

            paths = find_k_shortest_paths(global_graph, t1, t2, k=k, max_path_length=max_path_length)

            if not paths:
                if verbose:
                    print(f"    Warning: No paths found between {t1} and {t2}")
                continue

            for path in paths:
                path_tuple = tuple(path)
                if path_tuple in seen_paths:
                    continue
                seen_paths.add(path_tuple)

                weight = calculate_path_weight(global_graph, path)
                if weight == float('inf'):
                    continue

                all_paths_with_info.append({
                    'path': path,
                    'weight': weight,
                    'pair': pair
                })

    print(f"[Phase 1 Complete] Collected {len(all_paths_with_info)} unique paths from {len(seen_pairs)} table pairs")

    print(f"\n[Phase 2] Pruning paths using t-spanner={t_spanner}...")
    sparse_graph, pruned_paths = prune_paths_global(global_graph, all_paths_with_info, t=t_spanner, verbose=verbose, k=10)

    print(f"\n[Sparse Graph Construction Complete]")
    print(f"  - Processed {len(queries)} queries")
    print(f"  - Collected {len(all_paths_with_info)} candidate paths")
    print(f"  - Retained {len(pruned_paths)} paths after pruning")
    print(f"  - Sparse graph: {sparse_graph.number_of_nodes()} nodes, {sparse_graph.number_of_edges()} edges")
    print(f"  - Compared to global graph: {global_graph.number_of_nodes()} nodes, {global_graph.number_of_edges()} edges")

    if cache_file:
        save_pruned_graph(sparse_graph, cache_file, metadata)

    return sparse_graph


def build_steiner_for_query(required_tables: List[str],
                            sparse_graph: nx.Graph) -> Tuple[Optional[List[Tuple[str, str]]], List[str], str]:
    """
    Build Steiner tree for a single query (pure function, no verification)

    When all required tables cannot be fully connected, select the connected component
    containing the most required tables to build the Steiner tree.

    Args:
        required_tables: List of required table names for the query
        sparse_graph: Sparse graph

    Returns:
        (steiner_edges, covered_tables, status)
        - steiner_edges: List of edge tuples, None on failure
        - covered_tables: List of actually covered table names
        - status: 'success', 'partial', 'insufficient_tables', 'steiner_tree_error'
    """
    if len(required_tables) < 2:
        return None, required_tables, 'insufficient_tables'

    tables_in_graph = [t for t in required_tables if t in sparse_graph]
    if len(tables_in_graph) < 2:
        return None, required_tables, 'insufficient_tables'

    components = list(nx.connected_components(sparse_graph))
    best_tables = []
    for comp in components:
        tables_in_comp = [t for t in tables_in_graph if t in comp]
        if len(tables_in_comp) > len(best_tables):
            best_tables = tables_in_comp

    if len(best_tables) < 2:
        return None, required_tables, 'insufficient_tables'

    is_partial = len(best_tables) < len(required_tables)

    try:
        adj_list = convert_networkx_to_adjacency_list(sparse_graph)
        steiner_result = steiner_tree_2approx(adj_list, best_tables)
        steiner_edges = []
        if steiner_result and 'edges' in steiner_result:
            for edge_info in steiner_result['edges']:
                steiner_edges.append((edge_info['id1'], edge_info['id2']))
        status = 'partial' if is_partial else 'success'
        return steiner_edges, best_tables, status
    except Exception:
        return None, required_tables, 'steiner_tree_error'


def evaluate_query_on_sparse_graph(query_idx: int,
                                   required_tables: List[str],
                                   sparse_graph: nx.Graph,
                                   gt_edges: Set[Tuple[str, str]],
                                   verified_edges: Dict,
                                   verbose: bool = False) -> Dict:
    """
    Evaluate a single query on sparse graph (only compute metrics, no verification)

    Args:
        query_idx: Query index
        required_tables: List of required table names for the query
        sparse_graph: Sparse graph
        gt_edges: Ground truth edge set
        verified_edges: Verified edges dictionary
        verbose: Whether to print detailed output

    Returns:
        Evaluation result dictionary
    """
    steiner_edges, _, status = build_steiner_for_query(
        required_tables, sparse_graph
    )

    if steiner_edges is None:
        return {
            'query_idx': query_idx,
            'required_tables': required_tables,
            'status': status
        }

    valid_edges_after_llm = set()
    for edge in steiner_edges:
        edge_tuple = tuple(sorted(edge))
        edge_key = f"{edge_tuple[0]}<->{edge_tuple[1]}"
        result = verified_edges.get(edge_key)
        if result is None or result.get('is_valid', False):
            valid_edges_after_llm.add(edge_tuple)

    valid_edges_in_gt = valid_edges_after_llm & gt_edges
    edge_precision = len(valid_edges_in_gt) / len(valid_edges_after_llm) if valid_edges_after_llm else 0.0
    gt_match = valid_edges_after_llm.issubset(gt_edges)

    return {
        'query_idx': query_idx,
        'required_tables': required_tables,
        'status': status,
        'steiner_edges': list(valid_edges_after_llm),
        'steiner_edge_count': len(valid_edges_after_llm),
        'gt_match': gt_match,
        'valid_edges_count': len(valid_edges_after_llm),
        'valid_edges_in_gt_count': len(valid_edges_in_gt),
        'edge_precision': edge_precision
    }


def evaluate_all_queries(queries: List[List[str]],
                        sparse_graph: nx.Graph,
                        edge_validator: EdgeValidator,
                        candidate_pairs_dict: Dict,
                        gt_edges: Set[Tuple[str, str]],
                        verbose: bool = False) -> Tuple[List[Dict], Dict]:
    """
    Batch iterative verification of all queries

    Each round builds Steiner tree for all queries -> takes union of edges -> counts edge frequency
    to determine critical edges -> batch verification -> updates graph -> repeats until all edges verified.

    Args:
        queries: List of queries, each query is a list of table names
        sparse_graph: Sparse graph
        edge_validator: EdgeValidator instance
        candidate_pairs_dict: Candidate column pairs dictionary
        gt_edges: Ground truth edge set
        verbose: Whether to print detailed output

    Returns:
        (list of evaluation results, verified edges dictionary)
    """
    print("\n[Phase 3] Batch iterative verification of all queries...")

    verified_edges = {}
    round_idx = 0
    total_queries = len(queries)

    while True:
        print(f"\n--- Round {round_idx + 1} ---")

        all_steiner_trees = {}
        edge_frequency = Counter()

        for idx, required_tables in enumerate(queries):
            steiner_edges, _, status = build_steiner_for_query(
                required_tables, sparse_graph
            )
            if steiner_edges is not None:
                all_steiner_trees[idx] = steiner_edges
                for edge in steiner_edges:
                    edge_frequency[tuple(sorted(edge))] += 1

        all_edges_union = set()
        for edges in all_steiner_trees.values():
            for edge in edges:
                all_edges_union.add(tuple(sorted(edge)))

        unverified = [e for e in all_edges_union if f"{e[0]}<->{e[1]}" not in verified_edges]

        if not unverified:
            print(f"  All {len(all_edges_union)} edges verified, exiting iteration")
            break

        critical_edges = {e for e in unverified if edge_frequency[e] > 2}

        print(f"  Steiner trees: {len(all_steiner_trees)}/{total_queries}")
        print(f"  Edge union size: {len(all_edges_union)}, Unverified: {len(unverified)}, Critical edges: {len(critical_edges)}")

        for edge in tqdm(unverified, desc=f"Round {round_idx+1} verifying edges"):
            table1, table2 = edge
            edge_key = f"{table1}<->{table2}"
            candidate_pairs = candidate_pairs_dict.get(edge, [])
            is_critical = edge in critical_edges

            result = edge_validator.validate_edge(table1, table2, candidate_pairs, is_critical)
            verified_edges[edge_key] = result

            if edge_validator.llm_verifier.test_mode:
                print(f"\n{'='*80}")
                print("✓ Test mode: Completed verification of one edge")
                print(f"{'='*80}")
                print(f"Verified edge: {table1} <-> {table2}")
                print(f"Verification result: {'Valid' if result['is_valid'] else 'Invalid'}")
                print(f"Number of join column pairs found: {len(result.get('join_pairs', []))}")
                print(f"{'='*80}\n")
                print("Test mode complete, program will exit.")
                sys.exit(0)

        invalid_count = 0
        for edge in unverified:
            table1, table2 = edge
            edge_key = f"{table1}<->{table2}"
            result = verified_edges[edge_key]
            if result.get('is_valid', False):
                if sparse_graph.has_edge(table1, table2):
                    sparse_graph[table1][table2]['weight'] = 0
            else:
                if sparse_graph.has_edge(table1, table2):
                    sparse_graph.remove_edge(table1, table2)
                    invalid_count += 1

        print(f"  Verification result: {len(unverified) - invalid_count} valid, {invalid_count} invalid removed")

        if invalid_count == 0:
            print(f"  No invalid edges removed, graph unchanged, exiting iteration")
            break

        round_idx += 1

    print(f"\n[Phase 3.5] Computing final results for each query...")
    all_results = []

    with tqdm(total=total_queries, desc="Computing query results") as pbar:
        for query_idx, required_tables in enumerate(queries):
            result = evaluate_query_on_sparse_graph(
                query_idx, required_tables,
                sparse_graph, gt_edges,
                verified_edges, verbose
            )
            all_results.append(result)
            pbar.update(1)

    verification_base_dir = getattr(edge_validator, 'verification_base_dir', None)
    if verification_base_dir:
        os.makedirs(verification_base_dir, exist_ok=True)
        pruned_graph_path = os.path.join(verification_base_dir, 'pruned_graph.json')
        save_pruned_graph(sparse_graph, pruned_graph_path, metadata={
            'total_queries': total_queries,
            'verified_edges_count': len(verified_edges)
        })

        join_cols_path = os.path.join(verification_base_dir, 'pruned_graph_join_cols_version.json')
        save_pruned_graph_with_join_cols(sparse_graph, verified_edges, join_cols_path)

    return all_results, verified_edges


def print_evaluation_summary(results: List[Dict],
                            edge_validator: EdgeValidator,
                            llm_verifier: LLMVerifier,
                            gt_edges: Set[Tuple[str, str]],
                            gt_edges_by_basetable: Dict[str, Set[Tuple[str, str]]],
                            sparse_graph: nx.Graph,
                            verified_edges: Dict):
    """Print evaluation summary and statistics"""
    print("\n" + "=" * 80)
    print("Evaluation Summary")
    print("=" * 80)

    total_queries = len(results)
    successful_queries = sum(1 for r in results if r['status'] == 'success')
    partial_queries = sum(1 for r in results if r['status'] == 'partial')
    insufficient_queries = sum(1 for r in results if r['status'] == 'insufficient_tables')
    error_queries = sum(1 for r in results if r['status'] == 'steiner_tree_error')
    gt_matched_queries = sum(1 for r in results if r.get('gt_match', False))
    gt_matched_in_success = sum(1 for r in results if r['status'] == 'success' and r.get('gt_match', False))

    print(f"\nQuery Statistics:")
    print(f"  Total queries:           {total_queries}")
    print(f"  Successfully evaluated:  {successful_queries}")
    print(f"  Partially returned:      {partial_queries}")
    print(f"  Insufficient tables:     {insufficient_queries}")
    if error_queries:
        print(f"  Construction errors:     {error_queries}")
    print(f"  Ground Truth matches:    {gt_matched_in_success} (success only)")
    print(f"  GT match rate (success): {gt_matched_in_success / max(1, successful_queries):.2%}")

    print(f"\n" + "-" * 80)
    print(f"LLM Verification Accuracy (based on all verified edges):")
    print(f"-" * 80)

    tp = 0
    tn = 0
    fp = 0
    fn = 0
    fp_edges = []

    for edge_key, result in verified_edges.items():
        parts = edge_key.split("<->")
        edge_tuple = tuple(sorted(parts))
        is_gt = edge_tuple in gt_edges
        is_valid = result.get('is_valid', False)

        if is_gt and is_valid:
            tp += 1
        elif is_gt and not is_valid:
            fn += 1
        elif not is_gt and is_valid:
            fp += 1
            fp_edges.append(edge_tuple)
        else:
            tn += 1

    total_verified = tp + tn + fp + fn
    print(f"  Total verified edges:     {total_verified}")
    print(f"  TP (GT edge->valid):      {tp}")
    print(f"  TN (non-GT edge->invalid):{tn}")
    print(f"  FP (non-GT edge->valid):  {fp}")
    print(f"  FN (GT edge->invalid):    {fn}")

    accuracy = (tp + tn) / max(1, total_verified)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-9, precision + recall)

    print(f"  Accuracy:                 {accuracy:.2%}")
    print(f"  Precision:                {precision:.2%}")
    print(f"  Recall:                   {recall:.2%}")
    print(f"  F1 Score:                 {f1:.2%}")

    if fp_edges:
        print(f"\n  False Positive (FP) edges details:")
        for e in sorted(fp_edges):
            print(f"    - {e[0]} <-> {e[1]}")

    print(f"\n" + "-" * 80)
    print(f"GT Edge Verification Hit Statistics by Base Table:")
    print(f"-" * 80)

    total_gt_all = 0
    total_gt_verified_valid = 0

    for base_table in sorted(gt_edges_by_basetable.keys()):
        gt_set = gt_edges_by_basetable[base_table]
        gt_count = len(gt_set)

        verified_valid_count = 0
        fn_edges_for_table = []
        for edge in gt_set:
            edge_key = f"{edge[0]}<->{edge[1]}"
            result = verified_edges.get(edge_key)
            if result and result.get('is_valid', False):
                verified_valid_count += 1
            else:
                fn_edges_for_table.append(edge)

        total_gt_all += gt_count
        total_gt_verified_valid += verified_valid_count

        hit_rate = verified_valid_count / max(1, gt_count)
        print(f"  [{base_table}] {verified_valid_count}/{gt_count} ({hit_rate:.2%})")
        if fn_edges_for_table:
            for e in sorted(fn_edges_for_table):
                edge_key = f"{e[0]}<->{e[1]}"
                result = verified_edges.get(edge_key)
                if result:
                    reason = "LLM judged as invalid"
                elif sparse_graph.has_edge(e[0], e[1]):
                    reason = "In sparse graph but not verified"
                else:
                    reason = "Not in sparse graph"
                print(f"    FN: {e[0]} <-> {e[1]} ({reason})")

    total_base_tables = len(gt_edges_by_basetable)
    overall_rate = total_gt_verified_valid / max(1, total_gt_all)
    print(f"\n  Summary:")
    print(f"    Total base tables:                   {total_base_tables}")
    print(f"    Total GT edges:                      {total_gt_all}")
    print(f"    Correctly verified GT edges:         {total_gt_verified_valid}/{total_gt_all} ({overall_rate:.2%})")

    print("\n" + "-" * 80)
    edge_validator.print_statistics()
    print("-" * 80)
    llm_verifier.print_statistics()
    print("=" * 80)


def save_results(results: List[Dict], output_file: str):
    """Save evaluation results to JSON file"""
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n[Output] Results saved to: {output_file}")
    except Exception as e:
        print(f"[Error] Failed to save results: {e}")


def run_join_evaluation(
    json_file: str,
    csv_file: str,
    queries_file: str,
    tables_dir: str,
    output_file: str = 'evaluation_results.json',
    api_key: str = None,
    model: str = 'gpt-4o',
    sample_rows: int = 5,
    top_k_paths: int = 10,
    max_path_length: int = 100,
    t_spanner: float = 1.1,
    edge_verification_cache: str = 'edge_verification_cache',
    graph_cache: str = 'sparse_graph_cache.json',
    verbose: bool = False,
    test_mode: bool = False,
    overwrite: bool = False
) -> Dict:
    """
    Run join evaluation pipeline (callable interface)

    Args:
        json_file: JSON file containing prediction results
        csv_file: CSV file containing ground truth
        queries_file: LLM filter result JSON file (with selected_tables)
        tables_dir: Table data directory
        output_file: Result output file (default: evaluation_results.json)
        api_key: LLM API key (or set OPENAI_API_KEY environment variable)
        model: LLM model name (default: gpt-4o)
        sample_rows: Number of rows for coarse-grained verification (default: 5)
        top_k_paths: Number of shortest paths per table pair (default: 10)
        max_path_length: Maximum path length (default: 100)
        t_spanner: t-spanner parameter for pruning (default: 1.1)
        edge_verification_cache: Edge verification cache directory
        graph_cache: Sparse graph cache file
        verbose: Verbose output
        test_mode: Test mode - only verify one edge and output prompt
        overwrite: Overwrite mode - ignore all caches

    Returns:
        Dictionary containing evaluation results and statistics
    """
    print("\n[Step 1/6] Loading JSON prediction results...")
    json_data = load_json_data(json_file)
    if not json_data:
        raise ValueError("JSON data is empty or invalid.")

    print("\n[Step 2/6] Building global graph...")
    global_graph, edge_details = build_global_graph(json_data, verbose=verbose)

    print("\n[Step 3/6] Loading queries...")
    queries = load_queries_from_llm_filter(queries_file, verbose=verbose)
    print(f"  Loaded {len(queries)} queries")

    print("\n[Step 4/6] Loading ground truth...")
    _, gt_edges, gt_edges_by_basetable = load_ground_truth(csv_file, verbose=verbose)

    print("\n[Step 5/6] Building sparse graph...")
    sparse_graph = build_sparse_graph_from_queries(
        queries, global_graph,
        k=top_k_paths,
        max_path_length=max_path_length,
        t_spanner=t_spanner,
        cache_file=graph_cache,
        overwrite=overwrite,
        verbose=verbose
    )

    print("\n[Step 5.5/6] Extracting candidate column pairs...")
    candidate_pairs_dict = extract_candidate_pairs_from_json(json_data, edge_details)

    # Get prompt template path
    utils_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'utils')
    prompt_template_path = os.path.join(utils_dir, 'prompt.txt')

    llm_verifier = LLMVerifier(
        api_key=api_key,
        model=model,
        sample_rows=sample_rows,
        prompt_template_path=prompt_template_path,
        test_mode=test_mode
    )

    edge_validator = EdgeValidator(
        llm_verifier=llm_verifier,
        tables_dir=tables_dir,
        verification_base_dir=edge_verification_cache,
        overwrite=overwrite
    )

    if test_mode:
        print("\n" + "="*80)
        print("⚠️  Test mode enabled")
        print("="*80)
        print("- Will only verify the first edge")
        print("- Will not call LLM API")
        print("- Will only output Phase1 and Phase2 prompts")
        print("="*80 + "\n")

    print("\n[Step 6/6] Evaluating queries...")
    results, verified_edges = evaluate_all_queries(
        queries, sparse_graph,
        edge_validator, candidate_pairs_dict,
        gt_edges, verbose
    )

    print_evaluation_summary(results, edge_validator, llm_verifier,
                            gt_edges, gt_edges_by_basetable,
                            sparse_graph, verified_edges)

    save_results(results, output_file)

    # Return summary statistics
    return {
        'total_queries': len(results),
        'successful_queries': sum(1 for r in results if r['status'] == 'success'),
        'results': results,
        'verified_edges': verified_edges
    }


def main():
    """Command-line interface"""
    parser = argparse.ArgumentParser(
        description="LLM-based Join Evaluation",
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument('-j', '--json', required=True, help="JSON file containing prediction results")
    parser.add_argument('-c', '--csv', required=True, help="CSV file containing ground truth")
    parser.add_argument('--queries', required=True, help="LLM filter result JSON file (with selected_tables)")

    parser.add_argument('--tables', required=True, help="Table data directory")

    parser.add_argument('--api_key', help="LLM API key (or set OPENAI_API_KEY environment variable)")
    parser.add_argument('--model', default='gpt-4o', help="LLM model name (default: gpt-4o)")
    parser.add_argument('--sample_rows', type=int, default=5,
                       help="Number of rows for coarse-grained verification (default: 5)")

    parser.add_argument('--top_k_paths', type=int, default=10,
                       help="Number of shortest paths per table pair (default: 10)")
    parser.add_argument('--max_path_length', type=int, default=100,
                       help="Maximum path length (default: 100)")
    parser.add_argument('--t_spanner', type=float, default=1.1,
                       help="t-spanner parameter for pruning (default: 1.1)")

    parser.add_argument('--edge_verification_cache', default='edge_verification_cache',
                       help="Edge verification cache directory (default: edge_verification_cache)")
    parser.add_argument('--graph_cache', default='sparse_graph_cache.json',
                       help="Sparse graph cache file (default: sparse_graph_cache.json)")

    parser.add_argument('-o', '--output', default='evaluation_results.json',
                       help="Result output file (default: evaluation_results.json)")

    parser.add_argument('-v', '--verbose', action='store_true', help="Verbose output")
    parser.add_argument('--test_mode', action='store_true',
                       help="Test mode: only verify one edge and output prompt, without calling LLM API")
    parser.add_argument('--overwrite', action='store_true',
                       help="Overwrite mode: ignore all caches, force re-verification of all edges")

    args = parser.parse_args()

    try:
        run_join_evaluation(
            json_file=args.json,
            csv_file=args.csv,
            queries_file=args.queries,
            tables_dir=args.tables,
            output_file=args.output,
            api_key=args.api_key,
            model=args.model,
            sample_rows=args.sample_rows,
            top_k_paths=args.top_k_paths,
            max_path_length=args.max_path_length,
            t_spanner=args.t_spanner,
            edge_verification_cache=args.edge_verification_cache,
            graph_cache=args.graph_cache,
            verbose=args.verbose,
            test_mode=args.test_mode,
            overwrite=args.overwrite
        )
    except Exception as e:
        print(f"\n[Error] Execution failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
