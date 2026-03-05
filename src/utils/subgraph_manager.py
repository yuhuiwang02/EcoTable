"""
Subgraph Manager Module

This module handles:
- Merging user_selected_nodes with containment relationships
- Building subgraphs from merged sub_tables
- Caching subgraphs for efficient query matching
- Finding best matching subgraph for a given query
"""

import json
import os
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict
import networkx as nx


class SubgraphManager:
    """Manager for subgraph construction and caching"""

    def __init__(self, cache_file: str = "subgraph_cache.json"):
        """
        Initialize subgraph manager

        Args:
            cache_file: Path to cache file for subgraphs
        """
        self.cache_file = cache_file
        self.subgraphs = {}  # {subgraph_id: subgraph_data}
        self.base_table_subgraphs = defaultdict(list)  # {base_table: [subgraph_ids]}

        # Load cache if exists
        self._load_cache()

    def _load_cache(self):
        """Load subgraph cache from file"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.subgraphs = data.get('subgraphs', {})
                    self.base_table_subgraphs = defaultdict(list, data.get('base_table_subgraphs', {}))
                print(f"[SubgraphManager] Loaded {len(self.subgraphs)} subgraphs from cache")
            except Exception as e:
                print(f"[Warning] Failed to load subgraph cache: {e}")

    def save_cache(self):
        """Save subgraph cache to file"""
        try:
            data = {
                'subgraphs': self.subgraphs,
                'base_table_subgraphs': dict(self.base_table_subgraphs)
            }
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"[SubgraphManager] Saved {len(self.subgraphs)} subgraphs to cache")
        except Exception as e:
            print(f"[Warning] Failed to save subgraph cache: {e}")

    def merge_user_selected_nodes(self, base_table_name: str,
                                  all_queries: List[List[str]]) -> List[Set[str]]:
        """
        Merge user_selected_nodes that have containment relationships

        Args:
            base_table_name: Name of the base table
            all_queries: List of user_selected_nodes lists for this base table

        Returns:
            List of merged node sets (sub_tables)
        """
        if not all_queries:
            return []

        # Convert to sets for easier comparison
        node_sets = [set(query) for query in all_queries]

        # Merge sets with containment relationships
        merged = []
        used = [False] * len(node_sets)

        for i in range(len(node_sets)):
            if used[i]:
                continue

            current_set = node_sets[i].copy()
            used[i] = True

            # Check if any other set contains or is contained by current_set
            changed = True
            while changed:
                changed = False
                for j in range(len(node_sets)):
                    if used[j]:
                        continue

                    # Check containment relationship
                    if node_sets[j].issubset(current_set) or current_set.issubset(node_sets[j]):
                        # Merge the sets
                        current_set = current_set.union(node_sets[j])
                        used[j] = True
                        changed = True

            merged.append(current_set)

        print(f"[SubgraphManager] {base_table_name}: Merged {len(node_sets)} queries into {len(merged)} sub_tables")
        return merged

    def build_subgraph(self, base_table_name: str, sub_tables: Set[str],
                      global_graph: nx.Graph, split_stats: Dict,
                      top_k_paths: int = 10, max_path_length: int = 5) -> Dict:
        """
        Build a subgraph for a set of sub_tables

        Args:
            base_table_name: Name of the base table
            sub_tables: Set of node IDs (from user_selected_nodes)
            global_graph: The global graph
            split_stats: Split statistics data
            top_k_paths: Number of shortest paths to consider
            max_path_length: Maximum path length

        Returns:
            Subgraph data dict with:
                - 'subgraph_id': Unique identifier
                - 'base_table': Base table name
                - 'sub_tables': List of node IDs
                - 'subtable_paths': List of actual subtable paths
                - 'nodes': List of nodes in subgraph
                - 'edges': List of edges with weights
                - 'verified_edges': Dict of verified edges
        """
        # Convert node IDs to subtable paths
        subtable_paths = self._convert_nodes_to_paths(base_table_name, sub_tables, split_stats)

        if len(subtable_paths) < 2:
            print(f"[Warning] Not enough subtables for subgraph: {len(subtable_paths)}")
            return None

        # Extract table names from paths
        from testTrainQuality_new import extract_table_name
        table_names = [extract_table_name(path) for path in subtable_paths]

        # Find all shortest paths between pairs of tables
        from testTrainQuality_new import find_k_shortest_paths, path_to_edges
        from itertools import combinations

        all_edges = set()
        for t1, t2 in combinations(table_names, 2):
            paths = find_k_shortest_paths(global_graph, t1, t2, k=top_k_paths,
                                         max_path_length=max_path_length)
            for path in paths:
                edges = path_to_edges(path)
                all_edges.update(edges)

        # Build subgraph from collected edges
        subgraph = nx.Graph()
        edge_list = []
        for (n1, n2) in all_edges:
            if global_graph.has_edge(n1, n2):
                weight = global_graph[n1][n2]['weight']
                subgraph.add_edge(n1, n2, weight=weight)
                edge_list.append({
                    'node1': n1,
                    'node2': n2,
                    'weight': weight
                })

        # Generate subgraph ID
        subgraph_id = f"{base_table_name}_{'_'.join(sorted(sub_tables))}"

        subgraph_data = {
            'subgraph_id': subgraph_id,
            'base_table': base_table_name,
            'sub_tables': sorted(list(sub_tables)),
            'subtable_paths': subtable_paths,
            'nodes': list(subgraph.nodes()),
            'edges': edge_list,
            'verified_edges': {},  # Will be populated during verification
            'edge_count': len(edge_list),
            'node_count': len(subgraph.nodes())
        }

        print(f"[SubgraphManager] Built subgraph {subgraph_id}: "
              f"{subgraph_data['node_count']} nodes, {subgraph_data['edge_count']} edges")

        return subgraph_data

    def _convert_nodes_to_paths(self, base_table_name: str,
                               node_ids: Set[str], split_stats: Dict) -> List[str]:
        """Convert node IDs to subtable paths"""
        from testTrainQuality_new import convert_nodes_to_subtables
        return convert_nodes_to_subtables(base_table_name, list(node_ids), split_stats)

    def add_subgraph(self, subgraph_data: Dict):
        """Add a subgraph to the manager"""
        if not subgraph_data:
            return

        subgraph_id = subgraph_data['subgraph_id']
        base_table = subgraph_data['base_table']

        self.subgraphs[subgraph_id] = subgraph_data
        self.base_table_subgraphs[base_table].append(subgraph_id)

    def find_matching_subgraph(self, base_table_name: str,
                               user_selected_nodes: List[str]) -> Optional[str]:
        """
        Find the best matching subgraph for a query

        The best match is the smallest subgraph whose sub_tables completely
        contain the user_selected_nodes

        Args:
            base_table_name: Base table name
            user_selected_nodes: Query's user_selected_nodes

        Returns:
            Subgraph ID if found, None otherwise
        """
        query_nodes = set(user_selected_nodes)

        # Get all subgraphs for this base table
        candidate_subgraphs = self.base_table_subgraphs.get(base_table_name, [])

        if not candidate_subgraphs:
            return None

        # Find subgraphs that contain all query nodes
        matching_subgraphs = []
        for subgraph_id in candidate_subgraphs:
            subgraph_data = self.subgraphs[subgraph_id]
            subgraph_nodes = set(subgraph_data['sub_tables'])

            if query_nodes.issubset(subgraph_nodes):
                matching_subgraphs.append((subgraph_id, len(subgraph_nodes)))

        if not matching_subgraphs:
            return None

        # Return the smallest matching subgraph
        matching_subgraphs.sort(key=lambda x: x[1])
        return matching_subgraphs[0][0]

    def get_subgraph(self, subgraph_id: str) -> Optional[Dict]:
        """Get subgraph data by ID"""
        return self.subgraphs.get(subgraph_id)

    def update_verified_edges(self, subgraph_id: str, edge: Tuple[str, str],
                             verification_result: Dict):
        """
        Update verified edges for a subgraph

        Args:
            subgraph_id: Subgraph ID
            edge: Edge tuple (node1, node2)
            verification_result: Verification result dict
        """
        if subgraph_id not in self.subgraphs:
            return

        edge_key = f"{edge[0]}<->{edge[1]}"
        self.subgraphs[subgraph_id]['verified_edges'][edge_key] = verification_result

    def get_verified_edge(self, subgraph_id: str, edge: Tuple[str, str]) -> Optional[Dict]:
        """Get verification result for an edge"""
        if subgraph_id not in self.subgraphs:
            return None

        edge_key = f"{edge[0]}<->{edge[1]}"
        return self.subgraphs[subgraph_id]['verified_edges'].get(edge_key)

    def prune_subgraph(self, subgraph_id: str, edges_to_remove: List[Tuple[str, str]]) -> Dict:
        """
        Prune edges from a subgraph

        Args:
            subgraph_id: Subgraph ID
            edges_to_remove: List of edges to remove

        Returns:
            Updated subgraph data
        """
        if subgraph_id not in self.subgraphs:
            return None

        subgraph_data = self.subgraphs[subgraph_id]

        # Remove edges
        edges_to_remove_set = set()
        for edge in edges_to_remove:
            edges_to_remove_set.add(tuple(sorted(edge)))

        new_edges = []
        for edge_data in subgraph_data['edges']:
            edge_tuple = tuple(sorted([edge_data['node1'], edge_data['node2']]))
            if edge_tuple not in edges_to_remove_set:
                new_edges.append(edge_data)

        subgraph_data['edges'] = new_edges
        subgraph_data['edge_count'] = len(new_edges)

        # Rebuild node list (only nodes that are still connected)
        nodes = set()
        for edge_data in new_edges:
            nodes.add(edge_data['node1'])
            nodes.add(edge_data['node2'])

        subgraph_data['nodes'] = list(nodes)
        subgraph_data['node_count'] = len(nodes)

        print(f"[SubgraphManager] Pruned subgraph {subgraph_id}: "
              f"removed {len(edges_to_remove)} edges, "
              f"now {subgraph_data['node_count']} nodes, {subgraph_data['edge_count']} edges")

        return subgraph_data

    def get_statistics(self) -> Dict:
        """Get statistics about subgraphs"""
        total_subgraphs = len(self.subgraphs)
        total_edges = sum(sg['edge_count'] for sg in self.subgraphs.values())
        total_verified = sum(len(sg['verified_edges']) for sg in self.subgraphs.values())

        return {
            'total_subgraphs': total_subgraphs,
            'total_edges': total_edges,
            'total_verified_edges': total_verified,
            'base_tables': len(self.base_table_subgraphs)
        }

    def print_statistics(self):
        """Print subgraph statistics"""
        stats = self.get_statistics()
        print("\n" + "=" * 60)
        print("Subgraph Manager Statistics")
        print("=" * 60)
        print(f"Total Subgraphs:     {stats['total_subgraphs']}")
        print(f"Base Tables:         {stats['base_tables']}")
        print(f"Total Edges:         {stats['total_edges']}")
        print(f"Verified Edges:      {stats['total_verified_edges']}")
        print("=" * 60)
