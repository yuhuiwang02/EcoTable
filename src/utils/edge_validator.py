"""
Edge Validation Module

Handles edge validation with voting mechanism:
- Normal edges: 1 coarse + 1 fine verification
- Critical edges: 3 verifications with voting
- Critical edge definition:
  1. Appears 2+ times in top-k shortest paths
  2. Is a bridge edge (affects graph connectivity)
"""

import json
import os
from typing import Dict, List, Tuple, Set, Optional
from collections import Counter
import networkx as nx
from .llm_verifier import LLMVerifier
from .verification_utils import (
    check_verification_exists,
    load_verification_result,
    save_verification_result,
    save_verification_visualization,
    get_verification_filepath,
    get_edge_filename
)


class EdgeValidator:
    """Edge validator with voting mechanism"""

    def __init__(self, llm_verifier: LLMVerifier, tables_dir: str,
                 verification_base_dir: str = "verification_results",
                 overwrite: bool = False):
        """
        Initialize edge validator

        Args:
            llm_verifier: LLM verifier instance
            tables_dir: Directory containing table files
            verification_base_dir: Base directory for verification results
            overwrite: Whether to overwrite existing cache (default False)
        """
        self.llm_verifier = llm_verifier
        self.tables_dir = tables_dir
        self.verification_base_dir = verification_base_dir
        self.overwrite = overwrite

        # Create verification results directory
        os.makedirs(verification_base_dir, exist_ok=True)

        # Statistics
        self.stats = {
            'total_edges_validated': 0,
            'critical_edges': 0,
            'normal_edges': 0,
            'valid_edges': 0,
            'invalid_edges': 0,
            'cache_hits': 0
        }

    def _accumulate_cached_stats(self, cached_result: Dict, phase: str):
        """
        Accumulate cost/latency statistics from cached results into llm_verifier.stats,
        to simulate the overhead as if LLM was called each time.

        Args:
            cached_result: Cached verification result (contains tokens, input_tokens, output_tokens, latency)
            phase: 'coarse' or 'fine', used to update corresponding call count
        """
        self.llm_verifier.stats['total_tokens'] += cached_result.get('tokens', 0)
        self.llm_verifier.stats['input_tokens'] += cached_result.get('input_tokens', 0)
        self.llm_verifier.stats['output_tokens'] += cached_result.get('output_tokens', 0)
        self.llm_verifier.stats['total_latency'] += cached_result.get('latency', 0)
        self.llm_verifier.stats['total_calls'] += 1
        if phase == 'coarse':
            self.llm_verifier.stats['coarse_calls'] += 1
        elif phase == 'fine':
            # phase2 may contain multiple column pair calls, but cached result is aggregated, counted as 1
            self.llm_verifier.stats['fine_calls'] += 1

    def _restore_conversation_from_cache(self, table1_path: str, table2_path: str,
                                          verification_type: str, repeat_idx: int = 0):
        """
        Restore conversation history from readable cache file to LLMVerifier

        Args:
            table1_path: Path to the first table
            table2_path: Path to the second table
            verification_type: Verification type (coarse/fine)
            repeat_idx: Repeat index
        """
        readable_dir = os.path.join(self.verification_base_dir, f"{verification_type}_readable")
        filename = get_edge_filename(table1_path, table2_path)
        if repeat_idx > 0:
            filename = f"{repeat_idx}){filename}"
        readable_filepath = os.path.join(readable_dir, f"{filename}.json")

        if os.path.exists(readable_filepath):
            with open(readable_filepath, 'r', encoding='utf-8') as f:
                readable_data = json.load(f)
            conversation_history = readable_data.get('conversation_history', [])
            if conversation_history:
                self.llm_verifier.restore_conversation(table1_path, table2_path, conversation_history)
                print(f"[Conversation Restore] Restored conversation history from cache: {readable_filepath}")

    def identify_critical_edges(self, subgraph_data: Dict, global_graph: nx.Graph,
                               top_k: int = 10, max_path_length: int = 5) -> Set[Tuple[str, str]]:
        """
        Identify critical edges in the subgraph

        Critical edge definition:
        1. Edges appearing 2+ times in top-k shortest paths
        2. Bridge edges (removal disconnects the graph)

        Args:
            subgraph_data: Subgraph data dictionary
            global_graph: Global graph
            top_k: Number of shortest paths to consider
            max_path_length: Maximum path length

        Returns:
            Set of critical edge tuples
        """
        from testTrainQuality_new import find_k_shortest_paths, path_to_edges, extract_table_name
        from itertools import combinations

        # Build subgraph from edge list
        subgraph = nx.Graph()
        for edge_data in subgraph_data['edges']:
            subgraph.add_edge(edge_data['node1'], edge_data['node2'],
                            weight=edge_data['weight'])

        # Get subtable names
        subtable_paths = subgraph_data['subtable_paths']
        table_names = [extract_table_name(path) for path in subtable_paths]

        # Count edge appearances in shortest paths
        edge_counter = Counter()
        for t1, t2 in combinations(table_names, 2):
            paths = find_k_shortest_paths(global_graph, t1, t2, k=top_k,
                                         max_path_length=max_path_length)
            for path in paths:
                edges = path_to_edges(path)
                edge_counter.update(edges)

        # Edges appearing 2+ times
        frequent_edges = {edge for edge, count in edge_counter.items() if count >= 2}

        # Find bridge edges
        bridge_edges = set(nx.bridges(subgraph))
        # Normalize bridge edges (sorted tuples)
        bridge_edges = {tuple(sorted(edge)) for edge in bridge_edges}

        # Merge both criteria
        critical_edges = frequent_edges | bridge_edges

        print(f"[EdgeValidator] Identified {len(critical_edges)} critical edges:")
        print(f"  - {len(frequent_edges)} high-frequency edges (appear 2+ times in paths)")
        print(f"  - {len(bridge_edges)} bridge edges (affect connectivity)")

        return critical_edges

    def _parse_selected_pairs(self, selected_pairs: List[str]) -> List[Tuple[str, str]]:
        """
        Parse 'Table1.col1 <-> Table2.col2' format into (col1, col2) tuples

        Args:
            selected_pairs: List of strings in format "Table1.column_name <-> Table2.column_name"

        Returns:
            List of (col1_name, col2_name) tuples
        """
        column_pairs = []
        for pair_str in selected_pairs:
            parts = pair_str.split("<->")
            if len(parts) == 2:
                # Extract column names (everything after the last dot)
                col1_name = parts[0].strip().split(".")[-1].strip()
                col2_name = parts[1].strip().split(".")[-1].strip()
                column_pairs.append((col1_name, col2_name))
        return column_pairs

    def validate_edge(self, table1: str, table2: str, candidate_pairs: List[Dict],
                     is_critical: bool = False) -> Dict:
        """
        Validate edge between two tables

        Args:
            table1: First table name
            table2: Second table name
            candidate_pairs: Candidate column pairs from deep learning model
            is_critical: Whether this is a critical edge (requires voting)

        Returns:
            Validation result dictionary containing:
                - 'is_valid': bool
                - 'join_pairs': List of valid join column pairs
                - 'is_critical': bool
                - 'verification_count': int
                - 'votes': List of verification results (for critical edges)
                - 'reasoning': str
        """
        # Get table paths
        table1_path = self._find_table_path(table1)
        table2_path = self._find_table_path(table2)

        if not table1_path or not table2_path:
            print(f"[Warning] Cannot find table files: {table1} or {table2}")
            return {
                'is_valid': False,
                'join_pairs': [],
                'is_critical': is_critical,
                'verification_count': 0,
                'reasoning': 'Table files not found'
            }

        # Execute validation
        if is_critical:
            result = self._validate_critical_edge(table1_path, table2_path, candidate_pairs)
            self.stats['critical_edges'] += 1
        else:
            result = self._validate_normal_edge(table1_path, table2_path, candidate_pairs)
            self.stats['normal_edges'] += 1

        # Update statistics
        self.stats['total_edges_validated'] += 1
        if result['is_valid']:
            self.stats['valid_edges'] += 1
        else:
            self.stats['invalid_edges'] += 1

        return result

    def _validate_normal_edge(self, table1_path: str, table2_path: str,
                             candidate_pairs: List[Dict]) -> Dict:
        """
        Validate normal edge: 1 phase1 + 1 phase2 verification

        Args:
            table1_path: Path to the first table
            table2_path: Path to the second table
            candidate_pairs: Candidate column pairs

        Returns:
            Validation result dictionary
        """
        # Step 1: Check if phase1 verification already exists (stored in coarse/ directory for compatibility)
        coarse_exists, coarse_filepath = check_verification_exists(
            table1_path, table2_path, 'coarse', self.verification_base_dir, 0
        )

        # If overwrite=True, ignore cache
        if self.overwrite:
            coarse_exists = False

        if coarse_exists:
            print(f"[Cache Hit] Phase1 verification: {coarse_filepath}")
            phase1_result = load_verification_result(coarse_filepath)
            self.stats['cache_hits'] += 1
            self._accumulate_cached_stats(phase1_result, 'coarse')
            # Restore conversation history from readable cache for Phase2
            self._restore_conversation_from_cache(table1_path, table2_path, 'coarse', 0)
        else:
            # Call phase1 verification
            print(f"[LLM Call] Phase1 verification: {table1_path} <-> {table2_path}")
            phase1_result = self.llm_verifier.phase1_verification(
                table1_path, table2_path, candidate_pairs
            )
            # Save result
            coarse_filepath = get_verification_filepath(
                table1_path, table2_path, 'coarse', self.verification_base_dir, 0
            )
            save_verification_result(coarse_filepath, phase1_result)

            # Save visualization file
            conversation_history = self.llm_verifier.get_conversation_history(table1_path, table2_path)
            save_verification_visualization(
                table1_path, table2_path, 'coarse', self.verification_base_dir,
                phase1_result, repeat_idx=0, conversation_history=conversation_history
            )

        # Extract selected pairs from Phase1
        selected_pairs = phase1_result.get('selected_pairs', [])
        if not selected_pairs:
            return {
                'is_valid': False,
                'join_pairs': [],
                'is_critical': False,
                'verification_count': 1,
                'phase1_result': phase1_result,
                'reasoning': 'No valid column pairs found in Phase1 verification'
            }

        # Parse selected pairs into column tuples
        column_pairs = self._parse_selected_pairs(selected_pairs)

        # Step 2: Check if phase2 verification already exists (stored in fine/ directory for compatibility)
        fine_exists, fine_filepath = check_verification_exists(
            table1_path, table2_path, 'fine', self.verification_base_dir, 0
        )

        # If overwrite=True, ignore cache
        if self.overwrite:
            fine_exists = False

        if fine_exists:
            print(f"[Cache Hit] Phase2 verification: {fine_filepath}")
            phase2_result = load_verification_result(fine_filepath)
            self.stats['cache_hits'] += 1
            self._accumulate_cached_stats(phase2_result, 'fine')
        else:
            # Call phase2 verification
            print(f"[LLM Call] Phase2 verification: {table1_path} <-> {table2_path}")
            phase2_result = self.llm_verifier.phase2_verification(
                table1_path, table2_path, column_pairs
            )
            # Save result
            fine_filepath = get_verification_filepath(
                table1_path, table2_path, 'fine', self.verification_base_dir, 0
            )
            save_verification_result(fine_filepath, phase2_result)

            # Save visualization file
            conversation_history = self.llm_verifier.get_conversation_history(table1_path, table2_path)
            save_verification_visualization(
                table1_path, table2_path, 'fine', self.verification_base_dir,
                phase2_result, repeat_idx=0, conversation_history=conversation_history
            )

        valid_pairs = phase2_result.get('valid_pairs', [])

        return {
            'is_valid': len(valid_pairs) > 0,
            'join_pairs': valid_pairs,
            'is_critical': False,
            'verification_count': 2,
            'phase1_result': phase1_result,
            'phase2_result': phase2_result,
            'reasoning': f"Found {len(valid_pairs)} valid join column pairs"
        }

    def _validate_critical_edge(self, table1_path: str, table2_path: str,
                               candidate_pairs: List[Dict]) -> Dict:
        """
        Validate critical edge: 3 phase1+phase2 verifications with voting

        Args:
            table1_path: Path to the first table
            table2_path: Path to the second table
            candidate_pairs: Candidate column pairs

        Returns:
            Validation result dictionary with voting information
        """
        votes = []

        # Execute 3 verifications
        for i in range(3):
            # Check if phase1 verification already exists
            coarse_exists, coarse_filepath = check_verification_exists(
                table1_path, table2_path, 'coarse', self.verification_base_dir, i
            )

            # If overwrite=True, ignore cache
            if self.overwrite:
                coarse_exists = False

            if coarse_exists:
                print(f"[Cache Hit] Phase1 verification {i+1}/3: {coarse_filepath}")
                phase1_result = load_verification_result(coarse_filepath)
                self.stats['cache_hits'] += 1
                self._accumulate_cached_stats(phase1_result, 'coarse')
                # Restore conversation history from readable cache for Phase2
                self._restore_conversation_from_cache(table1_path, table2_path, 'coarse', i)
            else:
                # Call LLM for phase1 verification
                print(f"[LLM Call] Phase1 verification {i+1}/3: {table1_path} <-> {table2_path}")
                phase1_result = self.llm_verifier.phase1_verification(
                    table1_path, table2_path, candidate_pairs
                )
                # Save result
                coarse_filepath = get_verification_filepath(
                    table1_path, table2_path, 'coarse', self.verification_base_dir, i
                )
                save_verification_result(coarse_filepath, phase1_result)

                # Save visualization file
                conversation_history = self.llm_verifier.get_conversation_history(table1_path, table2_path)
                save_verification_visualization(
                    table1_path, table2_path, 'coarse', self.verification_base_dir,
                    phase1_result, repeat_idx=i, conversation_history=conversation_history
                )

            # Extract selected pairs
            selected_pairs = phase1_result.get('selected_pairs', [])
            if not selected_pairs:
                votes.append({
                    'vote_id': i + 1,
                    'is_valid': False,
                    'join_pairs': [],
                    'phase1_result': phase1_result,
                    'reasoning': 'No valid column pairs found in Phase1 verification'
                })
                continue

            # Parse column pairs
            column_pairs = self._parse_selected_pairs(selected_pairs)

            # Check if phase2 verification already exists
            fine_exists, fine_filepath = check_verification_exists(
                table1_path, table2_path, 'fine', self.verification_base_dir, i
            )

            # If overwrite=True, ignore cache
            if self.overwrite:
                fine_exists = False

            if fine_exists:
                print(f"[Cache Hit] Phase2 verification {i+1}/3: {fine_filepath}")
                phase2_result = load_verification_result(fine_filepath)
                self.stats['cache_hits'] += 1
                self._accumulate_cached_stats(phase2_result, 'fine')
            else:
                # Call LLM for phase2 verification
                print(f"[LLM Call] Phase2 verification {i+1}/3: {table1_path} <-> {table2_path}")
                phase2_result = self.llm_verifier.phase2_verification(
                    table1_path, table2_path, column_pairs
                )
                # Save result
                fine_filepath = get_verification_filepath(
                    table1_path, table2_path, 'fine', self.verification_base_dir, i
                )
                save_verification_result(fine_filepath, phase2_result)

                # Save visualization file
                conversation_history = self.llm_verifier.get_conversation_history(table1_path, table2_path)
                save_verification_visualization(
                    table1_path, table2_path, 'fine', self.verification_base_dir,
                    phase2_result, repeat_idx=i, conversation_history=conversation_history
                )

            valid_pairs = phase2_result.get('valid_pairs', [])

            votes.append({
                'vote_id': i + 1,
                'is_valid': len(valid_pairs) > 0,
                'join_pairs': valid_pairs,
                'phase1_result': phase1_result,
                'phase2_result': phase2_result,
                'reasoning': f"Found {len(valid_pairs)} valid join column pairs"
            })

        # Count votes
        valid_votes = sum(1 for vote in votes if vote['is_valid'])
        is_valid = valid_votes >= 2  # Majority voting

        # Aggregate join pairs from valid votes
        all_join_pairs = []
        for vote in votes:
            if vote['is_valid']:
                all_join_pairs.extend(vote['join_pairs'])

        # Deduplicate join pairs
        unique_pairs = {}
        for pair in all_join_pairs:
            key = f"{pair['column1']}<->{pair['column2']}"
            if key not in unique_pairs:
                unique_pairs[key] = pair

        return {
            'is_valid': is_valid,
            'join_pairs': list(unique_pairs.values()),
            'is_critical': True,
            'verification_count': 3,
            'votes': votes,
            'valid_votes': valid_votes,
            'reasoning': f"Voting result: {valid_votes}/3 votes are valid"
        }

    def _find_table_path(self, table_name: str) -> Optional[str]:
        """Find the full path of table file"""
        # Try direct path
        direct_path = os.path.join(self.tables_dir, f"{table_name}.csv")
        if os.path.exists(direct_path):
            return direct_path

        # Try searching in subdirectories
        for root, dirs, files in os.walk(self.tables_dir):
            for file in files:
                if file == f"{table_name}.csv" or file == table_name:
                    return os.path.join(root, file)

        return None

    def validate_subgraph_edges(self, subgraph_data: Dict, global_graph: nx.Graph,
                               candidate_pairs_dict: Dict[Tuple[str, str], List[Dict]],
                               top_k: int = 10, max_path_length: int = 5) -> Dict:
        """
        Validate all edges in the subgraph

        Args:
            subgraph_data: Subgraph data dictionary
            global_graph: Global graph
            candidate_pairs_dict: Dictionary mapping edge tuples to candidate column pairs
            top_k: Number of shortest paths for critical edge detection
            max_path_length: Maximum path length

        Returns:
            Dictionary of validation results for all edges
        """
        # Identify critical edges
        critical_edges = self.identify_critical_edges(subgraph_data, global_graph,
                                                     top_k, max_path_length)

        # Validate each edge
        validation_results = {}
        edges_to_validate = []

        for edge_data in subgraph_data['edges']:
            edge = tuple(sorted([edge_data['node1'], edge_data['node2']]))
            edges_to_validate.append((edge, edge in critical_edges))

        print(f"\n[EdgeValidator] Validating {len(edges_to_validate)} edges "
              f"({len(critical_edges)} critical edges)")

        for edge, is_critical in edges_to_validate:
            table1, table2 = edge
            candidate_pairs = candidate_pairs_dict.get(edge, [])

            print(f"\n  Validating edge: {table1} <-> {table2} "
                  f"({'critical' if is_critical else 'normal'})")

            result = self.validate_edge(table1, table2, candidate_pairs, is_critical)
            validation_results[f"{table1}<->{table2}"] = result

            print(f"    Result: {'valid' if result['is_valid'] else 'invalid'} "
                  f"({len(result['join_pairs'])} join pairs)")

        return validation_results

    def get_invalid_edges(self, validation_results: Dict) -> List[Tuple[str, str]]:
        """
        Get list of invalid edges from validation results

        Args:
            validation_results: Validation results dictionary

        Returns:
            List of invalid edge tuples
        """
        invalid_edges = []
        for edge_key, result in validation_results.items():
            if not result['is_valid']:
                # Parse edge key "table1<->table2"
                parts = edge_key.split('<->')
                if len(parts) == 2:
                    invalid_edges.append(tuple(sorted(parts)))

        return invalid_edges

    def apply_validation_to_graph(self, graph: nx.Graph, validation_results: Dict) -> nx.Graph:
        """
        Modify graph based on validation results:
        - Valid edges: Set weight to 0
        - Invalid edges: Remove from graph

        Args:
            graph: NetworkX graph object to modify
            validation_results: Validation results dictionary

        Returns:
            Modified graph object
        """
        modified_graph = graph.copy()

        valid_edges = []
        invalid_edges = []

        for edge_key, result in validation_results.items():
            # Parse edge key "table1<->table2"
            parts = edge_key.split('<->')
            if len(parts) != 2:
                continue

            table1, table2 = parts[0].strip(), parts[1].strip()
            edge = tuple(sorted([table1, table2]))

            if result['is_valid']:
                valid_edges.append(edge)
            else:
                invalid_edges.append(edge)

        # Process valid edges: Set weight to 0
        for edge in valid_edges:
            if modified_graph.has_edge(edge[0], edge[1]):
                modified_graph[edge[0]][edge[1]]['weight'] = 0.0
                print(f"[Graph Modification] Valid edge weight set to 0: {edge[0]} <-> {edge[1]}")

        # Process invalid edges: Remove from graph
        for edge in invalid_edges:
            if modified_graph.has_edge(edge[0], edge[1]):
                modified_graph.remove_edge(edge[0], edge[1])
                print(f"[Graph Modification] Removed invalid edge: {edge[0]} <-> {edge[1]}")

        print(f"\n[Graph Modification Statistics]:")
        print(f"  - Valid edges (weight set to 0): {len(valid_edges)}")
        print(f"  - Invalid edges (removed): {len(invalid_edges)}")
        print(f"  - Number of edges after modification: {modified_graph.number_of_edges()}")

        return modified_graph

    def get_statistics(self) -> Dict:
        """Get validation statistics"""
        return {
            'total_edges_validated': self.stats['total_edges_validated'],
            'critical_edges': self.stats['critical_edges'],
            'normal_edges': self.stats['normal_edges'],
            'valid_edges': self.stats['valid_edges'],
            'invalid_edges': self.stats['invalid_edges'],
            'cache_hits': self.stats['cache_hits'],
            'validation_rate': self.stats['valid_edges'] / max(1, self.stats['total_edges_validated'])
        }

    def print_statistics(self):
        """Print validation statistics"""
        stats = self.get_statistics()
        print("\n" + "=" * 60)
        print("Edge Validation Statistics")
        print("=" * 60)
        print(f"Total edges validated:  {stats['total_edges_validated']}")
        print(f"  - Critical edges:     {stats['critical_edges']}")
        print(f"  - Normal edges:       {stats['normal_edges']}")
        print(f"Valid edges:            {stats['valid_edges']}")
        print(f"Invalid edges:          {stats['invalid_edges']}")
        print(f"Validation pass rate:   {stats['validation_rate']:.2%}")
        print(f"Cache hits:             {stats['cache_hits']}")
        print("=" * 60)
