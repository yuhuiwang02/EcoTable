"""
Verification Utilities Module

Utility functions for edge verification file naming and detection.
"""

import os
import json
import hashlib
from typing import Tuple, Optional, Dict


def get_edge_filename(table1: str, table2: str) -> str:
    """
    Generate filename for edge verification based on table names

    Rules:
    - If table name doesn't contain 'node', use only CSV filename (no path)
    - If table name contains 'node', use relative path
    - Sort the two names and join with '-'
    - If filename exceeds 200 characters, truncate and append hash for uniqueness

    Args:
        table1: First table name/path
        table2: Second table name/path

    Returns:
        Edge filename (without extension)
    """
    def process_table_name(table_name: str) -> str:
        # Remove .csv extension
        if table_name.endswith('.csv'):
            table_name = table_name[:-4]

        # Unify path separators
        table_name = table_name.replace('\\', '/')

        # If contains 'node', keep relative path and replace / with _
        if 'node' in table_name.lower():
            return table_name.replace('/', '_')
        else:
            # Keep only filename (last segment)
            return table_name.rsplit('/', 1)[-1]

    name1 = process_table_name(table1)
    name2 = process_table_name(table2)

    # Sort and join with '-'
    sorted_names = sorted([name1, name2])
    filename = f"{sorted_names[0]}-{sorted_names[1]}"

    # Truncate and append hash if filename is too long
    if len(filename) > 200:
        full_hash = hashlib.md5(filename.encode()).hexdigest()[:12]
        filename = f"{filename[:187]}_{full_hash}"

    return filename


def get_verification_filepath(table1: str, table2: str,
                              verification_type: str,
                              base_dir: str,
                              repeat_idx: int = 0) -> str:
    """
    Get full file path for edge verification result

    Args:
        table1: First table name/path
        table2: Second table name/path
        verification_type: Verification type 'coarse' or 'fine'
        base_dir: Base directory for verification files
        repeat_idx: Index for repeated verification (0 means first verification)

    Returns:
        Full path to verification file
    """
    # Create subdirectory for verification type
    type_dir = os.path.join(base_dir, verification_type)
    os.makedirs(type_dir, exist_ok=True)

    # Get base filename
    filename = get_edge_filename(table1, table2)

    # Add repeat verification prefix if needed
    if repeat_idx > 0:
        filename = f"{repeat_idx}){filename}"

    return os.path.join(type_dir, f"{filename}.json")


def check_verification_exists(table1: str, table2: str,
                              verification_type: str,
                              base_dir: str,
                              repeat_idx: int = 0) -> Tuple[bool, Optional[str]]:
    """
    Check if verification result already exists

    Args:
        table1: First table name/path
        table2: Second table name/path
        verification_type: Verification type 'coarse' or 'fine'
        base_dir: Base directory for verification files
        repeat_idx: Index for repeated verification

    Returns:
        Tuple (exists: bool, filepath: str or None)
    """
    filepath = get_verification_filepath(table1, table2, verification_type,
                                        base_dir, repeat_idx)
    exists = os.path.exists(filepath)
    return exists, filepath if exists else None


def load_verification_result(filepath: str) -> Optional[Dict]:
    """
    Load verification result from file

    Args:
        filepath: Verification file path

    Returns:
        Verification result dictionary, or None on failure
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"[Error] Failed to load verification result from {filepath}: {e}")
        return None


def save_verification_result(filepath: str, result: Dict):
    """
    Save verification result to file

    Args:
        filepath: Path to save verification file
        result: Verification result dictionary
    """
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"[Saved] Verification result: {filepath}")
    except Exception as e:
        print(f"[Error] Failed to save verification result to {filepath}: {e}")


def save_verification_visualization(table1: str, table2: str,
                                    verification_type: str,
                                    base_dir: str,
                                    result: Dict,
                                    repeat_idx: int = 0,
                                    conversation_history: Optional[list] = None):
    """
    Save visualization version of verification result (more readable format)

    Args:
        table1: First table name/path
        table2: Second table name/path
        verification_type: Verification type 'coarse' or 'fine'
        base_dir: Base directory for verification files
        result: Verification result dictionary
        repeat_idx: Index for repeated verification
        conversation_history: Conversation history (optional)
    """
    try:
        # Create visualization directory
        viz_dir = os.path.join(base_dir, f"{verification_type}_readable")
        os.makedirs(viz_dir, exist_ok=True)

        # Get base filename
        filename = get_edge_filename(table1, table2)
        if repeat_idx > 0:
            filename = f"{repeat_idx}){filename}"

        viz_filepath = os.path.join(viz_dir, f"{filename}.json")

        # Build visualization data structure
        viz_data = {
            "metadata": {
                "table1": table1,
                "table2": table2,
                "verification_type": verification_type,
                "repeat_index": repeat_idx,
                "edge_key": f"{table1} <-> {table2}"
            },
            "verification_result": result
        }

        # If conversation history is provided, add to visualization data
        if conversation_history:
            viz_data["conversation_history"] = conversation_history

        # Add readability enhancements
        if verification_type == "coarse":
            # Visualization enhancement for coarse verification
            approved_count = len(result.get('approved_pairs', []))
            additional_count = len(result.get('additional_pairs', []))
            viz_data["summary"] = {
                "approved_pairs_count": approved_count,
                "additional_pairs_count": additional_count,
                "total_pairs": approved_count + additional_count,
                "has_valid_pairs": (approved_count + additional_count) > 0
            }

            # List all approved column pairs
            all_pairs = []
            for pair in result.get('approved_pairs', []):
                if pair.get('approved', False):
                    all_pairs.append(f"{pair['column1']} <-> {pair['column2']}")
            for pair in result.get('additional_pairs', []):
                all_pairs.append(f"{pair['column1']} <-> {pair['column2']}")
            viz_data["summary"]["all_approved_pairs"] = all_pairs

        elif verification_type == "fine":
            # Visualization enhancement for fine verification
            valid_pairs = result.get('valid_pairs', [])
            viz_data["summary"] = {
                "valid_pairs_count": len(valid_pairs),
                "has_valid_pairs": len(valid_pairs) > 0
            }

        # Save visualization file
        os.makedirs(os.path.dirname(viz_filepath), exist_ok=True)
        with open(viz_filepath, 'w', encoding='utf-8') as f:
            json.dump(viz_data, f, indent=2, ensure_ascii=False)

        print(f"[Saved] Visualization file: {viz_filepath}")

    except Exception as e:
        print(f"[Error] Failed to save visualization file: {e}")


def get_query_output_path(output_dir: str, query_idx: int) -> str:
    """
    Get output path for query join_path result

    Args:
        output_dir: Base output directory
        query_idx: Query index

    Returns:
        Full path to query output file
    """
    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(output_dir, f"{query_idx}.json")


def save_query_result(output_path: str, query_result: Dict):
    """
    Save query evaluation result (including join_path)

    Args:
        output_path: Path to save query result
        query_result: Query result dictionary
    """
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(query_result, f, indent=2, ensure_ascii=False)
        print(f"[Saved] Query result: {output_path}")
    except Exception as e:
        print(f"[Error] Failed to save query result to {output_path}: {e}")
