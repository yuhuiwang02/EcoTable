"""
Table Name Converter Interface
Provides reusable functions from convert_selected_tables.py
"""

import os
import json
from typing import List, Tuple, Dict, Optional


class TableNameConverter:
    """Convert table names to file paths"""

    def __init__(self, dataset_dir: str, suffixes: List[str] = None):
        """
        Initialize converter

        Args:
            dataset_dir: Dataset root directory
            suffixes: Folder suffixes to check (default: _normal, _no_overlap, _no_join_cols)
        """
        self.dataset_dir = dataset_dir
        self.suffixes = suffixes or ["_normal", "_no_overlap", "_no_join_cols"]
        self.file_index = {}
        self.dir_index = {}
        self._build_index()

    def _build_index(self):
        """Build file and directory indexes"""
        for dir_name in os.listdir(self.dataset_dir):
            dir_path = os.path.join(self.dataset_dir, dir_name)
            if not os.path.isdir(dir_path):
                continue
            self.dir_index[dir_name.lower()] = dir_name
            for fname in os.listdir(dir_path):
                if fname.endswith(".csv"):
                    self.file_index[fname.lower()] = f"{dir_name}/{fname}"

    def resolve_node_table(self, table_name: str) -> Optional[str]:
        """
        Resolve a node table name to its relative path

        Args:
            table_name: Table name (e.g., "project_node_0")

        Returns:
            Relative path (e.g., "project_normal/node_0.csv") or None
        """
        if "node" not in table_name:
            return None

        idx = table_name.index("node")
        if idx > 0 and table_name[idx - 1] == "_":
            prefix = table_name[:idx - 1]
            node_file = table_name[idx:] + ".csv"
        else:
            return None

        for suffix in self.suffixes:
            candidate_dir = (prefix + suffix).lower()
            if candidate_dir in self.dir_index:
                actual_dir = self.dir_index[candidate_dir]
                full_path = os.path.join(self.dataset_dir, actual_dir, node_file)
                if os.path.isfile(full_path):
                    return f"{actual_dir}/{node_file}"
        return None

    def convert_table_name(self, table_name: str) -> Optional[str]:
        """
        Convert a single table name to path

        Args:
            table_name: Table name

        Returns:
            Relative path or None if not found
        """
        # Handle node tables
        if "node" in table_name:
            return self.resolve_node_table(table_name)

        # Handle regular tables
        csv_key = (table_name + ".csv").lower()
        if csv_key in self.file_index:
            return self.file_index[csv_key]

        return None

    def convert_table_list(self, table_names: List[str],
                          skip_missing: bool = True) -> Tuple[List[str], List[str]]:
        """
        Convert a list of table names to paths

        Args:
            table_names: List of table names
            skip_missing: If True, skip missing tables; if False, raise error

        Returns:
            (converted_paths, missing_tables)
        """
        converted_paths = []
        missing_tables = []

        for table_name in table_names:
            path = self.convert_table_name(table_name)
            if path:
                converted_paths.append(path)
            else:
                missing_tables.append(table_name)
                if not skip_missing:
                    raise ValueError(f"Cannot resolve table: {table_name}")

        return converted_paths, missing_tables

    def convert_llm_filter_results(self, results: List[Dict],
                                   inplace: bool = True) -> List[Dict]:
        """
        Convert selected_tables in LLM filter results

        Args:
            results: List of result dictionaries with 'selected_tables' field
            inplace: If True, modify results in place; if False, return copy

        Returns:
            Results with converted table paths
        """
        if not inplace:
            import copy
            results = copy.deepcopy(results)

        total_converted = 0
        total_missing = 0

        for item in results:
            if 'selected_tables' not in item:
                continue

            table_names = item['selected_tables']
            converted, missing = self.convert_table_list(table_names, skip_missing=True)

            item['selected_tables'] = converted
            total_converted += len(converted)
            total_missing += len(missing)

            if missing:
                item['_conversion_warnings'] = missing

        print(f"[Table Converter] Converted {total_converted} tables, {total_missing} missing")

        return results


# Convenience functions for backward compatibility
def convert_table_names_to_paths(
    selected_tables: List[str],
    dataset_dir: str,
    suffixes: List[str] = None
) -> List[str]:
    """
    Simple function to convert table names to paths

    Args:
        selected_tables: List of table names
        dataset_dir: Dataset root directory
        suffixes: Folder suffixes to check

    Returns:
        List of converted paths (missing tables are skipped)
    """
    converter = TableNameConverter(dataset_dir, suffixes)
    converted, missing = converter.convert_table_list(selected_tables, skip_missing=True)

    if missing:
        print(f"  Warning: Could not resolve {len(missing)} tables: {missing[:3]}...")

    return converted


def convert_json_file(
    input_json: str,
    output_json: str,
    dataset_dir: str,
    mode: str = "report"
) -> Dict:
    """
    Convert selected_tables in a JSON file (original script functionality)

    Args:
        input_json: Input JSON file path
        output_json: Output JSON file path
        dataset_dir: Dataset root directory
        mode: Conversion mode ("strict", "skip", "report")

    Returns:
        Statistics dictionary
    """
    # Load JSON
    with open(input_json, 'r', encoding='utf-8') as f:
        data = json.load(f)

    converter = TableNameConverter(dataset_dir)

    # Convert
    if 'detailed_results' in data:
        detailed_results = data['detailed_results']
    else:
        detailed_results = data if isinstance(data, list) else []

    converted_count = 0
    error_count = 0
    errors = []
    clean_results = []

    for i, item in enumerate(detailed_results):
        sample_id = item.get('sample_id', i)

        if 'selected_tables' not in item or not item['selected_tables']:
            if mode == "skip":
                continue
            clean_results.append(item)
            continue

        converted, missing = converter.convert_table_list(
            item['selected_tables'],
            skip_missing=True
        )

        item['selected_tables'] = converted
        converted_count += len(converted)

        if missing:
            error_count += len(missing)
            errors.extend([(i, sample_id, t) for t in missing])
            if mode == "skip":
                continue

        clean_results.append(item)

    # Save result
    if mode == "skip":
        data['detailed_results'] = clean_results

    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    stats = {
        'converted_count': converted_count,
        'error_count': error_count,
        'errors': errors,
        'output_samples': len(clean_results)
    }

    print(f"Converted {converted_count} table entries")
    if errors:
        print(f"{error_count} tables could not be resolved")

    return stats
