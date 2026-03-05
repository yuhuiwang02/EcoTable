"""
Layer3 Transform Agent Interface
Provides interface for layer3/run.py - LLM-based data transformation on join graph
"""

import os
import sys
import json
import argparse
from typing import Dict, List, Optional

try:
    from .run import run as layer3_run, config as layer3_config
except ImportError:
    try:
        from run import run as layer3_run, config as layer3_config
    except ImportError:
        layer3_run = None
        layer3_config = None


class TransformAgentRunner:
    """
    Wrapper for layer3 transformation agent
    Executes LLM-based transformations on join graph edges
    """

    def __init__(
        self,
        graph_json: str,
        duckdb_path: str,
        output_dir: str,
        model: str = "gpt-4o",
        temperature: float = 0.1,
        top_p: float = 0.9,
        max_tokens: int = 2500,
        max_iterations: int = 10,
        max_workers: int = 8,
        test_mode: bool = False,
        overwriting: bool = False,
        eval_file: Optional[str] = None
    ):
        """
        Initialize transform agent runner

        Args:
            graph_json: Path to pruned graph JSON file (from join_evaluation)
            duckdb_path: Path to DuckDB file containing all tables
            output_dir: Output directory for transformation results
            model: LLM model name
            temperature: LLM temperature
            top_p: LLM top_p
            max_tokens: LLM max tokens
            max_iterations: Max iterations per edge transformation
            max_workers: Max parallel workers per color group
            test_mode: Test mode (no LLM calls)
            overwriting: Overwrite existing results
            eval_file: Optional evaluation results JSON (filter edges)
        """
        self.graph_json = graph_json
        self.duckdb_path = duckdb_path
        self.output_dir = output_dir
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.max_iterations = max_iterations
        self.max_workers = max_workers
        self.test_mode = test_mode
        self.overwriting = overwriting
        self.eval_file = eval_file

    def run(self) -> Dict:
        """
        Run transformation agent on the join graph

        Returns:
            Summary dictionary with results and timing
        """
        if layer3_run is None:
            raise ImportError(
                "Could not import layer3.run module. "
                "Make sure layer3/ directory exists and contains run.py"
            )

        # Validate inputs
        if not os.path.exists(self.graph_json):
            raise FileNotFoundError(f"Graph JSON not found: {self.graph_json}")
        if not os.path.exists(self.duckdb_path):
            raise FileNotFoundError(f"DuckDB file not found: {self.duckdb_path}")

        # Create argparse Namespace with our parameters
        args = argparse.Namespace(
            graph=self.graph_json,
            duckdb=self.duckdb_path,
            output_dir=self.output_dir,
            model=self.model,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            max_iterations=self.max_iterations,
            test_mode=self.test_mode,
            overwriting=self.overwriting,
            eval_file=self.eval_file,
            max_workers=self.max_workers
        )

        # Run layer3 transformation
        layer3_run(args)

        # Load and return summary
        summary_path = os.path.join(self.output_dir, "summary.json")
        if os.path.exists(summary_path):
            with open(summary_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            return {"error": "Summary file not found"}


def run_transform_agent(
    graph_json: str,
    duckdb_path: str,
    output_dir: str = "transform_output",
    model: str = "gpt-4o",
    max_iterations: int = 10,
    max_workers: int = 8,
    test_mode: bool = False,
    eval_file: Optional[str] = None,
    verbose: bool = True
) -> Dict:
    """
    Simple interface to run transformation agent

    Args:
        graph_json: Path to pruned graph JSON (from join_evaluation)
        duckdb_path: Path to DuckDB file with tables
        output_dir: Output directory
        model: LLM model name
        max_iterations: Max iterations per edge
        max_workers: Max parallel workers
        test_mode: Test mode (no LLM calls)
        eval_file: Optional evaluation results JSON
        verbose: Print progress

    Returns:
        Summary dictionary with results
    """
    if verbose:
        print(f"\n[Transform Agent] Starting...")
        print(f"  Graph: {graph_json}")
        print(f"  DuckDB: {duckdb_path}")
        print(f"  Output: {output_dir}")
        print(f"  Model: {model}")

    runner = TransformAgentRunner(
        graph_json=graph_json,
        duckdb_path=duckdb_path,
        output_dir=output_dir,
        model=model,
        max_iterations=max_iterations,
        max_workers=max_workers,
        test_mode=test_mode,
        eval_file=eval_file
    )

    summary = runner.run()

    if verbose:
        print(f"\n[Transform Agent] Completed!")
        if "error" not in summary:
            print(f"  Processed: {summary.get('processed_edges', 0)} edges")
            print(f"  Skipped: {summary.get('skipped_edges', 0)} edges")
            print(f"  Errors: {summary.get('error_edges', 0)} edges")
            timing = summary.get('timing', {})
            print(f"  Parallel time: {timing.get('total_parallel_time_seconds', 0):.1f}s")
            print(f"  Speedup: {timing.get('speedup_ratio', 0):.2f}x")

    return summary


def prepare_duckdb_from_tables(
    tables_dir: str,
    selected_tables: List[str],
    output_duckdb: str,
    verbose: bool = True
) -> str:
    """
    Create DuckDB file from selected CSV tables

    Args:
        tables_dir: Root directory containing table CSV files
        selected_tables: List of table paths (e.g., ["project/table.csv"])
        output_duckdb: Output DuckDB file path
        verbose: Print progress

    Returns:
        Path to created DuckDB file
    """
    try:
        import duckdb
    except ImportError:
        raise ImportError("duckdb package required. Install with: pip install duckdb")

    if verbose:
        print(f"\n[DuckDB] Creating database: {output_duckdb}")
        print(f"  Loading {len(selected_tables)} tables...")

    # Create DuckDB and load tables
    con = duckdb.connect(output_duckdb)

    loaded_count = 0
    for table_path in selected_tables:
        full_path = os.path.join(tables_dir, table_path)
        if not os.path.exists(full_path):
            if verbose:
                print(f"  Warning: Table not found: {full_path}")
            continue

        # Extract table name (use file stem without .csv)
        table_name = os.path.splitext(os.path.basename(table_path))[0]

        try:
            # Load CSV into DuckDB
            con.execute(
                f'CREATE TABLE "{table_name}" AS '
                f'SELECT * FROM read_csv_auto(\'{full_path}\')'
            )
            loaded_count += 1
            if verbose and loaded_count % 10 == 0:
                print(f"  Loaded {loaded_count} tables...")
        except Exception as e:
            if verbose:
                print(f"  Error loading {table_name}: {e}")

    con.close()

    if verbose:
        print(f"[DuckDB] Loaded {loaded_count}/{len(selected_tables)} tables")

    return output_duckdb
