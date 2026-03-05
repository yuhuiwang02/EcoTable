import argparse
import datetime
import glob
import hashlib
import json
import logging
import os
import re
import sys
import time
import shutil
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from .transform_agent.agent.agents import PromptAgent
import duckdb
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


#  Logger Configs {{{ #
logger = logging.getLogger("transform_agent")
logger.setLevel(logging.DEBUG)

datetime_str: str = datetime.datetime.now().strftime("%Y%m%d@%H%M%S")

os.makedirs("logs", exist_ok=True)

file_handler = logging.FileHandler(os.path.join("logs", "normal-{:}.log".format(datetime_str)), encoding="utf-8")
debug_handler = logging.FileHandler(os.path.join("logs", "debug-{:}.log".format(datetime_str)), encoding="utf-8")
stdout_handler = logging.StreamHandler(sys.stdout)

file_handler.setLevel(logging.INFO)
debug_handler.setLevel(logging.DEBUG)
stdout_handler.setLevel(logging.INFO)

formatter = logging.Formatter(
    fmt="\x1b[1;33m[%(asctime)s \x1b[31m%(levelname)s \x1b[32m%(module)s/%(lineno)d-%(threadName)s\x1b[1;33m] \x1b[0m%(message)s")
file_handler.setFormatter(formatter)
debug_handler.setFormatter(formatter)
stdout_handler.setFormatter(formatter)

stdout_handler.addFilter(logging.Filter("transform_agent"))

logger.addHandler(file_handler)
logger.addHandler(debug_handler)
logger.addHandler(stdout_handler)
#  }}} Logger Configs #


# ── Edge Folder Naming ───────────────────────────────────────────────────── #

def get_edge_folder_name(source: str, target: str) -> str:
    """
    Generate a deterministic, filesystem-safe folder name for an edge
    based on sorted node names joined by '-'.
    """
    sorted_nodes = sorted([source, target])
    raw_name = f"{sorted_nodes[0]}-{sorted_nodes[1]}"
    # Filesystem-safe character replacement
    safe = raw_name.replace("/", "_").replace("\\", "_").replace(" ", "_")
    safe = re.sub(r'[<>:"|?*]', '_', safe)
    if len(safe) > 200:
        hash_suffix = hashlib.md5(raw_name.encode()).hexdigest()[:8]
        safe = safe[:191] + "_" + hash_suffix
    return safe


# ── Edge Coloring ────────────────────────────────────────────────────────── #

def greedy_edge_coloring(target_edges):
    """
    Greedy edge coloring: assign colors so that no two edges sharing a node
    (source or target table) get the same color.

    Returns:
        dict mapping edge_idx -> color (int, 0-based)
    """
    coloring = {}
    node_used_colors = defaultdict(set)  # node_id -> set of colors

    for idx, edge in enumerate(target_edges):
        src = edge["source"]
        tgt = edge["target"]

        unavailable = node_used_colors[src] | node_used_colors[tgt]

        color = 0
        while color in unavailable:
            color += 1

        coloring[idx] = color
        node_used_colors[src].add(color)
        node_used_colors[tgt].add(color)

    return coloring


def group_by_color(coloring):
    """Group edge indices by their color. Returns {color: [edge_idx, ...]}."""
    groups = defaultdict(list)
    for edge_idx, color in coloring.items():
        groups[color].append(edge_idx)
    return dict(groups)


# ── Per-Edge Worker ──────────────────────────────────────────────────────── #

def process_single_edge(edge_idx, edge, edge_duckdb_path, agent, args, output_dir):
    """
    Process a single edge on its isolated DuckDB copy.
    All join column pairs accumulate transforms (no backup/restore).

    Returns:
        (edge_output_dict, elapsed_seconds, llm_elapsed_seconds)
    """
    start = time.time()

    source_table = edge["source"]
    target_table = edge["target"]
    join_columns = edge["join_columns"]
    weight = edge.get("weight", 0)

    logger.info(f"[Edge {edge_idx}] {source_table} <-> {target_table} ({len(join_columns)} pairs)")

    edge_output_dir = os.path.join(output_dir, get_edge_folder_name(source_table, target_table))
    os.makedirs(edge_output_dir, exist_ok=True)

    edge_results = []
    edge_llm_ms = 0.0
    for pair_idx, col_pair in enumerate(join_columns):
        if not isinstance(col_pair, (list, tuple)) or len(col_pair) != 2:
            logger.warning(f"[Edge {edge_idx}] Invalid pair: {col_pair}, skipping")
            continue

        join_col1, join_col2 = col_pair
        logger.info(f"[Edge {edge_idx}] {join_col1} <-> {join_col2}:")

        try:
            result = agent.run_new_workflow(
                duckdb_path=edge_duckdb_path,
                table1=source_table,
                table2=target_table,
                join_col1=join_col1,
                join_col2=join_col2,
                max_iterations=args.max_iterations,
                test_mode=args.test_mode,
                output_dir=edge_output_dir,
                pair_idx=pair_idx
            )

            edge_llm_ms += result.get('total_llm_latency_ms', 0.0)

            edge_results.append({
                "join_col1": join_col1,
                "join_col2": join_col2,
                **result
            })

        except Exception as e:
            logger.error(f"[Edge {edge_idx}] Failed pair {join_col1}-{join_col2}: {e}",
                        exc_info=True)
            edge_results.append({
                "join_col1": join_col1,
                "join_col2": join_col2,
                "error": str(e)
            })

    edge_output = {
        "source": source_table,
        "target": target_table,
        "weight": weight,
        "join_columns": join_columns,
        "results": edge_results
    }

    # Save edge result
    result_path = os.path.join(edge_output_dir, "result.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(edge_output, f, indent=2, ensure_ascii=False)

    elapsed = time.time() - start
    llm_elapsed = edge_llm_ms / 1000.0
    logger.info(f"[Edge {edge_idx}] Finished in {elapsed:.1f}s (LLM: {llm_elapsed:.1f}s)")

    return edge_output, elapsed, llm_elapsed


# ── DuckDB Merge ─────────────────────────────────────────────────────────── #

def merge_tables_from_edge_db(edge_idx, edge, edge_duckdb_path, work_duckdb):
    """
    Merge modified tables from an edge-specific DuckDB back into the main DuckDB.
    Same-color edges touch disjoint tables, so merge is conflict-free.
    """
    tables_to_merge = {edge["source"], edge["target"]}

    with duckdb.connect(work_duckdb) as con:
        con.execute(f"ATTACH '{edge_duckdb_path}' AS edge_db (READ_ONLY)")

        for table_name in tables_to_merge:
            try:
                con.execute(f'DROP TABLE IF EXISTS main."{table_name}"')
                con.execute(
                    f'CREATE TABLE main."{table_name}" AS '
                    f'SELECT * FROM edge_db.main."{table_name}"'
                )
                logger.debug(f"[Edge {edge_idx}] Merged table '{table_name}' back to main DB")
            except Exception as e:
                logger.error(f"[Edge {edge_idx}] Failed to merge table '{table_name}': {e}")

        con.execute("DETACH edge_db")


def safe_remove(path, retries=3, delay=0.5):
    """Remove a file with retries (Windows may hold DuckDB locks briefly)."""
    for i in range(retries):
        try:
            if os.path.exists(path):
                os.remove(path)
            return
        except PermissionError:
            if i < retries - 1:
                time.sleep(delay)
            else:
                logger.warning(f"Could not remove {path} after {retries} attempts")


# ── CLI Config ───────────────────────────────────────────────────────────── #

def config() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run transformation agent on a join graph"
    )

    # Graph input
    parser.add_argument("--graph", type=str, required=True,
                        help="Path to the pruned graph JSON file")
    parser.add_argument("--duckdb", type=str, required=True,
                        help="Path to the DuckDB file containing all tables")
    parser.add_argument("--output_dir", type=str, default="output",
                        help="Output directory for results")

    # LLM config
    parser.add_argument("--model", type=str, default="gpt-4o")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_tokens", type=int, default=2500)

    # Workflow config
    parser.add_argument("--max_iterations", type=int, default=10,
                        help="Max iterations per edge in the three-phase workflow")
    parser.add_argument("--test_mode", action="store_true", default=False,
                        help="Test mode: only run discovery + evaluation, output prompt without calling LLM")
    parser.add_argument("--overwriting", action="store_true", default=False)

    # Edge filter
    parser.add_argument("--eval_file", type=str, default=None,
                        help="Path to evaluation results JSON; only edges appearing "
                             "in steiner_edges will be processed")

    # Parallel config
    parser.add_argument("--max_workers", type=int, default=8,
                        help="Max parallel workers per color group (default: 8)")

    args = parser.parse_args()
    return args


# ── Main ─────────────────────────────────────────────────────────────────── #

def run(args: argparse.Namespace) -> None:
    logger.info("Args: %s", args)

    # Validate duckdb path
    if not os.path.exists(args.duckdb):
        logger.error(f"DuckDB file not found: {args.duckdb}")
        return

    # Load graph
    with open(args.graph, "r", encoding="utf-8") as f:
        graph_data = json.load(f)

    edges = graph_data.get("graph", {}).get("edges", [])
    metadata = graph_data.get("metadata", {})
    target_edges = [e for e in edges if "join_columns" in e]
    logger.info(f"Graph: {len(edges)} edges, {len(target_edges)} with join_columns")

    # Optional: filter to only edges referenced in evaluation results
    if args.eval_file:
        with open(args.eval_file, "r", encoding="utf-8") as f:
            eval_data = json.load(f)
        eval_edge_set = set()
        for query in eval_data:
            for e in query.get("steiner_edges", []):
                eval_edge_set.add(tuple(sorted(e)))
        before = len(target_edges)
        target_edges = [
            e for e in target_edges
            if tuple(sorted([e["source"], e["target"]])) in eval_edge_set
        ]
        logger.info(f"Eval filter: {before} -> {len(target_edges)} edges "
                     f"(eval file has {len(eval_edge_set)} unique edges)")

    if not target_edges:
        logger.info("No edges to process, exiting.")
        return

    # Create agent
    agent = PromptAgent(
        model=args.model,
        max_tokens=args.max_tokens,
        top_p=args.top_p,
        temperature=args.temperature,
    )

    if args.test_mode:
        logger.info("*** TEST MODE ENABLED - No LLM calls, no transformations ***")

    # Copy DuckDB once to output dir for safe modification
    os.makedirs(args.output_dir, exist_ok=True)
    work_duckdb = os.path.join(args.output_dir, os.path.basename(args.duckdb))
    duckdb_freshly_copied = False
    if not os.path.exists(work_duckdb) or args.overwriting:
        logger.debug(f"Copying DuckDB to {work_duckdb}")
        shutil.copy2(args.duckdb, work_duckdb)
        duckdb_freshly_copied = True
    else:
        logger.debug(f"Using existing DuckDB: {work_duckdb}")

    # ── Edge Coloring ──────────────────────────────────────────────────── #
    coloring = greedy_edge_coloring(target_edges)
    color_groups = group_by_color(coloring)
    num_colors = len(color_groups)

    logger.info(f"Edge coloring: {len(target_edges)} edges -> {num_colors} color(s)")

    # ── Process by Color Group ─────────────────────────────────────────── #
    all_results = [None] * len(target_edges)
    total_serial_time = 0.0
    total_parallel_time = 0.0
    llm_serial_time = 0.0
    llm_parallel_time = 0.0
    skipped_llm_time = 0.0
    per_color_timing = {}
    skipped_count = 0
    error_count = 0

    for color in sorted(color_groups):
        edge_indices = color_groups[color]

        logger.info(f"Color {color}/{num_colors-1}: {len(edge_indices)} edges {edge_indices}")

        # Filter out already-processed edges
        edges_to_process = []
        edges_to_replay = []
        for idx in edge_indices:
            edge_folder = os.path.join(args.output_dir,
                                       get_edge_folder_name(target_edges[idx]["source"],
                                                            target_edges[idx]["target"]))
            result_path = os.path.join(edge_folder, "result.json")
            if not args.overwriting and os.path.exists(result_path):
                logger.debug(f"[Edge {idx}] Skipping, result exists")
                skipped_count += 1
                try:
                    with open(result_path, "r", encoding="utf-8") as f:
                        all_results[idx] = json.load(f)
                except Exception:
                    pass
                # Extract LLM latency from conversation files for reuse accounting
                conv_files = glob.glob(os.path.join(edge_folder, "conversation_*.json"))
                reused_llm_ms = 0.0
                for cf in conv_files:
                    try:
                        with open(cf, "r", encoding="utf-8") as f:
                            conv_data = json.load(f)
                            reused_llm_ms += conv_data.get("total_latency_ms", 0.0)
                    except Exception:
                        pass
                skipped_llm_time += reused_llm_ms / 1000.0
                edges_to_replay.append(idx)
                continue
            edges_to_process.append(idx)

        # Replay cached transforms when work_duckdb is fresh
        # (existing work_duckdb already contains previous transforms)
        if duckdb_freshly_copied:
            for idx in edges_to_replay:
                if all_results[idx] is None:
                    continue
                has_ops = any(
                    r.get("operations_applied")
                    for r in all_results[idx].get("results", [])
                    if "error" not in r
                )
                if not has_ops:
                    continue
                replay_db_path = os.path.join(args.output_dir, f"_replay_{idx}.duckdb")
                shutil.copy2(work_duckdb, replay_db_path)
                try:
                    agent.replay_operations(replay_db_path, all_results[idx])
                    merge_tables_from_edge_db(
                        idx, target_edges[idx], replay_db_path, work_duckdb
                    )
                    logger.info(f"[Edge {idx}] Replayed cached transforms")
                except Exception as e:
                    logger.error(f"[Edge {idx}] Replay failed: {e}", exc_info=True)
                finally:
                    safe_remove(replay_db_path)
                    safe_remove(replay_db_path + ".wal")

        if not edges_to_process:
            logger.debug(f"Color {color}: all edges already processed, skipping")
            continue

        # Create per-edge DuckDB copies
        edge_db_paths = {}
        for idx in edges_to_process:
            edge_db_path = os.path.join(args.output_dir, f"_edge_{idx}.duckdb")
            shutil.copy2(work_duckdb, edge_db_path)
            edge_db_paths[idx] = edge_db_path

        # Determine effective worker count
        effective_workers = min(len(edges_to_process), args.max_workers)
        logger.debug(f"Launching {len(edges_to_process)} edges with {effective_workers} workers")

        # Process in parallel
        color_start = time.time()
        color_serial_sum = 0.0
        color_llm_sum = 0.0
        color_llm_times = []  # per-edge LLM times for parallel max calculation

        try:
            with ThreadPoolExecutor(max_workers=effective_workers) as pool:
                futures = {}
                for idx in edges_to_process:
                    future = pool.submit(
                        process_single_edge,
                        idx,
                        target_edges[idx],
                        edge_db_paths[idx],
                        agent,
                        args,
                        args.output_dir,
                    )
                    futures[future] = idx

                for future in as_completed(futures):
                    idx = futures[future]
                    try:
                        edge_output, elapsed, llm_elapsed = future.result()
                        all_results[idx] = edge_output
                        color_serial_sum += elapsed
                        total_serial_time += elapsed
                        color_llm_sum += llm_elapsed
                        color_llm_times.append(llm_elapsed)
                        llm_serial_time += llm_elapsed
                    except Exception as e:
                        logger.error(f"[Edge {idx}] Worker failed: {e}", exc_info=True)
                        error_count += 1
                        all_results[idx] = {
                            "source": target_edges[idx]["source"],
                            "target": target_edges[idx]["target"],
                            "error": str(e)
                        }
        finally:
            color_elapsed = time.time() - color_start
            total_parallel_time += color_elapsed
            color_llm_parallel = max(color_llm_times) if color_llm_times else 0.0
            llm_parallel_time += color_llm_parallel

            per_color_timing[str(color)] = {
                "wall_clock_seconds": round(color_elapsed, 2),
                "serial_sum_seconds": round(color_serial_sum, 2),
                "llm_serial_sum_seconds": round(color_llm_sum, 2),
                "llm_parallel_seconds": round(color_llm_parallel, 2),
                "edge_count": len(edges_to_process),
                "workers": effective_workers,
            }

            succeeded = sum(
                1 for i in edges_to_process
                if all_results[i] and "error" not in all_results[i]
            )
            logger.info(f"Color {color} done: {color_elapsed:.1f}s wall, "
                        f"{succeeded}/{len(edges_to_process)} ok")

            # Merge modified tables back to main DuckDB
            for idx in edges_to_process:
                if all_results[idx] and "error" not in all_results[idx]:
                    try:
                        merge_tables_from_edge_db(
                            idx, target_edges[idx],
                            edge_db_paths[idx], work_duckdb
                        )
                    except Exception as e:
                        logger.error(f"[Edge {idx}] Merge failed: {e}", exc_info=True)

            # Clean up edge-specific DuckDB files
            for idx in edges_to_process:
                edge_db_path = edge_db_paths[idx]
                safe_remove(edge_db_path)
                safe_remove(edge_db_path + ".wal")

    # ── Save Summary ───────────────────────────────────────────────────── #
    # Include reused LLM time from skipped edges in the totals
    llm_serial_time += skipped_llm_time
    llm_parallel_time += skipped_llm_time

    speedup = (total_serial_time / total_parallel_time) if total_parallel_time > 0 else 0
    llm_speedup = (llm_serial_time / llm_parallel_time) if llm_parallel_time > 0 else 0

    summary = {
        "total_edges": len(target_edges),
        "processed_edges": sum(1 for r in all_results if r is not None),
        "skipped_edges": skipped_count,
        "error_edges": error_count,
        "num_colors": num_colors,
        "timing": {
            "total_serial_time_seconds": round(total_serial_time, 2),
            "total_parallel_time_seconds": round(total_parallel_time, 2),
            "speedup_ratio": round(speedup, 2),
            "llm_serial_time_seconds": round(llm_serial_time, 2),
            "llm_parallel_time_seconds": round(llm_parallel_time, 2),
            "llm_speedup_ratio": round(llm_speedup, 2),
            "reused_llm_time_seconds": round(skipped_llm_time, 2),
            "per_color_group": per_color_timing,
        },
        "color_groups": {
            str(c): indices for c, indices in sorted(color_groups.items())
        },
        "results": [r for r in all_results if r is not None]
    }

    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info(f"\nDone: {summary['processed_edges']}/{len(target_edges)} edges, "
                f"{skipped_count} skipped, {error_count} errors")
    logger.info(f"Total: serial={total_serial_time:.1f}s, parallel={total_parallel_time:.1f}s, "
                f"speedup={speedup:.2f}x")
    logger.info(f"LLM:   serial={llm_serial_time:.1f}s, parallel={llm_parallel_time:.1f}s, "
                f"speedup={llm_speedup:.2f}x")
    logger.info(f"Summary: {summary_path}")


if __name__ == '__main__':
    args = config()
    run(args)
