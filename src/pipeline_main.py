"""
EcoTable Pipeline - Main Integration Script
Integrates Schema Linking, LLM Filter, Table Conversion, DeepJoin, Join Evaluation, Transform Agent
"""

import os
import json
import sys
import torch
import asyncio
from typing import List, Dict, Tuple, Optional
from pathlib import Path

# Import components from individual scripts
from .utils.schema_linking_DL_crossencoder import (
    CrossEncoderSchemaLinker,
    CrossEncoderTrainer,
    GlobalTablePool,
    load_data_with_global_pool
)
from .utils.LLM_Filter_nyc import LLMTableFilter, load_all_table_columns
from .utils.DeepJoin import DeepJoinClassify, load_tables
from .utils.join_evaluation import run_join_evaluation
from .utils.table_converter_interface import TableNameConverter
from .utils.transform_agent_interface import run_transform_agent, prepare_duckdb_from_tables


# ========================================
# Step 1: Schema Linking Interface
# ========================================

def schema_linking_inference(
    queries: List[Dict],
    test_table_pool: GlobalTablePool,
    model_path: str,
    device: str = 'cuda',
    top_k: int = 20,
    batch_size: int = 32
) -> List[Dict]:
    """
    Schema Linking inference interface - Use CrossEncoder model to select candidate tables

    Args:
        queries: Query list, each element contains {'query': str, 'project_name': str, ...}
        test_table_pool: Global table pool object
        model_path: Trained model path
        device: Device ('cuda' or 'cpu')
        top_k: Number of candidate tables to return per query
        batch_size: Batch size

    Returns:
        Result list, each element contains {'sample_id', 'query', 'table_scores', 'predicted_tables', ...}
    """
    print(f"\n[Schema Linking] Loading model from {model_path}...")

    # Load model
    model = CrossEncoderSchemaLinker(model_name='microsoft/deberta-v3-large', dropout=0.1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    # Create temporary trainer for inference
    trainer = CrossEncoderTrainer(
        model=model,
        train_dataloader=None,
        val_dataloader=None,
        val_data=None,
        train_table_pool=test_table_pool,
        test_data=queries,
        test_table_pool=test_table_pool,
        device=device
    )

    print(f"[Schema Linking] Running inference on {len(queries)} queries...")
    results, metrics = trainer.evaluate_on_full_table_pool(
        queries,
        top_k=top_k,
        max_samples=None,
        use_global_pool=True,
        save_detailed=True
    )

    print(f"[Schema Linking] Completed. Average F1: {metrics['f1']:.4f}")

    return results


# ========================================
# Step 2: LLM Filter Interface
# ========================================

async def llm_filter_inference(
    schema_linking_results: List[Dict],
    column_info_dict: Dict[str, List[str]],
    api_key: str,
    model: str = "gpt-4o",
    base_url: Optional[str] = None,
    top_n_candidates: int = 30
) -> List[Dict]:
    """
    LLM Filter inference interface - Semantic filtering of candidate tables

    Args:
        schema_linking_results: Schema Linking output results
        column_info_dict: Mapping from table name to column name list
        api_key: LLM API key
        model: LLM model name
        base_url: API base URL (optional)
        top_n_candidates: Number of candidate tables to pass to LLM

    Returns:
        Filtered result list, each element contains {'sample_id', 'query', 'selected_tables', ...}
    """
    print(f"\n[LLM Filter] Initializing with model {model}...")

    filter_obj = LLMTableFilter(
        api_key=api_key,
        model=model,
        base_url=base_url
    )

    filtered_results = []

    print(f"[LLM Filter] Processing {len(schema_linking_results)} queries...")

    for idx, item in enumerate(schema_linking_results, 1):
        query = item['query']
        table_scores = item.get('table_scores', [])

        # Build candidate table list (table_name, score)
        candidates = [(ts[0], ts[1]) for ts in table_scores[:top_n_candidates]]

        # Build column info
        column_info = {}
        for table_name, _ in candidates:
            if table_name in column_info_dict:
                column_info[table_name] = {col: '' for col in column_info_dict[table_name]}

        # Call LLM filter
        filter_result = await filter_obj.filter_candidates(
            query=query,
            candidates=candidates,
            column_info=column_info
        )

        filtered_results.append({
            'sample_id': item.get('sample_id', idx - 1),
            'query': query,
            'candidates': candidates,
            'selected_tables': filter_result['filtered_tables'],
            'temporal_scope': filter_result.get('temporal_scope'),
            'llm_interaction': filter_result.get('llm_interaction')
        })

        if idx % 10 == 0:
            print(f"  Processed {idx}/{len(schema_linking_results)} queries")

        # Avoid API rate limiting
        await asyncio.sleep(1.0)

    print(f"[LLM Filter] Completed.")

    return filtered_results


# ========================================
# Step 3: Table Name Conversion
# ========================================
# Now using table_converter_interface.py (wraps convert_selected_tables.py)


# ========================================
# Main Pipeline Function
# ========================================

async def run_ecotable_pipeline(
    # Input data
    queries: List[Dict],  # Query list

    # Schema Linking parameters
    schema_linking_model_path: str,
    test_table_pool: GlobalTablePool,
    schema_top_k: int = 30,

    # LLM Filter parameters
    llm_api_key: str = "",
    llm_model: str = "gpt-4o",
    llm_base_url: Optional[str] = None,
    column_stats_file: str = None,

    # DeepJoin parameters
    deepjoin_model_dir: str = "",
    deepjoin_output: str = "deepjoin_results.json",
    deepjoin_top_k: int = 6,

    # Join Evaluation parameters
    ground_truth_csv: str = "",
    tables_dir: str = "",
    evaluation_output: str = "evaluation_results.json",
    eval_api_key: Optional[str] = None,
    eval_model: str = "gpt-4o",

    # Transform Agent parameters (Step 6)
    enable_transform_agent: bool = False,
    transform_max_iterations: int = 10,
    transform_max_workers: int = 8,
    transform_test_mode: bool = False,

    # Intermediate result save paths
    output_dir: str = "pipeline_outputs",

    # Run mode
    device: str = 'cuda'
) -> Dict:
    """
    EcoTable complete pipeline (up to 6 steps)

    Args:
        queries: Query list, each contains {'query': str, 'project_name': str, ...}
        schema_linking_model_path: Schema Linking model path
        test_table_pool: Test set table pool
        schema_top_k: Number of candidate tables returned by Schema Linking
        llm_api_key: LLM API key
        llm_model: LLM model name
        llm_base_url: LLM API base URL
        column_stats_file: Table column info file (JSON format)
        deepjoin_model_dir: DeepJoin model directory
        deepjoin_output: DeepJoin result output file
        deepjoin_top_k: DeepJoin top-k joins to keep per table pair
        ground_truth_csv: Ground truth CSV file
        tables_dir: Table data directory
        evaluation_output: Evaluation result output file
        eval_api_key: LLM API key for evaluation phase
        eval_model: LLM model for evaluation phase
        enable_transform_agent: Whether to run transform agent (Step 6)
        transform_max_iterations: Max iterations for transform agent
        transform_max_workers: Max parallel workers for transform agent
        transform_test_mode: Test mode for transform agent (no LLM calls)
        output_dir: Intermediate result save directory
        device: Device to run on

    Returns:
        Dictionary containing results from each phase
    """
    os.makedirs(output_dir, exist_ok=True)

    total_steps = 6 if enable_transform_agent else 5

    # ========================================
    # Step 1: Schema Linking
    # ========================================
    print("\n" + "="*80)
    print(f"Step 1/{total_steps}: Schema Linking with CrossEncoder")
    print("="*80)

    schema_results = schema_linking_inference(
        queries=queries,
        test_table_pool=test_table_pool,
        model_path=schema_linking_model_path,
        device=device,
        top_k=schema_top_k
    )

    # Save intermediate results
    schema_output = os.path.join(output_dir, "step1_schema_linking_results.json")
    with open(schema_output, 'w', encoding='utf-8') as f:
        json.dump(schema_results, f, ensure_ascii=False, indent=2)
    print(f"[Step 1] Results saved to: {schema_output}")

    # ========================================
    # Step 2: LLM Filter
    # ========================================
    print("\n" + "="*80)
    print(f"Step 2/{total_steps}: LLM-based Table Filtering")
    print("="*80)

    # Load column info
    if column_stats_file:
        column_info_dict = load_all_table_columns(column_stats_file)
    else:
        column_info_dict = {}
        print("  Warning: No column_stats_file provided, using empty column info")

    llm_filter_results = await llm_filter_inference(
        schema_linking_results=schema_results,
        column_info_dict=column_info_dict,
        api_key=llm_api_key,
        model=llm_model,
        base_url=llm_base_url,
        top_n_candidates=30
    )

    # Save intermediate results
    llm_filter_output = os.path.join(output_dir, "step2_llm_filter_results.json")
    with open(llm_filter_output, 'w', encoding='utf-8') as f:
        json.dump({
            'metadata': {'model': llm_model},
            'detailed_results': llm_filter_results
        }, f, ensure_ascii=False, indent=2)
    print(f"[Step 2] Results saved to: {llm_filter_output}")

    # ========================================
    # Step 3: Convert Table Names to Paths
    # ========================================
    print("\n" + "="*80)
    print(f"Step 3/{total_steps}: Converting Table Names to Paths")
    print("="*80)

    # Use TableNameConverter from table_converter_interface.py
    # (which wraps the original convert_selected_tables.py logic)
    converter = TableNameConverter(tables_dir)
    llm_filter_results = converter.convert_llm_filter_results(
        llm_filter_results,
        inplace=True
    )

    # Update saved results
    with open(llm_filter_output, 'w', encoding='utf-8') as f:
        json.dump({
            'metadata': {'model': llm_model},
            'detailed_results': llm_filter_results
        }, f, ensure_ascii=False, indent=2)
    print(f"[Step 3] Updated results saved to: {llm_filter_output}")

    # ========================================
    # Step 4: DeepJoin - Predict Join Relationships
    # ========================================
    print("\n" + "="*80)
    print(f"Step 4/{total_steps}: DeepJoin - Predicting Join Relationships")
    print("="*80)

    # Collect all selected tables (deduplicate)
    all_selected_tables = set()
    for item in llm_filter_results:
        all_selected_tables.update(item['selected_tables'])

    print(f"[DeepJoin] Total unique tables selected: {len(all_selected_tables)}")

    # Call DeepJoin
    deepjoin_output_path = os.path.join(output_dir, deepjoin_output)

    print(f"[DeepJoin] Running prediction...")
    print(f"  Model dir: {deepjoin_model_dir}")
    print(f"  Tables dir: {tables_dir}")
    print(f"  Output: {deepjoin_output_path}")

    DeepJoinClassify(
        mode='predict',
        tables=tables_dir,
        model_dir=deepjoin_model_dir,
        links=None,
        output=deepjoin_output_path,
        top_k=deepjoin_top_k,
        isfolder=True
    )

    print(f"[Step 4] DeepJoin results saved to: {deepjoin_output_path}")

    # ========================================
    # Step 5: Join Evaluation
    # ========================================
    print("\n" + "="*80)
    print(f"Step 5/{total_steps}: Join Relationship Evaluation")
    print("="*80)

    evaluation_output_path = os.path.join(output_dir, evaluation_output)

    eval_results = run_join_evaluation(
        json_file=deepjoin_output_path,
        csv_file=ground_truth_csv,
        queries_file=llm_filter_output,
        tables_dir=tables_dir,
        output_file=evaluation_output_path,
        api_key=eval_api_key or llm_api_key,
        model=eval_model,
        sample_rows=5,
        top_k_paths=10,
        max_path_length=100,
        t_spanner=1.1,
        edge_verification_cache=os.path.join(output_dir, 'edge_verification_cache'),
        graph_cache=os.path.join(output_dir, 'sparse_graph_cache.json'),
        verbose=False,
        test_mode=False,
        overwrite=False
    )

    print(f"[Step 5] Evaluation results saved to: {evaluation_output_path}")

    # ========================================
    # Step 6: Transform Agent (Optional)
    # ========================================
    transform_summary = None
    if enable_transform_agent:
        print("\n" + "="*80)
        print("Step 6/6: Transform Agent - Data Transformation")
        print("="*80)

        # Get pruned graph with join columns from evaluation output
        pruned_graph_path = os.path.join(
            output_dir,
            'edge_verification_cache',
            'pruned_graph_join_cols_version.json'
        )

        if not os.path.exists(pruned_graph_path):
            print(f"  Warning: Pruned graph not found at {pruned_graph_path}")
            print(f"  Skipping transform agent step")
        else:
            # Prepare DuckDB from all selected tables
            print(f"[Transform Agent] Preparing DuckDB...")
            all_selected_tables_for_db = set()
            for item in llm_filter_results:
                all_selected_tables_for_db.update(item['selected_tables'])

            duckdb_path = os.path.join(output_dir, 'tables.duckdb')
            prepare_duckdb_from_tables(
                tables_dir=tables_dir,
                selected_tables=list(all_selected_tables_for_db),
                output_duckdb=duckdb_path,
                verbose=True
            )

            # Run transform agent
            transform_output_dir = os.path.join(output_dir, 'transform_agent_output')
            transform_summary = run_transform_agent(
                graph_json=pruned_graph_path,
                duckdb_path=duckdb_path,
                output_dir=transform_output_dir,
                model=eval_model,
                max_iterations=transform_max_iterations,
                max_workers=transform_max_workers,
                test_mode=transform_test_mode,
                eval_file=evaluation_output_path,
                verbose=True
            )

            print(f"[Step 6] Transform agent results saved to: {transform_output_dir}")
    else:
        print("\n[Step 6] Transform Agent skipped (enable_transform_agent=False)")

    # ========================================
    # Pipeline Complete
    # ========================================
    print("\n" + "="*80)
    print("Pipeline Complete!")
    print("="*80)

    result = {
        'schema_linking_results': schema_results,
        'llm_filter_results': llm_filter_results,
        'deepjoin_output': deepjoin_output_path,
        'evaluation_results': eval_results,
        'output_dir': output_dir
    }

    if transform_summary:
        result['transform_agent_summary'] = transform_summary

    return result


# ========================================
# Command Line Interface
# ========================================

def main():
    """Command line interface"""
    import argparse

    parser = argparse.ArgumentParser(
        description='EcoTable Pipeline - End-to-End Table Discovery, Join Evaluation, and Transformation'
    )

    # Required parameters
    parser.add_argument('--queries_file', required=True, help='Query file (JSON format)')
    parser.add_argument('--schema_model', required=True, help='Schema Linking model path')
    parser.add_argument('--test_data_file', required=True, help='Test data file (for building table pool)')
    parser.add_argument('--csv_dataset_root', required=True, help='CSV dataset root directory')
    parser.add_argument('--deepjoin_model_dir', required=True, help='DeepJoin model directory')
    parser.add_argument('--ground_truth_csv', required=True, help='Ground truth CSV file')
    parser.add_argument('--tables_dir', required=True, help='Table data directory')

    # LLM parameters
    parser.add_argument('--llm_api_key', required=True, help='LLM API key')
    parser.add_argument('--llm_model', default='gpt-4o', help='LLM model name')
    parser.add_argument('--llm_base_url', help='LLM API base URL')

    # Optional parameters
    parser.add_argument('--column_stats_file', help='Table column info file')
    parser.add_argument('--output_dir', default='pipeline_outputs', help='Output directory')
    parser.add_argument('--schema_top_k', type=int, default=30, help='Schema Linking top-k')
    parser.add_argument('--deepjoin_top_k', type=int, default=6, help='DeepJoin top-k')
    parser.add_argument('--device', default='cuda', help='Device (cuda/cpu)')

    # Transform Agent (Step 6)
    parser.add_argument('--enable_transform_agent', action='store_true',
                       help='Enable transform agent (Step 6)')
    parser.add_argument('--transform_max_iterations', type=int, default=10,
                       help='Transform agent max iterations per edge')
    parser.add_argument('--transform_max_workers', type=int, default=8,
                       help='Transform agent max parallel workers')
    parser.add_argument('--transform_test_mode', action='store_true',
                       help='Transform agent test mode (no LLM calls)')

    args = parser.parse_args()

    # Load query data
    with open(args.queries_file, 'r', encoding='utf-8') as f:
        queries_data = json.load(f)

    # Build test table pool
    print("Building test table pool...")
    test_table_pool = GlobalTablePool()
    test_table_pool.build_from_csv_folders(args.csv_dataset_root)

    # Prepare query format
    queries = []
    if isinstance(queries_data, dict):
        for project_name, project_queries in queries_data.items():
            for item in project_queries:
                item['project_name'] = project_name
                queries.append(item)
    elif isinstance(queries_data, list):
        queries = queries_data
    else:
        raise ValueError("Unsupported queries file format")

    print(f"Loaded {len(queries)} queries")

    # Run pipeline
    results = asyncio.run(run_ecotable_pipeline(
        queries=queries,
        schema_linking_model_path=args.schema_model,
        test_table_pool=test_table_pool,
        schema_top_k=args.schema_top_k,
        llm_api_key=args.llm_api_key,
        llm_model=args.llm_model,
        llm_base_url=args.llm_base_url,
        column_stats_file=args.column_stats_file,
        deepjoin_model_dir=args.deepjoin_model_dir,
        deepjoin_top_k=args.deepjoin_top_k,
        ground_truth_csv=args.ground_truth_csv,
        tables_dir=args.tables_dir,
        evaluation_output='evaluation_results.json',
        enable_transform_agent=args.enable_transform_agent,
        transform_max_iterations=args.transform_max_iterations,
        transform_max_workers=args.transform_max_workers,
        transform_test_mode=args.transform_test_mode,
        output_dir=args.output_dir,
        device=args.device
    ))

    print("\nAll results saved to:", args.output_dir)

    if args.enable_transform_agent and 'transform_agent_summary' in results:
        summary = results['transform_agent_summary']
        print(f"\nTransform Agent Summary:")
        print(f"  Processed: {summary.get('processed_edges', 0)} edges")
        print(f"  Skipped: {summary.get('skipped_edges', 0)} edges")
        print(f"  Errors: {summary.get('error_edges', 0)} edges")


if __name__ == "__main__":
    main()
