"""
LLM Filter Interface
Provides async function interface for LLM_Filter_nyc.py
"""

import asyncio
from typing import List, Dict, Optional, Tuple
from .LLM_Filter_nyc import LLMTableFilter, load_all_table_columns, make_serializable


async def llm_filter_candidates(
    query: str,
    candidates: List[Tuple[str, float]],
    column_info: Dict[str, Dict[str, str]],
    api_key: str,
    model: str = "gpt-4o",
    base_url: Optional[str] = None
) -> Dict:
    """
    Filter candidate tables using LLM (single query)

    Args:
        query: Natural language query
        candidates: Candidate table list, format: [(table_name, score), ...]
        column_info: Column info, format: {table_name: {column_name: '', ...}, ...}
        api_key: LLM API key
        model: LLM model name
        base_url: API base URL (optional)

    Returns:
        {
            'filtered_tables': List[str],  # Filtered table list
            'temporal_scope': str,  # Temporal scope
            'summary': Dict,  # Summary info
            'llm_interaction': Dict  # LLM interaction details
        }
    """
    filter_obj = LLMTableFilter(
        api_key=api_key,
        model=model,
        base_url=base_url
    )

    result = await filter_obj.filter_candidates(
        query=query,
        candidates=candidates,
        column_info=column_info
    )

    return result


async def llm_filter_batch(
    queries: List[str],
    candidates_list: List[List[Tuple[str, float]]],
    column_info_list: List[Dict[str, Dict[str, str]]],
    api_key: str,
    model: str = "gpt-4o",
    base_url: Optional[str] = None,
    sleep_interval: float = 1.0,
    verbose: bool = True
) -> List[Dict]:
    """
    Batch process multiple queries with LLM filter

    Args:
        queries: Query list
        candidates_list: Candidate table list for each query
        column_info_list: Column info for each query
        api_key: LLM API key
        model: LLM model name
        base_url: API base URL (optional)
        sleep_interval: Request interval (seconds)
        verbose: Whether to print detailed info

    Returns:
        Filter result list
    """
    if verbose:
        print(f"[LLM Filter] Processing {len(queries)} queries with model {model}")

    results = []
    filter_obj = LLMTableFilter(
        api_key=api_key,
        model=model,
        base_url=base_url
    )

    for idx, (query, candidates, column_info) in enumerate(zip(queries, candidates_list, column_info_list), 1):
        if verbose and idx % 10 == 0:
            print(f"  Processed {idx}/{len(queries)} queries")

        result = await filter_obj.filter_candidates(
            query=query,
            candidates=candidates,
            column_info=column_info
        )

        results.append(result)

        # Avoid API rate limiting
        if idx < len(queries):
            await asyncio.sleep(sleep_interval)

    if verbose:
        print(f"[LLM Filter] Completed {len(results)} queries")

    return results


def prepare_column_info(
    table_names: List[str],
    column_stats_file: str
) -> Dict[str, Dict[str, str]]:
    """
    Prepare column info from stats file

    Args:
        table_names: Table name list
        column_stats_file: Column stats file path

    Returns:
        Column info dict {table_name: {column_name: '', ...}, ...}
    """
    all_columns = load_all_table_columns(column_stats_file)

    column_info = {}
    for table_name in table_names:
        if table_name in all_columns:
            column_info[table_name] = {col: '' for col in all_columns[table_name]}

    return column_info


async def run_llm_filter_on_schema_results(
    schema_results: List[Dict],
    column_stats_file: str,
    api_key: str,
    model: str = "gpt-4o",
    base_url: Optional[str] = None,
    top_n_candidates: int = 30,
    verbose: bool = True
) -> List[Dict]:
    """
    Run LLM filter on Schema Linking results

    Args:
        schema_results: Schema Linking output results
        column_stats_file: Column stats file path
        api_key: LLM API key
        model: LLM model name
        base_url: API base URL (optional)
        top_n_candidates: Number of candidate tables to pass to LLM
        verbose: Whether to print detailed info

    Returns:
        Filtered result list, each contains:
        {
            'sample_id': int,
            'query': str,
            'candidates': List[Tuple[str, float]],
            'selected_tables': List[str],
            'temporal_scope': str,
            'llm_interaction': Dict
        }
    """
    if verbose:
        print(f"[LLM Filter] Loading column information from {column_stats_file}")

    all_columns = load_all_table_columns(column_stats_file)

    if verbose:
        print(f"[LLM Filter] Loaded column info for {len(all_columns)} tables")

    filter_obj = LLMTableFilter(
        api_key=api_key,
        model=model,
        base_url=base_url
    )

    filtered_results = []

    for idx, item in enumerate(schema_results, 1):
        if verbose and idx % 10 == 0:
            print(f"  Processing {idx}/{len(schema_results)} queries")

        query = item['query']
        table_scores = item.get('table_scores', [])

        # Build candidate table list
        candidates = [(ts[0], ts[1]) for ts in table_scores[:top_n_candidates]]

        # Build column info
        column_info = {}
        for table_name, _ in candidates:
            if table_name in all_columns:
                column_info[table_name] = {col: '' for col in all_columns[table_name]}

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

        # Avoid API rate limiting
        await asyncio.sleep(1.0)

    if verbose:
        print(f"[LLM Filter] Completed processing {len(filtered_results)} queries")

    return filtered_results


# Sync wrapper (if need to call in non-async environment)
def run_llm_filter_sync(
    schema_results: List[Dict],
    column_stats_file: str,
    api_key: str,
    model: str = "gpt-4o",
    base_url: Optional[str] = None,
    top_n_candidates: int = 30,
    verbose: bool = True
) -> List[Dict]:
    """
    Synchronous version of LLM filter interface

    Args:
        Same as run_llm_filter_on_schema_results

    Returns:
        Same as run_llm_filter_on_schema_results
    """
    return asyncio.run(run_llm_filter_on_schema_results(
        schema_results=schema_results,
        column_stats_file=column_stats_file,
        api_key=api_key,
        model=model,
        base_url=base_url,
        top_n_candidates=top_n_candidates,
        verbose=verbose
    ))
