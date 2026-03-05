"""
Schema Linking Interface
Provides function interface for schema_linking_DL_crossencoder.py
"""

import torch
from typing import List, Dict, Tuple
from .schema_linking_DL_crossencoder import (
    CrossEncoderSchemaLinker,
    CrossEncoderTrainer,
    GlobalTablePool
)


def load_schema_linking_model(model_path: str, device: str = 'cuda') -> Tuple[CrossEncoderSchemaLinker, str]:
    """
    Load trained Schema Linking model

    Args:
        model_path: Model weight file path
        device: Device ('cuda' or 'cpu')

    Returns:
        (model object, device)
    """
    model = CrossEncoderSchemaLinker(
        model_name='microsoft/deberta-v3-large',
        dropout=0.1
    )

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    return model, device


def schema_linking_predict(
    model: CrossEncoderSchemaLinker,
    queries: List[Dict],
    table_pool: GlobalTablePool,
    device: str = 'cuda',
    top_k: int = 20,
    batch_size: int = 32,
    verbose: bool = True
) -> Tuple[List[Dict], Dict]:
    """
    Run inference with Schema Linking model

    Args:
        model: Loaded CrossEncoderSchemaLinker model
        queries: Query list, each contains {'query': str, 'project_name': str, ...}
        table_pool: GlobalTablePool object
        device: Device
        top_k: Number of candidate tables to return per query
        batch_size: Batch size
        verbose: Whether to print detailed info

    Returns:
        (results, metrics)
        - results: Prediction result list
        - metrics: Overall metrics {'precision': float, 'recall': float, 'f1': float}
    """
    # Create temporary trainer for inference
    trainer = CrossEncoderTrainer(
        model=model,
        train_dataloader=None,
        val_dataloader=None,
        val_data=None,
        train_table_pool=table_pool,
        test_data=queries,
        test_table_pool=table_pool,
        device=device
    )

    if verbose:
        print(f"[Schema Linking] Running inference on {len(queries)} queries...")
        print(f"  Top-K: {top_k}")
        print(f"  Device: {device}")

    results, metrics = trainer.evaluate_on_full_table_pool(
        queries,
        top_k=top_k,
        max_samples=None,
        use_global_pool=True,
        save_detailed=True
    )

    if verbose:
        print(f"[Schema Linking] Completed.")
        print(f"  Average Precision: {metrics['precision']:.4f}")
        print(f"  Average Recall: {metrics['recall']:.4f}")
        print(f"  Average F1: {metrics['f1']:.4f}")

    return results, metrics


def build_table_pool_from_csv(csv_dataset_root: str, verbose: bool = True) -> GlobalTablePool:
    """
    Build global table pool from CSV folder

    Args:
        csv_dataset_root: CSV dataset root directory
        verbose: Whether to print detailed info

    Returns:
        GlobalTablePool object
    """
    if verbose:
        print(f"[Table Pool] Building from: {csv_dataset_root}")

    table_pool = GlobalTablePool()
    table_pool.build_from_csv_folders(csv_dataset_root)

    if verbose:
        all_tables = table_pool.get_all_tables()
        print(f"[Table Pool] Loaded {len(all_tables)} tables")

    return table_pool


def build_table_pool_from_json(json_file: str, add_project_prefix: bool = True, verbose: bool = True) -> GlobalTablePool:
    """
    Build global table pool from JSON file

    Args:
        json_file: JSON file path
        add_project_prefix: Whether to add project prefix for node tables
        verbose: Whether to print detailed info

    Returns:
        GlobalTablePool object
    """
    if verbose:
        print(f"[Table Pool] Building from: {json_file}")

    table_pool = GlobalTablePool()
    table_pool.build_from_json(json_file, add_project_prefix_to_nodes=add_project_prefix)

    if verbose:
        all_tables = table_pool.get_all_tables()
        print(f"[Table Pool] Loaded {len(all_tables)} tables")

    return table_pool


# Complete workflow interface
def run_schema_linking(
    queries: List[Dict],
    model_path: str,
    table_pool: GlobalTablePool,
    device: str = 'cuda',
    top_k: int = 20,
    verbose: bool = True
) -> Tuple[List[Dict], Dict]:
    """
    Run complete Schema Linking workflow

    Args:
        queries: Query list
        model_path: Model path
        table_pool: Global table pool
        device: Device
        top_k: Number of candidate tables to return
        verbose: Whether to print detailed info

    Returns:
        (results, metrics)
    """
    # Load model
    model, device = load_schema_linking_model(model_path, device)

    # Run inference
    results, metrics = schema_linking_predict(
        model=model,
        queries=queries,
        table_pool=table_pool,
        device=device,
        top_k=top_k,
        verbose=verbose
    )

    return results, metrics
