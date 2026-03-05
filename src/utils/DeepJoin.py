import os
import pandas as pd
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses
from typing import List, Tuple, Dict, Optional
import time
import logging
import argparse
import json
import random
from pathlib import Path
from collections import Counter
from tqdm import tqdm
import heapq

# ------------------------------
# Logging Configuration
# ------------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
torch.use_deterministic_algorithms(True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger('DeepJoin')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# ------------------------------
# Data Structure Definitions
# ------------------------------

class Table:
    def __init__(self, table_name: str, columns: Dict[str, List], unique_id: str = None):
        self.table_name = table_name
        self.unique_id = unique_id if unique_id else table_name
        self.columns = columns
        self.col_names = list(columns.keys())

    def __repr__(self):
        row_count = len(next(iter(self.columns.values()))) if self.columns else 0
        return f"Table({self.unique_id}, {len(self.columns)} columns, {row_count} rows)"

class BIModel:
    def __init__(self, tables: List[Table], joins: List[Tuple[Tuple[Table, str], Tuple[Table, str]]]):
        self.tables = tables
        self.joins = joins
    def __repr__(self):
        return f"BIModel({len(self.tables)} tables, {len(self.joins)} joins)"

# ------------------------------
# Data Loading
# ------------------------------

def normalize_table_name(name: str) -> str:
    return name.replace('\\', '/')

def parse_table_identifier(name: str) -> Tuple[str, str]:
    name = normalize_table_name(name)
    if '/' in name:
        parts = name.rsplit('/', 1)
        return (parts[0], parts[1])
    return (None, name)

def load_tables_from_dir(table_dir: str) -> Dict[str, Table]:
    tables = {}
    table_path = Path(table_dir)
    if not table_path.exists():
        logger.error(f"Table directory does not exist: {table_path}")
        return tables
    csv_files = list(table_path.glob("*.csv"))
    logger.info(f"Found {len(csv_files)} CSV files")
    for filepath in csv_files:
        try:
            df = pd.read_csv(filepath)
            table_name = filepath.stem
            columns = {col.strip(): df[col].tolist() for col in df.columns}
            tables[table_name] = Table(table_name, columns, unique_id=table_name)
        except Exception as e:
            logger.error(f"Failed to load table {filepath.name}: {e}")
    logger.info(f"Loading complete: {len(tables)} tables")
    return tables

def load_tables_from_folder_structure(table_dir: str) -> Dict[str, Table]:
    tables = {}
    table_path = Path(table_dir)
    if not table_path.exists():
        logger.error(f"Table directory does not exist: {table_path}")
        return tables

    subfolders = [f for f in table_path.iterdir() if f.is_dir()]
    logger.info(f"Found {len(subfolders)} subfolders")

    total_csv_count = 0
    for subfolder in subfolders:
        folder_name = subfolder.name
        csv_files = list(subfolder.glob("*.csv"))
        for filepath in csv_files:
            try:
                df = pd.read_csv(filepath)
                csv_name = filepath.stem
                unique_id = f"{folder_name}/{csv_name}"
                columns = {col: df[col].tolist() for col in df.columns}
                tables[unique_id] = Table(table_name=csv_name, columns=columns, unique_id=unique_id)
                total_csv_count += 1
            except Exception as e:
                logger.error(f"Failed to load table {filepath}: {e}")
    logger.info(f"Loading complete: total {total_csv_count} tables")
    return tables

def load_tables(table_dir: str, isfolder: bool = False) -> Dict[str, Table]:
    if isfolder:
        return load_tables_from_folder_structure(table_dir)
    else:
        return load_tables_from_dir(table_dir)

def find_table_by_name(tables: Dict[str, Table], name: str) -> Optional[Table]:
    name = normalize_table_name(name)
    if name in tables: return tables[name]
    if name.endswith('.csv'):
        name_no_ext = name[:-4]
        if name_no_ext in tables: return tables[name_no_ext]
    folder_name, csv_name = parse_table_identifier(name)
    if folder_name:
        candidate_key = f"{folder_name}/{csv_name}"
        if candidate_key in tables: return tables[candidate_key]
        if csv_name.endswith('.csv'):
            candidate_key = f"{folder_name}/{csv_name[:-4]}"
            if candidate_key in tables: return tables[candidate_key]
    return None

def load_links_csv(links_file: str, tables: Dict[str, Table]) -> BIModel:
    try:
        links_path = Path(links_file)
        if not links_path.exists():
            return BIModel([], [])

        links_df = pd.read_csv(links_path, header=None)
        tables_in_model = set()
        joins = []

        for idx, row in links_df.iterrows():
            try:
                t1n = normalize_table_name(str(row[0]).strip())
                t2n = normalize_table_name(str(row[1]).strip())
                if t1n.endswith('.csv'): t1n = t1n[:-4]
                if t2n.endswith('.csv'): t2n = t2n[:-4]

                c1n = str(row[2]).strip()
                c2n = str(row[3]).strip()

                t1 = find_table_by_name(tables, t1n)
                t2 = find_table_by_name(tables, t2n)

                if t1 and t2:
                    tables_in_model.add(t1.unique_id)
                    tables_in_model.add(t2.unique_id)
                    if c1n in t1.columns and c2n in t2.columns:
                        joins.append(((t1, c1n), (t2, c2n)))
            except Exception as e:
                logger.error(f"Error processing row {idx+1} in join relationship file: {e}")
                continue

        table_objs = [tables[name] for name in tables_in_model if name in tables]
        return BIModel(table_objs, joins)
    except Exception as e:
        logger.error(f"Failed to load join relationships: {e}")
        return BIModel([], [])


# ------------------------------
# DeepJoin Implementation
# ------------------------------

class DeepJoinClassifier:
    """
    DeepJoin: Joinable Table Discovery with Pre-trained Language Models
    Paper: DeepJoin: Joinable Table Discovery with Pre-trained Language Models (VLDB 2023)
    """

    def __init__(self, model_name='all-mpnet-base-v2'):
        """
        Initialize DeepJoin classifier

        Args:
            model_name: Pre-trained language model name, default is MPNet
        """
        # Try to load local model
        local_model_path = '/home/zhaohangyu/EcoTable/all-mpnet-base-v2-local'
        try:
            self.model = SentenceTransformer(local_model_path, device=device)
            logger.info(f"Loaded local model: {local_model_path}")
        except Exception:
            logger.warning("Failed to load local model, loading from HuggingFace")
            self.model = SentenceTransformer(local_model_path, device=device)

        # Set maximum sequence length
        self.max_seq_length = 384
        self.model.max_seq_length = self.max_seq_length
        logger.info(f"Model max sequence length: {self.max_seq_length}")

        self.logger = logging.getLogger('DeepJoinClassifier')
        self.column_embeddings = {}  # Store column embeddings

    def _column_to_text(self, table: Table, column_name: str, pattern='title-colname-stat-col') -> str:
        """
        Convert column to text sequence

        Args:
            table: Table object
            column_name: Column name
            pattern: Conversion pattern

        Returns:
            Text sequence
        """
        column_data = table.columns[column_name]

        # Sample frequent values
        sampled_values = self._sample_frequent_values(column_data)

        # Compute statistics
        n = len(set(column_data))
        value_lengths = [len(str(x)) for x in column_data if pd.notna(x)]
        max_len = max(value_lengths) if value_lengths else 0
        min_len = min(value_lengths) if value_lengths else 0
        avg_len = sum(value_lengths) / len(value_lengths) if value_lengths else 0

        # Generate text based on pattern
        if pattern == 'col':
            return ', '.join(sampled_values)
        elif pattern == 'colname-col':
            return f"{column_name}: {', '.join(sampled_values)}"
        elif pattern == 'colname-stat-col':
            return f"{column_name} contains {n} values ({max_len}, {min_len}, {avg_len:.1f}): {', '.join(sampled_values)}"
        elif pattern == 'title-colname-col':
            return f"{table.table_name}. {column_name}: {', '.join(sampled_values)}"
        elif pattern == 'title-colname-stat-col':
            return f"{table.unique_id}. {column_name} contains {n} values ({max_len}, {min_len}, {avg_len:.1f}): {', '.join(sampled_values)}"
        else:
            # Default to best pattern
            return f"{table.unique_id}. {column_name} contains {n} values ({max_len}, {min_len}, {avg_len:.1f}): {', '.join(sampled_values)}"

    def _sample_frequent_values(self, column: List, max_tokens=None) -> List[str]:
        """
        Sample most frequent cell values (frequency-based sampling strategy)

        Args:
            column: Column data
            max_tokens: Maximum token count

        Returns:
            List of sampled values
        """
        if max_tokens is None:
            max_tokens = self.max_seq_length - 50

        # Count frequencies
        str_values = [str(x) for x in column if pd.notna(x)]
        if not str_values:
            return []

        value_freq = Counter(str_values)
        # Sort by frequency
        distinct_values = sorted(list(value_freq.keys()), key=lambda x: value_freq[x], reverse=True)

        sampled_values = []
        current_tokens = 0
        tokenizer = self.model.tokenizer

        for value in distinct_values:
            # Limit individual value length
            if len(value) > 1000:
                value = value[:1000]

            tokens = tokenizer.tokenize(value)
            token_count = len(tokens)

            if token_count > max_tokens:
                sampled_values.append(value[:max_tokens * 4])
                break

            if sampled_values:
                token_count += 1  # Comma separator

            if current_tokens + token_count <= max_tokens:
                sampled_values.append(value)
                current_tokens += token_count
            else:
                break

        return sampled_values

    def _shuffle_column(self, column: List) -> List:
        """Randomly shuffle cell order in column (for data augmentation)"""
        shuffled = column.copy()
        random.shuffle(shuffled)
        return shuffled

    def train(self, bimodels: List[BIModel], batch_size=32, epochs=10, shuffle_rate=0.3):
        """
        Train DeepJoin model

        Args:
            bimodels: List of BI models (containing positive samples)
            batch_size: Batch size
            epochs: Number of training epochs
            shuffle_rate: Ratio of shuffled columns in data augmentation
        """
        try:
            logger.info("Preparing training data...")

            # Collect all positive samples
            positive_pairs = []
            for bm in bimodels:
                for (t1, c1), (t2, c2) in bm.joins:
                    positive_pairs.append(((t1, c1), (t2, c2)))

            logger.info(f"Number of positive samples: {len(positive_pairs)}")

            # Data augmentation: randomly shuffle some columns
            augmented_pairs = []
            num_to_shuffle = int(len(positive_pairs) * shuffle_rate)
            shuffle_indices = random.sample(range(len(positive_pairs)), num_to_shuffle)

            for idx, ((t1, c1), (t2, c2)) in enumerate(positive_pairs):
                if idx in shuffle_indices:
                    # Shuffle first column
                    shuffled_col = self._shuffle_column(t1.columns[c1])
                    # Create temporary table
                    temp_columns = t1.columns.copy()
                    temp_columns[c1] = shuffled_col
                    temp_table = Table(t1.table_name, temp_columns, t1.unique_id + "_shuffled")
                    augmented_pairs.append(((temp_table, c1), (t2, c2)))

            logger.info(f"Added samples after data augmentation: {len(augmented_pairs)}")
            all_positive_pairs = positive_pairs + augmented_pairs

            # Prepare training samples (using InputExample)
            train_examples = []
            for (t1, c1), (t2, c2) in all_positive_pairs:
                text1 = self._column_to_text(t1, c1)
                text2 = self._column_to_text(t2, c2)
                # InputExample's label is used for Multiple Negatives Ranking Loss
                train_examples.append(InputExample(texts=[text1, text2], label=1.0))

            logger.info(f"Total training samples: {len(train_examples)}")

            # Create DataLoader
            train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)

            # Use Multiple Negatives Ranking Loss
            # This loss function automatically uses in-batch negatives
            train_loss = losses.MultipleNegativesRankingLoss(self.model)

            # Fine-tune model
            logger.info("Starting pre-trained language model fine-tuning...")
            warmup_steps = int(len(train_dataloader) * epochs * 0.1)

            self.model.fit(
                train_objectives=[(train_dataloader, train_loss)],
                epochs=epochs,
                warmup_steps=warmup_steps,
                optimizer_params={'lr': 2e-5},
                show_progress_bar=True
            )

            logger.info("Training complete!")

        except Exception as e:
            logger.error(f"Error during training: {e}")
            logger.exception("Detailed error traceback:")
            raise

    def build_embeddings(self, tables: List[Table]):
        """
        Generate embeddings for all columns (no longer using FAISS index)

        Args:
            tables: List of tables
        """
        logger.info("Generating embeddings for all columns...")

        # Generate embeddings for all columns
        all_columns = []
        all_texts = []

        for table in tables:
            for col_name in table.col_names:
                all_columns.append((table, col_name))
                text = self._column_to_text(table, col_name)
                all_texts.append(text)

        logger.info(f"Total columns: {len(all_columns)}")

        # Batch encoding
        logger.info("Batch generating column embeddings...")
        embeddings = self.model.encode(
            all_texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True
        )

        # Store embeddings
        for i, (table, col_name) in enumerate(all_columns):
            key = (table.unique_id, col_name)
            self.column_embeddings[key] = embeddings[i]

        logger.info(f"Column embedding generation complete! Embedding dimension: {embeddings.shape[1]}")

    def get_column_embedding(self, table: Table, column_name: str) -> np.ndarray:
        """
        Get column embedding vector (compute on-the-fly if not available)

        Args:
            table: Table object
            column_name: Column name

        Returns:
            Column embedding vector
        """
        key = (table.unique_id, column_name)
        if key not in self.column_embeddings:
            # Compute embedding on-the-fly
            text = self._column_to_text(table, column_name)
            embedding = self.model.encode([text], convert_to_numpy=True)[0]
            self.column_embeddings[key] = embedding
        return self.column_embeddings[key]

    def compute_cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embedding vectors

        Args:
            emb1: First embedding vector
            emb2: Second embedding vector

        Returns:
            Cosine similarity (0-1)
        """
        # Normalize vectors
        emb1_norm = emb1 / (np.linalg.norm(emb1) + 1e-8)
        emb2_norm = emb2 / (np.linalg.norm(emb2) + 1e-8)
        # Compute cosine similarity
        similarity = np.dot(emb1_norm, emb2_norm)
        return float(similarity)

    def save_model(self, model_dir: str):
        """Save model"""
        try:
            os.makedirs(model_dir, exist_ok=True)

            # Save sentence-transformer model
            model_path = os.path.join(model_dir, "sentence_transformer")
            self.model.save(model_path)
            logger.info(f"Model saved to: {model_path}")

            # Save configuration
            config = {
                "max_seq_length": self.max_seq_length,
            }
            config_path = os.path.join(model_dir, "config.json")
            with open(config_path, 'w') as f:
                json.dump(config, f)
            logger.info(f"Configuration saved to: {config_path}")

        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise

    def load_model(self, model_dir: str):
        """Load model"""
        try:
            # Load sentence-transformer model
            model_path = os.path.join(model_dir, "sentence_transformer")
            self.model = SentenceTransformer(model_path, device=device)
            logger.info(f"Model loaded from {model_path}")

            # Load configuration
            config_path = os.path.join(model_dir, "config.json")
            with open(config_path, 'r') as f:
                config = json.load(f)
            self.max_seq_length = config["max_seq_length"]
            self.model.max_seq_length = self.max_seq_length
            logger.info(f"Configuration loaded")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

# ------------------------------
# Global Prediction (for multiple tables)
# ------------------------------

class GlobalOptimizer:
    """Global optimizer: batch prediction for multiple tables (using cosine similarity to traverse all column pairs)"""

    def __init__(self, deepjoin_clf: DeepJoinClassifier, top_k_edges=6):
        self.deepjoin_clf = deepjoin_clf
        self.top_k_edges = top_k_edges

    def predict_bi_model(self, tables: List[Table]) -> List[Tuple]:
        """
        Predict all joinable relationships between tables
        Traverse all column pairs of all table pairs, compute cosine similarity, retain Top-K

        Returns:
            [(table1, table2, column1, column2, probability), ...]
        """
        logger.info(f"Starting prediction of join relationships among {len(tables)} tables...")

        joins = []

        # Predict for each table pair
        for i, t1 in tqdm(enumerate(tables), desc="Predicting join probabilities", total=len(tables)):
            for j, t2 in enumerate(tables):
                if i >= j:  # Avoid duplicates and self-joins
                    continue

                # Use heap to retain Top-K column pairs (performance optimization)
                heap = []

                # Traverse all column pairs of t1 and t2
                for col1 in t1.col_names:
                    # Get col1 embedding
                    emb1 = self.deepjoin_clf.get_column_embedding(t1, col1)

                    for col2 in t2.col_names:
                        # Get col2 embedding
                        emb2 = self.deepjoin_clf.get_column_embedding(t2, col2)

                        # Compute cosine similarity
                        similarity = self.deepjoin_clf.compute_cosine_similarity(emb1, emb2)

                        # Use min-heap to retain Top-K (more efficient)
                        if similarity > 0:
                            if len(heap) < self.top_k_edges:
                                heapq.heappush(heap, (similarity, col1, col2))
                            elif similarity > heap[0][0]:
                                heapq.heapreplace(heap, (similarity, col1, col2))

                # Add Top-K column pairs to results
                for similarity, col1, col2 in heap:
                    joins.append((t1, t2, col1, col2, similarity))

        logger.info(f"Prediction complete, found {len(joins)} potential joins")
        return joins

# ------------------------------
# Main Process
# ------------------------------

def DeepJoinClassify(mode, tables, model_dir, links=None, output=None, top_k=6, isfolder=False):
    """
    DeepJoin main function

    Args:
        mode: 'train' or 'predict'
        tables: Table directory path
        model_dir: Model directory
        links: Positive samples file (required for training)
        output: Output file (required for prediction)
        top_k: Number of best joins to retain per table pair
        isfolder: Whether to use folder structure
    """
    try:
        logger.info(f"=== Starting {mode} mode ===")

        if mode == 'train':
            logger.info("Loading training data...")
            if not links:
                raise ValueError("Training mode requires --links parameter")

            tables_dict = load_tables(tables, isfolder)
            if not tables_dict:
                raise ValueError(f"Failed to load any tables, please check path: {tables}")

            bimodel = load_links_csv(links, tables_dict)
            if not bimodel.joins:
                logger.warning("No valid join relationships found")

            # Create DeepJoin classifier
            clf = DeepJoinClassifier()

            # Train model
            clf.train([bimodel], batch_size=32, epochs=10, shuffle_rate=0.3)

            # Save model
            clf.save_model(model_dir)
            logger.info("=== Training complete ===")

        elif mode == 'predict':
            logger.info("Loading prediction data...")
            if not output:
                raise ValueError("Prediction mode requires --output parameter")

            tables_dict = load_tables(tables, isfolder)
            if not tables_dict:
                raise ValueError(f"Failed to load any tables, please check path: {tables}")
            logger.info(f"Will predict for {len(tables_dict)} tables")

            # Load model
            clf = DeepJoinClassifier()
            clf.load_model(model_dir)

            # Generate column embeddings (no longer using FAISS index)
            clf.build_embeddings(list(tables_dict.values()))

            # Global prediction
            opt = GlobalOptimizer(clf, top_k)
            logger.info(f"Starting prediction, Top-K={top_k}")

            results = []
            for t1, t2, c1, c2, prob in opt.predict_bi_model(list(tables_dict.values())):
                results.append({
                    "table1": t1.unique_id,
                    "table2": t2.unique_id,
                    "column1": c1,
                    "column2": c2,
                    "prob": prob
                })

            logger.info(f"Prediction complete, found {len(results)} join relationships")

            # Save results
            with open(output, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to: {output}")
            logger.info("=== Prediction complete ===")

        else:
            raise ValueError(f"Unsupported mode: {mode}")

    except Exception as e:
        logger.error(f"Execution failed: {e}")
        logger.exception("Detailed error traceback:")
        raise

# ------------------------------
# Command Line Entry
# ------------------------------

if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description='DeepJoin - Table Join Relationship Prediction Tool Based on Pre-trained Language Models')
        parser.add_argument('--mode', required=True, choices=['train', 'predict'],
                            help='Run mode: train (training) or predict (prediction)')
        parser.add_argument('--links', help='Training mode: positive sample join relationship file path (CSV format)')
        parser.add_argument('--tables', required=True, help='Table data directory path')
        parser.add_argument('--model_dir', required=True, help='Model save/load directory')
        parser.add_argument('--output', help='Prediction mode: result output file path (JSON format)')
        parser.add_argument('--top_k', type=int, default=6, help='Number of best joins to retain per table pair during prediction (default 6)')
        parser.add_argument('--isfolder', action='store_true',
                            help='If set, indicates tables directory contains subfolder structure')
        args = parser.parse_args()

        # Parameter validation
        if args.mode == 'train' and not args.links:
            parser.error("Training mode (--mode train) requires --links parameter")
        if args.mode == 'predict' and not args.output:
            parser.error("Prediction mode (--mode predict) requires --output parameter")

        logger.info("=" * 60)
        logger.info("DeepJoin Starting")
        logger.info("=" * 60)

        DeepJoinClassify(args.mode, args.tables, args.model_dir,
                        args.links, args.output, args.top_k, args.isfolder)

        logger.info("=" * 60)
        logger.info("Program execution successfully completed")
        logger.info("=" * 60)

    except KeyboardInterrupt:
        logger.warning("\nProgram interrupted by user")
        exit(1)
    except Exception as e:
        logger.error("=" * 60)
        logger.error("Program execution failed")
        logger.error("=" * 60)
        logger.error(f"Error: {e}")
        logger.exception("Detailed error traceback:")
        exit(1)
