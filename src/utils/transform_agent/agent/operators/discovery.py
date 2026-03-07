"""
Discovery Operators for Fuzzy Join Detection

This module implements operators to discover potential fuzzy join candidates:
1. GetLeftAnti: Record matched values and return 5 sample rows (with headers) for LLM observation
2. EditDistScan: Find top candidates based on edit distance
3. EmbeddingScan: Find top candidates based on semantic similarity
"""

import duckdb
import re
import threading
import pandas as pd
from typing import List, Tuple, Dict, Any
from rapidfuzz.distance import Levenshtein
import numpy as np

# Pattern to detect UUID/hex-like values where embedding similarity is meaningless
_HEX_LIKE = re.compile(r'^[{]?[0-9a-fA-F\-]{20,}[}]?$')


class GetLeftAnti:
    """
    Record equi-join matched values, then return 5 random sample rows (with headers)
    from each table's unmatched portion for LLM to observe the overall table structure.
    """

    def __init__(self, duckdb_path: str):
        self.duckdb_path = duckdb_path

    def execute(self, table1: str, table2: str, join_col1: str, join_col2: str) -> Dict[str, Any]:
        """
        1. Find matched values (equi-join component)
        2. Return 5 random rows from unmatched portion of each table

        Args:
            table1: Name of the first table
            table2: Name of the second table
            join_col1: Join column in table1
            join_col2: Join column in table2

        Returns:
            Dictionary containing:
            - matched_values: list of values that already match (equi-join component)
            - matched_count: number of matched distinct values
            - table1_sample: DataFrame of 5 random unmatched rows from table1 (with all columns)
            - table2_sample: DataFrame of 5 random unmatched rows from table2 (with all columns)
            - table1_total_unmatched: total unmatched rows in table1
            - table2_total_unmatched: total unmatched rows in table2
        """
        with duckdb.connect(self.duckdb_path) as con:
            # Step 1: Find matched values (equi-join component)
            matched_query = f"""
                SELECT DISTINCT CAST(t1."{join_col1}" AS VARCHAR) as val
                FROM "{table1}" t1
                INNER JOIN "{table2}" t2
                ON CAST(t1."{join_col1}" AS VARCHAR) = CAST(t2."{join_col2}" AS VARCHAR)
                WHERE t1."{join_col1}" IS NOT NULL
            """
            matched_values = con.execute(matched_query).fetchdf()['val'].tolist()

            # Step 2: Get 5 random unmatched rows from table1 (all columns)
            table1_sample_query = f"""
                SELECT t1.*
                FROM "{table1}" t1
                WHERE CAST(t1."{join_col1}" AS VARCHAR) NOT IN (
                    SELECT DISTINCT CAST(t2."{join_col2}" AS VARCHAR)
                    FROM "{table2}" t2
                    WHERE t2."{join_col2}" IS NOT NULL
                )
                ORDER BY RANDOM()
                LIMIT 5
            """
            table1_sample = con.execute(table1_sample_query).fetchdf()

            # Step 3: Get 5 random unmatched rows from table2 (all columns)
            table2_sample_query = f"""
                SELECT t2.*
                FROM "{table2}" t2
                WHERE CAST(t2."{join_col2}" AS VARCHAR) NOT IN (
                    SELECT DISTINCT CAST(t1."{join_col1}" AS VARCHAR)
                    FROM "{table1}" t1
                    WHERE t1."{join_col1}" IS NOT NULL
                )
                ORDER BY RANDOM()
                LIMIT 5
            """
            table2_sample = con.execute(table2_sample_query).fetchdf()

            # Step 4: Count total unmatched
            t1_unmatched_count = con.execute(f"""
                SELECT COUNT(*) FROM "{table1}" t1
                WHERE CAST(t1."{join_col1}" AS VARCHAR) NOT IN (
                    SELECT DISTINCT CAST(t2."{join_col2}" AS VARCHAR)
                    FROM "{table2}" t2 WHERE t2."{join_col2}" IS NOT NULL
                )
            """).fetchone()[0]

            t2_unmatched_count = con.execute(f"""
                SELECT COUNT(*) FROM "{table2}" t2
                WHERE CAST(t2."{join_col2}" AS VARCHAR) NOT IN (
                    SELECT DISTINCT CAST(t1."{join_col1}" AS VARCHAR)
                    FROM "{table1}" t1 WHERE t1."{join_col1}" IS NOT NULL
                )
            """).fetchone()[0]

            return {
                "matched_values": matched_values,
                "matched_count": len(matched_values),
                "table1_sample": table1_sample,
                "table2_sample": table2_sample,
                "table1_total_unmatched": t1_unmatched_count,
                "table2_total_unmatched": t2_unmatched_count
            }


class EditDistScan:
    """
    Find top candidates based on edit distance (Levenshtein distance).
    Identifies potential matches due to spelling errors, case differences, or character variations.
    """

    def __init__(self, duckdb_path: str):
        self.duckdb_path = duckdb_path

    def execute(self, table1: str, table2: str, join_col1: str, join_col2: str,
                top_k: int = 10) -> List[Tuple[Any, Any, float]]:
        """
        Find top-k pairs with smallest edit distance (excluding exact matches).

        Args:
            table1: Name of the first table
            table2: Name of the second table
            join_col1: Join column in table1
            join_col2: Join column in table2
            top_k: Number of top candidates to return (default: 10)

        Returns:
            List of tuples (value1, value2, normalized_distance) sorted by distance
        """
        with duckdb.connect(self.duckdb_path) as con:
            # Get distinct values from both tables
            query1 = f"""
                SELECT DISTINCT CAST("{join_col1}" AS VARCHAR) as val
                FROM "{table1}"
                WHERE "{join_col1}" IS NOT NULL
            """

            query2 = f"""
                SELECT DISTINCT CAST("{join_col2}" AS VARCHAR) as val
                FROM "{table2}"
                WHERE "{join_col2}" IS NOT NULL
            """

            vals1 = con.execute(query1).fetchdf()['val'].tolist()
            vals2 = con.execute(query2).fetchdf()['val'].tolist()

            # Calculate edit distance for all pairs
            candidates = []
            for v1 in vals1:
                for v2 in vals2:
                    str1 = str(v1)
                    str2 = str(v2)

                    # Calculate Levenshtein distance
                    dist = Levenshtein.distance(str1, str2)

                    # Skip exact matches (distance = 0)
                    if dist == 0:
                        continue

                    # Normalize by max length
                    max_len = max(len(str1), len(str2))
                    if max_len > 0:
                        normalized_dist = dist / max_len
                    else:
                        normalized_dist = 0.0

                    # Only consider pairs with some similarity (normalized distance < 0.5)
                    if normalized_dist < 0.5:
                        candidates.append((v1, v2, normalized_dist))

            # Sort by distance and return top-k
            candidates.sort(key=lambda x: x[2])
            return candidates[:top_k]


class EmbeddingScan:
    """
    Find top candidates based on semantic similarity using sentence embeddings.
    Uses all-mpnet-base-v2 model for vectorization.
    """

    _class_lock = threading.Lock()
    _shared_model = None
    _shared_tokenizer = None

    def __init__(self, duckdb_path: str, model_path: str = None):
        self.duckdb_path = duckdb_path

        # Set default model path if not provided
        if model_path is None:
            model_path = "F:/workspace/codeForTransform/EcoTable/all-mpnet-base-v2-local"

        self.model_path = model_path

    def _load_model(self):
        """Thread-safe lazy loading of the embedding model (class-level singleton)."""
        if EmbeddingScan._shared_model is not None:
            return

        with EmbeddingScan._class_lock:
            if EmbeddingScan._shared_model is not None:
                return

            from transformers import AutoTokenizer, AutoModel
            import torch

            EmbeddingScan._shared_tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            EmbeddingScan._shared_model = AutoModel.from_pretrained(self.model_path)
            EmbeddingScan._shared_model.eval()

    def _get_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Get embeddings for a list of texts.

        Args:
            texts: List of text strings

        Returns:
            numpy array of shape (len(texts), embedding_dim)
        """
        import torch

        self._load_model()

        # Tokenize
        encoded = EmbeddingScan._shared_tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )

        # Get embeddings
        with torch.no_grad():
            outputs = EmbeddingScan._shared_model(**encoded)
            # Use mean pooling
            embeddings = outputs.last_hidden_state.mean(dim=1)

        return embeddings.numpy()

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def execute(self, table1: str, table2: str, join_col1: str, join_col2: str,
                top_k: int = 10) -> List[Tuple[Any, Any, float]]:
        """
        Find top-k pairs with highest semantic similarity (excluding exact matches).

        Args:
            table1: Name of the first table
            table2: Name of the second table
            join_col1: Join column in table1
            join_col2: Join column in table2
            top_k: Number of top candidates to return (default: 10)

        Returns:
            List of tuples (value1, value2, similarity_score) sorted by similarity (descending)
        """
        with duckdb.connect(self.duckdb_path) as con:
            # Get distinct values from both tables
            query1 = f"""
                SELECT DISTINCT CAST("{join_col1}" AS VARCHAR) as val
                FROM "{table1}"
                WHERE "{join_col1}" IS NOT NULL
            """

            query2 = f"""
                SELECT DISTINCT CAST("{join_col2}" AS VARCHAR) as val
                FROM "{table2}"
                WHERE "{join_col2}" IS NOT NULL
            """

            vals1 = con.execute(query1).fetchdf()['val'].tolist()
            vals2 = con.execute(query2).fetchdf()['val'].tolist()

            if len(vals1) == 0 or len(vals2) == 0:
                return []

            # Skip embedding scan entirely if most values look like UUIDs/hex
            # (embedding similarity is meaningless for random identifiers)
            hex_ratio1 = sum(1 for v in vals1 if _HEX_LIKE.match(str(v))) / len(vals1)
            hex_ratio2 = sum(1 for v in vals2 if _HEX_LIKE.match(str(v))) / len(vals2)
            if hex_ratio1 > 0.5 or hex_ratio2 > 0.5:
                return []

            # Convert to strings
            texts1 = [str(v) for v in vals1]
            texts2 = [str(v) for v in vals2]

            # Get embeddings
            embeddings1 = self._get_embeddings(texts1)
            embeddings2 = self._get_embeddings(texts2)

            # Calculate similarity for all pairs
            candidates = []
            for i, v1 in enumerate(vals1):
                for j, v2 in enumerate(vals2):
                    # Skip already-matched pairs (identical strings)
                    if str(v1) == str(v2):
                        continue
                    similarity = self._cosine_similarity(embeddings1[i], embeddings2[j])

                    # Only consider pairs with high similarity (> 0.7) but not exact match (< 1.0)
                    if 0.7 < similarity < 1.0:
                        candidates.append((v1, v2, float(similarity)))

            # Sort by similarity (descending) and return top-k
            candidates.sort(key=lambda x: x[2], reverse=True)
            return candidates[:top_k]
