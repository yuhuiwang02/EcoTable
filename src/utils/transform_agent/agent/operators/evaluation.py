"""
Evaluation Operators for Join Quality Assessment

This module implements operators aligned with EcoTable paper Section 5:
- O_val (Validation Operator): ValidateJoin - computes containment ratio

Additional operators for workflow control:
- TokenRatio: Calculate token overlap ratio (ignoring word order)
- Commit: Return the table with highest ExactRatio
"""

import duckdb
import logging
import pandas as pd
from typing import List, Dict, Any, Tuple

logger = logging.getLogger("transform_agent")


def _column_exists(con, table_name: str, column_name: str) -> bool:
    """Check if a column exists in a table."""
    cols = con.execute(f"PRAGMA table_info('{table_name}')").fetchdf()['name'].tolist()
    return column_name in cols


# ══════════════════════════════════════════════════════════════════════════════
# O_val: Validation Operator
# ══════════════════════════════════════════════════════════════════════════════

class ValidateJoin:
    """
    O_val: Validation operator from paper Section 5.

    Computes containment ratio: Containment(A, B) = |A ∩ B| / |A|
    Signals to LLM whether tables can be joined or need further transformation.

    This is the core validation operator that guides the iterative ReAct workflow.
    """

    def __init__(self, duckdb_path: str):
        self.duckdb_path = duckdb_path

    def execute(self, table1: str, table2: str,
                join_col1: str, join_col2: str) -> float:
        """
        Validate if two tables can be joined on given columns.

        Computes the containment ratio (exact match ratio) between join columns.

        Args:
            table1: Name of the first table
            table2: Name of the second table
            join_col1: Join column in table1
            join_col2: Join column in table2

        Returns:
            Containment ratio (0.0 to 1.0), where 1.0 means perfect joinability.
        """
        with duckdb.connect(self.duckdb_path) as con:
            # Guard: check column existence after structural transforms
            if not _column_exists(con, table1, join_col1):
                logger.warning(f"ValidateJoin: column '{join_col1}' not found in '{table1}', returning 0.0")
                return 0.0
            if not _column_exists(con, table2, join_col2):
                logger.warning(f"ValidateJoin: column '{join_col2}' not found in '{table2}', returning 0.0")
                return 0.0

            # Count total distinct values in table1
            total_query = f"""
                SELECT COUNT(DISTINCT "{join_col1}") as total
                FROM "{table1}"
                WHERE "{join_col1}" IS NOT NULL
            """
            total = con.execute(total_query).fetchone()[0]

            if total == 0:
                return 0.0

            # Count exact matches (intersection)
            match_query = f"""
                SELECT COUNT(DISTINCT CAST(t1."{join_col1}" AS VARCHAR)) as matches
                FROM "{table1}" t1
                INNER JOIN "{table2}" t2
                ON CAST(t1."{join_col1}" AS VARCHAR) = CAST(t2."{join_col2}" AS VARCHAR)
                WHERE t1."{join_col1}" IS NOT NULL
            """
            matches = con.execute(match_query).fetchone()[0]

            return matches / total


class ExactRatio(ValidateJoin):
    """
    Alias for ValidateJoin for backward compatibility.

    Calculate the exact match ratio between two columns.
    Computes the percentage of rows with exact string-to-string matches.
    """
    pass


class TokenRatio:
    """
    Calculate token overlap ratio between two columns.
    Computes overlap ignoring word order, allowing for token permutations.
    """

    def __init__(self, duckdb_path: str):
        self.duckdb_path = duckdb_path

    def _tokenize(self, text: str) -> set:
        """Tokenize text into words."""
        if pd.isna(text):
            return set()
        return set(str(text).lower().split())

    def _token_similarity(self, text1: str, text2: str) -> float:
        """Calculate token-based similarity between two texts."""
        tokens1 = self._tokenize(text1)
        tokens2 = self._tokenize(text2)

        if len(tokens1) == 0 and len(tokens2) == 0:
            return 1.0
        if len(tokens1) == 0 or len(tokens2) == 0:
            return 0.0

        intersection = tokens1.intersection(tokens2)
        union = tokens1.union(tokens2)

        return len(intersection) / len(union)

    def execute(self, table1: str, table2: str,
                join_col1: str, join_col2: str,
                threshold: float = 0.8) -> float:
        """
        Calculate token overlap ratio.

        Args:
            table1: Name of the first table
            table2: Name of the second table
            join_col1: Join column in table1
            join_col2: Join column in table2
            threshold: Similarity threshold to consider a match (default: 0.8)

        Returns:
            Token match ratio (0.0 to 1.0)
        """
        with duckdb.connect(self.duckdb_path) as con:
            # Guard: check column existence after structural transforms
            if not _column_exists(con, table1, join_col1):
                logger.warning(f"TokenRatio: column '{join_col1}' not found in '{table1}', returning 0.0")
                return 0.0
            if not _column_exists(con, table2, join_col2):
                logger.warning(f"TokenRatio: column '{join_col2}' not found in '{table2}', returning 0.0")
                return 0.0

            # Get distinct values from both tables
            vals1 = con.execute(f"""
                SELECT DISTINCT CAST("{join_col1}" AS VARCHAR) as val
                FROM "{table1}"
                WHERE "{join_col1}" IS NOT NULL
            """).fetchdf()['val'].tolist()

            vals2 = con.execute(f"""
                SELECT DISTINCT CAST("{join_col2}" AS VARCHAR) as val
                FROM "{table2}"
                WHERE "{join_col2}" IS NOT NULL
            """).fetchdf()['val'].tolist()

            if len(vals1) == 0:
                return 0.0

            # Count matches based on token similarity
            matches = 0
            for v1 in vals1:
                for v2 in vals2:
                    if self._token_similarity(v1, v2) >= threshold:
                        matches += 1
                        break  # Found a match for v1, move to next

            return matches / len(vals1)


class Commit:
    """
    Commit operator: Signal that no further modifications are needed.
    When the LLM determines that the current transformation result is satisfactory
    and no further improvement can be achieved, it uses Commit to submit the
    current best version as the final result, ending the iteration loop.
    """

    def __init__(self, duckdb_path: str):
        self.duckdb_path = duckdb_path

    def execute(self, best_table1: str, best_table2: str,
                best_score: float) -> Tuple[str, str, float]:
        """
        Commit the current best version as the final result.

        Args:
            best_table1: The current best version of table1
            best_table2: The current best version of table2
            best_score: The best ExactRatio score achieved so far

        Returns:
            Tuple of (best_table1, best_table2, best_score)
        """
        return best_table1, best_table2, best_score
