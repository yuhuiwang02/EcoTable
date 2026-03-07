"""
Transformation Operators for Fuzzy Join to Equi-Join Conversion

This module implements operators aligned with EcoTable paper Section 5:
- O_trans (Transformation Operators): Normalize, Split, Pivot

These are inverse operations of the noise injection methods in noise.py.
"""

import duckdb
import pandas as pd
import re
from typing import List, Dict, Any, Optional, Union
from datetime import datetime


class TransformationOperator:
    """
    Base class for O_trans (Transformation Operators) from paper Section 5.

    O_trans includes three categories:
    - Normalize: Handle format inconsistencies (text, date, numeric, fuzzy mapping)
    - Split: Split composite keys into atomic columns
    - Pivot: Restructure table schemas for alignment
    """

    def __init__(self, duckdb_path: str):
        self.duckdb_path = duckdb_path

    def execute(self, table_name: str, column_name: str, **kwargs) -> str:
        """
        Execute the transformation and return the new table name.

        Args:
            table_name: Name of the table to transform
            column_name: Name of the column to transform
            **kwargs: Additional parameters specific to each operator

        Returns:
            Name of the new transformed table
        """
        raise NotImplementedError("Subclasses must implement execute method")


# ══════════════════════════════════════════════════════════════════════════════
# Normalize Operators (O_trans: Normalize)
# ══════════════════════════════════════════════════════════════════════════════

class NormalizeOperator(TransformationOperator):
    """
    Base class for Normalize operators from paper Section 5.

    Normalize operators handle format inconsistencies within a column by
    generating different transformation code for different formats.
    Example: ISO strings ("2013-02-13") vs Unix timestamps ("1360713600")

    Subclasses: TextNorm, ToDate, ToNumeric, FuzzyMap
    """
    pass


class TextNorm(NormalizeOperator):
    """
    Text normalization operator.
    Converts text to uppercase or lowercase, removes unnecessary characters.

    Inverse operations for:
    - apply_lowercase: case normalization
    - apply_extra_space: whitespace removal
    - apply_special_char: special character removal
    """

    def execute(self, table_name: str, column_name: str,
                case: str = 'upper', remove_spaces: bool = True,
                remove_special_chars: bool = True,
                special_chars: str = '!@#$%&*-_') -> str:
        """
        Normalize text in a column.

        Args:
            table_name: Name of the table
            column_name: Name of the column to normalize
            case: 'upper', 'lower', or 'none' (default: 'upper')
            remove_spaces: Whether to remove extra spaces (default: True)
            remove_special_chars: Whether to remove special characters (default: True)
            special_chars: String of special characters to remove

        Returns:
            Name of the new table with normalized column
        """
        with duckdb.connect(self.duckdb_path) as con:
            # Build transformation SQL (cast to VARCHAR first to handle numeric columns)
            transform_expr = f'CAST("{column_name}" AS VARCHAR)'

            # Remove special characters
            if remove_special_chars:
                for char in special_chars:
                    transform_expr = f"REPLACE({transform_expr}, '{char}', '')"

            # Remove all whitespace
            if remove_spaces:
                transform_expr = f"REGEXP_REPLACE({transform_expr}, '\\s', '', 'g')"

            # Case normalization
            if case == 'upper':
                transform_expr = f"UPPER({transform_expr})"
            elif case == 'lower':
                transform_expr = f"LOWER({transform_expr})"

            # Create new table with transformed column (overwrite in place)
            columns_query = f"PRAGMA table_info('{table_name}')"
            columns = con.execute(columns_query).fetchdf()['name'].tolist()

            select_parts = []
            for col in columns:
                if col == column_name:
                    select_parts.append(f'{transform_expr} AS "{column_name}"')
                else:
                    select_parts.append(f'"{col}"')

            create_query = f"""
                CREATE OR REPLACE TABLE "{table_name}" AS
                SELECT {', '.join(select_parts)}
                FROM "{table_name}"
            """

            con.execute(create_query)

            return table_name


class ToDate(NormalizeOperator):
    """
    Date format unification operator.
    Converts inconsistent date strings to standard ISO format (YYYY-MM-DDTHH:MM:SS.000).

    Inverse operation for:
    - apply_datetime_format_change: date format standardization
    """

    def execute(self, table_name: str, column_name: str,
                input_formats: List[str] = None, **kwargs) -> str:
        """
        Unify date formats to fixed ISO standard: YYYY-MM-DDTHH:MM:SS.000

        Args:
            table_name: Name of the table
            column_name: Name of the column to transform
            input_formats: Ignored. Always uses built-in format list.

        Returns:
            Name of the new table with unified date column
        """
        # Always use built-in formats — LLM-provided formats use human-readable
        # notation (e.g. YYYY-MM-DD) which is incompatible with Python strptime.
        input_formats = [
            '%Y-%m-%dT%H:%M:%S.%f',
            '%Y/%m/%d %H:%M:%S',
            '%Y-%m-%d %H:%M:%S',
            '%Y/%m/%d',
            '%Y-%m-%d',
            '%d/%m/%Y',
            '%m/%d/%Y',
            '%d-%m-%Y',
            '%m-%d-%Y'
        ]

        with duckdb.connect(self.duckdb_path) as con:
            # Read the table
            df = con.execute(f'SELECT * FROM "{table_name}"').fetchdf()

            # Transform the date column
            def parse_date(val):
                if pd.isna(val):
                    return val

                val_str = str(val)

                # Try each format
                for fmt in input_formats:
                    try:
                        dt = datetime.strptime(val_str, fmt)
                        return dt.strftime('%Y-%m-%dT%H:%M:%S.000')
                    except:
                        continue

                # If no format works, return original
                return val

            df[column_name] = df[column_name].apply(parse_date)

            # Create new table
            con.execute(f'DROP TABLE IF EXISTS "{table_name}"')
            con.execute(f'CREATE TABLE "{table_name}" AS SELECT * FROM df')

            return table_name


class ToNumeric(NormalizeOperator):
    """
    Numeric extraction operator.
    Extracts numbers from strings containing symbols (%) or scientific notation.

    Inverse operations for:
    - apply_scientific_notation: extract number from scientific notation
    - apply_percentage_format: extract number from percentage
    """

    def execute(self, table_name: str, column_name: str,
                handle_percentage: bool = True,
                handle_scientific: bool = True) -> str:
        """
        Extract numeric values from strings.

        Args:
            table_name: Name of the table
            column_name: Name of the column to transform
            handle_percentage: Convert percentages to decimals (default: True)
            handle_scientific: Parse scientific notation (default: True)

        Returns:
            Name of the new table with numeric column
        """
        with duckdb.connect(self.duckdb_path) as con:
            # Read the table
            df = con.execute(f'SELECT * FROM "{table_name}"').fetchdf()

            # Transform the numeric column
            def parse_numeric(val):
                if pd.isna(val):
                    return val

                val_str = str(val).strip()

                # Handle percentage
                if handle_percentage and '%' in val_str:
                    try:
                        num_str = val_str.replace('%', '').strip()
                        return float(num_str) / 100.0
                    except:
                        pass

                # Handle scientific notation
                if handle_scientific:
                    try:
                        return float(val_str)
                    except:
                        pass

                # Try to extract any number
                try:
                    val_str = val_str.replace(',', '')
                    match = re.search(r'-?\d+\.?\d*', val_str)
                    if match:
                        return float(match.group())
                except:
                    pass

                return val

            df[column_name] = df[column_name].apply(parse_numeric)

            # Create new table
            con.execute(f'DROP TABLE IF EXISTS "{table_name}"')
            con.execute(f'CREATE TABLE "{table_name}" AS SELECT * FROM df')

            return table_name


class FuzzyMap(NormalizeOperator):
    """
    Fuzzy mapping operator.
    Maps abbreviations, aliases, and spelling errors to their standard names.

    Inverse operations for:
    - apply_keyboard_mistake: spelling error correction
    - apply_abbreviation: abbreviation expansion
    - apply_substring: partial match mapping
    """

    def execute(self, table_name: str, column_name: str,
                mapping: Dict[str, str]) -> str:
        """
        Apply fuzzy mapping to standardize values.

        Args:
            table_name: Name of the table
            column_name: Name of the column to transform
            mapping: Dictionary mapping fuzzy values to standard values

        Returns:
            Name of the new table with mapped column
        """
        with duckdb.connect(self.duckdb_path) as con:
            # Read the table
            df = con.execute(f'SELECT * FROM "{table_name}"').fetchdf()

            # Apply mapping
            def apply_mapping(val):
                if pd.isna(val):
                    return val

                val_str = str(val)

                # Direct match
                if val_str in mapping:
                    return mapping[val_str]

                # Case-insensitive match
                val_lower = val_str.lower()
                for key, value in mapping.items():
                    if key.lower() == val_lower:
                        return value

                # Partial match (if value contains key or key contains value)
                for key, value in mapping.items():
                    if key.lower() in val_lower or val_lower in key.lower():
                        return value

                return val

            df[column_name] = df[column_name].apply(apply_mapping)

            # Create new table
            con.execute(f'DROP TABLE IF EXISTS "{table_name}"')
            con.execute(f'CREATE TABLE "{table_name}" AS SELECT * FROM df')

            return table_name


# ══════════════════════════════════════════════════════════════════════════════
# Split Operator (O_trans: Split)
# ══════════════════════════════════════════════════════════════════════════════

class SplitOperator(TransformationOperator):
    """
    Base class for Split operators from paper Section 5.

    Split operators split string columns into atomic columns based on delimiters.
    Helps achieve exact matches when dealing with composite keys.

    Subclasses: SepSplit
    """
    pass


class SepSplit(SplitOperator):
    """
    Separator split operator.
    Splits merged columns based on explicit separators (e.g., comma, dash).

    Inverse operation for:
    - apply_column_level_noise: column merge (splits merged columns)
    """

    def execute(self, table_name: str, column_name: str,
                separator: str = '-', keep_part: int = 0,
                new_column_name: str = None,
                split_from: str = 'right',
                **kwargs) -> str:
        """
        Split a column by separator.

        After splitting, the original column is dropped and replaced by two
        properly-named columns derived from the original column name.
        E.g. "start_date-building_owner_name" split by "-" → "start_date" + "building_owner_name".

        Args:
            table_name: Name of the table
            column_name: Name of the column to split
            separator: Separator character (default: '-')
            keep_part: Which part to keep as the join column (0 for first, 1 for second)
            new_column_name: Ignored (kept for backward compatibility). Names are auto-derived.
            split_from: 'right' (rsplit at last separator, default) or 'left' (split at first separator).

        Returns:
            Name of the new table with split columns
        """
        # Handle string keep_part values from LLM (e.g. "left" -> 0, "right" -> 1)
        if keep_part is None:
            logger.warning(f"SepSplit: keep_part is None for column '{column_name}', skipping")
            return table_name
        if isinstance(keep_part, str):
            keep_part = 0 if keep_part.lower() in ('left', 'first', '0') else 1

        with duckdb.connect(self.duckdb_path) as con:
            # Read the table
            df = con.execute(f'SELECT * FROM "{table_name}"').fetchdf()

            # Split the column value
            def split_value(val, part_index):
                if pd.isna(val):
                    return val

                val_str = str(val)

                if split_from == 'right':
                    parts = val_str.rsplit(separator, 1)
                else:
                    parts = val_str.split(separator, 1)

                if part_index < len(parts):
                    return parts[part_index].strip()

                return val

            # Derive column names from the original column name
            if split_from == 'right':
                name_parts = column_name.rsplit(separator, 1)
            else:
                name_parts = column_name.split(separator, 1)

            if len(name_parts) == 2:
                kept_col_name = name_parts[keep_part]
                other_col_name = name_parts[1 - keep_part]
            else:
                # Cannot derive names from column name, use fallbacks
                kept_col_name = column_name
                other_col_name = new_column_name or f"{column_name}_split"

            other_part = 1 if keep_part == 0 else 0

            # Create new columns with derived names
            df[kept_col_name] = df[column_name].apply(lambda x: split_value(x, keep_part))
            df[other_col_name] = df[column_name].apply(lambda x: split_value(x, other_part))

            # Drop original column (it has been replaced by two properly-named columns)
            if kept_col_name != column_name:
                df.drop(columns=[column_name], inplace=True)

            # Create new table
            con.execute(f'DROP TABLE IF EXISTS "{table_name}"')
            con.execute(f'CREATE TABLE "{table_name}" AS SELECT * FROM df')

            return table_name


# ══════════════════════════════════════════════════════════════════════════════
# Pivot Operators (O_trans: Pivot)
# ══════════════════════════════════════════════════════════════════════════════

class PivotOperator(TransformationOperator):
    """
    Base class for Pivot operators from paper Section 5.

    Pivot operators restructure tables to align schemas for downstream join operations.
    Includes both pivot (long-to-wide) and unpivot (wide-to-long) transformations.

    Subclasses: Pivot, UnPivot
    """
    pass


class Pivot(PivotOperator):
    """
    Pivot operator: Convert long table to wide table.
    Transforms row values into column headers.

    Inverse operation for:
    - apply_unpivot_noise: unpivot (converts long table back to wide)
    """

    def execute(self, table_name: str,
                pivot_col: str,
                value_col: str,
                **kwargs) -> str:
        """
        Pivot a long table to wide format.

        Args:
            table_name: Name of the table
            pivot_col: Column whose values will become new column names
            value_col: Column whose values will fill the new columns

        Returns:
            Name of the new pivoted table
        """
        with duckdb.connect(self.duckdb_path) as con:
            # Read the table
            df = con.execute(f'SELECT * FROM "{table_name}"').fetchdf()

            # Auto-compute index_cols: all columns except pivot_col and value_col
            index_cols = [c for c in df.columns if c not in (pivot_col, value_col)]

            # Perform pivot operation
            pivoted_df = df.pivot_table(
                index=index_cols,
                columns=pivot_col,
                values=value_col,
                aggfunc='first'  # Use first value if there are duplicates
            ).reset_index()

            # Flatten column names if multi-level
            pivoted_df.columns = [str(col) for col in pivoted_df.columns]

            # Create new table
            con.execute(f'DROP TABLE IF EXISTS "{table_name}"')
            con.execute(f'CREATE TABLE "{table_name}" AS SELECT * FROM pivoted_df')

            return table_name


class UnPivot(PivotOperator):
    """
    UnPivot operator: Convert wide table to long table.
    Transforms column headers into row values.

    Inverse operation for:
    - apply_table_level_noise: pivot (converts wide table back to long)
    """

    def execute(self, table_name: str,
                id_vars: List[str],
                value_vars: List[str],
                var_name: str = 'variable',
                value_name: str = 'value') -> str:
        """
        Unpivot a wide table to long format.

        Args:
            table_name: Name of the table
            id_vars: Columns to keep as identifiers (won't be unpivoted)
            value_vars: Columns to unpivot (will become rows)
            var_name: Name for the new column containing variable names
            value_name: Name for the new column containing values

        Returns:
            Name of the new unpivoted table
        """
        with duckdb.connect(self.duckdb_path) as con:
            # Read the table
            df = con.execute(f'SELECT * FROM "{table_name}"').fetchdf()

            # Perform unpivot operation (melt)
            unpivoted_df = pd.melt(
                df,
                id_vars=id_vars,
                value_vars=value_vars,
                var_name=var_name,
                value_name=value_name
            )

            # Create new table
            con.execute(f'DROP TABLE IF EXISTS "{table_name}"')
            con.execute(f'CREATE TABLE "{table_name}" AS SELECT * FROM unpivoted_df')

            return table_name
