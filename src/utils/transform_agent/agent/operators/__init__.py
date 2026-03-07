"""
Operators module for fuzzy join to equi-join transformation.

This module provides operators aligned with EcoTable paper Section 5:

1. O_val (Validation Operator):
   - ValidateJoin: Computes containment ratio to validate joinability

2. O_trans (Transformation Operators):
   - Normalize: Handle format inconsistencies (TextNorm, ToDate, ToNumeric, FuzzyMap)
   - Split: Split composite keys (SepSplit)
   - Pivot: Restructure schemas (Pivot, UnPivot)

3. Discovery operators (implementation detail):
   - GetLeftAnti, EditDistScan, EmbeddingScan

4. Additional operators:
   - TokenRatio, Commit
"""

# Discovery operators
from .discovery import GetLeftAnti, EditDistScan, EmbeddingScan

# Transformation operators - Paper-aligned base classes
from .transformation import (
    TransformationOperator,
    NormalizeOperator,
    SplitOperator,
    PivotOperator,
)

# Transformation operators - Concrete implementations
from .transformation import (
    TextNorm,
    ToDate,
    ToNumeric,
    FuzzyMap,
    SepSplit,
    Pivot,
    UnPivot
)

# Evaluation operators
from .evaluation import ValidateJoin, ExactRatio, TokenRatio, Commit

__all__ = [
    # Discovery
    'GetLeftAnti',
    'EditDistScan',
    'EmbeddingScan',

    # Transformation - Base classes (paper-aligned)
    'TransformationOperator',
    'NormalizeOperator',
    'SplitOperator',
    'PivotOperator',

    # Transformation - Concrete implementations
    'TextNorm',
    'ToDate',
    'ToNumeric',
    'FuzzyMap',
    'SepSplit',
    'Pivot',
    'UnPivot',

    # Validation (O_val)
    'ValidateJoin',
    'ExactRatio',  # Alias for ValidateJoin

    # Additional
    'TokenRatio',
    'Commit',
]
