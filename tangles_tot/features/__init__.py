"""
This package provides tools for working with features in the context of the tree of tangles.
It contains functionality for reconstructing interpretations for the corners added to a feature system
by the tree of tangles uncrossing algorithm.
"""

from .logic import TextTerm
from .interpret_corner import interpret_feature, interpret_feature_array
from .label_tot import (
    label_corners_using_logic_term,
    label_conditioned_corners_using_logic_term,
    label_locations_using_logic_term,
)

__all__ = [
    "TextTerm",
    "interpret_feature",
    "interpret_feature_array",
    "label_corners_using_logic_term",
    "label_conditioned_corners_using_logic_term",
    "label_locations_using_logic_term",
]
