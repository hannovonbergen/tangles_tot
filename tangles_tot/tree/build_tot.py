from typing import Optional
import numpy as np
from tangles import TangleSweep
from tangles._typing import LessOrEqFunc
from tangles_tot.core import FeatureId, Feature
from tangles_tot.core.feature_tree import FeatureTree, Location
from .tree_of_tangles import TreeOfTangles


def build_tree_of_tangles_from_sweep(
    tangle_sweep: TangleSweep,
    agreement_value: Optional[int] = None,
) -> TreeOfTangles:
    if not isinstance(tangle_sweep, TangleSweep):
        raise ValueError(
            f"attribute {tangle_sweep}, passed in for tangle_sweep must be a TangleSweep"
        )
    is_le = tangle_sweep._algorithm._core_logic._le_func
    agreement_value = agreement_value or tangle_sweep.tree.limit + 1
    if agreement_value <= tangle_sweep.tree.limit:
        raise ValueError(
            f"input agreement value {agreement_value}"
            f"is not greater than the limit {tangle_sweep.tree.limit} of the search tree."
            "Cannot build a tree of tangles, which contains the efficient distinguishers of all"
            "maximal tangles of more than the specified agreement value if we can not ensure"
            "that we have found all tangles of this agreement range. Please continue sweeping to a lower value"
            "to fix this error."
        )
    _, efficient_distinguishers = tangle_sweep.tree.get_efficient_distinguishers(
        agreement=agreement_value
    )
    if not _are_efficient_distinguishers_nested(is_le, efficient_distinguishers):
        raise ValueError(
            "The efficient distinguishers of the tangles of the tangle sweep"
            "have not been uncrossed. Please uncross the efficient distinguishers of the tangle sweep"
            "before providing it to the build tree_of_tangles method"
        )
    feature_tree = _build_feature_tree_from_nested_features(
        efficient_distinguishers, is_le
    )
    return TreeOfTangles(
        feature_tree=feature_tree,
    )
