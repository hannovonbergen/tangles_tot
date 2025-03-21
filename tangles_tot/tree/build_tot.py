from typing import Optional
import numpy as np
from tangles_tot._tangles_lib import TangleSweep, LessOrEqFunc
from tangles_tot._typing import FeatureId, Feature
from .tree_of_tangles import TreeOfTangles
from .feature_tree import FeatureTree, FeatureEdge, Location


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
            f"The efficient distinguishers of the tangles of the tangle sweep"
            "have not been uncrossed. Please uncross the efficient distinguishers of the tangle sweep"
            "before providing it to the build tree_of_tangles method"
        )
    feature_tree = _build_feature_tree_from_nested_features(
        efficient_distinguishers, is_le
    )
    return TreeOfTangles(
        feature_tree=feature_tree,
    )


def _build_feature_tree_from_nested_features(
    efficient_distinguishers: np.ndarray,
    is_le: LessOrEqFunc,
) -> FeatureTree:
    _edges = {
        feature_id: FeatureEdge(
            feature_id=feature_id, specification=None, label=f"feature {feature_id}"
        )
        for feature_id in efficient_distinguishers
    }
    _locations, _locations_of_edge = _find_locations(efficient_distinguishers, is_le)

    return FeatureTree(
        _edges=_edges,
        _locations=_locations,
        _locations_of_edge=_locations_of_edge,
    )


def _find_locations(nested_feature_ids: np.ndarray, is_le: LessOrEqFunc) -> FeatureTree:
    _locations = []
    _locations_of_edge: dict[
        FeatureId, tuple[Optional[Location], Optional[Location]]
    ] = {}

    all_features = [(feature_id, 1) for feature_id in nested_feature_ids] + [
        (feature_id, -1) for feature_id in nested_feature_ids
    ]

    for feature in all_features:
        if _is_feature_in_location_already(feature, _locations_of_edge):
            continue
        inverse_of_other_elements_in_location = _find_maximal_features_less_than(
            feature, all_features, is_le
        )
        location_features = [feature] + [
            (feature_id, -specification)
            for (feature_id, specification) in inverse_of_other_elements_in_location
        ]
        _locations.append(
            Location(
                features=location_features,
                associated_tangle=None,
                node_idx=len(_locations),
                label=f"tangle {len(_locations)}",
            )
        )
        for feature_id, specification in location_features:
            if feature_id not in _locations_of_edge:
                _locations_of_edge[feature_id] = (None, None)
            if specification == 1:
                _locations_of_edge[feature_id] = (
                    _locations[-1],
                    _locations_of_edge[feature_id][1],
                )
            else:
                _locations_of_edge[feature_id] = (
                    _locations_of_edge[feature_id][0],
                    _locations[-1],
                )

    return _locations, _locations_of_edge


def _find_maximal_features_less_than(
    feature: Feature,
    all_features: list[Feature],
    is_le: LessOrEqFunc,
) -> list[Feature]:
    maximal_lesser_features = []

    for potential_feature in all_features:
        if potential_feature == feature:
            continue
        if not is_le(
            potential_feature[0], potential_feature[1], feature[0], feature[1]
        ):
            continue
        if any(
            [
                is_le(
                    potential_feature[0],
                    potential_feature[1],
                    current_feature[0],
                    current_feature[1],
                )
                for current_feature in maximal_lesser_features
            ]
        ):
            continue
        maximal_lesser_features = [
            current_feature
            for current_feature in maximal_lesser_features
            if not is_le(
                current_feature[0],
                current_feature[1],
                potential_feature[0],
                potential_feature[1],
            )
        ]
        maximal_lesser_features.append(potential_feature)

    return maximal_lesser_features


def _is_feature_in_location_already(
    feature: Feature,
    _locations_of_edge: dict[FeatureId, tuple[Optional[Location], Optional[Location]]],
) -> bool:
    if _locations_of_edge.get(feature[0]) is None:
        return False
    if feature[1] == 1 and _locations_of_edge.get(feature[0])[0] is None:
        return False
    if feature[1] == -1 and _locations_of_edge.get(feature[0])[1] is None:
        return False
    return True


def _are_efficient_distinguishers_nested(
    is_le: LessOrEqFunc,
    efficient_distinguishers: np.ndarray,
) -> bool:
    for i in range(len(efficient_distinguishers)):
        for j in range(i + 1, len(efficient_distinguishers)):
            if not _is_nested(i, j, is_le):
                return False
    return True


def _is_nested(feature_1: FeatureId, feature_2: FeatureId, is_le: LessOrEqFunc) -> bool:
    return (
        is_le(feature_1, 1, feature_2, 1)
        or is_le(feature_1, -1, feature_2, 1)
        or is_le(feature_1, -1, feature_2, 1)
        or is_le(feature_1, -1, feature_2, -1)
    )
