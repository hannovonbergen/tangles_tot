from typing import Union
import numpy as np
from tangles_tot._tangles_lib import FeatureSystem
from tangles_tot.search import UncrossingFeatureSystem
from tangles_tot.tree import TreeOfTangles, FeatureLabels, LocationLabels
from .interpret_corner import interpret_feature, interpret_feature_array


def label_corners_using_logic_term(
    tree_of_tangles: TreeOfTangles,
    feat_sys: Union[FeatureSystem, UncrossingFeatureSystem],
) -> FeatureLabels:
    feature_labels = {}

    for feature_id in tree_of_tangles.feature_ids():
        feature_labels[(feature_id, 1)] = interpret_feature(
            feature=(feature_id, 1), feat_sys=feat_sys
        )
        feature_labels[(feature_id, -1)] = interpret_feature(
            feature=(feature_id, -1), feat_sys=feat_sys
        )

    return feature_labels


def label_conditioned_corners_using_logic_term(
    tree_of_tangles: TreeOfTangles,
    feat_sys: Union[FeatureSystem, UncrossingFeatureSystem],
) -> FeatureLabels:
    feature_labels = {}

    all_features = [(feature_id, 1) for feature_id in tree_of_tangles.feature_ids()] + [
        (feature_id, -1) for feature_id in tree_of_tangles.feature_ids()
    ]

    for feature_id, spec in all_features:
        location = tree_of_tangles.feature_tree.get_location_containing(
            (feature_id, -spec)
        )
        conditions = [
            feature for feature in location.features if feature != (feature_id, -spec)
        ]
        assert (
            len(conditions) == len(location.features) - 1
        ), "critical error in conditioned feature labeling method"
        feature_labels[(feature_id, spec)] = interpret_feature(
            feature=(feature_id, spec),
            feat_sys=feat_sys,
            under_condition=conditions,
        )

    return feature_labels


def label_locations_using_logic_term(
    tree_of_tangles: TreeOfTangles,
    feat_sys: Union[FeatureSystem, UncrossingFeatureSystem],
) -> LocationLabels:
    if isinstance(feat_sys, FeatureSystem):
        feat_sys = UncrossingFeatureSystem.from_feature_system(feat_sys)
    location_labels = {}

    for location in tree_of_tangles.locations():
        ids = np.array([id for id, _ in location.features], dtype=int)
        specs = np.array([spec for _, spec in location.features], dtype=np.int8)
        location_array = feat_sys.compute_infimum(ids, specs)
        location_labels[location.node_idx] = interpret_feature_array(
            feature=location_array,
            original_features=feat_sys.get_original_features(),
            metadata=feat_sys.get_metadata_of_original_features(),
        )

    return location_labels
