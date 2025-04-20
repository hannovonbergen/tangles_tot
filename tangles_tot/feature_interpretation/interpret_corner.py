from typing import Union, Optional
import numpy as np
from tangles.separations import MetaData, FeatureSystem, SetSeparationSystem
from tangles_tot.core import Feature
from tangles_tot.core.reconstruct_term import reconstruct_logic_term, Term
from tangles_tot.search import UncrossingFeatureSystem

MetaDataType = Union[str, MetaData]
FeatureArray = np.ndarray


def interpret_feature_array(
    feature: np.ndarray,
    original_features: np.ndarray,
    under_condition: Optional[FeatureArray] = None,
) -> Term:
    """Interpret a feature array by representing it as a logical term.

    The original features can be interpretet as statements (whose names are given by the metadata)
    which elements of the groundset either have (if the corresponding value is 1) or not have
    (if the corresponding value is -1).

    Suppose feature was constructed from the original features from a combination of intersections,
    unions and complements of the original features.

    This method reconstructs a logical term which describes how it is possible to use the
    statements of the original features to obtain the same statement as the new feature.

    For example if feature is the intersection of original_feature[:, 0] and
    original_feature[:, 1] then feature would be represented by the statement
    (0, "and", 1).

    If we were to condition this statement under the statement 0, by putting original_feature[:, 0] into
    the under_condition argument, the output would be 1 since we assume 0 to be true.

    Args:
        feature: The feature to interpret.
        original_features: Array of features which are labeled for reference.
        under_condition: Optional feature. If provided condition the output statement on the under_condition feature being true.

    Returns:
        A TextTerm representing the reconstructed logical interpretation of the feature.
    """
    if under_condition is None:
        under_condition = np.ones(feature.shape[0], dtype=np.int8)
    return reconstruct_logic_term(
        vector=feature[under_condition == 1],
        original_vectors=original_features[under_condition == 1],
    )


def interpret_feature(
    feature: Feature,
    feat_sys: Union[FeatureSystem, SetSeparationSystem],
    under_condition: Optional[list[Feature]] = None,
) -> Term:
    """Interpret a feature of a feature system by representing it as a logical term.

    Very helpful if we added new corners to a FeatureSystem and we are curious about how we can
    describe these corners using the features we originally added to the FeatureSystem.

    This method reconstructs a logical term which describes how it is possible to use the
    statements of the original features to obtain the same statement as the corner feature.

    For example if feature is the intersection of original_feature[:, 0] with metadata "A" and
    original_feature[:, 1] with metadata "B" then feature would be represented by the statement
    "A and B".

    If we were to condition this statement under the statement "A", by putting original_feature[:, 0] into
    the under_condition argument, the output would be B since we assume A to be true from the beginning.

    Args:
        feature: The feature to interpret.
        feat_sys: The feature system containing information about all features.
        under_condition: Optional list of features. If provided condition the output statement on all of the feature being true.

    Returns:
        A TextTerm representing the reconstructed logical interpretation of the feature.
    """
    if isinstance(feat_sys, FeatureSystem):
        feat_sys = UncrossingFeatureSystem.from_feature_system(feat_sys)
    elif isinstance(feat_sys, SetSeparationSystem):
        feat_sys = UncrossingFeatureSystem.from_set_separation_system(feat_sys)
    if under_condition is None or len(under_condition) == 0:
        under_condition_feature = None
    else:
        condition_ids = []
        condition_spec = []
        for id, spec in under_condition:
            condition_ids.append(id)
            condition_spec.append(spec)
        under_condition_feature = feat_sys.compute_infimum(
            condition_ids, condition_spec
        )
    return interpret_feature_array(
        feature=feat_sys.get_feature(feature),
        original_features=feat_sys.get_original_features(),
        metadata=feat_sys.get_metadata_of_original_features(),
        under_condition=under_condition_feature,
    )
