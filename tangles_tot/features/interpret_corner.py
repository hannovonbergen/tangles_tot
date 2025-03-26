from typing import Union, Optional
import numpy as np
from tangles_tot._tangles_lib import MetaData, FeatureSystem, SetSeparationSystem
from tangles_tot._typing import Feature
from tangles_tot.search import UncrossingFeatureSystem
from .logic import TextTerm, _SemanticTextTerm

MetaDataType = Union[str, TextTerm, MetaData]
FeatureArray = np.ndarray


def interpret_feature_array(
    feature: np.ndarray,
    original_features: np.ndarray,
    metadata: list[MetaDataType],
    under_condition: Optional[FeatureArray] = None,
) -> TextTerm:
    """Interpret a feature array by representing it as a logical term.

    The original features can be interpretet as statements (whose names are given by the metadata)
    which elements of the groundset either have (if the corresponding value is 1) or not have
    (if the corresponding value is -1).

    Suppose feature was constructed from the original features from a combination of intersections,
    unions and complements of the original features.

    This method reconstructs a logical term which describes how it is possible to use the
    statements of the original features to obtain the same statement as the new feature.

    For example if feature is the intersection of original_feature[:, 0] with metadata "A" and
    original_feature[:, 1] with metadata "B" then feature would be represented by the statement
    "A and B".

    If we were to condition this statement under the statement "A", by putting original_feature[:, 0] into
    the under_condition argument, the output would be B since we assume A to be true from the beginning.

    Args:
        feature: The feature to interpret.
        original_features: Array of features which are labeled for reference.
        metadata: List containing labels for each feature.
        under_condition: Optional feature. If provided condition the output statement on the under_condition feature being true.

    Returns:
        A TextTerm representing the reconstructed logical interpretation of the feature.
    """
    if under_condition is None:
        under_condition = np.ones(feature.shape[0], dtype=np.int8)
    rec_log = _RecursionLogic(
        original_features[under_condition == 1], metadata, feature[under_condition == 1]
    )
    starting_approximation = np.ones(feature.shape, dtype=np.int8)[under_condition == 1]
    return _array_to_term_recursive(
        sep=feature[under_condition == 1],
        approximation=starting_approximation,
        next_term=_SemanticTextTerm.true(len(starting_approximation)),
        rec_log=rec_log,
    ).text


def interpret_feature(
    feature: Feature,
    feat_sys: Union[FeatureSystem, UncrossingFeatureSystem],
    under_condition: Optional[list[Feature]] = None,
) -> TextTerm:
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


def _array_to_term_recursive(
    sep: np.ndarray,
    approximation: np.ndarray,
    next_term: _SemanticTextTerm,
    rec_log: "_RecursionLogic",
) -> _SemanticTextTerm:
    next_sep = np.minimum(sep, next_term.array)
    next_approx = np.minimum(approximation, next_term.array)

    if np.all(next_approx == next_sep):
        return next_term
    if np.all(next_sep == -1):
        return _SemanticTextTerm.false(sep.shape[0])

    new_term = rec_log.find_best_term_extension(sep=next_sep, term=next_approx)

    first_term = _array_to_term_recursive(
        sep=next_sep,
        approximation=next_approx,
        next_term=new_term,
        rec_log=rec_log,
    )
    second_term = _array_to_term_recursive(
        sep=next_sep,
        approximation=next_approx,
        next_term=new_term.not_(),
        rec_log=rec_log,
    )
    or_term = rec_log.or_term(first_term, second_term, next_term)
    and_term = rec_log.and_term(next_term, or_term, approximation)
    return and_term


class _RecursionLogic:
    def __init__(
        self, features: np.ndarray, feature_labels: list[str], og_sep: np.ndarray
    ):
        self._features = features
        self._og_sep = og_sep
        self._terms = [
            _SemanticTextTerm(TextTerm(feature_labels[i]), features[:, i])
            for i in range(features.shape[1])
        ]

    def find_best_term_extension(
        self, sep: np.ndarray, term: np.ndarray
    ) -> _SemanticTextTerm:
        mask_ab = np.minimum(term, -sep) == 1
        mask_cd = np.minimum(term, sep) == 1
        a_ar = np.sum(self._features[mask_ab] == 1, axis=0)
        b_ar = np.sum(-self._features[mask_ab] == 1, axis=0)
        c_ar = np.sum(self._features[mask_cd] == 1, axis=0)
        d_ar = np.sum(-self._features[mask_cd] == 1, axis=0)

        nested_bias = np.maximum(a_ar * (c_ar == 0), b_ar * (d_ar == 0))
        if np.any(nested_bias) > 0:
            return self._terms[np.argmax(nested_bias)]
        scores_1 = np.zeros(self._features.shape[1], dtype=np.float64)
        scores_1[c_ar != 0] = a_ar[c_ar != 0] / c_ar[c_ar != 0]
        scores_2 = np.zeros(self._features.shape[1], dtype=np.float64)
        scores_2[d_ar != 0] = b_ar[d_ar != 0] / d_ar[d_ar != 0]
        return self._terms[np.argmax(np.maximum(scores_1, scores_2))]

    def or_term(
        self,
        first_term: _SemanticTextTerm,
        second_term: _SemanticTextTerm,
        text_term: _SemanticTextTerm,
    ) -> _SemanticTextTerm:
        if np.all(
            np.minimum(first_term.array, text_term.array)
            <= np.minimum(second_term.array, text_term.array)
        ):
            return second_term
        if np.all(
            np.minimum(second_term.array, text_term.array)
            <= np.minimum(first_term.array, text_term.array)
        ):
            return first_term
        if second_term.array.sum() <= first_term.array.sum():
            return first_term.or_(second_term)
        return second_term.or_(first_term)

    def and_term(
        self,
        text_term: _SemanticTextTerm,
        or_term: _SemanticTextTerm,
        approximation: np.ndarray,
    ) -> _SemanticTextTerm:
        if np.all(np.minimum(or_term.array, approximation) <= self._og_sep):
            return or_term
        return text_term.and_(or_term)
