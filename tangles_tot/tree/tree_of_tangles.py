from typing import Union
from tangles_tot._typing import Feature, FeatureId, Specification
from .feature_tree import FeatureTree

FeatureLabels = dict[Union[FeatureId, Feature], str]
LocationIdx = int
LocationLabels = dict[LocationIdx, str]
FeatureSpecification = dict[FeatureId, Specification]


class TreeOfTangles:
    """
    A tree of tangles.

    Attribute:
        feature_tree: A FeatureTree encoding the tree structure of the (unspecified) features of the tree of tangles.

    Note:
        TreeOfTangles are not intended to be built using the constructor but instead by using
        one of the factory methods.
    """

    def __init__(self, feature_tree: FeatureTree):
        """
        @private
        """
        self.feature_tree = feature_tree

    def label_features_by_id(self) -> FeatureLabels:
        """
        Returns labels of the features of the form "label {feature_id}" for
        each feature of the feature tree.
        """
        return {
            feature_id: f"feature {feature_id}"
            for feature_id in self.feature_tree.feature_ids()
        }

    def label_locations_by_idx(self) -> LocationLabels:
        """
        Returns label of the locations of the form "location {node_idx}" for
        each location of the feature tree.
        """
        return {
            location.node_idx: f"location {location.node_idx}"
            for location in self.feature_tree.locations()
        }

    def default_specification(self) -> FeatureSpecification:
        """
        Specifies every feature as "1", the default orientation.
        """
        return {feature_id: 1 for feature_id in self.feature_tree.feature_ids()}
