from .feature_tree import FeatureTree


class TreeOfTangles:
    """
    A tree of tangles.

    Attribute:
        feature_tree: A FeatureTree containing the (unspecified) features of the tree of tangles.

    Note:
        TreeOfTangles are not intended to be built using the constructor but instead by using
        one of the factory methods.
    """

    def __init__(self, feature_tree: FeatureTree):
        """
        @private
        """
        self.feature_tree = feature_tree

    def default_specification(self) -> FeatureTree:
        """
        Returns:
            A FeatureTree with every feature being specified by its default specification (1).
        """
        return self.feature_tree.with_specification(
            specification={
                feature_id: 1 for feature_id in self.feature_tree.feature_ids()
            }
        )
