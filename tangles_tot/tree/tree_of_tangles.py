from .feature_tree import FeatureTree


class TreeOfTangles:
    def __init__(self, feature_tree: FeatureTree):
        self.feature_tree = feature_tree

    def default_specification(self) -> FeatureTree:
        return self.feature_tree.with_specification(
            specification={
                feature_id: 1 for feature_id in self.feature_tree.feature_ids()
            }
        )
