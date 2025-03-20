from tangles_tot._testing.feature_trees import three_star
from .tree_of_tangles import TreeOfTangles


def test_feature_tree_of_tot():
    tot = TreeOfTangles(three_star(False))
    for edge in tot.feature_tree.edges():
        assert edge.specification is None


def test_feature_tree_default_orientation():
    tot = TreeOfTangles(three_star(False))
    for edge in tot.default_specification().edges():
        assert edge.specification == 1
