from tangles_tot._testing.feature_trees import three_star
from tangles_tot.tree import TreeOfTangles
from .feature_tree import plot_tree_of_tangles


def test_plot_tree_of_tangles_with_feature_tree():
    plot_tree_of_tangles(three_star(True))


def test_plot_tree_of_tangles_with_invalid_input():
    try:
        plot_tree_of_tangles("invalid input")
    except ValueError:
        assert True
        return
    assert (
        False
    ), "plot_tree_of_tangles did not raise a value error on input with invalid type"


def test_plot_tree_of_tangles_with_tree_of_tangles():
    plot_tree_of_tangles(TreeOfTangles(three_star(False)))
