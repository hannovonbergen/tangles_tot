import pytest
from tangles_tot._testing.feature_trees import three_star
from .tree_of_tangles import TreeOfTangles


@pytest.fixture
def tree_of_tangles() -> TreeOfTangles:
    return TreeOfTangles(three_star())


def test_label_feature_by_id(tree_of_tangles: TreeOfTangles):
    assert tree_of_tangles.label_features_by_id() == {
        0: "feature 0",
        1: "feature 1",
        2: "feature 2",
    }


def test_label_locations_by_idx(tree_of_tangles: TreeOfTangles):
    assert tree_of_tangles.label_locations_by_idx() == {
        0: "location 0",
        1: "location 1",
        2: "location 2",
        3: "location 3",
    }


def test_default_specification(tree_of_tangles: TreeOfTangles):
    assert tree_of_tangles.default_specification() == {
        0: 1,
        1: 1,
        2: 1,
    }
