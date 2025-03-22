import networkx as nx
from tangles_tot._testing.feature_trees import three_star
from .networkx_plot import feature_tree_to_nx


def test_three_star_to_nx():
    three_star_nx = feature_tree_to_nx(three_star())
    assert isinstance(three_star_nx, nx.Graph)
    assert three_star_nx.number_of_nodes() == 4
    assert three_star_nx.number_of_edges() == 3
    assert three_star_nx.degree(3) == 3
