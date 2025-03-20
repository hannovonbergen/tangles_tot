import networkx as nx
from tangles_tot._testing.feature_trees import three_star
from .networkx_plot import feature_tree_to_nx


def test_unoriented_three_star_to_nx():
    unoriented_three_star = three_star(False)
    unoriented_three_star_nx = feature_tree_to_nx(unoriented_three_star)
    assert isinstance(unoriented_three_star_nx, nx.Graph)
    assert unoriented_three_star_nx.number_of_nodes() == 4
    assert unoriented_three_star_nx.number_of_edges() == 3
    for i in range(3):
        assert unoriented_three_star_nx[i][3]["feature_id"] == i
        assert unoriented_three_star_nx[i][3]["label"] == f"feature {i}"


def test_oriented_three_star_to_nx():
    oriented_three_star = three_star(True)
    oriented_three_star_nx = feature_tree_to_nx(oriented_three_star)
    assert isinstance(oriented_three_star_nx, nx.DiGraph)
    assert oriented_three_star_nx.number_of_nodes() == 4
    assert oriented_three_star_nx.number_of_edges() == 3
    for i in range(3):
        assert oriented_three_star_nx.get_edge_data(3, i) is None
        assert oriented_three_star_nx[i][3]["feature_id"] == i
        assert oriented_three_star_nx[i][3]["label"] == f"feature {i}"
