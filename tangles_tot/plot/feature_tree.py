import networkx as nx
from tangles_tot.tree import FeatureTree
from .networkx_plot import feature_tree_to_nx


def plot_feature_tree(feature_tree: FeatureTree):
    graph = feature_tree_to_nx(feature_tree)
    pos = nx.layout.kamada_kawai_layout(graph)
    nx.draw_networkx_nodes(graph, pos=pos)
    nx.draw_networkx_edges(graph, pos=pos)
    nx.draw_networkx_labels(graph, pos=pos, labels=dict(graph.nodes(data="label")))
    nx.draw_networkx_edge_labels(
        graph,
        pos=pos,
        edge_labels={(a, b): label for (a, b, label) in graph.edges(data="label")},
    )
