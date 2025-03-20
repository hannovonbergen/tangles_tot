from typing import Union, Optional, Any
import networkx as nx
from tangles_tot.tree import FeatureTree, TreeOfTangles
from .networkx_plot import feature_tree_to_nx


def plot_feature_tree(feature_tree: FeatureTree, ax: Optional[Any] = None):
    graph = feature_tree_to_nx(feature_tree)
    pos = nx.layout.kamada_kawai_layout(graph)
    nx.draw_networkx_nodes(graph, pos=pos, ax=ax)
    nx.draw_networkx_edges(graph, pos=pos, ax=ax)
    nx.draw_networkx_labels(
        graph,
        pos=pos,
        labels=dict(graph.nodes(data="label")),
        ax=ax,
    )
    nx.draw_networkx_edge_labels(
        graph,
        pos=pos,
        edge_labels={(a, b): label for (a, b, label) in graph.edges(data="label")},
        ax=ax,
    )


def plot_tree_of_tangles(
    tree: Union[TreeOfTangles, FeatureTree],
    ax: Optional[Any] = None,
):
    if isinstance(tree, TreeOfTangles):
        feature_tree = tree.feature_tree
    elif isinstance(tree, FeatureTree):
        feature_tree = tree
    else:
        raise ValueError(f"tree {tree} must be of type TreeOfTangles or FeatureTree")
    plot_feature_tree(feature_tree, ax=ax)
