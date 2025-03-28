from typing import Union, Optional, Any
import networkx as nx
from tangles_tot.tree import (
    FeatureTree,
    TreeOfTangles,
    FeatureSpecification,
    FeatureLabels,
    LocationLabels,
)
from .networkx_plot import feature_tree_to_nx


def plot_feature_tree(
    feature_tree: FeatureTree,
    feature_labels: Optional[FeatureLabels] = None,
    location_labels: Optional[LocationLabels] = None,
    feature_specification: Optional[FeatureSpecification] = None,
    ax: Optional[Any] = None,
):
    """
    Plot the tree structure of the feature tree.

    The edges of the plotted tree correspond to the features in the feature tree.
    If the features are not specified, i.e. they are potential features,
    then neither are the edges of the tree are undirected.
    If the features are specified, then the edges are directed - pointing
    towards the feature.

    Args:
        feature_tree: The feature tree to plot.
        feature_labels: Optional labels for the edges of the tree.
        location_lables: Optional labels for the nodes of the tree.
        feature_specification: Optional specification for the edges of the tree.
        ax: Optional matplotlib axis object.
    """
    graph = feature_tree_to_nx(
        feature_tree=feature_tree,
        feature_labels=feature_labels,
        location_labels=location_labels,
        feature_specification=feature_specification,
    )
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
    feature_labels: Optional[FeatureLabels] = None,
    location_labels: Optional[LocationLabels] = None,
    feature_specification: Optional[FeatureSpecification] = None,
    ax: Optional[Any] = None,
):
    """
    Plot the tree structure of a tree of tangles.

    Args:
        feature_tree: The tree of tangles or feature tree to plot.
        ax: Optional matplotlib axis object.
    """
    if isinstance(tree, TreeOfTangles):
        feature_tree = tree.feature_tree
        feature_labels = feature_labels or tree.label_features_by_id()
        location_labels = location_labels or tree.label_locations_by_idx()
    elif isinstance(tree, FeatureTree):
        feature_tree = tree
    else:
        raise ValueError(f"tree {tree} must be of type TreeOfTangles or FeatureTree")
    plot_feature_tree(
        feature_tree,
        feature_labels=feature_labels,
        location_labels=location_labels,
        feature_specification=feature_specification,
        ax=ax,
    )
