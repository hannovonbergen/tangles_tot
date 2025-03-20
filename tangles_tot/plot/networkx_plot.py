from typing import Union
import networkx as nx
from tangles_tot.tree import FeatureTree

NXTree = Union[nx.Graph, nx.DiGraph]


def feature_tree_to_nx(feature_tree: FeatureTree) -> NXTree:
    """
    Builds a networkx graph representation of a FeatureTree.

    If the features in the feature_tree are unspecified, i.e. they are potential features, then
    the graph object is a nx.Graph object. If the features are specified then a directed graph
    is returned.

    Args:
        feature_tree: The feature tree to turn into a network graph (or directed graph) object.

    Returns:
        A networkx graph object which can be used to plot a graph representation of the feature tree.
        The nodes are the index of the node in the feature_tree, furthermore they have associated_tangle
        and label attributes.
        The edges have a feature_id attribute and a label attribute.
    """
    graph = nx.DiGraph() if _is_oriented(feature_tree) else nx.Graph()
    for location in feature_tree.locations():
        graph.add_node(
            location.node_idx,
            associated_tangle=location.associated_tangle,
            label=location.label,
        )
    for feature_id in feature_tree.feature_ids():
        feature_edge = feature_tree.get_edge(feature_id)
        first_node, second_node = (
            feature_tree.get_node_idx_of_location_containing((feature_id, -1)),
            feature_tree.get_node_idx_of_location_containing((feature_id, 1)),
        )
        if feature_edge.specification and feature_edge.specification == -1:
            second_node, first_node = first_node, second_node
        graph.add_edge(
            first_node,
            second_node,
            feature_id=feature_id,
            label=feature_edge.label,
        )

    return graph


def _is_oriented(feature_tree: FeatureTree) -> bool:
    for feature_id in feature_tree.feature_ids():
        if feature_tree.get_edge(feature_id).specification is not None:
            return True
    return False
