from typing import Union, Optional
import networkx as nx
from tangles_tot.tree import (
    FeatureTree,
    LocationLabels,
    FeatureLabels,
    FeatureSpecification,
)

NXTree = Union[nx.Graph, nx.DiGraph]


def feature_tree_to_nx(
    feature_tree: FeatureTree,
    location_labels: Optional[LocationLabels] = None,
    feature_labels: Optional[FeatureLabels] = None,
    feature_specification: Optional[FeatureSpecification] = None,
) -> NXTree:
    """
    Builds a networkx graph representation of a FeatureTree.

    If no specifications are given, then
    the graph object is a nx.Graph object. If the features are specified then a directed graph
    is returned.

    Args:
        feature_tree: The feature tree to turn into a network graph (or directed graph) object.
        location_lables: Optional labels for the nodes of the tree.
        feature_lables: Optional labels for directed or undirected edges of the tree.
        feature_specification: Optional specification for the edges of the tree, if provided,
            edge without an orientation are specified in both directions.

    Returns:
        A networkx graph object which can be used to plot a graph representation of the feature tree.
        The nodes are the index of the node in the feature_tree, furthermore they have associated_tangle
        and label attributes.
        The edges have a feature_id attribute and a label attribute.
    """
    specified = feature_specification is not None
    location_labels = location_labels or {}
    feature_labels = feature_labels or {}
    feature_specification = feature_specification or {}
    graph = nx.Graph() if not specified else nx.DiGraph()
    for location in feature_tree.locations():
        graph.add_node(
            location.node_idx,
            label=location_labels.get(location.node_idx, None) or "",
        )
    for feature_id in feature_tree.feature_ids():
        first_node, second_node = (
            feature_tree.get_node_idx_of_location_containing((feature_id, -1)),
            feature_tree.get_node_idx_of_location_containing((feature_id, 1)),
        )
        if feature_specification.get(feature_id, 0) != -1:
            label = (
                feature_labels.get((feature_id, 1), feature_labels.get(feature_id, ""))
                if specified
                else feature_labels.get(feature_id, "")
            )
            graph.add_edge(first_node, second_node, feature_id=feature_id, label=label)
        if feature_specification.get(feature_id, 0) != 1:
            graph.add_edge(
                first_node,
                second_node,
                feature_id=feature_id,
                label=feature_labels.get(
                    (feature_id, -1), feature_labels.get(feature_id, "")
                ),
            )

    return graph
