from typing import Union
import networkx as nx
from tangles_tot.tree import FeatureTree

NXTree = Union[nx.Graph, nx.DiGraph]


def feature_tree_to_nx(feature_tree: FeatureTree) -> NXTree:
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
