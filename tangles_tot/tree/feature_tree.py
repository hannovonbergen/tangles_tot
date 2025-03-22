from dataclasses import dataclass
from typing import Optional
from tangles_tot._typing import Specification, Feature, FeatureId, TangleId


@dataclass(frozen=True)
class Location:
    """
    A node of a FeatureTree, corresponds to a location, the minimal features
    contained in the pre-tangles of the partitions contained in the FeatureTree.

    Attributes:
        features: the list of minimal features.
        associated_tangle: Optionally contains the tangle id of the (unique) maximal tangle which contains the features of this location.
        node_idx: The index of the node in the FeatureTree.
        label: a description of the location.
    """

    features: list[Feature]
    node_idx: int


@dataclass
class FeatureTree:
    """
    Encodes the tree structure of a set of nested features.

    Consists of nodes which are Locations, connected by FeatureEdges.

    The FeatureEdges correspond to the nested features.
    """

    _edges: list[FeatureId]
    _locations: list[Location]
    _locations_of_edge: dict[FeatureId, tuple[Location, Location]]

    def feature_ids(self) -> list[FeatureId]:
        """The feature ids of the nested features in the feature tree."""
        return self._edges

    def contains_edge(self, feature_id: FeatureId) -> bool:
        """
        Checks if a feature id is contained in the feature tree.
        """
        return feature_id in self._edges

    def locations(self) -> list[Location]:
        """Returns a list of all of the nodes of the FeatureTree."""
        return self._locations

    def get_location(self, node_idx: int) -> Location:
        """
        Gets a location of the FeatureTree from a node index.
        """
        return self._locations[node_idx]

    def get_location_containing(self, feature: Feature) -> Location:
        if feature[1] == 1:
            return self._locations_of_edge[feature[0]][0]
        if feature[1] == -1:
            return self._locations_of_edge[feature[0]][1]
        raise ValueError(f"feature variable {feature} of type Feature has invalid form")

    def get_node_idx_of_location_containing(self, feature: Feature) -> int:
        return self.get_location_containing(feature).node_idx
