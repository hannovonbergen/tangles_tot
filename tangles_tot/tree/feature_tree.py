from dataclasses import dataclass
from typing import Optional
from tangles_tot._typing import Specification, Feature, FeatureId, TangleId


@dataclass(frozen=True)
class FeatureEdge:
    """
    An edge of a FeatureTree, corresponds either to a potential
    feature, a partition, if the specification is None, or
    to a feature with given specification.

    Attributes:
        feature_id: the feature id of the edge.
        specification: either None or the specification of the feature.
        label: a description of the feature.
    """

    feature_id: FeatureId
    specification: Optional[Specification]
    label: str


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
    associated_tangle: Optional[TangleId]
    node_idx: int
    label: str


@dataclass
class FeatureTree:
    """
    Encodes the tree structure of a set of nested features.

    Consists of nodes which are Locations, connected by FeatureEdges.

    The FeatureEdges correspond to the nested features.
    """

    _edges: dict[FeatureId, FeatureEdge]
    _locations: list[Location]
    _locations_of_edge: dict[FeatureId, tuple[Location, Location]]

    def feature_ids(self) -> list[FeatureId]:
        """The feature ids of the nested features in the feature tree."""
        return list(self._edges.keys())

    def get_edge(self, feature_id: FeatureId) -> Optional[FeatureEdge]:
        """
        Get the edge corresponding to the id of a feature.

        Returns:
            Either a FeatureEdge if one exists or None if the id of the feature is
            not contained in the feature_ids of the FeatureTree.
        """
        return self._edges.get(feature_id, None)

    def edges(self) -> list[FeatureEdge]:
        """Returns a list of all of the edges of the FeatureTree."""
        return list(self._edges.values())

    def locations(self) -> list[Location]:
        """Returns a list of all of the nodes of the FeatureTree."""
        return self._locations

    def get_location(
        self, node_idx: Optional[int] = None, tangle_id: Optional[TangleId] = None
    ) -> Optional[Location]:
        """
        Gets a location of the FeatureTree either using a node index or a tangle id.

        Raises:
            Value error if both node_idx and tangle_id or either are specified.

        Returns:
            A Location if a matching location exists and None otherwise.
        """
        if node_idx is None and tangle_id is None:
            raise ValueError(
                "to get a location you have to set either the node index or the tangle_id"
            )
        if node_idx is not None and tangle_id is not None:
            raise ValueError("please input either a node_idx or a tangle_id, not both.")
        if node_idx is not None:
            return self._locations[node_idx]
        for location in self._locations:
            if location.associated_tangle == tangle_id:
                return location
        return None

    def get_location_containing(self, feature: Feature) -> Location:
        if feature[1] == 1:
            return self._locations_of_edge[feature[0]][0]
        if feature[1] == -1:
            return self._locations_of_edge[feature[0]][1]
        raise ValueError(f"feature variable {feature} of type Feature has invalid form")

    def get_node_idx_of_location_containing(self, feature: Feature) -> int:
        return self.get_location_containing(feature).node_idx

    def with_specification(
        self, specification: Optional[dict[FeatureId, Specification]]
    ) -> "FeatureTree":
        """
        Returns a new FeatureTree without specifications, only containing non-specified
        partitions if no specification is provided, otherwise orients the
        features as specified in the specifiecation.

        Args:
            specification: Either None or a dictionary mapping EVERY feature id to a specification.

        Returns:
            a new FeatureTree.
        """
        specification = specification or {}
        return FeatureTree(
            _edges={
                feature_id: FeatureEdge(
                    feature_id=feature_id,
                    specification=specification.get(feature_id, None),
                    label=self._edges[feature_id].label,
                )
                for feature_id in self.feature_ids()
            },
            _locations=self._locations,
            _locations_of_edge=self._locations_of_edge,
        )
