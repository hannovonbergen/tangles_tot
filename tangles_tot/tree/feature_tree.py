from dataclasses import dataclass
from typing import Optional
from tangles_tot._typing import Specification, Feature, FeatureId, TangleId


@dataclass(frozen=True)
class FeatureEdge:
    feature_id: FeatureId
    specification: Optional[Specification]
    label: str


@dataclass(frozen=True)
class Location:
    features: list[Feature]
    associated_tangle: Optional[TangleId]
    node_idx: int
    label: str


@dataclass
class FeatureTree:
    _edges: dict[FeatureId, FeatureEdge]
    _locations: list[Location]
    _locations_of_edge: dict[FeatureId, tuple[Location, Location]]

    def feature_ids(self) -> list[FeatureId]:
        return list(self._edges.keys())

    def get_edge(self, feature_id: FeatureId) -> Optional[FeatureEdge]:
        return self._edges.get(feature_id, None)

    def edges(self) -> list[FeatureEdge]:
        return list(self._edges.values())

    def locations(self) -> list[Location]:
        return self._locations

    def get_location(
        self, node_idx: Optional[int] = None, tangle_id: Optional[TangleId] = None
    ) -> Optional[Location]:
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
