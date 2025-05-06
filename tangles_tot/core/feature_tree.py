from dataclasses import dataclass
import numpy as np
from .typing import Feature, FeatureId, LessOrEqFunc

Vertex = int
Edge = tuple[Vertex, Vertex]


@dataclass(frozen=True)
class DirectedEdge:
    source: Vertex
    target: Vertex

    def inverse(self) -> "DirectedEdge":
        return DirectedEdge(
            target=self.source,
            source=self.target,
        )


class Graph:
    def __init__(self):
        self.vertices = []
        self.edges = []

    def add_vertex(self, vertex: Vertex):
        if vertex not in self.vertices:
            self.vertices.append(vertex)

    def contains_vertex(self, vertex: Vertex) -> bool:
        return vertex in self.vertices

    def add_edge(self, a: Vertex, b: Vertex):
        if a == b:
            raise ValueError(
                f"entered edge with only one vertex {a}, loops are not allowed"
            )
        normalized_edge = (a, b) if a < b else (b, a)
        if normalized_edge not in self.edges:
            self.edges.append(normalized_edge)

    def directed_edges(self) -> list[DirectedEdge]:
        return [DirectedEdge(source=edge[0], target=edge[1]) for edge in self.edges] + [
            DirectedEdge(source=edge[1], target=edge[0]) for edge in self.edges
        ]


class Alpha:
    def __init__(self):
        self._map: dict[DirectedEdge, Feature] = {}

    def map_directed_edge_to(self, directed_edge: DirectedEdge, feature: Feature):
        self._map[directed_edge] = feature
        assert directed_edge.inverse() not in self._map or self._map[
            directed_edge.inverse()
        ] == (
            feature[0],
            -feature[1],
        ), "opposite directions must be mapped to inverse features"

    def __call__(self, directed_edge: DirectedEdge) -> Feature:
        return self._map[directed_edge]

    def feature_id_of(self, edge: Edge) -> FeatureId:
        return self(DirectedEdge(source=edge[0], target=edge[1]))[0]


@dataclass(frozen=True)
class Location:
    """
    A node of a FeatureTree, corresponds to a location, the minimal features
    contained in the pre-tangles of the partitions contained in the FeatureTree.

    Attributes:
        features: the list of minimal features.
        vertex: The vertex of the (graph theoretical) tree corresponding to the location.
    """

    features: list[Feature]
    vertex: Vertex


class FeatureTree:
    def __init__(self, tree: Graph, alpha: Alpha):
        self.tree = tree
        self.alpha = alpha

    @staticmethod
    def from_nested_features(nested_feature_ids: np.ndarray, is_le: LessOrEqFunc) -> "FeatureTree":
        if not _are_features_nested(nested_feature_ids, is_le):
            raise ValueError("provided features are not nested")
        return _build_feature_tree(nested_feature_ids, is_le)

    def get_location_at_vertex(self, vertex: Vertex) -> Location:
        return Location(
            features=[
                self.alpha(directed_edge)
                for directed_edge in self.tree.directed_edges()
                if directed_edge.target == vertex
            ],
            vertex=vertex,
        )

def _are_features_nested(nested_feature_ids: np.ndarray, is_le: LessOrEqFunc) -> bool:
    pass

def _build_feature_tree(nested_feature_ids: np.ndarray, is_le: LessOrEqFunc) -> FeatureTree:
    pass

def _build_feature_tree_from_nested_features(
    efficient_distinguishers: np.ndarray,
    is_le: LessOrEqFunc,
) -> FeatureTree:
    _edges = list(efficient_distinguishers)
    _locations, _locations_of_edge = _find_locations(efficient_distinguishers, is_le)

    return FeatureTree(
        _edges=_edges,
        _locations=_locations,
        _locations_of_edge=_locations_of_edge,
    )


def _find_locations(nested_feature_ids: np.ndarray, is_le: LessOrEqFunc) -> FeatureTree:
    _locations = []
    _locations_of_edge: dict[
        FeatureId, tuple[Optional[Location], Optional[Location]]
    ] = {}

    all_features = [(feature_id, 1) for feature_id in nested_feature_ids] + [
        (feature_id, -1) for feature_id in nested_feature_ids
    ]

    for feature in all_features:
        if _is_feature_in_location_already(feature, _locations_of_edge):
            continue
        inverse_of_other_elements_in_location = _find_maximal_features_less_than(
            feature, all_features, is_le
        )
        location_features = [feature] + [
            (feature_id, -specification)
            for (feature_id, specification) in inverse_of_other_elements_in_location
        ]
        _locations.append(
            Location(
                features=location_features,
                node_idx=len(_locations),
            )
        )
        for feature_id, specification in location_features:
            if feature_id not in _locations_of_edge:
                _locations_of_edge[feature_id] = (None, None)
            if specification == 1:
                _locations_of_edge[feature_id] = (
                    _locations[-1],
                    _locations_of_edge[feature_id][1],
                )
            else:
                _locations_of_edge[feature_id] = (
                    _locations_of_edge[feature_id][0],
                    _locations[-1],
                )

    return _locations, _locations_of_edge


def _find_maximal_features_less_than(
    feature: Feature,
    all_features: list[Feature],
    is_le: LessOrEqFunc,
) -> list[Feature]:
    maximal_lesser_features = []

    for potential_feature in all_features:
        if potential_feature == feature:
            continue
        if not is_le(
            potential_feature[0], potential_feature[1], feature[0], feature[1]
        ):
            continue
        if any(
            [
                is_le(
                    potential_feature[0],
                    potential_feature[1],
                    current_feature[0],
                    current_feature[1],
                )
                for current_feature in maximal_lesser_features
            ]
        ):
            continue
        maximal_lesser_features = [
            current_feature
            for current_feature in maximal_lesser_features
            if not is_le(
                current_feature[0],
                current_feature[1],
                potential_feature[0],
                potential_feature[1],
            )
        ]
        maximal_lesser_features.append(potential_feature)

    return maximal_lesser_features


def _is_feature_in_location_already(
    feature: Feature,
    _locations_of_edge: dict[FeatureId, tuple[Optional[Location], Optional[Location]]],
) -> bool:
    if _locations_of_edge.get(feature[0]) is None:
        return False
    if feature[1] == 1 and _locations_of_edge.get(feature[0])[0] is None:
        return False
    if feature[1] == -1 and _locations_of_edge.get(feature[0])[1] is None:
        return False
    return True


def _are_efficient_distinguishers_nested(
    is_le: LessOrEqFunc,
    efficient_distinguishers: np.ndarray,
) -> bool:
    for i in range(len(efficient_distinguishers)):
        for j in range(i + 1, len(efficient_distinguishers)):
            if not _is_nested(
                efficient_distinguishers[i], efficient_distinguishers[j], is_le
            ):
                return False
    return True


def _is_nested(feature_1: FeatureId, feature_2: FeatureId, is_le: LessOrEqFunc) -> bool:
    return (
        is_le(feature_1, 1, feature_2, 1)
        or is_le(feature_1, -1, feature_2, 1)
        or is_le(feature_1, 1, feature_2, -1)
        or is_le(feature_1, -1, feature_2, -1)
    )
