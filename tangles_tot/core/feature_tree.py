from dataclasses import dataclass
from .typing import Feature, FeatureId

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

    def get_location_at_vertex(self, vertex: Vertex) -> Location:
        return Location(
            features=[
                self.alpha(directed_edge)
                for directed_edge in self.tree.directed_edges()
                if directed_edge.target == vertex
            ],
            vertex=vertex,
        )
