from tangles_tot.core import FeatureTree, Graph, Alpha, DirectedEdge


def three_star() -> FeatureTree:
    tree = Graph()
    alpha = Alpha()
    for i in range(4):
        tree.add_vertex(i)
    for i in range(3):
        tree.add_edge(i, 3)
        alpha.map_directed_edge_to(DirectedEdge(source=i, target=3), (i, 1))
        alpha.map_directed_edge_to(DirectedEdge(source=3, target=i), (i, -1))

    return FeatureTree(
        tree=tree,
        alpha=alpha,
    )
