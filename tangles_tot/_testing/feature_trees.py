from tangles_tot.tree import FeatureTree, FeatureEdge, Location


def three_star(oriented: bool) -> FeatureTree:
    edges = {
        i: FeatureEdge(
            feature_id=i, specification=1 if oriented else None, label=f"feature {i}"
        )
        for i in range(3)
    }

    locations = [
        Location(
            features=[(i, -1)], associated_tangle=i, node_idx=i, label=f"tangle {i}"
        )
        for i in range(3)
    ] + [
        Location(
            features=[(0, 1), (1, 1), (2, 1)],
            associated_tangle=3,
            node_idx=3,
            label="tangle 3",
        )
    ]

    locations_of_edge = {
        i: (
            locations[3],
            locations[i],
        )
        for i in range(3)
    }

    return FeatureTree(
        _edges=edges, _locations=locations, _locations_of_edge=locations_of_edge
    )
