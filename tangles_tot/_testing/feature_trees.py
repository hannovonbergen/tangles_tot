from tangles_tot.tree import FeatureTree, Location


def three_star() -> FeatureTree:
    edges = [0, 1, 2]

    locations = [
        Location(
            features=[(i, -1)],
            node_idx=i,
        )
        for i in range(3)
    ] + [
        Location(
            features=[
                (0, 1),
                (1, 1),
                (2, 1),
            ],
            node_idx=3,
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
        _edges=edges,
        _locations=locations,
        _locations_of_edge=locations_of_edge,
    )
