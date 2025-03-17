import numpy as np
from tangles_tot._tangles_lib import FeatureSystem

SPECIFICATION_SET = np.array([1, -1], dtype=np.int8)


def generate_random_features(
    number_of_features: int, length_of_features: int
) -> np.ndarray:
    return np.random.choice(
        SPECIFICATION_SET, size=(length_of_features, number_of_features)
    )


def add_random_corners_to_feat_sys(feat_sys: FeatureSystem, number_of_corners: int):
    for _ in range(number_of_corners):
        length_of_feature_system = len(feat_sys)
        feat_sys.add_corner(
            np.random.randint(length_of_feature_system),
            np.random.choice(SPECIFICATION_SET),
            np.random.randint(length_of_feature_system),
            np.random.choice(SPECIFICATION_SET),
        )
