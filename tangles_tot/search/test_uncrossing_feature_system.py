import numpy as np
from tangles_tot._tangles_lib import FeatureSystem
from .uncrossing_feature_system import UncrossingFeatureSystem

SPECIFICATION_SET = np.array([1, -1], dtype=np.int8)


def test_creation_with_array_not_unique():
    features = np.array(
        [
            [1, -1, -1],
            [1, 1, -1],
            [-1, -1, 1],
        ]
    )
    metadata = ["a", "b", "c"]
    feat_sys = UncrossingFeatureSystem.with_array(features, metadata)
    assert len(feat_sys) == 2


def _generate_random_features(
    number_of_features: int, length_of_features: int
) -> np.ndarray:
    return np.random.choice(
        SPECIFICATION_SET, size=(length_of_features, number_of_features)
    )


def _add_random_corners_to_feat_sys(feat_sys: FeatureSystem, number_of_corners: int):
    for _ in range(number_of_corners):
        length_of_feature_system = len(feat_sys)
        feat_sys.add_corner(
            np.random.randint(length_of_feature_system),
            np.random.choice(SPECIFICATION_SET),
            np.random.randint(length_of_feature_system),
            np.random.choice(SPECIFICATION_SET),
        )


def test_creation_with_array():
    num_features = 10
    feature_length = 100
    while True:
        features = _generate_random_features(num_features, feature_length)
        metadata = list(range(num_features))
        feat_sys = UncrossingFeatureSystem.with_array(features, metadata)
        if len(feat_sys) == num_features:
            break
    assert feat_sys.original_ids == list(range(num_features))
    assert len(feat_sys.feat_sys) == num_features


def test_creation_with_feature_system():
    num_features = 10
    feature_length = 100
    number_of_corners_to_add = 100
    while True:
        original_features = _generate_random_features(num_features, feature_length)
        metadata = list(range(num_features))
        feat_sys = FeatureSystem.with_array(original_features, metadata=metadata)
        if len(feat_sys) == num_features:
            break

    _add_random_corners_to_feat_sys(feat_sys, number_of_corners_to_add)

    uncrossing_feat_sys = UncrossingFeatureSystem.from_feature_system(feat_sys)
    assert len(uncrossing_feat_sys) == len(feat_sys)
    assert uncrossing_feat_sys.get_number_of_original_features() == num_features
    assert uncrossing_feat_sys.original_ids == list(range(num_features))


def test_creation_with_feature_system_no_metadata():
    num_features = 10
    feature_length = 100
    number_of_corners_to_add = 100
    while True:
        original_features = _generate_random_features(num_features, feature_length)
        feat_sys = FeatureSystem.with_array(original_features)
        if len(feat_sys) == num_features:
            break

    _add_random_corners_to_feat_sys(feat_sys, number_of_corners_to_add)

    uncrossing_feat_sys = UncrossingFeatureSystem.from_feature_system(feat_sys)
    assert len(uncrossing_feat_sys) == len(feat_sys)
    assert uncrossing_feat_sys.get_number_of_original_features() == num_features
    assert uncrossing_feat_sys.original_ids == list(range(num_features))


def test_get_original_features():
    num_features = 10
    feature_length = 100
    number_of_corners_to_add = 100
    original_features = _generate_random_features(num_features, feature_length)
    feat_sys = UncrossingFeatureSystem.with_array(original_features)

    _add_random_corners_to_feat_sys(feat_sys, number_of_corners_to_add)

    assert np.all(
        feat_sys.get_original_features() == original_features[:, feat_sys.original_ids]
    )


def test_get_metadata_of_original_features_no_metadata():
    num_features = 10
    feature_length = 100
    number_of_corners_to_add = 100
    original_features = _generate_random_features(num_features, feature_length)
    feat_sys = UncrossingFeatureSystem.with_array(original_features)
    expected_metadata = [f"s{i}" for i in range(len(feat_sys))]

    _add_random_corners_to_feat_sys(feat_sys, number_of_corners_to_add)

    assert feat_sys.get_metadata_of_original_features() == expected_metadata


def test_get_metadata_of_original_features():
    num_features = 10
    feature_length = 100
    number_of_corners_to_add = 100
    original_features = _generate_random_features(num_features, feature_length)
    expected_metadata = [f"test{i}" for i in range(num_features)]
    feat_sys = UncrossingFeatureSystem.with_array(
        original_features, metadata=expected_metadata
    )

    _add_random_corners_to_feat_sys(feat_sys, number_of_corners_to_add)

    assert feat_sys.get_metadata_of_original_features() == expected_metadata
