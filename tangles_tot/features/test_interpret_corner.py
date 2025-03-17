import numpy as np
from tangles_tot._testing import generate_random_features
from .interpret_corner import interpret_feature_array


def test_interpret_feature_array_finds_input():
    num_features = 10
    feature_length = 100
    features = generate_random_features(num_features, feature_length)
    metadata = [str(i) for i in range(num_features)]
    for i in range(num_features):
        assert str(interpret_feature_array(features[:, i], features, metadata)) == str(
            i
        )


def test_interpret_feature_array_true_input():
    num_features = 10
    feature_length = 100
    features = generate_random_features(num_features, feature_length)
    metadata = [str(i) for i in range(num_features)]
    for _ in range(num_features):
        assert (
            str(
                interpret_feature_array(
                    np.ones(feature_length, dtype=np.int8), features, metadata
                )
            )
            == "true"
        )


def test_interpret_feature_array_inverse():
    num_features = 10
    feature_length = 100
    features = generate_random_features(num_features, feature_length)
    metadata = [str(i) for i in range(num_features)]
    for i in range(num_features):
        assert (
            str(interpret_feature_array(-features[:, i], features, metadata))
            == f"¬{str(i)}"
        )
