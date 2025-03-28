import pytest
import numpy as np
from tangles_tot._testing import (
    generate_random_features,
    generate_random_set_separations,
    add_random_corners_to_feat_sys,
)
from tangles_tot._tangles_lib import FeatureSystem, SetSeparationSystem
from tangles_tot.search import UncrossingFeatureSystem
from .interpret_corner import interpret_feature_array, interpret_feature


@pytest.mark.parametrize(
    "feature_generation",
    [generate_random_features] #generate_random_set_separations],
)
def test_interpret_feature_array_finds_input(feature_generation):
    num_features = 10
    feature_length = 100
    features = feature_generation(num_features, feature_length)
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


def test_interpret_feature_array_condition():
    num_features = 10
    feature_length = 100
    features = generate_random_features(num_features, feature_length)
    metadata = ["a", "b"] + [f"s{i}" for i in range(2, num_features)]
    a = features[:, 0]
    b = features[:, 1]
    a_and_b = np.minimum(a, b)
    a_or_b = np.maximum(a, b)
    assert (
        str(interpret_feature_array(a_and_b, features, metadata, under_condition=a))
        == "b"
    )
    assert str(interpret_feature_array(a_or_b, features, metadata, -b)) == "a"


feature_systems = [FeatureSystem, UncrossingFeatureSystem]


@pytest.mark.parametrize("FeatSys", feature_systems)
def test_interpret_feature_original_features(FeatSys):
    num_features = 10
    feature_length = 100
    number_of_corners = 100
    features = generate_random_features(num_features, feature_length)
    metadata = [str(i) for i in range(num_features)]
    feat_sys = FeatSys.with_array(features, metadata=metadata)
    add_random_corners_to_feat_sys(feat_sys, number_of_corners)
    for i in range(num_features):
        assert str(interpret_feature((i, 1), feat_sys)) == f"{i}"
        assert str(interpret_feature((i, -1), feat_sys)) == f"¬{i}"


@pytest.mark.parametrize("FeatSys", feature_systems)
def test_interpret_feature_under_condition(FeatSys):
    num_features = 10
    feature_length = 100
    number_of_corners = 100
    features = generate_random_features(num_features, feature_length)
    metadata = ["a", "b"] + [str(i) for i in range(2, num_features)]
    feat_sys = FeatSys.with_array(features, metadata=metadata)
    add_random_corners_to_feat_sys(feat_sys, number_of_corners)
    id_array, spec_array = feat_sys.get_corners(0, 1)
    a_or_b = (id_array[0], -spec_array[0])
    a_and_b = (id_array[3], spec_array[3])
    assert str(interpret_feature(a_and_b, feat_sys, under_condition=[(0, 1)])) == "b"
    assert str(interpret_feature(a_or_b, feat_sys, under_condition=[(1, -1)])) == "a"


@pytest.mark.parametrize("FeatSys", feature_systems)
def test_interpret_feature_under_empty_condition(FeatSys):
    num_features = 10
    feature_length = 100
    number_of_corners = 100
    features = generate_random_features(num_features, feature_length)
    metadata = ["a", "b"] + [str(i) for i in range(2, num_features)]
    feat_sys = FeatSys.with_array(features, metadata=metadata)
    add_random_corners_to_feat_sys(feat_sys, number_of_corners)
    assert str(interpret_feature((0, 1), feat_sys, under_condition=[])) == "a"
    assert str(interpret_feature((1, 1), feat_sys, under_condition=[])) == "b"


@pytest.mark.parametrize("FeatSys", feature_systems)
def test_interpret_feature_under_weird_conditions(FeatSys):
    num_features = 10
    feature_length = 100
    number_of_corners = 100
    features = generate_random_features(num_features, feature_length)
    metadata = ["a", "b"] + [str(i) for i in range(2, num_features)]
    feat_sys = FeatSys.with_array(features, metadata=metadata)
    add_random_corners_to_feat_sys(feat_sys, number_of_corners)
    assert str(interpret_feature((1, 1), feat_sys, under_condition=[(1, 1)])) == "true"
    assert (
        str(interpret_feature((1, 1), feat_sys, under_condition=[(1, -1)])) == "false"
    )
    assert (
        str(interpret_feature((0, 1), feat_sys, under_condition=[(0, 1), (0, -1)]))
        == "true"
    )
    assert (
        str(interpret_feature((0, 1), feat_sys, under_condition=[(0, 1), (1, 1)]))
        == "true"
    )
