import pytest
import numpy as np
from tangles_tot._tangles_lib import TangleSweep, LessOrEqFunc
from tangles_tot._typing import FeatureId, Specification
from .build_tot import (
    build_tree_of_tangles_from_sweep,
    _build_feature_tree_from_nested_features,
)


def test_build_tot_from_sweep_checks_for_tangle_sweep_type():
    try:
        build_tree_of_tangles_from_sweep(10)
    except ValueError:
        return
    assert False, "invalid tangle_sweep type did not raise exception"


@pytest.fixture
def mock_tangle_sweep_with_limit_ten() -> TangleSweep:
    mock_agreement_function = lambda _: 0
    mock_agreement_function.max_value = 10  # must be set for an agreement function
    mock_tangle_sweep = TangleSweep(mock_agreement_function, None, [0])
    return mock_tangle_sweep


@pytest.fixture
def mock_tangle_sweep_not_uncrossed() -> TangleSweep:
    mock_agreement_function = lambda _: 0
    mock_agreement_function.max_value = 10
    is_le = lambda _a, _b, _c, _d: False
    mock_tangle_sweep = TangleSweep(mock_agreement_function, is_le, [0])

    def mock_get_efficient_distinguishers(
        agreement: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        return None, np.array([1, 3, 5])

    mock_tangle_sweep.tree.get_efficient_distinguishers = (
        mock_get_efficient_distinguishers
    )
    return mock_tangle_sweep


def test_build_tot_from_sweep_agreement_value_too_low(
    mock_tangle_sweep_with_limit_ten: TangleSweep,
):
    assert mock_tangle_sweep_with_limit_ten.tree.limit == 10, "test setup failed"
    try:
        build_tree_of_tangles_from_sweep(
            tangle_sweep=mock_tangle_sweep_with_limit_ten,
            agreement_value=9,
        )
    except ValueError:
        return
    assert False, "invalid agreement_value parameter did not raise exception"


def test_build_tot_from_sweep_not_uncrossed(
    mock_tangle_sweep_not_uncrossed: TangleSweep,
):
    try:
        build_tree_of_tangles_from_sweep(
            tangle_sweep=mock_tangle_sweep_not_uncrossed,
        )
    except ValueError:
        return
    assert False, "efficient distinguishers not being uncrossed was not caught"


@pytest.fixture
def is_le_for_three_star() -> LessOrEqFunc:
    def is_le(
        feature_a: FeatureId,
        specification_a: Specification,
        feature_b: FeatureId,
        specification_b: Specification,
    ) -> bool:
        if feature_a == feature_b and specification_a == specification_b:
            return True
        if feature_a != feature_b and specification_a == -1 and specification_b == 1:
            return True
        return False

    return is_le


def test_build_from_nested(is_le_for_three_star: LessOrEqFunc):
    feature_tree = _build_feature_tree_from_nested_features(
        efficient_distinguishers=np.array([0, 1, 2]),
        is_le=is_le_for_three_star,
    )
    assert len(feature_tree.edges()) == 3
    assert len(feature_tree.locations()) == 4
    assert feature_tree.get_node_idx_of_location_containing(
        (0, 1)
    ) == feature_tree.get_node_idx_of_location_containing((1, 1))
    assert feature_tree.get_node_idx_of_location_containing(
        (1, 1)
    ) == feature_tree.get_node_idx_of_location_containing((2, 1))
