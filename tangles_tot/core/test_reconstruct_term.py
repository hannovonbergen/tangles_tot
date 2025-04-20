import numpy as np
import pytest
from tangles_tot._testing import generate_random_features
from .reconstruct_term import Term, AND, OR, NOT, TRUE, FALSE, reconstruct_logic_term


def test_text_term_constants():
    assert str(Term.true()) == "True"
    assert str(Term.false()) == "False"


def test_variables():
    assert str(Term.variable(1)) == "1"
    assert str(Term.variable("test")) == "test"


def test_invalid_variable():
    try:
        Term.variable(True)
    except:
        return
    assert False, "did not raise exception"


def test_confusing_variable():
    try:
        Term.variable("True")
    except:
        return
    assert False, "did not raise exception"


def test_text_term_and():
    a = Term.variable("a")
    b = Term.variable("b")
    c = Term.variable("c")
    a_or_b = a.or_(b)
    true = Term.true()
    false = Term.false()
    assert a.and_(b).term == ("a", AND, "b")
    assert str(a.and_(true)) == "a"
    assert str(a.and_(false)) == "False"


@pytest.mark.parametrize(
    "feature_generation",
    [generate_random_features],  # generate_random_set_separations],
)
def test_reconstruct_logic_term_finds_input(feature_generation):
    num_features = 10
    feature_length = 100
    features = feature_generation(num_features, feature_length)
    for i in range(num_features):
        assert reconstruct_logic_term(features[:, i], features).term == i


def test_reconstruct_logic_term_true_input():
    num_features = 10
    feature_length = 100
    features = generate_random_features(num_features, feature_length)
    for _ in range(num_features):
        assert (
            reconstruct_logic_term(
                np.ones(feature_length, dtype=np.int8), features
            ).term
            == TRUE
        )


def test_reconstruct_logic_term_inverse():
    num_features = 10
    feature_length = 100
    features = generate_random_features(num_features, feature_length)
    for i in range(num_features):
        assert reconstruct_logic_term(-features[:, i], features).term == (NOT, i)
