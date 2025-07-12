import numpy as np
import pytest
from tangles_tot._testing import generate_random_features, generate_random_set_separations
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
    not_a = a.not_()
    b = Term.variable("b")
    not_b = b.not_()
    a_or_b = a.or_(b)
    true = Term.true()
    false = Term.false()
    assert a_or_b.not_() == not_a.and_(not_b)
    assert a.and_(true) == a
    assert a.and_(false) == Term.false()


@pytest.mark.parametrize(
    "feature_generation",
    [generate_random_features, generate_random_set_separations],
)
def test_reconstruct_logic_term_finds_input(feature_generation):
    num_features = 10
    feature_length = 100
    features = feature_generation(num_features, feature_length)
    for i in range(num_features):
        assert reconstruct_logic_term(features[:, i], features).term == str(i)


@pytest.mark.parametrize(
    "feature_generation",
    [generate_random_features, generate_random_set_separations],
)
def test_reconstruct_logic_term_true_input(feature_generation):
    num_features = 10
    feature_length = 100
    features = feature_generation(num_features, feature_length)
    for _ in range(num_features):
        assert (
            reconstruct_logic_term(
                np.ones(feature_length, dtype=np.int8), features
            ).term
            == TRUE
        )

@pytest.mark.parametrize(
    "feature_generation",
    [generate_random_features, generate_random_set_separations],
)
def test_reconstruct_logic_term_inverse(feature_generation):
    num_features = 10
    feature_length = 100
    features = feature_generation(num_features, feature_length)
    for i in range(num_features):
        assert reconstruct_logic_term(-features[:, i], features).term == (NOT, str(i))

@pytest.mark.parametrize(
    "feature_generation",
    [generate_random_features, generate_random_set_separations],
)
def test_impossible_reconstruction(feature_generation):
    num_features = 10
    feature_length = 100
    features = feature_generation(num_features, feature_length)
    try:
        reconstruct_logic_term(features[:, 9], features[:, :9])
    except:
        return
    assert False, "impossible feature was found"

@pytest.mark.parametrize(
    "feature_generation",
    [generate_random_features, generate_random_set_separations],
)
def test_intersection_and_union(feature_generation):
    num_features = 10
    feature_length = 100
    features = feature_generation(num_features, feature_length)
    feature_a = features[:, 0]
    feature_b = features[:, 1]
    feature_c = features[:, 2]
    a_and_b_or_c = np.minimum(feature_a, np.maximum(feature_b, feature_c))
    reconstructed_logic_term = reconstruct_logic_term(a_and_b_or_c, features)
    # TODO better method for checking if logic terms are identical
    assert reconstructed_logic_term.term == ('0', 'and', ('1', 'or', (('not', '1'), 'and', '2'))) or reconstructed_logic_term.term == ('0', 'and', ('2', 'or', (('not', '2'), 'and', '1')))
