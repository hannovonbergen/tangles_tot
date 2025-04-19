from .logic import Term, AND, OR, NOT


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
