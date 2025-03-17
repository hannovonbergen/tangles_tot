from .logic import TextTerm


def test_text_term_constants():
    assert str(TextTerm.true()) == "true"
    assert str(TextTerm.false()) == "false"


def test_text_term_string_constructor():
    assert str(TextTerm.build_from("test")) == "test"


class TextTermTest(TextTerm):
    def __init__(self):
        super().__init__("test")


def test_text_term_subclass_constructor():
    assert str(TextTerm.build_from(TextTermTest())) == "test"


def test_text_term_invalid_constructor():
    try:
        TextTerm.build_from(0)
    except ValueError:
        assert True
        return
    assert False, "invalid input did not cause a value error"


def test_text_term_and():
    a = TextTerm("a")
    b = TextTerm("b")
    c = TextTerm("c")
    a_or_b = a.or_(b)
    true = TextTerm.true()
    false = TextTerm.false()
    assert str(a.and_(b)) == "a ∧ b"
    assert str(a.and_(b).and_(c)) == "a ∧ b ∧ c"
    assert str(a.and_(true)) == "a"
    assert str(a.and_(false)) == "false"
    assert str(a.and_(a_or_b)) == f"a ∧ ({a_or_b})"


def test_text_term_or():
    a = TextTerm("a")
    b = TextTerm("b")
    c = TextTerm("c")
    a_and_b = a.and_(b)
    true = TextTerm.true()
    false = TextTerm.false()
    assert str(a.or_(b)) == "a ∨ b"
    assert str(a.or_(b).or_(c)) == "a ∨ b ∨ c"
    assert str(a.or_(true)) == "true"
    assert str(a.or_(false)) == "a"
    assert str(a.or_(a_and_b)) == f"a ∨ ({a_and_b})"


def test_text_term_not():
    a = TextTerm("a")
    b = TextTerm("b")
    c = TextTerm("c")
    a_and_b = a.and_(b)
    a_and_b_or_c = (a.and_(b)).or_(c)
    true = TextTerm.true()
    false = TextTerm.false()
    assert str(true.not_()) == "false"
    assert str(false.not_()) == "true"
    assert str(a.not_()) == "¬a"
    assert str(a.not_().not_()) == "a"
    assert str(a_and_b.not_()) == f"¬({a_and_b})"
    assert str(a_and_b.not_().not_()) == str(a_and_b)
    assert str(a_and_b_or_c.not_().not_()) == str(a_and_b_or_c)


from tangles_tot._tangles_lib import MetaData


def test_text_term_from_metadata():
    assert str(TextTerm.build_from(MetaData("test"))) == "test"
    assert str(TextTerm.build_from(MetaData("test", orientation=1))) == "test"
    assert str(TextTerm.build_from(MetaData("test", orientation=-1))) == "¬test"
