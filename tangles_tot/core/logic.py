from typing import Union, Any

AND = "and"
OR = "or"
NOT = "not"
TRUE = True
FALSE = False

TermType = Union[tuple, str, bool]


class Term:
    def __init__(self, term: TermType):
        """
        @private
        """
        self._term = term

    @property
    def term(self) -> TermType:
        return self._term

    def __repr__(self) -> str:
        return str(self.term)

    def and_(self, other: "Term") -> "Term":
        if self.term is TRUE or other.term is FALSE:
            return other
        if self.term is FALSE or other.term is TRUE:
            return self
        return Term((self.term, AND, other.term))

    def or_(self, other: "Term") -> "Term":
        if self.term is TRUE or other.term is FALSE:
            return self
        if self.term is FALSE or other.term is TRUE:
            return other
        return Term((self.term, OR, other.term))

    def not_(self) -> "Term":
        if self.term is TRUE:
            return Term(FALSE)
        if self.term is FALSE:
            return Term(TRUE)
        if not isinstance(self.term, tuple):
            return Term((NOT, self.term))
        assert isinstance(self.term, tuple)
        if len(self.term) == 2:
            assert self.term[0] is NOT
            return Term(self.term[1])
        assert len(self.term) == 3
        assert self.term[1] in [AND, OR]
        return Term(
            (
                Term(self.term[0]).not_(),
                OR if self.term[1] is AND else AND,
                Term(self.term[2]).not_(),
            )
        )

    @staticmethod
    def true() -> "Term":
        return Term(TRUE)

    @staticmethod
    def false() -> "Term":
        return Term(FALSE)

    @staticmethod
    def variable(label: Any) -> "Term":
        if label in [TRUE, FALSE] or str(label) in [str(True), str(False)]:
            raise ValueError(
                f"{label} is used internally and is not a valid variable name"
            )
        return Term(label)
