from typing import Union, Any
import numpy as np

AND = "and"
OR = "or"
NOT = "not"
TRUE = "True"
FALSE = "False"

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
        if str(label) in [str(True), str(False)]:
            raise ValueError(
                f"{label} is used internally and is not a valid variable name"
            )
        return Term(label)


def reconstruct_logic_term(vector: np.ndarray, original_vectors: np.ndarray) -> Term:
    return _array_to_term_recursive(
        approximation_term=Term.true(),
        approximation=np.ones(original_vectors.shape[0], dtype=np.int8),
        original_vectors=original_vectors,
        target=vector,
    )


def _array_to_term_recursive(
    approximation_term: Term,
    approximation: np.ndarray,
    original_vectors: np.ndarray,
    target: np.ndarray,
) -> Term:
    next_approximation = np.minimum(target, approximation)

    if np.all(approximation == target):
        return approximation_term
    if np.all(next_approximation == -1):
        return Term.false()

    approximation_index = _find_best_term_extension(
        approximation=approximation,
        original_vectors=original_vectors,
        target=target,
    )

    first_term = _array_to_term_recursive(
        approximation_term=Term.variable(approximation_index),
        approximation=original_vectors[:, approximation_index],
        original_vectors=original_vectors,
        target=target,
    )
    second_term = _array_to_term_recursive(
        approximation_term=Term.variable(approximation_index).not_(),
        approximation=-original_vectors[:, approximation_index],
        original_vectors=original_vectors,
        target=target,
    )

    return approximation_term.and_(first_term.or_(second_term))


def _find_best_term_extension(
    approximation: np.ndarray,
    original_vectors: np.ndarray,
    target: np.ndarray,
) -> np.intp:
    mask_ab = np.minimum(approximation, np.maximum(-target, -approximation)) == 1
    mask_cd = np.minimum(approximation, target) == 1
    a_ar = np.sum(original_vectors[mask_ab] == 1, axis=0)
    b_ar = np.sum(original_vectors[mask_ab] == 1, axis=0)
    c_ar = np.sum(original_vectors[mask_cd] == 1, axis=0)
    d_ar = np.sum(original_vectors[mask_cd] == 1, axis=0)

    nested_bias = np.maximum(a_ar * (c_ar == 0), b_ar * (d_ar == 0))
    if np.any(nested_bias) > 0:
        return np.argmax(nested_bias)
    scores_1 = np.zeros(original_vectors.shape[1], dtype=np.float64)
    scores_1[c_ar != 0] = a_ar[c_ar != 0] / c_ar[c_ar != 0]
    scores_2 = np.zeros(original_vectors.shape[1], dtype=np.float64)
    scores_2[d_ar != 0] = b_ar[d_ar != 0] / d_ar[d_ar != 0]
    return np.argmax(np.maximum(scores_1, scores_2))
