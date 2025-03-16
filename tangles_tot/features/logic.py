from typing import Optional
import warnings
import numpy as np
from tangles_tot._tangles_lib import MetaData


class TextTerm:
    def __init__(self, text: str, _outer_operation: Optional[str] = None):
        self._text = text
        self._outer_operation = "" if not _outer_operation else _outer_operation

    @staticmethod
    def build_from(source) -> "TextTerm":
        if isinstance(source, TextTerm):
            return source
        if isinstance(source, str):
            return TextTerm(text=source)
        if isinstance(source, MetaData):
            if source.next is not None:
                warnings.warn(
                    f"metadata object has other options, which were associated to the same feature. Using {source.info} and not {source.next.info} for the logic terms."
                )
            return (
                TextTerm(text=str(source.info))
                if source.orientation != -1
                else TextTerm(text=str(source.info)).not_()
            )
        raise ValueError(f"Cannot build a text term from feature metadata {source}.")

    def and_(self, other_term: "TextTerm") -> "TextTerm":
        if self._text == "true":
            return other_term
        if other_term._text == "true":
            return self

        if self._text == "false":
            return self
        if other_term._text == "false":
            return other_term

        first_term_text = (
            self._text if self._outer_operation != "or" else f"({self._text})"
        )
        second_term_text = (
            other_term._text
            if other_term._outer_operation != "or"
            else f"({other_term._text})"
        )
        return TextTerm(
            text=f"{first_term_text} ∧ {second_term_text}",
            _outer_operation="and",
        )

    def or_(self, other_term: "TextTerm") -> "TextTerm":
        if self._text == "false":
            return other_term
        if other_term._text == "false":
            return self

        if self._text == "true":
            return self
        if other_term._text == "true":
            return other_term

        first_term_text = (
            self._text if self._outer_operation != "and" else f"({self._text})"
        )
        second_term_text = (
            other_term._text
            if other_term._outer_operation != "and"
            else f"({other_term._text})"
        )
        return TextTerm(
            text=f"{first_term_text} ∨ {second_term_text}",
            _outer_operation="or",
        )

    def not_(self) -> "TextTerm":
        if self._text == "true":
            return "false"
        if self._text == "false":
            return "true"
        if self._text[0] == "¬":
            new_text = self._text[1:]
            if new_text[0] == "(" and new_text[-1] == ")":
                new_text = new_text[1:-1]
            return TextTerm(new_text, _outer_operation=self._outer_operation)
        if self._outer_operation == "":
            return TextTerm(f"¬{self._text}")
        return TextTerm(f"¬({self._text})", _outer_operation=self._outer_operation)

    def __repr__(self) -> str:
        return self._text

    @staticmethod
    def true() -> "TextTerm":
        return TextTerm("true")

    @staticmethod
    def false() -> "TextTerm":
        return TextTerm("false")


class _SemanticTextTerm:
    def __init__(self, text: TextTerm, array: np.ndarray):
        self.text = text
        self.array = array

    def and_(self, other_term: "_SemanticTextTerm") -> "_SemanticTextTerm":
        return _SemanticTextTerm(
            text=self.text.and_(other_term.text),
            array=np.minimum(self.array, other_term.array),
        )

    def or_(self, other_term: "_SemanticTextTerm") -> "_SemanticTextTerm":
        return _SemanticTextTerm(
            text=self.text.or_(other_term.text),
            array=np.maximum(self.array, other_term.array),
        )

    def not_(self) -> "_SemanticTextTerm":
        return _SemanticTextTerm(text=self.text.not_(), array=-self.array)

    @staticmethod
    def true(n: int) -> "_SemanticTextTerm":
        return _SemanticTextTerm(text=TextTerm.true(), array=np.ones(n, dtype=np.int8))

    @staticmethod
    def false(n: int) -> "_SemanticTextTerm":
        return _SemanticTextTerm(
            text=TextTerm.false(), array=-np.ones(n, dtype=np.int8)
        )
