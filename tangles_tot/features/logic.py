from typing import Optional
import warnings
import numpy as np
from tangles_tot._tangles_lib import MetaData


class TextTerm:
    """Class representing a text-based logical term with operations."""

    def __init__(self, text: str, _outer_operation: Optional[str] = None):
        """Initialize a TextTerm instance.

        Args:
            text: The text content of the term.
        """
        self._text = text
        self._outer_operation = "" if not _outer_operation else _outer_operation

    @staticmethod
    def build_from(source) -> "TextTerm":
        """Build a TextTerm from various possible source types.

        Can either be a string or TextTerm directly or a MetaData object.
        In case of source being a MetaData object the TextTerm represents
        the label for the positive orientation of the feature the MetaData
        describes.

        Args:
            source: Can be a TextTerm instance, string, or MetaData object.

        Returns:
            A TextTerm instance constructed from the source.

        Raises:
            ValueError: If source type is not supported.
        """
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
        """Combine two terms with logical AND operation.

        Args:
            other_term: The other TextTerm instance to combine with.

        Returns:
            A new TextTerm representing the AND combination.
        """
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
        """Combine two terms with logical OR operation.

        Args:
            other_term: The other TextTerm instance to combine with.

        Returns:
            A new TextTerm representing the OR combination.
        """
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
        """Negate the current term.

        Returns:
            A new TextTerm representing the negation.
        """
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
        """Return string representation of the term."""
        return self._text

    @staticmethod
    def true() -> "TextTerm":
        """Return a TextTerm representing 'true'.

        Returns:
            A TextTerm instance with text set to 'true'.
        """
        return TextTerm("true")

    @staticmethod
    def false() -> "TextTerm":
        """Return a TextTerm representing 'false'.

        Returns:
            A TextTerm instance with text set to 'false'.
        """
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
