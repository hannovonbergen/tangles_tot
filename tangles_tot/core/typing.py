from typing import Union, Literal

FeatureId = int

Specification = Union[Literal[1], Literal[-1]]

Feature = tuple[FeatureId, Specification]

TangleId = Union[str, int]
