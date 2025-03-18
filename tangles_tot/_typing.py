from enum import Enum
from typing import Union

FeatureId = int


class Specification(Enum):
    DEFAULT = 1
    INVERSE = -1


Feature = tuple[FeatureId, Specification]

TangleId = Union[str, int]
