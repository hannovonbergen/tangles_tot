from typing import Union, Optional
import warnings
import numpy as np
from tangles_tot._tangles_lib import MetaData, Feature, FeatureSystem
from tangles_tot.search import UncrossingFeatureSystem
from .logic import TextTerm

MetaDataType = Union[str, TextTerm, MetaData]
FeatureArray = np.ndarray


def interpret_feature_array(
    feature: np.ndarray,
    original_features: np.ndarray,
    metadata: list[MetaDataType],
    under_condition: Optional[FeatureArray],
) -> TextTerm:
    pass


def interpret_feature(
    feature: Feature,
    feat_sys: Union[FeatureSystem, UncrossingFeatureSystem],
    under_condition: list[Feature],
) -> TextTerm:
    pass
