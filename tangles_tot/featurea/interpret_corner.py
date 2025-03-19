from typing import Union, Optional
import warnings
import numpy as np
from tangles_tot._tangles_lib import MetaData, Feature, CUSTOM_LABEL
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
