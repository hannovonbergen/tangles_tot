from dataclasses import dataclass
import numpy as np
from tangles.separations.system import MetaData
from tangles_tot.core import FeatureSystem, SetSeparationSystem

CUSTOM_LABEL = "custom"
INF_LABEL = "inf"

@dataclass
class OriginalFeatures:
    sep_ids: list[int]

    @staticmethod
    def from_feature_system(feat_sys: FeatureSystem) -> "OriginalFeatures":
        original_ids = []
        corner_ids = []
        for i in range(len(feat_sys)):
            metadata = feat_sys.feature_metadata(i)
            assert isinstance(metadata, MetaData)
            while metadata:
                if metadata.type == CUSTOM_LABEL:
                    original_ids.append(i)
                    break
                if metadata.type == INF_LABEL:
                    corner_ids.append(i)
                metadata = metadata.next
        if len(original_ids) == 0 and len(corner_ids) == 0:
            original_ids = list(range(len(feat_sys)))
        if len(original_ids) == 0:
            raise Exception(
                "could not determine which features were added by uncrossing and which were added by user. You can fix this by adding metadata to your features."
            )
        return OriginalFeatures(
            sep_ids=original_ids,
        )

    @staticmethod
    def from_set_separation_system(
        sep_sys: SetSeparationSystem,
    ) -> "OriginalFeatures":
        original_ids = []
        corner_ids = []
        for i in range(len(sep_sys)):
            metadata = sep_sys.separation_metadata(i)
            while metadata:
                assert isinstance(metadata, MetaData)
                if metadata.type == CUSTOM_LABEL:
                    original_ids.append(i)
                    break
                if metadata.type == INF_LABEL:
                    corner_ids.append(i)
                metadata = metadata.next
        if len(original_ids) == 0 and len(corner_ids) == 0:
            original_ids = list(range(len(sep_sys)))
        if len(original_ids) == 0:
            raise Exception(
                "could not determine which features were added by uncrossing and which were added by user. You can fix this by adding metadata to your features."
            )
        return OriginalFeatures(
            sep_ids=original_ids,
        )
