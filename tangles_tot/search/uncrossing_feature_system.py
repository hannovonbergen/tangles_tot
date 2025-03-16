from typing import Any, Optional
import numpy as np
from tangles_tot._tangles_lib import FeatureSystem, INF_LABEL


class UncrossingFeatureSystem:
    def __init__(self, feat_sys: FeatureSystem, original_ids: list[int]):
        self.feat_sys = feat_sys
        self.original_ids = original_ids

    @staticmethod
    def with_array(
        features: np.ndarray, metadata: Optional[Any] = None
    ) -> "UncrossingFeatureSystem":
        feat_sys = FeatureSystem.with_array(features, metadata=metadata)
        return UncrossingFeatureSystem(
            feat_sys=feat_sys, original_ids=list(range(len(feat_sys)))
        )

    @staticmethod
    def from_feature_system(feat_sys: FeatureSystem) -> "UncrossingFeatureSystem":
        original_ids = []
        for i in range(len(feat_sys)):
            metadata = feat_sys.feature_metadata(i)
            while metadata:
                if metadata.type == INF_LABEL:
                    break
                metadata = metadata.next
            if not metadata:
                original_ids.append(i)
        return UncrossingFeatureSystem(
            feat_sys=feat_sys,
            original_ids=original_ids,
        )

    def __len__(self) -> int:
        return len(self.feat_sys)

    def add_corner(
        self,
        feature_id_a: int,
        specification_a: int,
        feature_id_b: int,
        specification_b: int,
    ):
        self.feat_sys.add_corner(
            feature_id_a,
            specification_a,
            feature_id_b,
            specification_b,
        )

    def get_number_of_original_features(self) -> int:
        return len(self.original_ids)

    def get_original_features(self) -> np.ndarray:
        return self.feat_sys[self.original_ids]

    def get_metadata_of_original_features(self) -> list[Any]:
        metadata_list = []
        for original_id in self.original_ids:
            metadata = self.feat_sys.feature_metadata(original_id)
            if not metadata or metadata.info is None:
                metadata_list.append(f"s{original_id}")
            else:
                metadata_list.append(metadata.info)
        return metadata_list
