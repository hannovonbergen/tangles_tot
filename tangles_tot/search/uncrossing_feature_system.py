from typing import Any, Optional, Union
import numpy as np
from tangles_tot._tangles_lib import FeatureSystem, INF_LABEL, MetaData
from tangles_tot._typing import Feature


class UncrossingFeatureSystem:
    """
    A class which implements the functionality of a feature system and is additionally
    specialised for use cases involving uncrossing.

    Currently it manages a list of feature ids of the features which were not added by
    corners, the "original ids".

    Still very unstable and will probably change a lot between releases.
    """

    def __init__(self, feat_sys: FeatureSystem, original_ids: list[int]):
        self._feat_sys = feat_sys
        self._original_ids = original_ids

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
        return len(self._feat_sys)

    def add_corner(
        self,
        feature_id_a: int,
        specification_a: int,
        feature_id_b: int,
        specification_b: int,
    ):
        self._feat_sys.add_corner(
            feature_id_a,
            specification_a,
            feature_id_b,
            specification_b,
        )

    def compute_infimum(
        self,
        feat_ids: Union[np.ndarray, list[int]],
        specifications: Union[np.ndarray, list[int]],
    ):
        return self._feat_sys.compute_infimum(feat_ids, specifications)

    def get_number_of_original_features(self) -> int:
        return len(self._original_ids)

    def get_original_features(self) -> np.ndarray:
        return self._feat_sys[self._original_ids]

    def get_feature(self, feature: Feature) -> np.ndarray:
        return self._feat_sys[feature[0]] * feature[1]

    def get_metadata_of_original_features(self) -> list[Any]:
        metadata_list = []
        for original_id in self._original_ids:
            metadata = self._feat_sys.feature_metadata(original_id)
            if not metadata or metadata.info is None:
                metadata_list.append(f"s{original_id}")
            else:
                metadata_list.append(metadata.info)
        return metadata_list

    def count_big_side(self, feature_id: int) -> int:
        return self._feat_sys.count_big_side(feature_id)

    def side_counts(self, feature_id: int) -> tuple[int, int]:
        return self._feat_sys.side_counts(feature_id)

    def feature_size(self, feature_id: int) -> int:
        return self._feat_sys.feature_size(feature_id)

    def feature_and_complement_size(self, feature_id: int) -> tuple[int, int]:
        return self._feat_sys.feature_and_complement_size(feature_id)

    def all_feature_ids(self) -> np.ndarray:
        return self._feat_sys.all_feature_ids()

    def get_feature_ids(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self._feat_sys.get_feature_ids(features)

    def is_nested(self, feature_id_1: int, feature_id_2: int) -> bool:
        return self._feat_sys.is_nested(feature_id_1, feature_id_2)

    def feature_metadata(
        self, feature_ids: Union[int, list, np.ndarray, None]
    ) -> MetaData:
        return self._feat_sys.feature_metadata(feature_ids)

    def is_le(
        self,
        feature_id_1: int,
        specification_1: int,
        feature_id_2: int,
        specification_2: int,
    ) -> bool:
        return self._feat_sys.is_le(
            feature_id_1, specification_1, feature_id_2, specification_2
        )

    def is_subset(
        self,
        feature_id_1: int,
        specification_1: int,
        feature_id_2: int,
        specification_2: int,
    ) -> bool:
        return self._feat_sys.is_subset(
            feature_id_1, specification_1, feature_id_2, specification_2
        )

    def add_features(
        self, features: np.ndarray, metadata: Optional[Any] = None
    ) -> tuple[np.ndarray, np.ndarray]:
        previous_length = len(self._feat_sys)
        result = self._feat_sys.add_features(features, metadata)
        new_length = len(self._feat_sys)
        self._original_ids = self._original_ids + list(
            range(previous_length, new_length)
        )
        return result

    def get_corners(
        self, feature_id_1: int, feature_id_2: int
    ) -> tuple[np.ndarray, np.ndarray]:
        return self._feat_sys.get_corners(feature_id_1, feature_id_2)

    def copy(self) -> "UncrossingFeatureSystem":
        return UncrossingFeatureSystem(
            feat_sys=self._feat_sys.copy(), original_ids=self._original_ids.copy()
        )
