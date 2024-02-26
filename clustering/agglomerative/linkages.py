from abc import ABC, abstractmethod

from enum import auto, StrEnum
from typing import Type

import numpy as np
import pandas as pd


class LinkageMethodName(StrEnum):
    WARD: str = auto()
    SINGLE: str = auto()
    COMPLETE: str = auto()
    AVERAGE: str = auto()
    CENTROID: str = auto()


class LinkageMethod(ABC):
    @property
    @abstractmethod
    def name(self):
        pass

    @abstractmethod
    def linkage(self, cluster_1_data: pd.DataFrame, cluster_2_data: pd.DataFrame) -> float:
        pass


class WardLinkage(LinkageMethod):
    name = LinkageMethodName.WARD

    def linkage(self, cluster_1_data: pd.DataFrame, cluster_2_data: pd.DataFrame) -> float:
        centroid_1 = np.mean(cluster_1_data, axis=0)
        centroid_2 = np.mean(cluster_2_data, axis=0)

        dist_cluster_1 = np.sqrt(((cluster_1_data - centroid_1) ** 2).sum(axis=1))
        dist_cluster_2 = np.sqrt(((cluster_2_data - centroid_2) ** 2).sum(axis=1))

        merged_cluster = pd.concat([cluster_1_data, cluster_2_data])
        merged_centroid = np.mean(merged_cluster, axis=0)
        dist_merged = np.sqrt(((merged_cluster - merged_centroid) ** 2).sum(axis=1))

        return dist_merged.sum() - (dist_cluster_1.sum() + dist_cluster_2.sum())


class SingleLinkage(LinkageMethod):
    name = LinkageMethodName.SINGLE

    def linkage(self, cluster_1_data: pd.DataFrame, cluster_2_data: pd.DataFrame) -> float:
        distances = np.sqrt(((cluster_1_data.to_numpy()[:, np.newaxis] - cluster_2_data.to_numpy()) ** 2).sum(axis=2))
        return distances.min()


class CompleteLinkage(LinkageMethod):
    name = LinkageMethodName.COMPLETE

    def linkage(self, cluster_1_data: pd.DataFrame, cluster_2_data: pd.DataFrame) -> float:
        distances = np.sqrt(((cluster_1_data.to_numpy()[:, np.newaxis] - cluster_2_data.to_numpy()) ** 2).sum(axis=2))
        return distances.max()


class AverageLinkage(LinkageMethod):
    name = LinkageMethodName.AVERAGE

    def linkage(self, cluster_1_data: pd.DataFrame, cluster_2_data: pd.DataFrame) -> float:
        distances = np.sqrt(((cluster_1_data.to_numpy()[:, np.newaxis] - cluster_2_data.to_numpy()) ** 2).sum(axis=2))
        return distances.mean()


class CentroidLinkage(LinkageMethod):
    name = LinkageMethodName.CENTROID

    def linkage(self, cluster_1_data: pd.DataFrame, cluster_2_data: pd.DataFrame) -> float:
        centroid_1 = cluster_1_data.mean(axis=0)
        centroid_2 = cluster_2_data.mean(axis=0)
        return np.sqrt(np.sum((centroid_1 - centroid_2) ** 2))


class _LinkageManager(type):
    linkage_methods = {
        method.name.value: method for method in
        [WardLinkage, SingleLinkage, CompleteLinkage, AverageLinkage, CentroidLinkage]
    }

    def __getitem__(self, linkage_method_name: str) -> Type[LinkageMethod]:
        if linkage_method_name not in self.linkage_methods:
            raise KeyError(
                f'Linkage method does not exist: {linkage_method_name}. '
                f'Allowed linkage methods: {[m.value for m in self.linkage_methods.keys()]}'
            )
        return self.linkage_methods[linkage_method_name]


class LinkageManager(metaclass=_LinkageManager):
    pass
