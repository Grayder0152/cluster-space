from abc import ABC, abstractmethod

from enum import auto, StrEnum
from typing import Type

import numpy as np
import pandas as pd

from distances import DistanceMethodName, DistanceMethod, DistanceManager
from settings import CLUSTER_COL_NAME


def calc_centroids(vectors):
    np_vectors = np.array([v.toArray() for v in vectors])
    centroid = np.mean(np_vectors, axis=0)
    return centroid.tolist()


def sum_of_distances_to_centroid(vectors, centroid):
    centroid = np.array(centroid)
    points = np.array([v.toArray() for v in vectors])
    distances = np.sqrt(((points - centroid) ** 2).sum(axis=1))
    return float(distances.sum())


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
    def linkage(self, dataframe: pd.DataFrame, cluster_id: str, cluster_2: str) -> float:
        pass


class WardLinkage(LinkageMethod):
    name = LinkageMethodName.WARD

    def __init__(self, distance_method_name: str = DistanceMethodName.EUCLIDEAN.value):
        self.distance_method: DistanceMethod = DistanceManager[distance_method_name]()

    def linkage(self, dataframe: pd.DataFrame, cluster_1: str, cluster_2: str) -> float:  # 86.7
        cluster_1_data = dataframe[dataframe[CLUSTER_COL_NAME] == cluster_1].drop(CLUSTER_COL_NAME, axis=1)
        cluster_2_data = dataframe[dataframe[CLUSTER_COL_NAME] == cluster_2].drop(CLUSTER_COL_NAME, axis=1)
        centroid_1 = np.mean(cluster_1_data, axis=0)
        centroid_2 = np.mean(cluster_2_data, axis=0)

        # Векторизоване обчислення відстаней
        dist_cluster_1 = np.sqrt(((cluster_1_data - centroid_1) ** 2).sum(axis=1))
        dist_cluster_2 = np.sqrt(((cluster_2_data - centroid_2) ** 2).sum(axis=1))

        merged_cluster = pd.concat([cluster_1_data, cluster_2_data])
        merged_centroid = np.mean(merged_cluster, axis=0)
        dist_merged = np.sqrt(((merged_cluster - merged_centroid) ** 2).sum(axis=1))

        return dist_merged.sum() - (dist_cluster_1.sum() + dist_cluster_2.sum())
    # def linkage(self, dataframe: pd.DataFrame, cluster_1: str, cluster_2: str) -> float: # 89.3
    #     cluster_1 = dataframe[dataframe[CLUSTER_COL_NAME] == cluster_1].drop(CLUSTER_COL_NAME, axis=1)
    #     cluster_2 = dataframe[dataframe[CLUSTER_COL_NAME] == cluster_2].drop(CLUSTER_COL_NAME, axis=1)
    #     centroid_1 = np.mean(cluster_1, axis=0)
    #     centroid_2 = np.mean(cluster_2, axis=0)
    #     cluster_1['dist'] = cluster_1.apply(lambda point: self.distance_method.distance(point, centroid_1), axis=1)
    #     cluster_2['dist'] = cluster_2.apply(lambda point: self.distance_method.distance(point, centroid_2), axis=1)
    #
    #     merged_cluster = pd.concat([cluster_1, cluster_2]).drop('dist', axis=1)
    #     merged_centroid = np.mean(merged_cluster, axis=0)
    #     merged_cluster['dist'] = merged_cluster.apply(
    #         lambda point: self.distance_method.distance(point, merged_centroid), axis=1)
    #
    #     return merged_cluster['dist'].sum() - (cluster_1['dist'].sum() + cluster_2['dist'].sum())


class _LinkageManager(type):
    linkage_methods = {method.name.value: method for method in [WardLinkage, ]}

    def __getitem__(self, linkage_method_name: str) -> Type[LinkageMethod]:
        if linkage_method_name not in self.linkage_methods:
            raise KeyError(
                f'Linkage method does not exist: {linkage_method_name}. '
                f'Allowed linkage methods: {[m.value for m in self.linkage_methods.keys()]}'
            )
        return self.linkage_methods[linkage_method_name]


class LinkageManager(metaclass=_LinkageManager):
    pass


if __name__ == '__main__':
    from clustering.kmean import KMean
    from visualizer import Visualizer2D
    from extract_data import extract


    def prepare_df():
        km = KMean(10)

        df = extract('data_2.csv', delimiter=' ')
        _, clustered_df = km.clustering(df)

        vis = Visualizer2D()
        vis.plot(clustered_df)
        vis.show()

        return clustered_df


    df = prepare_df()

    ward = WardLinkage()
    print(ward.linkage(df, 0, 1))
