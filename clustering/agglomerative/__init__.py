import itertools
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from clustering import ClusteringMethod, ClusteringMethodName
from clustering.agglomerative.linkages import LinkageMethodName, LinkageManager
from distances import DistanceMethodName, DistanceManager
from settings import CLUSTER_COL_NAME


class Agglomerative(ClusteringMethod):
    name = ClusteringMethodName.AGGLOMERATIVE

    def __init__(
            self, k: int,
            distance_method_name: Optional[str] = None,
            linkage_method_name: Optional[str] = None
    ):
        distance_method_name = distance_method_name or DistanceMethodName.EUCLIDEAN.value
        linkage_method_name = linkage_method_name or LinkageMethodName.SINGLE.value

        self.k: int = k
        self.distance_method = DistanceManager[distance_method_name]()
        self.linkage_method = LinkageManager[linkage_method_name]()

    def _calculate_distance_matrix(self, dataframe: pd.DataFrame, clusters: Iterable) -> np.array:
        cluster_count = len(dataframe)
        distance_matrix = np.zeros((cluster_count, cluster_count))

        for cluster_1, cluster_2 in itertools.combinations(clusters, 2):
            cluster_1_data = dataframe[dataframe[CLUSTER_COL_NAME] == cluster_1].drop(CLUSTER_COL_NAME, axis=1)
            cluster_2_data = dataframe[dataframe[CLUSTER_COL_NAME] == cluster_2].drop(CLUSTER_COL_NAME, axis=1)
            dist = self.linkage_method.linkage(cluster_1_data, cluster_2_data)
            distance_matrix[cluster_1, cluster_2] = distance_matrix[cluster_2, cluster_1] = dist
        return distance_matrix

    def clustering(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        # TODO: required to be optimize
        max_iteration = len(dataframe)
        dataframe.reset_index(inplace=True, names=[CLUSTER_COL_NAME])
        clusters = dataframe[CLUSTER_COL_NAME].unique()
        distance_matrix = self._calculate_distance_matrix(dataframe, clusters)
        while max_iteration != self.k:
            lowes_distance = np.inf
            closest_clusters = (None, None)

            cluster_combination = itertools.combinations(clusters, 2)

            for cluster_1, cluster_2 in cluster_combination:
                dist = distance_matrix[cluster_1, cluster_2]
                if lowes_distance > dist:
                    lowes_distance = dist
                    closest_clusters = cluster_1, cluster_2

            augmented_cluster = closest_clusters[0]
            merged_cluster = closest_clusters[1]
            dataframe.loc[dataframe[CLUSTER_COL_NAME] == merged_cluster, CLUSTER_COL_NAME] = augmented_cluster

            clusters = dataframe[CLUSTER_COL_NAME].unique()
            cluster_1_data = dataframe[dataframe[CLUSTER_COL_NAME] == augmented_cluster].drop(CLUSTER_COL_NAME, axis=1)
            for i in clusters:
                cluster_2_data = dataframe[dataframe[CLUSTER_COL_NAME] == i].drop(CLUSTER_COL_NAME, axis=1)
                distance_matrix[augmented_cluster, i] = distance_matrix[
                    i, augmented_cluster] = self.linkage_method.linkage(cluster_1_data, cluster_2_data)

            max_iteration -= 1
        return dataframe


if __name__ == '__main__':
    from extract_data import extract
    from visualizer import Visualizer2D

    agg = Agglomerative(k=3, linkage_method_name=LinkageMethodName.SINGLE.value)

    df = extract('data_2.csv', delimiter=' ')

    clustered_df = agg.clustering(df)
    vis = Visualizer2D()
    vis.plot(clustered_df)
    vis.show()
