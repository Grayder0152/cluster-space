from typing import Optional

import numpy as np
import pandas as pd

from clustering import ClusteringMethod, ClusteringMethodName
from clustering.k_means.centroid_methods import InitialCentroidManager, CentroidMethodName, CentroidMethod
from distances import DistanceMethodName, DistanceManager
from settings import CLUSTER_COL_NAME


class KMeans(ClusteringMethod):
    name = ClusteringMethodName.K_MEANS

    def __init__(
            self, k: int,
            distance_method_name: Optional[str] = None,
            centroid_method_name: Optional[str] = None
    ):
        distance_method_name = distance_method_name or DistanceMethodName.EUCLIDEAN.value
        centroid_method_name = centroid_method_name or CentroidMethodName.K_MEAN_PP.value

        self.k: int = k
        self.distance_method = DistanceManager[distance_method_name]()
        self.centroid_method = InitialCentroidManager[centroid_method_name](
            k=k, distance_method_name=distance_method_name
        )

    def clustering(self, dataframe: pd.DataFrame):
        dataframe = dataframe.copy()
        centroids = self.centroid_method.get_centroids(dataframe)
        while True:
            clusters = self._get_clusters(dataframe, centroids)
            new_centroids = self._update_centroids(dataframe, clusters)
            if np.all(centroids == new_centroids):
                break
            centroids = new_centroids

        dataframe[CLUSTER_COL_NAME] = clusters
        return dataframe

    def _get_clusters(self, dataframe: pd.DataFrame, centroids: np.array) -> np.array:
        clusters = []
        for point in dataframe.to_numpy():
            cluster = np.argmin([self.distance_method.distance(point, centroid) for centroid in centroids])
            clusters.append(cluster)
        return np.array(clusters)

    def _update_centroids(self, dataframe: pd.DataFrame, clusters: np.array) -> np.array:
        new_centroids = []
        for i in range(self.k):
            points = dataframe[clusters == i].to_numpy()
            if len(points) > 0:
                new_centroid = np.mean(points, axis=0)
                new_centroids.append(new_centroid)
            else:
                new_centroids.append(np.random.rand(dataframe.shape[1]))
        return np.array(new_centroids)
