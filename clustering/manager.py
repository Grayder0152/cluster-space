from typing import Type

from clustering import ClusteringMethod
from clustering.agglomerative import Agglomerative
from clustering.k_means import KMeans
from clustering.mean_shift import MeanShift


class _ClusterManager(type):
    cluster_methods = {method.name: method for method in [KMeans, MeanShift, Agglomerative]}

    def __getitem__(self, cluster_method_name: str) -> Type[ClusteringMethod]:
        if cluster_method_name not in self.cluster_methods:
            raise KeyError(
                f'Cluster method does not exist: {cluster_method_name}. '
                f'Allowed cluster methods: {[m.value for m in self.cluster_methods.keys()]}'
            )
        return self.cluster_methods[cluster_method_name]


class ClusterManager(metaclass=_ClusterManager):
    pass
