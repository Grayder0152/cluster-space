from abc import ABC, abstractmethod
from enum import StrEnum, auto
from typing import Type

import numpy as np
import pandas as pd

from distances import DistanceManager, DistanceMethod, DistanceMethodName


class CentroidMethodName(StrEnum):
    K_MEAN_PP = 'k-mean++'
    RANDOM = auto()


class CentroidMethod(ABC):
    def __init__(self, k: int, **kwargs):
        self.k: int = k

    @property
    @abstractmethod
    def name(self) -> CentroidMethodName:
        pass

    @abstractmethod
    def get_centroids(self, data: pd.DataFrame) -> pd.DataFrame:
        pass


class KMeanPp(CentroidMethod):
    name = CentroidMethodName.K_MEAN_PP

    def __init__(self, k: int, distance_method_name: str = DistanceMethodName.EUCLIDEAN.value):
        super().__init__(k)
        self.distance_method: DistanceMethod = DistanceManager[distance_method_name]()

    def get_centroids(self, dataframe: pd.DataFrame) -> np.array:
        init_centroids = [dataframe.sample(n=1).to_numpy()[0]]

        np_df = dataframe.to_numpy()
        for _ in range(self.k - 1):
            distances = np.array([
                min(self.distance_method.distance(point, centroid) for centroid in init_centroids)
                for point in np_df
            ])
            probabilities = distances ** 2
            probabilities /= probabilities.sum()
            next_centroid = np_df[np.random.choice(len(np_df), p=probabilities)]
            init_centroids.append(next_centroid)

        return np.array(init_centroids)


class Random(CentroidMethod):
    name = CentroidMethodName.RANDOM

    def get_centroids(self, dataframe: pd.DataFrame) -> np.array:
        random_indices = np.random.choice(dataframe.index, size=self.k, replace=False)
        init_centroids = dataframe.iloc[random_indices].to_numpy()
        return init_centroids


class _InitialCentroidManager(type):
    centroid_methods = {method.name.value: method for method in [KMeanPp, Random]}

    def __getitem__(self, centroid_method_name: str) -> Type[CentroidMethod]:
        if centroid_method_name not in self.centroid_methods:
            raise KeyError(
                f'Centroid method does not exist: {centroid_method_name}. '
                f'Allowed centroid methods: {[m.value for m in self.centroid_methods.keys()]}'
            )
        return self.centroid_methods[centroid_method_name]


class InitialCentroidManager(metaclass=_InitialCentroidManager):
    pass
