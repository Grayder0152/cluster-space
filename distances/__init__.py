from abc import ABC, abstractmethod
from enum import StrEnum, auto
from typing import Type

import numpy as np


class DistanceMethodName(StrEnum):
    EUCLIDEAN = auto()


class DistanceMethod(ABC):
    @property
    @abstractmethod
    def name(self) -> DistanceMethodName:
        pass

    @staticmethod
    @abstractmethod
    def distance(point_1: np.array, point_2: np.array) -> float:
        pass

    def __repr__(self) -> str:
        return f'{self.name.value.title()} distance method'


class EuclideanDistance(DistanceMethod):
    name = DistanceMethodName.EUCLIDEAN

    @staticmethod
    def distance(point_1: np.array, point_2: np.array) -> float:
        return np.sqrt(np.sum((point_1 - point_2) ** 2))


class _DistanceManager(type):
    distance_methods = {method.name: method for method in [EuclideanDistance, ]}

    def __getitem__(self, distance_method_name: str) -> Type[DistanceMethod]:
        if distance_method_name not in self.distance_methods:
            raise KeyError(
                f'Distance method does not exist: {distance_method_name}. '
                f'Allowed distance methods: {[m.value for m in self.distance_methods.keys()]}'
            )
        return self.distance_methods[distance_method_name]


class DistanceManager(metaclass=_DistanceManager):
    pass


if __name__ == '__main__':
    print(DistanceManager['euclidean'])
