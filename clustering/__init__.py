from abc import ABC, abstractmethod
from enum import StrEnum, auto

import pandas as pd


class ClusteringMethodName(StrEnum):
    K_MEANS = auto()
    MEAN_SHIFT = auto()
    AGGLOMERATIVE = auto()


class ClusteringMethod(ABC):
    @property
    @abstractmethod
    def name(self) -> ClusteringMethodName:
        pass

    @staticmethod
    @abstractmethod
    def clustering(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        pass

    def __repr__(self) -> str:
        return f'{self.name.value.title()} distance method'
