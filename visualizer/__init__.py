import os
from abc import ABC, abstractmethod
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from clustering.k_means import KMeans
from settings import CLUSTER_COL_NAME, SAMPLE_DATASETS_DIR


class BaseVisualizer(ABC):
    figure = None
    axis = None

    def __init__(self, **kwargs) -> None:
        self.figure, self.axis = plt.subplots(1, 1, **kwargs)

    @abstractmethod
    def plot(self, dataframe: pd.DataFrame, feature_names: Optional[list] = None) -> None:
        pass

    @staticmethod
    def show():
        plt.show()


class VisualizerND(BaseVisualizer):
    figure = None
    axis = None

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.axis.set_title(f'{self.feature_count}D Visualization of Clusters')

    @property
    @abstractmethod
    def feature_count(self) -> int:
        pass

    def add_centroids(self, centroids: np.array) -> None:
        self.axis.scatter(*[centroids[:, i] for i in range(self.feature_count)], s=70, label='Centroids')
        self.axis.legend()


class Visualizer2D(VisualizerND):
    feature_count = 2

    def plot(self, dataframe: pd.DataFrame, feature_names: Optional[list[str, str]] = None) -> None:
        if feature_names:
            if len(feature_names) != self.feature_count:
                raise ValueError('Incorrect amount of feature names provided.')
            feature_1, feature_2 = feature_names
        else:
            feature_1, feature_2 = dataframe.columns.drop(CLUSTER_COL_NAME)[:self.feature_count]

        clusters = sorted(dataframe[CLUSTER_COL_NAME].unique())

        self.axis.set_xlabel(feature_1)
        self.axis.set_ylabel(feature_2)

        for cluster in clusters:
            cluster_data = dataframe[dataframe[CLUSTER_COL_NAME] == cluster]
            self.axis.scatter(cluster_data[feature_1], cluster_data[feature_2], label=f'Cluster {cluster}')

        self.axis.legend()


class Visualizer3D(VisualizerND):
    feature_count = 3

    def __init__(self):
        super().__init__(subplot_kw=dict(projection="3d"))

    def plot(self, dataframe: pd.DataFrame, feature_names: Optional[list[str, str, str]] = None) -> None:
        if feature_names:
            if len(feature_names) != self.feature_count:
                raise ValueError('Incorrect amount of feature names provided.')
            feature_1, feature_2, feature_3 = feature_names
        else:
            feature_1, feature_2, feature_3 = dataframe.columns.drop(CLUSTER_COL_NAME)[:self.feature_count]
        clusters = sorted(dataframe[CLUSTER_COL_NAME].unique())

        self.axis.set_xlabel(feature_1)
        self.axis.set_ylabel(feature_2)
        self.axis.set_zlabel(feature_3)

        xs = dataframe[feature_1]
        ys = dataframe[feature_2]
        zs = dataframe[feature_3]

        for cluster in clusters:
            self.axis.scatter(
                xs[dataframe[CLUSTER_COL_NAME] == cluster],
                ys[dataframe[CLUSTER_COL_NAME] == cluster],
                zs[dataframe[CLUSTER_COL_NAME] == cluster],
                s=50, alpha=0.6, edgecolors='w', label=f'Cluster {cluster}'
            )

        self.axis.legend()


class ParallelCoordinate(BaseVisualizer):
    def plot(self, dataframe: pd.DataFrame, feature_names: Optional[list[str, str, str]] = None) -> None:
        if not feature_names:
            feature_names = dataframe.columns

        pd.plotting.parallel_coordinates(
            dataframe[feature_names],
            class_column=CLUSTER_COL_NAME,
            ax=self.axis
        )
        self.axis.legend(title=CLUSTER_COL_NAME)


if __name__ == '__main__':
    km = KMeans(3)

    df = pd.read_csv(os.path.join(SAMPLE_DATASETS_DIR, 'wines.csv'),  sep='\s+')
    print(df)
    clustered_df = km.clustering(df)

    vis = Visualizer3D()
    vis.plot(clustered_df)
    # vis.add_centroids(centroids)
    vis.show()
