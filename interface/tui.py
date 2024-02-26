from abc import ABC, abstractmethod
from typing import Type

from clustering import ClusteringMethod
from clustering.agglomerative import Agglomerative
from clustering.k_means import KMeans
from clustering.manager import ClusterManager
from clustering.mean_shift import MeanShift
from extract_data import extract
from visualizer import Visualizer2D


class ClusteringTUI(ABC):
    @staticmethod
    @abstractmethod
    def get_instance():
        pass


class KMeansTUI(ClusteringTUI):
    @staticmethod
    def get_instance():
        k = int(input("Enter the number of clusters: "))
        distance_method_name = input("Enter the distance method name (optional): ") or None
        centroid_method_name = input("Enter the centroid method name (optional): ") or None
        return KMeans(k, distance_method_name, centroid_method_name)


class AgglomerativeTUI(ClusteringTUI):
    @staticmethod
    def get_instance():
        k = int(input("Enter the number of clusters: "))
        distance_method_name = input("Enter the distance method name (optional): ") or None
        linkage_method_name = input("Enter the linkage method name (optional): ") or None
        return Agglomerative(k, distance_method_name, linkage_method_name)


class MeanShiftTUI(ClusteringTUI):
    @staticmethod
    def get_instance():
        bandwidth = float(input("Enter max radius of cluster: "))
        return MeanShift(bandwidth)


clustering_tui: dict[Type[ClusteringMethod], ClusteringTUI] = {
    KMeans: KMeansTUI(),
    Agglomerative: AgglomerativeTUI(),
    MeanShift: MeanShiftTUI()
}


class TerminalUserInterface:
    # TODO: Refactor in a future
    def __init__(self):
        self.instance = None
        self.dataframe = None

    def _get_instance(self):
        while True:
            try:
                clustering_method_class = ClusterManager[input("Please enter your clustering method: ")]
            except KeyError as ex:
                print(str(ex))
                continue
            break
        while True:
            try:
                self.instance = clustering_tui[clustering_method_class].get_instance()
            except KeyError as ex:
                print(str(ex))
                continue
            break

    def _get_dataframe(self):
        while True:
            try:
                file_path = input("Please enter the csv file path: ")
                columns = input("Please enter the columns of the dataframe (through a space): ").split(' ') or None
                delimiter = input("Please enter the delimiter: ") or ','
                self.dataframe = extract(file_path, columns, delimiter)
            except KeyError as ex:
                print(str(ex))
                continue
            break

    def _clustering(self):
        clustered_df = self.instance.clustering(self.dataframe)
        vis = Visualizer2D()
        vis.plot(clustered_df)
        vis.show()

    def run(self):
        self._get_instance()
        self._get_dataframe()
        self._clustering()


if __name__ == '__main__':
    TerminalUserInterface().run()
