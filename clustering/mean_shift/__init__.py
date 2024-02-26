import numpy as np
import pandas as pd

from clustering import ClusteringMethod, ClusteringMethodName


class MeanShift(ClusteringMethod):
    name = ClusteringMethodName.MEAN_SHIFT

    def __init__(self, bandwidth: float):
        self.bandwidth = bandwidth

    def clustering(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        points = dataframe.to_numpy()
        n_points = len(points)
        clusters = np.zeros(n_points, dtype=int)
        centers = np.copy(points)

        while True:
            new_centers = []
            for i in range(n_points):
                center = centers[i]
                distances = np.linalg.norm(points - center, axis=1)
                in_bandwidth = points[distances < self.bandwidth]
                new_center = np.mean(in_bandwidth, axis=0)
                new_centers.append(new_center)

            new_centers = np.array(new_centers)
            if np.all(centers == new_centers):
                break
            centers = new_centers

        for i, point in enumerate(points):
            distances = np.linalg.norm(centers - point, axis=1)
            clusters[i] = np.argmin(distances)

        dataframe['cluster'] = clusters
        return dataframe


if __name__ == '__main__':
    from extract_data import extract
    from visualizer import Visualizer2D, Visualizer3D

    mean_shift = MeanShift(3.0)

    df = extract('data_2.csv', delimiter=' ')
    clustered_df = mean_shift.clustering(df)
    vis = Visualizer2D()
    vis.plot(clustered_df)
    vis.show()

    df = extract('temps.csv', columns=['t36', 't35', 't34'], delimiter=',')
    clustered_df = mean_shift.clustering(df)
    vis = Visualizer3D()
    vis.plot(clustered_df)
    vis.show()
