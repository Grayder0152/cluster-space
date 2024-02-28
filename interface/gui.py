import os
import sys
from typing import Optional

import pandas as pd
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QGridLayout,
    QPushButton, QFileDialog, QWidget,
    QComboBox, QLabel, QLineEdit, QSpinBox,
    QDoubleSpinBox, QHBoxLayout, QVBoxLayout
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from clustering.agglomerative import LinkageManager, Agglomerative
from clustering.k_means import KMeans, InitialCentroidManager
from clustering.manager import ClusterManager
from clustering.mean_shift import MeanShift
from distances import DistanceManager
from extract_data import extract
from settings import CLUSTER_COL_NAME, SAMPLE_DATASETS_DIR

os.environ["XDG_SESSION_TYPE"] = "xcb"

CLUSTER_TYPES = [' '.join(i.split('_')).title() for i in ClusterManager.cluster_methods.keys()]
DISTANCE_METHODS = [i.title() for i in DistanceManager.distance_methods.keys()]
CENTROID_METHODS = [i.title() for i in InitialCentroidManager.centroid_methods.keys()]
LINKAGE_METHODS = [i.title() for i in LinkageManager.linkage_methods.keys()]


class KMeansGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.load_data_btn: Optional[QPushButton] = None
        self.run_btn: Optional[QPushButton] = None

        self.clustering_type: QComboBox = QComboBox(self)
        self.distance_method: QComboBox = QComboBox(self)
        self.centroid_method: QComboBox = QComboBox(self)
        self.linkage_method: QComboBox = QComboBox(self)

        self.cluster_count: QSpinBox = QSpinBox(self)
        self.bandwidth: QDoubleSpinBox = QDoubleSpinBox(self)

        self.canvas: Optional[FigureCanvas] = None

        self.clustering_layout = QHBoxLayout()

        self.clustering_widgets = {
            'K Means': [self.cluster_count, self.distance_method, self.centroid_method],
            'Agglomerative': [self.cluster_count, self.distance_method, self.linkage_method],
            'Mean Shift': [self.bandwidth]
        }
        self.file_name: Optional[str] = None
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('Clustering')
        self.setGeometry(100, 100, 800, 600)

        layout = QVBoxLayout()
        top_l = QHBoxLayout()
        self.load_data_btn = QPushButton('Load Data')
        self.load_data_btn.clicked.connect(self.load_data)

        self.run_btn = QPushButton('Run Clustering')
        self.run_btn.clicked.connect(self.run_clustering)

        self.canvas = FigureCanvas(Figure(figsize=(5, 3)))

        self.clustering_type.addItems(CLUSTER_TYPES)
        self.clustering_type.currentTextChanged.connect(self.on_clustering_changed)

        self.distance_method.addItems(DISTANCE_METHODS)
        self.centroid_method.addItems(CENTROID_METHODS)
        self.linkage_method.addItems(LINKAGE_METHODS)
        self.cluster_count.setRange(1, 99)

        top_l.addWidget(self.clustering_type)
        top_l.addWidget(self.load_data_btn)

        self._set_clustering_widgets()

        layout.addLayout(top_l)
        layout.addLayout(self.clustering_layout)
        layout.addWidget(self.canvas)
        layout.addWidget(self.run_btn)

        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

    def _set_clustering_widgets(self):
        for widget in self.clustering_widgets[self.clustering_type.currentText()]:
            self.clustering_layout.addWidget(widget)

    def run_clustering(self):
        if self.file_name is not None:
            df = extract(self.file_name, delimiter=' ')
            if self.clustering_type.currentText() == 'K Means':
                clustering = KMeans(
                    k=self.cluster_count.value(),
                    distance_method_name=self.distance_method.currentText().lower(),
                    centroid_method_name=self.centroid_method.currentText().lower(),
                )
            elif self.clustering_type.currentText() == 'Agglomerative':
                clustering = Agglomerative(
                    k=self.cluster_count.value(),
                    distance_method_name=self.distance_method.currentText().lower(),
                    linkage_method_name=self.linkage_method.currentText().lower(),
                )
            elif self.clustering_type.currentText() == 'Mean Shift':
                clustering = MeanShift(
                    bandwidth=self.bandwidth.value()
                )
            else:
                return
            clustered_df = clustering.clustering(df)
            self.plot(clustered_df)

    def on_clustering_changed(self):
        for i in reversed(range(self.clustering_layout.count())):
            widget = self.clustering_layout.itemAt(i).widget()
            self.clustering_layout.removeWidget(widget)
            widget.setParent(None)

        self._set_clustering_widgets()

    def load_data(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        options |= QFileDialog.Directory

        self.file_name, _ = QFileDialog.getOpenFileName(
            self,
            caption="Data Files for Clustering",
            directory=SAMPLE_DATASETS_DIR,
            filter="CSV Filed (*.csv)", options=options
        )
        self.load_data_btn.setText(self.file_name.split('/')[-1])

    def plot(self, dataframe: pd.DataFrame, feature_names: Optional[list[str, str]] = None) -> None:
        feature_1, feature_2 = feature_names or dataframe.columns.drop(CLUSTER_COL_NAME)[:2]
        clusters = sorted(dataframe[CLUSTER_COL_NAME].unique())

        self.canvas.figure.clf()
        ax = self.canvas.figure.subplots()

        ax.set_xlabel(feature_1)
        ax.set_ylabel(feature_2)

        for cluster in clusters:
            cluster_data = dataframe[dataframe[CLUSTER_COL_NAME] == cluster]
            ax.scatter(cluster_data[feature_1], cluster_data[feature_2], label=f'Cluster {cluster}')

        ax.legend()
        self.canvas.draw()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = KMeansGUI()
    main_window.show()
    sys.exit(app.exec_())
