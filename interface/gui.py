import os
import sys
from typing import Optional

import pandas as pd
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QPushButton, QFileDialog, QWidget
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from clustering.k_means import KMeans
from extract_data import extract
from settings import CLUSTER_COL_NAME, SAMPLE_DATASETS_DIR

os.environ["XDG_SESSION_TYPE"] = "xcb"


class KMeansGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.load_data_button: Optional[QPushButton] = None
        self.canvas: Optional[FigureCanvas] = None

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('KMeans Clustering')
        self.setGeometry(100, 100, 800, 600)

        layout = QVBoxLayout()

        self.load_data_button = QPushButton('Завантажити дані')
        self.load_data_button.clicked.connect(self.load_data)

        self.canvas = FigureCanvas(Figure(figsize=(5, 3)))

        layout.addWidget(self.load_data_button)
        layout.addWidget(self.canvas)

        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

    def load_data(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        options |= QFileDialog.Directory

        file_name, _ = QFileDialog.getOpenFileName(
            self,
            caption="Data Files for Clustering",
            directory=SAMPLE_DATASETS_DIR,
            filter="CSV Filed (*.csv)", options=options
        )
        if file_name:
            print(file_name)
            df = extract(file_name, delimiter=' ')
            clustering = KMeans(3)
            clustered_df = clustering.clustering(df)
            self.plot(clustered_df)

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
