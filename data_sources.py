from typing import Tuple, List
import numpy as np
from sklearn.datasets import make_classification
from sklearn import datasets
import matplotlib.pyplot as plt


def label_to_colors(labels, cmap="tab10"):
    unique_labels = np.unique(labels)
    colormap = plt.get_cmap(cmap)
    label_to_color = {label: colormap(i) for i, label in enumerate(unique_labels)}
    return [label_to_color[label] for label in labels], [
        label_to_color[label] for label in unique_labels
    ]


def get_generated(num_points: int) -> Tuple[np.array, np.array, List[int]]:
    X, y = make_classification(
        n_features=1000,
        n_classes=10,
        n_samples=num_points,
        n_informative=4,
        random_state=5,
        n_clusters_per_class=1,
    )
    colors = label_to_colors(y)

    return (X, y, colors)


def get_MNIST(num_points: int) -> Tuple[np.array, np.array, List[int]]:
    # X, y = datasets.load_digits(return_X_y=True)
    mnist = datasets.fetch_openml("mnist_784", version=1)
    X_dataframe, y_series = mnist["data"], mnist["target"]
    # test with a subset
    X = X_dataframe.to_numpy()[0:num_points]
    y = y_series.to_numpy()[0:num_points]
    print(f"MNIST: data shape {X.shape} labels shape {y.shape}")
    import matplotlib.pyplot as plt

    # import matrix_viewer

    colors, unique_colors = label_to_colors(y)
    return (X, y, colors, unique_colors)
