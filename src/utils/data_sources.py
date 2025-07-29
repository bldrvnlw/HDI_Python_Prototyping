from typing import Tuple, List
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn import datasets
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import gzip
import pickle
import csv


def label_to_colors(labels, cmap="tab10", seed=42):
    unique_labels = np.unique(labels)
    n_labels = len(unique_labels)
    rng = np.random.default_rng(seed)
    # colormap = plt.get_cmap(cmap)
    if cmap == "random":
        # Generate a random colormap
        # colormap = np.random.rand(len(unique_labels), 3).tolist()
        colormap = rng.random(size=(n_labels, 3)).tolist()
    else:
        cmaptype = plt.get_cmap(cmap)
        if type(cmaptype) == mcolors.LinearSegmentedColormap:
            colormap = cmaptype(np.linspace(0, 1, n_labels))
        else:
            colormap = cmaptype.colors
    label_to_color = {label: colormap[i] for i, label in enumerate(unique_labels)}
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


def get_MNIST(num_points: int) -> Tuple[np.array, np.array, List[int], List[int]]:
    # X, y = datasets.load_digits(return_X_y=True)
    mnist = datasets.fetch_openml("mnist_784", version=1)
    X_dataframe, y_series = mnist["data"], mnist["target"]
    # test with a subset
    X = X_dataframe.to_numpy()[0:num_points]
    y = y_series.to_numpy()[0:num_points]
    print(f"MNIST: data shape {X.shape} labels shape {y.shape}")
    import matplotlib.pyplot as plt

    # import matrix_viewer
    col_list = [
        "#EE3333",
        "#FF9900",
        "#FFEE00",
        "#AACC11",
        "#44AA77",
        "#0099EE",
        "#0066BB",
        "#443388",
        "#992288",
        "#EE0077",
    ]
    cmap = mcolors.ListedColormap(col_list)
    colors, unique_colors = label_to_colors(y, cmap)
    return (X, y, colors, unique_colors)


def get_mouse_Zheng(num_points: int) -> Tuple[np.array, np.array, List[int], List[int]]:
    # Load the mouse dataset from a CSV file

    with gzip.open(r"C:\\Users\\bvanlew\\Downloads\\10x_mouse_zheng.pkl.gz", "rb") as f:
        data = pickle.load(f)

    X = data["pca_50"]  # Features
    y = data["CellType1"]  # Labels

    # Limit to num_points if necessary
    if num_points < len(X):
        X = X[:num_points]
        y = y[:num_points]

    colors, unique_colors = label_to_colors(y, "random")

    return (X, y, colors, unique_colors)


def get_hypomap(num_points: int) -> Tuple[np.array, np.array, List[int], List[int]]:
    with open(r"C:\\Users\\bvanlew\\Downloads\\hypomap.pkl", "rb") as f:
        data = pickle.load(f)

    X = data["pca_50"]  # Features
    y = data["C2_name"]  # Labels

    # Limit to num_points if necessary
    if num_points < len(X):
        X = X[:num_points]
        y = y[:num_points]

    colors, unique_colors = label_to_colors(y, "gist_rainbow")

    return (X, y, colors, unique_colors)


def get_xmas_tree() -> Tuple[np.array, np.array, List[int], List[int]]:
    data = np.loadtxt(
        r"C:\\Users\\bvanlew\\Downloads\\xmas\\data.csv",
        delimiter=",",
        skiprows=1,
        dtype=int,
    )
    X = data[..., 1:]
    df = pd.read_csv(r"C:\\Users\\bvanlew\\Downloads\\xmas\\metadata.csv")
    colors = df["main_color"].to_list()
    unique_colors = np.unique(colors)
    y = df["main_label"].to_list()
    return (X, y, colors, unique_colors)


def get_wikiword_350000(
    num_points: int,
) -> Tuple[np.array, np.array, List[int], List[int]]:
    data = np.load(r"D:\\Data\\ML\\wiki-news-300d-1M.vec\\vec350000.npy")
    X = data[:num_points, ...]
    y = None
    colors = np.full(num_points, "#1F1B53")
    unique_colors = np.unique(colors)
    return (X, y, colors, unique_colors)
