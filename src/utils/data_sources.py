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

    def rgb_to_hex(r, g, b):
        def rescale(x):
            return int(max(0, min(x * 255, 255)))

        return "#{0:02x}{1:02x}{2:02x}".format(rescale(r), rescale(g), rescale(b))

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
    label_to_color = {
        label: rgb_to_hex(*colormap[i][0:3]) for i, label in enumerate(unique_labels)
    }

    return (
        [label_to_color[label] for label in labels],
        [label_to_color[label] for label in unique_labels],
        label_to_color,
    )


def get_generated(num_points: int) -> Tuple[np.array, np.array, List[int]]:
    X, y = make_classification(
        n_features=1000,
        n_classes=10,
        n_samples=num_points,
        n_informative=4,
        random_state=5,
        n_clusters_per_class=1,
    )
    colors, unique_colors, color_key = label_to_colors(y)

    return (X, y, colors)


def get_MNIST(num_points: int) -> dict:
    # X, y = datasets.load_digits(return_X_y=True)
    mnist = datasets.fetch_openml("mnist_784", version=1)
    X_dataframe, y_series = mnist["data"], mnist["target"]
    # test with a subset
    X = X_dataframe.to_numpy()[0:num_points]
    y = y_series.to_numpy()[0:num_points]
    print(f"MNIST: data shape {X.shape} labels shape {y.shape}")
    import matplotlib.pyplot as plt

    # import matrix_viewer
    col_key = {
        "0": "#EE3333",
        "1": "#FF9900",
        "2": "#FFEE00",
        "3": "#AACC11",
        "4": "#44AA77",
        "5": "#0099EE",
        "6": "#0066BB",
        "7": "#443388",
        "8": "#992288",
        "9": "#EE0077",
    }

    return {"X": X, "label": y, "col_key": col_key}


def get_mouse_Zheng(num_points: int) -> dict:
    # Load the mouse dataset from a CSV file

    with gzip.open(r"C:\\Users\\bvanlew\\Downloads\\10x_mouse_zheng.pkl.gz", "rb") as f:
        data = pickle.load(f)

    X = data["pca_50"]  # Features
    y = data["CellType1"]  # Labels

    # Limit to num_points if necessary
    if num_points < len(X):
        X = X[:num_points]
        y = y[:num_points]

    colors, unique_colors, col_key = label_to_colors(y, "random")

    return {"X": X, "label": y, "col_key": col_key}


def get_hypomap(num_points: int) -> pd.DataFrame:
    with open(r"C:\\Users\\bvanlew\\Downloads\\hypomap.pkl", "rb") as f:
        data = pickle.load(f)

    X = data["pca_50"]  # Features
    y = data["C2_name"]  # Labels

    # Limit to num_points if necessary
    if num_points < len(X):
        X = X[:num_points]
        y = y[:num_points]

    colors, unique_colors, col_key = label_to_colors(y, "gist_rainbow")

    return {"X": X, "label": y, "col_key": col_key}


def get_xmas_tree() -> pd.DataFrame:
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
    return pd.DataFrame(
        {"points": X, "labels": y, "colors": colors, "unique_colors": unique_colors}
    )


def get_wikiword_350000(
    num_points: int,
) -> pd.DataFrame:
    data = np.load(r"D:\\Data\\ML\\wiki-news-300d-1M.vec\\vec350000.npy")
    X = data[:num_points, ...]
    y = np.full((num_points), 0)
    col_key = {0: "#1F1B53"}
    # unique_colors = np.unique(colors)
    return {"X": X, "label": y, "col_key": col_key}
