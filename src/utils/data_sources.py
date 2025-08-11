from typing import Tuple, List
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn import datasets
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from scipy import io
import gzip
import pickle
import csv
from PIL import Image
import glob
import os
from pathlib import Path

# https://cs.nyu.edu/home/people/in_memoriam/roweis/data.html


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


def get_hypomap(num_points: int) -> dict:
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


def get_xmas_tree() -> dict:
    data = np.loadtxt(
        r"C:\\Users\\bvanlew\\Downloads\\xmas\\data.csv",
        delimiter=",",
        skiprows=1,
        dtype=int,
    )
    X = data[..., 1:]
    df = pd.read_csv(r"C:\\Users\\bvanlew\\Downloads\\xmas\\metadata.csv")
    labels = df["sub_label"].unique()
    color_key = {}
    for label in labels:
        color = df.loc[df["sub_label"] == label]["sub_color"].iloc[0]
        color_key[label] = color

    all_labels = df["sub_label"].tolist()
    return {"X": X, "label": all_labels, "col_key": color_key}


def get_wikiword_350000(
    num_points: int,
) -> dict:
    data = np.load(r"D:\\Data\\ML\\wiki-news-300d-1M.vec\\vec350000.npy")
    X = data[:num_points, ...]
    y = ["w"] * num_points
    col_key = {"w": "#210056"}
    # unique_colors = np.unique(colors)
    return {"X": X, "label": y, "col_key": col_key}


def load_word2vec_bin(fname):
    with open(fname, "rb") as f:
        # 1. Read header
        header = f.readline()
        vocab_size, vector_size = map(int, header.split())

        # 2. Preallocate
        vocab = []
        vectors = np.zeros((vocab_size, vector_size), dtype=np.float32)

        # 3. Read each word + vector
        for i in range(vocab_size):
            # Read the word (ends at space)
            word_bytes = bytearray()
            while True:
                ch = f.read(1)
                if ch == b" ":
                    break
                word_bytes.extend(ch)
            word = word_bytes.decode("utf-8")

            # Read the vector
            vec = np.frombuffer(f.read(4 * vector_size), dtype=np.float32)
            vectors[i] = vec

            vocab.append(word)

    return vocab, vectors


def get_word2vec(num_points: int):
    # from https://code.google.com/archive/p/word2vec/
    vocab, vectors = load_word2vec_bin(
        "C:\\Users\\bvanlew\\Downloads\\word2vec3x10_6_300.bin\\GoogleNews-vectors-negative300.bin"
    )
    X = vectors[:num_points, ...]
    y = ["w"] * num_points
    col_key = {"w": "#210056"}
    # unique_colors = np.unique(colors)
    return {"X": X, "label": y, "col_key": col_key}


def load_coil20_images(path, target_size=(128, 128), limit=1400):
    """
    Loads COIL-20 PNG images into a NumPy array.
    see https://www.cs.columbia.edu/CAVE/software/softlib/coil-20.php

    Parameters:
        path (str): Directory containing the PNG images.
        target_size (tuple): (height, width), default (128, 128).
        limit (int): Maximum number of images to load.

    Returns:
        images (np.ndarray): shape (limit, 16384) float32 normalized [0,1]
        filenames (list): corresponding filenames
    """
    p = Path(path)
    files = sorted(list(p.glob("**/*.png")))

    # Optionally limit to first 1400 files
    files = files[:limit]

    num_images = len(files)
    h, w = target_size
    data = np.zeros((num_images, h * w), dtype=np.float32)

    for i, file in enumerate(files):
        img = Image.open(file).convert("L")  # 'L' = grayscale
        img = img.resize((w, h))  # just in case
        arr = np.array(img, dtype=np.float32) / 255.0  # normalize to [0, 1]
        data[i] = arr.flatten()

    return data, files


def get_coil20(num_points: int) -> dict:
    # Example usage
    coil_path = "D:\\Data\\ML\\coil-20-proc\\coil-20-proc\\"
    data, filenames = load_coil20_images(coil_path, limit=1400)

    X = data[:num_points, ...]
    names = [Path(f).name for f in filenames[:num_points]]
    y = [name.split("_")[0] for name in names]
    colors, unique_colors, col_key = label_to_colors(y, "random")
    return {"X": X, "label": y, "col_key": col_key}


def get_frey_faces(num_points: int) -> dict:
    data = io.loadmat(r"D:\\Data\\ML\\frey_rawface.mat")
    X = data["ff"].transpose()[:num_points]
    y = ["w"] * num_points
    col_key = {"w": "#210056"}
    # unique_colors = np.unique(colors)
    return {"X": X, "label": y, "col_key": col_key}


def load_mnist(path, kind="train"):
    import os
    import gzip
    import numpy as np

    """Load MNIST data from `path`"""
    labels_path = os.path.join(path, "%s-labels-idx1-ubyte.gz" % kind)
    images_path = os.path.join(path, "%s-images-idx3-ubyte.gz" % kind)

    with gzip.open(labels_path, "rb") as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)

    with gzip.open(images_path, "rb") as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(
            len(labels), 784
        )

    return images, labels


def get_fashion(num_points: int) -> dict:
    images, labels = load_mnist(r"D:\\Data\\ML\\fashion", kind="train")
    colors, unique_colors, color_key = label_to_colors(labels[:num_points])
    return {
        "X": images[:num_points],
        "label": labels[:num_points],
        "col_key": color_key,
    }
