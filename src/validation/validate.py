import sys
import os

# To debug set this in terminal
# $env:PYTHONPATH = "D:\TempProj\HDI_Python_Prototyping\src"
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

import pandas as pd
import torch
from scipy import stats
import numpy as np
import pathlib
import pytest
from sklearn.datasets import load_iris, make_classification, make_regression
from sklearn.manifold import trustworthiness
from optimize.nn_points_torch import NNPointsTorch
import umap

# from pyDRMetrics.pyDRMetrics import *

from utils.nnp_util import spearmanr_torch, trustworthiness_torch, continuity_torch


def test_spearman():
    df = pd.read_csv(
        pathlib.Path(__file__).parents[1] / "./data/dsExerciseWeightLoss.csv"
    )

    # correlated data - numpy arrays
    ex = df["Exercise"].to_numpy()
    wl = df["WeightLoss"].to_numpy()

    # correlated data - torch tensors
    ex_t = NNPointsTorch(df["Exercise"].to_numpy())
    wl_t = NNPointsTorch(df["WeightLoss"].to_numpy())

    scipy_sr = stats.spearmanr(ex, wl)
    tensor_sr = spearmanr_torch(ex_t, wl_t)
    print(f"{scipy_sr[0]}")
    print(f"{tensor_sr}")
    # accuracy test 1e-6
    assert scipy_sr[0] == pytest.approx(float(tensor_sr.cpu()))
    print("test_spearman passed")


def test_trustworthiness():
    X_train = load_iris().data
    reducer = umap.UMAP()
    UMAPembedding = reducer.fit_transform(X_train)
    trust = trustworthiness(X_train, UMAPembedding, n_neighbors=reducer.n_neighbors)
    cont = trustworthiness(UMAPembedding, X_train, n_neighbors=reducer.n_neighbors)
    print(f"scipy Trustworthiness: {trust}")
    X_train_t = NNPointsTorch(X_train)
    UMAPembedding_t = NNPointsTorch(UMAPembedding)
    trust_torch = trustworthiness_torch(
        X_train_t, UMAPembedding_t, n_neighbors=reducer.n_neighbors
    )
    print(f"torch Trustworthiness: {trust_torch}")
    rel_accuracy = 2e-4
    # accuracy test 2e-4
    assert trust == pytest.approx(trust_torch, rel=2e-4)
    print(f"test_trustworthiness passed at relative accuracy {rel_accuracy}")
    cont_torch = continuity_torch(
        X_train_t, UMAPembedding_t, n_neighbors=reducer.n_neighbors
    )
    print(f"scipy Continuity: {cont}")
    print(f"torch Continuity: {cont_torch}")
    assert cont == pytest.approx(cont_torch, rel=2e-4)
    print(f"test_continuity passed at relative accuracy {rel_accuracy}")
    # drm = DRMetrics(X_train, UMAPembedding)
    # print(f"pyDRMetrics : T {np.mean(drm.T)} C {np.mean(drm.C)}")


if __name__ == "__main__":
    # test_spearman()
    test_trustworthiness()
