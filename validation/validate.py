import pandas as pd
import torch
from scipy import stats
import numpy as np
import sys
import os
import pathlib
import pytest

sys.path.append(str(pathlib.Path(__file__).parents[1] / "./utils"))
print(f"sys path: {sys.path}")
from nnp_util import spearmanr_torch


def test_spearman():
    df = pd.read_csv(
        pathlib.Path(__file__).parents[1] / "./data/dsExerciseWeightLoss.csv"
    )

    # correlated data - numpy arrays
    ex = df["Exercise"].to_numpy()
    wl = df["WeightLoss"].to_numpy()

    # correlated data - torch tensors
    ex_t = torch.tensor(df["Exercise"].to_numpy(), device="cuda", dtype=torch.float32)
    wl_t = torch.tensor(df["WeightLoss"].to_numpy(), device="cuda", dtype=torch.float32)

    scipy_sr = stats.spearmanr(ex, wl)
    tensor_sr = spearmanr_torch(ex_t, wl_t)
    print(f"{scipy_sr[0]}")
    print(f"{tensor_sr}")
    assert scipy_sr[0] == pytest.approx(float(tensor_sr.cpu()))
    print("test_spearman passed")


if __name__ == "__main__":
    test_spearman()
