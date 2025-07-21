# wget the raw data
# wget2 https://datasets.cellxgene.cziscience.com/fb9c1950-d7c0-4a48-9475-a165edeab79c.h5ad -O hypomap.h5ad

import scanpy as sc
import pickle

hypomap = sc.read_h5ad(r"C:\\Users\\bvanlew\\Downloads\\hypomap.h5ad")
hmPCA = sc.pp.pca(hypomap.X)
C0_name = hypomap.obs["C0_named"].to_list()
C1_name = hypomap.obs["C1_named"].to_list()
C2_name = hypomap.obs["C2_named"].to_list()
C3_name = hypomap.obs["C3_named"].to_list()
C4_name = hypomap.obs["C4_named"].to_list()

data_dict = {
    "pca_50": hmPCA,
    "C0_name": C0_name,
    "C1_name": C1_name,
    "C2_name": C2_name,
    "C3_name": C3_name,
    "C4_name": C4_name,
}

with open(r"C:\\Users\\bvanlew\\Downloads\\hypomap.pkl", "wb") as f:
    pickle.dump(data_dict, f)
