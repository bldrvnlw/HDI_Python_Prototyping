from shaders.shader import (
    BoundsShader,
    StencilShader,
    FieldComputationShader,
    InterpolationShader,
    ForcesShader,
    UpdateShader,
    CenterScaleShader,
)
import sys
import math
import time
import numpy as np
from kp import Manager
import umap
from nptsne import TextureTsne, KnnAlgorithm
from optimize.nn_points_torch import NNPointsTorch
from itertools import chain

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# from openTSNE import affinity
from utils.prob_utils import (
    compute_annoy_distances,
    compute_hnsw_distances,
    euclidian_sqrdistance_matrix,
    compute_perplexity_probs_numba,
    get_random_uniform_circular_embedding,
    getProbabilitiesOpenTSNE,
    symmetrize_P,
    calculate_normalization_Q,
    compute_Qnorm_cuda,
)

from utils.metrics import compute_coranking_matrix, rnx_auc_crm

from utils.nnp_util import (
    compute_nnp,
    neighborhood_preservation_torch,
    neighborhood_hit_torch,
    get_spearman_and_stress,
    trustworthiness_torch,
    continuity_torch,
)

from shaders.persistent_tensors import (
    LinearProbabilityMatrix,
    PersistentTensors,
    ShaderBuffers,
)

from utils.data_sources import (
    get_generated,
    get_MNIST,
    get_mouse_Zheng,
    get_hypomap,
    get_xmas_tree,
    get_wikiword_350000,
    get_word2vec,
    get_coil20,
    get_frey_faces,
    get_fashion,
)

from utils.base import pairwise_l2_distances

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from sklearn.manifold import trustworthiness

bounds_shader = BoundsShader()
stencil_shader = StencilShader()
fields_shader = FieldComputationShader()
interpolation_shader = InterpolationShader()
forces_shader = ForcesShader()
update_shader = UpdateShader()
centerscale_shader = CenterScaleShader()
# 1M digits in the range 0-128

num_iterations = 1000  # was 1000

# data = get_xmas_tree()
# num_points = data["X"].shape[0]  # all in xmas_tree
# perplexity = 13
# num_iterations = 500

### Hypomap
# perplexity = 50  # was 30
# num_points = 433369  # was 433369
# data = get_hypomap(num_points=num_points)

### Zheng
# perplexity = 30  # was 30
# num_points = 1306127  # all in Zheng 1306127
# data = get_mouse_Zheng(num_points=num_points)

### MNIST
# perplexity = 30  # was 30
# num_points = 60000  # was 70000 all in MNIST
# data = get_MNIST(num_points=num_points)

### Wikiword
# perplexity = 50  # was 50
# num_points = 350000  # was 350000
# data = get_wikiword_350000(num_points=num_points)

### Word2vec
# perplexity = 80  # was
# num_points = 1000000  # was 3000000
# data = get_word2vec(num_points=num_points)

# coil-20
# perplexity = 30  # was 30
# num_points = 1400  # was 1400
# data = get_coil20(num_points=num_points)

# frey_faces
# perplexity = 30  # was 30
# num_points = 1965  # was 1965
# data = get_frey_faces(num_points=num_points)

# fashion MNIST like but more challenging
perplexity = 30  # was 30
num_points = 60000  # was 60000
data = get_fashion(num_points)

X = data["X"]
# Things to do
metrics = False  # calculate the metrics
reuse_nptsne_distributions = True  # Use the distrubutions from nptsne
graphics_hv = True  # Display plots using HoloViews + datashader


knn_algorithm = "HNSW"  # Annoy or HNSW
# randomly initialize the embedding
points = get_random_uniform_circular_embedding(num_points, 0.1)

perplexity_multiplier = 3  # was 3
nn = perplexity * perplexity_multiplier

# perform old nptsne
tsne = TextureTsne(True, perplexity=perplexity, knn_algorithm=KnnAlgorithm.HNSW)
embed_nptsne = tsne.fit_transform(X)
embed_nptsne = embed_nptsne.reshape(num_points, 2)
tembed_nptsne = NNPointsTorch(embed_nptsne)
nptsne_klvalues = tsne.kl_values

nptsne_probabilities = tsne.transition_matrix

num_rows = len(nptsne_probabilities)
distrib_matrix = None

if reuse_nptsne_distributions:
    neigh_rows = []
    prob_rows = []
    nptsne_indices = np.zeros(num_rows * 2, dtype=int)
    offset = 0
    for pos, point_prob_tups in enumerate(nptsne_probabilities):
        nptsne_neigh, nptsne_prob = zip(*point_prob_tups)
        neigh_rows.append(nptsne_neigh)
        prob_rows.append(nptsne_prob)
        nptsne_indices[pos * 2] = offset
        nptsne_indices[pos * 2 + 1] = len(nptsne_neigh)
        offset += len(nptsne_neigh)

    nptsne_prob_np = np.fromiter(chain.from_iterable(prob_rows), dtype=float)
    nptsne_neigh_np = np.fromiter(chain.from_iterable(neigh_rows), dtype=int)

    distrib_matrix = LinearProbabilityMatrix(
        neighbours=nptsne_neigh_np,
        probabilities=nptsne_prob_np,
        indices=nptsne_indices,
    )

else:
    if knn_algorithm == "Annoy":
        print("Using Annoy")
        distances, neighbours, indices = compute_annoy_distances(
            data=X,
            num_trees=int(math.sqrt(num_points)),
            nn=nn,
        )
    else:
        print("Using HNSW")
        distances, neighbours, indices = compute_hnsw_distances(data=X, nn=nn)

    # # print(f"dist {distances.shape} distances {distances}")
    # # print(f"indices {indices.shape} indices {indices}")
    # # print(f"neigh {neighbours.shape} neighbours {neighbours}")

    # # Compute the perplexity probabilities
    P, sigmas = compute_perplexity_probs_numba(distances, perplexity=perplexity)
    # Symmetrize and flatten to indexes 1D arrays
    neighbours, probabilities, indices = symmetrize_P(P, neighbours, nn)
    print(
        "Ratio of symmetrized High Dimensional Probability to num_points\n"
        f"(should be approx 1): {probabilities.sum()/num_points}"
    )
    # # # print(f"Perplexity matrix {P.shape} sigmas {sigmas.shape}")

    # Or Using OpenTSNE
    # neighbours, probabilities, indices = getProbabilitiesOpenTSNE(X, perplexity=perplexity)
    # Create a manager for the shaders
    # and persistent buffers

    distrib_matrix = LinearProbabilityMatrix(
        neighbours=neighbours,
        probabilities=probabilities,
        indices=indices,
    )


mgr = Manager()
# Create the persistent buffers
persistent_tensors = PersistentTensors(
    mgr=mgr, num_points=num_points, prob_matrix=distrib_matrix
)


# for all iterations

start_exaggeration = 4.0  # was 4.0  # should decay at a certain point
end_exaggeration = 1.0
decay_start = 250  # was 250
decay_length = 150  # was 201500

vulkan_klvalues = np.zeros((num_iterations,))

# plt.figure(0)
# plt.scatter(points[:, 0], points[:, 1], c=colors, alpha=0.7)
a_num_points = np.array([num_points], dtype=np.uint32)
persistent_tensors.set_tensor_data(ShaderBuffers.NUM_POINTS, a_num_points)


print("Starting GPU iterations")
import time

start = time.time()
for i in range(num_iterations):
    exaggeration = start_exaggeration
    if i > decay_start and i < decay_start + decay_length:
        decay_fraction = float(i - decay_start) / float(decay_length)
        decay_range = start_exaggeration - end_exaggeration
        exaggeration = start_exaggeration - (decay_fraction * decay_range)
    elif i >= decay_start + decay_length:
        exaggeration = 1.0

    # print("**********************************************************")
    # print(f"iteration number: {i} Exaggeration factor: {exaggeration}")
    bounds = bounds_shader.compute(
        mgr=mgr,
        num_points=num_points,
        padding=0.1,
        points=points,
        persistent_tensors=persistent_tensors,
    )
    # print(f"Bounds {bounds}")

    MINIMUM_FIELDS_SIZE = 5
    RESOLUTION_SCALING = 2
    range_x = abs(bounds[1][0] - bounds[0][0])
    range_y = abs(bounds[1][1] - bounds[0][1])

    # assume adaptive resolution (scales with points range) with a minimum size
    width = int(max(RESOLUTION_SCALING * range_x, MINIMUM_FIELDS_SIZE))
    height = int(max(RESOLUTION_SCALING * range_y, MINIMUM_FIELDS_SIZE))

    # This width and height is used for the size of the point "plot"

    # print(f"Bounds range + resolution scaling: Width {width} Height {height}")
    stencil = stencil_shader.compute(
        mgr=mgr,
        width=width,
        height=height,
        num_points=num_points,
        persistent_tensors=persistent_tensors,
    )
    # print(f"Stencil shape {stencil.shape} dtype {stencil.dtype}")
    # print(stencil)
    # matrix_viewer.view(stencil[..., 0])
    # matrix_viewer.show_with_pyplot()
    fields = fields_shader.compute(
        mgr=mgr,
        num_points=num_points,
        stencil=stencil,
        width=width,
        height=height,
        persistent_tensors=persistent_tensors,
    )

    # print(f"Fields shape {fields.shape} dtype {fields.dtype}")

    interpolation_shader.compute(
        mgr=mgr,
        num_points=num_points,
        fields=fields,
        width=width,
        height=height,
        persistent_tensors=persistent_tensors,
    )

    if interpolation_shader.sumQ[0] == 0:
        print("!!!! Breaking out due to interpolation sum 0 !!!!")
        break

    # q_norm = calculate_normalization_Q(points)
    # q_norm = compute_Qnorm_cuda(points)
    # print(f"q_norm:  {q_norm}")
    # q_norm = interpolation_shader.sumQ[0]
    sum_kl = forces_shader.compute(
        mgr=mgr,
        num_points=num_points,
        exaggeration=exaggeration,
        persistent_tensors=persistent_tensors,
    )
    vulkan_klvalues[i] = sum_kl
    print(f"Iteration {i} Sum KL {sum_kl} ")
    if i % 1 == 0:
        pass
        #

    update_shader.compute(
        mgr=mgr,
        num_points=num_points,
        eta=200.0,
        minimum_gain=0.1,
        iteration=i,
        momentum=0.2,  # was 0.2
        momentum_switch=250,  # was 250,
        momentum_final=0.5,  # was 0.5,
        gain_multiplier=4.0,
        persistent_tensors=persistent_tensors,
    )
    updated_points = persistent_tensors.get_tensor_data(ShaderBuffers.POSITION)
    # print(f"New points {updated_points}")
    bounds = bounds_shader.compute(
        mgr=mgr,
        num_points=num_points,
        padding=0.1,
        points=updated_points,  # skip this because it is in the tensor
        persistent_tensors=persistent_tensors,
    )
    updated_bounds = persistent_tensors.get_tensor_data(ShaderBuffers.BOUNDS)
    # print(f"Updated bounds after point move {updated_bounds}")
    # if exaggeration <= 1.2:
    #    print("Breaking for low exaggeration")
    #    break
    centerscale_shader.compute(
        mgr=mgr,
        num_points=num_points,
        exaggeration=exaggeration,
        persistent_tensors=persistent_tensors,
    )
    # get the updated points
    points = persistent_tensors.get_tensor_data(ShaderBuffers.POSITION).reshape(
        num_points, 2
    )
    # print(f"Centered points {updated_points}")

    # xy = points.reshape(num_points, 2)
    # if (i - 1) % 50 == 0:
    #    plt.figure(i)
    #    plt.scatter(xy[:, 0], xy[:, 1], c=colors, alpha=0.7)
    #    plt.show(block=False)
    # time.sleep(0.5)

end = time.time()
print(f"Elapsed time {end-start}")
points = persistent_tensors.get_tensor_data(ShaderBuffers.POSITION)
xy = points.reshape(num_points, 2)
print(f"xy.shape {xy.shape}")

tXY = NNPointsTorch(xy)
tX = NNPointsTorch(X)

# perform UMAP
reducer = umap.UMAP(n_neighbors=perplexity * perplexity_multiplier)
UMAPembedding = reducer.fit_transform(X)
tUMAPembedding = NNPointsTorch(UMAPembedding)


if metrics:
    NNP_VK = neighborhood_preservation_torch(tX, tXY, nr_neighbors=90)
    NNP_UMAP = neighborhood_preservation_torch(
        embed=tUMAPembedding, X=tX, nr_neighbors=90
    )
    NNP_NPTSNE = neighborhood_preservation_torch(
        embed=tembed_nptsne, X=tX, nr_neighbors=90
    )

    y = data["y"]
    if y is not None:
        int_labels = y.astype(np.int32)
        NHIT_VK = neighborhood_hit_torch(xy, int_labels, nr_neighbors=90)
        NHIT_UMAP = neighborhood_hit_torch(UMAPembedding, int_labels, nr_neighbors=90)
        NHIT_NPTSNE = neighborhood_hit_torch(embed_nptsne, int_labels, nr_neighbors=90)
    else:
        NHIT_VK = ""
        NHIT_UMAP = ""
        NHIT_NPTSNE = ""

    tDX = NNPointsTorch(tX.get_pairwise_l2_distances().cpu().numpy())
    tDXY = NNPointsTorch(tXY.get_pairwise_l2_distances().cpu().numpy())

    tDUMAPembedding = NNPointsTorch(
        tUMAPembedding.get_pairwise_l2_distances().cpu().numpy()
    )
    tDembed_nptsne = NNPointsTorch(
        tembed_nptsne.get_pairwise_l2_distances().cpu().numpy()
    )
    SPEARMANR_VK, STRESS_VK = get_spearman_and_stress(tDX, tDXY)
    SPEARMANR_UMAP, STRESS_UMAP = get_spearman_and_stress(tDX, tDUMAPembedding)
    SPEARMANR_NPTSNE, STRESS_NPTSNE = get_spearman_and_stress(tDX, tDembed_nptsne)

    TRUST_VK = trustworthiness_torch(tX, tXY)
    TRUST_UMAP = trustworthiness_torch(tX, tUMAPembedding)
    TRUST_NPTSNE = trustworthiness_torch(tX, tembed_nptsne)

    CONT_VK = continuity_torch(tX, tXY)
    CONT_UMAP = continuity_torch(tX, tUMAPembedding)
    CONT_NPTSNE = continuity_torch(tX, tembed_nptsne)

    print("**************")
    print(f"NNP value VK: {NNP_VK}")  # approx same as area under RNX curve
    print(f"NHIT value VK: {NHIT_VK}")
    print(f"SPEARMAN-R value VK: {SPEARMANR_VK}")
    print(f"STRESS value VK: {STRESS_VK}")
    print(f"TRUSTWORTHINESS value VK: {TRUST_VK}")
    print(f"CONTINUITY value VK: {CONT_VK}")
    print("**************")
    print(f"NNP value UMAP: {NNP_UMAP}")
    print(f"NHIT value UMAP: {NHIT_UMAP}")
    print(f"SPEARMAN-R value UMAP: {SPEARMANR_UMAP}")
    print(f"STRESS value UMAP: {STRESS_UMAP}")
    print(f"TRUSTWORTHINESS value VK: {TRUST_UMAP}")
    print(f"CONTINUITY value VK: {CONT_UMAP}")
    print("**************")
    print(f"NNP value nptsne: {NNP_NPTSNE}")
    print(f"NHIT value nptsne: {NHIT_NPTSNE}")
    print(f"SPEARMAN-R value nptsne: {SPEARMANR_NPTSNE}")
    print(f"STRESS value nptsne: {STRESS_NPTSNE}")
    print(f"TRUSTWORTHINESS value VK: {TRUST_NPTSNE}")
    print(f"CONTINUITY value VK: {CONT_NPTSNE}")
    print("**************")

    neighbors = [int(perplexity / 3), perplexity, perplexity * 3, perplexity * 9]
    trust = [trustworthiness_torch(tX, tXY, n_neighbors=int(k)) for k in neighbors]
    print(f"Trustworthiness VK for neighbors {neighbors} : {trust}")

# QNX = compute_coranking_matrix(data_ld=xy, data_hd=X)


if graphics_hv:
    alpha = 1 / math.log10(num_points)
    size = 5 / math.log10(num_points)
    import plotly.io as pio
    import holoviews as hv
    from holoviews import opts
    from holoviews.operation.datashader import datashade, dynspread
    import datashader as ds
    import datashader.transfer_functions as tf
    from datashader.mpl_ext import dsshow, alpha_colormap
    import pandas as pd
    from mpl_toolkits.axes_grid1 import ImageGrid, Grid
    from functools import partial
    import panel as pn

    hv.extension("bokeh")
    pn.extension()

    dynspread.max_px = 15
    dynspread.threshold = 0.5

    # Sliders for dynspread parameters
    # point_size_slider = pn.widgets.IntSlider(
    #     name="Point Size", start=1, end=20, value=5
    # )
    # max_px_slider = pn.widgets.IntSlider(
    #     name="Max Spread (px)", start=1, end=20, value=5
    # )
    threshold_slider = pn.widgets.FloatSlider(
        name="Threshold", start=0.0, end=1.0, step=0.05, value=0.5
    )

    # pio.renderers.default = "png"

    alpha = 1 / math.log10(num_points)
    size = 10 / math.log10(num_points)
    no_categories = len(data["col_key"]) == 1
    # ax = make_subplots(rows=1, cols=3, subplot_titles=("UMAP", "nptsne", "vulkan"))
    # fig, axs = plt.subplots(nrows=1, ncols=3)
    # dataframes comprise the x and y coords and the labels
    df1 = pd.DataFrame(
        dict(xe=UMAPembedding[:, 0], ye=UMAPembedding[:, 1], label=data["label"])
    )
    df2 = pd.DataFrame(
        dict(xe=embed_nptsne[:, 0], ye=embed_nptsne[:, 1], label=data["label"])
    )
    df3 = pd.DataFrame(dict(xe=xy[:, 0], ye=xy[:, 1], label=data["label"]))

    # print(df1["label"].unique())
    # print(df2["label"].unique())
    # print(df3["label"].unique())
    # print(data["col_key"].keys())
    # print(df1[["xe", "ye"]].isna().sum())
    # print(df2[["xe", "ye"]].isna().sum())
    # print(df3[["xe", "ye"]].isna().sum())
    pw = 600
    ph = 600

    @pn.depends(threshold_slider)
    def create_composition(threshold):
        overlay1 = hv.NdOverlay(
            {
                label: hv.Points(
                    group,
                    kdims=["xe", "ye"],
                    label="umap",
                )
                for label, group in df1.groupby("label")
            }
        )
        sct1 = dynspread(
            datashade(
                overlay1,
                aggregator=ds.count() if no_categories else "count_cat",
                color_key=data["col_key"],
            ).opts(width=pw, height=ph),
            threshold=threshold,
        )
        # print(f"overlay keys: {overlay1.keys()}")
        # print(f"color key keys: {data["col_key"].keys()}")

        overlay2 = hv.NdOverlay(
            {
                label: hv.Points(
                    group,
                    kdims=["xe", "ye"],
                    label="nptsne",
                )
                for label, group in df2.groupby("label")
            }
        )
        sct2 = dynspread(
            datashade(
                overlay2,
                aggregator=ds.count() if no_categories else "count_cat",
                color_key=data["col_key"],
            ).opts(width=pw, height=ph),
            threshold=threshold,
        )

        overlay3 = hv.NdOverlay(
            {
                label: hv.Points(
                    group,
                    kdims=["xe", "ye"],
                    label="vulkan",
                )
                for label, group in df3.groupby("label")
            }
        )
        sct3 = dynspread(
            datashade(
                overlay3,
                aggregator=ds.count() if no_categories else "count_cat",
                color_key=data["col_key"],
            ).opts(width=pw, height=ph),
            threshold=threshold,
        )

        empty_umap = hv.Text(0.5, 0.5, "No KL curve for umap").opts(
            width=pw, height=ph, xaxis=None, yaxis=None, bgcolor="white"
        )

        nptsne_kl_points = [(i, nptsne_klvalues[i]) for i in range(num_iterations)]
        nptsne_kl_curve = hv.Curve(nptsne_kl_points).opts(width=pw, height=ph)

        vulkan_kl_points = [(i, vulkan_klvalues[i]) for i in range(num_iterations)]
        vulkan_kl_curve = hv.Curve(vulkan_kl_points).opts(width=pw, height=ph)

        composition = (
            (sct1 + sct2 + sct3 + empty_umap + nptsne_kl_curve + vulkan_kl_curve)
            .opts(shared_axes=False)
            .cols(3)
        )
        return composition

    layout = pn.Row(
        create_composition,
        pn.Column(threshold_slider),
    )
    pn.panel(layout).show()


input("Press enter to finish...")
mgr.destroy()
print("Iterations complete")
