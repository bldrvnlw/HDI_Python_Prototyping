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
# X, y, colors, unique_colors = get_xmas_tree()
# num_points = X.shape[0]  # all in xmas_tree
# perplexity = 15
# perplexity = 90  # was 30
# num_points = 50000  # was 433369
# X, y, colors, unique_colors = get_hypomap(num_points=num_points)
# num_points = 1306127  # all in Zheng 1306127
# X, y, colors, unique_colors = get_mouse_Zheng(num_points=num_points)
perplexity = 30  # was 30
num_points = 70000  # was 70000 all in MNIST
X, y, colors, unique_colors = get_MNIST(num_points=num_points)
# Wikiword
# perplexity = 90  # was 30
# num_points = 100000  # was 70000 all in MNIST
# X, y, colors, unique_colors = get_wikiword_350000(num_points=num_points)

metrics = False
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

nptsne_prob_np = np.array(prob_rows)
nptsne_prob_np = nptsne_prob_np.flatten()
nptsne_neigh_np = np.array(neigh_rows)
nptsne_neigh_np = nptsne_neigh_np.flatten()

prob_matrix = LinearProbabilityMatrix(
    neighbours=nptsne_neigh_np,
    probabilities=nptsne_prob_np,
    indices=nptsne_indices,
)

self_own_probs = True

if self_own_probs:
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

    prob_matrix = LinearProbabilityMatrix(
        neighbours=neighbours,
        probabilities=probabilities,
        indices=indices,
    )


mgr = Manager()
# Create the persistent buffers
persistent_tensors = PersistentTensors(
    mgr=mgr, num_points=num_points, prob_matrix=prob_matrix
)


# for all iterations

start_exaggeration = 4.0  # was 4.0  # should decay at a certain point
end_exaggeration = 1.0
decay_start = 250  # was 250
decay_length = 150  # was 201500

klvalues = np.zeros((num_iterations,))

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
    klvalues[i] = sum_kl
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

fig, axs = plt.subplots(figsize=(20, 20), nrows=2, ncols=3)

axs[0][0].scatter(
    UMAPembedding[:, 0],
    UMAPembedding[:, 1],
    c=colors,
    s=10 / math.log10(num_points),
    alpha=0.5,
)

if metrics:
    axs[0][0].set_title(f"UMAP nnp: {NNP_UMAP}")
axs[0][0].set_ylabel("embeddings: ", size="large")


axs[0][1].scatter(
    embed_nptsne[:, 0],
    embed_nptsne[:, 1],
    c=colors,
    s=10 / math.log10(num_points),
    alpha=0.5,
)
if metrics:
    axs[0][1].set_title(f"nptsne nnp: {NNP_NPTSNE}")


axs[0][2].scatter(
    xy[:, 0], xy[:, 1], c=colors, s=10 / math.log10(num_points), alpha=0.5
)

custom = [
    Line2D(
        [],
        [],
        marker=".",
        markersize=8.0,
        color=unique_colors[i],
        linestyle="None",
    )
    for i in range(0, len(unique_colors))
]
axs[0][2].legend(
    custom,
    [x for x in range(0, len(unique_colors))],
    loc="lower left",
    title="Cluster",
    prop={"size": 8},
)

if metrics:
    axs[0][2].set_title(f"HDILib VK nnp: {NNP_VK}")

axs[1][0].set_ylabel("KL-div: ", size="large")
axs[1][1].plot(range(0, num_iterations), nptsne_klvalues)
axs[1][2].plot(range(0, num_iterations), klvalues)

fig.tight_layout()
# can linit range to range(1,perplexity) but using definition from

# print(f"Area Under RNX Curve: {rnx_auc_crm(QNX)}")

plt.show()
input("Press enter to finish...")
mgr.destroy()
print("Iterations complete")
