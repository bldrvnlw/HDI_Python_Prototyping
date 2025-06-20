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
from openTSNE import affinity
from prob_utils import (
    compute_annoy_probabilities,
    euclidian_sqrdistance_matrix,
    compute_perplexity_probs_numba,
    symmetrize_probs,
    get_random_uniform_circular_embedding,
    getProbabilitiesOpenTSNE,
)
from shaders.persistent_tensors import (
    LinearProbabilityMatrix,
    PersistentTensors,
    ShaderBuffers,
)

from data_sources import get_generated, get_MNIST
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

bounds_shader = BoundsShader()
stencil_shader = StencilShader()
fields_shader = FieldComputationShader()
interpolation_shader = InterpolationShader()
forces_shader = ForcesShader()
update_shader = UpdateShader()
centerscale_shader = CenterScaleShader()
# 1M digits in the range 0-128

num_points = 20000
perplexity = 30
perplexity_multiplier = 3
nn = perplexity * perplexity_multiplier + 1

X, y, colors, unique_colors = get_MNIST(num_points=num_points)

# randomly initialize the embedding
points = get_random_uniform_circular_embedding(num_points, 0.1)

distances, neighbours, indices = compute_annoy_probabilities(
    data=X,
    num_trees=int(math.sqrt(num_points)),
    nn=nn,
)

# print(f"dist {distances.shape} distances {distances}")
# print(f"indices {indices.shape} indices {indices}")
# print(f"neigh {neighbours.shape} neighbours {neighbours}")

# Compute the perplexity probabilities
P, sigmas = compute_perplexity_probs_numba(distances, perplexity=perplexity)
P_sym = symmetrize_probs(P, neighbours, num_points, nn)

# print(f"Perplexity matrix {P.shape} sigmas {sigmas.shape}")

prob_matrix = LinearProbabilityMatrix(
    neighbours=np.ravel(neighbours),
    probabilities=np.ravel(P_sym),
    indices=indices,
)

# Or Using OpenTSNE
# neighbours, probabilities, indices = getProbabilitiesOpenTSNE(X, perplexity=perplexity)
# Create a manager for the shaders
# and persistent buffers


# prob_matrix = LinearProbabilityMatrix(
#    neighbours=neighbours,
#    probabilities=probabilities,
#    indices=indices,
# )


mgr = Manager()
# Create the persistent buffers
persistent_tensors = PersistentTensors(
    mgr=mgr, num_points=num_points, prob_matrix=prob_matrix
)


# for all iterations
num_iterations = 1000
start_exaggeration = 4.0  # should decay at a certain point
end_exaggeration = 1.0
decay_start = 250
decay_length = 200

# plt.figure(0)
# plt.scatter(points[:, 0], points[:, 1], c=colors, alpha=0.7)
a_num_points = np.array([num_points], dtype=np.uint32)
persistent_tensors.set_tensor_data(ShaderBuffers.NUM_POINTS, a_num_points)

print("Starting GPU iterations")
for i in range(num_iterations):
    exaggeration = start_exaggeration
    if i > decay_start and i < decay_start + decay_length:
        decay_fraction = float(i - decay_start) / float(decay_length)
        decay_range = start_exaggeration - end_exaggeration
        exaggeration = start_exaggeration - (decay_fraction * decay_range)
    elif i >= decay_start + decay_length:
        exaggeration = 1.0

    print("**********************************************************")
    print(f"iteration number: {i} Exaggeration factor: {exaggeration}")
    bounds = bounds_shader.compute(
        mgr=mgr,
        num_points=num_points,
        padding=0.1,
        points=points,
        persistent_tensors=persistent_tensors,
    )
    print(f"Bounds {bounds}")

    MINIMUM_FIELDS_SIZE = 5
    RESOLUTION_SCALING = 2
    range_x = bounds[1][0] - bounds[0][0]
    range_y = bounds[1][1] - bounds[0][1]

    # assume adaptive resolution (scales with points range) with a minimum size
    width = int(max(RESOLUTION_SCALING * range_x, MINIMUM_FIELDS_SIZE))
    height = int(max(RESOLUTION_SCALING * range_y, MINIMUM_FIELDS_SIZE))

    # This width and height is used for the size of the point "plot"

    print(f"Bounds range + resolution scaling: Width {width} Height {height}")
    stencil = stencil_shader.compute(
        mgr=mgr,
        width=width,
        height=height,
        num_points=num_points,
        persistent_tensors=persistent_tensors,
    )
    print(f"Stencil shape {stencil.shape} dtype {stencil.dtype}")
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

    print(f"Fields shape {fields.shape} dtype {fields.dtype}")

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

    forces_shader.compute(
        mgr=mgr,
        num_points=num_points,
        exaggeration=exaggeration,
        persistent_tensors=persistent_tensors,
    )

    update_shader.compute(
        mgr=mgr,
        num_points=num_points,
        eta=200.0,
        minimum_gain=0.1,
        iteration=i,
        momentum=0.2,
        momentum_switch=250,
        momentum_final=0.5,
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
    print(f"Updated bounds after point move {updated_bounds}")
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

    xy = points.reshape(num_points, 2)
    if (i - 1) % 50 == 0:
        plt.figure(i)
        plt.scatter(xy[:, 0], xy[:, 1], c=colors, alpha=0.7)
        plt.show(block=False)
        # time.sleep(0.5)


points = persistent_tensors.get_tensor_data(ShaderBuffers.POSITION)
xy = points.reshape(num_points, 2)
print(f"xy.shape {xy.shape}")
# mgr.destroy()
plt.figure(i)
scatter = plt.scatter(xy[:, 0], xy[:, 1], c=colors, alpha=0.7)
custom = [
    Line2D([], [], marker=".", markersize=10, color=unique_colors[i], linestyle="None")
    for i in range(0, 10)
]
plt.legend(custom, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], loc="lower left", title="Digits")
plt.show()
mgr.destroy()
print("Iterations complete")
