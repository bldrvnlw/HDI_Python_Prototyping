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
import numpy as np
from kp import Manager
from prob_utils import (
    compute_annoy_probabilities,
    euclidian_sqrdistance_matrix,
    compute_perplexity_probs,
    symmetrize_P,
    get_random_uniform_circular_embedding,
)
from shaders.persistent_tensors import (
    LinearProbabilityMatrix,
    PersistentTensors,
    ShaderBuffers,
)
from sklearn.datasets import make_classification

# a number of points for the test
num_points = 2**16

# Genrate some random clustered data
X, y = make_classification(
    n_features=1000,
    n_classes=10,
    n_samples=num_points,
    n_informative=4,
    random_state=5,
    n_clusters_per_class=1,
)

# randomly initialize the embedding
points = get_random_uniform_circular_embedding(num_points, 0.1)

bounds_shader = BoundsShader()
stencil_shader = StencilShader()
fields_shader = FieldComputationShader()
interpolation_shader = InterpolationShader()
forces_shader = ForcesShader()
update_shader = UpdateShader()
centerscale_shader = CenterScaleShader()
# 1M digits in the range 0-128


perplexity = 30
perplexity_multiplier = 3
nn = perplexity * perplexity_multiplier + 1

distances, neighbours, indices = compute_annoy_probabilities(
    data=X,
    num_trees=4,
    nn=nn,
)

# print(f"dist {distances.shape} distances {distances}")
# print(f"indices {indices.shape} indices {indices}")
# print(f"neigh {neighbours.shape} neighbours {neighbours}")

# D = euclidian_sqrdistance_matrix(points)
# Compute the perplexity probabilities
(P, sigmas) = compute_perplexity_probs(distances, perplexity=perplexity)
# P_s = symmetrize_P(P)
P_s = P  # skip symmetrization for now

print(f"Perplexity matrix {P.shape} sigmas {sigmas.shape}")
# print(f"Perplexity matrix {P}")
print(f"sigmas {sigmas}")

# Create a manager for the shaders
# and persistent buffers
mgr = Manager()

prob_matrix = LinearProbabilityMatrix(
    neighbours=neighbours,
    probabilities=P_s,
    indices=indices,
)

# Create the persistent buffers
persistent_tensors = PersistentTensors(
    mgr=mgr, num_points=num_points, prob_matrix=prob_matrix
)


# for all iterations
num_iterations = 500
print("Starting GPU iterations")
for i in range(num_iterations):
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

    width = RESOLUTION_SCALING * int(max(range_x, MINIMUM_FIELDS_SIZE))
    height = RESOLUTION_SCALING * int(max(range_y, MINIMUM_FIELDS_SIZE))

    print(f"Width {width} Height {height}")
    stencil = stencil_shader.compute(
        mgr=mgr,
        width=width,
        height=width,
        num_points=num_points,
        persistent_tensors=persistent_tensors,
    )
    print(f"Stencil shape {stencil.shape} dtype {stencil.dtype}")

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
        break

    exaggeration = 4.0  # should decay at a certain point
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
    print(f"New bounds {updated_bounds}")
    centerscale_shader.compute(
        mgr=mgr,
        num_points=num_points,
        exaggeration=exaggeration,
        persistent_tensors=persistent_tensors,
    )
    # get the updated points
    points = persistent_tensors.get_tensor_data(ShaderBuffers.POSITION)
    # print(f"Centered points {updated_points}")

print("Iterations complete")
mgr.destroy()
# stencil = stencil.reshape(height, width, 4)  # colours

# np.set_printoptions(threshold=sys.maxsize)
# print(f"Stencil {stencil}")
