from shaders.shader import (
    BoundsShader,
    StencilShader,
    FieldComputationShader,
    InterpolationShader,
)
import sys
import numpy as np
from kp import Manager
from prob_utils import (
    compute_annoy_probabilities,
    euclidian_sqrdistance_matrix,
    compute_perplexity_probs,
    symmetrize_P,
)

bounds_shader = BoundsShader()
stencil_shader = StencilShader()
fields_shader = FieldComputationShader()
interpolation_shader = InterpolationShader()
# 1M digits in the range 0-128
points = (np.random.rand(2**12, 2).astype(np.float32) * 128) - (64, 64)
print(f"Points {points}")
print(f"Points max {np.max(points)} min {np.min(points)}")

perplexity = 30
perplexity_multiplier = 3
nn = perplexity * perplexity_multiplier + 1

indices, distances = compute_annoy_probabilities(
    data=points,
    num_trees=4,
    nn=nn,
)

print(f"distances {distances}")
exit(0)

# D = euclidian_sqrdistance_matrix(points)
# Compute the perplexity probabilities
(P, sigmas) = compute_perplexity_probs(distances, perplexity=perplexity)
P_s = symmetrize_P(P)

print(f"Perplexity matrix {P.shape} sigmas {sigmas.shape}")
print(f"Perplexity matrix {P}")
print(f"sigmas {sigmas}")
exit(0)

mgr = Manager()

# for all iterations
for i in range(1):
    bounds = bounds_shader.compute(
        mgr=mgr,
        num_points=points.shape[0],
        padding=0.1,
        points=points,
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
        bound_min=bounds[0],
        bound_max=bounds[1],
        width=width,
        height=width,
        points=points,
    )
    print(f"Stencil shape {stencil.shape} dtype {stencil.dtype}")

    fields = fields_shader.compute(
        mgr=mgr,
        points=points,
        bounds=bounds,
        stencil=stencil,
        width=width,
        height=height,
        position_buffer=1,
    )

    print(f"Fields shape {fields.shape} dtype {fields.dtype}")

    interpolation_shader.compute(
        mgr=mgr,
        points=points,
        bounds=bounds,
        fields=fields,
        width=width,
        height=height,
    )

    if interpolation_shader.sumQ[0] == 0:
        break


mgr.destroy()
# stencil = stencil.reshape(height, width, 4)  # colours

# np.set_printoptions(threshold=sys.maxsize)
# print(f"Stencil {stencil}")
