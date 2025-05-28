from shaders.shader import BoundsShader
import numpy as np
from kp import Manager

bounds_shader = BoundsShader()
# 1M digits in the range 0-128
points = np.random.rand(2**16, 2).astype(np.float32) * 128
print(f"Points {points}")
print(f"Points max {np.max(points)} min {np.min(points)}")

mgr = Manager()
bounds = bounds_shader.compute(
    mgr=mgr,
    num_points=points.shape[0],
    padding=0.1,
    points=points,
)
print(f"Bounds {bounds}")
