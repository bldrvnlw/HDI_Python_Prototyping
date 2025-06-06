from .hdi_shader_source import HDIShaderSource
import subprocess
import kp
import numpy as np
import math
import sys
from typing import Annotated


class Shader:
    def __init__(self, name: str, shader_name: str):
        self.name = name
        self.code = HDIShaderSource()[shader_name]

    def __repr__(self):
        return f"Shader(name={self.name}, code={self.code})"

    def __str__(self):
        return f"Shader: {self.name}\nCode:\n{self.code}"

    def compile(self):
        open("tmp_kp_shader.glsl.comp", "w").write(self.code)
        try:
            stdout = subprocess.check_output(
                [
                    "glslangValidator",
                    "-V",
                    "-e",
                    "main",
                    "--target-env",
                    "vulkan1.3",
                    "tmp_kp_shader.glsl.comp",
                    "-o",
                    "tmp_kp_shader.glsl.comp.spv",
                ],
                stderr=subprocess.STDOUT,
            )
        except subprocess.CalledProcessError as err:
            e = "Could not compile hlsl to Spir-V:\n" + err.output.decode()
            raise Exception(e)

        self.spv = open("tmp_kp_shader.glsl.comp.spv", "rb").read()


class BoundsShader:
    def __init__(self):
        self.shader_code = Shader("Bounds Shader", "bounds")
        self.shader_code.compile()

    def compute(
        self,
        mgr: kp.Manager,
        num_points: int,
        padding: float,
        points: np.array,
    ):
        """_summary_

        Args:
            num_points (_type_): _description_
            padding (_type_): _description_
            points (_type_): _description_
        """
        #
        points_in = mgr.tensor(points)
        # print(points_in.data(), points_in.data().shape)
        bounds_out = mgr.tensor(np.zeros((1, 4), dtype=np.float32))
        # print(bounds_out.data(), bounds_out.data().shape)

        params = [bounds_out, points_in]
        workgroup = (128, 1, 1)
        algorithm = mgr.algorithm(
            tensors=params,
            spirv=self.shader_code.spv,
            workgroup=workgroup,
            push_consts=[num_points, padding],
        )
        seq = mgr.sequence()
        seq.record(kp.OpSyncDevice(params))
        seq.record(kp.OpAlgoDispatch(algorithm))
        seq.record(kp.OpSyncLocal([bounds_out]))
        seq.eval()
        return bounds_out.data().reshape(2, 2)


class StencilShader:
    def __init__(self):
        self.shader_code = Shader("Stencil Shader", "stencil")
        self.shader_code.compile()

    def compute(
        self,
        mgr: kp.Manager,
        bound_min: Annotated[np.ndarray, "float32", "2"],
        bound_max: Annotated[np.ndarray, "float32", "2"],
        width: int,
        height: int,
        points: np.array,
    ):
        print(bound_min, bound_max)
        print(f"Number of points: {points.shape[0]}")
        np.set_printoptions(threshold=sys.maxsize)
        # print(f"Points  {points}")
        stencil_np_array = np.zeros((height, width, 4), dtype=np.uint8)
        stencil_out = mgr.image(stencil_np_array, width, height, 4)
        points_in = mgr.tensor(points)
        print(f"Tensor points in shape: {points_in.size()}")
        push_constants = [
            bound_min[0],
            bound_min[1],
            bound_max[0],
            bound_max[1],
            float(width),
            float(height),
        ]

        algorithm = mgr.algorithm(
            tensors=[
                points_in,
                stencil_out,
            ],  # The stencil_tensor is the only output parameter
            spirv=self.shader_code.spv,
            workgroup=[points.shape[0], 1, 1],  # defaults to tensor[0].size,1,1
            push_consts=push_constants,
        )
        seq = mgr.sequence()
        seq.record(kp.OpSyncDevice([points_in, stencil_out]))
        seq.record(kp.OpAlgoDispatch(algorithm))
        seq.record(kp.OpSyncLocal([stencil_out]))
        seq.eval()
        generated_stencil = (
            np.array(stencil_out.data()).reshape(height, width, 4).astype(np.uint8)
        )
        print(f"Generated stencil shape: {generated_stencil.shape}")
        print(f"Generated stencil dtype: {generated_stencil.dtype}")

        # print(f"Stencil: {generated_stencil[..., 0]}")  # Print only the first channel
        np.savetxt("stencil_output.txt", generated_stencil[..., 0], fmt="%i")
        # input("Press Enter to continue...")

        return generated_stencil


class FieldComputationShader:
    def __init__(self):
        self.shader_code = Shader("Field Computation Shader", "compute_fields")
        self.shader_code.compile()
        self.kernel_radius: float = 32.0
        self.kernel_width: int = int(self.kernel_radius * 2 + 1)
        self.TSNEKernel: np.array = np.zeros(
            (self.kernel_width, self.kernel_width, 4), dtype=np.float32
        )
        self.func_support: float = 6.5
        # self.generateRasterTSNEKernel(self.func_support)
        self.field_texture: np.array = None

    def generateRasterTSNEKernel(self, func_support: float):

        for i in range(self.kernel_width):
            for j in range(self.kernel_width):
                x = (i - self.kernel_radius) / (self.kernel_radius * func_support)
                y = (j - self.kernel_radius) / (self.kernel_radius * func_support)
                tstud = 1.0 / (1.0 + x * x + y * y)
                off = (j * self.kernel_width + i) * 4
                self.TSNEKernel[i, j, 0] = tstud
                self.TSNEKernel[i, j, 1] = tstud * tstud * x
                self.TSNEKernel[i, j, 2] = tstud * tstud * y
                self.TSNEKernel[i, j, 3] = 0

    def compute(
        self,
        mgr: kp.Manager,
        points: np.array,
        bounds: np.array,
        stencil: np.array,
        width: int,
        height: int,
        position_buffer: int,
    ):
        # np.set_printoptions(threshold=sys.maxsize)
        # print(f"TSNEKernel shape: {self.TSNEKernel.shape}")
        # print(f"TSNEKernel dtype: {self.TSNEKernel.dtype}")
        # print(f"TSNEKernel: {self.TSNEKernel}")
        field_out = mgr.image(
            np.zeros((height, width, 4), dtype=np.float32), width, height, 4
        )
        points_in = mgr.tensor(points)
        bounds_in = mgr.tensor(bounds)
        stencil_in = mgr.image(stencil, width, height, 4)
        params = [points_in, bounds_in, field_out, stencil_in]
        print(f"Num points: {points.shape[0]}")
        push_constants = [
            float(points.shape[0]),
            float(width),
            float(height),
            self.func_support,
        ]
        algorithm = mgr.algorithm(
            tensors=params,  # The stencil_tensor is the only output parameter
            spirv=self.shader_code.spv,
            workgroup=[width, height, 1],  # Global dispatch 1 thread per pixel
            push_consts=push_constants,
        )
        seq = mgr.sequence()
        seq.record(kp.OpSyncDevice(params))
        seq.record(kp.OpAlgoDispatch(algorithm))
        seq.record(kp.OpSyncLocal([field_out]))
        seq.eval()

        print(f"Field shape {field_out.data().shape} ")
        generated_field = (
            np.array(field_out.data()).reshape(height, width, 4).astype(np.float32)
        )
        np.savetxt("field_output.txt", generated_field[..., 1], fmt="%f")
        # print(f"Field data {field_out.data()}")
        return generated_field


class InterpolationShader:
    def __init__(self):
        self.shader_code = Shader("Interpolation Shader", "interp_fields")
        self.shader_code.compile()

    def compute(
        self,
        mgr: kp.Manager,
        points: np.array,
        bounds: np.array,
        fields: np.array,
        width: int,
        height: int,
    ):

        self.interp_fields: np.array = np.zeros((points.shape[0], 4), dtype=np.float32)
        interp_fields_out = mgr.tensor(self.interp_fields)
        self.sumQ = np.zeros((1), dtype=np.float32)
        sum_out = mgr.tensor(self.sumQ)

        points_in = mgr.tensor(points)
        fields_in = mgr.tensor(np.reshape(fields, (height * width, 4)))

        params = [points_in, fields_in, interp_fields_out, sum_out]
        print(f"Width height: {width} : {height}")
        push_constants = [
            float(bounds[0, 0]),
            float(bounds[0, 1]),
            float(bounds[1, 0]),
            float(bounds[1, 1]),
            float(width),
            float(height),
            float(points.shape[0]),
        ]
        algorithm = mgr.algorithm(
            tensors=params,  # The stencil_tensor is the only output parameter
            spirv=self.shader_code.spv,
            workgroup=[1, 1, 1],  # Global dispatch 1 thread per pixel
            push_consts=push_constants,
        )
        seq = mgr.sequence()
        seq.record(kp.OpSyncDevice(params))
        seq.record(kp.OpAlgoDispatch(algorithm))
        seq.record(kp.OpSyncLocal([interp_fields_out, sum_out]))
        seq.eval()

        self.interp_fields = (
            np.array(interp_fields_out.data())
            .reshape(points.shape[0], 4)
            .astype(np.float32)
        )

        self.sumQ = sum_out.data()
        # return (interp_fields, )
        print(f"Sum: {self.sumQ}")
        print(f"Interpolation {self.interp_fields} ")


class ForcesShader:
    def __init__(self):
        self.shader_code = Shader("Forces Shader", "compute_forces")
        self.shader_code.compile()

    def compute(
        self,
        mgr: kp.Manager,
        points: np.array,
        bounds: np.array,
        fields: np.array,
        width: int,
        height: int,
    ):
        pass
