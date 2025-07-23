from .hdi_shader_source import HDIShaderSource
import subprocess
import kp
import numpy as np
import math
import sys
from typing import Annotated
from shaders.persistent_tensors import PersistentTensors, ShaderBuffers
import time


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
                    "-S",
                    "comp",
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
        persistent_tensors: PersistentTensors,
    ):
        """_summary_

        Args:
            num_points (_type_): _description_
            padding (_type_): _description_
            points (_type_): _description_
        """
        #
        persistent_tensors.set_tensor_data(ShaderBuffers.POSITION, points)
        points_in = persistent_tensors.get_tensor(ShaderBuffers.POSITION)
        bounds_out = persistent_tensors.get_tensor(ShaderBuffers.BOUNDS)
        # debug = persistent_tensors.get_tensor(ShaderBuffers.POS_DEBUG)
        num_points_tensor = persistent_tensors.get_tensor(ShaderBuffers.NUM_POINTS)

        params = [bounds_out, points_in, num_points_tensor]
        algorithm = mgr.algorithm(
            tensors=params,
            spirv=self.shader_code.spv,
            workgroup=[1, 1, 1],
            push_consts=[padding],
        )
        (
            mgr.sequence()
            .record(kp.OpSyncDevice(params))
            .record(kp.OpAlgoDispatch(algorithm))
            .eval()
        )

        seq = mgr.sequence()
        seq.eval_async(kp.OpSyncLocal([bounds_out]))
        seq.eval_await()

        # dbg_data = persistent_tensors.get_tensor_data(ShaderBuffers.POS_DEBUG)
        return bounds_out.data().reshape(2, 2)


class StencilShader:
    def __init__(self):
        self.shader_code = Shader("Stencil Shader", "stencil")
        self.shader_code.compile()

    def compute(
        self,
        mgr: kp.Manager,
        width: int,
        height: int,
        num_points: int,
        persistent_tensors: PersistentTensors,
    ):
        bounds = persistent_tensors.get_tensor_data(ShaderBuffers.BOUNDS)
        points_in = persistent_tensors.get_tensor(ShaderBuffers.POSITION)
        debug = persistent_tensors.get_tensor(ShaderBuffers.POS_DEBUG)

        # print(f"bounds {bounds}")
        # np.set_printoptions(threshold=sys.maxsize)
        # print(f"Points  {points}")
        stencil_np_array = np.zeros((height, width, 4), dtype=np.uint8)
        stencil_out = mgr.image(stencil_np_array, width, height, 4)
        # print(f"Tensor points in shape: {points_in.size()}")
        # print(f"stencil size width: {width} height: {height}")
        push_constants = [
            bounds[0],
            bounds[1],
            bounds[2],
            bounds[3],
            float(width),
            float(height),
        ]

        algorithm = mgr.algorithm(
            tensors=[
                points_in,
                stencil_out,
                debug,
            ],  # The stencil_tensor is the only output parameter
            spirv=self.shader_code.spv,
            workgroup=[num_points, 1, 1],  # defaults to tensor[0].size,1,1
            push_consts=push_constants,
        )
        (
            mgr.sequence()
            .record(kp.OpSyncDevice([points_in, stencil_out]))
            .record(kp.OpAlgoDispatch(algorithm))
            .eval()
        )

        seq = mgr.sequence()
        seq.eval_async(kp.OpSyncLocal([stencil_out, debug]))
        seq.eval_await()

        generated_stencil = (
            np.array(stencil_out.data()).reshape(height, width, 4).astype(np.uint8)
        )
        debug = persistent_tensors.get_tensor_data(ShaderBuffers.POS_DEBUG)

        # print(
        #    f"Generated stencil shape: {generated_stencil.shape}  dtype:"
        #    f" {generated_stencil.dtype}"
        # )

        # print(f"Stencil: {generated_stencil[..., 0]}")  # Print only the first channel
        # np.savetxt("stencil_output.txt", generated_stencil[..., 0], fmt="%i")
        # input("Press Enter to continue...")

        return generated_stencil


class FieldComputationShader:
    def __init__(self):
        self.shader_code = Shader("Field Computation Shader", "compute_fields")
        self.shader_code.compile()
        self.kernel_radius: float = 32.0
        self.kernel_width: int = int(self.kernel_radius * 2 + 1)
        # self.TSNEKernel: np.array = np.zeros(
        #    (self.kernel_width, self.kernel_width, 4), dtype=np.float32
        # )
        self.func_support: float = 6.5
        self.field_texture: np.array = None

    def compute(
        self,
        mgr: kp.Manager,
        num_points: int,
        stencil: np.array,
        width: int,
        height: int,
        persistent_tensors: PersistentTensors,
    ):
        field_out = mgr.image(
            np.zeros((height, width, 4), dtype=np.float32), width, height, 4
        )
        points_in = persistent_tensors.get_tensor(ShaderBuffers.POSITION)
        bounds_in = persistent_tensors.get_tensor(ShaderBuffers.BOUNDS)
        num_points_tensor = persistent_tensors.get_tensor(ShaderBuffers.NUM_POINTS)
        stencil_in = mgr.image(stencil, width, height, 4)
        params = [points_in, bounds_in, field_out, stencil_in, num_points_tensor]
        # print(f"Num points: {num_points}")
        push_constants = [float(width), float(height), self.func_support]
        algorithm = mgr.algorithm(
            tensors=params,  # The stencil_tensor is the only output parameter
            spirv=self.shader_code.spv,
            workgroup=[width, height, 1],  # Global dispatch 1 thread per pixel
            push_consts=push_constants,
        )
        (
            mgr.sequence()
            .record(kp.OpSyncDevice(params))
            .record(kp.OpAlgoDispatch(algorithm))
            .eval()
        )

        seq = mgr.sequence()
        seq.eval_async(kp.OpSyncLocal([field_out]))
        seq.eval_await()

        # print(f"Field shape {field_out.data().shape} ")
        generated_field = (
            np.array(field_out.data()).reshape(height, width, 4).astype(np.float32)
        )
        # np.savetxt("field_output.txt", generated_field[..., 1], fmt="%f")
        # print(f"Field data {field_out.data()}")
        return generated_field


class InterpolationShader:
    def __init__(self):
        self.shader_code = Shader("Interpolation Shader", "interp_fields")
        self.shader_code.compile()

    def compute(
        self,
        mgr: kp.Manager,
        num_points: int,
        fields: np.array,
        width: int,
        height: int,
        persistent_tensors: PersistentTensors,
    ):

        interp_fields_out = persistent_tensors.get_tensor(ShaderBuffers.INTERP_FIELDS)
        sum_out = persistent_tensors.get_tensor(ShaderBuffers.SUM_Q)

        points_in = persistent_tensors.get_tensor(ShaderBuffers.POSITION)
        bounds_in = persistent_tensors.get_tensor(ShaderBuffers.BOUNDS)
        num_points_tensor = persistent_tensors.get_tensor(ShaderBuffers.NUM_POINTS)
        fieldimage_in = mgr.image(fields, width, height, 4)

        params = [
            points_in,
            interp_fields_out,
            sum_out,
            bounds_in,
            num_points_tensor,
            fieldimage_in,
        ]
        # print(f"Width: {width}  height : {height}")
        push_constants = [
            float(width),
            float(height),
        ]
        algorithm = mgr.algorithm(
            tensors=params,  # The stencil_tensor is the only output parameter
            spirv=self.shader_code.spv,
            workgroup=[1, 1, 1],  # Global dispatch 1 thread per pixel
            push_consts=push_constants,
        )
        (
            mgr.sequence()
            .record(kp.OpSyncDevice(params))
            .record(kp.OpAlgoDispatch(algorithm))
            .eval()
        )

        seq = mgr.sequence()
        seq.eval_async(kp.OpSyncLocal([interp_fields_out, sum_out]))
        seq.eval_await()

        # self.interp_fields = (
        #    np.array(interp_fields_out.data()).reshape(num_points, 4).astype(np.float32)
        # )

        self.sumQ = sum_out.data()
        # print(f"Interpolation Sum: {self.sumQ}")


class ForcesShader:
    def __init__(self):
        self.shader_code = Shader("Forces Shader", "compute_forces")
        self.shader_code.compile()

    def compute(
        self,
        mgr: kp.Manager,
        num_points: int,
        exaggeration: float,
        persistent_tensors: PersistentTensors,
    ):
        points_in = persistent_tensors.get_tensor(ShaderBuffers.POSITION)
        neighbours_in = persistent_tensors.get_tensor(ShaderBuffers.NEIGHBOUR)
        prob_in = persistent_tensors.get_tensor(ShaderBuffers.PROBABILITIES)
        index_in = persistent_tensors.get_tensor(ShaderBuffers.INDEX)
        interp_fields_in = persistent_tensors.get_tensor(ShaderBuffers.INTERP_FIELDS)
        gradients_out = persistent_tensors.get_tensor(ShaderBuffers.GRADIENTS)
        num_points_tensor = persistent_tensors.get_tensor(ShaderBuffers.NUM_POINTS)
        sum_kl = persistent_tensors.get_tensor(ShaderBuffers.KL)

        params = [
            points_in,
            neighbours_in,
            index_in,
            prob_in,
            interp_fields_in,
            gradients_out,
            num_points_tensor,
            sum_kl,
        ]
        sumq_data = persistent_tensors.get_tensor_data(ShaderBuffers.SUM_Q)

        push_constants = [
            exaggeration,
            sumq_data[0],
        ]

        grid_size = int(math.sqrt(num_points) + 1)
        # print(f"Grid size: {grid_size} for num_points: {num_points}")
        algorithm = mgr.algorithm(
            tensors=params,  # The stencil_tensor is the only output parameter
            spirv=self.shader_code.spv,
            workgroup=[grid_size, grid_size, 1],  # Global dispatch 1 thread per pixel
            push_consts=push_constants,
        )
        (
            mgr.sequence()
            .record(kp.OpSyncDevice(params))
            .record(kp.OpAlgoDispatch(algorithm))
            .eval()
        )

        seq = mgr.sequence()
        seq.eval_async(kp.OpSyncLocal([gradients_out, sum_kl]))
        seq.eval_await()

        grad = gradients_out.data()
        sum_kl = sum_kl.data()
        return sum_kl[0]
        # print(f"Sum KL: {sum_kl}")
        # return (interp_fields, )
        # print(f"Gradient: {grad}")


class UpdateShader:
    def __init__(self):
        self.shader_code = Shader("Point Update Shader", "update")
        self.shader_code.compile()

    def compute(
        self,
        mgr: kp.Manager,
        num_points: int,
        eta: float,
        minimum_gain: float,
        iteration: int,
        momentum: float,
        momentum_switch: int,
        momentum_final: float,
        gain_multiplier: float,
        persistent_tensors: PersistentTensors,
    ):
        points_in = persistent_tensors.get_tensor(ShaderBuffers.POSITION)
        gradients_in = persistent_tensors.get_tensor(ShaderBuffers.GRADIENTS)
        prev_gradients_in = persistent_tensors.get_tensor(ShaderBuffers.PREV_GRADIENTS)
        gain_in = persistent_tensors.get_tensor(ShaderBuffers.GAIN)
        num_points_tensor = persistent_tensors.get_tensor(ShaderBuffers.NUM_POINTS)

        params = [
            points_in,
            gradients_in,
            prev_gradients_in,
            gain_in,
            num_points_tensor,
        ]

        push_constants = [
            float(eta),
            float(minimum_gain),
            float(iteration),
            float(momentum_switch),
            float(momentum),
            float(momentum_final),
            float(gain_multiplier),
        ]
        num_workgroups = int(num_points * 2 / 64) + 1
        grid_size = int(math.sqrt(num_workgroups) + 1)
        algorithm = mgr.algorithm(
            tensors=params,  # The stencil_tensor is the only output parameter
            spirv=self.shader_code.spv,
            workgroup=[grid_size, grid_size, 1],  # Global dispatch 1 thread per pixel
            push_consts=push_constants,
        )
        (
            mgr.sequence()
            .record(kp.OpSyncDevice(params))
            .record(kp.OpAlgoDispatch(algorithm))
            .eval()
        )

        seq = mgr.sequence()
        seq.eval_async(kp.OpSyncLocal([points_in, prev_gradients_in, gain_in]))
        seq.eval_await()

        updated_points = points_in.data()
        # print(f"Updated points: {updated_points}")


class CenterScaleShader:
    def __init__(self):
        self.shader_code = Shader("Update Embedding ", "center_scale")
        self.shader_code.compile()

    def compute(
        self,
        mgr: kp.Manager,
        num_points: int,
        exaggeration: float,
        persistent_tensors: PersistentTensors,
    ):
        points_in = persistent_tensors.get_tensor(ShaderBuffers.POSITION)
        bounds_in = persistent_tensors.get_tensor(ShaderBuffers.BOUNDS)
        # debug = persistent_tensors.get_tensor(ShaderBuffers.POS_DEBUG)
        num_points_tensor = persistent_tensors.get_tensor(ShaderBuffers.NUM_POINTS)

        params = [points_in, bounds_in, num_points_tensor]
        if exaggeration > 1.2:
            scale = 1.0
            diameter = 0.1
        else:
            scale = 0.0
            diameter = 0.0

        # print(f"Center/scale scale: {scale} diameter: {diameter}")
        push_constants = [
            float(num_points),
            float(scale),
            float(diameter),
        ]
        num_workgroups = int(num_points / 128) + 1
        grid_size = int(math.sqrt(num_workgroups)) + 1

        algorithm = mgr.algorithm(
            tensors=params,  # The stencil_tensor is the only output parameter
            spirv=self.shader_code.spv,
            workgroup=[grid_size, grid_size, 1],  # Global dispatch 1 thread per pixel
            push_consts=push_constants,
        )
        (
            mgr.sequence()
            .record(kp.OpSyncDevice(params))
            .record(kp.OpAlgoDispatch(algorithm))
            .eval()
        )

        seq = mgr.sequence()
        seq.eval_async(kp.OpSyncLocal([points_in]))
        seq.eval_await()
        # dbg_np = persistent_tensors.get_tensor_data(ShaderBuffers.POS_DEBUG)
        # updated_points = np.reshape(points_in.data(), (num_points, 2))
        # print(
        #    f"Update max: {np.max(updated_points, axis=0)} min:"
        #    f" {np.min(updated_points, axis=0)}"
        # )
        # print(f"Updated points after center and scale: {updated_points}")
