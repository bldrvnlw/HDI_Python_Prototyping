from .hdi_shader_source import HDIShaderSource
import subprocess
import kp
import numpy as np
import math


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
        seq.record(kp.OpTensorSyncDevice(params))
        seq.record(kp.OpAlgoDispatch(algorithm))
        seq.record(kp.OpTensorSyncLocal([bounds_out]))
        seq.eval()
        return bounds_out.data().reshape(2, 2)
