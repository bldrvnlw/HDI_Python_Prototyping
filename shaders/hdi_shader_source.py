import os


class HDIShaderSource(dict):
    """The supportes shader names are hardcoded for this test"""

    def __init__(self):
        self._suffix = ".glsl.comp"
        self._directory = os.path.join(os.path.dirname(__file__), "sources")
        keys = {
            "bounds": "",
            "stencil": "",
            "compute_fields": "",
            "compute_forces": "",
            "interp_fields": "",
            "update": "",
        }
        self._loaded = set()
        super().__init__({key: None for key in keys})

    def __getitem__(self, key):
        """Lazy load the shader source code from files."""
        if key not in self:
            raise KeyError(f"Shader source '{key}' not defined in HDIShaderSource.")
        if key not in self._loaded:
            filename = os.path.join(self._directory, f"{key}{self._suffix}")
            with open(filename, "r", encoding="utf-8") as f:
                self[key] = f.read()
            self._loaded.add(key)
        return super().__getitem__(key)
