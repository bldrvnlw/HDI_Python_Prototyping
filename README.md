## Prototype a Vulkan Kompute Project tSNE implementation in Pyton

Due to lack of compute shader support on MacOS OpenGL this is an experiment to 
port the HDLLib compute shaders to Vulkan using the Vulkan Kompute Project library.

The Vulkan Compute Project library introduces a tensor abstraction to greatly
simplify writing compute shaders. These "tensors" are simply CPU arrays that can be easily
mapped to GPU buffers using simple operations. This is similar to pytorch.

Regarding the compute shaders, the glslangValidator provides an easy route to
compile shaders, with minimal changes, to the SPIRV form consumed by Vulkan. 
These shaders could be applied largely unchanged apart from two points:

1. Uniforms are replaced bu push constants. Unfortunately these push constants cannot
be integers so some tweaks are needed.

2. It does not seem to be possible to get a GLSL sampler working in the Vulkan
Kompute Project. Instead a simple implementation of bilinear sampling was created.

### Python and C++

This project is created in python to allow rapid development however the Vulkan Kompute
project provides an almost identical API for C++. If anything the C++ API is more extensive. This means that the 

In order to have access the the image type in python the master branch of the Vulkan Kompute Project was build as the current (version 0.9 at June 2025) does not have the 
image support while it is already in the master.

### NN and probability calculations

Where possible existing python libraries (annoy, hnsw) have been used for the knn calculations. Probabilities are calculated in a combination of python, numba and pytorch for performance.

### Test data

The following sets are tested

Name | # data points | # dimensions
--- | --- | ---
MNIST | 70000 | 784
Zheng mouse | 1306127 | 50


HYPOMAP 433,369 https://cellxgene.cziscience.com/collections/d0941303-7ce3-4422-9249-cf31eb98c480 see https://scanpy.readthedocs.io/en/stable/generated/scanpy.read_h5ad.html and https://scverse-tutorials.readthedocs.io/en/latest/notebooks/anndata_getting_started.html 

For other large data sets see https://www.openml.org/ use search term `#Instances > 1e5 AND #Features > 50`


https://github.com/niderhoff/big-data-datasets

### Sets compatible with [scanpy](https://scanpy.readthedocs.io/en/stable/index.html)  :

[Tabula Muris Senis](https://figshare.com/articles/dataset/Processed_files_to_use_with_scanpy_/8273102?file=23938934)

