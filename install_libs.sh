#!/bin/bash

eval "$(/path_to_conda/miniconda3/bin/conda shell.bash hook)" 
# conda create -n torch260 python=3.11 -y
conda activate torch260

conda install -y -c conda-forge gcc=12.4.0 gxx=12.4.0
conda install -y -c conda-forge numpy=1.26.4
conda install -y nvidia/label/cuda-12.4.1::cuda
conda install -y conda-forge::libopenblas
conda install -y cmake ninja
# conda install conda-forge::cudnn



CUDNN_DIR=/path_to_cudnn/cudnn-linux-ppc64le-9.0.0.312_cuda12-archive
echo "CUDNN_DIR: $CUDNN_DIR"
echo "CONDA_PREFIX: $CONDA_PREFIX"
rsync -av $CUDNN_DIR/include/ $CONDA_PREFIX/include/
rsync -av $CUDNN_DIR/lib/ $CONDA_PREFIX/lib/


conda install -c conda-forge -y astunparse
conda install -c conda-forge -y expecttest
conda install -c conda-forge -y hypothesis
conda install -c conda-forge -y psutil
conda install -c conda-forge -y pyyaml
conda install -c conda-forge -y requests
conda install -c conda-forge -y setuptools
conda install -c conda-forge -y types-dataclasses
conda install -c conda-forge -y typing-extensions
conda install -c conda-forge -y sympy
conda install -c conda-forge -y filelock
conda install -c conda-forge -y networkx
conda install -c conda-forge -y jinja2
conda install -c conda-forge -y fsspec
conda install -c conda-forge -y lintrunner
conda install -c conda-forge -y packaging
conda install -c conda-forge -y optree
