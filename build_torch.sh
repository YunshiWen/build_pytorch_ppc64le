#!/bin/bash
#SBATCH --job-name=torch         # create a short name for your job
#SBATCH -n 1
#SBATCH --partition=dcs-2024     # appropriate partition; if not specified, slurm will automatically do it for you
#SBATCH --gres=gpu:1             # number of allocated gpus per node
#SBATCH --output=./dcs/torch_build.log  # Standard output and error log
#SBATCH --time=06:00:00          # total run time limit (HH:MM:SS)


# module use /opt/nvidia/hpc_sdk/modulefiles
# module spider
# module load nvhpc
# module load gcc

eval "$(/path_to_conda/miniconda3/bin/conda shell.bash hook)" 
conda activate torch260

nvcc --version
gcc --version

# Paths
export CC=$(which gcc)
export CXX=$(which g++)
export CUDA_HOME="$CONDA_PREFIX"
export CUDA_TOOLKIT_ROOT_DIR="$CONDA_PREFIX"
export CUDACXX="$CONDA_PREFIX/bin/nvcc"

export CUDNN_ROOT="$CONDA_PREFIX"
export CUDNN_INCLUDE_DIR="$CONDA_PREFIX/include"
export CUDNN_LIBRARY="$CONDA_PREFIX/lib/libcudnn.so"

export PATH="$CONDA_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export CMAKE_PREFIX_PATH="${CONDA_PREFIX}:${CMAKE_PREFIX_PATH:-}"
export LDFLAGS="-Wl,-rpath,$CONDA_PREFIX/lib"


# For debugging - print the libraries in your conda environment
echo "Checking for CUDA libraries in $CONDA_PREFIX/lib:"
ls -l $CONDA_PREFIX/lib/libcudart*
ls -l $CONDA_PREFIX/lib/libcupti*
ls -l $CONDA_PREFIX/lib/libopenblas*
ls -l $CONDA_PREFIX/lib/libgomp*


export CXXFLAGS=""  
export CFLAGS=""


python setup.py clean

export PYTORCH_BUILD_VERSION="2.6.0+cu124"
export PYTORCH_BUILD_NUMBER=0
python setup.py bdist_wheel
echo "Wheel file created in dist/ directory:"
ls -lh dist/*.whl
