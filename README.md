# Building PyTorch from Source for PPC64LE

> **Disclaimer:** This is an unofficial example of building PyTorch on a ppc64le system. It was tested on the [RPI CCI AiMOS cluster](https://cci.rpi.edu/aimos) for distributed data parallel (DDP) training using up to 96 V100 32GB GPUs. The author is not responsible for any potential or unknown issues arising from the use of this guide.

Official PyTorch distributions do not include ppc64le (IBM POWER) wheels. This repository provides scripts and documentation for building PyTorch 2.6.0 from source on ppc64le with CUDA support on a SLURM-managed GPU cluster.

## Pre-built Wheels

If you just need a wheel, pre-built binaries for Python 3.11 are available in the [Releases](https://github.com/YunshiWen/build_pytorch_ppc64le/releases):

| Release | CUDA | Python | Download |
|---|---|---|---|
| v2.6.0+cu121 | 12.1 | 3.11 | [torch-2.6.0+cu121-cp311-cp311-linux_ppc64le.whl](https://github.com/YunshiWen/build_pytorch_ppc64le/releases/download/v2.6.0%2Bcu121/torch-2.6.0+cu121-cp311-cp311-linux_ppc64le.whl) |
| v2.6.0+cu124 | 12.4 | 3.11 | [torch-2.6.0+cu124-cp311-cp311-linux_ppc64le.whl](https://github.com/YunshiWen/build_pytorch_ppc64le/releases/download/v2.6.0%2Bcu124/torch-2.6.0+cu124-cp311-cp311-linux_ppc64le.whl) |

To install a pre-built wheel, follow Steps 3–5 below to set up the runtime dependencies (you can omit `gcc`, `gxx`, `cmake`, and `ninja`), then run:

```bash
pip install /path/to/torch-2.6.0+cu124-cp311-cp311-linux_ppc64le.whl
```

---

## Building from Source

### Step 1 — Clone PyTorch

```bash
git clone https://github.com/pytorch/pytorch
cd pytorch
git checkout v2.6.0
git submodule sync && git submodule update --init --recursive
```

### Step 2 — Create a Conda Environment

```bash
conda create -n torch260 python=3.11
conda activate torch260
```

### Step 3 — Install Build Tools and CUDA

> You may substitute a different CUDA version (e.g., `cuda-12.1.0`).

```bash
conda install -y -c conda-forge gcc=12.4.0 gxx=12.4.0
conda install -y -c conda-forge numpy=1.26.4
conda install -y nvidia/label/cuda-12.4.1::cuda
conda install -y conda-forge::libopenblas
conda install -y cmake ninja
```

### Step 4 — Install CUDNN 9.0

> **Note:** Conda only provides CUDNN 8.9 (bundled with CUDA 11.8), which can cause compatibility issues. CUDNN 9.0 must be downloaded manually from NVIDIA and copied into the conda environment.

```bash
wget https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-ppc64le/cudnn-linux-ppc64le-9.0.0.312_cuda12-archive.tar.xz
tar xf cudnn-linux-ppc64le-9.0.0.312_cuda12-archive.tar.xz

CUDNN_DIR=/path/to/cudnn-linux-ppc64le-9.0.0.312_cuda12-archive
rsync -av $CUDNN_DIR/include/ $CONDA_PREFIX/include/
rsync -av $CUDNN_DIR/lib/ $CONDA_PREFIX/lib/
```

### Step 5 — Install PyTorch Python Dependencies

> **Note:** These must be installed one at a time. A single `conda install` with all packages can time out due to solver complexity.

```bash
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
```

Alternatively, run [`install_libs.sh`](./install_libs.sh) which automates Steps 3–5.

### Step 6 — Configure the Build Script

Open [`build_torch.sh`](./build_torch.sh) and verify these two settings:

```bash
conda activate torch260            # Must match the environment name from Step 2
export PYTORCH_BUILD_VERSION="2.6.0+cu124"  # Must be <pytorch_version>+<cuda_version>
```

> **Important:** An incorrect `PYTORCH_BUILD_VERSION` will still produce a working wheel but will cause version-check failures with downstream packages such as HuggingFace Transformers and Accelerate.

### Step 7 — Submit the Build Job

```bash
sbatch build_torch.sh
```

The build takes approximately **90–120 minutes**. Output is written to `dist/torch-2.6.0+cu124-cp311-cp311-linux_ppc64le.whl`.

### Step 8 — Install the Wheel

```bash
# In a new or existing conda environment (with Steps 3–5 dependencies installed):
pip install dist/torch-2.6.0+cu124-cp311-cp311-linux_ppc64le.whl
```

### Step 9 — Verify the Installation

```bash
sbatch verify.sh        # Run on a GPU node via SLURM
# or directly:
python verify_pytorch.py
```

`verify_pytorch.py` checks PyTorch and CUDA versions, runs CPU/GPU tensor operations, tests a small neural network, and validates autograd.

---

## Known Issues on the DCS Cluster

This PyTorch build is expected to work on most of the DCS nodes. However, some issues appear randomly on a small set of nodes.

### Issue Summary

| Issue | Symptom | Solution |
|---|---|---|
| Missing CUDA driver | `torch.cuda.is_available()` returns `False` | Exclude the node or request a different one |
| Stale GPU memory | CUDA errors at job start | Exclude known problematic nodes |
| NCCL / InfiniBand memory error | `NCCL WARN Call to ibv_reg_mr failed with error Cannot allocate memory` | Add `--exclusive` to the SBATCH file |
