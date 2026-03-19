# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains scripts for building **PyTorch 2.6.0 from source** on IBM POWER9 (ppc64le) architecture with CUDA support on a SLURM-managed DCS GPU cluster. Official PyTorch wheels do not exist for ppc64le, so this pipeline builds them locally.

**Target environment**: SLURM cluster, `dcs-2024` GPU partition, conda environment `torch260`, Python 3.11, GCC 12.4.0

## Build Workflow

### 1. Prepare environment
```bash
# Clone PyTorch v2.6.0
git clone --recursive --branch v2.6.0 https://github.com/pytorch/pytorch
cd pytorch

# Create conda env and install dependencies
bash install_libs.sh
```

### 2. Build PyTorch wheel (submit SLURM job, ~90-120 minutes)
```bash
sbatch build_torch.sh
```
Output wheel lands in `dist/torch-2.6.0+cu124-cp311-cp311-linux_ppc64le.whl`

### 3. Install and verify
```bash
pip install dist/torch-2.6.0+cu124-cp311-cp311-linux_ppc64le.whl
sbatch verify.sh        # verify on a GPU node
# or directly:
python verify_pytorch.py
```

## Key Scripts

| Script | Purpose |
|---|---|
| `install_libs.sh` | Install conda deps (gcc 12.4, CUDA 12.4.1, cudnn 9.0, 17 Python packages) |
| `build_torch.sh` | SLURM batch job that sets env vars and runs `python setup.py bdist_wheel` |
| `verify_pytorch.py` | Test suite: version check, CUDA, CPU/GPU tensors, neural net, autograd |
| `verify.sh` | SLURM wrapper to run `verify_pytorch.py` on a GPU node |

## Critical Build Details

- **`PYTORCH_BUILD_VERSION`** in `build_torch.sh` must match the format `2.6.0+cu124` — this string is embedded in the wheel filename and `torch.__version__`
- CUDNN 9.0 must be manually downloaded from NVIDIA and installed via `rsync` into `$CONDA_PREFIX` (conda only provides CUDNN 8.9, which causes compatibility issues)
- Dependencies in `install_libs.sh` are installed one at a time (sequential `conda install` calls) to avoid conda solver timeouts

## Pre-built Wheels

`dist/` contains ready-to-install wheels for Python 3.11 (ppc64le):
- `torch-2.6.0+cu121-cp311-cp311-linux_ppc64le.whl` — CUDA 12.1
- `torch-2.6.0+cu124-cp311-cp311-linux_ppc64le.whl` — CUDA 12.4
