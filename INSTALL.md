# ManiFlow Training: Installation

## Quickstart

This guide sets up **ManiFlow** in a single conda environment.

## Prerequisites

- Linux with an NVIDIA GPU and a driver that supports CUDA 12.8
- Conda (Miniforge, Miniconda or Anaconda)
- A C compiler and make (Triton compiles small helpers at runtime):
  ```bash
  sudo apt install build-essential
  ```

---

## 1. Create Environment & Install Core Dependencies

```bash
conda create -y -n maniflow-tb python=3.10
conda activate maniflow-tb
# CUDA toolkit (nvcc): DeepSpeed queries it on import
conda install -c nvidia cuda-toolkit=12.8 -y
cd HierarchicalVLA
pip install -r requirements.txt
```

## 2. Install FFmpeg & TorchCodec

The dataloader decodes videos with TorchCodec, which needs the FFmpeg shared libraries:

```bash
conda install -y -c conda-forge "ffmpeg=7"
pip install --no-cache-dir "torchcodec==0.5" --index-url https://download.pytorch.org/whl/cu128

# conda-forge FFmpeg needs a newer libstdc++ than some systems ship (e.g. Ubuntu 22.04);
# let the environment's own libraries win, otherwise TorchCodec can fail to load.
conda env config vars set LD_LIBRARY_PATH="$CONDA_PREFIX/lib"
conda deactivate && conda activate maniflow-tb
```

## 3. Install Packages in Editable Mode

```bash
# Install this repository
pip install -e .
```

Optional, only for `scripts/train_maniflow.py --optimizer muon`:

```bash
pip install "muon-optimizer @ git+https://github.com/KellerJordan/Muon@f98f1cacc0263b04290753e32be8d498c1efc806"
```

Training downloads pretrained weights from the Hugging Face Hub on first use (the timm RGB backbone and `t5-small`
for language conditioning); set `HF_HOME` to choose the cache location.

The unit tests (`make test`) additionally need `pip install pytest`.
