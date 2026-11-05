# ManiFlow Training

Training scripts for the ManiFlow multi-modal policy (RGB, optionally with depth) on LeRobot-format datasets.
The checkpoints are evaluated with ToyBlocks' policy server (see [Evaluation](#3-evaluation)).
Setup is described in [INSTALL.md](INSTALL.md).
A checkpoint trained with these scripts is on Hugging Face: [allenai/maniflow-tb](https://huggingface.co/allenai/maniflow-tb).

---

## Quick Start

### 1. Running locally

```bash
# Maniflow path (from the repository root)
CUDA_VISIBLE_DEVICES=0 WANDB_MODE=offline \
python scripts/train_maniflow.py \
  --num_gpus_per_node 1 \
  --run_root_dir runs \
  --duration 100 \
  --save_interval 100 \
  --logging_frequency 10 \
  --dbs 4 \
  --num_workers 0 \
  --data_root_dir /path/to/lerobot_dataset \
  --max_episodes "500" --depth --depth_rg8 rg8_55_224
```

### 2. Full training runs

With RGB:

```bash
python scripts/train_maniflow.py \
  --rgb_224 --data_root_dir /path/to/lerobot_dataset \
  --max_episodes 5000
```

With Depth with rg8_55:

```bash
python scripts/train_maniflow.py \
  --rgb_224 --data_root_dir /path/to/lerobot_dataset \
  --max_episodes 5000 --depth --depth_rg8 rg8_55_224
```

`scripts/train_maniflow.py` uses all visible GPUs by default (`--num_gpus_per_node`). For several nodes, run it on
every node with the same flags plus `--num_nodes N --node_rank R --master_addr <node 0> --run_id <name>`.
`--resume <run_id>` continues a run from its latest checkpoint. `python scripts/train_maniflow.py --help` lists all options.

### 3. Evaluation

Checkpoints are evaluated with ToyBlocks' policy server, which loads the model with this repository. A run directory
`runs/<run_id>/` holds what the server needs: `checkpoints/steps_<N>_model.safetensors` (the weights), `config.yaml`
and `stats.json`. The trained checkpoint on Hugging Face,
[allenai/maniflow-tb](https://huggingface.co/allenai/maniflow-tb), has the same layout:

```bash
hf download allenai/maniflow-tb --local-dir maniflow-tb
```

## License

This code is released under the [Apache License 2.0](LICENSE).
