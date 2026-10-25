# Qwen3.5-9B OVC training

The release contains one configuration, [qwen3_5_9b_ovc.json](../configs/qwen3_5_9b_ovc.json),
for **16-node, 128-GPU training** (8 GPUs per node). See the [README](../README.md) for the commands.

## Environment

Use Python 3.12 and a clean CUDA environment. The dependency baseline is in
[requirements/training.txt](../requirements/training.txt). It includes Qwen3.5-capable
Transformers and vLLM versions; the old Transformers 4.57 / vLLM 0.11 stack is not
appropriate for this model. See the [official model instructions](https://huggingface.co/Qwen/Qwen3.5-9B)
and [vLLM recipe](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html).

Install FlashAttention after PyTorch, using `--no-build-isolation` and CUDA-compatible
wheels or a local build. For the hybrid linear-attention fast path, also install
`causal-conv1d` and `flash-linear-attention` against the same PyTorch/CUDA stack.
The dependency versions are a starting baseline, not a full GPU-training validation record.

The preset uses BF16, FSDP2 parameter/optimizer offload, a 20,480-token maximum
sequence length and one response per forward micro-batch. Provision 16 allocated
nodes with 8 idle GPUs each and sufficient host memory. Configure the Ray object
store on each node when starting the cluster; the example below uses 32 GiB per node.
Memory requirements depend on image resolution and response lengths.

## Data

The preprocessor accepts raw parquet with these columns:

| Column | Content |
| --- | --- |
| `problem` | User question, including the image placeholders expected by the processor |
| `answer` | Reference answer for the training reward |
| `images` | Image list in the format supported by the verl multimodal dataset |

It creates `prompt`, `images`, `reward_model` and `extra_info` fields for verl.
`--val-size N` holds out the first N rows; those rows are excluded from training.
`--val-size 0` converts all rows to training format and requires a separate
validation parquet at launch. Neither mode overwrites an existing output directory.

Separate parquet paths do not prove content independence: remove duplicated
questions/source examples across training and validation when preparing the input.
Training-time validation computes the same reward on a held-out split.

## Cluster setup

Install the same training environment on all 16 nodes. Model, train/validation
parquet and the output directory must be accessible at the **same absolute paths**
on every node, using shared storage. All nodes need a compatible CUDA/NCCL network.
Use a dedicated job allocation: Ray resource checks do not detect jobs started
outside Ray. Choose the allocated physical GPUs through `CUDA_VISIBLE_DEVICES`
when starting each Ray node.

On the head node, replace the example address with its private cluster address:

```bash
export HEAD_IP=10.0.0.1
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ray start --head \
  --node-ip-address="$HEAD_IP" --port=6379 --num-gpus=8 \
  --object-store-memory=34359738368 --dashboard-host=127.0.0.1
```

On each of the other **15 nodes**, use the same head address:

```bash
export HEAD_IP=10.0.0.1
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ray start \
  --address="$HEAD_IP:6379" --num-gpus=8 --object-store-memory=34359738368
```

Run the training driver **once on the head node**:

```bash
python -m ovc_release.train \
  --model /models/Qwen3.5-9B \
  --train-file /datasets/math77k/ovc/train.parquet \
  --val-file /datasets/math77k/ovc/val.parquet \
  --output /outputs/qwen35-ovc \
  --ray-address "$HEAD_IP:6379" --launch
```

The launcher requires exactly 16 live GPU nodes with 8 GPUs each and 128 free
Ray GPU resources. An additional CPU-only head is allowed. It connects to the
existing cluster; it does not create a local cluster or stop existing jobs.

## Configuration and launch

Use `python -m ovc_release.train --help` for the single training entry point.
It prints the resolved configuration by default. `--launch` requires a local
Qwen3.5-9B checkpoint, separate train/validation files, a fresh output directory
and `--ray-address` for the dedicated 128-GPU cluster. It never stops existing GPU jobs.

The preset preserves the original batch, rollout, credit and optimizer settings.
The topology is 16 nodes × 8 GPUs. The global training batch remains 256 prompts,
with 8 rollouts per prompt and a 64-prompt actor mini-batch: 512 responses across
128 data-parallel ranks, or 4 responses per rank with micro-batch size 1.
The FSDP shard group remains 8 GPUs, with 16 replica groups; it does not limit
the job to 8 GPUs. These settings preserve the optimizer batch and learning rate.
The budget is
at most 602 updates over two epochs; the available data can end training sooner.

Training snapshots `verl/`, `recipe/math77k/` and the shared credit modules into
`OUTPUT/runtime/`, checks source hashes and records the effective configuration
in `OUTPUT/config.json`. The Qwen3.5 engine entry is `verl.trainer.main_ppo`.
Hidden layer 16 is passed to Ray workers through `VERL_DELTA_H_LAYER_IDX` so that
OVC receives the intermediate states used for visual credit.

For source-only inspection:

```bash
python -m ovc_release.runtime --output /outputs/qwen35-runtime
```

For local method changes, pass `--allow-source-edits` to the training or runtime
entry point. The run still records the actual copied hashes and refuses overwrite.

## Model export

Merge a saved actor checkpoint using that run's source snapshot:

```bash
PYTHONPATH=/outputs/qwen35-ovc/runtime python -m verl.model_merger merge --backend fsdp \
  --local_dir /outputs/qwen35-ovc/checkpoints/global_step_STEP/actor \
  --target_dir /models/qwen35-ovc-final
```

## Evaluation

[lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval)

## Development checks

```bash
python -m pip install pytest
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q --import-mode=importlib tests
PYTHONDONTWRITEBYTECODE=1 python scripts/check_release.py
```

These CPU checks cover the training entry, source snapshot, disjoint data
conversion, reward and token-level credit gradients. Full model loading,
rollout, distributed backpropagation and checkpointing require a GPU training run.
