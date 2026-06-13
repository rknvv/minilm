#!/bin/bash

# DDP, 2xH100
NUM_NODES=1
NUM_GPUS_PER_NODE=2
NODE_RANK=0
MASTER_ADDR=localhost
MASTER_PORT=29500

# NCCL: measured-defaults for a single NVLink node; re-measure on the box
# (see out_profile/REPORT.md, H100 bring-up checklist).
export NCCL_NVLS_ENABLE=1                    # NVLink SHARP all-reduce (H100/NVSwitch)
export TORCH_NCCL_AVOID_RECORD_STREAMS=1     # less sync/memory churn with gradient_as_bucket_view
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}        # visibility without log spam

# Persist the Inductor cache: max-autotune cold compile is ~7 min; /tmp dies
# with the container, the workspace survives stop/start (and recycle too if
# it is volume-backed).
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-$(dirname "$0")/.inductor_cache}

# Contingency for allocator fragmentation on multi-day runs (slight perf cost,
# enable only if a mid-run OOM appears despite stable peak memory):
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

torchrun \
  --nproc_per_node=$NUM_GPUS_PER_NODE \
  --nnodes=$NUM_NODES \
  --node_rank=$NODE_RANK \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  main.py \
  --yaml_path="./configs/cpt_gemma_1b.yaml"
