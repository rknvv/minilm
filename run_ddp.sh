#!/bin/bash

# DDP, 2xH100
NUM_NODES=1
NUM_GPUS_PER_NODE=2
NODE_RANK=0
MASTER_ADDR=localhost
MASTER_PORT=29500

export NCCL_NVLS_ENABLE=1
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}

export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-$(dirname "$0")/.inductor_cache}

torchrun \
  --nproc_per_node=$NUM_GPUS_PER_NODE \
  --nnodes=$NUM_NODES \
  --node_rank=$NODE_RANK \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  main.py \
  --yaml_path="./configs/cpt_gemma_1b.yaml"
