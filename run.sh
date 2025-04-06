#!/bin/bash

# DDP
NUM_NODES=1
NUM_GPUS_PER_NODE=4
NODE_RANK=0
MASTER_ADDR=localhost
MASTER_PORT=29500

# ModelArgs
MODEL_DIM=512
MODEL_N_LAYERS=16
MODEL_N_HEADS=8
MODEL_N_KV_HEADS=8
MODEL_VOCAB_SIZE=8192
MODEL_MULTIPLE_OF=256
MODEL_FFN_DIM_MULTIPLIER=""
MODEL_NORM_EPS=1e-5
MODEL_MAX_BATCH_SIZE=32
MODEL_MAX_SEQ_LEN=512
MODEL_FLASH_ATTN=false
MODEL_DROPOUT=0.1

# TrainConfig
OUT_DIR="out"
DATASET_DIR="./data/pretrain"
resume_from_checkpoint=false
EVAL_INTERVAL=10
LOG_INTERVAL=1
EVAL_ITERS=1
ALWAYS_SAVE_CHECKPOINT=true
WANDB_LOG=false
WANDB_PROJECT="minilm-pretrain"
WANDB_RUN_NAME="run_$(date +%Y%m%d_%H%M%S)"
GRADIENT_ACCUMULATION_STEPS=1
TRAIN_BATCH_SIZE=1
EVAL_BATCH_SIZE=1
CONTEXT_LENGTH=4
NUM_WORKERS=4
LEARNING_RATE=6e-4
MAX_ITERS=600000
WEIGHT_DECAY=0.1
BETA1=0.9
BETA2=0.95
GRAD_CLIP=1.0
DECAY_LR=true
WARMUP_ITERS=1
LR_DECAY_ITERS=600000
DEVICE="cpu"
DTYPE="bfloat16"
COMPILE=false
BACKEND="nccl"
EVAL_ONLY=false


OS="$(uname -s)"
USE_DDP=true

if [[ "$OS" == "Darwin" || "$OS" == MINGW* || "$OS" == CYGWIN* ]]; then
  echo "Running in single-GPU mode..."
  USE_DDP=false
fi
CMD=(
  main.py
  --out_dir "$OUT_DIR"
  --dataset_dir "$DATASET_DIR"
  --resume_from_checkpoint "$resume_from_checkpoint"
  --eval_interval "$EVAL_INTERVAL"
  --log_interval "$LOG_INTERVAL"
  --eval_iters "$EVAL_ITERS"
  --always_save_checkpoint "$ALWAYS_SAVE_CHECKPOINT"
  --wandb_log "$WANDB_LOG"
  --wandb_project "$WANDB_PROJECT"
  --wandb_run_name "$WANDB_RUN_NAME"
  --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS"
  --train_batch_size "$TRAIN_BATCH_SIZE"
  --eval_batch_size "$EVAL_BATCH_SIZE"
  --context_length "$CONTEXT_LENGTH"
  --num_workers "$NUM_WORKERS"
  --learning_rate "$LEARNING_RATE"
  --max_iters "$MAX_ITERS"
  --weight_decay "$WEIGHT_DECAY"
  --beta1 "$BETA1"
  --beta2 "$BETA2"
  --grad_clip "$GRAD_CLIP"
  --decay_lr "$DECAY_LR"
  --warmup_iters "$WARMUP_ITERS"
  --lr_decay_iters "$LR_DECAY_ITERS"
  --device "$DEVICE"
  --dtype "$DTYPE"
  --compile "$COMPILE"
  --backend "$BACKEND"
  --eval_only "$EVAL_ONLY"
  --model_dim "$MODEL_DIM"
  --model_n_layers "$MODEL_N_LAYERS"
  --model_n_heads "$MODEL_N_HEADS"
  --model_n_kv_heads "$MODEL_N_KV_HEADS"
  --model_vocab_size "$MODEL_VOCAB_SIZE"
  --model_multiple_of "$MODEL_MULTIPLE_OF"
  $(if [ -n "$MODEL_FFN_DIM_MULTIPLIER" ]; then echo "--model_ffn_dim_multiplier $MODEL_FFN_DIM_MULTIPLIER"; fi)
  --model_norm_eps "$MODEL_NORM_EPS"
  --model_max_batch_size "$MODEL_MAX_BATCH_SIZE"
  --model_max_seq_len "$MODEL_MAX_SEQ_LEN"
  $(if [ "$MODEL_FLASH_ATTN" = true ]; then echo "--model_flash_attn"; fi)
  --model_dropout "$MODEL_DROPOUT"
)

if [ "$USE_DDP" = true ]; then
  torchrun \
    --nproc_per_node=$NUM_GPUS_PER_NODE \
    --nnodes=$NUM_NODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    "${CMD[@]}"
else
  python "${CMD[@]}"
fi
