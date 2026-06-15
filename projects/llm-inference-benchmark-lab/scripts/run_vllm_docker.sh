#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME="${CONTAINER_NAME:-vllm-inference}"
IMAGE="${IMAGE:-vllm/vllm-openai:latest}"
MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen2.5-0.5b-smoke}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
PORT="${PORT:-8000}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.80}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
HF_CACHE_DIR="${HF_CACHE_DIR:-$HOME/.cache/huggingface}"
OFFLINE="${OFFLINE:-1}"

if [[ "${OFFLINE}" == "1" && "${MODEL}" != /* ]]; then
  echo "OFFLINE=1 requires MODEL to be an absolute local snapshot path." >&2
  echo "Example: MODEL=$HOME/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/<revision>" >&2
  exit 2
fi

docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true

docker run -d \
  --name "${CONTAINER_NAME}" \
  --gpus all \
  --ipc=host \
  -p "${PORT}:8000" \
  -e "HF_HUB_OFFLINE=${OFFLINE}" \
  -e "TRANSFORMERS_OFFLINE=${OFFLINE}" \
  -v "${HF_CACHE_DIR}:/root/.cache/huggingface" \
  "${IMAGE}" \
  --model "/root/.cache/huggingface${MODEL#${HF_CACHE_DIR}}" \
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --served-model-name "${SERVED_MODEL_NAME}"
