# Experiment: vLLM Baseline On Pink

## Goal

Measure a first vLLM serving baseline on `pink` and understand how latency and throughput change with concurrency, prompt length, output length, and GPU count.

## Environment

- Machine: `pink`
- GPU: 2 x NVIDIA GeForce RTX 4090, 24 GB each
- Driver: NVIDIA 580.142, CUDA 13.0 reported by `nvidia-smi`
- Docker image: not selected yet; Docker 29.1.5 is installed
- Python: 3.12.2 system Python
- Framework: vLLM
- Model: `Qwen/Qwen2.5-7B-Instruct`

## Setup

```bash
ssh pink
nvidia-smi
Get-Content -Raw projects/llm-inference-benchmark-lab/scripts/inspect_pink.sh | ssh pink bash -s
```

## Server Command

```bash
SNAP=$HOME/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775
docker run -d \
  --name vllm-smoke \
  --gpus all \
  --ipc=host \
  -p 8000:8000 \
  -e HF_HUB_OFFLINE=1 \
  -e TRANSFORMERS_OFFLINE=1 \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-openai:latest \
  --model /root/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.80 \
  --max-model-len 2048 \
  --served-model-name qwen2.5-0.5b-smoke
```

## Benchmark Command

```bash
python ~/mlsys-inference-smoke/scripts/benchmark_client.py \
  --model qwen2.5-0.5b-smoke \
  --prompt-length short \
  --max-tokens 64 \
  --requests 1 \
  --out ~/mlsys-inference-smoke/results/vllm_smoke.csv
```

## Benchmark Matrix

| GPU Count | Concurrency | Prompt Length | Output Length |
| ---: | ---: | ---: | ---: |
| 1 | 1, 2, 4, 8, 16 | short, medium, long | short, medium |
| 2 | 1, 2, 4, 8, 16 | short, medium, long | short, medium |

## Results

| GPU Count | Concurrency | TTFT | TPOT | Output tok/s | Peak GPU Memory | Failure Rate |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 0.9094 s | 0.0021 s | 61.27 | 19,778 MiB | 0.0 |

## Interpretation

- `pink` has two idle RTX 4090 GPUs, a recent NVIDIA driver, Docker, Python 3.12.2, and enough disk space.
- Docker GPU runtime works: `docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi` sees both RTX 4090 GPUs.
- The vLLM Docker image was already present on `pink`.
- Starting from the Hugging Face model id failed inside the container because vLLM could not retrieve the Hugging Face file list from container networking.
- Starting from the cached local snapshot path with `HF_HUB_OFFLINE=1` worked.
- The first smoke request succeeded and wrote `~/mlsys-inference-smoke/results/vllm_smoke.csv`.
- The first request had high TTFT because the logs showed a Triton kernel JIT compilation during inference: `_compute_slot_mapping_kernel`.
- The first attempt to pipe the inspection script from Windows exposed a CRLF issue, so shell scripts are now forced to LF with `.gitattributes`.

## Next Step

- Install or activate vLLM.
- Start the vLLM server with `TENSOR_PARALLEL_SIZE=1`.
- Run the one-request smoke benchmark and paste the summary here.

## Resume / Interview Sentence

```text
I built a small LLM inference benchmark lab to measure vLLM serving behavior on dual RTX 4090 GPUs, tracking TTFT, TPOT, throughput, memory, and failure rate across concurrency and sequence-length sweeps.
```
