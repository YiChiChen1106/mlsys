# Serving Benchmarking

Serving benchmarks should explain the latency-throughput tradeoff of an inference system, not just produce one tokens-per-second number.

## Minimum Metrics

- TTFT: time to first token.
- TPOT: time per output token after the first token.
- End-to-end latency: full request duration.
- Output throughput: generated tokens per second.
- Request throughput: completed requests per second.
- GPU memory peak.
- Failure rate.

## Variables To Sweep

- Framework: vLLM, SGLang, llama.cpp.
- Model: start with a 7B/8B class model.
- GPU count: 1 x RTX 4090, then 2 x RTX 4090.
- Concurrency: 1, 2, 4, 8, 16.
- Prompt length: short, medium, long.
- Output length: short, medium.

## Record Format

Each experiment note should include:

- Goal.
- Environment.
- Model and framework.
- Commands.
- Benchmark matrix.
- Results table.
- Interpretation.
- Next step.

Use `templates/inference-experiment.md`.
