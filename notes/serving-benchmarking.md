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

## Benchmark Pitfalls From vLLM Baseline

- Warmup should be outside the measured window. Otherwise cold-start work such as Triton JIT or CUDA graph setup can pollute steady-state numbers.
- Concurrent throughput should use wall-clock time, not summed request latency. Summed latency double-counts overlapping work.
- Token throughput should use server-reported completion tokens when available. Text whitespace splitting is not tokenizer-accurate.
- Prefill experiments need server-reported prompt tokens, not just a human label such as short or long.
- Repeated prompts can make prefix cache hide prefill work.
- Request-id-only varied prompts can still leak prefix-cache state across prompt buckets if different buckets share the same long prefix. A per-run prompt salt helps break cross-experiment reuse.
- Context admission depends on `prompt_tokens + requested max_tokens`, not only actual generated tokens.

## Interview Sentence

```text
When I benchmark an LLM server, I separate TTFT, TPOT, end-to-end latency, throughput, memory, and failures. For prefill experiments I vary and salt prompts to avoid prefix-cache contamination, and for decode experiments I sweep output length because latency should scale roughly with generated tokens.
```

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
