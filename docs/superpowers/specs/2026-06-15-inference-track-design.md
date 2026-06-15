# Inference Track Design

## Goal

Create a public learning and project track for AI infra inference systems inside the existing MLsys learning lab. The track should be easy to continue from GitHub, Obsidian, or another agent thread without mixing with the GPU kernel track.

## Repository Model

The repository has two long-running tracks:

- GPU kernel optimization: custom operators, Triton/CUDA kernels, bandwidth analysis, and profiling.
- LLM inference systems: serving frameworks, KV cache, batching, scheduler behavior, and benchmark methodology.

Both tracks live in the same repository because they share the same ML systems performance mindset. They stay separate through naming, directories, and ownership rules.

## File Ownership

This inference thread owns:

- `projects/llm-inference-benchmark-lab/`
- `notes/llm-inference-systems.md`
- `notes/kv-cache.md`
- `notes/batching-and-scheduling.md`
- `notes/serving-benchmarking.md`
- `experiments/inference-*.md`
- `templates/inference-experiment.md`

The GPU kernel thread owns:

- `projects/gpu-mode-*/`
- `notes/vector-sum-reduction.md`
- `notes/triton-basics.md`
- `notes/gpu-kernel-optimization.md`
- `experiments/vectorsum-*.md`

Shared files are edited carefully and only for navigation or status:

- `README.md`
- `START_HERE.md`
- `AGENTS.md`
- `notes/mlsys-learning-hub.md`
- `daily/YYYY-MM-DD.md`

## First Project

The first inference project is `llm-inference-benchmark-lab`.

Scope for the first milestone:

- Run vLLM on `pink`.
- Serve one 7B/8B class model through an OpenAI-compatible endpoint.
- Benchmark single-GPU and two-GPU configurations.
- Sweep concurrency and prompt/output length.
- Record TTFT, TPOT, throughput, failure rate, and GPU memory usage.
- Publish a short experiment report with commands, environment, raw observations, and interpretation.

SGLang and llama.cpp are intentionally later milestones. The first milestone should build one clean baseline before adding more frameworks.

## Learning Loop

Each work session follows this loop:

1. Read `START_HERE.md`, this project README, the latest daily log, and the relevant inference note.
2. Run one small experiment or improve one script.
3. Save raw small outputs under the project directory when useful.
4. Write a public experiment note under `experiments/`.
5. Add distilled concepts to `notes/`.
6. Update `START_HERE.md` and the daily log with the next action.
7. Commit progress with a small, descriptive commit message.

## GitHub and Obsidian

The Obsidian vault is the repository. Public notes are first-class project artifacts, not private scratch files. Large model files, generated weights, large logs, and benchmark output dumps stay out of git.

The repo should be readable from GitHub without Obsidian, while Obsidian links can still be used for local navigation.

## Success Criteria

After setup, another agent should be able to continue the inference track by reading:

1. `AGENTS.md`
2. `START_HERE.md`
3. `projects/llm-inference-benchmark-lab/README.md`
4. The latest `daily/*.md`

The reader should understand the current track, owned files, current milestone, next command to try, and what not to touch.
