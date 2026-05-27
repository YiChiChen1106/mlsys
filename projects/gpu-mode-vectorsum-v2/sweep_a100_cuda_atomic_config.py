from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path


LEADERBOARD = "vectorsum_v2"
GPU = "A100"


@dataclass(frozen=True)
class Candidate:
    threads: int
    items_per_thread: int
    filename: str

    @property
    def elements_per_block(self) -> int:
        return self.threads * self.items_per_thread

    @property
    def module_name(self) -> str:
        return f"vectorsum_cuda_atomic_t{self.threads}_i{self.items_per_thread}"


DEFAULT_CANDIDATES = (
    Candidate(threads=256, items_per_thread=32, filename="submission_cuda_atomic_t256_i32.py"),
    Candidate(threads=256, items_per_thread=64, filename="submission_cuda_atomic_t256_i64.py"),
    Candidate(threads=512, items_per_thread=32, filename="submission_cuda_atomic_t512_i32.py"),
    Candidate(threads=256, items_per_thread=128, filename="submission_cuda_atomic_t256_i128.py"),
)


def validate_candidate(threads: int, items_per_thread: int) -> None:
    if threads < 32 or threads > 1024 or threads % 32 != 0:
        raise ValueError("threads must be a multiple of 32 between 32 and 1024")
    if items_per_thread < 1:
        raise ValueError("items_per_thread must be positive")


def parse_candidate_specs(raw_values: list[str]) -> tuple[Candidate, ...]:
    if not raw_values:
        return DEFAULT_CANDIDATES

    candidates: list[Candidate] = []
    for raw in raw_values:
        for part in raw.split(","):
            part = part.strip().lower()
            if not part:
                continue

            if "x" not in part:
                raise ValueError("candidate specs must look like 256x32")
            raw_threads, raw_items = part.split("x", 1)
            threads = int(raw_threads)
            items_per_thread = int(raw_items)
            validate_candidate(threads, items_per_thread)
            candidates.append(
                Candidate(
                    threads=threads,
                    items_per_thread=items_per_thread,
                    filename=f"submission_cuda_atomic_t{threads}_i{items_per_thread}.py",
                )
            )

    return tuple(candidates)


def render_submission(source: str, candidate: Candidate) -> str:
    validate_candidate(candidate.threads, candidate.items_per_thread)

    rendered, threads_count = re.subn(
        r"^THREADS = \d+$",
        f"THREADS = {candidate.threads}",
        source,
        count=1,
        flags=re.MULTILINE,
    )
    rendered, items_count = re.subn(
        r"^ITEMS_PER_THREAD = \d+$",
        f"ITEMS_PER_THREAD = {candidate.items_per_thread}",
        rendered,
        count=1,
        flags=re.MULTILINE,
    )
    rendered, name_count = re.subn(
        r'name="vectorsum_cuda_atomic_[^"]+"',
        f'name="{candidate.module_name}"',
        rendered,
        count=1,
    )

    if threads_count != 1 or items_count != 1 or name_count != 1:
        raise ValueError("submission source must define THREADS, ITEMS_PER_THREAD, and load_inline name once")
    return rendered


def write_variants(source_path: Path, output_dir: Path, candidates: tuple[Candidate, ...]) -> list[Candidate]:
    source = source_path.read_text()
    output_dir.mkdir(parents=True, exist_ok=True)

    for candidate in candidates:
        (output_dir / candidate.filename).write_text(render_submission(source, candidate))

    manifest = {
        "leaderboard": LEADERBOARD,
        "gpu": GPU,
        "submission": str(source_path),
        "candidates": [asdict(candidate) | {"elements_per_block": candidate.elements_per_block} for candidate in candidates],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return list(candidates)


def run_benchmark(candidate: Candidate, output_dir: Path) -> dict[str, object]:
    submission_path = output_dir / candidate.filename
    result_path = output_dir / f"benchmark-cuda-atomic-t{candidate.threads}-i{candidate.items_per_thread}-a100.json"
    command = [
        "popcorn-cli",
        "submit",
        "--no-tui",
        "--leaderboard",
        LEADERBOARD,
        "--gpu",
        GPU,
        "--mode",
        "benchmark",
        "--output",
        str(result_path),
        str(submission_path),
    ]

    started = time.time()
    completed = subprocess.run(command, text=True, capture_output=True)
    elapsed_seconds = time.time() - started

    return {
        "candidate": asdict(candidate) | {"elements_per_block": candidate.elements_per_block},
        "command": command,
        "returncode": completed.returncode,
        "elapsed_seconds": elapsed_seconds,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "result_file": str(result_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate and optionally benchmark CUDA atomic config variants.")
    parser.add_argument("--submission", default="submission_cuda_atomic.py", help="Base CUDA atomic file to mutate.")
    parser.add_argument(
        "--output-dir",
        default="outputs/cuda-atomic-config-sweep",
        help="Directory for generated submissions and benchmark outputs.",
    )
    parser.add_argument(
        "--candidates",
        nargs="*",
        default=[],
        help="Candidate specs like 256x32 or comma-separated groups like 256x32,256x64.",
    )
    parser.add_argument("--run", action="store_true", help="Run Popcorn A100 benchmark for each generated variant.")
    args = parser.parse_args()

    source_path = Path(args.submission)
    output_dir = Path(args.output_dir)
    candidates = parse_candidate_specs(args.candidates)
    written = write_variants(source_path, output_dir, candidates)

    print("Generated candidates:")
    for candidate in written:
        print(
            f"  {candidate.filename}: "
            f"THREADS={candidate.threads}, "
            f"ITEMS_PER_THREAD={candidate.items_per_thread}, "
            f"ELEMENTS_PER_BLOCK={candidate.elements_per_block}"
        )

    if not args.run:
        return

    results = []
    for candidate in written:
        print(
            f"\nRunning A100 benchmark for "
            f"THREADS={candidate.threads}, ITEMS_PER_THREAD={candidate.items_per_thread}"
        )
        result = run_benchmark(candidate, output_dir)
        results.append(result)
        print(result["stdout"])
        if result["returncode"] != 0:
            print(result["stderr"])
            raise SystemExit(result["returncode"])

    summary_path = output_dir / "benchmark-summary.json"
    summary_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {summary_path}")


if __name__ == "__main__":
    main()
