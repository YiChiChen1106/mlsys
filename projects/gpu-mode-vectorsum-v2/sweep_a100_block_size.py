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
MAX_OFFICIAL_SIZE = 52_428_800
DEFAULT_BLOCK_SIZES = (4096, 8192, 16384, 32768)


@dataclass(frozen=True)
class Candidate:
    block_size: int
    num_blocks: int
    final_block_size: int
    filename: str


def next_power_of_two(value: int) -> int:
    if value < 1:
        raise ValueError("value must be positive")
    return 1 << (value - 1).bit_length()


def candidate_for_block_size(block_size: int) -> Candidate:
    if block_size < 1:
        raise ValueError("block_size must be positive")

    num_blocks = (MAX_OFFICIAL_SIZE + block_size - 1) // block_size
    final_block_size = next_power_of_two(num_blocks)
    return Candidate(
        block_size=block_size,
        num_blocks=num_blocks,
        final_block_size=final_block_size,
        filename=f"submission_bs{block_size}.py",
    )


def render_submission(source: str, candidate: Candidate) -> str:
    rendered, block_count = re.subn(
        r"^BLOCK_SIZE = \d+$",
        f"BLOCK_SIZE = {candidate.block_size}",
        source,
        count=1,
        flags=re.MULTILINE,
    )
    rendered, final_count = re.subn(
        r"^FINAL_BLOCK_SIZE = \d+$",
        f"FINAL_BLOCK_SIZE = {candidate.final_block_size}",
        rendered,
        count=1,
        flags=re.MULTILINE,
    )

    if block_count != 1 or final_count != 1:
        raise ValueError("submission source must define BLOCK_SIZE and FINAL_BLOCK_SIZE once")
    return rendered


def write_variants(source_path: Path, output_dir: Path, block_sizes: tuple[int, ...]) -> list[Candidate]:
    source = source_path.read_text()
    output_dir.mkdir(parents=True, exist_ok=True)

    candidates = [candidate_for_block_size(block_size) for block_size in block_sizes]
    for candidate in candidates:
        (output_dir / candidate.filename).write_text(render_submission(source, candidate))

    manifest = {
        "leaderboard": LEADERBOARD,
        "gpu": GPU,
        "max_official_size": MAX_OFFICIAL_SIZE,
        "candidates": [asdict(candidate) for candidate in candidates],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return candidates


def run_benchmark(candidate: Candidate, output_dir: Path) -> dict[str, object]:
    submission_path = output_dir / candidate.filename
    result_path = output_dir / f"benchmark-bs{candidate.block_size}-a100.json"
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
        "candidate": asdict(candidate),
        "command": command,
        "returncode": completed.returncode,
        "elapsed_seconds": elapsed_seconds,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "result_file": str(result_path),
    }


def parse_block_sizes(raw_values: list[str]) -> tuple[int, ...]:
    if not raw_values:
        return DEFAULT_BLOCK_SIZES

    block_sizes: list[int] = []
    for raw in raw_values:
        for part in raw.split(","):
            part = part.strip()
            if part:
                block_sizes.append(int(part))
    return tuple(block_sizes)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate and optionally benchmark A100 BLOCK_SIZE variants.")
    parser.add_argument("--submission", default="submission.py", help="Base submission file to mutate.")
    parser.add_argument(
        "--output-dir",
        default="outputs/block-size-sweep",
        help="Directory for generated submissions and benchmark outputs.",
    )
    parser.add_argument(
        "--block-sizes",
        nargs="*",
        default=[],
        help="Block sizes as space-separated values or comma-separated groups.",
    )
    parser.add_argument("--run", action="store_true", help="Run Popcorn A100 benchmark for each generated variant.")
    args = parser.parse_args()

    source_path = Path(args.submission)
    output_dir = Path(args.output_dir)
    block_sizes = parse_block_sizes(args.block_sizes)
    candidates = write_variants(source_path, output_dir, block_sizes)

    print("Generated candidates:")
    for candidate in candidates:
        print(
            f"  {candidate.filename}: "
            f"BLOCK_SIZE={candidate.block_size}, "
            f"num_blocks={candidate.num_blocks}, "
            f"FINAL_BLOCK_SIZE={candidate.final_block_size}"
        )

    if not args.run:
        return

    results = []
    for candidate in candidates:
        print(f"\nRunning A100 benchmark for BLOCK_SIZE={candidate.block_size}")
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
