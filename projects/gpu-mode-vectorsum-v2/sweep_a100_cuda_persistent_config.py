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
    grid_blocks: int
    filename: str

    @property
    def module_name(self) -> str:
        return f"vectorsum_cuda_persistent_g{self.grid_blocks}"


DEFAULT_CANDIDATES = (
    Candidate(grid_blocks=256, filename="submission_cuda_persistent_g256.py"),
    Candidate(grid_blocks=512, filename="submission_cuda_persistent_g512.py"),
    Candidate(grid_blocks=1024, filename="submission_cuda_persistent_g1024.py"),
    Candidate(grid_blocks=2048, filename="submission_cuda_persistent_g2048.py"),
)


def validate_candidate(grid_blocks: int) -> None:
    if grid_blocks < 1:
        raise ValueError("grid_blocks must be positive")


def parse_candidate_specs(raw_values: list[str]) -> tuple[Candidate, ...]:
    if not raw_values:
        return DEFAULT_CANDIDATES

    candidates: list[Candidate] = []
    for raw in raw_values:
        for part in raw.split(","):
            part = part.strip()
            if not part:
                continue

            grid_blocks = int(part)
            validate_candidate(grid_blocks)
            candidates.append(
                Candidate(
                    grid_blocks=grid_blocks,
                    filename=f"submission_cuda_persistent_g{grid_blocks}.py",
                )
            )

    return tuple(candidates)


def render_submission(source: str, candidate: Candidate) -> str:
    validate_candidate(candidate.grid_blocks)

    rendered, grid_count = re.subn(
        r"^GRID_BLOCKS = \d+$",
        f"GRID_BLOCKS = {candidate.grid_blocks}",
        source,
        count=1,
        flags=re.MULTILINE,
    )
    rendered, name_count = re.subn(
        r'name="vectorsum_cuda_persistent_[^"]+"',
        f'name="{candidate.module_name}"',
        rendered,
        count=1,
    )

    if grid_count != 1 or name_count != 1:
        raise ValueError("submission source must define GRID_BLOCKS and load_inline name once")
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
        "candidates": [asdict(candidate) for candidate in candidates],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return list(candidates)


def run_benchmark(candidate: Candidate, output_dir: Path) -> dict[str, object]:
    submission_path = output_dir / candidate.filename
    result_path = output_dir / f"benchmark-cuda-persistent-g{candidate.grid_blocks}-a100.json"
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate and optionally benchmark CUDA persistent-grid variants.")
    parser.add_argument("--submission", default="submission_cuda_persistent.py", help="Base CUDA persistent file.")
    parser.add_argument(
        "--output-dir",
        default="outputs/cuda-persistent-config-sweep",
        help="Directory for generated submissions and benchmark outputs.",
    )
    parser.add_argument(
        "--grid-blocks",
        nargs="*",
        default=[],
        help="Grid block counts as space-separated values or comma-separated groups.",
    )
    parser.add_argument("--run", action="store_true", help="Run Popcorn A100 benchmark for each generated variant.")
    args = parser.parse_args()

    source_path = Path(args.submission)
    output_dir = Path(args.output_dir)
    candidates = parse_candidate_specs(args.grid_blocks)
    written = write_variants(source_path, output_dir, candidates)

    print("Generated candidates:")
    for candidate in written:
        print(f"  {candidate.filename}: GRID_BLOCKS={candidate.grid_blocks}")

    if not args.run:
        return

    results = []
    for candidate in written:
        print(f"\nRunning A100 benchmark for GRID_BLOCKS={candidate.grid_blocks}")
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
