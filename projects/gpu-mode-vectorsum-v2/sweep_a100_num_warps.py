from __future__ import annotations

import argparse
import json
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path


LEADERBOARD = "vectorsum_v2"
GPU = "A100"
DEFAULT_PARTIAL_WARPS = (4, 8, 16)
DEFAULT_FINAL_WARPS = (4, 8, 16)
ALLOWED_WARPS = {1, 2, 4, 8, 16, 32}


@dataclass(frozen=True)
class Candidate:
    num_warps: int
    final_num_warps: int
    filename: str


def parse_warps(raw_values: list[str], default: tuple[int, ...]) -> tuple[int, ...]:
    if not raw_values:
        return default

    warps: list[int] = []
    for raw in raw_values:
        for part in raw.split(","):
            part = part.strip()
            if not part:
                continue

            value = int(part)
            if value not in ALLOWED_WARPS:
                raise ValueError(f"num_warps must be one of {sorted(ALLOWED_WARPS)}")
            warps.append(value)
    return tuple(warps)


def candidate_grid(partial_warps: tuple[int, ...], final_warps: tuple[int, ...]) -> list[Candidate]:
    return [
        Candidate(
            num_warps=num_warps,
            final_num_warps=final_num_warps,
            filename=f"submission_wp{num_warps}_wf{final_num_warps}.py",
        )
        for num_warps in partial_warps
        for final_num_warps in final_warps
    ]


def render_submission(source: str, candidate: Candidate) -> str:
    partial_before = """    _partial_sum_kernel[(num_blocks,)](
        x,
        partial,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
"""
    partial_after = f"""    _partial_sum_kernel[(num_blocks,)](
        x,
        partial,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps={candidate.num_warps},
    )
"""
    final_before = """    _final_sum_kernel[(1,)](
        partial,
        output,
        num_blocks,
        FINAL_BLOCK_SIZE=FINAL_BLOCK_SIZE,
    )
"""
    final_after = f"""    _final_sum_kernel[(1,)](
        partial,
        output,
        num_blocks,
        FINAL_BLOCK_SIZE=FINAL_BLOCK_SIZE,
        num_warps={candidate.final_num_warps},
    )
"""

    if source.count(partial_before) != 1 or source.count(final_before) != 1:
        raise ValueError("submission source must have the expected two-stage kernel launches")

    return source.replace(partial_before, partial_after).replace(final_before, final_after)


def write_variants(
    source_path: Path,
    output_dir: Path,
    partial_warps: tuple[int, ...],
    final_warps: tuple[int, ...],
) -> list[Candidate]:
    source = source_path.read_text()
    output_dir.mkdir(parents=True, exist_ok=True)

    candidates = candidate_grid(partial_warps, final_warps)
    for candidate in candidates:
        (output_dir / candidate.filename).write_text(render_submission(source, candidate))

    manifest = {
        "leaderboard": LEADERBOARD,
        "gpu": GPU,
        "submission": str(source_path),
        "candidates": [asdict(candidate) for candidate in candidates],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return candidates


def run_benchmark(candidate: Candidate, output_dir: Path) -> dict[str, object]:
    submission_path = output_dir / candidate.filename
    result_path = output_dir / f"benchmark-wp{candidate.num_warps}-wf{candidate.final_num_warps}-a100.json"
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
    parser = argparse.ArgumentParser(description="Generate and optionally benchmark A100 num_warps variants.")
    parser.add_argument("--submission", default="submission.py", help="Base submission file to mutate.")
    parser.add_argument(
        "--output-dir",
        default="outputs/num-warps-sweep",
        help="Directory for generated submissions and benchmark outputs.",
    )
    parser.add_argument(
        "--partial-warps",
        nargs="*",
        default=[],
        help="Partial kernel num_warps values as space-separated values or comma-separated groups.",
    )
    parser.add_argument(
        "--final-warps",
        nargs="*",
        default=[],
        help="Final kernel num_warps values as space-separated values or comma-separated groups.",
    )
    parser.add_argument("--run", action="store_true", help="Run Popcorn A100 benchmark for each generated variant.")
    args = parser.parse_args()

    source_path = Path(args.submission)
    output_dir = Path(args.output_dir)
    partial_warps = parse_warps(args.partial_warps, DEFAULT_PARTIAL_WARPS)
    final_warps = parse_warps(args.final_warps, DEFAULT_FINAL_WARPS)
    candidates = write_variants(source_path, output_dir, partial_warps, final_warps)

    print("Generated candidates:")
    for candidate in candidates:
        print(
            f"  {candidate.filename}: "
            f"partial num_warps={candidate.num_warps}, "
            f"final num_warps={candidate.final_num_warps}"
        )

    if not args.run:
        return

    results = []
    for candidate in candidates:
        print(
            f"\nRunning A100 benchmark for "
            f"partial num_warps={candidate.num_warps}, final num_warps={candidate.final_num_warps}"
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
