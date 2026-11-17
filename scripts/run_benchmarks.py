#!/usr/bin/env python3
"""Run the benchmarks used in the thesis.

Check if the host is configured correctly, build csvbench and run benchmarks.
The results go to `bench-data/<dataset>-<machine>-<label>.jsonl`. A dataset whose
result file is already complete is skipped, so the same command can be run
again after an interruption.

On macOS there is no pinning and no performance counters: csvbench measures time
only and asks the scheduler for the performance cores, and `--cpu` is not used.
"""

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

MACOS = sys.platform == "darwin"

REPO = Path(__file__).resolve().parent.parent
CSVBENCH = REPO / "target/release/csvbench"
RESULTS = REPO / "bench-data"

SAMPLES = 3
WARMUP_MS = 50
MIN_SAMPLE_MS = 20
CAPACITY = "64K"

# wait peroid after building the csvbench binary so benchmark results are not affected
COOLDOWN_S = 30

DATASETS = ["bench-github-balanced", "bench-kaggle-balanced"]

PARSER_SETS = {
    "all": ["*"],
    "scalar": ["compaction/*/-/*", "zerocopy/plain/-/*"],
    "best-simd": [
        "*/simd/nibble-map-64-*/bitshift",
    ],
    "third-party": ["csveee/*", "csv-core/dfa"],
}

# Highest value of kernel.perf_event_paranoid that still lets csvbench read the performance
# counters. It only counts its own user space code, which the kernel allows up to this level.
MAX_PERF_EVENT_PARANOID = 2
# The machine counts as busy above this 1 minute load average.
MAX_LOAD = 3.0 if MACOS else 0.5


def read(path: str) -> str | None:
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def low_power_mode() -> bool:
    """Whether macOS runs in Low Power Mode, which lowers the clock."""
    try:
        settings = subprocess.run(
            ["pmset", "-g"], capture_output=True, text=True
        ).stdout
    except OSError:
        return False
    return any(line.split() == ["lowpowermode", "1"] for line in settings.splitlines())


def problems(cpu: int | None, datasets: list[str], datasets_dir: Path) -> list[str]:
    """Everything that would make the results unreliable or not reproducible."""
    found = []

    if MACOS:
        if low_power_mode():
            found.append(
                "Low Power Mode is on, it lowers the clock: sudo pmset -a lowpowermode 0"
            )
    else:
        found += linux_problems(cpu)

    load = os.getloadavg()[0]
    if load > MAX_LOAD:
        found.append(
            f"the machine is busy (load {load:.2f}), other work disturbs the timing"
        )

    changed = subprocess.run(
        [
            "git",
            "status",
            "--porcelain",
            "--",
            "parser",
            "harness",
            "Cargo.toml",
            "Cargo.lock",
            ".cargo",
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if changed:
        found.append("uncommitted changes in the parser or harness")

    for name in datasets:
        if not (datasets_dir / f"{name}.json").exists():
            found.append(f"dataset {name!r} not found in {datasets_dir}")

    return found


def linux_problems(cpu: int) -> list[str]:
    """Pinning, counter permissions and frequency settings, which only Linux has."""
    found = []

    if cpu not in os.sched_getaffinity(0):
        found.append(f"CPU {cpu} does not exist or this process may not run on it")
    if cpu == 0:
        found.append("CPU 0 handles most interrupts, pin to another one")

    paranoid = read("/proc/sys/kernel/perf_event_paranoid")
    if paranoid is None or int(paranoid) > MAX_PERF_EVENT_PARANOID:
        found.append(
            f"kernel.perf_event_paranoid is {paranoid}, the performance counters need "
            f"{MAX_PERF_EVENT_PARANOID} or lower: "
            f"sudo sysctl kernel.perf_event_paranoid={MAX_PERF_EVENT_PARANOID}"
        )

    governor = read(f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_governor")
    if governor is not None and governor != "performance":
        found.append(
            f"the frequency governor of CPU {cpu} is {governor!r}, not 'performance': "
            "sudo cpupower frequency-set -g performance"
        )

    siblings = read(f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list")
    if siblings and siblings != str(cpu):
        found.append(
            f"CPU {cpu} shares its core with other hyperthreads ({siblings}). Turn them off: "
            "echo 0 | sudo tee /sys/devices/system/cpu/cpu<N>/online"
        )
    return found


def skipped_files(args: argparse.Namespace) -> list[str]:
    """Dataset files csvbench would leave out: not in the data store, of the wrong size, or
    without a usable dialect. If not checked they are skipped silently."""
    cmd = [
        str(CSVBENCH),
        "list",
        "--datasets-dir",
        str(args.datasets_dir),
        "--corpus",
        str(args.corpus),
    ]
    for dataset in args.dataset:
        cmd += ["--dataset", dataset]
    listing = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    if listing.returncode != 0:
        return [listing.stderr.strip() or "csvbench list failed"]
    return [
        line.strip() for line in listing.stdout.splitlines() if " skipped: " in line
    ]


def notes(cpu: int | None) -> list[str]:
    """Report settings that should be kept consistent across runs."""
    if MACOS:
        return [
            "macOS: no pinning and no performance counters, measuring time only",
            f"load {os.getloadavg()[0]:.2f}",
        ]
    found = []
    if read(f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_governor") is None:
        found.append(
            f"CPU {cpu} has no frequency governor, so its setting cannot be checked"
        )
    no_turbo = read("/sys/devices/system/cpu/intel_pstate/no_turbo")
    boost = read("/sys/devices/system/cpu/cpufreq/boost")
    if no_turbo is not None or boost is not None:
        on = no_turbo == "0" if no_turbo is not None else boost == "1"
        found.append(f"turbo is {'on' if on else 'off'}")
    return found


def is_complete(result: Path) -> bool:
    """A finished csvbench run ends with a line of type `end`. A run killed while writing
    leaves a partial last line."""
    lines = result.read_text().splitlines()
    try:
        return bool(lines) and json.loads(lines[-1]).get("type") == "end"
    except json.JSONDecodeError:
        return False


def command(
    dataset: str, parsers: list[str], result: Path, args: argparse.Namespace
) -> list[str]:
    cmd = [
        str(CSVBENCH),
        "run",
        "--dataset",
        dataset,
        "--datasets-dir",
        str(args.datasets_dir),
        "--corpus",
        str(args.corpus),
        "--samples",
        str(args.samples),
        "--warmup",
        str(WARMUP_MS),
        "--min-sample-time",
        str(MIN_SAMPLE_MS),
        "--capacity",
        CAPACITY,
        "--output",
        str(result),
    ]
    if args.cpu is not None:
        cmd += ["--cpu", str(args.cpu)]
    for parser in parsers:
        cmd += ["--parser", parser]
    return cmd


def main() -> int:
    cli = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    cli.add_argument(
        "--cpu",
        type=int,
        required=not MACOS,
        help="CPU to pin the benchmark to (Linux only, required there)",
    )
    cli.add_argument(
        "--set",
        action="append",
        choices=list(PARSER_SETS),
        help="which parser sets to measure (repeatable, merged into one run; default: all)",
    )
    cli.add_argument(
        "-p",
        "--parser",
        action="append",
        metavar="GLOB",
        help="also measure these parsers, as globs like for csvbench (repeatable). "
        "Added to the --set sets; without --set only these are measured",
    )
    cli.add_argument(
        "-d",
        "--dataset",
        action="append",
        metavar="NAME",
        help=f"measure only this dataset (repeatable, default: {' and '.join(DATASETS)})",
    )
    cli.add_argument(
        "--samples",
        type=int,
        default=SAMPLES,
        help=f"measured samples per file and parser (default {SAMPLES})",
    )
    cli.add_argument(
        "--label",
        help="last part of the result file names (default: the sets joined by `+`, "
        "plus `custom` with --parser)",
    )
    cli.add_argument(
        "--machine",
        default=socket.gethostname().split(".")[0],
        help="name of this machine in the result file names (default: the hostname)",
    )
    cli.add_argument("--datasets-dir", type=Path, default=REPO / "scripts/datasets")
    cli.add_argument("--corpus", type=Path, default=REPO / "scripts/data/corpus")
    cli.add_argument(
        "--dry-run",
        action="store_true",
        help="check and print the commands, measure nothing",
    )
    cli.add_argument(
        "--force",
        action="store_true",
        help="measure even if a check fails (for trying things out)",
    )
    args = cli.parse_args()
    if MACOS and args.cpu is not None:
        cli.error("--cpu: macOS cannot pin to a CPU")
    args.dataset = args.dataset or DATASETS
    # --parser alone measures just those parsers, with --set it adds to the sets
    sets = list(dict.fromkeys(args.set or ([] if args.parser else ["all"])))
    globs = [glob for name in sets for glob in PARSER_SETS[name]] + (args.parser or [])
    parsers = list(dict.fromkeys(globs))
    label = args.label or "+".join(sets + (["custom"] if args.parser else []))

    for note in notes(args.cpu):
        print(f"note: {note}")
    found = problems(args.cpu, args.dataset, args.datasets_dir)
    for problem in found:
        print(f"problem: {problem}", file=sys.stderr)
    if found and not args.force:
        print("not measuring. Fix the problems, or pass --force.", file=sys.stderr)
        return 1

    if not args.dry_run:
        # build the csvbench binary including csveee support
        started = time.monotonic()
        subprocess.run(
            [
                "cargo",
                "build",
                "--release",
                "-p",
                "harness",
                "--bin",
                "csvbench",
                "--features",
                "csveee",
            ],
            cwd=REPO,
            check=True,
        )
        if time.monotonic() - started > 10:
            print(f"built; waiting {COOLDOWN_S} s for the CPU to cool down", flush=True)
            time.sleep(COOLDOWN_S)

    skipped = skipped_files(args) if CSVBENCH.exists() else []
    for line in skipped[:20]:
        print(f"problem: {line}", file=sys.stderr)
    if skipped and not args.force:
        print(
            f"not measuring: csvbench would skip {len(skipped)} files of the datasets.",
            file=sys.stderr,
        )
        return 1

    for dataset in args.dataset:
        result = RESULTS / f"{dataset}-{args.machine}-{label}.jsonl"
        if result.exists():
            if is_complete(result):
                print(f"{result.name}: already complete, skipping")
                continue
            print(
                f"{result}: left over from an interrupted run. Delete or move it, then run again.",
                file=sys.stderr,
            )
            return 1
        cmd = command(dataset, parsers, result, args)
        print(" ".join(cmd), flush=True)
        if not args.dry_run:
            subprocess.run(cmd, cwd=REPO, check=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
