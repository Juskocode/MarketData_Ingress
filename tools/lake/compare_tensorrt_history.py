#!/usr/bin/env python3
"""Compare direct TensorRT and CUDA Graph latency on the pinned historical snapshots."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import median
import subprocess

from replay_public_history import BATCH_SIZE, FIXTURES, Fixture, verify_snapshot


SCOPES = ("end_to_end", "device_compute")
QUANTILES = ("p50", "p99")


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def invoke(command: list[str], log: Path, timeout: int) -> None:
    with log.open("x", encoding="utf-8") as stream:
        result = subprocess.run(
            command, stdout=stream, stderr=subprocess.STDOUT, timeout=timeout,
        )
    if result.returncode:
        raise RuntimeError(f"{command[0]} exited {result.returncode}. See {log}")


def validate_metrics(
    metrics: dict, fixture: Fixture, fingerprint: str, model: Path, graph: bool,
) -> None:
    iterations = fixture.rows // BATCH_SIZE
    columns = len(fixture.features)
    expected = {
        "schema_version": 2,
        "model": str(model),
        "backend": "tensorrt-gpu",
        "requested_backend": "tensorrt",
        "samples": iterations,
        "input_size": columns,
        "batch_size": BATCH_SIZE,
    }
    if (
        any(metrics.get(key) != value for key, value in expected.items())
        or metrics.get("fallback_used") is not False
        or metrics.get("cuda_graph") is not graph
    ):
        raise ValueError("Replay backend, graph mode or request shape does not match")

    source = metrics["input"]
    if (
        source["source"] != "lake-f32-v1" or source["dataset_rows"] != fixture.rows
        or source["row_offset"] != 0 or source["fingerprint"] != fingerprint
    ):
        raise ValueError("Native replay fingerprint or row window does not match")

    identity = metrics["validation"]
    if identity != {
        "kind": "identity",
        "tolerance": 0,
        "compared_elements": iterations * BATCH_SIZE * columns,
        "mismatches": 0,
        "max_abs_error": 0,
        "passed": True,
    }:
        raise ValueError("FP32 identity validation did not cover all inputs with zero mismatches")

    environment = metrics["environment"]
    if not all(environment.get(field) for field in (
        "device", "backend_version", "compute_capability", "cuda_runtime",
        "cuda_driver", "device_memory_bytes",
    )):
        raise ValueError("Native metrics are missing TensorRT, CUDA or GPU provenance")

    for scope in SCOPES:
        distribution = metrics["latency_us"][scope]
        if (
            not distribution or distribution["samples"] != iterations
            or not all(math.isfinite(distribution[q]) for q in QUANTILES)
            or not (0 <= distribution["p50"] <= distribution["p99"] <= distribution["max"])
            or not math.isfinite(distribution["max"])
            or distribution["max"] <= 0
        ):
            raise ValueError(f"Missing or invalid {scope} latency measurements")


def main() -> None:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--history", type=Path, required=True, help="Completed public replay directory")
    cli.add_argument("--ingress", type=Path, required=True, help="TensorRT-enabled marketdata_ingress")
    cli.add_argument("--output", type=Path, required=True, help="New directory for GPU evidence")
    cli.add_argument("--repetitions", type=int, default=6, help="Even number of direct/graph pairs")
    cli.add_argument("--warmup", type=int, default=256, help="Warmup requests before each measured run")
    args = cli.parse_args()
    if args.repetitions < 2 or args.repetitions % 2:
        cli.error("--repetitions must be an even integer of at least 2")
    if args.warmup < 1:
        cli.error("--warmup must be positive")

    ingress = args.ingress.resolve()
    builder = ingress.with_name("marketdata_build_engine")
    for binary in (ingress, builder):
        if not binary.is_file() or not os.access(binary, os.X_OK):
            cli.error(f"Missing executable: {binary}. Build with -DMD_REQUIRE_TENSORRT=ON.")

    try:
        prepared = {}
        columns = len(next(iter(FIXTURES.values())).features)
        for key, fixture in FIXTURES.items():
            if len(fixture.features) != columns:
                raise ValueError("Public snapshot feature counts differ")
            snapshot = (args.history / key / "replay.mdlake").resolve()
            manifest = json.loads(snapshot.with_suffix(".mdlake.json").read_text(encoding="utf-8"))
            iterations, unused_rows = divmod(fixture.rows, BATCH_SIZE)
            fingerprint = verify_snapshot(snapshot, fixture, manifest, iterations * BATCH_SIZE)
            prepared[key] = (snapshot, iterations, unused_rows, fingerprint)

        out = args.output.resolve()
        out.mkdir(parents=True, exist_ok=False)
        model = out / "identity.engine"
        width = columns * BATCH_SIZE
        invoke([str(builder), "--output", str(model), "--input-size", str(width)],
               out / "build-engine.log", timeout=180)
        engine_digest = sha256(model)

        build = environment = None
        results = []
        for key, fixture in FIXTURES.items():
            snapshot, iterations, unused_rows, fingerprint = prepared[key]
            directory = out / key
            directory.mkdir()
            metrics_by_mode = {"direct": [], "graph": []}
            runs = []

            for pair in range(1, args.repetitions + 1):
                modes = ("direct", "graph") if pair % 2 else ("graph", "direct")
                for mode in modes:
                    filename = f"{pair:02d}-{mode}"
                    metrics_path = directory / f"{filename}.json"
                    command = [
                        str(ingress), "--backend", "tensorrt", "--model", str(model),
                        "--input-file", str(snapshot), "--input-size", str(columns),
                        "--batch", str(BATCH_SIZE), "--iterations", str(iterations),
                        "--warmup", str(args.warmup), "--verify-identity", "--tolerance", "0",
                        "--json-out", str(metrics_path),
                    ]
                    if mode == "graph":
                        command.append("--cuda-graph")
                    invoke(command, directory / f"{filename}.log", timeout=60)
                    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                    try:
                        validate_metrics(metrics, fixture, fingerprint, model, mode == "graph")
                        if build is None:
                            build, environment = metrics["build"], metrics["environment"]
                        elif build != metrics["build"] or environment != metrics["environment"]:
                            raise ValueError("Build or GPU environment changed during the comparison")
                    except (ValueError, TypeError, KeyError) as error:
                        raise ValueError(f"{metrics_path}: {error}") from error
                    metrics_by_mode[mode].append(metrics)
                    runs.append({
                        "pair": pair,
                        "mode": mode,
                        "metrics": metrics_path.relative_to(out).as_posix(),
                    })

            medians = {
                mode: {
                    scope: {
                        quantile: median(run["latency_us"][scope][quantile] for run in mode_runs)
                        for quantile in QUANTILES
                    }
                    for scope in SCOPES
                }
                for mode, mode_runs in metrics_by_mode.items()
            }
            results.append({
                "dataset": key,
                "source_sha256": fixture.sha256,
                "snapshot_sha256": fixture.snapshot_sha256,
                "fingerprint": fingerprint,
                "requests_per_run": iterations,
                "replayed_rows_per_run": iterations * BATCH_SIZE,
                "unused_tail_rows": unused_rows,
                "zero_mismatch_runs": len(runs),
                "medians_of_run_percentiles_us": medians,
                "runs": runs,
            })

        if sha256(model) != engine_digest:
            raise ValueError("The serialized identity engine changed during the comparison")
        report = {
            "schema_version": 1,
            "backend": "tensorrt-gpu",
            "build": build,
            "environment": environment,
            "engine": {"kind": "fp32_identity", "shape": [1, width], "sha256": engine_digest},
            "batch_size": BATCH_SIZE,
            "features_per_row": columns,
            "warmup_requests_per_run": args.warmup,
            "paired_repetitions": args.repetitions,
            "datasets": results,
        }
        document = json.dumps(report, indent=2) + "\n"
        (out / "report.json").write_text(document, encoding="utf-8")
        print(document, end="")
    except (OSError, ValueError, RuntimeError, TypeError, KeyError, subprocess.TimeoutExpired) as error:
        cli.exit(2, f"TensorRT historical comparison: {error}\n")


if __name__ == "__main__":
    main()
