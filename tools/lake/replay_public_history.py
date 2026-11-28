#!/usr/bin/env python3
"""Exercise the native replay with pinned public trades and quotes."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date
import hashlib
import json
import os
from pathlib import Path
import subprocess
from tempfile import NamedTemporaryFile
from urllib.parse import urlsplit
from urllib.request import urlopen

import pyarrow as pa

import binance_aggtrades
import prepare


BATCH_SIZE = 8
MAX_DOWNLOAD_BYTES = 16 * 1024 * 1024


@dataclass(frozen=True)
class Fixture:
    url: str
    sha256: str
    snapshot_sha256: str
    features: tuple[str, ...]
    order_by: tuple[str, ...]
    rows: int
    filters: tuple[str, ...] = ()


FIXTURES = {
    "binance-trades": Fixture(
        "https://data.binance.vision/data/spot/daily/aggTrades/BTCUSDT/"
        "BTCUSDT-aggTrades-2025-02-01.zip",
        "815ba283fd809ae260d981bd1f7083a742e23aeaaf42282ed8bed5306592d503",
        "39a47ae08fd6c344a90c80efa6d72b518cb603584f1c6b69957b4669129ab8a9",
        ("price", "quantity", "quote_quantity", "taker_side"),
        ("event_time", "sequence"),
        3457,
        ("--timestamp-column", "event_time", "--start", "2025-02-01T00:00:00Z",
         "--end", "2025-02-01T00:05:00Z", "--symbol-column", "symbol", "--symbol", "BTCUSDT"),
    ),
    "crypto-lake-quotes": Fixture(
        "https://s3.eu-west-1.amazonaws.com/sample.crypto.lake/level_1/"
        "exchange=BINANCE/symbol=BTC-USDT/dt=2022-10-02/"
        "converted_a849a1d5a1154281958cfaf2e9f3d844.snappy.parquet",
        "fb38074929fb49eacac4c84ef3f4db1c3edaf36d4e7acd861ba6f684394f8e25",
        "28e420fdb2b97988e48ba326ef623447e86007550356b6c53103e3baf2cd817e",
        ("bid_0_price", "ask_0_price", "bid_0_size", "ask_0_size"),
        ("receipt_timestamp",),
        3064,
    ),
}


def check_cache(path: Path, expected: str) -> None:
    with path.open("rb") as stream:
        if os.fstat(stream.fileno()).st_size > MAX_DOWNLOAD_BYTES:
            raise ValueError(f"Cached source exceeds the download bound: {path}")
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != expected:
        raise ValueError(f"Cached source SHA-256 mismatch: {path}. Remove it before retrying.")


def fetch(fixture: Fixture, cache: Path, offline: bool) -> Path:
    name = Path(urlsplit(fixture.url).path).name
    path = cache / name
    if path.exists():
        check_cache(path, fixture.sha256)
        return path
    if offline:
        raise FileNotFoundError(f"No cached source for offline replay: {path}")

    cache.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with (
            urlopen(fixture.url, timeout=30) as response,
            NamedTemporaryFile(dir=cache, delete=False) as output,
        ):
            temporary = Path(output.name)
            digest = hashlib.sha256()
            size = 0
            for chunk in iter(lambda: response.read(64 * 1024), b""):
                size += len(chunk)
                if size > MAX_DOWNLOAD_BYTES:
                    raise ValueError(f"Download exceeds the byte bound: {name}")
                digest.update(chunk)
                output.write(chunk)
            if digest.hexdigest() != fixture.sha256:
                raise ValueError(f"Downloaded source SHA-256 mismatch: {name}")
        try:
            os.link(temporary, path)
        except FileExistsError:
            check_cache(path, fixture.sha256)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path


def verify_snapshot(snapshot: Path, fixture: Fixture, manifest: dict, replayed_rows: int) -> str:
    data = snapshot.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != manifest["sha256"] or digest != fixture.snapshot_sha256:
        raise ValueError("Snapshot SHA-256 does not match the pinned FP32 output")
    columns = len(fixture.features)
    row_bytes = columns * 4
    payload_size = fixture.rows * row_bytes
    header_size = prepare.HEADER.size
    header = prepare.HEADER.pack(prepare.MAGIC, fixture.rows, columns, payload_size)
    if data[:header_size] != header or len(data) != header_size + payload_size:
        raise ValueError("Snapshot shape does not match the pinned historical window")

    fingerprint = 14695981039346656037
    for byte in memoryview(data)[header_size:header_size + replayed_rows * row_bytes]:
        fingerprint = ((fingerprint ^ byte) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return f"{fingerprint:016x}"


def replay(key: str, fixture: Fixture, ingress: Path, directory: Path, archive: Path) -> dict:
    directory.mkdir()
    parquet = archive
    source_rows = None
    if key == "binance-trades":
        parquet = directory / "trades.parquet"
        source_rows = binance_aggtrades.convert(
            archive, parquet, symbol="BTCUSDT", day=date(2025, 2, 1), max_rows=1_000_000,
        )

    snapshot = directory / "replay.mdlake"
    manifest = prepare.prepare(prepare.parser().parse_args([
        str(parquet), "--output", str(snapshot), "--columns", ",".join(fixture.features),
        "--order-by", ",".join(fixture.order_by), "--max-rows", str(fixture.rows),
        *fixture.filters,
    ]))
    if manifest["rows"] != fixture.rows:
        raise ValueError(f"{key}: expected {fixture.rows} snapshot rows, got {manifest['rows']}")

    iterations, unused = divmod(fixture.rows, BATCH_SIZE)
    replayed = iterations * BATCH_SIZE
    fingerprint = verify_snapshot(snapshot, fixture, manifest, replayed)
    metrics_path = directory / "metrics.json"
    run = subprocess.run([
        str(ingress), "--backend", "mock", "--input-file", str(snapshot),
        "--input-size", str(len(fixture.features)), "--batch", str(BATCH_SIZE),
        "--warmup", "16", "--iterations", str(iterations), "--json-out", str(metrics_path),
    ], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, timeout=60)
    if run.returncode:
        raise RuntimeError(f"{key}: native replay exited {run.returncode}\n{run.stderr.strip()}")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if (
        metrics["backend"] != "mock-cpp-cpu" or metrics["fallback_used"]
        or metrics["batch_size"] != BATCH_SIZE or metrics["input_size"] != len(fixture.features)
        or metrics["samples"] != iterations or metrics["input"]["source"] != "lake-f32-v1"
        or metrics["input"]["dataset_rows"] != fixture.rows
        or metrics["input"]["fingerprint"] != fingerprint
    ):
        raise ValueError(f"{key}: native metrics disagree with the verified replay workload")

    return {
        "dataset": key,
        "source_url": fixture.url,
        "source_sha256": fixture.sha256,
        "source_bytes": archive.stat().st_size,
        "snapshot_sha256": manifest["sha256"],
        "source_rows": fixture.rows if source_rows is None else source_rows,
        "start_inclusive": manifest["start_inclusive"],
        "end_exclusive": manifest["end_exclusive"],
        "ordered_by": list(fixture.order_by),
        "features": list(fixture.features),
        "snapshot_rows": fixture.rows,
        "replayed_rows": replayed,
        "unused_tail_rows": unused,
        "batch_size": BATCH_SIZE,
        "requests": iterations,
        "fingerprint": fingerprint,
        "build": metrics["build"],
        "backend": metrics["backend"],
        "p99_us": metrics["latency_us"]["end_to_end"]["p99"],
        "metrics": f"{key}/metrics.json",
    }


def main() -> None:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--ingress", type=Path, required=True, help="Built marketdata_ingress binary")
    cli.add_argument("--output", type=Path, required=True, help="New directory for the run")
    cli.add_argument("--cache", type=Path, default=Path("out/public-history-cache"))
    cli.add_argument("--dataset", choices=FIXTURES, help="Run one dataset (default: both)")
    cli.add_argument("--offline", action="store_true", help="Require verified cached sources")
    args = cli.parse_args()
    ingress = args.ingress.resolve()
    if not ingress.is_file() or not os.access(ingress, os.X_OK):
        cli.error(f"Native replay binary is missing or not executable: {ingress}")
    keys = (args.dataset,) if args.dataset else FIXTURES
    try:
        args.output.mkdir(parents=True, exist_ok=False)
        runs = []
        for key in keys:
            fixture = FIXTURES[key]
            archive = fetch(fixture, args.cache, args.offline)
            runs.append(replay(key, fixture, ingress, args.output / key, archive))
        report = json.dumps({"schema_version": 1, "runs": runs}, indent=2) + "\n"
        (args.output / "report.json").write_text(report, encoding="utf-8")
        print(report, end="")
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired, pa.ArrowException) as error:
        cli.exit(2, f"public history replay: {error}\n")


if __name__ == "__main__":
    main()
