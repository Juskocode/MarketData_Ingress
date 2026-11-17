#!/usr/bin/env python3
"""Prepare a bounded Parquet snapshot for native FP32 inference replay."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import struct
import tempfile
from urllib.parse import urlsplit

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.fs as fs

MAGIC = b"MDLAKE01"
HEADER = struct.Struct("<8sQQQ32x")


def positive(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def utc_time(value: str) -> dt.datetime:
    result = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("Time bounds must include an explicit UTC offset")
    return result.astimezone(dt.timezone.utc)


def prepare(args: argparse.Namespace) -> dict:
    source = urlsplit(args.source)
    if source.username or source.password or source.query or source.fragment:
        raise ValueError("Use filesystem credential providers, not secrets in source URIs")
    columns = args.columns.split(",")
    order = args.order_by.split(",") if args.order_by else []
    if not all(columns) or len(set(columns)) != len(columns):
        raise ValueError("Feature columns must be nonempty and unique")
    if len(set(order)) != len(order) or any(not name for name in order):
        raise ValueError("Order columns must be nonempty and unique")
    output = Path(args.output)
    if args.max_rows > (2**63 - 1) // max(1, len(columns)) // 4:
        raise ValueError("Requested snapshot dimensions overflow")
    manifest_path = Path(str(output) + ".json")
    if output.exists() or manifest_path.exists():
        raise FileExistsError("Output/manifest already exists; choose a new immutable snapshot name")
    if source.scheme:
        filesystem, path = fs.FileSystem.from_uri(args.source)
    else:
        filesystem, path = fs.LocalFileSystem(), str(Path(args.source).resolve())
    dataset = ds.dataset(path, format="parquet", filesystem=filesystem, partitioning="hive")
    required = columns + order
    if args.timestamp_column:
        required.append(args.timestamp_column)
    if args.symbol_column:
        required.append(args.symbol_column)
    for name in required:
        if dataset.schema.get_field_index(name) < 0:
            raise ValueError(f"Column {name!r} is missing or ambiguous")
    for name in columns:
        kind = dataset.schema.field(name).type
        if not (pa.types.is_integer(kind) or pa.types.is_floating(kind)):
            raise ValueError(f"Feature {name!r} must be numeric, got {kind}")
    predicate = ds.scalar(True)
    if args.start or args.end:
        if not args.timestamp_column:
            raise ValueError("Time bounds require --timestamp-column")
        kind = dataset.schema.field(args.timestamp_column).type
        if not pa.types.is_timestamp(kind) or not kind.tz:
            raise ValueError("Time filtering requires an Arrow timestamp column with timezone")
        start = utc_time(args.start) if args.start else None
        end = utc_time(args.end) if args.end else None
        if start and end and start >= end:
            raise ValueError("--start must precede --end")
        if start:
            predicate &= ds.field(args.timestamp_column) >= pa.scalar(start, type=kind)
        if end:
            predicate &= ds.field(args.timestamp_column) < pa.scalar(end, type=kind)
    if args.symbol:
        if not args.symbol_column:
            raise ValueError("--symbol requires --symbol-column")
        predicate &= ds.field(args.symbol_column).isin(args.symbol)
    selected = list(dict.fromkeys(columns + order))
    # Stable path order, then physical row order. A global sort is bounded by
    # max_rows; exceeding the bound fails rather than silently truncating data.
    fragments = []
    for fragment in dataset.get_fragments(filter=predicate):
        if len(fragments) >= args.max_fragments:
            raise ValueError("Too many fragments; narrow the source partition or filters")
        fragments.append(fragment)
    fragments.sort(key=lambda item: item.path)
    batches, rows = [], 0
    for fragment in fragments:
        scanner = fragment.scanner(schema=dataset.schema, columns=selected,
                                   filter=predicate, batch_size=min(args.scan_rows, args.max_rows + 1),
                                   batch_readahead=1, fragment_readahead=1, use_threads=False)
        for batch in scanner.to_batches():
            rows += batch.num_rows
            if rows > args.max_rows:
                raise ValueError("Row bound exceeded; narrow filters or explicitly raise --max-rows")
            if batch.num_rows:
                batches.append(batch)
    if not rows:
        raise ValueError("No rows matched the requested snapshot")
    table = pa.Table.from_batches(batches).combine_chunks()
    if any(table[name].null_count for name in selected):
        raise ValueError("Null feature/order values are not allowed; prepare features upstream")
    if order:
        tie = "__md_ingress_row__"
        if tie in selected:
            raise ValueError(f"Reserved ordering column {tie}")
        table = table.append_column(tie, pa.array(np.arange(rows, dtype=np.int64)))
        table = table.take(pc.sort_indices(table, sort_keys=[(key, "ascending") for key in order + [tie]]))
    matrix = np.empty((rows, len(columns)), dtype="<f4")
    with np.errstate(over="ignore", invalid="ignore"):
        for index, name in enumerate(columns):
            matrix[:, index] = table[name].combine_chunks().to_numpy(zero_copy_only=False)
    if not np.isfinite(matrix).all():
        raise ValueError("Features contain NaN/infinity or overflow after FP32 conversion")
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "format": "lake-f32-v1", "rows": rows, "features": len(columns),
        "columns": columns, "order_by": order, "source": args.source,
        "fragment_count": len(fragments), "fragments": [item.path for item in fragments],
        "start_inclusive": args.start, "end_exclusive": args.end, "symbols": args.symbol,
        "pyarrow_version": pa.__version__, "numpy_version": np.__version__,
    }
    # Link completed temp files into place: atomic publication without clobber.
    temporaries = []
    installed = False
    try:
        with tempfile.NamedTemporaryFile(dir=output.parent, delete=False) as stream:
            temporaries.append(Path(stream.name))
            header = HEADER.pack(MAGIC, rows, len(columns), matrix.nbytes)
            stream.write(header)
            stream.write(memoryview(matrix).cast("B"))
            stream.flush()
            os.fsync(stream.fileno())
        digest = hashlib.sha256()
        with temporaries[0].open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        manifest["sha256"] = digest.hexdigest()
        with tempfile.NamedTemporaryFile(mode="w", dir=output.parent, delete=False) as stream:
            temporaries.append(Path(stream.name))
            json.dump(manifest, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporaries[0], output)
        installed = True
        os.link(temporaries[1], manifest_path)
    except BaseException:
        if installed:
            output.unlink()
        raise
    finally:
        for temporary in temporaries:
            temporary.unlink(missing_ok=True)
    return manifest


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("source", help="Parquet path, partitioned directory or Arrow filesystem URI")
    result.add_argument("--output", required=True)
    result.add_argument("--columns", required=True, help="Ordered comma-separated numeric model features")
    result.add_argument("--max-rows", type=positive, required=True, help="Fail rather than truncate beyond this bound")
    result.add_argument("--max-fragments", type=positive, default=1000)
    result.add_argument("--scan-rows", type=positive, default=65536)
    result.add_argument("--order-by", default="", help="Ascending sort columns, ties preserve source order")
    result.add_argument("--timestamp-column")
    result.add_argument("--start", help="Inclusive ISO timestamp with offset")
    result.add_argument("--end", help="Exclusive ISO timestamp with offset")
    result.add_argument("--symbol-column")
    result.add_argument("--symbol", action="append", default=[])
    return result


if __name__ == "__main__":
    cli = parser()
    try:
        print(json.dumps(prepare(cli.parse_args()), indent=2))
    except (ValueError, OSError, pa.ArrowException) as error:
        cli.exit(2, f"prepare: {error}\n")
