"""Convert a daily Binance Vision spot aggregate trade archive to Parquet."""
from __future__ import annotations

import csv
from collections.abc import Iterator
from datetime import date, timedelta
import io
import math
import os
from pathlib import Path
import re
import tempfile
import zipfile

import pyarrow as pa
import pyarrow.parquet as pq

SCHEMA = pa.schema([
    pa.field("event_time", pa.timestamp("us", tz="UTC"), nullable=False),
    pa.field("symbol", pa.string(), nullable=False),
    pa.field("sequence", pa.int64(), nullable=False),
    pa.field("price", pa.float64(), nullable=False),
    pa.field("quantity", pa.float64(), nullable=False),
    pa.field("quote_quantity", pa.float64(), nullable=False),
    pa.field("taker_side", pa.int8(), nullable=False),
])

BATCH_ROWS = 16_384
MAX_ARCHIVE_BYTES = 128 * 1024 * 1024
MAX_CSV_BYTES = 512 * 1024 * 1024
MAX_LINE_CHARS = 4096
MICROSECOND_START = date(2025, 1, 1)
EPOCH = date(1970, 1, 1)
HEADERS = (
    ("a", "p", "q", "f", "l", "t", "m", "m"),
    ("aggtradeid", "price", "quantity", "firsttradeid", "lasttradeid",
     "transacttime", "isbuyermaker", "isbestmatch"),
)
Trade = tuple[int, int, float, float, float, int]


def _lines(stream: io.TextIOWrapper) -> Iterator[str]:
    while line := stream.readline(MAX_LINE_CHARS + 1):
        if len(line) > MAX_LINE_CHARS:
            raise ValueError("Aggregate trade record exceeds the CSV line bound")
        yield line


def _boolean(value: str) -> bool:
    match value.lower():
        case "true":
            return True
        case "false":
            return False
    raise ValueError("Expected a boolean field")


def _parse_record(record: list[str], timestamp_scale: int, start_us: int, end_us: int) -> Trade:
    sequence_text, price_text, quantity_text, first_text, last_text, timestamp_text, maker, best = record
    sequence = int(sequence_text)
    if not 0 <= sequence < 2**63 or not 0 <= int(first_text) <= int(last_text):
        raise ValueError("Invalid trade IDs")
    timestamp_us = int(timestamp_text) * timestamp_scale
    if not start_us <= timestamp_us < end_us:
        raise ValueError("Timestamp outside archive day or incorrect unit")
    price = float(price_text)
    quantity = float(quantity_text)
    quote_quantity = price * quantity
    if not all(math.isfinite(value) and value > 0 for value in (price, quantity, quote_quantity)):
        raise ValueError("Price, quantity and quote quantity must be finite and positive")
    taker_side = -1 if _boolean(maker) else 1
    _boolean(best)
    return timestamp_us, sequence, price, quantity, quote_quantity, taker_side


def _write_batch(writer: pq.ParquetWriter, records: list[Trade], symbol: str) -> None:
    timestamps, sequences, prices, quantities, quote_quantities, sides = zip(*records)
    writer.write_batch(pa.RecordBatch.from_arrays(
        [pa.array(values, type=field.type) for field, values in zip(
            SCHEMA, (timestamps, [symbol] * len(records), sequences, prices,
                     quantities, quote_quantities, sides), strict=True,
        )],
        schema=SCHEMA,
    ))


def convert(archive: Path, output: Path, *, symbol: str, day: date, max_rows: int) -> int:
    """Convert a complete daily archive without extracting it or overwriting output."""
    if not re.fullmatch(r"[A-Z0-9]{3,24}", symbol):
        raise ValueError("Expected an uppercase Binance spot symbol")
    if max_rows <= 0:
        raise ValueError("Aggregate trade row limit must be positive")
    if output.exists():
        raise FileExistsError(f"Parquet output already exists: {output}")
    if archive.stat().st_size > MAX_ARCHIVE_BYTES:
        raise ValueError("Aggregate trade ZIP exceeds the archive size bound")

    csv_name = f"{symbol}-aggTrades-{day.isoformat()}.csv"
    start_us = (day - EPOCH) // timedelta(microseconds=1)
    end_us = start_us + 86_400_000_000
    timestamp_scale = 1 if day >= MICROSECOND_START else 1000
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    records: list[Trade] = []
    with zipfile.ZipFile(archive) as source:
        members = source.infolist()
        if len(members) != 1 or members[0].filename != csv_name:
            raise ValueError(f"Expected one CSV named {csv_name} in the ZIP")
        member = members[0]
        if member.file_size > MAX_CSV_BYTES:
            raise ValueError("Aggregate trade CSV exceeds the expanded size bound")
        with tempfile.NamedTemporaryFile(dir=output.parent, delete=False) as staging:
            staged_path = Path(staging.name)
        try:
            with pq.ParquetWriter(staged_path, SCHEMA, compression="zstd",
                                  use_dictionary=["symbol"]) as writer:
                with source.open(member) as compressed, io.TextIOWrapper(
                    compressed, encoding="utf-8-sig", newline=""
                ) as stream:
                    for line_number, record in enumerate(csv.reader(_lines(stream), strict=True), 1):
                        if line_number == 1 and tuple(field.lower().replace("_", "")
                                                      for field in record) in HEADERS:
                            continue
                        if rows == max_rows:
                            raise ValueError(f"Aggregate trade row bound exceeded ({max_rows})")
                        try:
                            trade = _parse_record(record, timestamp_scale, start_us, end_us)
                        except ValueError as error:
                            raise ValueError(f"Invalid aggregate trade on CSV row {line_number}: {error}") from error
                        records.append(trade)
                        rows += 1
                        if len(records) == BATCH_ROWS:
                            _write_batch(writer, records, symbol)
                            records.clear()
                if not rows:
                    raise ValueError("Aggregate trade archive has no data rows")
                if records:
                    _write_batch(writer, records, symbol)
            os.link(staged_path, output)
        finally:
            staged_path.unlink(missing_ok=True)
    return rows
