#!/usr/bin/env python3
"""Parquet -> bounded snapshot -> native replay integration."""
import datetime as dt
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "lake"))
import prepare

INGRESS = sys.argv.pop(1)
COMPARE = sys.argv.pop(1)


class LakePreparationTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / "parquet"
        self.source.mkdir()
        epoch = dt.datetime(2026, 11, 16, tzinfo=dt.timezone.utc)
        for name, seconds in (("a", [4, 0, 2, 6]), ("b", [5, 1, 3, 7])):
            table = pa.table({
                "time": pa.array([epoch + dt.timedelta(seconds=n) for n in seconds],
                                 type=pa.timestamp("ns", tz="UTC")),
                "symbol": ["OTHER" if n == 2 else "TSLA" for n in seconds],
                "bid": [100.0 + n for n in seconds],
                "ask": [101.0 + n for n in seconds],
            })
            pq.write_table(table, self.source / f"{name}.parquet", row_group_size=2)
        self.output = self.root / "snapshot.mdlake"

    def tearDown(self):
        self.temp.cleanup()

    def args(self, *extra):
        return prepare.parser().parse_args([
            str(self.source), "--output", str(self.output), "--columns", "ask,bid",
            "--max-rows", "16", *extra,
        ])

    def test_filtered_sorted_native_replay(self):
        manifest = prepare.prepare(self.args(
            "--timestamp-column", "time", "--start", "2026-11-16T00:00:00Z",
            "--end", "2026-11-16T00:00:07Z", "--symbol-column", "symbol", "--symbol", "TSLA",
            "--order-by", "time", "--scan-rows", "2"))
        binary = self.output.read_bytes()
        magic, rows, features, size = struct.unpack_from("<8sQQQ", binary)
        self.assertEqual((magic, rows, features, size), (b"MDLAKE01", 6, 2, 48))
        matrix = np.frombuffer(binary, dtype="<f4", offset=64).reshape(6, 2)
        np.testing.assert_array_equal(matrix[:, 0], [101, 102, 104, 105, 106, 107])
        np.testing.assert_array_equal(matrix[:, 1], matrix[:, 0] - 1)
        self.assertEqual(manifest["sha256"], hashlib.sha256(binary).hexdigest())
        self.assertEqual(manifest["columns"], ["ask", "bid"])
        metrics = self.root / "metrics.json"
        result = subprocess.run([
            INGRESS, "--backend", "mock", "--input-file", str(self.output),
            "--input-size", "2", "--batch", "2", "--iterations", "3", "--warmup", "2",
            "--json-out", str(metrics)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(metrics.read_text())
        self.assertEqual(report["input"]["source"], "lake-f32-v1")
        self.assertEqual(report["input"]["dataset_rows"], 6)
        self.assertEqual(report["samples"], 3)
        self.assertEqual(len(report["input"]["fingerprint"]), 16)
        comparison = subprocess.run([COMPARE, "--baseline", str(metrics), "--candidate", str(metrics)])
        self.assertEqual(comparison.returncode, 0)
        altered = self.root / "altered.json"
        report["input"]["fingerprint"] = "different"
        altered.write_text(json.dumps(report))
        comparison = subprocess.run([COMPARE, "--baseline", str(metrics), "--candidate", str(altered)],
                                    capture_output=True, text=True)
        self.assertEqual(comparison.returncode, 4)
        self.assertIn("input_workload", comparison.stdout + comparison.stderr)
        # Selecting more requests than complete batches fails without repeating rows.
        overflow = subprocess.run([INGRESS, "--backend", "mock", "--input-file", str(self.output),
            "--input-size", "2", "--iterations", "7"], capture_output=True)
        self.assertNotEqual(overflow.returncode, 0)

    def test_unsorted_export_preserves_fragment_and_row_order(self):
        manifest = prepare.prepare(self.args("--scan-rows", "2"))
        binary = self.output.read_bytes()
        expected = np.array([105, 101, 103, 107, 106, 102, 104, 108], dtype="<f4")
        expected_matrix = np.column_stack((expected, expected - 1)).astype("<f4")
        self.assertEqual(binary, prepare.HEADER.pack(b"MDLAKE01", 8, 2, 64)
                         + expected_matrix.tobytes())
        self.assertEqual(manifest["sha256"], hashlib.sha256(binary).hexdigest())

    def test_row_bound_fails_without_partial_output(self):
        with self.assertRaisesRegex(ValueError, "Row bound exceeded"):
            prepare.prepare(self.args("--max-rows", "3", "--scan-rows", "2"))
        self.assertEqual(set(self.root.iterdir()), {self.source})

    def test_fragment_bound(self):
        with self.assertRaisesRegex(ValueError, "Too many fragments"):
            prepare.prepare(self.args("--max-fragments", "1"))

    def test_empty_snapshot(self):
        with self.assertRaisesRegex(ValueError, "No rows"):
            prepare.prepare(self.args("--symbol-column", "symbol", "--symbol", "MISSING"))

    def test_null_and_nonfinite_features(self):
        for values in ([None, 1.0], [float("nan"), 1.0], [float("inf"), 1.0], [1e300, 1.0]):
            source = self.root / "invalid.parquet"
            pq.write_table(pa.table({"bid": values, "ask": [1.0, 2.0]}), source)
            args = self.args()
            args.source = str(source)
            with self.assertRaises(ValueError):
                prepare.prepare(args)
            self.assertFalse(self.output.exists())

    def test_output_is_immutable(self):
        prepare.prepare(self.args())
        before = self.output.read_bytes()
        with self.assertRaises(FileExistsError):
            prepare.prepare(self.args())
        self.assertEqual(before, self.output.read_bytes())

    def test_explicit_timezones(self):
        with self.assertRaisesRegex(ValueError, "explicit UTC offset"):
            prepare.prepare(self.args("--timestamp-column", "time", "--start", "2026-11-16T00:00:00"))

    def test_nanosecond_window_includes_start_and_excludes_end(self):
        source = self.root / "nanoseconds.parquet"
        epoch = 1_704_067_200_000_000_000
        time = pa.array([epoch + offset for offset in (0, 998, 999, 1000)],
                        type=pa.timestamp("ns", tz="UTC"))
        pq.write_table(pa.table({"time": time, "price": [1.0, 2.0, 3.0, 4.0]}), source)
        args = self.args("--columns", "price", "--timestamp-column", "time",
                         "--start", "2024-01-01T01:00:00.000000999+01:00",
                         "--end", "2024-01-01T00:00:00.000001Z")
        args.source = str(source)
        self.assertEqual(prepare.prepare(args)["rows"], 1)
        self.assertEqual(self.output.read_bytes()[prepare.HEADER.size:], struct.pack("<f", 3.0))

    def test_bounds_finer_than_the_source_unit_are_rejected(self):
        source = self.root / "milliseconds.parquet"
        time = pa.array([1_704_067_200_000], type=pa.timestamp("ms", tz="UTC"))
        pq.write_table(pa.table({"time": time, "price": [1.0]}), source)
        args = self.args("--columns", "price", "--timestamp-column", "time",
                         "--end", "2024-01-01T00:00:00.000001Z")
        args.source = str(source)
        with self.assertRaisesRegex(ValueError, "aligned to the column's ms precision"):
            prepare.prepare(args)
        self.assertFalse(self.output.exists())

    def test_numeric_and_present_features(self):
        with self.assertRaisesRegex(ValueError, "must be numeric"):
            prepare.prepare(self.args("--columns", "symbol,bid"))
        with self.assertRaisesRegex(ValueError, "missing or ambiguous"):
            prepare.prepare(self.args("--columns", "missing,bid"))

    def test_no_embedded_credentials(self):
        args = self.args()
        args.source = "s3://user:password@bucket/data"
        with self.assertRaisesRegex(ValueError, "credential providers"):
            prepare.prepare(args)


if __name__ == "__main__":
    unittest.main()
