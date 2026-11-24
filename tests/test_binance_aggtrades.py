"""Offline fixtures for the Binance Vision daily spot aggregate trade adapter."""
from datetime import date, datetime, timezone
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "lake"))
from binance_aggtrades import SCHEMA, convert


class BinanceAggregateTradesTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.archive = self.root / "source.zip"
        self.output = self.root / "trades.parquet"
        self.symbol = "BTCUSDT"

    def write_zip(self, day, contents, member=None):
        if member is None:
            member = f"{self.symbol}-aggTrades-{day.isoformat()}.csv"
        with zipfile.ZipFile(self.archive, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr(member, contents)

    def test_millisecond_archive_projects_taker_side_and_quote_quantity(self):
        day = date(2024, 12, 31)
        self.write_zip(day, "11,125.25,2.5,14,15,1735603200007,True,True\n"
                            "12,100,0.25,16,16,1735603200219,False,True\n")
        with patch("binance_aggtrades.BATCH_ROWS", 1):
            self.assertEqual(convert(self.archive, self.output, symbol=self.symbol, day=day, max_rows=2), 2)
        self.assertEqual(pq.read_metadata(self.output).num_row_groups, 2)
        result = pq.read_table(self.output)
        self.assertEqual(result.schema, SCHEMA)
        self.assertEqual(result["event_time"].to_pylist(), [
            datetime(2024, 12, 31, 0, 0, 0, 7000, tzinfo=timezone.utc),
            datetime(2024, 12, 31, 0, 0, 0, 219000, tzinfo=timezone.utc),
        ])
        self.assertEqual(result["symbol"].to_pylist(), [self.symbol, self.symbol])
        self.assertEqual(result["sequence"].to_pylist(), [11, 12])
        self.assertEqual(result["quote_quantity"].to_pylist(), [313.125, 25])
        self.assertEqual(result["taker_side"].to_pylist(), [-1, 1])

    def test_microsecond_archive_accepts_header_and_preserves_sub_millisecond_time(self):
        day = date(2025, 1, 1)
        self.write_zip(day, "\ufeffagg_trade_id,price,quantity,first_trade_id,last_trade_id,transact_time,"
                            "is_buyer_maker,is_best_match\r\n"
                            "1,0.2,50,1,1,1735689600010866,False,True\r\n")
        self.assertEqual(convert(self.archive, self.output, symbol=self.symbol, day=day, max_rows=1), 1)
        result = pq.read_table(self.output)
        self.assertEqual(result["event_time"].to_pylist(), [
            datetime(2025, 1, 1, 0, 0, 0, 10866, tzinfo=timezone.utc),
        ])

    def test_bound_failure_does_not_install_a_partial_file(self):
        day = date(2025, 1, 1)
        self.write_zip(day, "11,25,2,14,15,1735689600010866,True,True\n"
                            "12,20,1,16,16,1735689600010867,False,True\n")
        with patch("binance_aggtrades.BATCH_ROWS", 1), self.assertRaisesRegex(ValueError, "row bound"):
            convert(self.archive, self.output, symbol=self.symbol, day=day, max_rows=1)
        self.assertEqual(set(self.root.iterdir()), {self.archive})

    def test_mismatched_units_and_malformed_fields_are_rejected(self):
        day = date(2025, 1, 1)
        for record, error in [
            ("11,25,2,14,15,1735689600010,True,True\n", "incorrect unit"),
            ("11,25,2,14,15,1735776000000000,True,True\n", "outside archive day"),
            ("11,NaN,2,14,15,1735689600010866,True,True\n", "finite and positive"),
            ("11,25,2,14,15,1735689600010866,none,True\n", "boolean"),
            ("11,25,2,15,14,1735689600010866,True,True\n", "Invalid trade IDs"),
        ]:
            with self.subTest(error=error):
                self.write_zip(day, record)
                with self.assertRaisesRegex(ValueError, error):
                    convert(self.archive, self.output, symbol=self.symbol, day=day, max_rows=1)
                self.assertFalse(self.output.exists())

    def test_wrong_member_and_large_expanded_size_are_rejected(self):
        day = date(2025, 1, 1)
        record = "1,0.2,50,1,1,1735689600010866,False,True\n"
        self.write_zip(day, record, "../wrong.csv")
        with self.assertRaisesRegex(ValueError, "Expected one CSV"):
            convert(self.archive, self.output, symbol=self.symbol, day=day, max_rows=1)
        self.write_zip(day, record)
        with patch("binance_aggtrades.MAX_CSV_BYTES", 1), self.assertRaisesRegex(ValueError, "expanded size"):
            convert(self.archive, self.output, symbol=self.symbol, day=day, max_rows=1)
        self.assertFalse(self.output.exists())

    def test_output_is_immutable(self):
        day = date(2025, 1, 1)
        self.output.write_text("existing output")
        with self.assertRaises(FileExistsError):
            convert(self.archive, self.output, symbol=self.symbol, day=day, max_rows=1)
        self.assertEqual(self.output.read_text(), "existing output")


if __name__ == "__main__":
    unittest.main()
