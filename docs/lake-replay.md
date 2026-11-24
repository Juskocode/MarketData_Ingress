# Data-lake replay

Prepare numeric model features from a bounded Parquet snapshot, then replay them through the same native CPU/TensorRT API. Python/Arrow is an offline preparation step. The native executable has no Arrow, Python or cloud SDK dependency.

## Reproduce with public data

The [README quickstart](../README.md#replay-real-lake-features) uses Python 3.11+ and downloads two fixed inputs. The runner caches verified originals under `out/public-history-cache`. Use `--offline` to replay from that cache or `--dataset binance-trades` and `--dataset crypto-lake-quotes` to select an individual workload. Use a new output directory for each run.

| Source | Workload | Snapshot rows | Native requests × 8 | Replay fingerprint |
| --- | --- | ---: | ---: | --- |
| [Binance Vision spot aggregate trades](https://github.com/binance/binance-public-data) | BTCUSDT, 2025-02-01 00:00–00:05 UTC | 3,457 | 432 | `26346c499d7b43f9` |
| [Crypto Lake Level 1 quotes](https://crypto-lake.com/data/) | BTC-USDT, 2022-10-01 22:00–22:06 UTC | 3,064 | 383 | `4c468d19878148a1` |

The Binance converter reads the full daily ZIP in bounded batches. It stages 859,311 trades as typed Parquet, normalizes timestamps to UTC, and prepares price, base quantity, quote quantity and taker side from the five-minute window. A buyer-maker flag produces a sell-side value of `-1`. A buyer taker produces `+1`. The fixture leaves one final trade outside the last complete batch. [Publisher checksum](https://data.binance.vision/data/spot/daily/aggTrades/BTCUSDT/BTCUSDT-aggTrades-2025-02-01.zip.CHECKSUM)

The Crypto Lake object is used directly as Parquet. Replay orders the four bid and ask price and size features by the object's integer nanosecond receipt clock. The object is stored under the provider's October 2 partition. Its receipt times are on October 1 UTC. [Parquet object](https://s3.eu-west-1.amazonaws.com/sample.crypto.lake/level_1/exchange=BINANCE/symbol=BTC-USDT/dt=2022-10-02/converted_a849a1d5a1154281958cfaf2e9f3d844.snappy.parquet)

Both fixtures use a pinned source SHA-256 and a pinned FP32 snapshot SHA-256. The runner verifies the snapshot header, computes the fingerprint from the bytes of the full batches, and checks the native metrics against the expected shape, row count and fingerprint. The acceptance run uses the CPU mock backend. For TensorRT latency or output validation, run the resulting snapshots with a compatible model on GPU hardware.

Review the provider terms before reusing the data. The examples keep archives, derived Parquet and FP32 rows in local ignored directories. The manual CI job uploads aggregate reports and timing metrics. [Binance Vision dataset terms](https://github.com/binance/binance-public-data/blob/master/TERMS_AND_CONDITIONS.md) · [Crypto Lake terms](https://crypto-lake.com/terms-of-service/)

## Prepare a snapshot

```bash
python3 -m venv build-lake-venv
build-lake-venv/bin/python -m pip install -r tools/lake/requirements.txt

build-lake-venv/bin/python tools/lake/prepare.py /data/market-features \
  --columns bid,ask,bid_size,ask_size \
  --timestamp-column event_time \
  --start 2026-11-16T14:30:00Z --end 2026-11-16T14:35:00Z \
  --symbol-column symbol --symbol AAPL --symbol MSFT \
  --order-by event_time,sequence \
  --max-rows 100000 --output out/opening.mdlake
```

The output directory is created if needed. `--columns` order is the model's feature order, with one sample per row. This tool does not derive trading features, normalize values, construct time windows, resample or apply an exchange calendar. Prepare those features upstream and use a held-out time window when assessing a model. FP32 conversion may lose precision; reject or scale high-resolution timestamps and large identifiers upstream rather than feeding them as numeric model features.

The source can be a local Parquet file, a Hive-partitioned directory, or an Arrow filesystem URI such as `s3://bucket/features/date=2026-11-16`. Filesystem authentication uses Arrow's normal credential providers. Do not put credentials in a URI. S3/GCS/Azure access needs the corresponding Arrow filesystem support and environment credentials; only local Parquet was exercised in the integration suite, not a remote account.

Time filters use `[start, end)` and require timezone-aware Arrow timestamps and explicit UTC offsets in the bounds. Bounds must align exactly to the timestamp column's precision, including nanoseconds when the column supports them. Symbols are exact matches. The scanner projects only features/order columns and pushes the filters down. Fragments are processed in lexical path order, rows in physical order, with scanner prefetch limited. `--max-fragments` defaults to 1,000. Exceeding either bound fails the export.

An optional global `--order-by` sorts only the bounded selected rows. Ties preserve the original path/row order. Keep sources immutable during preparation; mutable lake tables should be exported from a versioned snapshot first. This is Parquet file/directory support, not an Iceberg/Delta transaction-log reader. Supply a committed snapshot's Parquet inputs rather than scanning a table directory containing obsolete files.

Nulls, nonnumeric features, non-finite FP32 conversions and empty selections fail. Exports never overwrite an existing snapshot or manifest. Without `--order-by`, preparation converts and writes one Arrow batch at a time, with `--scan-rows` controlling the batch size. Global sorting retains the bounded selection and requires memory proportional to `max_rows × projected_columns` plus sorting overhead. Neither row limit is a strict byte cap.

The `.json` sidecar records feature order, bounds, symbols, sort keys, library versions and a SHA-256 of the completed binary. Keep it with the snapshot.

## Native replay

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DMD_WITH_TENSORRT=OFF
cmake --build build --parallel

./build/bin/marketdata_ingress --backend mock \
  --input-file out/opening.mdlake --input-size 4 --batch 32 \
  --iterations 1000 --warmup 100 --json-out out/replay-metrics.json
```

For TensorRT, use `-DMD_REQUIRE_TENSORRT=ON`, the real model and `--backend tensorrt`. Add `--cuda-graph` to compare graph execution against ordinary enqueue. `input_size × batch` must match the fixed engine shape. `--verify-identity` is only for an identity model, not the arithmetic mock or a trained model.

`std::views::iota`/`transform` exposes contiguous row batches as borrowed `std::span<const float>`. Linux/macOS map the file read-only; other platforms use an aligned buffered fallback. No per-batch tensor allocation or row/column conversion occurs during replay. TensorRT still copies the mapped input into pinned staging before H2D, then lends its output buffer directly to the runner.

`--input-row-offset` selects the first row. Measured requests visit successive complete batches exactly once. The runner requires at least `iterations × batch` rows after the offset; it never pads the partial tail or silently wraps measured data. Warmup cycles through this selected window and measured replay then starts at the offset.

The preflight validates all selected FP32 values and computes their FNV-1a-64 fingerprint, touching those pages before warmup. These are warm-cache inference measurements. They exclude lake/network I/O, Parquet decoding, mapping, feature conversion, fingerprinting, input-view selection, validation and statistics; E2E includes backend staging/transfers/synchronization. This is inference replay, not an ingress-throughput or cold-storage benchmark. CPU mock numbers are not a model accuracy test.

Metrics keep schema v2 and add `input.source`, storage mode, content fingerprint, dataset row count and selected offset. The regression comparator rejects different replay content/offsets or synthetic-versus-replay comparisons unless `--allow-context-change` is explicit. FNV is a workload identifier, not a cryptographic integrity guarantee; the sidecar's SHA-256 serves artifact checking. Keep the mapped file immutable for the lifetime of its views; truncating it concurrently can fault the process.

## Tests

```bash
cmake -S . -B build-lake -DMD_WITH_TENSORRT=OFF \
  -DMD_BUILD_LAKE_TESTS=ON \
  -DPython3_EXECUTABLE="$PWD/build-lake-venv/bin/python"
cmake --build build-lake --parallel
ctest --test-dir build-lake --output-on-failure
```

The native suite checks zero-copy batch addresses/order, multiple real inputs through an identity engine, offsets, bounds, unsupported headers, truncation, overflow and NaN rejection. The Arrow suite prepares filtered/sorted multi-file Parquet, checks exact FP32 values and digest, runs the native CLI and checks workload-comparison rejection. It also covers empty selections, nulls, non-finite conversion, explicit timezones and immutable outputs. CI runs both; default native builds do not need Python.

## Binary format v1

All integers and FP32 data are little-endian. The 64-byte header is followed by row-major IEEE-754 FP32 features, without padding.

| Offset | Type | Meaning |
| ---: | --- | --- |
| 0 | 8 bytes | `MDLAKE01` |
| 8 | uint64 | sample rows, positive |
| 16 | uint64 | features per row, positive |
| 24 | uint64 | payload bytes, exactly rows × features × 4 |
| 32 | 32 bytes | reserved, must be zero |
| 64 | float32[] | row-major payload |

The native reader verifies exact file size, version, flags and overflow before exposing views. Big-endian hosts are rejected.
