# Real TensorRT hardware tests

The hosted CI matrix proves portable C++ behavior with the deterministic mock
backend. It does not prove that TensorRT loads an engine, produces correct GPU
output, supports CUDA Graph capture, or meets the device latency objective.

`scripts/tensorrt_real_tests.sh` covers those claims on an NVIDIA machine with
TensorRT 10 or newer. It never enables mock fallback.

## Test coverage

- Builds a fixed-shape identity engine with the repository's native C++ builder.
- Deserializes and executes that engine through the production TensorRT backend.
- Verifies every output element against deterministic input.
- Executes the normal enqueue path and the CUDA Graph path.
- Runs three warmed CUDA Graph benchmarks against the device p99 SLO.
- Compares repeated device p99 measurements for excessive regression.
- Rejects a missing engine, corrupt engine, and incompatible fixed input shape.
- Retains metrics JSON and negative-test logs as workflow artifacts.

## Local NVIDIA execution

```bash
cmake \
  -S . \
  -B build-real \
  -DCMAKE_BUILD_TYPE=Release \
  -DMD_REQUIRE_TENSORRT=ON \
  -DMD_BUILD_TEST_ENGINE=ON

cmake --build build-real --parallel
ctest --test-dir build-real --output-on-failure
./scripts/tensorrt_real_tests.sh build-real
```

The default hardware gate uses 5,000 warmup iterations, 50,000 measured
iterations, three repetitions, and a `1.0 us` device p99 target. Override these
only through explicit environment values such as:

```bash
MD_TEST_TARGET_US=0.9 \
MD_TEST_ITERATIONS=100000 \
./scripts/tensorrt_real_tests.sh build-real evidence
```

Set `MD_TEST_ENGINE` to use a prebuilt compatible identity engine instead of the
native builder.

## Historical replay comparison

Use the same Release binary, CUDA device, FP32 identity engine and historical
FP32 snapshots for both modes. On the NVIDIA host, build with
`MD_REQUIRE_TENSORRT=ON` and prepare the pinned data with Python 3.11 or newer.

```bash
python3 -m venv build-lake-venv
build-lake-venv/bin/python -m pip install -r tools/lake/requirements.txt
build-lake-venv/bin/python tools/lake/replay_public_history.py \
  --ingress build-real/bin/marketdata_ingress --output out/public-history
build-lake-venv/bin/python tools/lake/compare_tensorrt_history.py \
  --history out/public-history --ingress build-real/bin/marketdata_ingress \
  --output out/tensorrt-history
```

The comparison builds a fixed 32-element identity engine once. Both modes
verify every output exactly, use batch size 8, and require the expected
workload fingerprint. Six paired repetitions alternate which mode runs first.
`out/tensorrt-history/report.json` reports median per-run end-to-end and device
p50 and p99, with individual runs available alongside it.

End-to-end time includes pinned-buffer staging, CUDA transfers and stream
synchronization. The device event interval brackets TensorRT enqueue work and
excludes both transfers. CUDA events have approximately 0.5 µs resolution.
The identity model measures the engine and transfer path on real market
features. [NVIDIA CUDA event timing](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html)

Dispatch **TensorRT GPU Validation** with `historical_replay` selected to use
the same comparison on the repository's self-hosted NVIDIA runner. The
`tensorrt-historical-replay` artifact contains the aggregate and per-run
metrics with the driver and device inventory.

## Pull-request execution

The `TensorRT Real Hardware Tests` workflow always supports manual dispatch. To
run it for relevant pull requests, configure repository variable
`ENABLE_TENSORRT_PR_TESTS=true` and attach an NVIDIA runner matching the default
labels `self-hosted`, `linux`, `x64`, and `gpu`.

Use `TENSORRT_RUNNER_LABELS` to provide a JSON label array when the runner uses a
different label set. `TENSORRT_DEVICE_P99_TARGET_US`, `TENSORRT_TEST_WARMUP`, and
`TENSORRT_TEST_ITERATIONS` configure the hardware policy without changing code.

A skipped hardware job means no NVIDIA evidence was produced. It must never be
reported as a TensorRT pass. A successful run is backed by the uploaded engine,
metrics documents, comparison documents, and negative-test logs.
