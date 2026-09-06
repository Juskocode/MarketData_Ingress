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
