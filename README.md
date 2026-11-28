# MarketData_Ingress

A native C++23 edge-inference benchmark and deployment scaffold for fixed-shape TensorRT models. It measures the complete request path separately from GPU execution, enforces latency policy in CI, and never invokes Python on the inference hot path.

## What is implemented

- TensorRT 10+ runtime using the named-tensor API and `enqueueV3`.
- Dedicated non-blocking CUDA stream, persistent device buffers, pinned host staging buffers, and CUDA-event device timing.
- Borrowed `std::span` inference views with a compatible pointer-and-capacity API. TensorRT exposes its pinned staging buffers so native callers can fill/read them directly. The measured loop performs no vector growth or backend buffer allocation.
- End-to-end p50/p90/p95/p99/max/jitter plus GPU-compute distributions.
- Build revision, TensorRT/CUDA versions, GPU model, compute capability, and device-memory provenance in every metrics artifact.
- Strict TensorRT selection. A forced TensorRT run cannot silently become a mock benchmark.
- Deterministic C++ mock backend for development and hosted CI without a GPU.
- Metrics dashboard, sanitizer builds, CLI policy tests, staged-install checks, and a manual self-hosted GPU SLO workflow.

Requires a compiler and standard library with C++23 support (GCC 13+, Clang 18+ or recent Apple Clang).

## Build and test without a GPU

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DMD_WITH_TENSORRT=OFF \
  -DMD_BUILD_TESTS=ON
cmake --build build --parallel
ctest --test-dir build --output-on-failure
./scripts/deployability.sh build
./scripts/ci_pr_checks.sh build
```

## Production TensorRT build

Prerequisites are a CUDA Toolkit, TensorRT 10 or newer, `nvinfer`, and `nvonnxparser`.

```bash
cmake -S . -B build-trt \
  -DCMAKE_BUILD_TYPE=Release \
  -DMD_REQUIRE_TENSORRT=ON \
  -DMD_BUILD_TESTS=ON
cmake --build build-trt --parallel
```

`MD_REQUIRE_TENSORRT=ON` is the production-safe switch: configuration fails instead of quietly producing a mock-only binary.

The TensorRT build also produces a native engine generator. It creates a minimal fixed-shape identity network directly through the TensorRT C++ API, which is useful for installation checks and baseline overhead measurements:

```bash
./build-trt/bin/marketdata_build_engine \
  --output models/tiny.engine \
  --input-size 512
```

## Benchmark a fixed-shape engine

```bash
./build-trt/bin/marketdata_ingress \
  --backend tensorrt \
  --model models/tiny.engine \
  --input-size 512 \
  --batch 1 \
  --warmup 200 \
  --iterations 2000 \
  --verify-identity \
  --cuda-graph \
  --target-scope device \
  --target-us 1.0 \
  --json-out metrics.json
```

Use `--target-scope device` to gate TensorRT kernel execution measured with CUDA events. Use `--target-scope e2e` to gate the full C++ call, including host copies, CUDA transfers, synchronization, and result materialization.

`--verify-identity` is intended for the native identity engine. It compares every output element on every measured request with the deterministic input in C++ (including non-finite checks outside the timing interval), writes the mismatch count and maximum absolute error to JSON, and exits with code 5 on failure. Do not use that flag for a non-identity model.

`--cuda-graph` primes the fixed-shape context, captures H2D transfer, compute-timing events, `enqueueV3` and D2H transfer, and replays the entire captured request graph for each inference. Capture failure exits with code 6. Compare runs with and without this option; it primarily helps when host enqueue overhead dominates small kernels.

The runner intentionally supports one fixed-shape FP32 input and one fixed-shape FP32 output. This keeps the latency contract unambiguous. Build separate serialized engines for different shapes.

## Replay real lake features

Run the historical acceptance test after building. It exercises a Binance spot trade archive and a Crypto Lake Level 1 Parquet sample. Both fixtures are downloadable without API credentials.

```bash
python3 -m venv build-lake-venv
build-lake-venv/bin/python -m pip install -r tools/lake/requirements.txt
build-lake-venv/bin/python tools/lake/replay_public_history.py \
  --ingress build/bin/marketdata_ingress --output out/public-history
```

The test verifies source and snapshot SHA-256, checks native batch counts and fingerprints, and writes `out/public-history/report.json` with links to both metrics files. For cached runs, pass `--offline` and a new `--output` directory. See [data-lake replay](docs/lake-replay.md) for fixture provenance and how to replay your own Parquet data.

## Backend policy

- `--model mock` with `--backend auto` selects the mock backend.
- A non-mock model with `--backend auto` selects TensorRT.
- `--backend tensorrt` exits nonzero if TensorRT is unavailable or model loading fails.
- `--allow-mock-fallback` is the only way to permit a fallback; JSON records `fallback_used: true`.

## Metrics dashboard

Run a benchmark with `--json-out`, open `tools/visualization/index.html`, and drop the JSON file onto the page. The dashboard supports schema v2 and older flat metrics files.

## CI/CD

Every push and pull request runs GCC Release, GCC Debug with ASan/UBSan, Clang Release, CTest, source-policy checks, and staged deployability checks. Pull requests additionally run input-size stress, invalid-CLI, strict-backend, and threshold-failure tests.

Manually dispatch **MarketData Edge CI** to run the two public fixtures and download the `public-history-metrics` artifact. The standard Parquet integration tests use local fixtures.

The manual `TensorRT GPU Validation` workflow targets a self-hosted runner labeled `gpu` and `tensorrt`. It requires TensorRT at configure time, generates the native identity engine if its requested model path is absent, and uploads the measured JSON as evidence.

`cmake --build build --target package` creates a versioned `.tar.gz` containing the installed binary, documentation, and dashboard. A pushed `vMAJOR.MINOR.PATCH` tag triggers the release workflow only when it exactly matches the CMake project version; the workflow reruns deployability checks and publishes the package plus `SHA256SUMS` as immutable GitHub Release assets. GPU workflow packages are separate artifacts because TensorRT binaries depend on their deployment runtime.

## Documentation

- `docs/metrics.md`: schema, timing boundaries, SLO interpretation.
- `docs/troubleshooting.md`: build, runtime, accuracy, and latency diagnostics.
- `docs/deployability.md`: local and CI release gates.

## Reality of the 1 microsecond target

Sub-microsecond GPU compute is possible only for extremely small, fused workloads on suitable hardware. It is not a portable promise. PCIe transfers, launch overhead, and synchronization commonly make end-to-end latency much higher. This repository reports both boundaries so a device-only result cannot be presented as end-to-end performance.
