# Performance checks

Measure the same model, shape, timing scope, compiler, device and execution mode before comparing builds. GPU-compute latency excludes transfers; E2E includes transfers and synchronization. Mock results describe a CPU arithmetic fixture, not model inference or a market-data feed.

## C++23 buffer-view change

Local check on macOS arm64, Apple Clang 21, Release `-O3`, 100,000 measured requests after 1,000 warmups. Values below are medians of five separate runs against main at `65fd6a8`. CPU p99 is noisy at these short durations; these results do not establish a general speedup.

| FP32 input elements | Main mean (µs) | Candidate mean (µs) | Main p99 (µs) | Candidate p99 (µs) |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 0.046741 | 0.043404 | 0.084 | 0.084 |
| 4,096 | 0.290398 | 0.317038 | 0.417 | 0.459 |
| 65,536 | 5.337961 | 5.094091 | 7.042 | 7.708 |

Reproduce each shape five times with both binaries:

```bash
./build/bin/marketdata_ingress --backend mock --input-size 4096 \
  --warmup 1000 --iterations 100000 --json-out metrics.json
```

The TensorRT change removes two CPU staging copies for callers using the exposed pinned spans and captures the entire fixed-address request into one CUDA graph launch. NVIDIA runtime acceptance is required before claiming a GPU gain. `gpu_buffer_views` exercises changing data, both borrowed and external buffers, direct and graph execution, bad sizes and reload using a native identity engine. Run on the GPU host:

```bash
cmake -S . -B build-gpu -DCMAKE_BUILD_TYPE=Release -DMD_REQUIRE_TENSORRT=ON
cmake --build build-gpu --parallel
ctest --test-dir build-gpu -L gpu --output-on-failure
./scripts/tensorrt_real_tests.sh build-gpu
```
