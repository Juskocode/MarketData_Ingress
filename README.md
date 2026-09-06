# MarketData_Ingress

C++ edge-first inference scaffold using TensorRT when available, with a deterministic C++ mock fallback for environments where TensorRT is not installed.

- No Python is used on the hot path.
- Native performance loop is implemented in C++.
- Includes benchmark metrics, troubleshooting, deployability checks, PR tests, and a visual dashboard.

## Repository layout

- `src/` core implementation (benchmark + engine factory).
- `include/marketdata_infer/` public interfaces.
- `tools/visualization/` browser dashboard for metrics.
- `docs/` in-depth metrics and troubleshooting notes.
- `.github/workflows/ci.yml` CI pipeline.

## Build (CPU-only or no TensorRT)

```bash
cmake -S . -B build -DMD_WITH_TENSORRT=OFF -DMD_BUILD_TESTS=ON
cmake --build build -j 2
ctest --test-dir build --output-on-failure
```

## Build with TensorRT (if installed)

```bash
cmake -S . -B build -DMD_WITH_TENSORRT=ON -DMD_BUILD_TESTS=ON
cmake --build build -j 2
```

## Benchmark

```bash
./build/bin/marketdata_ingress --model mock --iterations 200 --json-out results.json
./build/bin/marketdata_ingress --model my_model.engine --backend tensorrt --iterations 1000 --target-us 1.0
```

`--target-us` applies a CI-style threshold on `p99_us`.

## JSON metrics and dashboard

```bash
./build/bin/marketdata_ingress --model mock --json-out metrics.json
```

Open `tools/visualization/index.html` in a browser and load `metrics.json`.

## CI behavior

- Push to `main`: build + unit tests + deployability script.
- Pull request: same baseline plus extended smoke matrix checks.

## Troubleshooting and deployment checks

See:
- `docs/metrics.md`
- `docs/troubleshooting.md`
- `docs/deployability.md`

## Git start (your requested bootstrap)

```bash
echo "# MarketData_Ingress" > README.md
git init
git add README.md
git commit -m "first commit"
git branch -M main
git remote add origin git@github.com:Juskocode/MarketData_Ingress.git
git push -u origin main
```

## Notes on latency target

The codebase is optimized for low-latency inference, but `p99 < 1 µs` depends on platform, model topology, and deployment context. Use `--target-us 1.0` as a target policy and validate on representative hardware.
