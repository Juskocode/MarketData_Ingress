# Deployability Tests

These are not micro-benchmarks; they verify that a commit can be packaged and used in simple environments.

## Included checks

1. Configure + build passes.
2. Unit test passes (`marketdata_tests`).
3. CLI help renders and benchmark runs end-to-end.
4. Metrics JSON is produced with expected keys.

## Local run

```bash
cmake -S . -B build -DMD_WITH_TENSORRT=OFF -DMD_BUILD_TESTS=ON
cmake --build build -j 2
ctest --test-dir build --output-on-failure
./scripts/deployability.sh build
```

## PR run (heavier)

`build-and-smoke` remains quick and deterministic with mock backend. Add a real TensorRT pipeline check only in environments with GPU hardware and runtime.
