# Deployability gates

The deterministic hosted pipeline proves that the native binary can be built, tested, staged, invoked, and governed without requiring a GPU. It does not prove TensorRT performance.

## Local release gate

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

The deployability script validates CLI help, schema-v2 metrics, strict TensorRT failure, explicitly permitted fallback metadata, CMake installation, and execution of the staged binary.

The PR script runs five input volumes and verifies malformed-input rejection, threshold failure, unavailable device-scope rejection, and strict backend policy.

## Pull request matrix

- GCC Release runs all deterministic deployment and PR policy checks.
- GCC Debug runs unit and CLI tests with AddressSanitizer and UndefinedBehaviorSanitizer.
- Clang Release catches compiler-specific warnings and portability regressions.
- Source policy rejects Python embedding in C++ headers and sources.
- JSON benchmark evidence is uploaded from the release job.

## GPU release gate

Provision a self-hosted Linux runner version 2.329.0 or newer with labels `gpu` and `tensorrt`, CUDA Toolkit, TensorRT 10+, CMake, Ninja, and a fixed-shape model available at the workflow input path. Trigger `TensorRT GPU Validation` manually with the model shape and target.

A production release should require all of the following evidence:

1. `MD_REQUIRE_TENSORRT=ON` configuration and build pass.
2. Deterministic CTest suite passes on the GPU runner.
3. `backend` is `tensorrt-gpu` and `fallback_used` is `false`.
4. The selected p99 SLO passes over at least 2,000 iterations after 200 warmups.
5. Model-output accuracy was separately validated against a versioned dataset.
6. The JSON artifact records the exact engine and environment in release metadata.
