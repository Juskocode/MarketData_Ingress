# Troubleshooting Playbook

## 1) TensorRT missing / backend fallback
- Symptom: binary logs show `Using mock` or backend switches to `mock-cpp-cpu`.
- Cause: TensorRT headers/libs not discoverable at configure time.
- Fix: install TensorRT + CUDA runtime and build with `-DMD_WITH_TENSORRT=ON`.

## 2) High tail latency (`p99_us` unstable)
- Symptom: `p99_us` above expected budget while mean looks healthy.
- Likely causes:
  - Thermal throttling
  - Power management toggles
  - Shared GPU usage by other workloads
  - Dynamic input size changes and hidden allocations
- Fixes:
  - Fix input size/batch.
  - Keep warm-up > 20 iterations.
  - Pin model in persistent execution context.
  - Profile with TensorRT inspector.

## 3) CLI exits with code `3`
- Symptom: `--target-us` check fails.
- Cause: tail latency exceeds target.
- Fix:
  - Confirm hardware conditions.
  - Rebuild with explicit batch-size and input-size.
  - Validate model graph includes only required operators.

## 4) `infer` returns false
- Symptom: benchmark aborts with inference error.
- Causes:
  - Wrong model input size.
  - Engine/input binding mismatch.
  - Missing CUDA symbols/runtime.
- Fix: compare input/output tensor shapes and regenerate engine with fixed batch/profile.
