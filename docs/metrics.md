# Metrics and latency contract

Metrics schema version 2 separates latency domains that should never be compared as if they were the same measurement.

## Timing boundaries

`latency_us.end_to_end` is measured by `std::chrono::steady_clock` around the complete `infer` call. For TensorRT that includes copying the caller's vector into pinned memory, H2D transfer, enqueue, GPU work, D2H transfer, stream synchronization, and copying the result into the caller's vector.

`latency_us.device_compute` is measured by CUDA events immediately before and after `enqueueV3` on the same non-blocking stream. H2D and D2H transfers are outside the event interval. The field is `null` for the mock backend.

## JSON schema

```json
{
  "schema_version": 2,
  "model": "models/tiny.engine",
  "backend": "tensorrt-gpu",
  "requested_backend": "tensorrt",
  "fallback_used": false,
  "cuda_graph": true,
  "samples": 2000,
  "input_size": 512,
  "batch_size": 1,
  "latency_us": {
    "end_to_end": {
      "samples": 2000,
      "mean": 8.4,
      "min": 7.6,
      "p50": 8.1,
      "p90": 9.0,
      "p95": 9.3,
      "p99": 10.2,
      "max": 15.8,
      "stddev": 0.8
    },
    "device_compute": {
      "samples": 2000,
      "mean": 0.71,
      "min": 0.64,
      "p50": 0.70,
      "p90": 0.76,
      "p95": 0.79,
      "p99": 0.88,
      "max": 1.12,
      "stddev": 0.04
    }
  },
  "throughput_ips": 119047.6,
  "throughput_samples_per_second": 119047.6,
  "validation": {
    "kind": "identity",
    "tolerance": 0.000001,
    "compared_elements": 512,
    "mismatches": 0,
    "max_abs_error": 0.0,
    "passed": true
  },
  "slo": {
    "scope": "device_compute",
    "target_us": 1.0,
    "observed_p99_us": 0.88,
    "passed": true
  }
}
```

The flat `mean_us`, `p99_us`, and related fields remain as end-to-end compatibility aliases for simple collectors.

## Interpretation

- Compare p50 to p99 to understand tail amplification.
- Compare end-to-end p99 to device p99 to expose transfer, launch, synchronization, and host-copy overhead.
- Watch standard deviation alongside p99; a low mean with high jitter is not a stable low-latency service.
- `throughput_ips` counts inference calls. `throughput_samples_per_second` multiplies by batch size.
- Treat a run with `fallback_used: true` as diagnostic evidence only, never as TensorRT SLO evidence.
- Compare otherwise identical runs with `cuda_graph` false and true to quantify enqueue-bound overhead.
- Require `validation.passed: true` when benchmarking the generated identity engine. Other models need their own domain-specific golden-output validation.

Use at least 200 warmups and 2,000 measured iterations for a hardware claim. Record GPU model, clocks, power mode, TensorRT/CUDA versions, engine hash, and thermal state next to the JSON artifact.
