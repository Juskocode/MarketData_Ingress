# Metrics and Benchmarks

The benchmark exports a compact JSON schema that is easy to scrape in CI.

```json
{
  "model": "model.engine",
  "backend": "tensorrt-gpu",
  "samples": 200,
  "input_size": 512,
  "batch_size": 1,
  "mean_us": 12.34,
  "min_us": 3.1,
  "p50_us": 10.9,
  "p90_us": 13.4,
  "p95_us": 15.1,
  "p99_us": 17.8,
  "max_us": 25.0,
  "stddev_us": 4.2,
  "throughput_ips": 5400
}
```

## Contract checks in CI

- `p99_us` must be present in every JSON file.
- For PR checks, we enforce synthetic tests that do not regress CLI behavior.
- `--target-us` (optional) can be used locally and in CI to enforce an SLO.

## Visual diagnostics

Open `tools/visualization/index.html`, load any `--json-out` file and use the cards/bars to review latency shape.

## Recommended interpretation

- `p50_us`: central tendency for inference latency.
- `p95_us`/`p99_us`: tail latency; tune kernels and input pipeline if these drift.
- `throughput_ips`: stable if close to expected hardware throughput.
- `stddev_us`: jitter; high values often indicate contention or non-deterministic batching.
