# Benchmark regression gates

`marketdata_metrics_compare` compares two schema-v2 benchmark documents without
putting a scripting language on the inference path. It is a dependency-free C++
utility and returns a non-zero status when latency regresses, inputs are invalid,
or benchmark contexts are not comparable.

## Recommended workflow

Generate the baseline and candidate with the same engine, GPU, shape, batch,
CUDA Graph mode, warmup, iteration count, power mode, and clock policy. Store the
full JSON artifacts, not only the selected percentile.

```bash
marketdata_ingress \
  --backend tensorrt \
  --model model.engine \
  --input-size 32 \
  --batch 1 \
  --warmup 10000 \
  --iterations 100000 \
  --cuda-graph \
  --json-out baseline.json

marketdata_ingress \
  --backend tensorrt \
  --model model.engine \
  --input-size 32 \
  --batch 1 \
  --warmup 10000 \
  --iterations 100000 \
  --cuda-graph \
  --json-out candidate.json

marketdata_metrics_compare \
  --baseline baseline.json \
  --candidate candidate.json \
  --scope device \
  --metric p99 \
  --max-regression-percent 5 \
  --absolute-tolerance-us 0.02
```

The accepted upper bound is:

```text
baseline * (1 + max_regression_percent / 100) + absolute_tolerance_us
```

The utility writes one JSON object to standard output. Exit status `0` means the
candidate is within budget, `1` means a measured regression, `2` means invalid
input or command-line usage, and `4` means benchmark context mismatch.

## Comparability guard

By default, both documents must have identical values for:

- backend
- device
- input size
- batch size
- CUDA Graph mode

This catches common false comparisons, such as treating mock CPU results as a
TensorRT baseline or comparing a batch-1 engine with a batch-8 engine. Runtime,
driver, TensorRT, and build provenance remain in the source metrics documents for
review. Use `--allow-context-change` only for an intentional cross-environment
experiment; the output records that override.

## Choosing a gate

Use `--scope device` to isolate GPU execution and `--scope e2e` to protect the
application-visible latency, including transfers and launch overhead. A deployment
gate should normally require both an absolute SLO from `marketdata_ingress` and a
relative regression budget from this comparator.

Do not use a short local sample as a release baseline. Collect enough iterations
to stabilize the tail, retain the raw run metadata, and compare several runs when
thermal state or clock behavior is not controlled.
