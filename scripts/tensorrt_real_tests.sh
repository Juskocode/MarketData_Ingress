#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 <build-directory> [evidence-directory]" >&2
  exit 2
fi

build_directory=$1
evidence_directory=${2:-"$build_directory/tensorrt-real-evidence"}
ingress="$build_directory/bin/marketdata_ingress"
builder="$build_directory/bin/marketdata_build_engine"
comparator="$build_directory/bin/marketdata_metrics_compare"

input_size=${MD_TEST_INPUT_SIZE:-32}
batch_size=${MD_TEST_BATCH_SIZE:-1}
warmup=${MD_TEST_WARMUP:-5000}
iterations=${MD_TEST_ITERATIONS:-50000}
repeat_count=${MD_TEST_REPEAT_COUNT:-3}
target_us=${MD_TEST_TARGET_US:-1.0}
tolerance=${MD_TEST_IDENTITY_TOLERANCE:-0.000001}
max_regression_percent=${MD_TEST_MAX_REGRESSION_PERCENT:-50}
absolute_tolerance_us=${MD_TEST_ABSOLUTE_TOLERANCE_US:-0.10}

for executable in "$ingress" "$comparator"; do
  if [[ ! -x "$executable" ]]; then
    echo "required executable is missing: $executable" >&2
    exit 2
  fi
done

mkdir -p "$evidence_directory"

expect_failure() {
  local name=$1
  shift
  if "$@" >"$evidence_directory/$name.stdout.log" \
           2>"$evidence_directory/$name.stderr.log"; then
    echo "expected failure succeeded unexpectedly: $name" >&2
    exit 1
  fi
}

if [[ -n "${MD_TEST_ENGINE:-}" ]]; then
  engine=$MD_TEST_ENGINE
else
  if [[ ! -x "$builder" ]]; then
    echo "native TensorRT engine builder is missing: $builder" >&2
    exit 2
  fi
  engine="$evidence_directory/identity-${input_size}x${batch_size}.engine"
  "$builder" \
    --output "$engine" \
    --input-size "$input_size" \
    --batch "$batch_size"
fi

if [[ ! -s "$engine" ]]; then
  echo "TensorRT test engine is missing or empty: $engine" >&2
  exit 1
fi

common_arguments=(
  --backend tensorrt
  --model "$engine"
  --input-size "$input_size"
  --batch "$batch_size"
  --warmup "$warmup"
  --iterations "$iterations"
  --verify-identity
  --tolerance "$tolerance"
)

"$ingress" \
  "${common_arguments[@]}" \
  --json-out "$evidence_directory/standard.json"

for run in $(seq 1 "$repeat_count"); do
  "$ingress" \
    "${common_arguments[@]}" \
    --cuda-graph \
    --target-scope device \
    --target-us "$target_us" \
    --json-out "$evidence_directory/cuda-graph-$run.json"
done

for run in $(seq 2 "$repeat_count"); do
  "$comparator" \
    --baseline "$evidence_directory/cuda-graph-1.json" \
    --candidate "$evidence_directory/cuda-graph-$run.json" \
    --scope device \
    --metric p99 \
    --max-regression-percent "$max_regression_percent" \
    --absolute-tolerance-us "$absolute_tolerance_us" \
    >"$evidence_directory/regression-$run.json"
done

expect_failure missing-engine \
  "$ingress" \
  --backend tensorrt \
  --model "$evidence_directory/does-not-exist.engine" \
  --input-size "$input_size" \
  --batch "$batch_size" \
  --warmup 0 \
  --iterations 1

printf 'not-a-tensorrt-engine\n' >"$evidence_directory/corrupt.engine"
expect_failure corrupt-engine \
  "$ingress" \
  --backend tensorrt \
  --model "$evidence_directory/corrupt.engine" \
  --input-size "$input_size" \
  --batch "$batch_size" \
  --warmup 0 \
  --iterations 1

expect_failure wrong-fixed-shape \
  "$ingress" \
  --backend tensorrt \
  --model "$engine" \
  --input-size "$((input_size + 1))" \
  --batch "$batch_size" \
  --warmup 0 \
  --iterations 1

echo "[tensorrt-real] pass"
echo "engine: $engine"
echo "evidence: $evidence_directory"
echo "device p99 SLO: ${target_us} us"
