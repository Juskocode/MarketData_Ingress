#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR="${1:-build}"
BIN="${BUILD_DIR}/bin/marketdata_ingress"
TARGET_US="${MD_TARGET_US:-10000}"
MISSING_ENGINE="${BUILD_DIR}/missing-pr-check.engine"

for INPUT_SIZE in 16 128 512 1024 4096; do
  "${BIN}" \
    --backend mock \
    --model mock \
    --iterations 64 \
    --input-size "${INPUT_SIZE}" \
    --warmup 8 \
    --json-out "${BUILD_DIR}/marketdata_metrics_pr_${INPUT_SIZE}.json" \
    --target-us "${TARGET_US}" >/dev/null
done

if "${BIN}" --backend mock --iterations 0 >/dev/null 2>&1; then
  echo "Invalid zero iterations unexpectedly succeeded" >&2
  exit 1
fi

if "${BIN}" --backend mock --iterations 8 --target-us 0.000001 >/dev/null 2>&1; then
  echo "Impossible p99 threshold unexpectedly succeeded" >&2
  exit 1
fi

if "${BIN}" --backend mock --iterations 8 --target-scope device --target-us 10000 >/dev/null 2>&1; then
  echo "Device-only SLO unexpectedly succeeded on the mock backend" >&2
  exit 1
fi

if "${BIN}" --backend tensorrt --model "${MISSING_ENGINE}" --iterations 1 >/dev/null 2>&1; then
  echo "Strict TensorRT policy unexpectedly succeeded with a missing engine" >&2
  exit 1
fi

grep -q '"schema_version": 2' "${BUILD_DIR}/marketdata_metrics_pr_512.json"
grep -q '"passed": true' "${BUILD_DIR}/marketdata_metrics_pr_512.json"

echo "[pr] stress, policy, and threshold checks passed"
