#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR="${1:-build}"
BIN="${BUILD_DIR}/bin/marketdata_ingress"
METRICS="${BUILD_DIR}/marketdata_metrics_smoke.json"
FALLBACK_METRICS="${BUILD_DIR}/marketdata_metrics_fallback.json"
HELP_CAPTURE="${BUILD_DIR}/help.txt"
STAGE_DIR="${BUILD_DIR}/stage"
MISSING_ENGINE="${BUILD_DIR}/missing-for-deployability.engine"

if [[ ! -x "${BIN}" ]]; then
  echo "Missing executable: ${BIN}" >&2
  exit 1
fi

"${BIN}" --help > "${HELP_CAPTURE}" 2>&1
grep -q "Usage:" "${HELP_CAPTURE}"

"${BIN}" \
  --backend mock \
  --model mock \
  --iterations 16 \
  --warmup 4 \
  --json-out "${METRICS}" \
  --target-us 10000 >/dev/null

grep -q '"schema_version": 2' "${METRICS}"
grep -q '"end_to_end"' "${METRICS}"
grep -q '"device_compute": null' "${METRICS}"
grep -q '"fallback_used": false' "${METRICS}"

if "${BIN}" --backend tensorrt --model "${MISSING_ENGINE}" --iterations 1 >/dev/null 2>&1; then
  echo "Forced TensorRT unexpectedly fell back without permission" >&2
  exit 1
fi

"${BIN}" \
  --backend tensorrt \
  --model "${MISSING_ENGINE}" \
  --allow-mock-fallback \
  --iterations 4 \
  --warmup 1 \
  --json-out "${FALLBACK_METRICS}" >/dev/null
grep -q '"fallback_used": true' "${FALLBACK_METRICS}"
grep -q '"backend": "mock-cpp-cpu"' "${FALLBACK_METRICS}"

cmake --install "${BUILD_DIR}" --prefix "${STAGE_DIR}"
"${STAGE_DIR}/bin/marketdata_ingress" --help >/dev/null

echo "[deployability] pass"
echo "metrics: ${METRICS}"
echo "staged binary: ${STAGE_DIR}/bin/marketdata_ingress"
