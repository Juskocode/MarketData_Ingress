#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR="${1:-build}"
BIN="${BUILD_DIR}/bin/marketdata_ingress"
METRICS="${BUILD_DIR}/marketdata_metrics_smoke.json"
HELP_CAPTURE="${BUILD_DIR}/help.txt"

if [[ ! -x "${BIN}" ]]; then
  echo "Missing executable: ${BIN}" >&2
  exit 1
fi

"${BIN}" --help > "${HELP_CAPTURE}" 2>&1
if ! grep -q "Usage:" "${HELP_CAPTURE}"; then
  echo "Binary does not expose CLI help" >&2
  exit 1
fi

"${BIN}" --iterations 16 --warmup 4 --json-out "${METRICS}" --model mock >/dev/null
if ! grep -q '"p99_us"' "${METRICS}"; then
  echo "Metrics file malformed" >&2
  exit 1
fi

if [[ ! -s "${METRICS}" ]]; then
  echo "Empty metrics file" >&2
  exit 1
fi

echo "[deployability] pass"
echo "metrics: ${METRICS}"
