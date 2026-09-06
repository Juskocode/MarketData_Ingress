#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 <ingress-binary> <exporter-binary> <work-directory>" >&2
  exit 2
fi

ingress_binary=$1
exporter_binary=$2
work_directory=$3
metrics="$work_directory/metrics.json"
invalid_metrics="$work_directory/invalid.json"

mkdir -p "$work_directory"

"$ingress_binary" \
  --backend mock \
  --warmup 8 \
  --iterations 64 \
  --target-scope e2e \
  --target-us 10 \
  --json-out "$metrics" \
  >/dev/null

exposition=$("$exporter_binary" --metrics-file "$metrics" --once)

for expected in \
  'marketdata_exporter_last_load_success 1' \
  'marketdata_build_info{' \
  'backend="mock-cpp-cpu"' \
  'marketdata_inference_latency_us{scope="end_to_end",quantile="0.99"}' \
  'marketdata_inference_fallback_used 0' \
  'marketdata_inference_slo_pass 1'; do
  case "$exposition" in
    *"$expected"*) ;;
    *)
      echo "missing Prometheus metric: $expected" >&2
      exit 1
      ;;
  esac
done

case "$exposition" in
  *'scope="device_compute"'*)
    echo "mock metrics incorrectly exposed device timing" >&2
    exit 1
    ;;
esac

printf '{invalid json\n' >"$invalid_metrics"
set +e
"$exporter_binary" --metrics-file "$invalid_metrics" --once >/dev/null 2>&1
invalid_status=$?
set -e
if [[ $invalid_status -ne 2 ]]; then
  echo "invalid metrics should return 2, got $invalid_status" >&2
  exit 1
fi

echo "monitoring integration: pass"
