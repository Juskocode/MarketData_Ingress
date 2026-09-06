#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 || $3 != "--" ]]; then
  echo "usage: $0 <ingress-binary> <metrics-file> -- <benchmark arguments...>" >&2
  exit 2
fi

ingress_binary=$1
metrics_file=$2
shift 3

if [[ ! -x "$ingress_binary" ]]; then
  echo "ingress binary is not executable: $ingress_binary" >&2
  exit 2
fi

interval_seconds=${MD_MONITOR_INTERVAL_SECONDS:-10}
metrics_directory=$(dirname "$metrics_file")
mkdir -p "$metrics_directory"

while true; do
  temporary_file="${metrics_file}.tmp.$$"
  if "$ingress_binary" "$@" --json-out "$temporary_file"; then
    mv "$temporary_file" "$metrics_file"
  else
    rm -f "$temporary_file"
    echo "benchmark failed; preserving last valid metrics snapshot" >&2
  fi

  if [[ ${MD_MONITOR_RUN_ONCE:-0} == 1 ]]; then
    break
  fi
  sleep "$interval_seconds"
done
