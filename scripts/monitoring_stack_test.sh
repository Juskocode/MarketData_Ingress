#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <build-directory>" >&2
  exit 2
fi

build_directory=$1
ingress="$build_directory/bin/marketdata_ingress"
exporter="$build_directory/bin/marketdata_metrics_exporter"
evidence="$build_directory/monitoring-stack-evidence"
metrics="$evidence/latest.json"
compose_file=deploy/monitoring/compose.yaml
grafana_user=${GRAFANA_ADMIN_USER:-admin}
grafana_password=${GRAFANA_ADMIN_PASSWORD:-marketdata-local}

if [[ ! -x "$ingress" || ! -x "$exporter" ]]; then
  echo "monitoring test binaries are missing under $build_directory/bin" >&2
  exit 2
fi
if ! command -v ruby >/dev/null 2>&1; then
  echo "ruby is required for monitoring API assertions" >&2
  exit 2
fi

if docker compose version >/dev/null 2>&1; then
  compose=(docker compose -f "$compose_file")
elif command -v docker-compose >/dev/null 2>&1; then
  compose=(docker-compose -f "$compose_file")
else
  echo "Docker Compose is required" >&2
  exit 2
fi

mkdir -p "$evidence"

exporter_pid=""
cleanup() {
  "${compose[@]}" logs --no-color >"$evidence/compose.log" 2>&1 || true
  if [[ -n "$exporter_pid" ]]; then
    kill "$exporter_pid" >/dev/null 2>&1 || true
    wait "$exporter_pid" >/dev/null 2>&1 || true
  fi
  "${compose[@]}" down --remove-orphans >/dev/null 2>&1 || true
}
trap cleanup EXIT

"$ingress" \
  --backend mock \
  --warmup 100 \
  --iterations 1000 \
  --target-scope e2e \
  --target-us 10 \
  --json-out "$metrics" \
  >"$evidence/benchmark.log"

"$exporter" \
  --metrics-file "$metrics" \
  --listen-address 0.0.0.0 \
  --port 9108 \
  >"$evidence/exporter.log" 2>&1 &
exporter_pid=$!

"${compose[@]}" config -q
"${compose[@]}" up -d

for attempt in $(seq 1 60); do
  if curl -fsS http://127.0.0.1:9108/readyz >/dev/null 2>&1 &&
     curl -fsS http://127.0.0.1:9090/-/ready >/dev/null 2>&1 &&
     curl -fsS http://127.0.0.1:8428/health >/dev/null 2>&1 &&
     curl -fsS http://127.0.0.1:3000/api/health >/dev/null 2>&1; then
    break
  fi
  if [[ $attempt -eq 60 ]]; then
    echo "monitoring services did not become ready" >&2
    exit 1
  fi
  sleep 2
done

"${compose[@]}" exec -T prometheus \
  promtool check config /etc/prometheus/prometheus.yml \
  >"$evidence/promtool.log"
curl -fsS -X POST http://127.0.0.1:9090/-/reload >/dev/null
curl -fsS http://127.0.0.1:9108/metrics >"$evidence/exporter.prom"

for attempt in $(seq 1 30); do
  curl -fsSG \
    --data-urlencode 'query=up{job="marketdata_edge"}' \
    http://127.0.0.1:9090/api/v1/query \
    >"$evidence/prometheus-up.json"
  scrape_value=$(ruby -rjson -e '
    result = JSON.parse(File.read(ARGV.fetch(0))).dig("data", "result") || []
    puts(result.empty? ? "" : result.first.dig("value", 1))
  ' "$evidence/prometheus-up.json")
  [[ $scrape_value == 1 ]] && break
  if [[ $attempt -eq 30 ]]; then
    echo "Prometheus did not scrape the C++ exporter" >&2
    exit 1
  fi
  sleep 2
done

for attempt in $(seq 1 30); do
  curl -fsSG \
    --data-urlencode 'query=marketdata_build_info' \
    http://127.0.0.1:8428/api/v1/query \
    >"$evidence/victoriametrics-query.json"
  vm_series=$(ruby -rjson -e '
    puts((JSON.parse(File.read(ARGV.fetch(0))).dig("data", "result") || []).length)
  ' "$evidence/victoriametrics-query.json")
  [[ $vm_series -ge 1 ]] && break
  if [[ $attempt -eq 30 ]]; then
    echo "VictoriaMetrics did not receive remote-written series" >&2
    exit 1
  fi
  sleep 2
done

curl -fsS \
  -u "$grafana_user:$grafana_password" \
  http://127.0.0.1:3000/api/datasources/uid/victoriametrics \
  >"$evidence/grafana-datasource.json"
datasource_url=$(ruby -rjson -e '
  puts JSON.parse(File.read(ARGV.fetch(0))).fetch("url")
' "$evidence/grafana-datasource.json")
if [[ $datasource_url != http://victoriametrics:8428 ]]; then
  echo "Grafana VictoriaMetrics datasource URL is incorrect: $datasource_url" >&2
  exit 1
fi

curl -fsS \
  -u "$grafana_user:$grafana_password" \
  http://127.0.0.1:3000/api/dashboards/uid/marketdata-edge-pulse \
  >"$evidence/grafana-dashboard.json"
dashboard_panels=$(ruby -rjson -e '
  dashboard = JSON.parse(File.read(ARGV.fetch(0))).fetch("dashboard")
  abort "wrong dashboard title" unless dashboard.fetch("title") == "MarketData Edge Pulse"
  puts dashboard.fetch("panels").length
' "$evidence/grafana-dashboard.json")
if [[ $dashboard_panels -lt 10 ]]; then
  echo "Grafana dashboard is incomplete: $dashboard_panels panels" >&2
  exit 1
fi

curl -fsS http://127.0.0.1:9090/api/v1/alerts >"$evidence/prometheus-alerts.json"
firing_alerts=$(ruby -rjson -e '
  alerts = JSON.parse(File.read(ARGV.fetch(0))).dig("data", "alerts") || []
  puts alerts.count { |alert| alert["state"] == "firing" }
' "$evidence/prometheus-alerts.json")
if [[ $firing_alerts -ne 0 ]]; then
  echo "unexpected firing alerts in healthy mock monitoring test: $firing_alerts" >&2
  exit 1
fi

echo "[monitoring-stack] pass"
echo "Prometheus exporter up: $scrape_value"
echo "VictoriaMetrics series: $vm_series"
echo "Grafana panels: $dashboard_panels"
echo "Firing alerts: $firing_alerts"
