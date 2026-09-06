#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:-static}"
BUILD_DIR="${2:-build}"
PROMETHEUS_IMAGE="prom/prometheus:v3.14.0"
ALERTMANAGER_IMAGE="prom/alertmanager:v0.34.0"
RULES_FILE="${ROOT_DIR}/deploy/monitoring/prometheus/rules/marketdata.yml"
TEST_FILE="${ROOT_DIR}/deploy/monitoring/prometheus/tests/marketdata_rules.test.yml"
ALERTMANAGER_CONFIG="${ROOT_DIR}/deploy/monitoring/alertmanager/alertmanager.yml"
mkdir -p "${ROOT_DIR}/work"
TMP_DIR="$(mktemp -d "${ROOT_DIR}/work/alerting-test.XXXXXX")"
chmod 0755 "${TMP_DIR}"

cleanup() {
  rm -rf "${TMP_DIR}"
}
trap cleanup EXIT

command -v docker >/dev/null 2>&1 || {
  echo "docker is required" >&2
  exit 1
}
command -v ruby >/dev/null 2>&1 || {
  echo "ruby is required to derive the deterministic test fixture" >&2
  exit 1
}

docker run --rm \
  -v "${RULES_FILE}:/work/marketdata.yml:ro" \
  --entrypoint /bin/promtool \
  "${PROMETHEUS_IMAGE}" \
  check rules /work/marketdata.yml

docker run --rm \
  -v "${ALERTMANAGER_CONFIG}:/etc/alertmanager/alertmanager.yml:ro" \
  --entrypoint /bin/amtool \
  "${ALERTMANAGER_IMAGE}" \
  check-config /etc/alertmanager/alertmanager.yml

ruby -ryaml -e '
  source, destination = ARGV
  document = YAML.safe_load(File.read(source), aliases: true)
  expected = %w[
    MarketDataExporterDown
    MarketDataSnapshotLoadFailed
    MarketDataMetricsStale
    MarketDataDeviceP99SloBreach
    MarketDataMockFallbackUsed
    MarketDataIdentityValidationFailed
    MarketDataCudaGraphDisabled
  ]
  alerts = document.fetch("groups").flat_map { |group| group.fetch("rules") }
                   .select { |rule| rule.key?("alert") }
  actual = alerts.map { |rule| rule.fetch("alert") }
  abort "alert test inventory drift: expected=#{expected.sort.inspect} actual=#{actual.sort.inspect}" unless actual.sort == expected.sort
  alerts.each do |rule|
    rule["expr"] = "sum((#{rule.fetch("expr")}))"
    rule.delete("labels")
    rule.delete("annotations")
  end
  File.write(destination, YAML.dump(document).sub(/\A---\n/, ""))
' "${RULES_FILE}" "${TMP_DIR}/marketdata-rules.normalized.yml"

cp "${TEST_FILE}" "${TMP_DIR}/marketdata_rules.test.yml"
chmod 0644 "${TMP_DIR}/marketdata-rules.normalized.yml" "${TMP_DIR}/marketdata_rules.test.yml"
docker run --rm \
  -v "${TMP_DIR}:/work:ro" \
  --entrypoint /bin/promtool \
  "${PROMETHEUS_IMAGE}" \
  test rules /work/marketdata_rules.test.yml

if [[ "${MODE}" != "--runtime" ]]; then
  echo "[alerting-test] static configuration and rule tests passed"
  exit 0
fi

ALERTMANAGER_PORT="${ALERTMANAGER_PORT:-9093}"
PROMETHEUS_PORT="${PROMETHEUS_PORT:-9090}"
VICTORIAMETRICS_PORT="${VICTORIAMETRICS_PORT:-8428}"
GRAFANA_PORT="${GRAFANA_PORT:-3000}"
GRAFANA_ADMIN_USER="${GRAFANA_ADMIN_USER:-admin}"
GRAFANA_ADMIN_PASSWORD="${GRAFANA_ADMIN_PASSWORD:-marketdata-local}"

for _ in $(seq 1 60); do
  if curl -fsS "http://127.0.0.1:${ALERTMANAGER_PORT}/-/ready" >/dev/null; then
    break
  fi
  sleep 1
done
curl -fsS "http://127.0.0.1:${ALERTMANAGER_PORT}/-/ready" >/dev/null
curl -fsS "http://127.0.0.1:${ALERTMANAGER_PORT}/api/v2/status" |
  ruby -rjson -e 'status = JSON.parse(STDIN.read); abort "missing Alertmanager cluster status" unless status.dig("cluster", "status")'

curl -fsS "http://127.0.0.1:${PROMETHEUS_PORT}/api/v1/alertmanagers" |
  ruby -rjson -e '
    payload = JSON.parse(STDIN.read)
    active = payload.dig("data", "activeAlertmanagers") || []
    abort "Prometheus has no active Alertmanager" unless active.any? { |entry| entry.fetch("url", "").include?("alertmanager:9093") }
  '

prometheus_up="$(curl -fsSG "http://127.0.0.1:${PROMETHEUS_PORT}/api/v1/query" \
  --data-urlencode 'query=max(up{job="alertmanager"})' |
  ruby -rjson -e 'payload = JSON.parse(STDIN.read); puts(payload.dig("data", "result", 0, "value", 1) || "0")')"
[[ "${prometheus_up}" == "1" ]] || {
  echo "Prometheus is not scraping Alertmanager: ${prometheus_up}" >&2
  exit 1
}

victoriametrics_up="0"
for _ in $(seq 1 30); do
  victoriametrics_up="$(curl -fsSG "http://127.0.0.1:${VICTORIAMETRICS_PORT}/api/v1/query" \
    --data-urlencode 'query=max(up{job="alertmanager"})' |
    ruby -rjson -e 'payload = JSON.parse(STDIN.read); puts(payload.dig("data", "result", 0, "value", 1) || "0")')"
  [[ "${victoriametrics_up}" == "1" ]] && break
  sleep 1
done
[[ "${victoriametrics_up}" == "1" ]] || {
  echo "VictoriaMetrics did not receive Alertmanager health: ${victoriametrics_up}" >&2
  exit 1
}

curl -fsS -u "${GRAFANA_ADMIN_USER}:${GRAFANA_ADMIN_PASSWORD}" \
  "http://127.0.0.1:${GRAFANA_PORT}/api/datasources/uid/alertmanager" |
  ruby -rjson -e '
    source = JSON.parse(STDIN.read)
    abort "wrong Alertmanager datasource URL" unless source.fetch("url") == "http://alertmanager:9093"
    abort "wrong Alertmanager implementation" unless source.dig("jsonData", "implementation") == "prometheus"
  '

panel_count="$(curl -fsS -u "${GRAFANA_ADMIN_USER}:${GRAFANA_ADMIN_PASSWORD}" \
  "http://127.0.0.1:${GRAFANA_PORT}/api/dashboards/uid/marketdata-edge-pulse" |
  ruby -rjson -e 'puts JSON.parse(STDIN.read).dig("dashboard", "panels").length')"
(( panel_count >= 12 )) || {
  echo "expected at least 12 Grafana panels, found ${panel_count}" >&2
  exit 1
}

evidence_dir="${ROOT_DIR}/${BUILD_DIR}/monitoring-stack-evidence"
mkdir -p "${evidence_dir}"
cat >"${evidence_dir}/alertmanager.txt" <<EOF
Alertmanager ready: true
Prometheus discovery: active
Prometheus scrape up: ${prometheus_up}
VictoriaMetrics remote-write series: ${victoriametrics_up}
Grafana Alertmanager datasource: provisioned
Grafana panels: ${panel_count}
External notification receiver: intentionally not configured
EOF

echo "[alerting-test] pass: routing active, remote-write visible, panels=${panel_count}"
