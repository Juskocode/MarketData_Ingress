#include <arpa/inet.h>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <netinet/in.h>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

namespace {

volatile std::sig_atomic_t stop_requested = 0;
volatile std::sig_atomic_t listening_socket = -1;

void request_stop(int) {
  stop_requested = 1;
  if (listening_socket >= 0) {
    ::close(static_cast<int>(listening_socket));
    listening_socket = -1;
  }
}

std::size_t skip_space(const std::string_view document, std::size_t position) {
  while (position < document.size()) {
    const char value = document[position];
    if (value != ' ' && value != '\n' && value != '\r' && value != '\t') {
      break;
    }
    ++position;
  }
  return position;
}

std::optional<std::size_t> find_value(const std::string_view document,
                                      const std::string_view key) {
  const std::string needle = "\"" + std::string(key) + "\"";
  std::size_t search_from = 0;
  while (search_from < document.size()) {
    const std::size_t key_position = document.find(needle, search_from);
    if (key_position == std::string_view::npos) {
      return std::nullopt;
    }

    std::size_t slash_count = 0;
    for (std::size_t cursor = key_position; cursor > 0 && document[cursor - 1] == '\\';
         --cursor) {
      ++slash_count;
    }
    if (slash_count % 2 != 0) {
      search_from = key_position + needle.size();
      continue;
    }

    std::size_t value_position = skip_space(document, key_position + needle.size());
    if (value_position < document.size() && document[value_position] == ':') {
      return skip_space(document, value_position + 1);
    }
    search_from = key_position + needle.size();
  }
  return std::nullopt;
}

std::size_t required_value(const std::string_view document,
                           const std::string_view key) {
  const auto value = find_value(document, key);
  if (!value) {
    throw std::runtime_error("missing JSON field: " + std::string(key));
  }
  return *value;
}

bool is_null(const std::string_view document, const std::size_t position) {
  return document.substr(skip_space(document, position), 4) == "null";
}

std::size_t string_end(const std::string_view document,
                       const std::size_t position) {
  if (position >= document.size() || document[position] != '"') {
    throw std::runtime_error("expected a JSON string");
  }
  bool escaped = false;
  for (std::size_t cursor = position + 1; cursor < document.size(); ++cursor) {
    if (!escaped && document[cursor] == '"') {
      return cursor;
    }
    if (!escaped && document[cursor] == '\\') {
      escaped = true;
    } else {
      escaped = false;
    }
  }
  throw std::runtime_error("unterminated JSON string");
}

std::string string_at(const std::string_view document,
                      const std::size_t raw_position) {
  const std::size_t position = skip_space(document, raw_position);
  const std::size_t end = string_end(document, position);
  std::string value;
  for (std::size_t cursor = position + 1; cursor < end; ++cursor) {
    if (document[cursor] != '\\') {
      value.push_back(document[cursor]);
      continue;
    }
    if (++cursor >= end) {
      throw std::runtime_error("incomplete JSON escape sequence");
    }
    switch (document[cursor]) {
      case '"': value.push_back('"'); break;
      case '\\': value.push_back('\\'); break;
      case '/': value.push_back('/'); break;
      case 'b': value.push_back('\b'); break;
      case 'f': value.push_back('\f'); break;
      case 'n': value.push_back('\n'); break;
      case 'r': value.push_back('\r'); break;
      case 't': value.push_back('\t'); break;
      case 'u':
        if (cursor + 4 >= end) {
          throw std::runtime_error("incomplete JSON unicode escape");
        }
        value.push_back('?');
        cursor += 4;
        break;
      default: throw std::runtime_error("invalid JSON escape sequence");
    }
  }
  return value;
}

std::string string_value(const std::string_view document,
                         const std::string_view key) {
  return string_at(document, required_value(document, key));
}

std::optional<std::string> optional_string_value(
    const std::string_view document,
    const std::string_view key) {
  const auto position = find_value(document, key);
  if (!position || is_null(document, *position)) {
    return std::nullopt;
  }
  return string_at(document, *position);
}

std::size_t matching_container(const std::string_view document,
                               const std::size_t position) {
  if (position >= document.size() || document[position] != '{') {
    throw std::runtime_error("expected a JSON object");
  }
  std::size_t depth = 1;
  for (std::size_t cursor = position + 1; cursor < document.size(); ++cursor) {
    if (document[cursor] == '"') {
      cursor = string_end(document, cursor);
    } else if (document[cursor] == '{') {
      ++depth;
    } else if (document[cursor] == '}') {
      if (--depth == 0) {
        return cursor;
      }
    }
  }
  throw std::runtime_error("unterminated JSON object");
}

std::string_view object_at(const std::string_view document,
                           const std::size_t raw_position) {
  const std::size_t position = skip_space(document, raw_position);
  const std::size_t end = matching_container(document, position);
  return document.substr(position, end - position + 1);
}

std::string_view object_value(const std::string_view document,
                              const std::string_view key) {
  return object_at(document, required_value(document, key));
}

std::size_t scalar_end(const std::string_view document,
                       const std::size_t position) {
  std::size_t cursor = position;
  while (cursor < document.size()) {
    const char value = document[cursor];
    if (value == ',' || value == '}' || value == ']' || value == ' ' ||
        value == '\n' || value == '\r' || value == '\t') {
      break;
    }
    ++cursor;
  }
  return cursor;
}

double number_at(const std::string_view document,
                 const std::size_t raw_position) {
  const std::size_t position = skip_space(document, raw_position);
  const std::size_t end = scalar_end(document, position);
  const std::string token(document.substr(position, end - position));
  char* parsed_end = nullptr;
  errno = 0;
  const double value = std::strtod(token.c_str(), &parsed_end);
  if (errno == ERANGE || parsed_end != token.c_str() + token.size() ||
      !std::isfinite(value)) {
    throw std::runtime_error("expected a finite JSON number");
  }
  return value;
}

double number_value(const std::string_view document,
                    const std::string_view key) {
  return number_at(document, required_value(document, key));
}

std::optional<double> optional_number_value(const std::string_view document,
                                            const std::string_view key) {
  const auto position = find_value(document, key);
  if (!position || is_null(document, *position)) {
    return std::nullopt;
  }
  return number_at(document, *position);
}

bool bool_at(const std::string_view document, const std::size_t raw_position) {
  const std::size_t position = skip_space(document, raw_position);
  if (document.substr(position, 4) == "true") {
    return true;
  }
  if (document.substr(position, 5) == "false") {
    return false;
  }
  throw std::runtime_error("expected a JSON boolean");
}

bool bool_value(const std::string_view document, const std::string_view key) {
  return bool_at(document, required_value(document, key));
}

std::uint64_t unsigned_value(const std::string_view document,
                             const std::string_view key) {
  const double value = number_value(document, key);
  if (value < 0.0 || value > static_cast<double>(std::numeric_limits<std::uint64_t>::max()) ||
      std::floor(value) != value) {
    throw std::runtime_error("field must be an unsigned integer: " +
                             std::string(key));
  }
  return static_cast<std::uint64_t>(value);
}

struct Distribution {
  std::uint64_t samples;
  double mean;
  double minimum;
  double p50;
  double p90;
  double p95;
  double p99;
  double maximum;
  double stddev;
};

Distribution parse_distribution(const std::string_view object) {
  return Distribution{
      unsigned_value(object, "samples"),
      number_value(object, "mean"),
      number_value(object, "min"),
      number_value(object, "p50"),
      number_value(object, "p90"),
      number_value(object, "p95"),
      number_value(object, "p99"),
      number_value(object, "max"),
      number_value(object, "stddev"),
  };
}

struct Snapshot {
  std::string version;
  std::string revision;
  std::string model;
  std::string backend;
  std::string backend_version;
  std::string device;
  std::uint64_t samples;
  std::uint64_t input_size;
  std::uint64_t batch_size;
  bool fallback_used;
  bool cuda_graph;
  double throughput_ips;
  double throughput_samples_per_second;
  Distribution end_to_end;
  std::optional<Distribution> device_compute;
  std::optional<bool> validation_passed;
  std::optional<std::uint64_t> validation_elements;
  std::string slo_scope;
  std::optional<double> slo_target_us;
  std::optional<double> slo_observed_p99_us;
  std::optional<bool> slo_passed;
};

Snapshot parse_snapshot(const std::string_view document) {
  if (unsigned_value(document, "schema_version") != 2) {
    throw std::runtime_error("unsupported metrics schema version");
  }

  const std::string_view build = object_value(document, "build");
  const std::string_view environment = object_value(document, "environment");
  const std::string_view latency = object_value(document, "latency_us");
  const std::string_view end_to_end = object_value(latency, "end_to_end");
  const std::size_t device_position = required_value(latency, "device_compute");
  std::optional<Distribution> device_distribution;
  if (!is_null(latency, device_position)) {
    device_distribution = parse_distribution(object_at(latency, device_position));
  }

  std::optional<bool> validation_passed;
  std::optional<std::uint64_t> validation_elements;
  const std::size_t validation_position = required_value(document, "validation");
  if (!is_null(document, validation_position)) {
    const std::string_view validation = object_at(document, validation_position);
    validation_passed = bool_value(validation, "passed");
    if (find_value(validation, "checked_elements")) {
      validation_elements = unsigned_value(validation, "checked_elements");
    }
  }

  const std::string_view slo = object_value(document, "slo");
  std::optional<bool> slo_passed;
  const auto slo_pass_position = find_value(slo, "passed");
  if (slo_pass_position && !is_null(slo, *slo_pass_position)) {
    slo_passed = bool_at(slo, *slo_pass_position);
  }

  return Snapshot{
      string_value(build, "version"),
      string_value(build, "revision"),
      string_value(document, "model"),
      string_value(document, "backend"),
      optional_string_value(environment, "backend_version").value_or("unknown"),
      string_value(environment, "device"),
      unsigned_value(document, "samples"),
      unsigned_value(document, "input_size"),
      unsigned_value(document, "batch_size"),
      bool_value(document, "fallback_used"),
      bool_value(document, "cuda_graph"),
      number_value(document, "throughput_ips"),
      number_value(document, "throughput_samples_per_second"),
      parse_distribution(end_to_end),
      device_distribution,
      validation_passed,
      validation_elements,
      string_value(slo, "scope"),
      optional_number_value(slo, "target_us"),
      optional_number_value(slo, "observed_p99_us"),
      slo_passed,
  };
}

std::string escape_label(const std::string_view value) {
  std::string escaped;
  escaped.reserve(value.size());
  for (const char character : value) {
    switch (character) {
      case '\\': escaped += "\\\\"; break;
      case '"': escaped += "\\\""; break;
      case '\n': escaped += "\\n"; break;
      default: escaped.push_back(character); break;
    }
  }
  return escaped;
}

void write_distribution(std::ostringstream& output,
                        const std::string_view scope,
                        const Distribution& distribution) {
  const std::string labels = "scope=\"" + escape_label(scope) + "\"";
  output << "marketdata_inference_latency_us{" << labels << ",quantile=\"0.5\"} "
         << distribution.p50 << '\n'
         << "marketdata_inference_latency_us{" << labels << ",quantile=\"0.9\"} "
         << distribution.p90 << '\n'
         << "marketdata_inference_latency_us{" << labels << ",quantile=\"0.95\"} "
         << distribution.p95 << '\n'
         << "marketdata_inference_latency_us{" << labels << ",quantile=\"0.99\"} "
         << distribution.p99 << '\n'
         << "marketdata_inference_latency_us_sum{" << labels << "} "
         << distribution.mean * static_cast<double>(distribution.samples) << '\n'
         << "marketdata_inference_latency_us_count{" << labels << "} "
         << distribution.samples << '\n'
         << "marketdata_inference_latency_mean_us{" << labels << "} "
         << distribution.mean << '\n'
         << "marketdata_inference_latency_min_us{" << labels << "} "
         << distribution.minimum << '\n'
         << "marketdata_inference_latency_max_us{" << labels << "} "
         << distribution.maximum << '\n'
         << "marketdata_inference_latency_stddev_us{" << labels << "} "
         << distribution.stddev << '\n';
}

std::string render_snapshot(const Snapshot& snapshot,
                            const double file_timestamp_seconds) {
  std::ostringstream output;
  output << std::setprecision(12)
         << "# HELP marketdata_build_info Build, model, backend, and device provenance.\n"
         << "# TYPE marketdata_build_info gauge\n"
         << "marketdata_build_info{version=\"" << escape_label(snapshot.version)
         << "\",revision=\"" << escape_label(snapshot.revision)
         << "\",model=\"" << escape_label(snapshot.model)
         << "\",backend=\"" << escape_label(snapshot.backend)
         << "\",backend_version=\"" << escape_label(snapshot.backend_version)
         << "\",device=\"" << escape_label(snapshot.device) << "\"} 1\n"
         << "# HELP marketdata_metrics_file_timestamp_seconds Modification time of the loaded snapshot.\n"
         << "# TYPE marketdata_metrics_file_timestamp_seconds gauge\n"
         << "marketdata_metrics_file_timestamp_seconds " << file_timestamp_seconds << '\n'
         << "# HELP marketdata_inference_run_samples Measured inference iterations in the snapshot.\n"
         << "# TYPE marketdata_inference_run_samples gauge\n"
         << "marketdata_inference_run_samples " << snapshot.samples << '\n'
         << "# HELP marketdata_inference_input_elements Elements per input sample.\n"
         << "# TYPE marketdata_inference_input_elements gauge\n"
         << "marketdata_inference_input_elements " << snapshot.input_size << '\n'
         << "# HELP marketdata_inference_batch_size Samples per inference batch.\n"
         << "# TYPE marketdata_inference_batch_size gauge\n"
         << "marketdata_inference_batch_size " << snapshot.batch_size << '\n'
         << "# HELP marketdata_inference_fallback_used Whether mock fallback was used.\n"
         << "# TYPE marketdata_inference_fallback_used gauge\n"
         << "marketdata_inference_fallback_used " << (snapshot.fallback_used ? 1 : 0) << '\n'
         << "# HELP marketdata_inference_cuda_graph_enabled Whether CUDA Graph replay was enabled.\n"
         << "# TYPE marketdata_inference_cuda_graph_enabled gauge\n"
         << "marketdata_inference_cuda_graph_enabled " << (snapshot.cuda_graph ? 1 : 0) << '\n'
         << "# HELP marketdata_inference_throughput_inferences_per_second Completed inference calls per second.\n"
         << "# TYPE marketdata_inference_throughput_inferences_per_second gauge\n"
         << "marketdata_inference_throughput_inferences_per_second "
         << snapshot.throughput_ips << '\n'
         << "# HELP marketdata_inference_throughput_samples_per_second Completed samples per second.\n"
         << "# TYPE marketdata_inference_throughput_samples_per_second gauge\n"
         << "marketdata_inference_throughput_samples_per_second "
         << snapshot.throughput_samples_per_second << '\n'
         << "# HELP marketdata_inference_latency_us Inference latency distribution summary.\n"
         << "# TYPE marketdata_inference_latency_us summary\n"
         << "# HELP marketdata_inference_latency_mean_us Mean inference latency.\n"
         << "# TYPE marketdata_inference_latency_mean_us gauge\n"
         << "# HELP marketdata_inference_latency_min_us Minimum inference latency.\n"
         << "# TYPE marketdata_inference_latency_min_us gauge\n"
         << "# HELP marketdata_inference_latency_max_us Maximum inference latency.\n"
         << "# TYPE marketdata_inference_latency_max_us gauge\n"
         << "# HELP marketdata_inference_latency_stddev_us Latency standard deviation.\n"
         << "# TYPE marketdata_inference_latency_stddev_us gauge\n";

  write_distribution(output, "end_to_end", snapshot.end_to_end);
  if (snapshot.device_compute) {
    write_distribution(output, "device_compute", *snapshot.device_compute);
    output << "# HELP marketdata_inference_host_overhead_p99_us Difference between end-to-end and device p99.\n"
           << "# TYPE marketdata_inference_host_overhead_p99_us gauge\n"
           << "marketdata_inference_host_overhead_p99_us "
           << snapshot.end_to_end.p99 - snapshot.device_compute->p99 << '\n';
  }

  output << "# HELP marketdata_inference_validation_enabled Whether output validation was performed.\n"
         << "# TYPE marketdata_inference_validation_enabled gauge\n"
         << "marketdata_inference_validation_enabled "
         << (snapshot.validation_passed ? 1 : 0) << '\n'
         << "# HELP marketdata_inference_validation_pass Whether output validation passed.\n"
         << "# TYPE marketdata_inference_validation_pass gauge\n"
         << "marketdata_inference_validation_pass "
         << (snapshot.validation_passed.value_or(false) ? 1 : 0) << '\n'
         << "# HELP marketdata_inference_validation_elements Number of validated output elements.\n"
         << "# TYPE marketdata_inference_validation_elements gauge\n"
         << "marketdata_inference_validation_elements "
         << snapshot.validation_elements.value_or(0) << '\n'
         << "# HELP marketdata_inference_slo_configured Whether an SLO was configured for this run.\n"
         << "# TYPE marketdata_inference_slo_configured gauge\n"
         << "marketdata_inference_slo_configured " << (snapshot.slo_target_us ? 1 : 0) << '\n'
         << "# HELP marketdata_inference_slo_target_us Configured p99 latency objective.\n"
         << "# TYPE marketdata_inference_slo_target_us gauge\n"
         << "marketdata_inference_slo_target_us{scope=\"" << escape_label(snapshot.slo_scope)
         << "\"} " << snapshot.slo_target_us.value_or(0.0) << '\n'
         << "# HELP marketdata_inference_slo_observed_p99_us Observed p99 selected by the SLO.\n"
         << "# TYPE marketdata_inference_slo_observed_p99_us gauge\n"
         << "marketdata_inference_slo_observed_p99_us{scope=\""
         << escape_label(snapshot.slo_scope) << "\"} "
         << snapshot.slo_observed_p99_us.value_or(snapshot.end_to_end.p99) << '\n'
         << "# HELP marketdata_inference_slo_pass Whether the configured p99 objective passed.\n"
         << "# TYPE marketdata_inference_slo_pass gauge\n"
         << "marketdata_inference_slo_pass "
         << (snapshot.slo_passed.value_or(false) ? 1 : 0) << '\n';
  return output.str();
}

std::string read_file(const std::string& path) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream) {
    throw std::runtime_error("cannot open metrics file: " + path);
  }
  std::ostringstream contents;
  contents << stream.rdbuf();
  if (!stream.good() && !stream.eof()) {
    throw std::runtime_error("cannot read metrics file: " + path);
  }
  return contents.str();
}

double file_timestamp(const std::string& path) {
  const auto file_time = std::filesystem::last_write_time(path);
  const auto system_time = std::chrono::time_point_cast<std::chrono::system_clock::duration>(
      file_time - std::filesystem::file_time_type::clock::now() +
      std::chrono::system_clock::now());
  return std::chrono::duration<double>(system_time.time_since_epoch()).count();
}

double unix_time_now() {
  return std::chrono::duration<double>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

struct ExporterState {
  std::uint64_t scrapes = 0;
  std::uint64_t failures = 0;
  double last_success_timestamp = 0.0;
};

std::string render_exporter_state(const ExporterState& state,
                                  const bool load_success) {
  std::ostringstream output;
  output << std::setprecision(12)
         << "# HELP marketdata_exporter_scrapes_total Metrics scrape attempts.\n"
         << "# TYPE marketdata_exporter_scrapes_total counter\n"
         << "marketdata_exporter_scrapes_total " << state.scrapes << '\n'
         << "# HELP marketdata_exporter_load_failures_total Snapshot load failures.\n"
         << "# TYPE marketdata_exporter_load_failures_total counter\n"
         << "marketdata_exporter_load_failures_total " << state.failures << '\n'
         << "# HELP marketdata_exporter_last_load_success Whether the latest snapshot load succeeded.\n"
         << "# TYPE marketdata_exporter_last_load_success gauge\n"
         << "marketdata_exporter_last_load_success " << (load_success ? 1 : 0) << '\n'
         << "# HELP marketdata_exporter_last_success_timestamp_seconds Latest successful snapshot load.\n"
         << "# TYPE marketdata_exporter_last_success_timestamp_seconds gauge\n"
         << "marketdata_exporter_last_success_timestamp_seconds "
         << state.last_success_timestamp << '\n';
  return output.str();
}

std::string load_for_scrape(const std::string& path, ExporterState& state) {
  ++state.scrapes;
  try {
    const Snapshot snapshot = parse_snapshot(read_file(path));
    state.last_success_timestamp = unix_time_now();
    return render_exporter_state(state, true) +
           render_snapshot(snapshot, file_timestamp(path));
  } catch (const std::exception&) {
    ++state.failures;
    return render_exporter_state(state, false);
  }
}

void send_all(const int socket, const std::string_view response) {
  std::size_t sent = 0;
  while (sent < response.size()) {
    const ssize_t result = ::send(socket, response.data() + sent,
                                  response.size() - sent, 0);
    if (result <= 0) {
      return;
    }
    sent += static_cast<std::size_t>(result);
  }
}

void send_response(const int socket,
                   const int status,
                   const std::string_view reason,
                   const std::string_view content_type,
                   const std::string& body) {
  std::ostringstream response;
  response << "HTTP/1.1 " << status << ' ' << reason << "\r\n"
           << "Content-Type: " << content_type << "\r\n"
           << "Content-Length: " << body.size() << "\r\n"
           << "Cache-Control: no-store\r\n"
           << "Connection: close\r\n\r\n"
           << body;
  send_all(socket, response.str());
}

bool snapshot_is_ready(const std::string& path, std::string& error) {
  try {
    static_cast<void>(parse_snapshot(read_file(path)));
    return true;
  } catch (const std::exception& failure) {
    error = failure.what();
    return false;
  }
}

void handle_connection(const int client,
                       const std::string& metrics_path,
                       ExporterState& state) {
  char buffer[8192]{};
  const ssize_t received = ::recv(client, buffer, sizeof(buffer) - 1, 0);
  if (received <= 0) {
    return;
  }
  const std::string_view request(buffer, static_cast<std::size_t>(received));
  const std::size_t line_end = request.find("\r\n");
  const std::string_view request_line = request.substr(0, line_end);
  const std::size_t first_space = request_line.find(' ');
  const std::size_t second_space = first_space == std::string_view::npos
                                       ? std::string_view::npos
                                       : request_line.find(' ', first_space + 1);
  if (first_space == std::string_view::npos || second_space == std::string_view::npos) {
    send_response(client, 400, "Bad Request", "text/plain; charset=utf-8",
                  "bad request\n");
    return;
  }
  if (request_line.substr(0, first_space) != "GET") {
    send_response(client, 405, "Method Not Allowed", "text/plain; charset=utf-8",
                  "method not allowed\n");
    return;
  }

  std::string_view path = request_line.substr(first_space + 1,
                                              second_space - first_space - 1);
  const std::size_t query = path.find('?');
  if (query != std::string_view::npos) {
    path = path.substr(0, query);
  }

  if (path == "/healthz") {
    send_response(client, 200, "OK", "text/plain; charset=utf-8", "ok\n");
  } else if (path == "/readyz") {
    std::string error;
    if (snapshot_is_ready(metrics_path, error)) {
      send_response(client, 200, "OK", "text/plain; charset=utf-8", "ready\n");
    } else {
      send_response(client, 503, "Service Unavailable", "text/plain; charset=utf-8",
                    "not ready: " + error + "\n");
    }
  } else if (path == "/metrics") {
    send_response(client, 200, "OK",
                  "text/plain; version=0.0.4; charset=utf-8",
                  load_for_scrape(metrics_path, state));
  } else {
    send_response(client, 404, "Not Found", "text/plain; charset=utf-8",
                  "not found\n");
  }
}

int create_listener(const std::string& address, const std::uint16_t port) {
  const int socket = ::socket(AF_INET, SOCK_STREAM, 0);
  if (socket < 0) {
    throw std::runtime_error("cannot create listening socket: " +
                             std::string(std::strerror(errno)));
  }

  int reuse = 1;
  if (::setsockopt(socket, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse)) != 0) {
    ::close(socket);
    throw std::runtime_error("cannot configure listening socket");
  }

  sockaddr_in endpoint{};
  endpoint.sin_family = AF_INET;
  endpoint.sin_port = htons(port);
  const std::string numeric_address = address == "localhost" ? "127.0.0.1" : address;
  if (::inet_pton(AF_INET, numeric_address.c_str(), &endpoint.sin_addr) != 1) {
    ::close(socket);
    throw std::runtime_error("listen address must be an IPv4 address or localhost");
  }
  if (::bind(socket, reinterpret_cast<const sockaddr*>(&endpoint), sizeof(endpoint)) != 0) {
    const std::string error = std::strerror(errno);
    ::close(socket);
    throw std::runtime_error("cannot bind exporter socket: " + error);
  }
  if (::listen(socket, 16) != 0) {
    const std::string error = std::strerror(errno);
    ::close(socket);
    throw std::runtime_error("cannot listen on exporter socket: " + error);
  }
  return socket;
}

std::uint16_t parse_port(const std::string& text) {
  char* end = nullptr;
  errno = 0;
  const unsigned long value = std::strtoul(text.c_str(), &end, 10);
  if (errno == ERANGE || end != text.c_str() + text.size() || value == 0 ||
      value > 65535) {
    throw std::runtime_error("--port requires an integer from 1 to 65535");
  }
  return static_cast<std::uint16_t>(value);
}

void print_usage(const char* executable) {
  std::cout
      << "Usage: " << executable << " --metrics-file <metrics.json> [options]\n\n"
      << "Options:\n"
      << "  --listen-address <IPv4>  HTTP bind address (default: 0.0.0.0)\n"
      << "  --port <N>               HTTP port (default: 9108)\n"
      << "  --once                   Print one Prometheus snapshot and exit\n"
      << "  --self-test              Run parser and exposition unit tests\n"
      << "  --help                   Show this help\n";
}

int self_test() {
  const std::string document = R"json({
    "schema_version": 2,
    "build": {"version": "0.2.0", "revision": "abc123"},
    "model": "identity.engine",
    "backend": "tensorrt-gpu",
    "requested_backend": "tensorrt",
    "fallback_used": false,
    "cuda_graph": true,
    "environment": {
      "backend_version": "10.0",
      "device": "GPU 0",
      "compute_capability": "8.9",
      "cuda_runtime": "12.0",
      "cuda_driver": "12.0",
      "device_memory_bytes": 1000
    },
    "samples": 100,
    "input_size": 32,
    "batch_size": 1,
    "latency_us": {
      "end_to_end": {"samples":100,"mean":2.0,"min":1.0,"p50":1.5,"p90":2.2,"p95":2.4,"p99":2.8,"max":3.0,"stddev":0.2},
      "device_compute": {"samples":100,"mean":0.5,"min":0.3,"p50":0.4,"p90":0.6,"p95":0.7,"p99":0.8,"max":0.9,"stddev":0.1}
    },
    "mean_us": 2.0,
    "min_us": 1.0,
    "p50_us": 1.5,
    "p90_us": 2.2,
    "p95_us": 2.4,
    "p99_us": 2.8,
    "max_us": 3.0,
    "stddev_us": 0.2,
    "throughput_ips": 500000,
    "throughput_samples_per_second": 500000,
    "validation": {"passed": true, "checked_elements": 32, "max_abs_error": 0.0},
    "slo": {"scope":"device_compute","target_us":1.0,"observed_p99_us":0.8,"passed":true}
  })json";

  const Snapshot snapshot = parse_snapshot(document);
  const std::string exposition = render_snapshot(snapshot, 1234.0);
  const auto require = [&](const std::string_view expected) {
    if (exposition.find(expected) == std::string::npos) {
      throw std::runtime_error("self-test missing exposition: " +
                               std::string(expected));
    }
  };
  require("marketdata_build_info{");
  require("backend=\"tensorrt-gpu\"");
  require("marketdata_inference_latency_us{scope=\"device_compute\",quantile=\"0.99\"} 0.8");
  require("marketdata_inference_host_overhead_p99_us 2");
  require("marketdata_inference_validation_pass 1");
  require("marketdata_inference_slo_pass 1");

  bool rejected = false;
  try {
    static_cast<void>(parse_snapshot("{}"));
  } catch (const std::runtime_error&) {
    rejected = true;
  }
  if (!rejected) {
    throw std::runtime_error("self-test accepted malformed metrics");
  }

  std::cout << "metrics exporter self-test: pass\n";
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    std::string metrics_file;
    std::string listen_address = "0.0.0.0";
    std::uint16_t port = 9108;
    bool once = false;

    const auto value_after = [&](const int index, const std::string_view option) {
      if (index + 1 >= argc) {
        throw std::runtime_error(std::string(option) + " requires a value");
      }
      return std::string(argv[index + 1]);
    };

    for (int index = 1; index < argc; ++index) {
      const std::string argument = argv[index];
      if (argument == "--help") {
        print_usage(argv[0]);
        return 0;
      }
      if (argument == "--self-test") {
        return self_test();
      }
      if (argument == "--metrics-file") {
        metrics_file = value_after(index, argument);
        ++index;
      } else if (argument == "--listen-address") {
        listen_address = value_after(index, argument);
        ++index;
      } else if (argument == "--port") {
        port = parse_port(value_after(index, argument));
        ++index;
      } else if (argument == "--once") {
        once = true;
      } else {
        throw std::runtime_error("unknown option: " + argument);
      }
    }

    if (metrics_file.empty()) {
      throw std::runtime_error("--metrics-file is required");
    }
    if (once) {
      ExporterState state;
      state.scrapes = 1;
      state.last_success_timestamp = unix_time_now();
      const Snapshot snapshot = parse_snapshot(read_file(metrics_file));
      std::cout << render_exporter_state(state, true)
                << render_snapshot(snapshot, file_timestamp(metrics_file));
      return 0;
    }

    std::signal(SIGINT, request_stop);
    std::signal(SIGTERM, request_stop);
    std::signal(SIGPIPE, SIG_IGN);
    const int server = create_listener(listen_address, port);
    listening_socket = server;
    std::cerr << "marketdata metrics exporter listening on " << listen_address
              << ':' << port << " using " << metrics_file << '\n';

    ExporterState state;
    while (!stop_requested) {
      sockaddr_in peer{};
      socklen_t peer_size = sizeof(peer);
      const int client = ::accept(server, reinterpret_cast<sockaddr*>(&peer), &peer_size);
      if (client < 0) {
        if (stop_requested) {
          break;
        }
        if (errno == EINTR) {
          continue;
        }
        throw std::runtime_error("exporter accept failed: " +
                                 std::string(std::strerror(errno)));
      }
      handle_connection(client, metrics_file, state);
      ::close(client);
    }
    if (listening_socket >= 0) {
      ::close(server);
      listening_socket = -1;
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "metrics exporter error: " << error.what() << '\n';
    return 2;
  }
}
