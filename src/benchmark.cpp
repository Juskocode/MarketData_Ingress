#include "marketdata_infer/engine.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace mdedge {

namespace {

double percentile(const std::vector<double>& sorted, double p) {
  if (sorted.empty()) {
    return 0.0;
  }
  const double position = std::clamp(p, 0.0, 1.0) * static_cast<double>(sorted.size() - 1);
  const size_t lower = static_cast<size_t>(std::floor(position));
  const size_t upper = static_cast<size_t>(std::ceil(position));
  const double weight = position - static_cast<double>(lower);
  return sorted[lower] + ((sorted[upper] - sorted[lower]) * weight);
}

LatencyDistribution summarize(std::vector<double> values) {
  LatencyDistribution result;
  if (values.empty()) {
    return result;
  }

  std::sort(values.begin(), values.end());
  result.samples = values.size();
  result.mean_us = std::accumulate(values.begin(), values.end(), 0.0) /
      static_cast<double>(values.size());
  result.min_us = values.front();
  result.p50_us = percentile(values, 0.50);
  result.p90_us = percentile(values, 0.90);
  result.p95_us = percentile(values, 0.95);
  result.p99_us = percentile(values, 0.99);
  result.max_us = values.back();

  double variance = 0.0;
  for (const double value : values) {
    const double delta = value - result.mean_us;
    variance += delta * delta;
  }
  result.stddev_us = std::sqrt(variance / static_cast<double>(values.size()));
  return result;
}

std::string escape_json(const std::string& value) {
  std::ostringstream out;
  for (const char raw : value) {
    const auto c = static_cast<unsigned char>(raw);
    switch (c) {
      case '\"': out << "\\\""; break;
      case '\\': out << "\\\\"; break;
      case '\b': out << "\\b"; break;
      case '\f': out << "\\f"; break;
      case '\n': out << "\\n"; break;
      case '\r': out << "\\r"; break;
      case '\t': out << "\\t"; break;
      default:
        if (c < 0x20U) {
          out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
              << static_cast<unsigned int>(c) << std::dec << std::setfill(' ');
        } else {
          out << static_cast<char>(c);
        }
    }
  }
  return out.str();
}

void write_distribution(std::ostringstream& out, const LatencyDistribution& distribution, int indent) {
  const std::string pad(static_cast<size_t>(indent), ' ');
  out << "{\n"
      << pad << "  \"samples\": " << distribution.samples << ",\n"
      << pad << "  \"mean\": " << distribution.mean_us << ",\n"
      << pad << "  \"min\": " << distribution.min_us << ",\n"
      << pad << "  \"p50\": " << distribution.p50_us << ",\n"
      << pad << "  \"p90\": " << distribution.p90_us << ",\n"
      << pad << "  \"p95\": " << distribution.p95_us << ",\n"
      << pad << "  \"p99\": " << distribution.p99_us << ",\n"
      << pad << "  \"max\": " << distribution.max_us << ",\n"
      << pad << "  \"stddev\": " << distribution.stddev_us << '\n'
      << pad << '}';
}

const char* scope_name(LatencyScope scope) {
  return scope == LatencyScope::DeviceCompute ? "device_compute" : "end_to_end";
}

} // namespace

InferenceStats run_benchmark(InferenceEngine& engine, const RunOptions& opts) {
  if (opts.input_size == 0 || opts.batch_size == 0 || opts.iterations == 0) {
    throw std::runtime_error("Input size, batch size, and iterations must be greater than zero");
  }
  if (opts.input_size > std::numeric_limits<size_t>::max() / opts.batch_size) {
    throw std::runtime_error("Input size multiplied by batch size overflows size_t");
  }

  const size_t total_input = opts.input_size * opts.batch_size;
  const size_t engine_capacity = engine.input_elements_per_batch();
  if (engine_capacity != 0 && total_input != engine_capacity) {
    throw std::runtime_error("Input element count must exactly match the fixed TensorRT input shape");
  }

  std::vector<float> input(total_input);
  for (size_t i = 0; i < input.size(); ++i) {
    input[i] = static_cast<float>(i % 97U) / 97.0F;
  }
  std::vector<float> output;

  for (size_t i = 0; i < opts.warmup; ++i) {
    if (!engine.infer(input, output)) {
      throw std::runtime_error("Warmup inference failed");
    }
  }

  std::vector<double> end_to_end_latencies;
  std::vector<double> device_latencies;
  end_to_end_latencies.reserve(opts.iterations);
  device_latencies.reserve(opts.iterations);

  for (size_t i = 0; i < opts.iterations; ++i) {
    const auto start = std::chrono::steady_clock::now();
    if (!engine.infer(input, output)) {
      throw std::runtime_error("Inference failed during benchmark");
    }
    const auto end = std::chrono::steady_clock::now();
    end_to_end_latencies.push_back(
        std::chrono::duration<double, std::micro>(end - start).count());
    if (const auto device_latency = engine.last_device_latency_us()) {
      device_latencies.push_back(*device_latency);
    }
  }

  InferenceStats stats;
  stats.end_to_end = summarize(std::move(end_to_end_latencies));
  if (!device_latencies.empty()) {
    stats.device_compute = summarize(std::move(device_latencies));
  }
  if (opts.verify_identity) {
    OutputValidation validation;
    validation.compared_elements = std::min(input.size(), output.size());
    validation.mismatches = input.size() > output.size()
        ? input.size() - output.size()
        : output.size() - input.size();
    for (size_t i = 0; i < validation.compared_elements; ++i) {
      const double error = std::abs(static_cast<double>(output[i]) - static_cast<double>(input[i]));
      validation.max_abs_error = std::max(validation.max_abs_error, error);
      if (error > opts.validation_tolerance) {
        ++validation.mismatches;
      }
    }
    validation.passed = validation.mismatches == 0;
    stats.validation = validation;
  }

  const double total_seconds =
      (stats.end_to_end.mean_us * static_cast<double>(stats.end_to_end.samples)) / 1e6;
  if (total_seconds > 0.0) {
    stats.throughput_inferences_per_second =
        static_cast<double>(stats.end_to_end.samples) / total_seconds;
    stats.throughput_samples_per_second =
        (static_cast<double>(stats.end_to_end.samples) * static_cast<double>(opts.batch_size)) /
        total_seconds;
  }
  stats.input_elements_per_batch = total_input;
  stats.backend = engine.backend_name();
  return stats;
}

std::string format_metrics_json(const RunOptions& options, const InferenceStats& stats) {
  const LatencyDistribution* selected = &stats.end_to_end;
  bool selected_available = true;
  if (options.target_scope == LatencyScope::DeviceCompute) {
    selected_available = stats.device_compute.has_value();
    if (selected_available) {
      selected = &*stats.device_compute;
    }
  }

  std::ostringstream out;
  out << std::fixed << std::setprecision(6);
  out << "{\n"
      << "  \"schema_version\": 2,\n"
      << "  \"model\": \"" << escape_json(options.model_path) << "\",\n"
      << "  \"backend\": \"" << escape_json(stats.backend) << "\",\n"
      << "  \"requested_backend\": \"" << escape_json(options.requested_backend) << "\",\n"
      << "  \"fallback_used\": " << (options.fallback_used ? "true" : "false") << ",\n"
      << "  \"samples\": " << stats.end_to_end.samples << ",\n"
      << "  \"input_size\": " << options.input_size << ",\n"
      << "  \"batch_size\": " << options.batch_size << ",\n"
      << "  \"latency_us\": {\n"
      << "    \"end_to_end\": ";
  write_distribution(out, stats.end_to_end, 4);
  out << ",\n    \"device_compute\": ";
  if (stats.device_compute) {
    write_distribution(out, *stats.device_compute, 4);
  } else {
    out << "null";
  }
  out << "\n  },\n"
      << "  \"mean_us\": " << stats.end_to_end.mean_us << ",\n"
      << "  \"min_us\": " << stats.end_to_end.min_us << ",\n"
      << "  \"p50_us\": " << stats.end_to_end.p50_us << ",\n"
      << "  \"p90_us\": " << stats.end_to_end.p90_us << ",\n"
      << "  \"p95_us\": " << stats.end_to_end.p95_us << ",\n"
      << "  \"p99_us\": " << stats.end_to_end.p99_us << ",\n"
      << "  \"max_us\": " << stats.end_to_end.max_us << ",\n"
      << "  \"stddev_us\": " << stats.end_to_end.stddev_us << ",\n"
      << "  \"throughput_ips\": " << stats.throughput_inferences_per_second << ",\n"
      << "  \"throughput_samples_per_second\": " << stats.throughput_samples_per_second << ",\n"
      << "  \"validation\": ";
  if (stats.validation) {
    out << "{\n"
        << "    \"kind\": \"identity\",\n"
        << "    \"tolerance\": " << options.validation_tolerance << ",\n"
        << "    \"compared_elements\": " << stats.validation->compared_elements << ",\n"
        << "    \"mismatches\": " << stats.validation->mismatches << ",\n"
        << "    \"max_abs_error\": " << stats.validation->max_abs_error << ",\n"
        << "    \"passed\": " << (stats.validation->passed ? "true" : "false") << "\n"
        << "  }";
  } else {
    out << "null";
  }
  out << ",\n"
      << "  \"slo\": {\n"
      << "    \"scope\": \"" << scope_name(options.target_scope) << "\",\n"
      << "    \"target_us\": ";
  if (options.target_latency_us > 0.0) {
    out << options.target_latency_us;
  } else {
    out << "null";
  }
  out << ",\n    \"observed_p99_us\": ";
  if (selected_available) {
    out << selected->p99_us;
  } else {
    out << "null";
  }
  out << ",\n    \"passed\": ";
  if (options.target_latency_us <= 0.0) {
    out << "null";
  } else {
    out << ((selected_available && selected->p99_us <= options.target_latency_us) ? "true" : "false");
  }
  out << "\n  }\n}\n";
  return out.str();
}

void print_metrics(const RunOptions& options, const InferenceStats& stats) {
  (void)options;
  const auto& e2e = stats.end_to_end;
  std::cout << "\n============================================================\n"
            << "Backend          : " << stats.backend << '\n'
            << "Measured samples : " << e2e.samples << '\n'
            << std::fixed << std::setprecision(3)
            << "E2E mean / p99   : " << e2e.mean_us << " / " << e2e.p99_us << " us\n";
  if (stats.device_compute) {
    std::cout << "GPU mean / p99   : " << stats.device_compute->mean_us << " / "
              << stats.device_compute->p99_us << " us\n";
  } else {
    std::cout << "GPU mean / p99   : n/a (backend has no device timer)\n";
  }
  if (stats.validation) {
    std::cout << "Output validation : " << (stats.validation->passed ? "pass" : "FAIL")
              << " (" << stats.validation->mismatches << " mismatches)\n";
  }
  std::cout << "E2E min / max    : " << e2e.min_us << " / " << e2e.max_us << " us\n"
            << "E2E jitter       : " << e2e.stddev_us << " us\n"
            << "Inference rate   : " << stats.throughput_inferences_per_second << " / s\n"
            << "Sample rate      : " << stats.throughput_samples_per_second << " / s\n"
            << "============================================================\n\n";
}

} // namespace mdedge
