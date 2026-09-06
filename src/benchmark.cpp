#include "marketdata_infer/engine.h"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <sstream>
#include <cmath>
#include <stdexcept>

namespace mdedge {

namespace {

double percentile(std::vector<double> values, double p) {
  if (values.empty()) return 0.0;
  std::sort(values.begin(), values.end());
  auto idx = static_cast<size_t>(std::clamp(p, 0.0, 1.0) * (values.size() - 1));
  return values[idx];
}

double mean_value(const std::vector<double>& values) {
  if (values.empty()) return 0.0;
  double sum = std::accumulate(values.begin(), values.end(), 0.0);
  return sum / values.size();
}

double stddev_value(const std::vector<double>& values, double mean) {
  if (values.empty()) return 0.0;
  double variance = 0.0;
  for (double value : values) {
    double diff = value - mean;
    variance += diff * diff;
  }
  return std::sqrt(variance / values.size());
}

} // namespace

InferenceStats run_benchmark(InferenceEngine& engine, const RunOptions& opts) {
  if (opts.input_size == 0 || opts.batch_size == 0 || opts.iterations == 0) {
    throw std::runtime_error("Invalid benchmark options");
  }

  size_t total_input = opts.input_size * opts.batch_size;
  if (engine.input_elements_per_batch() != 0 && total_input > engine.input_elements_per_batch()) {
    throw std::runtime_error("Input exceeds model input capacity");
  }

  std::vector<float> input(total_input);
  for (size_t i = 0; i < input.size(); ++i) {
    input[i] = static_cast<float>(i % 97) / 97.0f;
  }
  std::vector<float> output;

  for (size_t i = 0; i < opts.warmup; ++i) {
    if (!engine.infer(input, output)) {
      throw std::runtime_error("Warmup inference failed");
    }
  }

  std::vector<double> latencies;
  latencies.reserve(opts.iterations);

  for (size_t i = 0; i < opts.iterations; ++i) {
    const auto start = std::chrono::high_resolution_clock::now();
    if (!engine.infer(input, output)) {
      throw std::runtime_error("Inference failed during benchmark");
    }
    const auto end = std::chrono::high_resolution_clock::now();
    double elapsed_us = std::chrono::duration<double, std::micro>(end - start).count();
    latencies.push_back(elapsed_us);
  }

  std::vector<double> sorted = latencies;
  std::sort(sorted.begin(), sorted.end());

  double sum = mean_value(sorted);
  double total_seconds = std::accumulate(sorted.begin(), sorted.end(), 0.0) / 1e6;

  InferenceStats stats;
  stats.samples = sorted.size();
  stats.mean_us = sum;
  stats.min_us = sorted.front();
  stats.max_us = sorted.back();
  stats.p50_us = percentile(sorted, 0.50);
  stats.p90_us = percentile(sorted, 0.90);
  stats.p95_us = percentile(sorted, 0.95);
  stats.p99_us = percentile(sorted, 0.99);
  stats.stddev_us = stddev_value(sorted, sum);
  stats.input_elements_per_batch = static_cast<double>(total_input);
  stats.throughput_samples_per_second = total_seconds > 0 ? (static_cast<double>(opts.iterations) / total_seconds) : 0.0;
  stats.backend = engine.backend_name();
  return stats;
}

std::string format_metrics_json(const RunOptions& options, const InferenceStats& stats) {
  std::ostringstream out;
  out << "{\n";
  out << "  \"model\": \"" << options.model_path << "\",\n";
  out << "  \"backend\": \"" << stats.backend << "\",\n";
  out << "  \"samples\": " << stats.samples << ",\n";
  out << "  \"input_size\": " << options.input_size << ",\n";
  out << "  \"batch_size\": " << options.batch_size << ",\n";
  out << "  \"mean_us\": " << std::fixed << std::setprecision(6) << stats.mean_us << ",\n";
  out << "  \"min_us\": " << std::fixed << std::setprecision(6) << stats.min_us << ",\n";
  out << "  \"p50_us\": " << std::fixed << std::setprecision(6) << stats.p50_us << ",\n";
  out << "  \"p90_us\": " << std::fixed << std::setprecision(6) << stats.p90_us << ",\n";
  out << "  \"p95_us\": " << std::fixed << std::setprecision(6) << stats.p95_us << ",\n";
  out << "  \"p99_us\": " << std::fixed << std::setprecision(6) << stats.p99_us << ",\n";
  out << "  \"max_us\": " << std::fixed << std::setprecision(6) << stats.max_us << ",\n";
  out << "  \"stddev_us\": " << std::fixed << std::setprecision(6) << stats.stddev_us << ",\n";
  out << "  \"throughput_ips\": " << std::fixed << std::setprecision(6) << stats.throughput_samples_per_second << "\n";
  out << "}\n";
  return out.str();
}

void print_metrics(const RunOptions& options, const InferenceStats& stats) {
  (void)options;
  std::cout << "\n\n======================================================\n";
  std::cout << "Backend       : " << stats.backend << "\n";
  std::cout << "Samples       : " << stats.samples << "\n";
  std::cout << "Mean latency  : " << std::fixed << std::setprecision(3) << stats.mean_us << " us\n";
  std::cout << "p50/p90/p95/p99: " << stats.p50_us << " / " << stats.p90_us << " / " << stats.p95_us << " / " << stats.p99_us << " us\n";
  std::cout << "Min/Max      : " << stats.min_us << " / " << stats.max_us << " us\n";
  std::cout << "Stddev       : " << stats.stddev_us << " us\n";
  std::cout << "Throughput   : " << stats.throughput_samples_per_second << " inferences/s\n";
  std::cout << "======================================================\n\n";
}

} // namespace mdedge
