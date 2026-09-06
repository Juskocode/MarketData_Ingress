#include "marketdata_infer/engine.h"

#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
  }
}

class TimedTestEngine final : public mdedge::InferenceEngine {
public:
  bool load(const std::string&) override { return true; }

  bool infer(const std::vector<float>& input, std::vector<float>& output) override {
    if (input.empty()) {
      return false;
    }
    output = input;
    device_latency_us_ += 0.05;
    return true;
  }

  size_t input_elements_per_batch() const override { return 16; }
  const char* backend_name() const override { return "timed-test"; }
  std::optional<double> last_device_latency_us() const override {
    return device_latency_us_;
  }

private:
  double device_latency_us_ = 0.1;
};

} // namespace

int main() {
  mdedge::RunOptions options;
  options.input_size = 16;
  options.batch_size = 1;
  options.iterations = 20;
  options.warmup = 4;
  options.target_latency_us = 1.0;
  options.target_scope = mdedge::LatencyScope::DeviceCompute;
  options.requested_backend = "test";
  options.model_path = "model\"with\\escapes.engine";

  TimedTestEngine timed_engine;
  check(timed_engine.load(options.model_path), "timed test engine loads");
  const auto stats = mdedge::run_benchmark(timed_engine, options);
  check(stats.end_to_end.samples == options.iterations, "E2E sample count matches iterations");
  check(stats.end_to_end.mean_us >= 0.0, "E2E mean is non-negative");
  check(stats.end_to_end.p50_us <= stats.end_to_end.p99_us, "percentiles are monotonic");
  check(stats.device_compute.has_value(), "device timing is captured when backend exposes it");
  check(stats.device_compute && stats.device_compute->samples == options.iterations,
        "device sample count matches iterations");

  const std::string json = mdedge::format_metrics_json(options, stats);
  check(json.find("\"schema_version\": 2") != std::string::npos, "schema version is exported");
  check(json.find("\"device_compute\": {") != std::string::npos, "device distribution is exported");
  check(json.find("model\\\"with\\\\escapes.engine") != std::string::npos,
        "JSON strings are escaped");

  auto mock = mdedge::make_engine(mdedge::BackendHint::Mock, false);
  check(mock != nullptr, "mock factory returns an engine");
  check(mock && mock->load("mock"), "mock engine loads");
  if (mock) {
    options.input_size = 32;
    options.iterations = 8;
    options.warmup = 1;
    const auto mock_stats = mdedge::run_benchmark(*mock, options);
    check(!mock_stats.device_compute.has_value(), "mock does not invent GPU timing");
    check(mock_stats.throughput_samples_per_second > 0.0, "throughput is calculated");
  }

  bool rejected_zero = false;
  try {
    options.input_size = 0;
    mdedge::run_benchmark(timed_engine, options);
  } catch (const std::runtime_error&) {
    rejected_zero = true;
  }
  check(rejected_zero, "zero-sized input is rejected");

  bool rejected_overflow = false;
  try {
    options.input_size = std::numeric_limits<size_t>::max();
    options.batch_size = 2;
    mdedge::run_benchmark(timed_engine, options);
  } catch (const std::runtime_error&) {
    rejected_overflow = true;
  }
  check(rejected_overflow, "input multiplication overflow is rejected");

  if (failures != 0) {
    std::cerr << failures << " test(s) failed\n";
    return 1;
  }
  std::cout << "all tests passed\n";
  return 0;
}
