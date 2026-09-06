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
  explicit TimedTestEngine(bool corrupt_output = false) : corrupt_output_(corrupt_output) {}

  bool load(const std::string&) override { return true; }

  bool infer_into(
      const float* input,
      size_t input_elements,
      float* output,
      size_t output_capacity,
      size_t& output_elements) override {
    output_elements = 0;
    if (!input || !output || input_elements == 0 || output_capacity < input_elements) {
      return false;
    }
    for (size_t i = 0; i < input_elements; ++i) {
      output[i] = input[i];
    }
    if (corrupt_output_) {
      output[0] += 1.0F;
    }
    output_elements = input_elements;
    device_latency_us_ += 0.05;
    return true;
  }

  size_t input_elements_per_batch() const override { return 16; }
  size_t output_elements_per_batch() const override { return 16; }
  const char* backend_name() const override { return "timed-test"; }
  std::optional<double> last_device_latency_us() const override {
    return device_latency_us_;
  }
  bool enable_cuda_graph() override {
    cuda_graph_ = true;
    return true;
  }
  bool cuda_graph_enabled() const override { return cuda_graph_; }
  mdedge::RuntimeMetadata runtime_metadata() const override {
    mdedge::RuntimeMetadata metadata;
    metadata.backend_version = "test-v1";
    metadata.device_name = "virtual-gpu";
    metadata.compute_capability = "9.9";
    return metadata;
  }

private:
  bool corrupt_output_ = false;
  bool cuda_graph_ = false;
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
  options.verify_identity = true;
  options.requested_backend = "test";
  options.model_path = "model\"with\\escapes.engine";

  TimedTestEngine timed_engine;
  check(timed_engine.load(options.model_path), "timed test engine loads");
  check(timed_engine.enable_cuda_graph(), "test engine enables CUDA graph mode");
  const auto stats = mdedge::run_benchmark(timed_engine, options);
  check(stats.end_to_end.samples == options.iterations, "E2E sample count matches iterations");
  check(stats.end_to_end.mean_us >= 0.0, "E2E mean is non-negative");
  check(stats.end_to_end.p50_us <= stats.end_to_end.p99_us, "percentiles are monotonic");
  check(stats.device_compute.has_value(), "device timing is captured when backend exposes it");
  check(stats.device_compute && stats.device_compute->samples == options.iterations,
        "device sample count matches iterations");
  check(stats.validation && stats.validation->passed, "identity output validation passes");
  check(stats.cuda_graph, "execution mode is exported from the engine");
  check(stats.runtime.device_name == "virtual-gpu", "runtime metadata is collected");

  const std::string json = mdedge::format_metrics_json(options, stats);
  check(json.find("\"schema_version\": 2") != std::string::npos, "schema version is exported");
  check(json.find("\"device_compute\": {") != std::string::npos, "device distribution is exported");
  check(json.find("\"kind\": \"identity\"") != std::string::npos, "validation is exported");
  check(json.find("\"cuda_graph\": true") != std::string::npos, "CUDA graph mode is exported");
  check(json.find("\"device\": \"virtual-gpu\"") != std::string::npos, "device metadata is exported");
  check(json.find("model\\\"with\\\\escapes.engine") != std::string::npos,
        "JSON strings are escaped");

  TimedTestEngine corrupt_engine(true);
  check(corrupt_engine.load("corrupt"), "corrupt test engine loads");
  const auto corrupt_stats = mdedge::run_benchmark(corrupt_engine, options);
  check(corrupt_stats.validation && !corrupt_stats.validation->passed,
        "incorrect identity output fails validation");
  check(corrupt_stats.validation && corrupt_stats.validation->mismatches == 1,
        "identity validation reports the mismatch count");

  options.verify_identity = false;
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
    check(mock_stats.runtime.device_name == "host-cpu", "mock runtime identifies the host CPU path");
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
