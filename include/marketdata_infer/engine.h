#pragma once

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace mdedge {

enum class BackendHint {
  Auto,
  TensorRT,
  Mock,
};

enum class LatencyScope {
  EndToEnd,
  DeviceCompute,
};

struct RunOptions {
  std::string model_path = "mock";
  size_t input_size = 512;
  size_t batch_size = 1;
  size_t iterations = 200;
  size_t warmup = 20;
  double target_latency_us = 0.0;
  std::string metrics_path;
  BackendHint backend = BackendHint::Auto;
  LatencyScope target_scope = LatencyScope::EndToEnd;
  bool allow_mock_fallback = false;
  bool fallback_used = false;
  bool verify_identity = false;
  double validation_tolerance = 1e-6;
  std::string requested_backend = "auto";
};

struct LatencyDistribution {
  size_t samples = 0;
  double mean_us = 0.0;
  double min_us = 0.0;
  double p50_us = 0.0;
  double p90_us = 0.0;
  double p95_us = 0.0;
  double p99_us = 0.0;
  double max_us = 0.0;
  double stddev_us = 0.0;
};

struct OutputValidation {
  bool passed = false;
  size_t compared_elements = 0;
  size_t mismatches = 0;
  double max_abs_error = 0.0;
};

struct InferenceStats {
  LatencyDistribution end_to_end;
  std::optional<LatencyDistribution> device_compute;
  std::optional<OutputValidation> validation;
  double throughput_inferences_per_second = 0.0;
  double throughput_samples_per_second = 0.0;
  size_t input_elements_per_batch = 0;
  std::string backend;
};

class InferenceEngine {
public:
  virtual ~InferenceEngine() = default;
  virtual bool load(const std::string& model_path) = 0;
  virtual bool infer(const std::vector<float>& input, std::vector<float>& output) = 0;
  virtual size_t input_elements_per_batch() const = 0;
  virtual const char* backend_name() const = 0;

  virtual std::optional<double> last_device_latency_us() const {
    return std::nullopt;
  }
};

std::unique_ptr<InferenceEngine> make_engine(
    BackendHint preferred,
    bool allow_mock_fallback = false);

InferenceStats run_benchmark(InferenceEngine& engine, const RunOptions& opts);
std::string format_metrics_json(const RunOptions& options, const InferenceStats& stats);
void print_metrics(const RunOptions& options, const InferenceStats& stats);

} // namespace mdedge
