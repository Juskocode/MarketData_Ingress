#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace mdedge {

enum class BackendHint {
  Auto,
  TensorRT,
  Mock,
};

struct RunOptions {
  std::string model_path = "mock";
  size_t input_size = 512;
  size_t batch_size = 1;
  size_t iterations = 200;
  size_t warmup = 20;
  double target_latency_us = 0.0; // 0 disables threshold checks
  std::string metrics_path;
  BackendHint backend = BackendHint::Auto;
};

struct InferenceStats {
  size_t samples = 0;
  double mean_us = 0.0;
  double min_us = 0.0;
  double p50_us = 0.0;
  double p90_us = 0.0;
  double p95_us = 0.0;
  double p99_us = 0.0;
  double max_us = 0.0;
  double stddev_us = 0.0;
  double throughput_samples_per_second = 0.0;
  double input_elements_per_batch = 0.0;
  std::string backend;
};

class InferenceEngine {
public:
  virtual ~InferenceEngine() = default;
  virtual bool load(const std::string& model_path) = 0;
  virtual bool infer(const std::vector<float>& input, std::vector<float>& output) = 0;
  virtual size_t input_elements_per_batch() const = 0;
  virtual const char* backend_name() const = 0;
};

std::unique_ptr<InferenceEngine> make_engine(
    BackendHint preferred,
    bool allow_mock_fallback = true);

InferenceStats run_benchmark(InferenceEngine& engine, const RunOptions& opts);

std::string format_metrics_json(const RunOptions& options, const InferenceStats& stats);
void print_metrics(const RunOptions& options, const InferenceStats& stats);

} // namespace mdedge
