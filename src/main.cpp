#include "marketdata_infer/engine.h"

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void print_help(const char* exe) {
  std::cout << "Usage: " << exe << " [options]\n"
            << "\n"
            << "ML edge inference benchmark using C++ and TensorRT (if available)\n"
            << "\n"
            << "Options:\n"
            << "  --model <path>            Path to .engine or .onnx model (default: mock)\n"
            << "  --batch <N>               Batch size (default: 1)\n"
            << "  --input-size <N>          Input vector size per batch (default: 512)\n"
            << "  --iterations <N>          Benchmark iterations (default: 200)\n"
            << "  --warmup <N>              Warmup iterations (default: 20)\n"
            << "  --json-out <path>         Write metrics JSON to file\n"
            << "  --backend tensorrt|mock    Force backend\n"
            << "  --target-us <us>          CI threshold check (p99 must be <= value)\n"
            << "  --help                    Show this help and exit\n";
}

} // namespace

int main(int argc, char** argv) {
  mdedge::RunOptions options;
  try {
    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      auto require_value = [&](const std::string& flag) {
        if (i + 1 >= argc) {
          throw std::runtime_error("Missing value for " + flag);
        }
        return argv[++i];
      };

      if (arg == "--help" || arg == "-h") {
        print_help(argv[0]);
        return 0;
      } else if (arg == "--model") {
        options.model_path = require_value(arg);
      } else if (arg == "--batch") {
        options.batch_size = static_cast<size_t>(std::stoull(require_value(arg)));
      } else if (arg == "--input-size") {
        options.input_size = static_cast<size_t>(std::stoull(require_value(arg)));
      } else if (arg == "--iterations") {
        options.iterations = static_cast<size_t>(std::stoull(require_value(arg)));
      } else if (arg == "--warmup") {
        options.warmup = static_cast<size_t>(std::stoull(require_value(arg)));
      } else if (arg == "--json-out") {
        options.metrics_path = require_value(arg);
      } else if (arg == "--target-us") {
        options.target_latency_us = std::stod(require_value(arg));
      } else if (arg == "--backend") {
        auto value = require_value(arg);
        if (value == "tensorrt") {
          options.backend = mdedge::BackendHint::TensorRT;
        } else if (value == "mock") {
          options.backend = mdedge::BackendHint::Mock;
        } else {
          throw std::runtime_error("Unsupported backend value: " + value);
        }
      } else {
        throw std::runtime_error("Unknown argument: " + arg);
      }
    }

  auto engine = mdedge::make_engine(options.backend);
  if (!engine) {
    throw std::runtime_error("No engine backend available");
  }

  if (!engine->load(options.model_path)) {
    std::cerr << "Primary backend failed to load model; trying deterministic mock fallback\n";
    if (options.backend == mdedge::BackendHint::TensorRT) {
      engine = mdedge::make_engine(mdedge::BackendHint::Mock);
      options.model_path = "mock";
      if (!engine || !engine->load(options.model_path)) {
        std::cerr << "Fallback backend failed\n";
        return 2;
      }
    } else {
      return 2;
    }
  }

  const auto stats = mdedge::run_benchmark(*engine, options);
  mdedge::print_metrics(options, stats);
  const std::string json = mdedge::format_metrics_json(options, stats);

  if (!options.metrics_path.empty()) {
    std::ofstream out(options.metrics_path);
    if (!out) {
      std::cerr << "Failed to write metrics to " << options.metrics_path << '\n';
      return 2;
    }
    out << json;
  }

  std::cout << json;

  if (options.target_latency_us > 0.0 && stats.p99_us > options.target_latency_us) {
    std::cerr << "Latency target missed: p99=" << stats.p99_us << " us > " << options.target_latency_us << " us\n";
    return 3;
  }

  return 0;
  } catch (const std::exception& ex) {
    std::cerr << "Error: " << ex.what() << '\n';
    print_help(argv[0]);
    return 1;
  }
}
