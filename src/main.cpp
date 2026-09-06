#include "marketdata_infer/engine.h"

#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

namespace {

void print_help(const char* exe) {
  std::cout << "Usage: " << exe << " [options]\n"
            << "\n"
            << "Low-latency C++ inference benchmark with a TensorRT 10+ backend\n"
            << "\n"
            << "Options:\n"
            << "  --model <path>               Fixed-shape .engine or .onnx model (default: mock)\n"
            << "  --backend auto|tensorrt|mock Backend policy (default: auto)\n"
            << "  --allow-mock-fallback        Permit an explicit TensorRT-to-mock fallback\n"
            << "  --batch <N>                  Batch size (default: 1)\n"
            << "  --input-size <N>             Elements per sample (default: 512)\n"
            << "  --iterations <N>             Measured iterations (default: 200)\n"
            << "  --warmup <N>                 Warmup iterations (default: 20)\n"
            << "  --json-out <path>            Write metrics JSON to file\n"
            << "  --target-us <us>             Fail when selected p99 exceeds this value\n"
            << "  --target-scope e2e|device    SLO metric (default: e2e)\n"
            << "  --verify-identity            Require output to equal deterministic input\n"
            << "  --tolerance <value>          Identity absolute tolerance (default: 1e-6)\n"
            << "  --help                       Show this help and exit\n";
}
size_t parse_size(const std::string& value, const std::string& flag, bool allow_zero) {
  size_t consumed = 0;
  const unsigned long long parsed = std::stoull(value, &consumed);
  if (consumed != value.size() || (!allow_zero && parsed == 0) ||
      parsed > std::numeric_limits<size_t>::max()) {
    throw std::runtime_error("Invalid value for " + flag + ": " + value);
  }
  return static_cast<size_t>(parsed);
}

double parse_target(const std::string& value) {
  size_t consumed = 0;
  const double parsed = std::stod(value, &consumed);
  if (consumed != value.size() || !std::isfinite(parsed) || parsed <= 0.0) {
    throw std::runtime_error("--target-us must be a finite value greater than zero");
  }
  return parsed;
}

double parse_tolerance(const std::string& value) {
  size_t consumed = 0;
  const double parsed = std::stod(value, &consumed);
  if (consumed != value.size() || !std::isfinite(parsed) || parsed < 0.0) {
    throw std::runtime_error("--tolerance must be a finite non-negative value");
  }
  return parsed;
}

const char* backend_name(mdedge::BackendHint backend) {
  switch (backend) {
    case mdedge::BackendHint::TensorRT: return "tensorrt";
    case mdedge::BackendHint::Mock: return "mock";
    case mdedge::BackendHint::Auto: return "auto";
  }
  return "unknown";
}

} // namespace

int main(int argc, char** argv) {
  mdedge::RunOptions options;

  try {
    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      auto require_value = [&](const std::string& flag) -> std::string {
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
        options.batch_size = parse_size(require_value(arg), arg, false);
      } else if (arg == "--input-size") {
        options.input_size = parse_size(require_value(arg), arg, false);
      } else if (arg == "--iterations") {
        options.iterations = parse_size(require_value(arg), arg, false);
      } else if (arg == "--warmup") {
        options.warmup = parse_size(require_value(arg), arg, true);
      } else if (arg == "--json-out") {
        options.metrics_path = require_value(arg);
      } else if (arg == "--target-us") {
        options.target_latency_us = parse_target(require_value(arg));
      } else if (arg == "--target-scope") {
        const std::string value = require_value(arg);
        if (value == "e2e") {
          options.target_scope = mdedge::LatencyScope::EndToEnd;
        } else if (value == "device") {
          options.target_scope = mdedge::LatencyScope::DeviceCompute;
        } else {
          throw std::runtime_error("Unsupported target scope: " + value);
        }
      } else if (arg == "--backend") {
        const std::string value = require_value(arg);
        if (value == "auto") {
          options.backend = mdedge::BackendHint::Auto;
        } else if (value == "tensorrt") {
          options.backend = mdedge::BackendHint::TensorRT;
        } else if (value == "mock") {
          options.backend = mdedge::BackendHint::Mock;
        } else {
          throw std::runtime_error("Unsupported backend value: " + value);
        }
      } else if (arg == "--allow-mock-fallback") {
        options.allow_mock_fallback = true;
      } else if (arg == "--verify-identity") {
        options.verify_identity = true;
      } else if (arg == "--tolerance") {
        options.validation_tolerance = parse_tolerance(require_value(arg));
      } else {
        throw std::runtime_error("Unknown argument: " + arg);
      }
    }

    mdedge::BackendHint resolved_backend = options.backend;
    if (resolved_backend == mdedge::BackendHint::Auto) {
      resolved_backend = options.model_path == "mock"
          ? mdedge::BackendHint::Mock
          : mdedge::BackendHint::TensorRT;
    }
    options.requested_backend = backend_name(resolved_backend);

    auto use_mock_fallback = [&]() -> std::unique_ptr<mdedge::InferenceEngine> {
      if (!options.allow_mock_fallback || resolved_backend == mdedge::BackendHint::Mock) {
        return nullptr;
      }
      std::cerr << "TensorRT unavailable or model load failed; using explicitly permitted mock fallback\n";
      options.fallback_used = true;
      auto fallback = mdedge::make_engine(mdedge::BackendHint::Mock, false);
      if (fallback && fallback->load(options.model_path)) {
        return fallback;
      }
      return nullptr;
    };

    auto engine = mdedge::make_engine(resolved_backend, false);
    if (!engine) {
      engine = use_mock_fallback();
      if (!engine) {
        std::cerr << "Requested TensorRT backend is not built. Configure with "
                  << "-DMD_REQUIRE_TENSORRT=ON, or deliberately pass --allow-mock-fallback.\n";
        return 2;
      }
    } else if (!engine->load(options.model_path)) {
      engine = use_mock_fallback();
      if (!engine) {
        std::cerr << "Backend failed to load model: " << options.model_path << '\n';
        return 2;
      }
    }

    const auto stats = mdedge::run_benchmark(*engine, options);
    mdedge::print_metrics(options, stats);
    const std::string json = mdedge::format_metrics_json(options, stats);

    if (!options.metrics_path.empty()) {
      std::ofstream out(options.metrics_path, std::ios::trunc);
      if (!out) {
        std::cerr << "Failed to write metrics to " << options.metrics_path << '\n';
        return 2;
      }
      out << json;
      if (!out) {
        std::cerr << "Failed while writing metrics to " << options.metrics_path << '\n';
        return 2;
      }
    }

    std::cout << json;

    if (stats.validation && !stats.validation->passed) {
      std::cerr << "Output validation failed: " << stats.validation->mismatches
                << " mismatches, max absolute error="
                << stats.validation->max_abs_error << '\n';
      return 5;
    }

    if (options.target_latency_us > 0.0) {
      const mdedge::LatencyDistribution* selected = &stats.end_to_end;
      if (options.target_scope == mdedge::LatencyScope::DeviceCompute) {
        if (!stats.device_compute) {
          std::cerr << "Device-compute target requested, but this backend exposes no device timing\n";
          return 4;
        }
        selected = &*stats.device_compute;
      }
      if (selected->p99_us > options.target_latency_us) {
        std::cerr << "Latency target missed: p99=" << selected->p99_us
                  << " us > " << options.target_latency_us << " us\n";
        return 3;
      }
    }

    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "Error: " << ex.what() << '\n';
    print_help(argv[0]);
    return 1;
  }
}
