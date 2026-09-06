#include <NvInfer.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

namespace {

class Logger final : public nvinfer1::ILogger {
public:
  void log(Severity severity, const char* message) noexcept override {
    if (severity <= Severity::kWARNING) {
      std::cerr << "[TensorRT] " << message << '\n';
    }
  }
};

template <typename T>
using TrtUniquePtr = std::unique_ptr<T>;

size_t parse_positive_size(const std::string& value, const std::string& flag) {
  size_t consumed = 0;
  const unsigned long long parsed = std::stoull(value, &consumed);
  if (consumed != value.size() || parsed == 0 ||
      parsed > static_cast<unsigned long long>(std::numeric_limits<int32_t>::max())) {
    throw std::runtime_error(flag + " must be between 1 and INT32_MAX");
  }
  return static_cast<size_t>(parsed);
}

void print_help(const char* executable) {
  std::cout << "Usage: " << executable << " [options]\n\n"
            << "Build a fixed-shape FP32 TensorRT identity engine in native C++.\n\n"
            << "Options:\n"
            << "  --output <path>       Engine output (default: models/tiny.engine)\n"
            << "  --input-size <N>      Elements in the [1, N] tensor (default: 512)\n"
            << "  --workspace-mib <N>   TensorRT build workspace (default: 16)\n"
            << "  --help                Show this help and exit\n";
}

} // namespace

int main(int argc, char** argv) {
  std::string output_path = "models/tiny.engine";
  size_t input_size = 512;
  size_t workspace_mib = 16;

  try {
    for (int i = 1; i < argc; ++i) {
      const std::string argument = argv[i];
      auto require_value = [&](const std::string& flag) -> std::string {
        if (i + 1 >= argc) {
          throw std::runtime_error("Missing value for " + flag);
        }
        return argv[++i];
      };

      if (argument == "--help" || argument == "-h") {
        print_help(argv[0]);
        return 0;
      }
      if (argument == "--output") {
        output_path = require_value(argument);
      } else if (argument == "--input-size") {
        input_size = parse_positive_size(require_value(argument), argument);
      } else if (argument == "--workspace-mib") {
        workspace_mib = parse_positive_size(require_value(argument), argument);
      } else {
        throw std::runtime_error("Unknown argument: " + argument);
      }
    }

    if (workspace_mib > std::numeric_limits<size_t>::max() / (1024U * 1024U)) {
      throw std::runtime_error("Workspace size overflows size_t");
    }

    Logger logger;
    TrtUniquePtr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(logger));
    if (!builder) {
      throw std::runtime_error("Could not create TensorRT builder");
    }
    TrtUniquePtr<nvinfer1::INetworkDefinition> network(builder->createNetworkV2(0U));
    if (!network) {
      throw std::runtime_error("Could not create TensorRT network");
    }

    nvinfer1::Dims input_dimensions{};
    input_dimensions.nbDims = 2;
    input_dimensions.d[0] = 1;
    input_dimensions.d[1] = static_cast<int64_t>(input_size);
    auto* input = network->addInput("input", nvinfer1::DataType::kFLOAT, input_dimensions);
    if (!input) {
      throw std::runtime_error("Could not add the input tensor");
    }
    auto* identity = network->addIdentity(*input);
    if (!identity || !identity->getOutput(0)) {
      throw std::runtime_error("Could not add the identity layer");
    }
    auto* output = identity->getOutput(0);
    output->setName("output");
    network->markOutput(*output);

    TrtUniquePtr<nvinfer1::IBuilderConfig> config(builder->createBuilderConfig());
    if (!config) {
      throw std::runtime_error("Could not create TensorRT builder configuration");
    }
    config->setMemoryPoolLimit(
        nvinfer1::MemoryPoolType::kWORKSPACE,
        workspace_mib * 1024U * 1024U);

    TrtUniquePtr<nvinfer1::IHostMemory> serialized(
        builder->buildSerializedNetwork(*network, *config));
    if (!serialized) {
      throw std::runtime_error("TensorRT could not build the identity engine");
    }

    const std::filesystem::path destination(output_path);
    if (destination.has_parent_path()) {
      std::filesystem::create_directories(destination.parent_path());
    }
    std::ofstream output_file(destination, std::ios::binary | std::ios::trunc);
    if (!output_file) {
      throw std::runtime_error("Could not open engine output: " + output_path);
    }
    output_file.write(
        static_cast<const char*>(serialized->data()),
        static_cast<std::streamsize>(serialized->size()));
    if (!output_file) {
      throw std::runtime_error("Could not write complete engine: " + output_path);
    }

    std::cout << "Built TensorRT identity engine\n"
              << "  output: " << output_path << '\n'
              << "  input:  [1, " << input_size << "] FP32\n"
              << "  bytes:  " << serialized->size() << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "Error: " << error.what() << '\n';
    return 1;
  }
}
