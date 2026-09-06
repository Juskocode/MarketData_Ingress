#ifdef MD_WITH_TENSORRT

#include "marketdata_infer/tensorrt_engine.h"

#include <NvOnnxParser.h>

#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace mdedge {

namespace {

std::optional<size_t> checked_volume(const nvinfer1::Dims& dims) {
  if (dims.nbDims <= 0) {
    return std::nullopt;
  }
  size_t result = 1;
  for (int32_t i = 0; i < dims.nbDims; ++i) {
    if (dims.d[i] <= 0) {
      return std::nullopt;
    }
    const auto dimension = static_cast<size_t>(dims.d[i]);
    if (result > std::numeric_limits<size_t>::max() / dimension) {
      return std::nullopt;
    }
    result *= dimension;
  }
  return result;
}

bool cuda_ok(cudaError_t status, const char* operation) {
  if (status == cudaSuccess) {
    return true;
  }
  std::cerr << operation << " failed: " << cudaGetErrorString(status) << '\n';
  return false;
}

template <typename T>
using TrtUniquePtr = std::unique_ptr<T>;

} // namespace

TensorRTInferenceEngine::TensorRTInferenceEngine() = default;

TensorRTInferenceEngine::~TensorRTInferenceEngine() {
  destroyResources();
}

void TensorRTInferenceEngine::Logger::log(
    nvinfer1::ILogger::Severity severity,
    const char* msg) noexcept {
  if (severity <= nvinfer1::ILogger::Severity::kWARNING) {
    std::cerr << "[TensorRT] " << msg << '\n';
  }
}

void TensorRTInferenceEngine::destroyResources() {
  if (stream_) {
    cudaStreamSynchronize(stream_);
  }
  if (device_start_) {
    cudaEventDestroy(device_start_);
    device_start_ = nullptr;
  }
  if (device_stop_) {
    cudaEventDestroy(device_stop_);
    device_stop_ = nullptr;
  }
  if (graph_execution_) {
    cudaGraphExecDestroy(graph_execution_);
    graph_execution_ = nullptr;
  }
  if (stream_) {
    cudaStreamDestroy(stream_);
    stream_ = nullptr;
  }
  if (input_host_) {
    cudaFreeHost(input_host_);
    input_host_ = nullptr;
  }
  if (output_host_) {
    cudaFreeHost(output_host_);
    output_host_ = nullptr;
  }
  if (input_device_) {
    cudaFree(input_device_);
    input_device_ = nullptr;
  }
  if (output_device_) {
    cudaFree(output_device_);
    output_device_ = nullptr;
  }

  delete context_;
  context_ = nullptr;
  delete engine_;
  engine_ = nullptr;
  delete runtime_;
  runtime_ = nullptr;

  input_bytes_ = 0;
  output_bytes_ = 0;
  input_name_.clear();
  output_name_.clear();
  last_device_latency_us_.reset();
}

bool TensorRTInferenceEngine::initializeExecution() {
  if (!engine_) {
    return false;
  }

  const int32_t io_count = engine_->getNbIOTensors();
  for (int32_t i = 0; i < io_count; ++i) {
    const char* name = engine_->getIOTensorName(i);
    if (!name) {
      continue;
    }
    const auto mode = engine_->getTensorIOMode(name);
    if (mode == nvinfer1::TensorIOMode::kINPUT) {
      if (!input_name_.empty()) {
        std::cerr << "Only one input tensor is supported by this low-latency runner\n";
        return false;
      }
      input_name_ = name;
    } else if (mode == nvinfer1::TensorIOMode::kOUTPUT) {
      if (!output_name_.empty()) {
        std::cerr << "Only one output tensor is supported by this low-latency runner\n";
        return false;
      }
      output_name_ = name;
    }
  }

  if (input_name_.empty() || output_name_.empty()) {
    std::cerr << "Engine must expose exactly one input and one output tensor\n";
    return false;
  }
  if (engine_->getTensorDataType(input_name_.c_str()) != nvinfer1::DataType::kFLOAT ||
      engine_->getTensorDataType(output_name_.c_str()) != nvinfer1::DataType::kFLOAT) {
    std::cerr << "This runner currently requires FP32 input and output tensors\n";
    return false;
  }

  const auto input_elements = checked_volume(engine_->getTensorShape(input_name_.c_str()));
  const auto output_elements = checked_volume(engine_->getTensorShape(output_name_.c_str()));
  if (!input_elements || !output_elements) {
    std::cerr << "Dynamic or invalid tensor shape detected; build a fixed-shape engine for this runner\n";
    return false;
  }
  input_bytes_ = *input_elements * sizeof(float);
  output_bytes_ = *output_elements * sizeof(float);

  context_ = engine_->createExecutionContext();
  if (!context_) {
    std::cerr << "Could not create TensorRT execution context\n";
    return false;
  }
  if (!cuda_ok(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), "cudaStreamCreateWithFlags") ||
      !cuda_ok(cudaEventCreate(&device_start_), "cudaEventCreate(start)") ||
      !cuda_ok(cudaEventCreate(&device_stop_), "cudaEventCreate(stop)") ||
      !cuda_ok(cudaMalloc(&input_device_, input_bytes_), "cudaMalloc(input)") ||
      !cuda_ok(cudaMalloc(&output_device_, output_bytes_), "cudaMalloc(output)") ||
      !cuda_ok(cudaMallocHost(&input_host_, input_bytes_), "cudaMallocHost(input)") ||
      !cuda_ok(cudaMallocHost(&output_host_, output_bytes_), "cudaMallocHost(output)")) {
    return false;
  }

  if (!context_->setTensorAddress(input_name_.c_str(), input_device_) ||
      !context_->setTensorAddress(output_name_.c_str(), output_device_)) {
    std::cerr << "Could not bind TensorRT tensor addresses\n";
    return false;
  }
  return true;
}

bool TensorRTInferenceEngine::loadSerializedEngine(const std::string& model_path) {
  std::ifstream file(model_path, std::ios::binary);
  if (!file) {
    std::cerr << "Failed to open serialized engine: " << model_path << '\n';
    return false;
  }

  std::vector<char> engine_data(
      (std::istreambuf_iterator<char>(file)),
      std::istreambuf_iterator<char>());
  if (engine_data.empty()) {
    std::cerr << "Serialized engine is empty: " << model_path << '\n';
    return false;
  }

  runtime_ = nvinfer1::createInferRuntime(logger_);
  if (!runtime_) {
    std::cerr << "Could not create TensorRT runtime\n";
    return false;
  }
  engine_ = runtime_->deserializeCudaEngine(engine_data.data(), engine_data.size());
  if (!engine_) {
    std::cerr << "Could not deserialize TensorRT engine\n";
    return false;
  }
  return initializeExecution();
}

bool TensorRTInferenceEngine::loadOnnx(const std::string& model_path) {
  TrtUniquePtr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(logger_));
  if (!builder) {
    std::cerr << "Could not create TensorRT builder\n";
    return false;
  }

  TrtUniquePtr<nvinfer1::INetworkDefinition> network(builder->createNetworkV2(0U));
  if (!network) {
    std::cerr << "Could not create TensorRT network\n";
    return false;
  }
  TrtUniquePtr<nvonnxparser::IParser> parser(nvonnxparser::createParser(*network, logger_));
  if (!parser || !parser->parseFromFile(
          model_path.c_str(),
          static_cast<int32_t>(nvinfer1::ILogger::Severity::kWARNING))) {
    std::cerr << "Failed to parse ONNX model: " << model_path << '\n';
    if (parser) {
      for (int32_t i = 0; i < parser->getNbErrors(); ++i) {
        std::cerr << "[ONNX] " << parser->getError(i)->desc() << '\n';
      }
    }
    return false;
  }

  TrtUniquePtr<nvinfer1::IBuilderConfig> config(builder->createBuilderConfig());
  if (!config) {
    std::cerr << "Could not create TensorRT builder configuration\n";
    return false;
  }
  config->setMemoryPoolLimit(
      nvinfer1::MemoryPoolType::kWORKSPACE,
      static_cast<size_t>(256U) * 1024U * 1024U);

  TrtUniquePtr<nvinfer1::IHostMemory> serialized(
      builder->buildSerializedNetwork(*network, *config));
  if (!serialized) {
    std::cerr << "Could not build serialized TensorRT engine from ONNX\n";
    return false;
  }

  runtime_ = nvinfer1::createInferRuntime(logger_);
  if (!runtime_) {
    std::cerr << "Could not create TensorRT runtime\n";
    return false;
  }
  engine_ = runtime_->deserializeCudaEngine(serialized->data(), serialized->size());
  if (!engine_) {
    std::cerr << "Could not deserialize the engine built from ONNX\n";
    return false;
  }
  return initializeExecution();
}

bool TensorRTInferenceEngine::load(const std::string& model_path) {
  destroyResources();
  if (model_path.size() >= 5 &&
      model_path.compare(model_path.size() - 5, 5, ".onnx") == 0) {
    return loadOnnx(model_path);
  }
  return loadSerializedEngine(model_path);
}

bool TensorRTInferenceEngine::infer(
    const std::vector<float>& input,
    std::vector<float>& output) {
  last_device_latency_us_.reset();
  if (!context_ || !stream_ || !input_device_ || !output_device_ ||
      !input_host_ || !output_host_ || input.empty()) {
    return false;
  }
  if (input.size() * sizeof(float) != input_bytes_) {
    std::cerr << "Input size mismatch: received " << input.size()
              << " FP32 elements, expected " << (input_bytes_ / sizeof(float)) << '\n';
    return false;
  }

  output.resize(output_bytes_ / sizeof(float));
  std::memcpy(input_host_, input.data(), input_bytes_);

  if (!cuda_ok(cudaMemcpyAsync(
          input_device_, input_host_, input_bytes_, cudaMemcpyHostToDevice, stream_),
          "cudaMemcpyAsync(H2D)") ||
      !cuda_ok(cudaEventRecord(device_start_, stream_), "cudaEventRecord(start)")) {
    return false;
  }
  if (graph_execution_) {
    if (!cuda_ok(cudaGraphLaunch(graph_execution_, stream_), "cudaGraphLaunch")) {
      return false;
    }
  } else if (!context_->enqueueV3(stream_)) {
      std::cerr << "TensorRT enqueueV3 failed\n";
      return false;
  }
  if (!cuda_ok(cudaEventRecord(device_stop_, stream_), "cudaEventRecord(stop)") ||
      !cuda_ok(cudaMemcpyAsync(
          output_host_, output_device_, output_bytes_, cudaMemcpyDeviceToHost, stream_),
          "cudaMemcpyAsync(D2H)") ||
      !cuda_ok(cudaStreamSynchronize(stream_), "cudaStreamSynchronize")) {
    return false;
  }

  float elapsed_ms = 0.0F;
  if (!cuda_ok(
          cudaEventElapsedTime(&elapsed_ms, device_start_, device_stop_),
          "cudaEventElapsedTime")) {
    return false;
  }
  last_device_latency_us_ = static_cast<double>(elapsed_ms) * 1000.0;
  std::memcpy(output.data(), output_host_, output_bytes_);
  return true;
}

size_t TensorRTInferenceEngine::input_elements_per_batch() const {
  return input_bytes_ / sizeof(float);
}

const char* TensorRTInferenceEngine::backend_name() const {
  return "tensorrt-gpu";
}

std::optional<double> TensorRTInferenceEngine::last_device_latency_us() const {
  return last_device_latency_us_;
}

bool TensorRTInferenceEngine::enable_cuda_graph() {
  if (graph_execution_) {
    return true;
  }
  if (!context_ || !stream_) {
    return false;
  }

  if (!context_->enqueueV3(stream_) ||
      !cuda_ok(cudaStreamSynchronize(stream_), "CUDA graph priming synchronization")) {
    std::cerr << "Could not prime TensorRT context for CUDA graph capture\n";
    return false;
  }

  if (!cuda_ok(
          cudaStreamBeginCapture(stream_, cudaStreamCaptureModeGlobal),
          "cudaStreamBeginCapture")) {
    return false;
  }
  const bool enqueue_succeeded = context_->enqueueV3(stream_);
  cudaGraph_t captured_graph = nullptr;
  const cudaError_t capture_status = cudaStreamEndCapture(stream_, &captured_graph);
  if (!enqueue_succeeded || capture_status != cudaSuccess || !captured_graph) {
    if (captured_graph) {
      cudaGraphDestroy(captured_graph);
    }
    std::cerr << "CUDA graph capture is unsupported for this engine";
    if (capture_status != cudaSuccess) {
      std::cerr << ": " << cudaGetErrorString(capture_status);
    }
    std::cerr << '\n';
    return false;
  }

  const cudaError_t instantiate_status = cudaGraphInstantiate(
      &graph_execution_, captured_graph, nullptr, nullptr, 0);
  cudaGraphDestroy(captured_graph);
  if (!cuda_ok(instantiate_status, "cudaGraphInstantiate")) {
    graph_execution_ = nullptr;
    return false;
  }
  return true;
}

bool TensorRTInferenceEngine::cuda_graph_enabled() const {
  return graph_execution_ != nullptr;
}

} // namespace mdedge

#endif
