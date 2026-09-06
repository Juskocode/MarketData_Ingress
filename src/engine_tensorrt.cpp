#ifdef MD_WITH_TENSORRT

#include "marketdata_infer/tensorrt_engine.h"

#include <NvOnnxParser.h>

#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <string_view>
#include <vector>

#include <cuda_runtime_api.h>

namespace mdedge {

namespace {

size_t volume(const nvinfer1::Dims& dims) {
  size_t v = 1;
  for (int i = 0; i < dims.nbDims; ++i) {
    v *= static_cast<size_t>(dims.d[i]);
  }
  return v;
}

} // namespace

TensorRTInferenceEngine::TensorRTInferenceEngine() = default;

TensorRTInferenceEngine::~TensorRTInferenceEngine() {
  destroyResources();
}

void TensorRTInferenceEngine::Logger::log(nvinfer1::ILogger::Severity severity, const char* msg) noexcept {
  if (severity <= nvinfer1::ILogger::Severity::kWARNING) {
    std::cerr << "[TRT] " << msg << '\n';
  }
}

void TensorRTInferenceEngine::destroyResources() {
  if (context_) {
    context_->destroy();
    context_ = nullptr;
  }
  if (engine_) {
    engine_->destroy();
    engine_ = nullptr;
  }
  if (runtime_) {
    runtime_->destroy();
    runtime_ = nullptr;
  }
  if (input_device_) {
    cudaFree(input_device_);
    input_device_ = nullptr;
  }
  if (output_device_) {
    cudaFree(output_device_);
    output_device_ = nullptr;
  }
  bindings_[0] = bindings_[1] = nullptr;
  input_binding_ = -1;
  output_binding_ = -1;
  input_bytes_ = 0;
  output_bytes_ = 0;
}

bool TensorRTInferenceEngine::loadSerializedEngine(const std::string& model_path) {
  std::ifstream file(model_path, std::ios::binary);
  if (!file) {
    std::cerr << "Failed to open serialized engine: " << model_path << '\n';
    return false;
  }

  std::vector<char> engine_data((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
  runtime_ = nvinfer1::createInferRuntime(logger_);
  if (!runtime_) {
    std::cerr << "Could not create TensorRT runtime\n";
    return false;
  }

  engine_ = runtime_->deserializeCudaEngine(engine_data.data(), engine_data.size(), nullptr);
  if (!engine_) {
    std::cerr << "Could not deserialize engine\n";
    return false;
  }

  if (engine_->getNbBindings() < 2) {
    std::cerr << "Engine has fewer than two bindings\n";
    return false;
  }

  input_binding_ = 0;
  output_binding_ = 1;
  while (input_binding_ < engine_->getNbBindings() && !engine_->bindingIsInput(input_binding_)) {
    ++input_binding_;
  }
  while (output_binding_ < engine_->getNbBindings() && engine_->bindingIsInput(output_binding_)) {
    ++output_binding_;
  }
  if (input_binding_ >= engine_->getNbBindings() || output_binding_ >= engine_->getNbBindings()) {
    std::cerr << "Could not determine input/output bindings\n";
    return false;
  }

  auto input_shape = engine_->getBindingDimensions(input_binding_);
  auto output_shape = engine_->getBindingDimensions(output_binding_);
  input_bytes_ = volume(input_shape) * sizeof(float);
  output_bytes_ = volume(output_shape) * sizeof(float);

  if (cudaMalloc(&input_device_, input_bytes_) != cudaSuccess ||
      cudaMalloc(&output_device_, output_bytes_) != cudaSuccess) {
    std::cerr << "Could not allocate CUDA buffers\n";
    return false;
  }

  bindings_[input_binding_] = input_device_;
  bindings_[output_binding_] = output_device_;

  context_ = engine_->createExecutionContext();
  if (!context_) {
    std::cerr << "Could not create execution context\n";
    return false;
  }

  return true;
}

bool TensorRTInferenceEngine::loadOnnx(const std::string& model_path) {
  auto builder = nvinfer1::createInferBuilder(logger_);
  if (!builder) {
    std::cerr << "Could not create TensorRT builder\n";
    return false;
  }

  auto network = builder->createNetworkV2(1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));
  if (!network) {
    std::cerr << "Could not create TensorRT network\n";
    return false;
  }

  auto parser = nvonnxparser::createParser(*network, logger_);
  if (!parser || !parser->parseFromFile(model_path.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
    std::cerr << "Failed to parse ONNX model\n";
    if (parser) parser->destroy();
    network->destroy();
    builder->destroy();
    return false;
  }

  auto config = builder->createBuilderConfig();
  if (!config) {
    parser->destroy();
    network->destroy();
    builder->destroy();
    return false;
  }

  if (builder->platformHasFastFp16()) {
    config->setFlag(nvinfer1::BuilderFlag::kFP16);
  }
  config->setMaxWorkspaceSize(1UL << 20);

  auto raw_engine = builder->buildEngineWithConfig(*network, *config);
  if (!raw_engine) {
    std::cerr << "Could not build TensorRT engine from ONNX\n";
    config->destroy();
    parser->destroy();
    network->destroy();
    builder->destroy();
    return false;
  }

  runtime_ = nvinfer1::createInferRuntime(logger_);
  if (!runtime_) {
    raw_engine->destroy();
    config->destroy();
    parser->destroy();
    network->destroy();
    builder->destroy();
    return false;
  }

  auto serialized = raw_engine->serialize();
  if (!serialized) {
    std::cerr << "Could not serialize built engine\n";
    runtime_->destroy();
    raw_engine->destroy();
    config->destroy();
    parser->destroy();
    network->destroy();
    builder->destroy();
    return false;
  }

  engine_ = runtime_->deserializeCudaEngine(serialized->data(), serialized->size(), nullptr);
  serialized->destroy();
  raw_engine->destroy();
  config->destroy();
  parser->destroy();
  network->destroy();
  builder->destroy();

  if (!engine_) {
    std::cerr << "Could not deserialize built engine\n";
    return false;
  }

  // same extraction logic
  input_binding_ = 0;
  output_binding_ = 1;
  while (input_binding_ < engine_->getNbBindings() && !engine_->bindingIsInput(input_binding_)) {
    ++input_binding_;
  }
  while (output_binding_ < engine_->getNbBindings() && engine_->bindingIsInput(output_binding_)) {
    ++output_binding_;
  }

  auto input_shape = engine_->getBindingDimensions(input_binding_);
  auto output_shape = engine_->getBindingDimensions(output_binding_);
  input_bytes_ = volume(input_shape) * sizeof(float);
  output_bytes_ = volume(output_shape) * sizeof(float);

  if (cudaMalloc(&input_device_, input_bytes_) != cudaSuccess ||
      cudaMalloc(&output_device_, output_bytes_) != cudaSuccess) {
    std::cerr << "Could not allocate CUDA buffers\n";
    return false;
  }

  bindings_[input_binding_] = input_device_;
  bindings_[output_binding_] = output_device_;

  context_ = engine_->createExecutionContext();
  return context_ != nullptr;
}

bool TensorRTInferenceEngine::load(const std::string& model_path) {
  destroyResources();

  if (model_path.size() >= 5 &&
      model_path.compare(model_path.size() - 5, 5, ".onnx") == 0) {
    return loadOnnx(model_path);
  }
  return loadSerializedEngine(model_path);
}

bool TensorRTInferenceEngine::infer(const std::vector<float>& input, std::vector<float>& output) {
  if (!context_ || input.empty() || !input_device_ || !output_device_) {
    return false;
  }
  if (input.size() * sizeof(float) != input_bytes_) {
    std::cerr << "Input size mismatch\n";
    return false;
  }

  cudaError_t err = cudaMemcpyAsync(input_device_, input.data(), input_bytes_, cudaMemcpyHostToDevice, 0);
  if (err != cudaSuccess) {
    std::cerr << "cudaMemcpyAsync H2D failed\n";
    return false;
  }

  if (!context_->enqueueV2(bindings_, 0, nullptr)) {
    std::cerr << "Inference enqueue failed\n";
    return false;
  }

  output.resize(output_bytes_ / sizeof(float));
  err = cudaMemcpyAsync(output.data(), output_device_, output_bytes_, cudaMemcpyDeviceToHost, 0);
  if (err != cudaSuccess) {
    std::cerr << "cudaMemcpyAsync D2H failed\n";
    return false;
  }
  if (cudaStreamSynchronize(0) != cudaSuccess) {
    std::cerr << "cudaStreamSynchronize failed\n";
    return false;
  }

  return true;
}

size_t TensorRTInferenceEngine::input_elements_per_batch() const {
  return input_bytes_ / sizeof(float);
}

const char* TensorRTInferenceEngine::backend_name() const {
  return "tensorrt-gpu";
}

} // namespace mdedge

#endif
