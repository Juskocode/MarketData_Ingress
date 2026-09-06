#pragma once

#ifdef MD_WITH_TENSORRT

#include "marketdata_infer/engine.h"

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <optional>
#include <string>

namespace mdedge {

class TensorRTInferenceEngine final : public InferenceEngine {
public:
  TensorRTInferenceEngine();
  ~TensorRTInferenceEngine() override;

  TensorRTInferenceEngine(const TensorRTInferenceEngine&) = delete;
  TensorRTInferenceEngine& operator=(const TensorRTInferenceEngine&) = delete;

  bool load(const std::string& model_path) override;
  bool infer(const std::vector<float>& input, std::vector<float>& output) override;
  size_t input_elements_per_batch() const override;
  const char* backend_name() const override;
  std::optional<double> last_device_latency_us() const override;

private:
  bool loadSerializedEngine(const std::string& model_path);
  bool loadOnnx(const std::string& model_path);
  bool initializeExecution();
  void destroyResources();

  class Logger final : public nvinfer1::ILogger {
  public:
    void log(nvinfer1::ILogger::Severity severity, const char* msg) noexcept override;
  } logger_;

  nvinfer1::IRuntime* runtime_ = nullptr;
  nvinfer1::ICudaEngine* engine_ = nullptr;
  nvinfer1::IExecutionContext* context_ = nullptr;

  void* input_device_ = nullptr;
  void* output_device_ = nullptr;
  void* input_host_ = nullptr;
  void* output_host_ = nullptr;
  size_t input_bytes_ = 0;
  size_t output_bytes_ = 0;
  std::string input_name_;
  std::string output_name_;

  cudaStream_t stream_ = nullptr;
  cudaEvent_t device_start_ = nullptr;
  cudaEvent_t device_stop_ = nullptr;
  std::optional<double> last_device_latency_us_;
};

} // namespace mdedge

#endif
