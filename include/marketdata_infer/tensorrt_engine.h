#pragma once

#ifdef MD_WITH_TENSORRT

#include "marketdata_infer/engine.h"
#include <NvInfer.h>

namespace mdedge {

class TensorRTInferenceEngine final : public InferenceEngine {
public:
  TensorRTInferenceEngine();
  ~TensorRTInferenceEngine() override;

  bool load(const std::string& model_path) override;
  bool infer(const std::vector<float>& input, std::vector<float>& output) override;
  size_t input_elements_per_batch() const override;
  const char* backend_name() const override;

private:
  bool loadSerializedEngine(const std::string& model_path);
  bool loadOnnx(const std::string& model_path);
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
  size_t input_bytes_ = 0;
  size_t output_bytes_ = 0;
  int input_binding_ = -1;
  int output_binding_ = -1;
  void* bindings_[2] = {nullptr, nullptr};
};

} // namespace mdedge

#endif
