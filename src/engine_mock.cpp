#include "marketdata_infer/mock_engine.h"

namespace mdedge {

bool MockInferenceEngine::load(const std::string& model_path) {
  (void)model_path;
  input_elements_ = 0;
  return true;
}

bool MockInferenceEngine::infer_into(
    const float* input,
    size_t input_elements,
    float* output,
    size_t output_capacity,
    size_t& output_elements) {
  output_elements = 0;
  if (!input || !output || input_elements == 0 || output_capacity < input_elements) {
    return false;
  }
  for (size_t i = 0; i < input_elements; ++i) {
    output[i] = (input[i] * 0.25F) + 0.333F;
  }
  output_elements = input_elements;
  return true;
}

size_t MockInferenceEngine::input_elements_per_batch() const {
  return input_elements_;
}

size_t MockInferenceEngine::output_elements_per_batch() const {
  return 0;
}

const char* MockInferenceEngine::backend_name() const {
  return "mock-cpp-cpu";
}

RuntimeMetadata MockInferenceEngine::runtime_metadata() const {
  RuntimeMetadata metadata;
  metadata.backend_version = "deterministic-v1";
  metadata.device_name = "host-cpu";
  return metadata;
}

} // namespace mdedge
