#pragma once

#include "marketdata_infer/engine.h"

namespace mdedge {

class MockInferenceEngine final : public InferenceEngine {
public:
  bool load(const std::string& model_path) override;
  bool infer_into(
      const float* input,
      size_t input_elements,
      float* output,
      size_t output_capacity,
      size_t& output_elements) override;
  size_t input_elements_per_batch() const override;
  size_t output_elements_per_batch() const override;
  const char* backend_name() const override;
  RuntimeMetadata runtime_metadata() const override;

private:
  size_t input_elements_ = 512;
};

} // namespace mdedge
