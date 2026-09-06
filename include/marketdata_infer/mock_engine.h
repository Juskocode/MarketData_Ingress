#pragma once

#include "marketdata_infer/engine.h"

namespace mdedge {

class MockInferenceEngine final : public InferenceEngine {
public:
  bool load(const std::string& model_path) override;
  bool infer(const std::vector<float>& input, std::vector<float>& output) override;
  size_t input_elements_per_batch() const override;
  const char* backend_name() const override;
  RuntimeMetadata runtime_metadata() const override;

private:
  size_t input_elements_ = 512;
};

} // namespace mdedge
