#include "marketdata_infer/mock_engine.h"

#include <algorithm>
#include <cstdlib>

namespace mdedge {

bool MockInferenceEngine::load(const std::string& model_path) {
  (void)model_path;
  input_elements_ = 0;
  return true;
}

bool MockInferenceEngine::infer(const std::vector<float>& input, std::vector<float>& output) {
  if (input.size() == 0) {
    return false;
  }
  output.resize(input.size());
  std::transform(input.begin(), input.end(), output.begin(), [](float x) {
    return (x * 0.25f) + 0.333f;
  });
  return true;
}

size_t MockInferenceEngine::input_elements_per_batch() const {
  return input_elements_;
}

const char* MockInferenceEngine::backend_name() const {
  return "mock-cpp-cpu";
}

} // namespace mdedge
