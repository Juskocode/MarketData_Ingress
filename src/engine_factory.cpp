#include "marketdata_infer/engine.h"
#include "marketdata_infer/mock_engine.h"
#ifdef MD_WITH_TENSORRT
#include "marketdata_infer/tensorrt_engine.h"
#endif

namespace mdedge {

std::unique_ptr<InferenceEngine> make_engine(BackendHint preferred, bool allow_mock_fallback) {
#ifdef MD_FORCE_MOCK_BACKEND
  (void)preferred;
  (void)allow_mock_fallback;
  return std::make_unique<MockInferenceEngine>();
#else
#ifdef MD_WITH_TENSORRT
  if (preferred == BackendHint::TensorRT || preferred == BackendHint::Auto) {
    return std::make_unique<TensorRTInferenceEngine>();
  }
#endif
  if (preferred == BackendHint::Mock) {
    return std::make_unique<MockInferenceEngine>();
  }
  if (allow_mock_fallback) {
    return std::make_unique<MockInferenceEngine>();
  }
  return nullptr;
#endif
}

} // namespace mdedge
