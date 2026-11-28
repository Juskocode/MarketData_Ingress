#include "marketdata_infer/tensorrt_engine.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <ranges>
#include <vector>

int main(int argc, char** argv) {
  if (argc != 2) return 2;
  mdedge::TensorRTInferenceEngine engine;
  if (!engine.load(argv[1])) return 1;
  auto input = engine.host_input_buffer();
  auto output = engine.host_output_buffer();
  if (input.empty() || input.size() != output.size()) return 1;
  std::vector<float> external_input(input.size());
  std::vector<float> external_output(output.size());
  for (const bool graph : {false, true}) {
    if (graph && (!engine.enable_cuda_graph() || !engine.enable_cuda_graph())) return 1;
    for (const auto request : std::views::iota(0, 16)) {
      // Changing each request exposes graphs that accidentally replay stale data.
      std::ranges::transform(std::views::iota(size_t{0}, input.size()), input.begin(),
          [request](size_t i) { return static_cast<float>(i + static_cast<size_t>(request)); });
      const auto result = engine.infer(input, output);
      if (!result || !std::ranges::equal(input, *result)) return 1;
      const auto timing = engine.last_device_latency_us();
      if (!timing || !std::isfinite(*timing) || *timing < 0.0) return 1;
      std::ranges::copy(input, external_input.begin());
      const auto external = engine.infer(external_input, external_output);
      if (!external || !std::ranges::equal(external_input, *external)) return 1;
    }
    if (engine.infer(input.first(input.size() - 1), output) ||
        engine.infer(input, output.first(output.size() - 1)) ||
        engine.infer({}, output)) return 1;
    if (!engine.infer(input, output)) return 1;
  }
  // Reload destroys graph/staging resources and must leave the engine usable.
  if (!engine.load(argv[1]) || engine.cuda_graph_enabled()) return 1;
  if (!engine.infer(external_input, external_output)) return 1;
  std::cout << "TensorRT changing-input identity: pinned/external, direct/graph, reload PASS\n";
}
