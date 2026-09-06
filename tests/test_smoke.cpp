#include "marketdata_infer/engine.h"

#include <cassert>
#include <iostream>

int main() {
  mdedge::RunOptions options;
  options.input_size = 16;
  options.batch_size = 1;
  options.iterations = 16;
  options.warmup = 4;

  auto engine = mdedge::make_engine(mdedge::BackendHint::Mock, true);
  assert(engine != nullptr);

  bool loaded = engine->load("mock");
  assert(loaded);

  auto stats = mdedge::run_benchmark(*engine, options);
  assert(stats.samples == options.iterations);
  assert(stats.mean_us >= 0.0);

  std::string json = mdedge::format_metrics_json(options, stats);
  assert(json.find("\"backend\"") != std::string::npos);

  std::cout << "test_smoke passed\n";
  return 0;
}
