#include "marketdata_infer/engine.h"
#include "marketdata_infer/lake_dataset.h"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <vector>

namespace {
void put64(std::vector<char>& bytes, size_t offset, uint64_t value) {
  for (size_t i = 0; i < 8; ++i) bytes[offset + i] = static_cast<char>((value >> (8 * i)) & 0xffU);
}
std::vector<char> fixture() {
  std::vector<char> data(64 + 20 * sizeof(float), 0);
  std::ranges::copy(std::string_view("MDLAKE01"), data.begin());
  put64(data, 8, 10);
  put64(data, 16, 2);
  put64(data, 24, 80);
  for (size_t i = 0; i < 20; ++i) {
    const auto bits = std::bit_cast<uint32_t>(static_cast<float>(i) + 0.5F);
    for (size_t j = 0; j < 4; ++j) data[64 + i * 4 + j] = static_cast<char>((bits >> (j * 8)) & 0xffU);
  }
  return data;
}
void write(const std::filesystem::path& path, const std::vector<char>& data) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  out.write(data.data(), static_cast<std::streamsize>(data.size()));
  if (!out) throw std::runtime_error("Cannot write fixture");
}
class IdentityEngine final : public mdedge::InferenceEngine {
public:
  bool load(const std::string&) override { return true; }
  bool infer_into(const float* input, size_t n, float* out, size_t capacity, size_t& written) override {
    if (n != 4 || capacity < n) return false;
    first_values.push_back(input[0]);
    std::copy_n(input, n, out);
    written = n;
    return true;
  }
  size_t input_elements_per_batch() const override { return 4; }
  size_t output_elements_per_batch() const override { return 4; }
  const char* backend_name() const override { return "test-identity"; }
  std::vector<float> first_values;
};
void require(bool value, const char* message) {
  if (!value) throw std::runtime_error(message);
}
template<class F> void rejects(F&& fn) {
  bool rejected = false;
  try { fn(); } catch (const std::exception&) { rejected = true; }
  require(rejected, "Invalid dataset/shape unexpectedly accepted");
}
}
int main() {
  const auto root = std::filesystem::temp_directory_path() /
      ("md-lake-test-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
  std::filesystem::create_directory(root);
  struct Cleanup {
    std::filesystem::path path;
    ~Cleanup() { std::error_code ignored; std::filesystem::remove_all(path, ignored); }
  } cleanup{root};
  try {
    const auto path = root / "valid.mdlake";
    write(path, fixture());
    mdedge::LakeDataset dataset(path.string());
    require(dataset.rows() == 10 && dataset.features() == 2, "Dimensions differ");
    const auto batches = dataset.batches(2, 2);
    require(batches.size() == 4 && batches[1][0] == 8.5F, "Batch views differ");
    require(batches[1].data() == batches[0].data() + 4, "Batch views copied data");
    require(dataset.batches(3).size() == 3, "Partial batch must not be padded");
    rejects([&] { (void)dataset.batches(0); });
    rejects([&] { (void)dataset.batches(1, 11); });
    const auto fingerprint = dataset.validate_and_fingerprint(2, 6);
    require(fingerprint.size() == 16, "Missing workload fingerprint");
    require(fingerprint != dataset.validate_and_fingerprint(0, 6), "Content identity ignores data");

    IdentityEngine engine;
    mdedge::RunOptions options;
    options.input_file = path.string();
    options.input_size = 2;
    options.batch_size = 2;
    options.iterations = 3;
    options.warmup = 1;
    options.input_row_offset = 2;
    options.verify_identity = true;
    auto stats = mdedge::run_benchmark(engine, options);
    require(stats.validation && stats.validation->passed && stats.validation->compared_elements == 12,
            "Replay identity validation failed");
    require(engine.first_values == std::vector<float>{4.5F, 4.5F, 8.5F, 12.5F},
            "Warmup/measured replay rows changed or repeated");
    require(stats.input_fingerprint == fingerprint && stats.input_source == "lake-f32-v1",
            "Missing replay provenance");
    options.iterations = 5;
    rejects([&] { (void)mdedge::run_benchmark(engine, options); });
    options.iterations = 3;
    options.input_size = 3;
    rejects([&] { (void)mdedge::run_benchmark(engine, options); });

    const auto bad = root / "bad.mdlake";
    for (const auto kind : {0, 1, 2, 3, 4, 5}) {
      auto data = fixture();
      if (kind == 0) data.resize(10);
      if (kind == 1) data[0] = 'x';
      if (kind == 2) data.pop_back();
      if (kind == 3) put64(data, 16, std::numeric_limits<uint64_t>::max());
      if (kind == 4) data[32] = 1;
      if (kind == 5) put64(data, 8, 0);
      write(bad, data);
      rejects([&] { mdedge::LakeDataset invalid(bad.string()); });
    }
    auto nonfinite = fixture();
    const auto nan = std::bit_cast<uint32_t>(std::numeric_limits<float>::quiet_NaN());
    for (size_t i = 0; i < 4; ++i) nonfinite[64 + i] = static_cast<char>((nan >> (8 * i)) & 0xffU);
    write(bad, nonfinite);
    mdedge::LakeDataset invalid_values(bad.string());
    rejects([&] { (void)invalid_values.validate_and_fingerprint(0, 1); });
    std::cout << "Lake replay: views, row order, bounds, malformed data, identity PASS\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
