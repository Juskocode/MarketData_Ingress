#pragma once

#include <cstddef>
#include <memory>
#include <ranges>
#include <span>
#include <stdexcept>
#include <string>

namespace mdedge {

// Read-only, row-major FP32 replay. Keep the file immutable while views exist.
class LakeDataset {
public:
  explicit LakeDataset(const std::string& path);
  ~LakeDataset();
  LakeDataset(const LakeDataset&) = delete;
  LakeDataset& operator=(const LakeDataset&) = delete;

  [[nodiscard]] size_t rows() const;
  [[nodiscard]] size_t features() const;
  [[nodiscard]] const char* storage_kind() const;
  [[nodiscard]] std::string validate_and_fingerprint(size_t row_offset, size_t row_count) const;

  // Views borrow this dataset. Partial final batches are excluded, never padded.
  [[nodiscard]] auto batches(size_t batch_size, size_t row_offset = 0) const {
    if (batch_size == 0 || row_offset > rows()) {
      throw std::invalid_argument("Invalid replay batch size or row offset");
    }
    const size_t count = (rows() - row_offset) / batch_size;
    return std::views::iota(size_t{0}, count) |
        std::views::transform([this, batch_size, row_offset](size_t i) {
          return values().subspan((row_offset + i * batch_size) * features(),
                                  batch_size * features());
        });
  }

private:
  struct Storage;
  std::unique_ptr<Storage> storage_;
  [[nodiscard]] std::span<const float> values() const;
};

} // namespace mdedge
