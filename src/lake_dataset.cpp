#include "marketdata_infer/lake_dataset.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <fstream>
#include <limits>
#include <iomanip>
#include <sstream>
#include <system_error>
#include <vector>

#if defined(__unix__) || defined(__APPLE__)
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace mdedge {
namespace {
constexpr size_t header_bytes = 64;
constexpr std::array<char, 8> magic{'M', 'D', 'L', 'A', 'K', 'E', '0', '1'};

uint64_t read_u64(const std::byte* address) {
  uint64_t value = 0;
  std::memcpy(&value, address, sizeof(value));
  if constexpr (std::endian::native == std::endian::big) value = std::byteswap(value);
  return value;
}
} // namespace

struct LakeDataset::Storage {
  size_t row_count = 0;
  size_t feature_count = 0;
  size_t bytes = 0;
  const std::byte* data = nullptr;
#if defined(__unix__) || defined(__APPLE__)
  void* mapping = MAP_FAILED;
  ~Storage() { if (mapping != MAP_FAILED) munmap(mapping, bytes); }
#else
  std::vector<float> buffer; // Aligned portable fallback; includes the header.
#endif
};

LakeDataset::LakeDataset(const std::string& path) : storage_(std::make_unique<Storage>()) {
  static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559);
  if constexpr (std::endian::native != std::endian::little) {
    throw std::runtime_error("FP32 replay requires a little-endian host");
  }
  auto& s = *storage_;
#if defined(__unix__) || defined(__APPLE__)
  const int fd = open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd < 0) throw std::system_error(errno, std::generic_category(), "Open replay file");
  struct stat info{};
  if (fstat(fd, &info) != 0) {
    const int error = errno;
    close(fd);
    throw std::system_error(error, std::generic_category(), "Stat replay file");
  }
  if (!S_ISREG(info.st_mode) || info.st_size < static_cast<off_t>(header_bytes) ||
      static_cast<uintmax_t>(info.st_size) > std::numeric_limits<size_t>::max()) {
    close(fd);
    throw std::runtime_error("Replay must be a regular file with a complete header");
  }
  s.bytes = static_cast<size_t>(info.st_size);
  s.mapping = mmap(nullptr, s.bytes, PROT_READ, MAP_PRIVATE, fd, 0);
  const int map_error = errno;
  close(fd);
  if (s.mapping == MAP_FAILED) {
    throw std::system_error(map_error, std::generic_category(), "Map replay file");
  }
  s.data = static_cast<const std::byte*>(s.mapping);
#else
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  const auto size = file.tellg();
  if (!file || size < static_cast<std::streamoff>(header_bytes) ||
      static_cast<uintmax_t>(size) > std::numeric_limits<size_t>::max() ||
      size % static_cast<std::streamoff>(sizeof(float)) != 0) {
    throw std::runtime_error("Replay must have a complete FP32 header and payload");
  }
  s.bytes = static_cast<size_t>(size);
  s.buffer.resize(s.bytes / sizeof(float));
  file.seekg(0);
  if (!file.read(reinterpret_cast<char*>(s.buffer.data()), size)) {
    throw std::runtime_error("Could not read replay file");
  }
  s.data = reinterpret_cast<const std::byte*>(s.buffer.data());
#endif
  if (std::memcmp(s.data, magic.data(), magic.size()) != 0) {
    throw std::runtime_error("Unrecognized replay format (expected MDLAKE01)");
  }
  const auto rows = read_u64(s.data + 8);
  const auto features = read_u64(s.data + 16);
  const auto payload = read_u64(s.data + 24);
  if (rows == 0 || features == 0 || rows > std::numeric_limits<size_t>::max() ||
      features > std::numeric_limits<size_t>::max() ||
      features > (std::numeric_limits<size_t>::max() - header_bytes) / sizeof(float) ||
      rows > (std::numeric_limits<size_t>::max() - header_bytes) / sizeof(float) / features ||
      payload != rows * features * sizeof(float) || payload != s.bytes - header_bytes) {
    throw std::runtime_error("Replay dimensions/payload are empty, truncated or overflow");
  }
  if (!std::ranges::all_of(std::span(s.data + 32, header_bytes - 32),
                          [](std::byte value) { return value == std::byte{0}; })) {
    throw std::runtime_error("Replay reserved flags must be zero");
  }
  s.row_count = static_cast<size_t>(rows);
  s.feature_count = static_cast<size_t>(features);
}

LakeDataset::~LakeDataset() = default;
size_t LakeDataset::rows() const { return storage_->row_count; }
size_t LakeDataset::features() const { return storage_->feature_count; }
std::span<const float> LakeDataset::values() const {
  return {reinterpret_cast<const float*>(storage_->data + header_bytes), rows() * features()};
}
const char* LakeDataset::storage_kind() const {
#if defined(__unix__) || defined(__APPLE__)
  return "mmap";
#else
  return "buffered";
#endif
}
std::string LakeDataset::validate_and_fingerprint(size_t offset, size_t count) const {
  if (offset > rows() || count > rows() - offset) {
    throw std::invalid_argument("Replay validation range exceeds dataset");
  }
  // Non-cryptographic content identity for reproducible workload comparisons.
  uint64_t hash = 14695981039346656037ULL;
  for (float value : values().subspan(offset * features(), count * features())) {
    if (!std::isfinite(value)) throw std::runtime_error("Replay contains non-finite FP32 input");
    const auto bits = std::bit_cast<uint32_t>(value);
    for (const auto shift : {0, 8, 16, 24}) {
      hash ^= (bits >> shift) & 0xffU;
      hash *= 1099511628211ULL;
    }
  }
  std::ostringstream out;
  out << std::hex << std::setw(16) << std::setfill('0') << hash;
  return out.str();
}
} // namespace mdedge
