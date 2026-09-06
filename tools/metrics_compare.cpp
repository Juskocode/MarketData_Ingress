#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

struct Span {
  std::size_t open;
  std::size_t close;
};

class JsonReader {
 public:
  explicit JsonReader(const std::string& text) : text_(text) {}

  Span root_object() const {
    const std::size_t start = skip_space(0);
    const Span root = object_at(start);
    if (skip_space(root.close + 1) != text_.size()) {
      throw std::runtime_error("unexpected content after the JSON document");
    }
    return root;
  }

  std::size_t required(const Span object, const std::string_view key) const {
    const auto value = find(object, key);
    if (!value) {
      throw std::runtime_error("missing JSON field: " + std::string(key));
    }
    return *value;
  }

  Span object_at(const std::size_t position) const {
    const std::size_t start = skip_space(position);
    if (start >= text_.size() || text_[start] != '{') {
      throw std::runtime_error("expected a JSON object");
    }
    return Span{start, matching_delimiter(start)};
  }

  bool is_null(const std::size_t position) const {
    const std::size_t start = skip_space(position);
    return text_.compare(start, 4, "null") == 0;
  }

  std::string string_at(const std::size_t position) const {
    return parse_string(skip_space(position)).first;
  }

  bool bool_at(const std::size_t position) const {
    const std::size_t start = skip_space(position);
    if (text_.compare(start, 4, "true") == 0) {
      return true;
    }
    if (text_.compare(start, 5, "false") == 0) {
      return false;
    }
    throw std::runtime_error("expected a JSON boolean");
  }

  double number_at(const std::size_t position) const {
    const std::size_t start = skip_space(position);
    const std::size_t end = skip_value(start);
    const std::string token = text_.substr(start, end - start);
    char* parsed_end = nullptr;
    errno = 0;
    const double value = std::strtod(token.c_str(), &parsed_end);
    if (errno == ERANGE || parsed_end != token.c_str() + token.size() ||
        !std::isfinite(value)) {
      throw std::runtime_error("expected a finite JSON number");
    }
    return value;
  }

 private:
  std::optional<std::size_t> find(const Span object,
                                  const std::string_view wanted) const {
    std::size_t position = skip_space(object.open + 1);
    while (position < object.close) {
      const auto [key, after_key] = parse_string(position);
      position = skip_space(after_key);
      if (position >= object.close || text_[position] != ':') {
        throw std::runtime_error("expected ':' after a JSON object key");
      }
      const std::size_t value = skip_space(position + 1);
      if (key == wanted) {
        return value;
      }
      position = skip_space(skip_value(value));
      if (position == object.close) {
        break;
      }
      if (text_[position] != ',') {
        throw std::runtime_error("expected ',' between JSON object fields");
      }
      position = skip_space(position + 1);
    }
    return std::nullopt;
  }

  std::size_t skip_space(std::size_t position) const {
    while (position < text_.size()) {
      const char value = text_[position];
      if (value != ' ' && value != '\n' && value != '\r' && value != '\t') {
        break;
      }
      ++position;
    }
    return position;
  }

  std::pair<std::string, std::size_t> parse_string(
      const std::size_t position) const {
    if (position >= text_.size() || text_[position] != '"') {
      throw std::runtime_error("expected a JSON string");
    }

    std::string value;
    for (std::size_t cursor = position + 1; cursor < text_.size(); ++cursor) {
      const char current = text_[cursor];
      if (current == '"') {
        return {value, cursor + 1};
      }
      if (current != '\\') {
        value.push_back(current);
        continue;
      }
      if (++cursor >= text_.size()) {
        throw std::runtime_error("unterminated JSON escape sequence");
      }
      switch (text_[cursor]) {
        case '"': value.push_back('"'); break;
        case '\\': value.push_back('\\'); break;
        case '/': value.push_back('/'); break;
        case 'b': value.push_back('\b'); break;
        case 'f': value.push_back('\f'); break;
        case 'n': value.push_back('\n'); break;
        case 'r': value.push_back('\r'); break;
        case 't': value.push_back('\t'); break;
        case 'u':
          if (cursor + 4 >= text_.size()) {
            throw std::runtime_error("incomplete JSON unicode escape");
          }
          for (int offset = 1; offset <= 4; ++offset) {
            const char digit = text_[cursor + static_cast<std::size_t>(offset)];
            const bool hexadecimal =
                (digit >= '0' && digit <= '9') ||
                (digit >= 'a' && digit <= 'f') ||
                (digit >= 'A' && digit <= 'F');
            if (!hexadecimal) {
              throw std::runtime_error("invalid JSON unicode escape");
            }
          }
          value.append(text_, cursor - 1, 6);
          cursor += 4;
          break;
        default: throw std::runtime_error("invalid JSON escape sequence");
      }
    }
    throw std::runtime_error("unterminated JSON string");
  }

  std::size_t matching_delimiter(const std::size_t position) const {
    const char opening = text_[position];
    if (opening != '{' && opening != '[') {
      throw std::runtime_error("expected a JSON container");
    }

    std::vector<char> expected;
    expected.push_back(opening == '{' ? '}' : ']');
    for (std::size_t cursor = position + 1; cursor < text_.size(); ++cursor) {
      if (text_[cursor] == '"') {
        cursor = parse_string(cursor).second - 1;
        continue;
      }
      if (text_[cursor] == '{') {
        expected.push_back('}');
      } else if (text_[cursor] == '[') {
        expected.push_back(']');
      } else if (text_[cursor] == '}' || text_[cursor] == ']') {
        if (expected.empty() || text_[cursor] != expected.back()) {
          throw std::runtime_error("mismatched JSON container delimiter");
        }
        expected.pop_back();
        if (expected.empty()) {
          return cursor;
        }
      }
    }
    throw std::runtime_error("unterminated JSON container");
  }

  std::size_t skip_value(const std::size_t position) const {
    const std::size_t start = skip_space(position);
    if (start >= text_.size()) {
      throw std::runtime_error("missing JSON value");
    }
    if (text_[start] == '"') {
      return parse_string(start).second;
    }
    if (text_[start] == '{' || text_[start] == '[') {
      return matching_delimiter(start) + 1;
    }

    std::size_t cursor = start;
    while (cursor < text_.size()) {
      const char current = text_[cursor];
      if (current == ',' || current == '}' || current == ']' || current == ' ' ||
          current == '\n' || current == '\r' || current == '\t') {
        break;
      }
      ++cursor;
    }
    if (cursor == start) {
      throw std::runtime_error("missing JSON scalar value");
    }
    return cursor;
  }

  const std::string& text_;
};

struct RunMetrics {
  double latency_us;
  std::string backend;
  std::string device;
  std::uint64_t input_size;
  std::uint64_t batch_size;
  bool cuda_graph;
};

std::uint64_t exact_unsigned(const JsonReader& reader,
                             const std::size_t position,
                             const std::string_view field) {
  const double value = reader.number_at(position);
  if (value < 0.0 || value > static_cast<double>(std::numeric_limits<std::uint64_t>::max()) ||
      std::floor(value) != value) {
    throw std::runtime_error("field must be an unsigned integer: " +
                             std::string(field));
  }
  return static_cast<std::uint64_t>(value);
}

RunMetrics parse_metrics(const std::string& document,
                         const std::string_view scope,
                         const std::string_view metric) {
  JsonReader reader(document);
  const Span root = reader.root_object();
  const Span latency = reader.object_at(reader.required(root, "latency_us"));
  const std::string_view scope_key = scope == "e2e" ? "end_to_end" : "device_compute";
  const std::size_t scope_value = reader.required(latency, scope_key);
  if (reader.is_null(scope_value)) {
    throw std::runtime_error("requested latency scope is null: " +
                             std::string(scope_key));
  }
  const Span distribution = reader.object_at(scope_value);
  const Span environment = reader.object_at(reader.required(root, "environment"));

  return RunMetrics{
      reader.number_at(reader.required(distribution, metric)),
      reader.string_at(reader.required(root, "backend")),
      reader.string_at(reader.required(environment, "device")),
      exact_unsigned(reader, reader.required(root, "input_size"), "input_size"),
      exact_unsigned(reader, reader.required(root, "batch_size"), "batch_size"),
      reader.bool_at(reader.required(root, "cuda_graph")),
  };
}

std::string read_file(const std::string& path) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream) {
    throw std::runtime_error("cannot open metrics file: " + path);
  }
  std::ostringstream contents;
  contents << stream.rdbuf();
  if (!stream.good() && !stream.eof()) {
    throw std::runtime_error("cannot read metrics file: " + path);
  }
  return contents.str();
}

double parse_nonnegative(const std::string& text, const std::string_view option) {
  char* end = nullptr;
  errno = 0;
  const double value = std::strtod(text.c_str(), &end);
  if (errno == ERANGE || end != text.c_str() + text.size() ||
      !std::isfinite(value) || value < 0.0) {
    throw std::runtime_error(std::string(option) +
                             " requires a finite non-negative number");
  }
  return value;
}

std::vector<std::string> context_differences(const RunMetrics& baseline,
                                             const RunMetrics& candidate) {
  std::vector<std::string> differences;
  if (baseline.backend != candidate.backend) differences.emplace_back("backend");
  if (baseline.device != candidate.device) differences.emplace_back("device");
  if (baseline.input_size != candidate.input_size) differences.emplace_back("input_size");
  if (baseline.batch_size != candidate.batch_size) differences.emplace_back("batch_size");
  if (baseline.cuda_graph != candidate.cuda_graph) differences.emplace_back("cuda_graph");
  return differences;
}

std::string join(const std::vector<std::string>& values) {
  std::ostringstream output;
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (index != 0) output << ',';
    output << values[index];
  }
  return output.str();
}

void print_usage(const char* executable) {
  std::cout
      << "Usage: " << executable << " --baseline <metrics.json> --candidate <metrics.json> [options]\n\n"
      << "Options:\n"
      << "  --scope e2e|device              Distribution to compare (default: e2e)\n"
      << "  --metric mean|p50|p90|p95|p99|max\n"
      << "                                  Statistic to compare (default: p99)\n"
      << "  --max-regression-percent <N>    Allowed relative increase (default: 5)\n"
      << "  --absolute-tolerance-us <N>     Additional absolute budget (default: 0)\n"
      << "  --allow-context-change          Skip backend/device/shape/graph equality gate\n"
      << "  --self-test                     Run parser and comparison unit tests\n"
      << "  --help                          Show this help\n";
}

std::string sample_document(const double e2e_p99,
                            const std::string_view device_distribution,
                            const std::string_view backend = "tensorrt",
                            const std::uint64_t input_size = 8) {
  std::ostringstream document;
  document << "{\"backend\":\"" << backend
           << "\",\"environment\":{\"device\":\"gpu-0\"},"
           << "\"input_size\":" << input_size
           << ",\"batch_size\":1,\"cuda_graph\":true,"
           << "\"latency_us\":{\"end_to_end\":{\"mean\":0.5,\"p50\":0.6,"
           << "\"p90\":0.7,\"p95\":0.75,\"p99\":" << e2e_p99
           << ",\"max\":0.9},\"device_compute\":" << device_distribution << "}}";
  return document.str();
}

int self_test() {
  const RunMetrics baseline = parse_metrics(
      sample_document(0.8, "{\"mean\":0.2,\"p99\":0.3}"), "e2e", "p99");
  const RunMetrics candidate = parse_metrics(
      sample_document(0.84, "{\"mean\":0.2,\"p99\":0.31}"), "e2e", "p99");
  if (std::abs(baseline.latency_us - 0.8) > 1e-12 ||
      std::abs(candidate.latency_us - 0.84) > 1e-12 ||
      !context_differences(baseline, candidate).empty()) {
    throw std::runtime_error("self-test failed: valid comparison");
  }

  const RunMetrics changed = parse_metrics(
      sample_document(0.7, "{\"mean\":0.2,\"p99\":0.3}", "tensorrt", 16),
      "e2e", "p99");
  if (context_differences(baseline, changed) != std::vector<std::string>{"input_size"}) {
    throw std::runtime_error("self-test failed: context mismatch");
  }

  bool rejected_null_device = false;
  try {
    static_cast<void>(parse_metrics(sample_document(0.8, "null"), "device", "p99"));
  } catch (const std::runtime_error&) {
    rejected_null_device = true;
  }
  if (!rejected_null_device) {
    throw std::runtime_error("self-test failed: null device distribution accepted");
  }

  const double limit = baseline.latency_us * 1.05;
  if (candidate.latency_us > limit + 1e-12 || !(0.85 > limit)) {
    throw std::runtime_error("self-test failed: regression budget");
  }

  std::cout << "metrics comparator self-test: pass\n";
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    std::string baseline_path;
    std::string candidate_path;
    std::string scope = "e2e";
    std::string metric = "p99";
    double max_regression_percent = 5.0;
    double absolute_tolerance_us = 0.0;
    bool allow_context_change = false;

    const auto value_after = [&](const int index, const std::string_view option) {
      if (index + 1 >= argc) {
        throw std::runtime_error(std::string(option) + " requires a value");
      }
      return std::string(argv[index + 1]);
    };

    for (int index = 1; index < argc; ++index) {
      const std::string argument = argv[index];
      if (argument == "--help") {
        print_usage(argv[0]);
        return 0;
      }
      if (argument == "--self-test") {
        return self_test();
      }
      if (argument == "--baseline") {
        baseline_path = value_after(index, argument);
        ++index;
      } else if (argument == "--candidate") {
        candidate_path = value_after(index, argument);
        ++index;
      } else if (argument == "--scope") {
        scope = value_after(index, argument);
        ++index;
      } else if (argument == "--metric") {
        metric = value_after(index, argument);
        ++index;
      } else if (argument == "--max-regression-percent") {
        max_regression_percent = parse_nonnegative(value_after(index, argument), argument);
        ++index;
      } else if (argument == "--absolute-tolerance-us") {
        absolute_tolerance_us = parse_nonnegative(value_after(index, argument), argument);
        ++index;
      } else if (argument == "--allow-context-change") {
        allow_context_change = true;
      } else {
        throw std::runtime_error("unknown option: " + argument);
      }
    }

    if (baseline_path.empty() || candidate_path.empty()) {
      throw std::runtime_error("--baseline and --candidate are required");
    }
    if (scope != "e2e" && scope != "device") {
      throw std::runtime_error("--scope must be e2e or device");
    }
    const std::vector<std::string> valid_metrics = {"mean", "p50", "p90", "p95", "p99", "max"};
    bool valid_metric = false;
    for (const auto& value : valid_metrics) {
      if (metric == value) valid_metric = true;
    }
    if (!valid_metric) {
      throw std::runtime_error("unsupported metric: " + metric);
    }

    const RunMetrics baseline = parse_metrics(read_file(baseline_path), scope, metric);
    const RunMetrics candidate = parse_metrics(read_file(candidate_path), scope, metric);
    const std::vector<std::string> differences = context_differences(baseline, candidate);
    if (!allow_context_change && !differences.empty()) {
      std::cerr << "incomparable benchmark context: " << join(differences) << '\n';
      return 4;
    }

    const double limit = baseline.latency_us * (1.0 + max_regression_percent / 100.0) +
                         absolute_tolerance_us;
    const bool passed = candidate.latency_us <= limit;
    const std::optional<double> relative_change =
        baseline.latency_us == 0.0
            ? std::nullopt
            : std::optional<double>((candidate.latency_us / baseline.latency_us - 1.0) * 100.0);

    std::cout << std::fixed << std::setprecision(9)
              << "{\"status\":\"" << (passed ? "pass" : "regression")
              << "\",\"scope\":\"" << scope
              << "\",\"metric\":\"" << metric
              << "\",\"baseline_us\":" << baseline.latency_us
              << ",\"candidate_us\":" << candidate.latency_us
              << ",\"limit_us\":" << limit
              << ",\"relative_change_percent\":";
    if (relative_change) {
      std::cout << *relative_change;
    } else {
      std::cout << "null";
    }
    std::cout << ",\"max_regression_percent\":" << max_regression_percent
              << ",\"absolute_tolerance_us\":" << absolute_tolerance_us
              << ",\"context_override\":" << (allow_context_change ? "true" : "false")
              << "}\n";
    return passed ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << "metrics comparison error: " << error.what() << '\n';
    return 2;
  }
}
