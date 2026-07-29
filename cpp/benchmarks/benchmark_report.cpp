#include "benchmark_report.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <initializer_list>
#include <limits>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "lunar_path_planner/v3/codec/jcs_canonicalizer.hpp"
#include "lunar_path_planner/v3/crypto/sha256.hpp"

namespace lunar::planning::v3 {
namespace {

using Nanoseconds = std::chrono::nanoseconds;
using Json = nlohmann::json;

constexpr std::int64_t kP95LatencyTargetNanoseconds = 1'000'000'000LL;
constexpr std::uint64_t kMinimumBenchmarkMemoryBytes = 1'073'741'824ULL;
constexpr std::uint32_t kMaximumSchemaRevision = 2'147'483'647U;

struct PlatformAccumulator final {
  std::vector<Nanoseconds> measured;
  std::vector<Nanoseconds> cold_start;
  std::array<std::vector<Nanoseconds>, kCanonicalBenchmarkModules.size()>
      modules;
  bool result_independence_check_passed{true};
};

[[nodiscard]] Error Invalid(std::string field_path,
                            std::string message) {
  return Error{
      .code = ErrorCode::kInvalidArgument,
      .field_path = std::move(field_path),
      .message = std::move(message),
  };
}

[[nodiscard]] bool HasExactKeys(
    const Json& value,
    std::initializer_list<std::string_view> keys) {
  if (!value.is_object() || value.size() != keys.size()) {
    return false;
  }
  return std::all_of(
      keys.begin(), keys.end(),
      [&](const std::string_view key) {
        return value.contains(std::string{key});
      });
}

[[nodiscard]] Result<std::string> ReadString(
    const Json& object, std::string_view key,
    std::string field_path) {
  const auto iterator = object.find(std::string{key});
  if (iterator == object.end() || !iterator->is_string()) {
    return Invalid(std::move(field_path),
                   "field must be a JSON string");
  }
  return iterator->get<std::string>();
}

[[nodiscard]] Result<std::uint64_t> ReadUnsigned(
    const Json& object, std::string_view key,
    std::string field_path) {
  const auto iterator = object.find(std::string{key});
  if (iterator == object.end() ||
      (!iterator->is_number_unsigned() &&
       !iterator->is_number_integer())) {
    return Invalid(std::move(field_path),
                   "field must be a non-negative integer");
  }
  if (iterator->is_number_integer() &&
      iterator->get<std::int64_t>() < 0) {
    return Invalid(std::move(field_path),
                   "field must be a non-negative integer");
  }
  return iterator->get<std::uint64_t>();
}

[[nodiscard]] Result<std::size_t> ReadSize(
    const Json& object, std::string_view key,
    std::string field_path) {
  auto value =
      ReadUnsigned(object, key, std::move(field_path));
  if (!IsOk(value)) {
    return std::get<Error>(value);
  }
  const auto raw = std::get<std::uint64_t>(value);
  if (raw > std::numeric_limits<std::size_t>::max()) {
    return Invalid(std::string{key},
                   "integer exceeds size_t");
  }
  return static_cast<std::size_t>(raw);
}

[[nodiscard]] Result<bool> ReadBool(
    const Json& object, std::string_view key,
    std::string field_path) {
  const auto iterator = object.find(std::string{key});
  if (iterator == object.end() || !iterator->is_boolean()) {
    return Invalid(std::move(field_path),
                   "field must be a JSON boolean");
  }
  return iterator->get<bool>();
}

[[nodiscard]] bool IsAsciiAlphaNumeric(char value) noexcept {
  return (value >= 'A' && value <= 'Z') ||
         (value >= 'a' && value <= 'z') ||
         (value >= '0' && value <= '9');
}

[[nodiscard]] bool IsValidIdentifier(
    std::string_view identifier) noexcept {
  if (identifier.empty() || identifier.size() > 128U ||
      !IsAsciiAlphaNumeric(identifier.front())) {
    return false;
  }
  return std::all_of(identifier.begin(), identifier.end(), [](char value) {
    return IsAsciiAlphaNumeric(value) || value == '.' || value == '_' ||
           value == ':' || value == '/' || value == '-';
  });
}

[[nodiscard]] bool IsLowerHexDigest(
    std::string_view digest) noexcept {
  return digest.size() == 64U &&
         std::all_of(digest.begin(), digest.end(), [](char value) {
           return (value >= '0' && value <= '9') ||
                  (value >= 'a' && value <= 'f');
         });
}

[[nodiscard]] bool IsValidContentRef(const ContentRef& ref) noexcept {
  return IsValidIdentifier(ref.id) && ref.revision >= 1U &&
         ref.revision <= kMaximumSchemaRevision &&
         IsLowerHexDigest(ref.content_hash);
}

[[nodiscard]] Result<ContentRef> DecodeContentRef(
    const Json& value, std::string field_path) {
  if (!HasExactKeys(
          value, {"id", "revision", "content_hash"})) {
    return Invalid(std::move(field_path),
                   "ContentRef has missing or unknown fields");
  }
  auto id =
      ReadString(value, "id", field_path + ".id");
  if (!IsOk(id)) {
    return std::get<Error>(id);
  }
  auto revision =
      ReadUnsigned(value, "revision", field_path + ".revision");
  if (!IsOk(revision)) {
    return std::get<Error>(revision);
  }
  auto hash = ReadString(
      value, "content_hash", field_path + ".content_hash");
  if (!IsOk(hash)) {
    return std::get<Error>(hash);
  }
  const auto raw_revision = std::get<std::uint64_t>(revision);
  if (raw_revision > std::numeric_limits<std::uint32_t>::max()) {
    return Invalid(field_path + ".revision",
                   "revision exceeds uint32");
  }
  ContentRef ref{
      .id = std::get<std::string>(std::move(id)),
      .revision = static_cast<std::uint32_t>(raw_revision),
      .content_hash = std::get<std::string>(std::move(hash)),
  };
  if (!IsValidContentRef(ref)) {
    return Invalid(std::move(field_path),
                   "ContentRef is not canonical");
  }
  return ref;
}

[[nodiscard]] Result<PlatformType> DecodePlatformType(
    const Json& value, std::string field_path) {
  if (!value.is_string()) {
    return Invalid(std::move(field_path),
                   "platform_type must be a string");
  }
  const std::string platform = value.get<std::string>();
  if (platform == "WHEELED") {
    return PlatformType::kWheeled;
  }
  if (platform == "LEGGED") {
    return PlatformType::kLegged;
  }
  if (platform == "HOPPER") {
    return PlatformType::kHopper;
  }
  return Invalid(std::move(field_path),
                 "unknown platform_type");
}

[[nodiscard]] std::size_t PlatformIndex(
    PlatformType platform_type) noexcept {
  switch (platform_type) {
    case PlatformType::kWheeled:
      return 0U;
    case PlatformType::kLegged:
      return 1U;
    case PlatformType::kHopper:
      return 2U;
  }
  return kCanonicalBenchmarkPlatforms.size();
}

[[nodiscard]] std::string_view PlatformName(
    PlatformType platform_type) noexcept {
  switch (platform_type) {
    case PlatformType::kWheeled:
      return "WHEELED";
    case PlatformType::kLegged:
      return "LEGGED";
    case PlatformType::kHopper:
      return "HOPPER";
  }
  return "UNKNOWN";
}

[[nodiscard]] std::size_t ModuleIndex(
    BenchmarkModuleName module_name) noexcept {
  switch (module_name) {
    case BenchmarkModuleName::kSchemaAndContractValidation:
      return 0U;
    case BenchmarkModuleName::kSafeProjectionAndTerminalResolution:
      return 1U;
    case BenchmarkModuleName::kPlatformSearch:
      return 2U;
    case BenchmarkModuleName::kContinuousReferenceGeneration:
      return 3U;
    case BenchmarkModuleName::kContinuousValidation:
      return 4U;
    case BenchmarkModuleName::kBundlePackagingAndSerialization:
      return 5U;
  }
  return kCanonicalBenchmarkModules.size();
}

[[nodiscard]] bool IsCanonicalTimingBoundary(
    const BenchmarkTimingBoundary& boundary) noexcept {
  constexpr std::array<BenchmarkExcludedActivity, 3U> kExcluded{
      BenchmarkExcludedActivity::kRawSensorMapping,
      BenchmarkExcludedActivity::kProcessStartup,
      BenchmarkExcludedActivity::kRemoteRpcOrDiskIo,
  };
  return boundary.starts_before ==
             BenchmarkTimingStart::kRequestSchemaAndSnapshotValidation &&
         boundary.ends_after ==
             BenchmarkTimingEnd::kPlanningResponseSerialization &&
         boundary.included == kCanonicalBenchmarkModules &&
         boundary.excluded == kExcluded &&
         boundary.cold_start_reported_separately;
}

[[nodiscard]] bool IsValidBuildType(
    BenchmarkBuildType build_type) noexcept {
  switch (build_type) {
    case BenchmarkBuildType::kRelease:
    case BenchmarkBuildType::kRelWithDebInfo:
      return true;
  }
  return false;
}

[[nodiscard]] Result<std::array<const BenchmarkPlatformSuite*, 3U>>
ValidateProfileImpl(const BenchmarkProfile& profile) {
  if (!IsValidContentRef(profile.content_ref)) {
    return Invalid("benchmark_profile.content_ref",
                   "benchmark profile content_ref is not canonical");
  }
  if (!IsValidContentRef(profile.content.algorithm_config_ref)) {
    return Invalid("benchmark_profile.content.algorithm_config_ref",
                   "algorithm config content_ref is not canonical");
  }
  const auto& hardware = profile.content.hardware;
  if (hardware.cpu_model.empty() || hardware.cpu_model.size() > 256U ||
      hardware.physical_core_count < 1U ||
      hardware.logical_core_count < 1U ||
      hardware.logical_core_count < hardware.physical_core_count ||
      hardware.memory_bytes < kMinimumBenchmarkMemoryBytes ||
      hardware.os_name.empty() || hardware.os_name.size() > 128U ||
      hardware.os_version.empty() || hardware.os_version.size() > 128U ||
      hardware.compiler_id.empty() ||
      hardware.compiler_id.size() > 128U ||
      hardware.compiler_version.empty() ||
      hardware.compiler_version.size() > 128U ||
      !IsValidBuildType(hardware.build_type)) {
    return Invalid("benchmark_profile.content.hardware",
                   "hardware profile violates the benchmark schema");
  }
  if (profile.content.planner_thread_count < 1U) {
    return Invalid("benchmark_profile.content.planner_thread_count",
                   "planner thread count must be positive");
  }
  if (profile.content.measured_sample_count_per_platform < 1U) {
    return Invalid(
        "benchmark_profile.content.measured_sample_count_per_platform",
        "measured sample count must be positive");
  }
  if (profile.content.p95_latency_target.value.count() !=
      kP95LatencyTargetNanoseconds) {
    return Invalid("benchmark_profile.content.p95_latency_target_ns",
                   "P95 target must be the declared strict one-second bound");
  }
  if (!IsCanonicalTimingBoundary(profile.content.timing_boundary)) {
    return Invalid("benchmark_profile.content.timing_boundary",
                   "timing boundary does not match the declared API boundary");
  }
  if (!profile.content
           .planner_result_must_be_independent_of_latency_target) {
    return Invalid(
        "benchmark_profile.content."
        "planner_result_must_be_independent_of_latency_target",
        "planner-result independence must be declared");
  }
  if (profile.content.platform_suites.size() !=
      kCanonicalBenchmarkPlatforms.size()) {
    return Invalid("benchmark_profile.content.platform_suites",
                   "exactly three platform suites are required");
  }

  std::array<const BenchmarkPlatformSuite*, 3U> suites{};
  for (const auto& suite : profile.content.platform_suites) {
    const std::size_t index = PlatformIndex(suite.platform_type);
    if (index >= suites.size() || suites[index] != nullptr) {
      return Invalid("benchmark_profile.content.platform_suites",
                     "exactly one suite per platform is required");
    }
    if (!IsValidContentRef(suite.safety_capability_ref)) {
      return Invalid(
          "benchmark_profile.content.platform_suites."
          "safety_capability_ref",
          "suite capability content_ref is not canonical");
    }
    if (suite.scenario_mix.empty() || suite.scenario_mix.size() > 256U) {
      return Invalid(
          "benchmark_profile.content.platform_suites.scenario_mix",
          "scenario mix must contain between one and 256 entries");
    }
    for (const auto& scenario : suite.scenario_mix) {
      if (!IsValidContentRef(scenario.scenario_ref) ||
          !std::isfinite(scenario.weight) || scenario.weight <= 0.0) {
        return Invalid(
            "benchmark_profile.content.platform_suites.scenario_mix",
            "scenario refs must be canonical and weights finite positive");
      }
    }
    suites[index] = &suite;
  }
  if (std::any_of(suites.begin(), suites.end(),
                  [](const auto* suite) { return suite == nullptr; })) {
    return Invalid("benchmark_profile.content.platform_suites",
                   "exactly one suite per platform is required");
  }
  return suites;
}

[[nodiscard]] Result<bool> ValidateHeaderImpl(
    const BenchmarkReportHeader& header) {
  if (!IsValidIdentifier(header.report_id)) {
    return Invalid("benchmark_report_header.report_id",
                   "report id is not canonical");
  }
  if (header.report_revision < 1U ||
      header.report_revision > kMaximumSchemaRevision) {
    return Invalid("benchmark_report_header.report_revision",
                   "report revision is outside the schema range");
  }
  if (!IsValidIdentifier(header.generated_at.clock_id)) {
    return Invalid("benchmark_report_header.generated_at.clock_id",
                   "generated_at clock id is not canonical");
  }
  return true;
}

[[nodiscard]] Result<BenchmarkProfile> DecodeProfileImpl(
    std::string_view payload) {
  const Json document = Json::parse(payload);
  if (!HasExactKeys(
          document, {"schema_version", "content_ref", "content"})) {
    return Invalid("benchmark_profile",
                   "profile has missing or unknown top-level fields");
  }
  auto schema = ReadString(
      document, "schema_version",
      "benchmark_profile.schema_version");
  if (!IsOk(schema)) {
    return std::get<Error>(schema);
  }
  if (std::get<std::string>(schema) !=
      BenchmarkProfile::kSchemaVersion) {
    return Invalid("benchmark_profile.schema_version",
                   "unsupported benchmark profile schema");
  }
  auto content_ref = DecodeContentRef(
      document.at("content_ref"),
      "benchmark_profile.content_ref");
  if (!IsOk(content_ref)) {
    return std::get<Error>(content_ref);
  }

  const Json& content_json = document.at("content");
  if (!HasExactKeys(
          content_json,
          {"hardware",
           "algorithm_config_ref",
           "planner_thread_count",
           "warmup_sample_count",
           "measured_sample_count_per_platform",
           "p95_latency_target_ns",
           "platform_suites",
           "timing_boundary",
           "record_module_timings",
           "planner_result_must_be_independent_of_latency_target"})) {
    return Invalid("benchmark_profile.content",
                   "content has missing or unknown fields");
  }

  const Json& hardware_json = content_json.at("hardware");
  if (!HasExactKeys(
          hardware_json,
          {"cpu_model", "physical_core_count",
           "logical_core_count", "memory_bytes", "os_name",
           "os_version", "compiler_id", "compiler_version",
           "build_type"})) {
    return Invalid("benchmark_profile.content.hardware",
                   "hardware has missing or unknown fields");
  }
  auto cpu = ReadString(
      hardware_json, "cpu_model",
      "benchmark_profile.content.hardware.cpu_model");
  auto physical = ReadSize(
      hardware_json, "physical_core_count",
      "benchmark_profile.content.hardware.physical_core_count");
  auto logical = ReadSize(
      hardware_json, "logical_core_count",
      "benchmark_profile.content.hardware.logical_core_count");
  auto memory = ReadUnsigned(
      hardware_json, "memory_bytes",
      "benchmark_profile.content.hardware.memory_bytes");
  auto os_name = ReadString(
      hardware_json, "os_name",
      "benchmark_profile.content.hardware.os_name");
  auto os_version = ReadString(
      hardware_json, "os_version",
      "benchmark_profile.content.hardware.os_version");
  auto compiler = ReadString(
      hardware_json, "compiler_id",
      "benchmark_profile.content.hardware.compiler_id");
  auto compiler_version = ReadString(
      hardware_json, "compiler_version",
      "benchmark_profile.content.hardware.compiler_version");
  auto build = ReadString(
      hardware_json, "build_type",
      "benchmark_profile.content.hardware.build_type");
  for (const auto* result :
       {&cpu, &os_name, &os_version, &compiler,
        &compiler_version, &build}) {
    if (!IsOk(*result)) {
      return std::get<Error>(*result);
    }
  }
  if (!IsOk(physical)) {
    return std::get<Error>(physical);
  }
  if (!IsOk(logical)) {
    return std::get<Error>(logical);
  }
  if (!IsOk(memory)) {
    return std::get<Error>(memory);
  }
  BenchmarkBuildType build_type;
  const auto& build_string = std::get<std::string>(build);
  if (build_string == "RELEASE") {
    build_type = BenchmarkBuildType::kRelease;
  } else if (build_string == "RELWITHDEBINFO") {
    build_type = BenchmarkBuildType::kRelWithDebInfo;
  } else {
    return Invalid("benchmark_profile.content.hardware.build_type",
                   "unknown build type");
  }

  auto algorithm_ref = DecodeContentRef(
      content_json.at("algorithm_config_ref"),
      "benchmark_profile.content.algorithm_config_ref");
  if (!IsOk(algorithm_ref)) {
    return std::get<Error>(algorithm_ref);
  }
  auto planner_threads = ReadSize(
      content_json, "planner_thread_count",
      "benchmark_profile.content.planner_thread_count");
  auto warmups = ReadSize(
      content_json, "warmup_sample_count",
      "benchmark_profile.content.warmup_sample_count");
  auto measured = ReadSize(
      content_json, "measured_sample_count_per_platform",
      "benchmark_profile.content.measured_sample_count_per_platform");
  if (!IsOk(planner_threads)) {
    return std::get<Error>(planner_threads);
  }
  if (!IsOk(warmups)) {
    return std::get<Error>(warmups);
  }
  if (!IsOk(measured)) {
    return std::get<Error>(measured);
  }
  auto target = ReadString(
      content_json, "p95_latency_target_ns",
      "benchmark_profile.content.p95_latency_target_ns");
  if (!IsOk(target)) {
    return std::get<Error>(target);
  }
  if (std::get<std::string>(target) != "1000000000") {
    return Invalid(
        "benchmark_profile.content.p95_latency_target_ns",
        "P95 target must be the canonical one-second string");
  }

  const Json& suites_json = content_json.at("platform_suites");
  if (!suites_json.is_array()) {
    return Invalid("benchmark_profile.content.platform_suites",
                   "platform_suites must be an array");
  }
  std::vector<BenchmarkPlatformSuite> suites;
  suites.reserve(suites_json.size());
  for (std::size_t suite_index = 0U;
       suite_index < suites_json.size(); ++suite_index) {
    const Json& suite_json = suites_json[suite_index];
    const std::string suite_path =
        "benchmark_profile.content.platform_suites[" +
        std::to_string(suite_index) + "]";
    if (!HasExactKeys(
            suite_json,
            {"platform_type", "safety_capability_ref",
             "scenario_mix"})) {
      return Invalid(suite_path,
                     "suite has missing or unknown fields");
    }
    auto platform = DecodePlatformType(
        suite_json.at("platform_type"),
        suite_path + ".platform_type");
    if (!IsOk(platform)) {
      return std::get<Error>(platform);
    }
    auto capability_ref = DecodeContentRef(
        suite_json.at("safety_capability_ref"),
        suite_path + ".safety_capability_ref");
    if (!IsOk(capability_ref)) {
      return std::get<Error>(capability_ref);
    }
    const Json& scenarios_json = suite_json.at("scenario_mix");
    if (!scenarios_json.is_array()) {
      return Invalid(suite_path + ".scenario_mix",
                     "scenario_mix must be an array");
    }
    std::vector<BenchmarkScenarioEntry> scenarios;
    scenarios.reserve(scenarios_json.size());
    for (std::size_t scenario_index = 0U;
         scenario_index < scenarios_json.size(); ++scenario_index) {
      const Json& scenario_json =
          scenarios_json[scenario_index];
      const std::string scenario_path =
          suite_path + ".scenario_mix[" +
          std::to_string(scenario_index) + "]";
      if (!HasExactKeys(
              scenario_json, {"scenario_ref", "weight"})) {
        return Invalid(
            scenario_path,
            "scenario has missing or unknown fields");
      }
      auto scenario_ref = DecodeContentRef(
          scenario_json.at("scenario_ref"),
          scenario_path + ".scenario_ref");
      if (!IsOk(scenario_ref)) {
        return std::get<Error>(scenario_ref);
      }
      if (!scenario_json.at("weight").is_number()) {
        return Invalid(scenario_path + ".weight",
                       "scenario weight must be numeric");
      }
      scenarios.push_back(
          BenchmarkScenarioEntry{
              .scenario_ref =
                  std::get<ContentRef>(
                      std::move(scenario_ref)),
              .weight =
                  scenario_json.at("weight").get<double>(),
          });
    }
    suites.push_back(
        BenchmarkPlatformSuite{
            .platform_type =
                std::get<PlatformType>(platform),
            .safety_capability_ref =
                std::get<ContentRef>(
                    std::move(capability_ref)),
            .scenario_mix = std::move(scenarios),
        });
  }

  const Json& boundary_json =
      content_json.at("timing_boundary");
  if (!HasExactKeys(
          boundary_json,
          {"starts_before", "ends_after", "included", "excluded",
           "cold_start_reported_separately"})) {
    return Invalid(
        "benchmark_profile.content.timing_boundary",
        "timing boundary has missing or unknown fields");
  }
  auto starts = ReadString(
      boundary_json, "starts_before",
      "benchmark_profile.content.timing_boundary.starts_before");
  auto ends = ReadString(
      boundary_json, "ends_after",
      "benchmark_profile.content.timing_boundary.ends_after");
  auto cold_separate = ReadBool(
      boundary_json, "cold_start_reported_separately",
      "benchmark_profile.content.timing_boundary."
      "cold_start_reported_separately");
  if (!IsOk(starts)) {
    return std::get<Error>(starts);
  }
  if (!IsOk(ends)) {
    return std::get<Error>(ends);
  }
  if (!IsOk(cold_separate)) {
    return std::get<Error>(cold_separate);
  }
  constexpr std::array<std::string_view, 6U> kIncludedNames{
      "SCHEMA_AND_CONTRACT_VALIDATION",
      "SAFE_PROJECTION_AND_TERMINAL_RESOLUTION",
      "PLATFORM_SEARCH",
      "CONTINUOUS_REFERENCE_GENERATION",
      "CONTINUOUS_VALIDATION",
      "BUNDLE_PACKAGING_AND_SERIALIZATION",
  };
  constexpr std::array<std::string_view, 3U> kExcludedNames{
      "RAW_SENSOR_MAPPING",
      "PROCESS_STARTUP",
      "REMOTE_RPC_OR_DISK_IO",
  };
  const Json& included = boundary_json.at("included");
  const Json& excluded = boundary_json.at("excluded");
  if (std::get<std::string>(starts) !=
          "REQUEST_SCHEMA_AND_SNAPSHOT_VALIDATION" ||
      std::get<std::string>(ends) !=
          "PLANNING_RESPONSE_SERIALIZATION" ||
      !std::get<bool>(cold_separate) ||
      !included.is_array() ||
      included.size() != kIncludedNames.size() ||
      !excluded.is_array() ||
      excluded.size() != kExcludedNames.size()) {
    return Invalid(
        "benchmark_profile.content.timing_boundary",
        "timing boundary does not match the declared API boundary");
  }
  for (std::size_t index = 0U;
       index < kIncludedNames.size(); ++index) {
    if (!included[index].is_string() ||
        included[index].get<std::string>() !=
            kIncludedNames[index]) {
      return Invalid(
          "benchmark_profile.content.timing_boundary.included",
          "included module order is not canonical");
    }
  }
  for (std::size_t index = 0U;
       index < kExcludedNames.size(); ++index) {
    if (!excluded[index].is_string() ||
        excluded[index].get<std::string>() !=
            kExcludedNames[index]) {
      return Invalid(
          "benchmark_profile.content.timing_boundary.excluded",
          "excluded activity order is not canonical");
    }
  }

  auto record_modules = ReadBool(
      content_json, "record_module_timings",
      "benchmark_profile.content.record_module_timings");
  auto independent = ReadBool(
      content_json,
      "planner_result_must_be_independent_of_latency_target",
      "benchmark_profile.content."
      "planner_result_must_be_independent_of_latency_target");
  if (!IsOk(record_modules)) {
    return std::get<Error>(record_modules);
  }
  if (!IsOk(independent)) {
    return std::get<Error>(independent);
  }

  auto canonical = JcsCanonicalizer::Canonicalize(content_json);
  if (!IsOk(canonical)) {
    return std::get<Error>(canonical);
  }
  auto digest =
      Sha256Hex(std::get<std::string>(canonical));
  if (!IsOk(digest)) {
    return std::get<Error>(digest);
  }
  const ContentRef decoded_profile_ref =
      std::get<ContentRef>(content_ref);
  if (std::get<Sha256Digest>(digest) !=
      decoded_profile_ref.content_hash) {
    return Invalid(
        "benchmark_profile.content_ref.content_hash",
        "profile content hash does not match JCS content");
  }

  BenchmarkProfile profile{
      .content_ref = decoded_profile_ref,
      .content =
          BenchmarkProfileContent{
              .hardware =
                  BenchmarkHardwareProfile{
                      .cpu_model =
                          std::get<std::string>(std::move(cpu)),
                      .physical_core_count =
                          std::get<std::size_t>(physical),
                      .logical_core_count =
                          std::get<std::size_t>(logical),
                      .memory_bytes =
                          std::get<std::uint64_t>(memory),
                      .os_name =
                          std::get<std::string>(
                              std::move(os_name)),
                      .os_version =
                          std::get<std::string>(
                              std::move(os_version)),
                      .compiler_id =
                          std::get<std::string>(
                              std::move(compiler)),
                      .compiler_version =
                          std::get<std::string>(
                              std::move(compiler_version)),
                      .build_type = build_type,
                  },
              .algorithm_config_ref =
                  std::get<ContentRef>(
                      std::move(algorithm_ref)),
              .planner_thread_count =
                  std::get<std::size_t>(planner_threads),
              .warmup_sample_count =
                  std::get<std::size_t>(warmups),
              .measured_sample_count_per_platform =
                  std::get<std::size_t>(measured),
              .p95_latency_target =
                  DurationNanoseconds{
                      Nanoseconds{kP95LatencyTargetNanoseconds}},
              .platform_suites = std::move(suites),
              .timing_boundary = BenchmarkTimingBoundary{},
              .record_module_timings =
                  std::get<bool>(record_modules),
              .planner_result_must_be_independent_of_latency_target =
                  std::get<bool>(independent),
          },
  };
  const auto validation = ValidateProfileImpl(profile);
  if (!IsOk(validation)) {
    return std::get<Error>(validation);
  }
  return profile;
}

[[nodiscard]] std::size_t NearestRankIndex(
    std::size_t sample_count, std::size_t percentile_numerator) noexcept {
  constexpr std::size_t kDenominator = 100U;
  const std::size_t quotient = sample_count / kDenominator;
  const std::size_t remainder = sample_count % kDenominator;
  const std::size_t rank =
      quotient * percentile_numerator +
      (remainder * percentile_numerator + kDenominator - 1U) /
          kDenominator;
  return rank - 1U;
}

[[nodiscard]] Nanoseconds RoundedMeanTiesToEven(
    const std::vector<Nanoseconds>& sorted) noexcept {
  const std::uint64_t count =
      static_cast<std::uint64_t>(sorted.size());
  std::uint64_t floor_mean = 0U;
  std::uint64_t remainder = 0U;

  for (const auto value : sorted) {
    const auto raw = static_cast<std::uint64_t>(value.count());
    floor_mean += raw / count;
    const std::uint64_t term_remainder = raw % count;
    if (term_remainder != 0U &&
        remainder >= count - term_remainder) {
      ++floor_mean;
      remainder -= count - term_remainder;
    } else {
      remainder += term_remainder;
    }
  }

  const bool above_half = remainder > count / 2U;
  const bool exactly_half =
      count % 2U == 0U && remainder == count / 2U;
  if (above_half || (exactly_half && floor_mean % 2U != 0U)) {
    ++floor_mean;
  }
  return Nanoseconds{static_cast<std::int64_t>(floor_mean)};
}

[[nodiscard]] LatencyStatistics BuildStatistics(
    std::vector<Nanoseconds> values) {
  std::sort(values.begin(), values.end());
  return LatencyStatistics{
      .mean = DurationNanoseconds{RoundedMeanTiesToEven(values)},
      .median =
          DurationNanoseconds{
              values[NearestRankIndex(values.size(), 50U)]},
      .p95 =
          DurationNanoseconds{
              values[NearestRankIndex(values.size(), 95U)]},
      .p99 =
          DurationNanoseconds{
              values[NearestRankIndex(values.size(), 99U)]},
      .maximum = DurationNanoseconds{values.back()},
      .sample_count = values.size(),
  };
}

[[nodiscard]] std::string_view ModuleName(
    BenchmarkModuleName module_name) noexcept {
  switch (module_name) {
    case BenchmarkModuleName::kSchemaAndContractValidation:
      return "SCHEMA_AND_CONTRACT_VALIDATION";
    case BenchmarkModuleName::kSafeProjectionAndTerminalResolution:
      return "SAFE_PROJECTION_AND_TERMINAL_RESOLUTION";
    case BenchmarkModuleName::kPlatformSearch:
      return "PLATFORM_SEARCH";
    case BenchmarkModuleName::kContinuousReferenceGeneration:
      return "CONTINUOUS_REFERENCE_GENERATION";
    case BenchmarkModuleName::kContinuousValidation:
      return "CONTINUOUS_VALIDATION";
    case BenchmarkModuleName::kBundlePackagingAndSerialization:
      return "BUNDLE_PACKAGING_AND_SERIALIZATION";
  }
  return "UNKNOWN";
}

[[nodiscard]] Json EncodeContentRef(const ContentRef& ref) {
  return Json{
      {"id", ref.id},
      {"revision", ref.revision},
      {"content_hash", ref.content_hash},
  };
}

[[nodiscard]] Json EncodeDuration(
    DurationNanoseconds duration) {
  return std::to_string(duration.value.count());
}

[[nodiscard]] Json EncodeStatistics(
    const LatencyStatistics& statistics) {
  return Json{
      {"mean_ns", EncodeDuration(statistics.mean)},
      {"median_ns", EncodeDuration(statistics.median)},
      {"p95_ns", EncodeDuration(statistics.p95)},
      {"p99_ns", EncodeDuration(statistics.p99)},
      {"maximum_ns", EncodeDuration(statistics.maximum)},
      {"sample_count", statistics.sample_count},
  };
}

[[nodiscard]] bool ValidStatistics(
    const LatencyStatistics& statistics) noexcept {
  return statistics.sample_count >= 1U &&
         statistics.mean.value.count() >= 0 &&
         statistics.median.value.count() >= 0 &&
         statistics.p95.value.count() >= 0 &&
         statistics.p99.value.count() >= 0 &&
         statistics.maximum.value.count() >= 0 &&
         statistics.median.value <= statistics.p95.value &&
         statistics.p95.value <= statistics.p99.value &&
         statistics.p99.value <= statistics.maximum.value;
}

[[nodiscard]] Result<bool> ValidateReportImpl(
    const BenchmarkReport& report) {
  const BenchmarkReportHeader header{
      .report_id = report.report_id,
      .report_revision = report.report_revision,
      .generated_at = report.generated_at,
  };
  auto header_validation = ValidateHeaderImpl(header);
  if (!IsOk(header_validation)) {
    return std::get<Error>(header_validation);
  }
  if (!IsValidContentRef(report.algorithm_config_ref)) {
    return Invalid("benchmark_report.algorithm_config_ref",
                   "algorithm config ref is not canonical");
  }
  if (!IsValidContentRef(report.benchmark_profile_ref)) {
    return Invalid("benchmark_report.benchmark_profile_ref",
                   "benchmark profile ref is not canonical");
  }
  if (report.platform_results.size() !=
      kCanonicalBenchmarkPlatforms.size()) {
    return Invalid("benchmark_report.platform_results",
                   "exactly three platform results are required");
  }
  std::array<bool, 3U> seen{};
  for (const auto& result : report.platform_results) {
    const std::size_t platform_index =
        PlatformIndex(result.platform_type);
    if (platform_index >= seen.size() || seen[platform_index]) {
      return Invalid("benchmark_report.platform_results",
                     "exactly one result per platform is required");
    }
    seen[platform_index] = true;
    if (!IsValidContentRef(result.safety_capability_ref)) {
      return Invalid(
          "benchmark_report.platform_results.safety_capability_ref",
          "capability ref is not canonical");
    }
    if (!ValidStatistics(result.latency)) {
      return Invalid(
          "benchmark_report.platform_results.latency",
          "latency statistics are invalid");
    }
    if (result.p95_latency_target.value.count() !=
            kP95LatencyTargetNanoseconds ||
        result.p95_latency_target_met !=
            (result.latency.p95.value <
             result.p95_latency_target.value)) {
      return Invalid(
          "benchmark_report.platform_results."
          "p95_latency_target_met",
          "P95 target comparison is inconsistent");
    }
    if (result.cold_start.first_call.value.count() < 0 ||
        result.cold_start.sample_count < 1U) {
      return Invalid(
          "benchmark_report.platform_results.cold_start",
          "cold-start statistics are invalid");
    }
    if (result.module_timings.size() >
        kCanonicalBenchmarkModules.size()) {
      return Invalid(
          "benchmark_report.platform_results.module_timings",
          "too many module timings");
    }
    std::array<bool, 6U> module_seen{};
    for (const auto& module : result.module_timings) {
      const std::size_t module_index =
          ModuleIndex(module.module_name);
      if (module_index >= module_seen.size() ||
          module_seen[module_index] ||
          !ValidStatistics(module.statistics)) {
        return Invalid(
            "benchmark_report.platform_results.module_timings",
            "module timings must be unique and valid");
      }
      module_seen[module_index] = true;
    }
  }
  return true;
}

[[nodiscard]] Result<std::array<PlatformAccumulator, 3U>>
ValidateAndGroupSamples(
    const BenchmarkProfile& profile,
    std::span<const BenchmarkSample> samples,
    const std::array<const BenchmarkPlatformSuite*, 3U>& suites) {
  std::array<PlatformAccumulator, 3U> grouped{};
  for (std::size_t sample_index = 0U; sample_index < samples.size();
       ++sample_index) {
    const auto& sample = samples[sample_index];
    const std::size_t platform_index =
        PlatformIndex(sample.platform_type);
    if (platform_index >= grouped.size()) {
      return Invalid("samples[" + std::to_string(sample_index) +
                         "].platform_type",
                     "sample platform is invalid");
    }
    if (sample.safety_capability_ref !=
        suites[platform_index]->safety_capability_ref) {
      return Invalid("samples[" + std::to_string(sample_index) +
                         "].safety_capability_ref",
                     "sample capability ref does not match its suite");
    }
    if (sample.api_latency.value.count() < 0) {
      return Invalid("samples[" + std::to_string(sample_index) +
                         "].api_latency",
                     "API latency must be non-negative");
    }
    if (sample.termination_reason.empty()) {
      return Invalid("samples[" + std::to_string(sample_index) +
                         "].termination_reason",
                     "termination reason must be recorded");
    }

    auto& accumulator = grouped[platform_index];
    accumulator.result_independence_check_passed &=
        sample.result_independence_check_passed;

    if (sample.sample_kind != BenchmarkSampleKind::kColdStart &&
        sample.sample_kind != BenchmarkSampleKind::kMeasured) {
      return Invalid("samples[" + std::to_string(sample_index) +
                         "].sample_kind",
                     "sample kind is invalid");
    }
    if (sample.sample_kind == BenchmarkSampleKind::kColdStart) {
      if (!sample.modules.empty()) {
        return Invalid("samples[" + std::to_string(sample_index) +
                           "].modules",
                       "cold-start module timings are reported separately");
      }
      accumulator.cold_start.push_back(sample.api_latency.value);
      continue;
    }

    if (!profile.content.record_module_timings) {
      if (!sample.modules.empty()) {
        return Invalid("samples[" + std::to_string(sample_index) +
                           "].modules",
                       "module timings were not declared by the profile");
      }
    } else {
      if (sample.modules.size() !=
          kCanonicalBenchmarkModules.size()) {
        return Invalid("samples[" + std::to_string(sample_index) +
                           "].modules",
                       "all six module timings are required");
      }
      std::array<bool, kCanonicalBenchmarkModules.size()> seen{};
      for (const auto& module : sample.modules) {
        const std::size_t module_index =
            ModuleIndex(module.module_name);
        if (module_index >= seen.size() || seen[module_index] ||
            module.latency.value.count() < 0 ||
            module.latency.value > sample.api_latency.value) {
          return Invalid("samples[" + std::to_string(sample_index) +
                             "].modules",
                         "module timings must be unique, non-negative, "
                         "and bounded by API latency");
        }
        seen[module_index] = true;
        accumulator.modules[module_index].push_back(
            module.latency.value);
      }
    }
    accumulator.measured.push_back(sample.api_latency.value);
  }

  for (const auto platform_type : kCanonicalBenchmarkPlatforms) {
    const std::size_t index = PlatformIndex(platform_type);
    const std::string prefix =
        "samples." + std::string(PlatformName(platform_type));
    if (grouped[index].cold_start.size() != 1U) {
      return Invalid(prefix + ".cold_start",
                     "exactly one cold-start sample is required");
    }
    if (grouped[index].measured.size() !=
        profile.content.measured_sample_count_per_platform) {
      return Invalid(prefix + ".measured",
                     "measured sample count does not match the profile");
    }
  }
  return grouped;
}

}  // namespace

Result<bool> ValidateBenchmarkProfile(
    const BenchmarkProfile& profile) noexcept {
  try {
    const auto validation = ValidateProfileImpl(profile);
    if (!IsOk(validation)) {
      return std::get<Error>(validation);
    }
    return true;
  } catch (const std::exception& error) {
    return Error{
        .code = ErrorCode::kResourceLimit,
        .field_path = "benchmark_profile",
        .message = std::string{"profile validation failed: "} +
                   error.what(),
    };
  } catch (...) {
    return Error{
        .code = ErrorCode::kResourceLimit,
        .field_path = "benchmark_profile",
        .message = "profile validation failed",
    };
  }
}

Result<bool> ValidateBenchmarkReportHeader(
    const BenchmarkReportHeader& header) noexcept {
  try {
    return ValidateHeaderImpl(header);
  } catch (const std::exception& error) {
    return Error{
        .code = ErrorCode::kResourceLimit,
        .field_path = "benchmark_report_header",
        .message = std::string{"header validation failed: "} +
                   error.what(),
    };
  } catch (...) {
    return Error{
        .code = ErrorCode::kResourceLimit,
        .field_path = "benchmark_report_header",
        .message = "header validation failed",
    };
  }
}

Result<BenchmarkProfile> DecodeBenchmarkProfile(
    std::string_view payload) noexcept {
  try {
    return DecodeProfileImpl(payload);
  } catch (const nlohmann::json::exception& error) {
    return Error{
        .code = ErrorCode::kSchemaMismatch,
        .field_path = "benchmark_profile",
        .message = std::string{"invalid benchmark profile JSON: "} +
                   error.what(),
    };
  } catch (const std::exception& error) {
    return Error{
        .code = ErrorCode::kResourceLimit,
        .field_path = "benchmark_profile",
        .message = std::string{"profile decode failed: "} +
                   error.what(),
    };
  } catch (...) {
    return Error{
        .code = ErrorCode::kResourceLimit,
        .field_path = "benchmark_profile",
        .message = "profile decode failed",
    };
  }
}

Result<std::string> EncodeBenchmarkReport(
    const BenchmarkReport& report) noexcept {
  try {
    const auto validation = ValidateReportImpl(report);
    if (!IsOk(validation)) {
      return std::get<Error>(validation);
    }
    Json platform_results = Json::array();
    for (const auto& platform : report.platform_results) {
      Json modules = Json::array();
      for (const auto& module : platform.module_timings) {
        modules.push_back(
            Json{
                {"module_name",
                 std::string{ModuleName(module.module_name)}},
                {"statistics",
                 EncodeStatistics(module.statistics)},
            });
      }
      platform_results.push_back(
          Json{
              {"platform_type",
               std::string{PlatformName(platform.platform_type)}},
              {"safety_capability_ref",
               EncodeContentRef(
                   platform.safety_capability_ref)},
              {"latency", EncodeStatistics(platform.latency)},
              {"p95_latency_target_ns",
               EncodeDuration(
                   platform.p95_latency_target)},
              {"p95_latency_target_met",
               platform.p95_latency_target_met},
              {"cold_start",
               Json{
                   {"first_call_ns",
                    EncodeDuration(
                        platform.cold_start.first_call)},
                   {"sample_count",
                    platform.cold_start.sample_count},
               }},
              {"module_timings", std::move(modules)},
              {"result_independence_check_passed",
               platform.result_independence_check_passed},
          });
    }
    const Json document{
        {"schema_version",
         std::string{BenchmarkReport::kSchemaVersion}},
        {"report_id", report.report_id},
        {"report_revision", report.report_revision},
        {"generated_at",
         Json{
             {"clock_id", report.generated_at.clock_id},
             {"tick_ns",
              std::to_string(
                  report.generated_at.tick.count())},
         }},
        {"algorithm_config_ref",
         EncodeContentRef(report.algorithm_config_ref)},
        {"benchmark_profile_ref",
         EncodeContentRef(report.benchmark_profile_ref)},
        {"platform_results", std::move(platform_results)},
    };
    return JcsCanonicalizer::Canonicalize(document);
  } catch (const std::exception& error) {
    return Error{
        .code = ErrorCode::kResourceLimit,
        .field_path = "benchmark_report",
        .message = std::string{"report encoding failed: "} +
                   error.what(),
    };
  } catch (...) {
    return Error{
        .code = ErrorCode::kResourceLimit,
        .field_path = "benchmark_report",
        .message = "report encoding failed",
    };
  }
}

Result<BenchmarkReport> BuildBenchmarkReport(
    const BenchmarkProfile& profile,
    std::span<const BenchmarkSample> samples,
    const BenchmarkReportHeader& header) noexcept {
  try {
    const auto header_validation = ValidateHeaderImpl(header);
    if (!IsOk(header_validation)) {
      return std::get<Error>(header_validation);
    }

    const auto profile_validation = ValidateProfileImpl(profile);
    if (!IsOk(profile_validation)) {
      return std::get<Error>(profile_validation);
    }
    const auto& suites = std::get<
        std::array<const BenchmarkPlatformSuite*, 3U>>(
        profile_validation);

    const auto sample_validation =
        ValidateAndGroupSamples(profile, samples, suites);
    if (!IsOk(sample_validation)) {
      return std::get<Error>(sample_validation);
    }
    auto grouped =
        std::get<std::array<PlatformAccumulator, 3U>>(
            sample_validation);

    BenchmarkReport report{
        .report_id = header.report_id,
        .report_revision = header.report_revision,
        .generated_at = header.generated_at,
        .algorithm_config_ref = profile.content.algorithm_config_ref,
        .benchmark_profile_ref = profile.content_ref,
        .platform_results = {},
    };
    report.platform_results.reserve(
        kCanonicalBenchmarkPlatforms.size());

    for (const auto platform_type : kCanonicalBenchmarkPlatforms) {
      const std::size_t platform_index =
          PlatformIndex(platform_type);
      auto& accumulator = grouped[platform_index];
      PlatformBenchmarkResult platform_result{
          .platform_type = platform_type,
          .safety_capability_ref =
              suites[platform_index]->safety_capability_ref,
          .latency = BuildStatistics(
              std::move(accumulator.measured)),
          .p95_latency_target =
              profile.content.p95_latency_target,
          .p95_latency_target_met = false,
          .cold_start =
              ColdStartStatistics{
                  .first_call = DurationNanoseconds{
                      accumulator.cold_start.front()},
                  .sample_count = accumulator.cold_start.size(),
              },
          .module_timings = {},
          .result_independence_check_passed =
              accumulator.result_independence_check_passed,
      };
      platform_result.p95_latency_target_met =
          platform_result.latency.p95.value <
          platform_result.p95_latency_target.value;

      if (profile.content.record_module_timings) {
        platform_result.module_timings.reserve(
            kCanonicalBenchmarkModules.size());
        for (std::size_t module_index = 0U;
             module_index < kCanonicalBenchmarkModules.size();
             ++module_index) {
          platform_result.module_timings.push_back(
              BenchmarkModuleTiming{
                  .module_name =
                      kCanonicalBenchmarkModules[module_index],
                  .statistics = BuildStatistics(std::move(
                      accumulator.modules[module_index])),
              });
        }
      }
      report.platform_results.push_back(
          std::move(platform_result));
    }
    return report;
  } catch (const std::exception& error) {
    return Error{
        .code = ErrorCode::kResourceLimit,
        .field_path = "benchmark_report",
        .message = std::string{"benchmark report construction failed: "} +
                   error.what(),
    };
  } catch (...) {
    return Error{
        .code = ErrorCode::kResourceLimit,
        .field_path = "benchmark_report",
        .message = "benchmark report construction failed",
    };
  }
}

}  // namespace lunar::planning::v3
