#include <array>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <variant>

#include "benchmark_report.hpp"
#include "benchmark_runner.hpp"
#include "fixtures/system/planner_v3_system_fixture.hpp"
#include "lunar_path_planner/v3/codec/semantic_validator.hpp"

namespace lunar::planning::v3 {
namespace {

struct CommandLine final {
  std::filesystem::path profile_path;
  std::filesystem::path output_path;
};

[[nodiscard]] Error Invalid(std::string field_path,
                            std::string message) {
  return {
      .code = ErrorCode::kInvalidArgument,
      .field_path = std::move(field_path),
      .message = std::move(message),
  };
}

[[nodiscard]] Result<CommandLine> ParseCommandLine(
    int argc, char** argv) {
  std::optional<std::filesystem::path> profile_path;
  std::optional<std::filesystem::path> output_path;
  constexpr std::string_view profile_prefix{"--profile="};
  constexpr std::string_view output_prefix{"--output="};

  for (int index = 1; index < argc; ++index) {
    const std::string_view argument{argv[index]};
    if (argument == "--profile" || argument == "--output") {
      if (index + 1 >= argc) {
        return Invalid(
            argument == "--profile" ? "cli.profile" : "cli.output",
            "option requires a path");
      }
      const std::filesystem::path path{argv[++index]};
      auto& destination =
          argument == "--profile" ? profile_path : output_path;
      if (destination.has_value()) {
        return Invalid(
            argument == "--profile" ? "cli.profile" : "cli.output",
            "option may only be specified once");
      }
      destination = path;
      continue;
    }
    if (argument.starts_with(profile_prefix) ||
        argument.starts_with(output_prefix)) {
      const bool is_profile =
          argument.starts_with(profile_prefix);
      const std::string_view value =
          argument.substr(is_profile ? profile_prefix.size()
                                     : output_prefix.size());
      auto& destination =
          is_profile ? profile_path : output_path;
      if (value.empty()) {
        return Invalid(is_profile ? "cli.profile" : "cli.output",
                       "option requires a path");
      }
      if (destination.has_value()) {
        return Invalid(is_profile ? "cli.profile" : "cli.output",
                       "option may only be specified once");
      }
      destination = std::filesystem::path{std::string{value}};
      continue;
    }
    return Invalid("cli", "unknown option: " + std::string{argument});
  }

  if (!profile_path.has_value()) {
    return Invalid("cli.profile", "--profile is required");
  }
  if (!output_path.has_value()) {
    return Invalid("cli.output", "--output is required");
  }
  if (profile_path->empty()) {
    return Invalid("cli.profile", "profile path must not be empty");
  }
  if (output_path->empty()) {
    return Invalid("cli.output", "output path must not be empty");
  }
  return CommandLine{
      .profile_path = std::move(*profile_path),
      .output_path = std::move(*output_path),
  };
}

[[nodiscard]] Result<std::string> ReadTextFile(
    const std::filesystem::path& path) {
  std::ifstream input{path, std::ios::binary};
  if (!input.is_open()) {
    return Invalid("cli.profile",
                   "cannot open profile: " + path.string());
  }
  std::string payload{
      std::istreambuf_iterator<char>{input},
      std::istreambuf_iterator<char>{}};
  if (!input.good() && !input.eof()) {
    return Invalid("cli.profile",
                   "cannot read profile: " + path.string());
  }
  return payload;
}

[[nodiscard]] Result<bool> WriteTextFile(
    const std::filesystem::path& path,
    std::string_view payload) {
  const std::filesystem::path parent = path.parent_path();
  if (!parent.empty()) {
    std::error_code create_error;
    std::filesystem::create_directories(parent, create_error);
    if (create_error) {
      return Invalid(
          "cli.output",
          "cannot create output directory: " +
              parent.string() + ": " + create_error.message());
    }
  }

  std::ofstream output{
      path, std::ios::binary | std::ios::trunc};
  if (!output.is_open()) {
    return Invalid("cli.output",
                   "cannot open output: " + path.string());
  }
  output.write(payload.data(),
               static_cast<std::streamsize>(payload.size()));
  output.flush();
  if (!output.good()) {
    return Invalid("cli.output",
                   "cannot write output: " + path.string());
  }
  return true;
}

[[nodiscard]] Result<bool> ValidateRequest(
    const PlanningRequest& request) {
  const ValidationReport validation =
      SemanticValidator{}.Validate(request);
  if (!validation.ok()) {
    return Invalid(
        "benchmark_workload." + request.request_id,
        "fixture request is not semantically valid: " +
            system_test::DescribeIssues(validation));
  }
  return true;
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

[[nodiscard]] Result<BenchmarkRunResult> RunBenchmark(
    const BenchmarkProfile& profile) {
  auto wheeled = system_test::MakeSystemScenario(
      PlatformType::kWheeled);
  auto legged = system_test::MakeSystemScenario(
      PlatformType::kLegged);
  auto hopper = system_test::MakeSystemScenario(
      PlatformType::kHopper);

  for (const PlanningRequest* request :
       std::array<const PlanningRequest*, 3U>{
           &wheeled.request, &legged.request, &hopper.request}) {
    const auto validation = ValidateRequest(*request);
    if (!IsOk(validation)) {
      return std::get<Error>(validation);
    }
  }

  auto wheeled_planner = system_test::MakePlanner(wheeled);
  auto legged_planner = system_test::MakePlanner(legged);
  auto hopper_planner = system_test::MakePlanner(hopper);
  if (!wheeled_planner || !legged_planner || !hopper_planner) {
    return Invalid("benchmark_workload.planner",
                   "failed to create default PlannerV3 instances");
  }

  const std::array workloads{
      BenchmarkPlatformWorkload{
          PlatformType::kWheeled, &wheeled.request,
          wheeled_planner.get()},
      BenchmarkPlatformWorkload{
          PlatformType::kLegged, &legged.request,
          legged_planner.get()},
      BenchmarkPlatformWorkload{
          PlatformType::kHopper, &hopper.request,
          hopper_planner.get()},
  };

  SteadyBenchmarkClock clock;
  JsonBenchmarkResponseEncoder encoder;
  const BenchmarkReportHeader header{
      .report_id = "planner-v3-api-benchmark-report",
      .report_revision = 1U,
      .generated_at =
          ClockStamp{
              .clock_id = "steady-clock",
              .tick = clock.Now(),
          },
  };
  return RunPlannerApiBenchmark(
      profile, workloads, header, clock, encoder);
}

void PrintError(const Error& error) {
  std::cerr << error.field_path << ": " << error.message << '\n';
}

void PrintUsage(std::ostream& output) {
  output
      << "Usage: lpp_v3_planner_api_benchmark "
         "--profile <benchmark-profile.json> "
         "--output <benchmark-report.json>\n";
}

}  // namespace
}  // namespace lunar::planning::v3

int main(int argc, char** argv) {
  using namespace lunar::planning::v3;

  if (argc == 2 && std::string_view{argv[1]} == "--help") {
    PrintUsage(std::cout);
    return 0;
  }

  try {
    auto command_line = ParseCommandLine(argc, argv);
    if (!IsOk(command_line)) {
      PrintError(std::get<Error>(command_line));
      PrintUsage(std::cerr);
      return 2;
    }
    const auto& options = std::get<CommandLine>(command_line);

    auto payload = ReadTextFile(options.profile_path);
    if (!IsOk(payload)) {
      PrintError(std::get<Error>(payload));
      return 3;
    }
    auto profile =
        DecodeBenchmarkProfile(std::get<std::string>(payload));
    if (!IsOk(profile)) {
      PrintError(std::get<Error>(profile));
      return 4;
    }

    auto run = RunBenchmark(std::get<BenchmarkProfile>(profile));
    if (!IsOk(run)) {
      PrintError(std::get<Error>(run));
      return 5;
    }
    const auto& result = std::get<BenchmarkRunResult>(run);
    auto encoded = EncodeBenchmarkReport(result.report);
    if (!IsOk(encoded)) {
      PrintError(std::get<Error>(encoded));
      return 6;
    }
    auto written = WriteTextFile(
        options.output_path, std::get<std::string>(encoded));
    if (!IsOk(written)) {
      PrintError(std::get<Error>(written));
      return 7;
    }

    for (const auto& platform : result.report.platform_results) {
      std::cout << PlatformName(platform.platform_type)
                << " p95_ns=" << platform.latency.p95.value.count()
                << " target_met="
                << (platform.p95_latency_target_met ? "true" : "false")
                << '\n';
    }
    std::cout << "report=" << options.output_path.string()
              << " encoded_response_bytes="
              << result.encoded_response_bytes << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "benchmark: " << error.what() << '\n';
    return 8;
  } catch (...) {
    std::cerr << "benchmark: unexpected failure\n";
    return 8;
  }
}
