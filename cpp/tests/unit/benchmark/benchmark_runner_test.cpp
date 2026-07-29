#include "benchmark_runner.hpp"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include <gtest/gtest.h>

namespace lunar::planning::v3 {
namespace {

using namespace std::chrono_literals;

[[nodiscard]] ContentRef Ref(std::string id, char digit) {
  return {
      .id = std::move(id),
      .revision = 1U,
      .content_hash = std::string(64U, digit),
  };
}

[[nodiscard]] BenchmarkProfile Profile(
    std::size_t measured_count = 1U,
    std::size_t warmup_count = 1U) {
  BenchmarkProfile profile{
      .content_ref = Ref("benchmark-profile", 'a'),
      .content =
          BenchmarkProfileContent{
              .hardware =
                  {
                      .cpu_model = "test",
                      .physical_core_count = 1U,
                      .logical_core_count = 1U,
                      .memory_bytes = 1'073'741'824ULL,
                      .os_name = "test",
                      .os_version = "1",
                      .compiler_id = "test",
                      .compiler_version = "1",
                  },
              .algorithm_config_ref = Ref("algorithm", 'b'),
              .planner_thread_count = 1U,
              .warmup_sample_count = warmup_count,
              .measured_sample_count_per_platform = measured_count,
              .p95_latency_target = DurationNanoseconds{1s},
          },
  };
  profile.content.platform_suites = {
      {PlatformType::kWheeled, Ref("wheel-capability", 'c'),
       {{Ref("wheel-scenario", 'd'), 1.0}}},
      {PlatformType::kLegged, Ref("legged-capability", 'e'),
       {{Ref("legged-scenario", 'f'), 1.0}}},
      {PlatformType::kHopper, Ref("hopper-capability", '1'),
       {{Ref("hopper-scenario", '2'), 1.0}}},
  };
  return profile;
}

[[nodiscard]] std::shared_ptr<const SafetyCapabilityProfile>
Capability(PlatformType platform_type, ContentRef ref) {
  SafetyCapabilityContent content;
  switch (platform_type) {
    case PlatformType::kWheeled:
      content = WheeledCapability{};
      break;
    case PlatformType::kLegged:
      content = LeggedCapability{};
      break;
    case PlatformType::kHopper:
      content = HopperCapability{};
      break;
  }
  return std::make_shared<const SafetyCapabilityProfile>(
      SafetyCapabilityProfile{
          .content_ref = std::move(ref),
          .content = std::move(content),
      });
}

[[nodiscard]] PlanningRequest Request(
    PlatformType platform_type,
    const ContentRef& capability_ref,
    const ContentRef& algorithm_ref) {
  auto algorithm = std::make_shared<PlannerAlgorithmConfig>();
  algorithm->content_ref = algorithm_ref;
  algorithm->deterministic_execution.fixed_thread_count = 1U;
  return PlanningRequest{
      .request_id =
          platform_type == PlatformType::kWheeled
              ? "wheel-request"
              : (platform_type == PlatformType::kLegged
                     ? "legged-request"
                     : "hopper-request"),
      .request_time = {"mission", 1ns},
      .state_time = {"mission", 1ns},
      .frame_id = "map",
      .platform_type = platform_type,
      .current_state =
          platform_type == PlatformType::kHopper
              ? PlatformState{HopperState{}}
              : PlatformState{WheeledOrLeggedState{}},
      .safety_capability =
          Capability(platform_type, capability_ref),
      .algorithm_config = std::move(algorithm),
  };
}

class SpyPlanner final : public PlannerV3 {
 public:
  SpyPlanner(std::vector<std::string>& events,
             std::string label)
      : events_{events}, label_{std::move(label)} {}

  [[nodiscard]] PlanningResponse Plan(
      const PlanningRequest& request) noexcept override {
    events_.push_back("plan-" + label_);
    request_ids.push_back(request.request_id);
    return PlanningResponse{
        .request_id = request.request_id,
        .response_time = request.request_time,
        .planning_outcome = PlanningOutcome::kNoKnownSafeRoute,
        .execution_directive =
            ExecutionDirective::kHoldStationary,
        .reason_code = "NO_KNOWN_SAFE_ROUTE",
        .call_diagnostics =
            CallDiagnostics{
                .api_latency = DurationNanoseconds{0ns},
                .termination_reason = "NATURAL_COMPLETION",
            },
    };
  }

  std::vector<std::string> request_ids;

 private:
  std::vector<std::string>& events_;
  std::string label_;
};

class SequenceClock final : public BenchmarkClock {
 public:
  SequenceClock(std::vector<std::string>& events,
                std::vector<std::chrono::nanoseconds> ticks)
      : events_{events}, ticks_{std::move(ticks)} {}

  [[nodiscard]] std::chrono::nanoseconds Now() noexcept override {
    events_.push_back("clock");
    if (next_ >= ticks_.size()) {
      return ticks_.back();
    }
    return ticks_[next_++];
  }

 private:
  std::vector<std::string>& events_;
  std::vector<std::chrono::nanoseconds> ticks_;
  std::size_t next_{};
};

class SpyEncoder final : public BenchmarkResponseEncoder {
 public:
  explicit SpyEncoder(std::vector<std::string>& events)
      : events_{events} {}

  [[nodiscard]] Result<std::string> Encode(
      const PlanningResponse& response) override {
    events_.push_back("encode-" + response.request_id);
    return std::string{"encoded:"} + response.request_id;
  }

 private:
  std::vector<std::string>& events_;
};

struct WorkloadFixture final {
  BenchmarkProfile profile;
  std::vector<std::string> events;
  PlanningRequest wheel_request;
  PlanningRequest legged_request;
  PlanningRequest hopper_request;
  SpyPlanner wheel;
  SpyPlanner legged;
  SpyPlanner hopper;
  std::vector<BenchmarkPlatformWorkload> workloads;

  WorkloadFixture()
      : profile{Profile()},
        wheel_request{Request(
            PlatformType::kWheeled,
            profile.content.platform_suites[0].safety_capability_ref,
            profile.content.algorithm_config_ref)},
        legged_request{Request(
            PlatformType::kLegged,
            profile.content.platform_suites[1].safety_capability_ref,
            profile.content.algorithm_config_ref)},
        hopper_request{Request(
            PlatformType::kHopper,
            profile.content.platform_suites[2].safety_capability_ref,
            profile.content.algorithm_config_ref)},
        wheel{events, "wheel"},
        legged{events, "legged"},
        hopper{events, "hopper"},
        workloads{
            {PlatformType::kWheeled, &wheel_request, &wheel},
            {PlatformType::kLegged, &legged_request, &legged},
            {PlatformType::kHopper, &hopper_request, &hopper},
        } {}
};

[[nodiscard]] BenchmarkReportHeader Header() {
  return {
      .report_id = "runner-test",
      .report_revision = 1U,
      .generated_at = {"benchmark-clock", 100ns},
  };
}

TEST(BenchmarkRunnerTest,
     ProfileNeverEntersPlannerAndTimingEndsAfterEncoding) {
  WorkloadFixture fixture;
  SequenceClock clock{
      fixture.events,
      {0ns, 10ns, 20ns, 30ns, 40ns, 50ns,
       60ns, 70ns, 80ns, 90ns, 100ns, 110ns}};
  SpyEncoder encoder{fixture.events};

  const auto result = RunPlannerApiBenchmark(
      fixture.profile, fixture.workloads, Header(), clock, encoder);

  ASSERT_TRUE(IsOk(result));
  const auto& run = std::get<BenchmarkRunResult>(result);
  ASSERT_EQ(run.report.platform_results.size(), 3U);
  EXPECT_GT(run.encoded_response_bytes, 0U);
  EXPECT_EQ(fixture.wheel.request_ids,
            std::vector<std::string>(3U, "wheel-request"));
  EXPECT_EQ(fixture.legged.request_ids,
            std::vector<std::string>(3U, "legged-request"));
  EXPECT_EQ(fixture.hopper.request_ids,
            std::vector<std::string>(3U, "hopper-request"));

  ASSERT_GE(fixture.events.size(), 4U);
  EXPECT_EQ(fixture.events[0], "clock");
  EXPECT_EQ(fixture.events[1], "plan-wheel");
  EXPECT_EQ(fixture.events[2], "encode-wheel-request");
  EXPECT_EQ(fixture.events[3], "clock");
}

TEST(BenchmarkRunnerTest,
     OneSecondTargetDoesNotCutOffSlowNaturalCompletion) {
  WorkloadFixture fixture;
  fixture.profile.content.warmup_sample_count = 0U;
  SequenceClock clock{
      fixture.events,
      {0ns, 2s, 3s, 5s, 6s, 8s,
       9s, 11s, 12s, 14s, 15s, 17s}};
  SpyEncoder encoder{fixture.events};

  const auto result = RunPlannerApiBenchmark(
      fixture.profile, fixture.workloads, Header(), clock, encoder);

  ASSERT_TRUE(IsOk(result));
  const auto& report = std::get<BenchmarkRunResult>(result).report;
  for (const auto& platform : report.platform_results) {
    EXPECT_EQ(platform.latency.p95.value, 2s);
    EXPECT_FALSE(platform.p95_latency_target_met);
  }
  EXPECT_EQ(fixture.wheel.request_ids.size(), 2U);
  EXPECT_EQ(fixture.legged.request_ids.size(), 2U);
  EXPECT_EQ(fixture.hopper.request_ids.size(), 2U);
}

TEST(BenchmarkRunnerTest, RejectsDuplicatePlatformWorkload) {
  WorkloadFixture fixture;
  fixture.workloads[2].platform_type =
      PlatformType::kLegged;
  SequenceClock clock{fixture.events, {0ns}};
  SpyEncoder encoder{fixture.events};

  const auto result = RunPlannerApiBenchmark(
      fixture.profile, fixture.workloads, Header(), clock, encoder);

  ASSERT_FALSE(IsOk(result));
  EXPECT_EQ(std::get<Error>(result).field_path, "workloads");
  EXPECT_TRUE(fixture.events.empty());
}

TEST(BenchmarkRunnerTest, RejectsRequestCapabilityOrConfigMismatch) {
  WorkloadFixture fixture;
  fixture.wheel_request.safety_capability =
      Capability(PlatformType::kWheeled,
                 Ref("other-wheel-capability", '9'));
  SequenceClock clock{fixture.events, {0ns}};
  SpyEncoder encoder{fixture.events};

  auto result = RunPlannerApiBenchmark(
      fixture.profile, fixture.workloads, Header(), clock, encoder);

  ASSERT_FALSE(IsOk(result));
  EXPECT_EQ(std::get<Error>(result).field_path,
            "workloads.WHEELED.request.safety_capability");

  WorkloadFixture second_fixture;
  auto other_algorithm =
      std::make_shared<PlannerAlgorithmConfig>();
  other_algorithm->content_ref =
      Ref("other-algorithm", '9');
  second_fixture.hopper_request.algorithm_config =
      std::move(other_algorithm);
  SequenceClock second_clock{second_fixture.events, {0ns}};
  SpyEncoder second_encoder{second_fixture.events};
  result = RunPlannerApiBenchmark(
      second_fixture.profile, second_fixture.workloads, Header(),
      second_clock, second_encoder);

  ASSERT_FALSE(IsOk(result));
  EXPECT_EQ(std::get<Error>(result).field_path,
            "workloads.HOPPER.request.algorithm_config");
}

}  // namespace
}  // namespace lunar::planning::v3

#if defined(LPP_V3_BENCHMARK_RUNNER_STANDALONE_TEST_MAIN)
int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
#endif
