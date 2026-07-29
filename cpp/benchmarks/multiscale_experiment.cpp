#include "multiscale_experiment.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <map>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include <nlohmann/json.hpp>

#include "fixtures/system/planner_v3_system_fixture.hpp"
#include "lunar_path_planner/v3/codec/json_codec.hpp"

namespace lunar::planning::v3 {
namespace {

using Json = nlohmann::json;
using Clock = std::chrono::steady_clock;
using system_test::G1ReferenceScenario;
using system_test::MapScale;
using multiscale_experiment_internal::LatencyStatistics;

constexpr std::string_view kManifestSchema =
    "path-planner-v3-multiscale-experiment-manifest/v1";
constexpr std::string_view kLatencySchema =
    "path-planner-v3-multiscale-latency-sample/v1";
constexpr std::string_view kSummarySchema =
    "path-planner-v3-multiscale-experiment-summary/v1";
constexpr std::string_view kResponseSchema =
    "path-planner-v3-multiscale-response/v1";
constexpr std::size_t kCombinationCount = 27U;
constexpr std::size_t kBallisticSampleCount = 21U;

struct Combination final {
  PlatformType platform;
  MapScale scale;
  G1ReferenceScenario scene;
};

struct G1ReferenceProvenance final {
  std::string_view source_scenario_id;
  std::string_view source_scenario_hash;
  std::string_view density_profile;
};

struct Sample final {
  Combination combination;
  std::string sample_kind;
  std::size_t sample_index{};
  DurationNanoseconds latency;
  PlanningOutcome planning_outcome{};
  std::string reason_code;
  std::string response_hash;
  bool outcome_correct{};
};

struct TimedInvocation final {
  Sample sample;
  PlanningResponse response;
};

struct Representative final {
  Combination combination;
  system_test::ScenarioDescription description;
  Vec3 map_minimum;
  Vec3 map_maximum;
  std::size_t map_width_cells{};
  std::size_t map_height_cells{};
  double map_resolution_m{};
  PlanningResponse response;
  std::string response_hash;
};

[[nodiscard]] const std::array<Combination, kCombinationCount>&
ExperimentMatrix() {
  static const std::array<Combination, kCombinationCount> matrix = [] {
    std::array<Combination, kCombinationCount> result{};
    constexpr std::array platforms{
        PlatformType::kWheeled,
        PlatformType::kLegged,
        PlatformType::kHopper,
    };
    constexpr std::array scales{
        MapScale::kTenMeter,
        MapScale::kHundredMeter,
        MapScale::kThousandMeter,
    };
    constexpr std::array scenes{
        G1ReferenceScenario::kLowKnown,
        G1ReferenceScenario::kMediumKnown,
        G1ReferenceScenario::kHighFrontier,
    };
    std::size_t index = 0U;
    for (const PlatformType platform : platforms) {
      for (const MapScale scale : scales) {
        for (const G1ReferenceScenario scene : scenes) {
          result[index++] = {platform, scale, scene};
        }
      }
    }
    return result;
  }();
  return matrix;
}

[[nodiscard]] std::string_view PlatformName(const PlatformType platform) {
  switch (platform) {
    case PlatformType::kWheeled:
      return "WHEELED";
    case PlatformType::kLegged:
      return "LEGGED";
    case PlatformType::kHopper:
      return "HOPPER";
  }
  throw std::invalid_argument{"unsupported platform"};
}

[[nodiscard]] std::string_view ScaleName(const MapScale scale) {
  switch (scale) {
    case MapScale::kTenMeter:
      return "TEN_METER";
    case MapScale::kHundredMeter:
      return "HUNDRED_METER";
    case MapScale::kThousandMeter:
      return "THOUSAND_METER";
  }
  throw std::invalid_argument{"unsupported scale"};
}

[[nodiscard]] std::string_view SceneName(
    const G1ReferenceScenario scene) {
  switch (scene) {
    case G1ReferenceScenario::kLowKnown:
      return "G1_LOW_KNOWN";
    case G1ReferenceScenario::kMediumKnown:
      return "G1_MEDIUM_KNOWN";
    case G1ReferenceScenario::kHighFrontier:
      return "G1_HIGH_FRONTIER";
  }
  throw std::invalid_argument{"unsupported scene"};
}

[[nodiscard]] G1ReferenceProvenance G1ReferenceFor(
    const G1ReferenceScenario scene) {
  switch (scene) {
    case G1ReferenceScenario::kLowKnown:
      return {
          .source_scenario_id =
              "validation/scenario-0027/standard-proxy/v1",
          .source_scenario_hash =
              "564835143c269d1ef4fb8d01cb7517cf4efb674bb1554c36467a1d59eb39f16c",
          .density_profile = "low",
      };
    case G1ReferenceScenario::kMediumKnown:
      return {
          .source_scenario_id =
              "validation/scenario-0007/standard-proxy/v1",
          .source_scenario_hash =
              "6f754169bb3c4837978e3e852258493f4b1c971e8a59148a9a8269fb306a95bc",
          .density_profile = "medium",
      };
    case G1ReferenceScenario::kHighFrontier:
      return {
          .source_scenario_id =
              "validation/scenario-0116/standard-proxy/v1",
          .source_scenario_hash =
              "49e4254dc8e5602be67cc2e9eb68af65093ec049b463e3581edfc9b5f352b554",
          .density_profile = "high",
      };
  }
  throw std::invalid_argument{"scene is not a G1 reference"};
}

[[nodiscard]] Json G1ReferenceJson(
    const G1ReferenceScenario scene) {
  const G1ReferenceProvenance provenance = G1ReferenceFor(scene);
  return {
      {"source_scenario_id", provenance.source_scenario_id},
      {"source_scenario_hash", provenance.source_scenario_hash},
      {"density_profile", provenance.density_profile},
      {"derivation", "g1_validation_reference_scaled/v1"},
      {"synthetic_source_kind",
       "synthetic_terrain_obstacle_proxy/v1"},
      {"physical_obstacle_cells_written", false},
  };
}

[[nodiscard]] std::string_view OutcomeName(
    const PlanningOutcome outcome) {
  switch (outcome) {
    case PlanningOutcome::kNewReferenceReady:
      return "NEW_REFERENCE_READY";
    case PlanningOutcome::kSafeFrontierReferenceReady:
      return "SAFE_FRONTIER_REFERENCE_READY";
    case PlanningOutcome::kNoKnownSafeRoute:
      return "NO_KNOWN_SAFE_ROUTE";
    case PlanningOutcome::kGoalInfeasible:
      return "GOAL_INFEASIBLE";
    case PlanningOutcome::kInvalidRequest:
      return "INVALID_REQUEST";
    case PlanningOutcome::kStaleInput:
      return "STALE_INPUT";
    case PlanningOutcome::kNumericalFailure:
      return "NUMERICAL_FAILURE";
    case PlanningOutcome::kResourceLimit:
      return "RESOURCE_LIMIT";
    case PlanningOutcome::kActiveReferenceInvalidated:
      return "ACTIVE_REFERENCE_INVALIDATED";
  }
  throw std::invalid_argument{"unsupported planning outcome"};
}

[[nodiscard]] PlanningOutcome ExpectedOutcome(
    const Combination& combination) {
  if (combination.scene ==
          G1ReferenceScenario::kHighFrontier &&
      combination.platform != PlatformType::kHopper) {
    return PlanningOutcome::kSafeFrontierReferenceReady;
  }
  return PlanningOutcome::kNewReferenceReady;
}

[[nodiscard]] Error Invalid(
    std::string field_path, std::string message) {
  return {
      .code = ErrorCode::kInvalidArgument,
      .field_path = std::move(field_path),
      .message = std::move(message),
  };
}

[[nodiscard]] Json Vec2Json(const Vec2 value) {
  return Json::array({value.x, value.y});
}

[[nodiscard]] Json Vec3XyJson(const Vec3 value) {
  return Json::array({value.x, value.y});
}

[[nodiscard]] Json CombinationJson(const Combination& combination) {
  return {
      {"platform", PlatformName(combination.platform)},
      {"scale", ScaleName(combination.scale)},
      {"scene", SceneName(combination.scene)},
      {"expected_planning_outcome",
       OutcomeName(ExpectedOutcome(combination))},
  };
}

[[nodiscard]] std::size_t NearestRankIndex(
    const std::size_t count, const std::size_t numerator,
    const std::size_t denominator) {
  const std::size_t rank =
      (numerator * count + denominator - 1U) / denominator;
  return rank - 1U;
}

[[nodiscard]] Json StatisticsJson(const LatencyStatistics& statistics) {
  return {
      {"sample_count", statistics.sample_count},
      {"correct_count", statistics.correct_count},
      {"success_rate", statistics.success_rate},
      {"p50_ns", statistics.p50.value.count()},
      {"p95_ns", statistics.p95.value.count()},
      {"maximum_ns", statistics.maximum.value.count()},
      {"p95_target_met", statistics.p95_target_met},
  };
}

[[nodiscard]] Result<std::string> ResponseHash(
    const PlanningResponse& response) {
  return system_test::ResponseHash(response);
}

[[nodiscard]] Result<TimedInvocation> InvokeTimed(
    PlannerV3& planner, const PlanningRequest& request,
    const Combination& combination, std::string sample_kind,
    const std::size_t sample_index) {
  const auto start = Clock::now();
  PlanningResponse response = planner.Plan(request);
  const std::string encoded =
      JsonCodec::EncodePlanningResponse(response);
  const auto end = Clock::now();
  static_cast<void>(encoded);
  auto hash = ResponseHash(response);
  if (!IsOk(hash)) {
    return std::get<Error>(std::move(hash));
  }
  return TimedInvocation{
      .sample =
          {
              .combination = combination,
              .sample_kind = std::move(sample_kind),
              .sample_index = sample_index,
              .latency =
                  DurationNanoseconds{
                      std::chrono::duration_cast<
                          std::chrono::nanoseconds>(end - start)},
              .planning_outcome = response.planning_outcome,
              .reason_code = response.reason_code,
              .response_hash =
                  std::get<std::string>(std::move(hash)),
              .outcome_correct =
                  response.planning_outcome ==
                  ExpectedOutcome(combination),
          },
      .response = std::move(response),
  };
}

[[nodiscard]] Result<PlanningResponse> InvokeUntimed(
    PlannerV3& planner, const PlanningRequest& request) {
  PlanningResponse response = planner.Plan(request);
  static_cast<void>(JsonCodec::EncodePlanningResponse(response));
  return response;
}

void AppendPath(
    const GeometricPath& path, std::vector<Vec2>& points) {
  if (const auto* spline =
          std::get_if<ClampedCubicBSplinePath>(&path)) {
    for (const PoseXyzYaw& point : spline->control_points) {
      points.push_back({point.position_m.x, point.position_m.y});
    }
    return;
  }
  const auto& chain = std::get<ValidatedPrimitiveChain>(path);
  if (!chain.primitives.empty()) {
    const Vec3 start = chain.primitives.front().start_pose.position_m;
    points.push_back({start.x, start.y});
  }
  for (const ValidatedPrimitive& primitive : chain.primitives) {
    points.push_back(
        {primitive.end_pose.position_m.x,
         primitive.end_pose.position_m.y});
  }
}

[[nodiscard]] std::vector<Vec2> ReferencePolyline(
    const ReferenceBundle& bundle) {
  std::vector<Vec2> result;
  if (const auto* wheel =
          std::get_if<WheeledReference>(&bundle.platform_reference)) {
    for (const WheeledSegment& segment : wheel->segments) {
      if (const auto* drive = std::get_if<DriveSegment>(&segment)) {
        AppendPath(drive->geometric_path, result);
      } else {
        const Vec3 position =
            std::get<SpinSegment>(segment).fixed_position_m;
        result.push_back({position.x, position.y});
      }
    }
  } else if (const auto* legged =
                 std::get_if<LeggedBodyReference>(
                     &bundle.platform_reference)) {
    AppendPath(legged->geometric_path, result);
  } else {
    const auto& hopper =
        std::get<HopperReference>(bundle.platform_reference);
    const Vec3 launch =
        hopper.jump_boundary.nominal_launch_state.position_m;
    result.push_back({launch.x, launch.y});
    const Vec3 landing =
        hopper.next_landing_region.landing_plane.origin_m;
    result.push_back({landing.x, landing.y});
  }
  return result;
}

[[nodiscard]] Json RegionsJson(
    const std::vector<system_test::ScenarioRegion>& regions) {
  Json result = Json::array();
  for (const auto& region : regions) {
    std::string_view kind;
    switch (region.kind) {
      case system_test::ScenarioRegion::Kind::kHardObstacle:
        kind = "HARD_OBSTACLE";
        break;
      case system_test::ScenarioRegion::Kind::
          kSyntheticTerrainObstacleProxy:
        kind = "SYNTHETIC_TERRAIN_OBSTACLE_PROXY";
        break;
      case system_test::ScenarioRegion::Kind::kUnknown:
        kind = "UNKNOWN";
        break;
      case system_test::ScenarioRegion::Kind::kHighSlope:
        kind = "HIGH_SLOPE";
        break;
      case system_test::ScenarioRegion::Kind::kHighRoughness:
        kind = "HIGH_ROUGHNESS";
        break;
    }
    Json polygon = Json::array();
    for (const Vec2 vertex : region.polygon_xy_m) {
      polygon.push_back(Vec2Json(vertex));
    }
    Json encoded_region{
        {"kind", std::string{kind}},
        {"polygon_xy_m", std::move(polygon)},
    };
    if (region.kind ==
        system_test::ScenarioRegion::Kind::
            kSyntheticTerrainObstacleProxy) {
      encoded_region["source_kind"] =
          "synthetic_terrain_obstacle_proxy/v1";
      encoded_region["physical_obstacle_cells_written"] = false;
    }
    result.push_back(std::move(encoded_region));
  }
  return result;
}

[[nodiscard]] Json BallisticArcJson(const HopperReference& reference) {
  const Vec3 position =
      reference.jump_boundary.nominal_launch_state.position_m;
  const Vec3 velocity =
      reference.jump_boundary.nominal_launch_state.linear_velocity_mps;
  const double flight_time_s =
      std::chrono::duration<double>(
          reference.jump_boundary.ballistic_flight_time.value)
          .count();
  Json result = Json::array();
  for (std::size_t index = 0U;
       index < kBallisticSampleCount; ++index) {
    const double fraction =
        static_cast<double>(index) /
        static_cast<double>(kBallisticSampleCount - 1U);
    const double time_s = fraction * flight_time_s;
    const double x = position.x + velocity.x * time_s;
    const double z =
        position.z + velocity.z * time_s -
        0.5 * 1.62 * time_s * time_s;
    result.push_back(Json::array({x, z}));
  }
  return result;
}

[[nodiscard]] Json LandingPolygonJson(
    const HopperReference& reference) {
  const LandingPlane& plane =
      reference.next_landing_region.landing_plane;
  Json result = Json::array();
  for (const Vec2 uv :
       reference.next_landing_region.convex_polygon.vertices_uv) {
    const Vec3 point{
        plane.origin_m.x + uv.x * plane.basis_u.x +
            uv.y * plane.basis_v.x,
        plane.origin_m.y + uv.x * plane.basis_u.y +
            uv.y * plane.basis_v.y,
        plane.origin_m.z + uv.x * plane.basis_u.z +
            uv.y * plane.basis_v.z,
    };
    result.push_back(Vec3XyJson(point));
  }
  return result;
}

[[nodiscard]] Result<Json> RepresentativeJson(
    const Representative& representative) {
  if (!representative.response.new_reference_bundle.has_value()) {
    return Invalid(
        "responses.new_reference_bundle",
        "representative response has no new reference bundle");
  }
  const ReferenceBundle& bundle =
      *representative.response.new_reference_bundle;
  Json polyline = Json::array();
  for (const Vec2 point : ReferencePolyline(bundle)) {
    polyline.push_back(Vec2Json(point));
  }
  Json result{
      {"schema_version", kResponseSchema},
      {"scale", ScaleName(representative.combination.scale)},
      {"scene", SceneName(representative.combination.scene)},
      {"platform", PlatformName(representative.combination.platform)},
      {"map_bounds",
       {{"minimum_xy_m", Vec3XyJson(representative.map_minimum)},
        {"maximum_xy_m", Vec3XyJson(representative.map_maximum)}}},
      {"start_xy_m", Vec2Json(representative.description.start_xy_m)},
      {"goal_xy_m", Vec2Json(representative.description.goal_xy_m)},
      {"regions", RegionsJson(representative.description.regions)},
      {"reference_polyline_xy_m", std::move(polyline)},
      {"g1_reference",
       G1ReferenceJson(representative.combination.scene)},
      {"planning_outcome",
       OutcomeName(representative.response.planning_outcome)},
      {"reason_code", representative.response.reason_code},
      {"response_hash", representative.response_hash},
  };
  if (const auto* hopper =
          std::get_if<HopperReference>(&bundle.platform_reference)) {
    result["ballistic_arc_xz_m"] = BallisticArcJson(*hopper);
    result["landing_polygon_xy_m"] = LandingPolygonJson(*hopper);
  }
  return result;
}

[[nodiscard]] Json SampleJson(const Sample& sample) {
  return {
      {"schema_version", kLatencySchema},
      {"scale", ScaleName(sample.combination.scale)},
      {"scene", SceneName(sample.combination.scene)},
      {"platform", PlatformName(sample.combination.platform)},
      {"sample_kind", sample.sample_kind},
      {"sample_index", sample.sample_index},
      {"latency_ns", sample.latency.value.count()},
      {"planning_outcome", OutcomeName(sample.planning_outcome)},
      {"reason_code", sample.reason_code},
      {"response_hash", sample.response_hash},
  };
}

[[nodiscard]] Json ManifestJson(
    const MultiscaleExperimentConfig& config,
    const std::vector<Representative>& representatives) {
  Json matrix = Json::array();
  for (const Representative& representative : representatives) {
    Json combination = CombinationJson(representative.combination);
    combination["g1_reference"] =
        G1ReferenceJson(representative.combination.scene);
    combination["map_dimensions"] = {
        {"width_m", representative.description.width_m},
        {"height_m", representative.description.height_m},
        {"width_cells", representative.map_width_cells},
        {"height_cells", representative.map_height_cells},
        {"resolution_m", representative.map_resolution_m},
    };
    matrix.push_back(std::move(combination));
  }
#ifdef NDEBUG
  constexpr std::string_view build_type = "Release";
#else
  constexpr std::string_view build_type = "Debug";
#endif
  return {
      {"schema_version", kManifestSchema},
      {"build_metadata",
       {{"build_type", build_type},
#ifdef _MSC_VER
        {"compiler", "MSVC"},
        {"compiler_version", _MSC_VER}
#else
        {"compiler", "unknown"},
        {"compiler_version", 0}
#endif
       }},
      {"warmup_count", config.warmup_count},
      {"measured_count_per_combination",
       config.measured_count_per_combination},
      {"timing_boundary",
       {{"starts_immediately_before", "planner->Plan(request)"},
        {"ends_only_after",
         "JsonCodec::EncodePlanningResponse(response)"}}},
      {"p95_target_ns", config.p95_target.value.count()},
      {"matrix", std::move(matrix)},
  };
}

template <class Predicate>
[[nodiscard]] Result<LatencyStatistics> GroupStatistics(
    const std::vector<Sample>& samples,
    const DurationNanoseconds target,
    Predicate predicate) {
  std::vector<DurationNanoseconds> latencies;
  std::size_t correct_count = 0U;
  for (const Sample& sample : samples) {
    if (sample.sample_kind == "MEASURED" && predicate(sample)) {
      latencies.push_back(sample.latency);
      if (sample.outcome_correct) {
        ++correct_count;
      }
    }
  }
  return multiscale_experiment_internal::CalculateStatistics(
      latencies, correct_count, target);
}

[[nodiscard]] Result<Json> SummaryJson(
    const MultiscaleExperimentConfig& config,
    const std::vector<Sample>& samples,
    const bool all_outcomes_correct) {
  Json scenario_results = Json::array();
  Json scale_platform_results = Json::array();
  Json platform_results = Json::array();

  for (const Combination& combination : ExperimentMatrix()) {
    auto statistics = GroupStatistics(
        samples, config.p95_target,
        [&combination](const Sample& sample) {
          return sample.combination.platform == combination.platform &&
                 sample.combination.scale == combination.scale &&
                 sample.combination.scene == combination.scene;
        });
    if (!IsOk(statistics)) {
      return std::get<Error>(std::move(statistics));
    }
    Json entry = CombinationJson(combination);
    entry.update(
        StatisticsJson(std::get<LatencyStatistics>(statistics)));
    scenario_results.push_back(std::move(entry));
  }

  constexpr std::array platforms{
      PlatformType::kWheeled,
      PlatformType::kLegged,
      PlatformType::kHopper,
  };
  constexpr std::array scales{
      MapScale::kTenMeter,
      MapScale::kHundredMeter,
      MapScale::kThousandMeter,
  };
  bool all_platform_p95_below_target = true;
  for (const PlatformType platform : platforms) {
    for (const MapScale scale : scales) {
      auto statistics = GroupStatistics(
          samples, config.p95_target,
          [platform, scale](const Sample& sample) {
            return sample.combination.platform == platform &&
                   sample.combination.scale == scale;
          });
      if (!IsOk(statistics)) {
        return std::get<Error>(std::move(statistics));
      }
      Json entry{
          {"platform", PlatformName(platform)},
          {"scale", ScaleName(scale)},
      };
      entry.update(
          StatisticsJson(std::get<LatencyStatistics>(statistics)));
      scale_platform_results.push_back(std::move(entry));
    }
    auto statistics = GroupStatistics(
        samples, config.p95_target,
        [platform](const Sample& sample) {
          return sample.combination.platform == platform;
        });
    if (!IsOk(statistics)) {
      return std::get<Error>(std::move(statistics));
    }
    const auto& value = std::get<LatencyStatistics>(statistics);
    all_platform_p95_below_target &=
        value.p95_target_met;
    Json entry{{"platform", PlatformName(platform)}};
    entry.update(StatisticsJson(value));
    platform_results.push_back(std::move(entry));
  }

  const auto counts =
      multiscale_experiment_internal::ExperimentCallCountsFor(config);
  return Json{
      {"schema_version", kSummarySchema},
      {"measured_call_count", counts.measured_call_count},
      {"cold_start_call_count", counts.cold_start_call_count},
      {"all_outcomes_correct", all_outcomes_correct},
      {"all_platform_p95_below_target",
       all_platform_p95_below_target},
      {"p95_target_ns", config.p95_target.value.count()},
      {"scenario_results", std::move(scenario_results)},
      {"scale_platform_results", std::move(scale_platform_results)},
      {"platform_results", std::move(platform_results)},
  };
}

[[nodiscard]] Result<std::filesystem::path> WriteJsonFile(
    const std::filesystem::path& path, const Json& document) {
  std::ofstream output{path, std::ios::binary | std::ios::trunc};
  if (!output) {
    return Invalid(path.string(), "cannot open output file");
  }
  output << document.dump(2) << '\n';
  if (!output) {
    return Invalid(path.string(), "cannot write output file");
  }
  return path;
}

template <class Range, class Encoder>
[[nodiscard]] Result<std::filesystem::path> WriteJsonLines(
    const std::filesystem::path& path, const Range& values,
    Encoder encoder) {
  std::ofstream output{path, std::ios::binary | std::ios::trunc};
  if (!output) {
    return Invalid(path.string(), "cannot open output file");
  }
  for (const auto& value : values) {
    auto encoded = encoder(value);
    if constexpr (
        std::is_same_v<decltype(encoded), Result<Json>>) {
      if (!IsOk(encoded)) {
        return std::get<Error>(std::move(encoded));
      }
      output << std::get<Json>(std::move(encoded)).dump() << '\n';
    } else {
      output << encoded.dump() << '\n';
    }
  }
  if (!output) {
    return Invalid(path.string(), "cannot write output file");
  }
  return path;
}

}  // namespace

namespace multiscale_experiment_internal {

Result<LatencyStatistics> CalculateStatistics(
    const std::vector<DurationNanoseconds>& samples,
    const std::size_t correct_count,
    const DurationNanoseconds p95_target) noexcept {
  try {
    if (samples.empty()) {
      return Invalid("samples", "latency samples must not be empty");
    }
    if (correct_count > samples.size()) {
      return Invalid(
          "correct_count", "correct count exceeds sample count");
    }
    if (p95_target.value.count() <= 0) {
      return Invalid("p95_target", "P95 target must be positive");
    }
    std::vector<std::chrono::nanoseconds> sorted;
    sorted.reserve(samples.size());
    for (const DurationNanoseconds sample : samples) {
      if (sample.value.count() < 0) {
        return Invalid("samples", "latency must be non-negative");
      }
      sorted.push_back(sample.value);
    }
    std::sort(sorted.begin(), sorted.end());
    return LatencyStatistics{
        .sample_count = sorted.size(),
        .correct_count = correct_count,
        .success_rate =
            static_cast<double>(correct_count) /
            static_cast<double>(sorted.size()),
        .p50 =
            DurationNanoseconds{
                sorted[NearestRankIndex(sorted.size(), 50U, 100U)]},
        .p95 =
            DurationNanoseconds{
                sorted[NearestRankIndex(sorted.size(), 95U, 100U)]},
        .maximum = DurationNanoseconds{sorted.back()},
        .p95_target_met =
            sorted[NearestRankIndex(sorted.size(), 95U, 100U)] <
            p95_target.value,
    };
  } catch (const std::exception& exception) {
    return Invalid("statistics", exception.what());
  } catch (...) {
    return Invalid("statistics", "statistics calculation failed");
  }
}

Result<LatencyStatistics> RunInvocations(
    const std::size_t sample_count,
    const DurationNanoseconds p95_target,
    const Invocation& invocation) noexcept {
  try {
    if (!invocation) {
      return Invalid("invocation", "invocation must be provided");
    }
    std::vector<DurationNanoseconds> latencies;
    latencies.reserve(sample_count);
    std::size_t correct_count = 0U;
    for (std::size_t index = 0U; index < sample_count; ++index) {
      const InvocationMeasurement measurement = invocation(index);
      latencies.push_back(measurement.latency);
      if (measurement.outcome_correct) {
        ++correct_count;
      }
    }
    return CalculateStatistics(
        latencies, correct_count, p95_target);
  } catch (const std::exception& exception) {
    return Invalid("invocation", exception.what());
  } catch (...) {
    return Invalid("invocation", "sample invocation failed");
  }
}

ExperimentCallCounts ExperimentCallCountsFor(
    const MultiscaleExperimentConfig& config) noexcept {
  return {
      .combination_count = ExperimentMatrix().size(),
      .cold_start_call_count = ExperimentMatrix().size(),
      .measured_call_count =
          ExperimentMatrix().size() *
          config.measured_count_per_combination,
  };
}

}  // namespace multiscale_experiment_internal

Result<ExperimentRunPaths> RunMultiscaleExperiment(
    const MultiscaleExperimentConfig& config,
    const std::filesystem::path& output_root) noexcept {
  try {
    if (output_root.empty()) {
      return Invalid("output_root", "output root must not be empty");
    }
    if (config.measured_count_per_combination == 0U) {
      return Invalid(
          "measured_count_per_combination",
          "measured count must be positive");
    }
    if (config.p95_target.value.count() <= 0) {
      return Invalid("p95_target", "P95 target must be positive");
    }

    std::vector<Sample> samples;
    samples.reserve(
        ExperimentMatrix().size() *
        (config.measured_count_per_combination + 1U));
    std::vector<Representative> representatives;
    representatives.reserve(ExperimentMatrix().size());
    bool all_outcomes_correct = true;

    for (const Combination& combination : ExperimentMatrix()) {
      auto scenario = system_test::MakeMultiscaleSystemScenario(
          combination.platform, combination.scale,
          combination.scene);
      auto planner = system_test::MakePlanner(scenario);

      auto cold = InvokeTimed(
          *planner, scenario.request, combination, "COLD", 0U);
      if (!IsOk(cold)) {
        return std::get<Error>(std::move(cold));
      }
      const TimedInvocation& cold_invocation =
          std::get<TimedInvocation>(cold);
      const Sample& cold_sample = cold_invocation.sample;
      all_outcomes_correct &= cold_sample.outcome_correct;
      const GridGeometry& geometry =
          scenario.request.map_snapshot->geometry();
      representatives.push_back(
          {
              .combination = combination,
              .description = scenario.description,
              .map_minimum =
                  scenario.request.map_snapshot->bounds().minimum_m,
              .map_maximum =
                  scenario.request.map_snapshot->bounds().maximum_m,
              .map_width_cells = geometry.width,
              .map_height_cells = geometry.height,
              .map_resolution_m = geometry.resolution_m,
              .response =
                  std::move(std::get<TimedInvocation>(cold).response),
              .response_hash = cold_sample.response_hash,
          });
      samples.push_back(
          std::move(std::get<TimedInvocation>(cold).sample));

      for (std::size_t index = 0U;
           index < config.warmup_count; ++index) {
        auto warmup = InvokeUntimed(*planner, scenario.request);
        if (!IsOk(warmup)) {
          return std::get<Error>(std::move(warmup));
        }
        all_outcomes_correct &=
            std::get<PlanningResponse>(warmup).planning_outcome ==
            ExpectedOutcome(combination);
      }
      for (std::size_t index = 0U;
           index < config.measured_count_per_combination; ++index) {
        auto measured = InvokeTimed(
            *planner, scenario.request, combination, "MEASURED",
            index);
        if (!IsOk(measured)) {
          return std::get<Error>(std::move(measured));
        }
        all_outcomes_correct &=
            std::get<TimedInvocation>(measured)
                .sample.outcome_correct;
        samples.push_back(
            std::move(
                std::get<TimedInvocation>(measured).sample));
      }
    }

    auto summary = SummaryJson(
        config, samples, all_outcomes_correct);
    if (!IsOk(summary)) {
      return std::get<Error>(std::move(summary));
    }

    std::error_code create_error;
    std::filesystem::create_directories(
        output_root, create_error);
    if (create_error) {
      return Invalid(
          "output_root",
          "cannot create output root: " +
              create_error.message());
    }
    const ExperimentRunPaths paths{
        .manifest_json = output_root / "manifest.json",
        .latency_samples_jsonl =
            output_root / "latency-samples.jsonl",
        .summary_json = output_root / "summary.json",
        .responses_jsonl = output_root / "responses.jsonl",
    };
    auto manifest_write = WriteJsonFile(
        paths.manifest_json,
        ManifestJson(config, representatives));
    if (!IsOk(manifest_write)) {
      return std::get<Error>(std::move(manifest_write));
    }
    auto latency_write = WriteJsonLines(
        paths.latency_samples_jsonl, samples,
        [](const Sample& sample) { return SampleJson(sample); });
    if (!IsOk(latency_write)) {
      return std::get<Error>(std::move(latency_write));
    }
    auto summary_write = WriteJsonFile(
        paths.summary_json, std::get<Json>(std::move(summary)));
    if (!IsOk(summary_write)) {
      return std::get<Error>(std::move(summary_write));
    }
    auto response_write = WriteJsonLines(
        paths.responses_jsonl, representatives,
        [](const Representative& representative) {
          return RepresentativeJson(representative);
        });
    if (!IsOk(response_write)) {
      return std::get<Error>(std::move(response_write));
    }
    return paths;
  } catch (const std::exception& exception) {
    return Invalid("multiscale_experiment", exception.what());
  } catch (...) {
    return Invalid(
        "multiscale_experiment", "experiment run failed");
  }
}

}  // namespace lunar::planning::v3
