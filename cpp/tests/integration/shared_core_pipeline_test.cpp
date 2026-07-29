#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <ranges>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include "lunar_path_planner/v3/cache/deterministic_cache.hpp"
#include "lunar_path_planner/v3/codec/jcs_canonicalizer.hpp"
#include "lunar_path_planner/v3/codec/json_codec.hpp"
#include "lunar_path_planner/v3/codec/semantic_validator.hpp"
#include "lunar_path_planner/v3/corridor/convex_corridor.hpp"
#include "lunar_path_planner/v3/crypto/sha256.hpp"
#include "lunar_path_planner/v3/goal/terminal_resolver.hpp"
#include "lunar_path_planner/v3/map/safe_projection.hpp"
#include "lunar_path_planner/v3/search/ara_star.hpp"
#include "lunar_path_planner/v3/search/candidate_ranker.hpp"

#if defined(LPP_V3_STANDALONE_GTEST_MAIN)
int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
#endif

namespace lpp = lunar::planning::v3;
using Json = nlohmann::json;

namespace {

constexpr std::size_t kMapWidth = 8U;
constexpr std::size_t kMapHeight = 8U;
constexpr std::size_t kKnownWidth = 6U;
constexpr double kPi = 3.14159265358979323846;

[[nodiscard]] lpp::ContentRef Ref(std::string id,
                                  std::uint32_t revision,
                                  char digest_digit) {
  return {
      .id = std::move(id),
      .revision = revision,
      .content_hash = std::string(64U, digest_digit),
  };
}

[[nodiscard]] lpp::DeterministicVectorSet3 ErrorBall(
    double radius) {
  return lpp::EuclideanBall3{
      .center = {0.0, 0.0, 0.0},
      .radius = radius,
  };
}

[[nodiscard]] lpp::WheeledOrLeggedErrorBounds CertifiedError() {
  return {
      .position_bound_m = ErrorBall(0.05),
      .yaw_bound_rad = {.center = 0.0, .half_width = 0.01},
      .linear_velocity_bound_mps = ErrorBall(0.01),
      .yaw_rate_bound_radps = {.center = 0.0, .half_width = 0.01},
  };
}

[[nodiscard]] std::size_t Index(std::size_t x,
                                std::size_t y,
                                std::size_t width = kMapWidth) {
  return y * width + x;
}

[[nodiscard]] std::shared_ptr<const lpp::ImmutableMapSnapshot>
MakeMapSnapshot() {
  const std::size_t count = kMapWidth * kMapHeight;
  lpp::MapSnapshotInput input{
      .snapshot_ref = Ref("case-a-map", 7U, 'a'),
      .map_revision = 7U,
      .immutable_data_handle = "case-a-map-data",
      .source_time =
          {.clock_id = "mission",
           .tick = std::chrono::nanoseconds{1000}},
      .bounds =
          {
              .minimum_m = {0.0, 0.0, -1.0},
              .maximum_m = {8.0, 8.0, 1.0},
          },
      .geometry =
          {
              .width = kMapWidth,
              .height = kMapHeight,
              .resolution_m = 1.0,
              .origin_m = {0.0, 0.0},
              .frame_id = "map",
          },
      .layer_manifest =
          {
              {lpp::LayerKind::kKnownMask,
               Ref("case-a-known", 1U, '1')},
              {lpp::LayerKind::kElevation,
               Ref("case-a-elevation", 1U, '2')},
              {lpp::LayerKind::kTerrainNormal,
               Ref("case-a-normal", 1U, '3')},
              {lpp::LayerKind::kRoughness,
               Ref("case-a-roughness", 1U, '4')},
              {lpp::LayerKind::kHardObstacle,
               Ref("case-a-obstacle", 1U, '5')},
              {lpp::LayerKind::kConfidence,
               Ref("case-a-confidence", 1U, '6')},
          },
      .known_mask = std::vector<std::uint8_t>(count, 1U),
      .elevation_m = std::vector<float>(count, 0.0F),
      .normal_x = std::vector<float>(count, 0.0F),
      .normal_y = std::vector<float>(count, 0.0F),
      .normal_z = std::vector<float>(count, 1.0F),
      .roughness_m = std::vector<float>(count, 0.0F),
      .hard_obstacle_mask = std::vector<std::uint8_t>(count, 0U),
      .confidence = std::vector<float>(count, 1.0F),
  };
  for (std::size_t y = 0U; y < kMapHeight; ++y) {
    for (std::size_t x = kKnownWidth; x < kMapWidth; ++x) {
      input.known_mask[Index(x, y)] = 0U;
      input.confidence[Index(x, y)] = 0.0F;
    }
  }

  auto created = lpp::ImmutableMapSnapshot::Create(input);
  if (!lpp::IsOk(created)) {
    const auto& error = std::get<lpp::Error>(created);
    throw std::runtime_error{error.field_path + ": " + error.message};
  }
  return std::get<
      std::shared_ptr<const lpp::ImmutableMapSnapshot>>(
      std::move(created));
}

[[nodiscard]] lpp::WheelMotionPrimitive MakePrimitive(
    std::string id,
    lpp::WheelMotionPrimitive::Kind kind,
    lpp::PoseXyzYaw end_pose,
    std::chrono::nanoseconds duration,
    char digest_digit) {
  return {
      .primitive_id = std::move(id),
      .kind = kind,
      .relative_end_pose = end_pose,
      .nominal_duration = {.value = duration},
      .swept_geometry_ref =
          Ref("case-a-sweep-" +
                  std::string(1U, digest_digit),
              1U, digest_digit),
  };
}

[[nodiscard]] std::shared_ptr<
    const lpp::SafetyCapabilityProfile>
MakeCapability() {
  using Kind = lpp::WheelMotionPrimitive::Kind;
  lpp::WheeledCapability wheel{
      .frame_id = "map",
      .collision_envelope =
          {
              .vertices_xy_m =
                  {{-0.25, -0.20},
                   {0.25, -0.20},
                   {0.25, 0.20},
                   {-0.25, 0.20}},
              .minimum_z_m = -0.10,
              .maximum_z_m = 0.50,
          },
      .motion_model_ref = Ref("case-a-wheel-motion", 2U, 'b'),
      .analytic_cost_model_ref = Ref("case-a-wheel-cost", 2U, 'c'),
      .hard_limits =
          {
              .maximum_forward_speed_mps = 2.0,
              .maximum_reverse_speed_mps = 1.0,
              .maximum_spin_rate_radps = 1.5,
              .maximum_forward_acceleration_mps2 = 1.0,
              .maximum_braking_deceleration_mps2 = 1.0,
              .maximum_yaw_acceleration_radps2 = 2.0,
              .maximum_lateral_acceleration_mps2 = 1.0,
              .maximum_drive_curvature_per_m = 2.0,
              .maximum_slope_rad = 0.60,
              .minimum_clearance_m = 0.0,
          },
      .certified_state_error_bounds = CertifiedError(),
      .motion_primitives =
          {
              MakePrimitive("forward-line", Kind::kDriveForwardLine,
                            {{1.0, 0.0, 0.0}, 0.0},
                            std::chrono::seconds{1}, '0'),
              MakePrimitive("forward-arc", Kind::kDriveForwardArc,
                            {{1.0, 0.5, 0.0}, 0.5},
                            std::chrono::seconds{1}, '1'),
              MakePrimitive("reverse-line", Kind::kDriveReverseLine,
                            {{-1.0, 0.0, 0.0}, 0.0},
                            std::chrono::seconds{1}, '2'),
              MakePrimitive("reverse-arc", Kind::kDriveReverseArc,
                            {{-1.0, -0.5, 0.0}, -0.5},
                            std::chrono::seconds{1}, '3'),
              MakePrimitive("spin-cw", Kind::kSpinCw,
                            {{0.0, 0.0, 0.0}, -kPi / 2.0},
                            std::chrono::milliseconds{500}, '4'),
              MakePrimitive("spin-ccw", Kind::kSpinCcw,
                            {{0.0, 0.0, 0.0}, kPi / 2.0},
                            std::chrono::milliseconds{500}, '5'),
              MakePrimitive("stop-switch", Kind::kStopAndSwitch,
                            {{0.0, 0.0, 0.0}, 0.0},
                            std::chrono::milliseconds{100}, '6'),
          },
  };
  return std::make_shared<const lpp::SafetyCapabilityProfile>(
      lpp::SafetyCapabilityProfile{
          .content_ref =
              Ref("case-a-wheel-capability", 3U, 'd'),
          .content = std::move(wheel),
      });
}

[[nodiscard]] lpp::CorridorConfig CorridorConfig() {
  return {
      .maximum_regions = 8U,
      .maximum_inflation_iterations = 64U,
      .maximum_halfplanes_per_region = 16U,
      .maximum_split_depth = 4U,
      .minimum_overlap_m = 0.10,
      .sampling_spacing_m = 0.50,
  };
}

[[nodiscard]] lpp::SmoothingConfig SmoothingConfig() {
  return {
      .maximum_scp_iterations = 8U,
      .maximum_trust_region_reductions = 4U,
      .initial_trust_region_m = 1.0,
      .minimum_trust_region_m = 0.10,
      .constraint_tolerance = 1.0e-6,
      .maximum_time_increase =
          {.value = std::chrono::seconds{1}},
  };
}

[[nodiscard]] lpp::TimeScalingConfig TimeScalingConfig() {
  return {
      .maximum_adaptive_samples = 32U,
      .minimum_parameter_step = 0.01,
      .maximum_forward_passes = 4U,
      .maximum_backward_passes = 4U,
      .enable_jerk_smoothing = false,
      .maximum_jerk_smoothing_iterations = 0U,
  };
}

[[nodiscard]] std::shared_ptr<const lpp::PlannerAlgorithmConfig>
MakeAlgorithmConfig() {
  lpp::PlannerAlgorithmConfig config{
      .content_ref =
          Ref("case-a-algorithm-config", 5U, 'e'),
      .time_equivalence_tolerance =
          {.value = std::chrono::milliseconds{100}},
      .max_input_skew = {.value = std::chrono::seconds{1}},
      .error_bound_model_id = "case-a-error-bound-model",
      .projection_cache_capacity = 2U,
      .ara_star =
          {
              .initial_epsilon = 2.0,
              .epsilon_decrement = 0.5,
              .target_epsilon = 1.0,
              .resource_caps =
                  {
                      .maximum_expanded_states = 64U,
                      .maximum_reopened_states = 64U,
                      .maximum_generated_candidates = 4U,
                      .maximum_open_states = 64U,
                      .maximum_memory_bytes = 1'048'576U,
                  },
          },
      .wheeled =
          {
              .state_lattice =
                  {.xy_resolution_m = 1.0,
                   .yaw_bin_count = 8U,
                   .maximum_terminal_candidates = 4U},
              .corridor = CorridorConfig(),
              .smoothing = SmoothingConfig(),
              .time_scaling = TimeScalingConfig(),
              .continuous_validation_maximum_subdivisions = 16U,
          },
      .legged =
          {
              .pose_lattice =
                  {.xy_resolution_m = 1.0,
                   .yaw_bin_count = 8U,
                   .maximum_terminal_candidates = 4U},
              .maximum_height_interval_splits = 4U,
              .corridor = CorridorConfig(),
              .smoothing = SmoothingConfig(),
              .time_scaling = TimeScalingConfig(),
              .continuous_validation_maximum_subdivisions = 16U,
          },
      .hopper =
          {
              .maximum_landing_regions = 8U,
              .maximum_graph_nodes = 64U,
              .maximum_graph_out_degree = 8U,
              .yaw_partition_count = 8U,
              .support_direction_count = 8U,
              .maximum_nominal_aim_points_per_region = 8U,
              .maximum_full_certification_attempts = 8U,
              .maximum_interval_subdivision_depth = 8U,
              .maximum_collision_subdivision_depth = 8U,
              .maximum_root_iterations = 16U,
              .maximum_flight_tube_sections = 16U,
              .landing_region_inflation_iterations = 8U,
              .landing_region_maximum_split_depth = 4U,
              .landing_region_maximum_vertices = 16U,
          },
      .learned_cost_policy =
          {
              .mode =
                  lpp::LearnedCostPolicy::Mode::
                      kOptionalBoundedSoftCost,
              .maximum_inference_evaluations = 64U,
              .maximum_absolute_energy_correction = 0.25,
              .maximum_absolute_nonfatal_risk_correction = 0.25,
          },
      .learned_cost_model_ref =
          Ref("case-a-learned-model", 4U, '9'),
      .deterministic_execution =
          {
              .fixed_thread_count = 1U,
              .stable_candidate_order = true,
              .preallocated_memory_pools = true,
          },
  };
  return std::make_shared<const lpp::PlannerAlgorithmConfig>(
      std::move(config));
}

[[nodiscard]] std::shared_ptr<const lpp::LearnedCostSnapshot>
MakeLearnedSnapshot(const lpp::ImmutableMapSnapshot& map) {
  const std::size_t count = map.geometry().CellCount();
  lpp::LearnedCostSnapshotInput input{
      .snapshot_ref = Ref("case-a-learned-snapshot", 2U, 'f'),
      .model_ref = Ref("case-a-learned-model", 4U, '9'),
      .source_map_snapshot_ref = map.snapshot_ref(),
      .source_map_revision = map.map_revision(),
      .frame_id = "map",
      .feature_contract_id = "case-a-features",
      .output_contract_id = "case-a-soft-cost",
      .generated_at =
          {.clock_id = "mission",
           .tick = std::chrono::nanoseconds{999}},
      .certified_energy_correction_bounds = {-0.25, 0.25},
      .certified_nonfatal_risk_correction_bounds = {-0.25, 0.25},
      .energy_correction = std::vector<float>(count, 0.125F),
      .nonfatal_risk_correction =
          std::vector<float>(count, 0.0625F),
  };
  auto created = lpp::LearnedCostSnapshot::Create(input, count);
  if (!lpp::IsOk(created)) {
    const auto& error = std::get<lpp::Error>(created);
    throw std::runtime_error{error.field_path + ": " + error.message};
  }
  return std::get<
      std::shared_ptr<const lpp::LearnedCostSnapshot>>(
      std::move(created));
}

template <class T>
[[nodiscard]] std::shared_ptr<const T> OpaqueResolvedObject() {
  auto owner = std::make_shared<const std::max_align_t>();
  const auto* opaque =
      reinterpret_cast<const T*>(owner.get());
  return std::shared_ptr<const T>{
      std::move(owner), opaque};
}

class CaseRegistry final : public lpp::ContractObjectRegistry {
 public:
  CaseRegistry()
      : map_{MakeMapSnapshot()},
        capability_{MakeCapability()},
        config_{MakeAlgorithmConfig()},
        learned_{MakeLearnedSnapshot(*map_)} {
    const auto& wheel =
        std::get<lpp::WheeledCapability>(capability_->content);
    bindings_.motion_model = {
        .content_ref = wheel.motion_model_ref,
        .object = OpaqueResolvedObject<lpp::MotionModel>(),
    };
    bindings_.analytic_cost_model = {
        .content_ref = wheel.analytic_cost_model_ref,
        .object = OpaqueResolvedObject<lpp::AnalyticCostModel>(),
    };
  }

  [[nodiscard]] std::shared_ptr<const lpp::ImmutableMapSnapshot>
  FindMapSnapshot(const lpp::ContentRef& ref,
                  std::string_view handle) const override {
    return ref == map_->snapshot_ref() &&
                   handle == map_->immutable_data_handle()
               ? map_
               : nullptr;
  }

  [[nodiscard]]
  std::shared_ptr<const lpp::SafetyCapabilityProfile>
  FindSafetyCapability(const lpp::ContentRef& ref) const override {
    return ref == capability_->content_ref ? capability_ : nullptr;
  }

  [[nodiscard]] std::shared_ptr<const lpp::PlannerAlgorithmConfig>
  FindAlgorithmConfig(const lpp::ContentRef& ref) const override {
    return ref == config_->content_ref ? config_ : nullptr;
  }

  [[nodiscard]] std::shared_ptr<const lpp::LearnedCostSnapshot>
  FindLearnedCost(const lpp::ContentRef& ref,
                  std::string_view handle) const override {
    return ref == learned_->snapshot_ref() &&
                   handle == "case-a-learned-data"
               ? learned_
               : nullptr;
  }

  [[nodiscard]] lpp::Result<lpp::ResolvedCapabilityBindings>
  ResolveCapabilityBindings(
      const lpp::SafetyCapabilityProfile& profile) const override {
    if (profile.content_ref != capability_->content_ref) {
      return lpp::Error{
          .code = lpp::ErrorCode::kMissingRegistryObject,
          .field_path = "safety_capability_ref",
          .message = "case capability is absent",
      };
    }
    return bindings_;
  }

  [[nodiscard]] lpp::Result<
      std::shared_ptr<const lpp::ImmutableContractObject>>
  Resolve(const lpp::ContentRef&,
          lpp::ContractObjectKind) const override {
    return lpp::Error{
        .code = lpp::ErrorCode::kMissingRegistryObject,
        .field_path = "content_ref",
        .message = "case registry has no activation object",
    };
  }

 private:
  std::shared_ptr<const lpp::ImmutableMapSnapshot> map_;
  std::shared_ptr<const lpp::SafetyCapabilityProfile> capability_;
  std::shared_ptr<const lpp::PlannerAlgorithmConfig> config_;
  std::shared_ptr<const lpp::LearnedCostSnapshot> learned_;
  lpp::ResolvedCapabilityBindings bindings_;
};

[[nodiscard]] std::filesystem::path FixturePath(
    std::string_view name) {
  std::filesystem::path source{__FILE__};
  if (source.is_relative()) {
    source = std::filesystem::absolute(source);
  }
  return source.parent_path().parent_path() / "fixtures" /
         std::filesystem::path{name};
}

[[nodiscard]] std::string ReadText(
    const std::filesystem::path& path) {
  std::ifstream input{path, std::ios::binary};
  if (!input.is_open()) {
    throw std::runtime_error{"cannot open fixture: " +
                             path.string()};
  }
  return {std::istreambuf_iterator<char>{input},
          std::istreambuf_iterator<char>{}};
}

[[nodiscard]] lpp::PlanningRequest LoadCaseARequest() {
  const CaseRegistry registry;
  auto decoded = lpp::JsonCodec::DecodePlanningRequest(
      ReadText(FixturePath("shared_core_case_a.json")), registry);
  if (!lpp::IsOk(decoded)) {
    const auto& error = std::get<lpp::Error>(decoded);
    throw std::runtime_error{error.field_path + ": " + error.message};
  }
  return std::get<lpp::PlanningRequest>(std::move(decoded));
}

struct FakeState final {
  lpp::StateKey key{};
  lpp::Cell cell;
};

struct FakeEdge final {
  lpp::DurationNanoseconds duration;
  lpp::SecondaryCostVector secondary_costs;
  std::vector<lpp::Cell> sampled_cells;
  std::vector<lpp::PathControlPoint> centerline;
};

class FakeGroundAdapter final {
 public:
  FakeGroundAdapter(const lpp::SafeProjection& projection,
                    lpp::Cell start,
                    lpp::Cell terminal)
      : projection_{projection},
        start_{start},
        terminal_{terminal} {
    for (std::int32_t x = start_.x; x <= terminal_.x; ++x) {
      sampled_cells_.push_back({x, start_.y});
      centerline_.push_back(
          {.s_m = static_cast<double>(x - start_.x),
           .position_m =
               {static_cast<double>(x) + 0.5,
                static_cast<double>(start_.y) + 0.5}});
    }
  }

  [[nodiscard]] lpp::StateKey Key(
      const FakeState& state) const {
    return state.key;
  }

  [[nodiscard]] std::vector<
      lpp::SearchTransition<FakeState, FakeEdge>>
  Expand(const FakeState& state) const {
    if (state.key != 0U) {
      return {};
    }
    return {
        {
            .stable_edge_id = "case-a-route-b",
            .successor = {.key = 2U, .cell = terminal_},
            .edge = MakeEdge(std::chrono::milliseconds{2050},
                             {1.0, 0.20, 0.10}),
        },
        {
            .stable_edge_id = "case-a-route-a",
            .successor = {.key = 1U, .cell = terminal_},
            .edge = MakeEdge(std::chrono::seconds{2},
                             {2.0, 0.10, 0.20}),
        },
    };
  }

  [[nodiscard]] bool HardFeasible(
      const lpp::SearchTransition<FakeState, FakeEdge>&
          transition) const {
    return transition.successor.cell == terminal_ &&
           std::ranges::all_of(
               transition.edge.sampled_cells,
               [&](lpp::Cell cell) {
                 return projection_.Known(cell) &&
                        projection_.HardFeasible(cell);
               });
  }

  [[nodiscard]] lpp::DurationNanoseconds TransitionTime(
      const lpp::SearchTransition<FakeState, FakeEdge>&
          transition) const {
    return transition.edge.duration;
  }

  [[nodiscard]] lpp::DurationNanoseconds
  AdmissibleTimeHeuristic(
      const FakeState& state,
      const lpp::SearchProblem<FakeState>& problem) const {
    static_cast<void>(problem);
    return {
        .value = state.key == 0U
                     ? std::chrono::seconds{1}
                     : std::chrono::nanoseconds{0},
    };
  }

  [[nodiscard]] lpp::SecondaryCostVector SecondaryCosts(
      const lpp::SearchTransition<FakeState, FakeEdge>&
          transition) const {
    return transition.edge.secondary_costs;
  }

  [[nodiscard]] bool IsTerminal(
      const FakeState& state,
      const lpp::SearchProblem<FakeState>& problem) const {
    static_cast<void>(problem);
    return state.key != 0U && state.cell == terminal_;
  }

 private:
  [[nodiscard]] FakeEdge MakeEdge(
      std::chrono::nanoseconds duration,
      lpp::SecondaryCostVector secondary_costs) const {
    return {
        .duration = {.value = duration},
        .secondary_costs = secondary_costs,
        .sampled_cells = sampled_cells_,
        .centerline = centerline_,
    };
  }

  const lpp::SafeProjection& projection_;
  lpp::Cell start_;
  lpp::Cell terminal_;
  std::vector<lpp::Cell> sampled_cells_;
  std::vector<lpp::PathControlPoint> centerline_;
};

using ProjectionCache =
    lpp::DeterministicCache<lpp::SafeProjectionCacheKey,
                            lpp::SafeProjection>;
using FakeSearchResult =
    lpp::SearchResult<FakeState, FakeEdge>;

struct SharedCoreArtifacts final {
  lpp::ValidationReport validation;
  bool projection_cache_hit{};
  std::shared_ptr<const lpp::SafeProjection> projection;
  lpp::ResolvedTerminalSet terminal;
  FakeSearchResult search;
  lpp::TimeEquivalentPool pool;
  lpp::CorridorResult corridor;
};

void AddIssue(SharedCoreArtifacts& artifacts,
              std::string field_path,
              std::string reason_code,
              std::string message) {
  artifacts.validation.issues.push_back(
      {.field_path = std::move(field_path),
       .reason_code = std::move(reason_code),
       .message = std::move(message)});
}

void AddResultError(SharedCoreArtifacts& artifacts,
                    const lpp::Error& error,
                    std::string_view reason_code) {
  AddIssue(artifacts, error.field_path, std::string{reason_code},
           error.message);
}

[[nodiscard]] lpp::SafeProjectionCacheKey ProjectionKey(
    const lpp::PlanningRequest& request) {
  const auto& map = *request.map_snapshot;
  return lpp::MakeSafeProjectionCacheKey(
      map.snapshot_ref().id, std::to_string(map.map_revision()),
      std::string{map.LayerManifestHash()},
      request.safety_capability->content_ref,
      request.algorithm_config->content_ref,
      request.learned_cost_snapshot.has_value()
          ? std::optional<lpp::ContentRef>{
                request.learned_cost_snapshot->snapshot_ref}
          : std::nullopt,
      request.frame_id, map.geometry().resolution_m,
      request.algorithm_config->error_bound_model_id);
}

[[nodiscard]] lpp::Cell StartCell(
    const lpp::PlanningRequest& request) {
  const auto& state =
      std::get<lpp::WheeledOrLeggedState>(request.current_state);
  const auto& geometry = request.map_snapshot->geometry();
  return {
      .x = static_cast<std::int32_t>(
          std::floor((state.position_m.x - geometry.origin_m.x) /
                     geometry.resolution_m)),
      .y = static_cast<std::int32_t>(
          std::floor((state.position_m.y - geometry.origin_m.y) /
                     geometry.resolution_m)),
  };
}

[[nodiscard]] std::vector<lpp::CandidateScore> CandidateScores(
    const FakeSearchResult& search) {
  std::vector<lpp::CandidateScore> scores;
  scores.reserve(search.candidates.size());
  for (const auto& candidate : search.candidates) {
    scores.push_back(
        {.stable_candidate_id = candidate.stable_path_id,
         .total_time = candidate.total_time,
         .energy = candidate.secondary_costs.energy,
         .risk = candidate.secondary_costs.risk,
         .smoothness = candidate.secondary_costs.smoothness,
         .fully_hard_validated = candidate.fully_hard_validated});
  }
  return scores;
}

[[nodiscard]] lpp::ValidationReport ValidateSharedArtifacts(
    const lpp::PlanningRequest& request,
    const SharedCoreArtifacts& artifacts) {
  lpp::ValidationReport report;
  const auto issue = [&](std::string path, std::string code,
                         std::string message) {
    report.issues.push_back(
        {.field_path = std::move(path),
         .reason_code = std::move(code),
         .message = std::move(message)});
  };

  if (!artifacts.projection) {
    issue("projection", "projection_required",
          "shared projection is required");
    return report;
  }
  const auto& projection = *artifacts.projection;
  const std::size_t count = projection.geometry.CellCount();
  if (projection.source_map() != request.map_snapshot ||
      projection.capability_ref() !=
          request.safety_capability->content_ref) {
    issue("projection", "projection_lineage_mismatch",
          "projection must join exact request objects");
  }
  const bool shapes_match =
      count != 0U && projection.known_mask.size() == count &&
      projection.hard_feasible_mask.size() == count &&
      projection.esdf_clearance_m.size() == count &&
      projection.resolved_energy.size() == count &&
      projection.resolved_nonfatal_risk.size() == count;
  if (!shapes_match) {
    issue("projection", "projection_shape_mismatch",
          "all projection layers must match the grid");
  }
  if (shapes_match) {
    for (std::size_t index = 0U; index < count; ++index) {
      if (projection.known_mask[index] == 0U &&
          projection.hard_feasible_mask[index] != 0U) {
        issue("projection.hard_feasible_mask",
              "unknown_cell_marked_feasible",
              "unknown cells cannot be executable");
        break;
      }
    }
  }
  if (projection.soft_cost_source !=
      lpp::SoftCostSource::kAnalyticPlusPinnedLearned) {
    issue("projection.soft_cost_source",
          "pinned_learned_cost_required",
          "case A must use the fixed learned snapshot");
  }
  if (artifacts.terminal.kind != lpp::TerminalKind::kSafeFrontier ||
      artifacts.terminal.candidates.empty() ||
      !artifacts.terminal.unresolved_tail.has_value()) {
    issue("terminal", "safe_frontier_required",
          "unknown-strip case must stop at a safe frontier");
  }
  for (const auto& terminal : artifacts.terminal.candidates) {
    if (!terminal.requires_zero_speed ||
        !projection.HardFeasible(terminal.cell)) {
      issue("terminal.candidates",
            "invalid_frontier_terminal",
            "frontier terminals must be hard feasible stops");
      break;
    }
  }
  if (artifacts.search.status != lpp::SearchStatus::kSolved ||
      artifacts.search.candidates.size() != 2U) {
    issue("search", "two_solved_candidates_required",
          "case A must produce two search candidates");
  }
  for (const auto& path : artifacts.search.candidates) {
    const bool endpoint_matches =
        !path.states.empty() &&
        std::ranges::any_of(
            artifacts.terminal.candidates,
            [&](const lpp::TerminalCandidate& terminal) {
              return terminal.cell == path.states.back().cell;
            });
    if (!path.fully_hard_validated || !endpoint_matches ||
        path.edges.size() != 1U) {
      issue("search.candidates",
            "invalid_search_candidate",
            "every search path must be fully validated and terminal");
      break;
    }
  }
  if (artifacts.pool.ordered_candidates.size() != 2U ||
      !std::ranges::all_of(
          artifacts.pool.ordered_candidates,
          [](const lpp::CandidateScore& candidate) {
            return candidate.fully_hard_validated;
          })) {
    issue("pool", "time_equivalent_pool_invalid",
          "both validated candidates must be time equivalent");
  }
  if (artifacts.corridor.status !=
          lpp::CorridorStatus::kCertified ||
      artifacts.corridor.cells.empty()) {
    issue("corridor", "certified_corridor_required",
          "selected ground path must have a certified corridor");
  }
  for (const auto& cell : artifacts.corridor.cells) {
    if (cell.stable_cell_id.empty() ||
        cell.half_planes.size() < 4U ||
        !std::ranges::all_of(
            cell.half_planes, [](const lpp::HalfPlane2& plane) {
              return std::isfinite(
                         plane.outward_unit_normal.x) &&
                     std::isfinite(
                         plane.outward_unit_normal.y) &&
                     std::isfinite(plane.upper_offset_m) &&
                     std::abs(std::hypot(
                                  plane.outward_unit_normal.x,
                                  plane.outward_unit_normal.y) -
                              1.0) <= 1.0e-9;
            })) {
      issue("corridor.cells", "invalid_corridor_cell",
            "corridor cells need stable canonical planes");
      break;
    }
  }
  return report;
}

[[nodiscard]] SharedCoreArtifacts RunSharedCoreWithFakeAdapter(
    const lpp::PlanningRequest& request,
    ProjectionCache& cache) {
  SharedCoreArtifacts artifacts;
  artifacts.validation =
      lpp::SemanticValidator{}.Validate(request);
  if (!artifacts.validation.ok()) {
    return artifacts;
  }

  const auto key = ProjectionKey(request);
  artifacts.projection = cache.Get(key);
  artifacts.projection_cache_hit =
      artifacts.projection != nullptr;
  if (!artifacts.projection) {
    auto built = lpp::BuildSafeProjection(
        {.map = request.map_snapshot,
         .capability = request.safety_capability,
         .algorithm_config = request.algorithm_config,
         .learned_cost =
             request.learned_cost_snapshot.has_value()
                 ? request.learned_cost_snapshot->resolved_snapshot
                 : nullptr});
    if (!lpp::IsOk(built)) {
      AddResultError(artifacts, std::get<lpp::Error>(built),
                     "safe_projection_failed");
      return artifacts;
    }
    cache.Publish(
        key, std::make_shared<const lpp::SafeProjection>(
                 std::get<lpp::SafeProjection>(std::move(built))));
    artifacts.projection = cache.Get(key);
    if (!artifacts.projection) {
      AddIssue(artifacts, "projection_cache",
               "cache_publication_missing",
               "published projection must be immediately readable");
      return artifacts;
    }
  }

  const lpp::Cell start = StartCell(request);
  auto resolved = lpp::ResolveTerminal(
      {.projection = *artifacts.projection,
       .goal_region = request.goal,
       .start_cell = start,
       .capability = *request.safety_capability,
       .algorithm_config = *request.algorithm_config});
  if (!lpp::IsOk(resolved)) {
    AddResultError(artifacts, std::get<lpp::Error>(resolved),
                   "terminal_resolution_failed");
    return artifacts;
  }
  artifacts.terminal =
      std::get<lpp::ResolvedTerminalSet>(std::move(resolved));
  if (artifacts.terminal.candidates.empty()) {
    AddIssue(artifacts, "terminal.candidates",
             "terminal_candidate_required",
             "case A requires a safe frontier candidate");
    return artifacts;
  }

  const lpp::Cell terminal =
      artifacts.terminal.candidates.front().cell;
  const FakeGroundAdapter adapter{
      *artifacts.projection, start, terminal};
  artifacts.search = lpp::RunAraStar<FakeState, FakeEdge>(
      adapter,
      {.start = {.key = 0U, .cell = start},
       .limits = request.algorithm_config->ara_star.resource_caps},
      request.algorithm_config->ara_star);
  if (artifacts.search.status != lpp::SearchStatus::kSolved) {
    AddIssue(artifacts, "search", "search_not_solved",
             artifacts.search.reason_code);
    return artifacts;
  }

  const auto scores = CandidateScores(artifacts.search);
  auto ranked = lpp::CandidateRanker::BuildTimeEquivalentPool(
      scores, request.algorithm_config->time_equivalence_tolerance);
  if (!lpp::IsOk(ranked)) {
    AddResultError(artifacts, std::get<lpp::Error>(ranked),
                   "candidate_ranking_failed");
    return artifacts;
  }
  artifacts.pool =
      std::get<lpp::TimeEquivalentPool>(std::move(ranked));

  const std::string& selected_id =
      artifacts.pool.ordered_candidates.front().stable_candidate_id;
  const auto selected = std::ranges::find_if(
      artifacts.search.candidates,
      [&](const auto& candidate) {
        return candidate.stable_path_id == selected_id;
      });
  if (selected == artifacts.search.candidates.end() ||
      selected->edges.size() != 1U) {
    AddIssue(artifacts, "pool.selected_candidate",
             "selected_candidate_missing",
             "ranked candidate must retain its validated edge");
    return artifacts;
  }

  const auto& corridor_config =
      request.algorithm_config->wheeled.corridor;
  artifacts.corridor = lpp::BuildConvexCorridor(
      {.projection = *artifacts.projection,
       .validated_centerline = selected->edges.front().centerline,
       .tightening =
           {.footprint_support_radius_m = 0.25,
            .tracking_error_bound_m = 0.10,
            .additional_margin_m = 0.05,
            .platform_half_planes = {}},
       .max_planes =
           corridor_config.maximum_halfplanes_per_region,
       .max_iterations =
           corridor_config.maximum_inflation_iterations});

  artifacts.validation =
      ValidateSharedArtifacts(request, artifacts);
  return artifacts;
}

[[nodiscard]] Json ContentRefJson(const lpp::ContentRef& ref) {
  return {
      {"id", ref.id},
      {"revision", ref.revision},
      {"content_hash", ref.content_hash},
  };
}

[[nodiscard]] std::string MaskHash(
    std::span<const std::uint8_t> mask) {
  std::string bytes;
  bytes.reserve(mask.size());
  for (const std::uint8_t byte : mask) {
    bytes.push_back(static_cast<char>(byte));
  }
  auto hash = lpp::Sha256Hex(bytes);
  if (!lpp::IsOk(hash)) {
    throw std::runtime_error{"hard-mask SHA-256 failed"};
  }
  return std::get<lpp::Sha256Digest>(std::move(hash));
}

[[nodiscard]] std::string TerminalKindName(
    lpp::TerminalKind kind) {
  switch (kind) {
    case lpp::TerminalKind::kGoal:
      return "GOAL";
    case lpp::TerminalKind::kSafeFrontier:
      return "SAFE_FRONTIER";
    case lpp::TerminalKind::kGoalInfeasible:
      return "GOAL_INFEASIBLE";
    case lpp::TerminalKind::kNoKnownSafeRoute:
      return "NO_KNOWN_SAFE_ROUTE";
  }
  throw std::runtime_error{"unknown terminal kind"};
}

[[nodiscard]] std::string SearchStatusName(
    lpp::SearchStatus status) {
  switch (status) {
    case lpp::SearchStatus::kSolved:
      return "SOLVED";
    case lpp::SearchStatus::kNoPath:
      return "NO_PATH";
    case lpp::SearchStatus::kResourceLimit:
      return "RESOURCE_LIMIT";
    case lpp::SearchStatus::kInvalidAdapterCost:
      return "INVALID_ADAPTER_COST";
  }
  throw std::runtime_error{"unknown search status"};
}

[[nodiscard]] std::string CanonicalSharedArtifactJson(
    const SharedCoreArtifacts& artifacts) {
  if (!artifacts.validation.ok() || !artifacts.projection) {
    throw std::runtime_error{
        "cannot encode invalid shared-core artifacts"};
  }

  Json terminal_candidates = Json::array();
  for (const auto& candidate : artifacts.terminal.candidates) {
    terminal_candidates.push_back(
        {{"stable_id", candidate.stable_id},
         {"cell", {candidate.cell.x, candidate.cell.y}},
         {"requires_zero_speed", candidate.requires_zero_speed}});
  }

  Json search_candidates = Json::array();
  for (const auto& candidate : artifacts.search.candidates) {
    search_candidates.push_back(
        {{"stable_id", candidate.stable_path_id},
         {"total_time_ns",
          std::to_string(candidate.total_time.value.count())},
         {"fully_hard_validated",
          candidate.fully_hard_validated}});
  }

  Json ordered_candidates = Json::array();
  for (const auto& candidate :
       artifacts.pool.ordered_candidates) {
    ordered_candidates.push_back(candidate.stable_candidate_id);
  }

  Json corridor_cells = Json::array();
  for (const auto& cell : artifacts.corridor.cells) {
    Json planes = Json::array();
    for (const auto& plane : cell.half_planes) {
      planes.push_back(
          {{"normal",
            {plane.outward_unit_normal.x,
             plane.outward_unit_normal.y}},
           {"upper_offset_m", plane.upper_offset_m}});
    }
    corridor_cells.push_back(
        {{"stable_id", cell.stable_cell_id},
         {"centerline_s_begin_m", cell.centerline_s_begin},
         {"centerline_s_end_m", cell.centerline_s_end},
         {"half_planes", std::move(planes)}});
  }

  const auto learned_ref =
      artifacts.projection->soft_cost_source ==
              lpp::SoftCostSource::kAnalyticPlusPinnedLearned
          ? std::optional<lpp::ContentRef>{
                Ref("case-a-learned-snapshot", 2U, 'f')}
          : std::nullopt;
  Json root{
      {"schema_version",
       "path-planner-v3-shared-core-evidence/v1"},
      {"safe_projection",
       {{"source_map_snapshot_ref",
         ContentRefJson(
             artifacts.projection->source_map()->snapshot_ref())},
        {"source_capability_ref",
         ContentRefJson(artifacts.projection->capability_ref())},
        {"hard_mask_sha256",
         MaskHash(artifacts.projection->hard_feasible_mask)},
        {"learned_cost_source",
         artifacts.projection->soft_cost_source ==
                 lpp::SoftCostSource::kAnalyticPlusPinnedLearned
             ? "ANALYTIC_PLUS_PINNED_LEARNED"
             : "ANALYTIC_ONLY"},
        {"learned_cost_snapshot_ref",
         learned_ref.has_value() ? ContentRefJson(*learned_ref)
                                 : Json{nullptr}}}},
      {"terminal",
       {{"kind", TerminalKindName(artifacts.terminal.kind)},
        {"reason_code", artifacts.terminal.reason_code},
        {"candidates", std::move(terminal_candidates)},
        {"unresolved_tail_reason",
         artifacts.terminal.unresolved_tail.has_value()
             ? artifacts.terminal.unresolved_tail->reason_code
             : ""}}},
      {"search",
       {{"status", SearchStatusName(artifacts.search.status)},
        {"achieved_epsilon", artifacts.search.achieved_epsilon},
        {"expanded_states", artifacts.search.expansions},
        {"candidates", std::move(search_candidates)}}},
      {"time_equivalent_pool",
       {{"minimum_validated_time_ns",
         std::to_string(
             artifacts.pool.minimum_validated_time.value.count())},
        {"tolerance_ns",
         std::to_string(artifacts.pool.tolerance.value.count())},
        {"ordered_candidate_ids",
         std::move(ordered_candidates)}}},
      {"corridor",
       {{"status", "CERTIFIED"},
        {"reason_code", artifacts.corridor.reason_code},
        {"cells", std::move(corridor_cells)}}},
  };
  auto canonical = lpp::JcsCanonicalizer::Canonicalize(root);
  if (!lpp::IsOk(canonical)) {
    const auto& error = std::get<lpp::Error>(canonical);
    throw std::runtime_error{error.field_path + ": " + error.message};
  }
  return std::get<std::string>(std::move(canonical));
}

[[nodiscard]] std::string LoadExpectedCaseAJson() {
  const Json expected = Json::parse(
      ReadText(FixturePath("shared_core_case_a.expected.json")));
  auto canonical = lpp::JcsCanonicalizer::Canonicalize(expected);
  if (!lpp::IsOk(canonical)) {
    const auto& error = std::get<lpp::Error>(canonical);
    throw std::runtime_error{error.field_path + ": " + error.message};
  }
  return std::get<std::string>(std::move(canonical));
}

[[nodiscard]] std::string FormatIssues(
    const lpp::ValidationReport& report) {
  std::string result;
  for (const auto& issue : report.issues) {
    result += issue.field_path + ":" + issue.reason_code + "\n";
  }
  return result;
}

TEST(SharedCorePipeline,
     ProducesTheSameValidatedArtifactsFromFixedInput) {
  const auto request = LoadCaseARequest();
  ProjectionCache first_cache{
      request.algorithm_config->projection_cache_capacity};
  ProjectionCache second_cache{
      request.algorithm_config->projection_cache_capacity};

  const auto first =
      RunSharedCoreWithFakeAdapter(request, first_cache);
  const auto cached =
      RunSharedCoreWithFakeAdapter(request, first_cache);
  const auto second =
      RunSharedCoreWithFakeAdapter(request, second_cache);

  ASSERT_TRUE(first.validation.ok())
      << FormatIssues(first.validation);
  ASSERT_TRUE(cached.validation.ok())
      << FormatIssues(cached.validation);
  ASSERT_TRUE(second.validation.ok())
      << FormatIssues(second.validation);
  EXPECT_FALSE(first.projection_cache_hit);
  EXPECT_TRUE(cached.projection_cache_hit);
  EXPECT_FALSE(second.projection_cache_hit);
  ASSERT_EQ(first.search.status, lpp::SearchStatus::kSolved);
  ASSERT_EQ(first.pool.ordered_candidates.size(), 2U);
  ASSERT_EQ(first.corridor.status,
            lpp::CorridorStatus::kCertified);
  EXPECT_EQ(CanonicalSharedArtifactJson(first),
            CanonicalSharedArtifactJson(cached));
  EXPECT_EQ(CanonicalSharedArtifactJson(first),
            CanonicalSharedArtifactJson(second));
  EXPECT_EQ(CanonicalSharedArtifactJson(first),
            LoadExpectedCaseAJson());
}

}  // namespace
