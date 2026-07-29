#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <ranges>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include <benchmark/benchmark.h>
#include <nlohmann/json.hpp>

#include "lunar_path_planner/v3/cache/deterministic_cache.hpp"
#include "lunar_path_planner/v3/codec/semantic_validator.hpp"
#include "lunar_path_planner/v3/corridor/convex_corridor.hpp"
#include "lunar_path_planner/v3/goal/terminal_resolver.hpp"
#include "lunar_path_planner/v3/map/safe_projection.hpp"
#include "lunar_path_planner/v3/search/ara_star.hpp"
#include "lunar_path_planner/v3/search/candidate_ranker.hpp"

namespace lpp = lunar::planning::v3;
using Json = nlohmann::json;

namespace {

constexpr std::size_t kUnknownStripWidth = 2U;
constexpr std::int64_t kP95TargetNs = 1'000'000'000LL;
constexpr std::size_t kIterationsPerProfile = 1000U;
constexpr std::string_view kProfileSchema =
    "path-planner-v3-shared-core-benchmark-profile/v1";
constexpr std::string_view kBuildId =
    "lpp-v3-cpp20-cleanroom/v1";
constexpr std::string_view kDependencyIds =
    "benchmark-1.9.5;eigen-5.0.1;nlohmann-json-3.12.0";

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

[[nodiscard]] lpp::WheeledOrLeggedErrorBounds ErrorBounds() {
  return {
      .position_bound_m = ErrorBall(0.05),
      .yaw_bound_rad = {.center = 0.0, .half_width = 0.01},
      .linear_velocity_bound_mps = ErrorBall(0.01),
      .yaw_rate_bound_radps = {.center = 0.0, .half_width = 0.01},
  };
}

[[nodiscard]] std::size_t CellIndex(std::size_t x,
                                    std::size_t y,
                                    std::size_t width) {
  return y * width + x;
}

[[nodiscard]] std::shared_ptr<const lpp::ImmutableMapSnapshot>
MakeMap(std::size_t size) {
  const std::size_t count = size * size;
  const std::string suffix = std::to_string(size);
  lpp::MapSnapshotInput input{
      .snapshot_ref =
          Ref("benchmark-map-" + suffix, 1U, 'a'),
      .map_revision = 1U,
      .immutable_data_handle = "benchmark-map-data-" + suffix,
      .source_time =
          {.clock_id = "benchmark",
           .tick = std::chrono::nanoseconds{100}},
      .bounds =
          {
              .minimum_m = {0.0, 0.0, -1.0},
              .maximum_m = {static_cast<double>(size),
                            static_cast<double>(size), 1.0},
          },
      .geometry =
          {
              .width = size,
              .height = size,
              .resolution_m = 1.0,
              .origin_m = {0.0, 0.0},
              .frame_id = "map",
          },
      .layer_manifest =
          {
              {lpp::LayerKind::kKnownMask,
               Ref("benchmark-known-" + suffix, 1U, '1')},
              {lpp::LayerKind::kElevation,
               Ref("benchmark-elevation-" + suffix, 1U, '2')},
              {lpp::LayerKind::kTerrainNormal,
               Ref("benchmark-normal-" + suffix, 1U, '3')},
              {lpp::LayerKind::kRoughness,
               Ref("benchmark-roughness-" + suffix, 1U, '4')},
              {lpp::LayerKind::kHardObstacle,
               Ref("benchmark-obstacle-" + suffix, 1U, '5')},
              {lpp::LayerKind::kConfidence,
               Ref("benchmark-confidence-" + suffix, 1U, '6')},
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
  for (std::size_t y = 0U; y < size; ++y) {
    for (std::size_t x = size - kUnknownStripWidth; x < size; ++x) {
      input.known_mask[CellIndex(x, y, size)] = 0U;
      input.confidence[CellIndex(x, y, size)] = 0.0F;
    }
  }
  auto created = lpp::ImmutableMapSnapshot::Create(input);
  if (!lpp::IsOk(created)) {
    return nullptr;
  }
  return std::get<
      std::shared_ptr<const lpp::ImmutableMapSnapshot>>(
      std::move(created));
}

[[nodiscard]] lpp::WheelMotionPrimitive Primitive(
    std::string id,
    lpp::WheelMotionPrimitive::Kind kind,
    lpp::PoseXyzYaw end,
    char digest_digit) {
  return {
      .primitive_id = std::move(id),
      .kind = kind,
      .relative_end_pose = end,
      .nominal_duration =
          {.value = std::chrono::milliseconds{500}},
      .swept_geometry_ref =
          Ref("benchmark-sweep-" +
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
      .motion_model_ref = Ref("benchmark-wheel-motion", 1U, 'b'),
      .analytic_cost_model_ref =
          Ref("benchmark-wheel-cost", 1U, 'c'),
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
      .certified_state_error_bounds = ErrorBounds(),
      .motion_primitives =
          {
              Primitive("forward-line", Kind::kDriveForwardLine,
                        {{1.0, 0.0, 0.0}, 0.0}, '0'),
              Primitive("forward-arc", Kind::kDriveForwardArc,
                        {{1.0, 0.5, 0.0}, 0.5}, '1'),
              Primitive("reverse-line", Kind::kDriveReverseLine,
                        {{-1.0, 0.0, 0.0}, 0.0}, '2'),
              Primitive("reverse-arc", Kind::kDriveReverseArc,
                        {{-1.0, -0.5, 0.0}, -0.5}, '3'),
              Primitive("spin-cw", Kind::kSpinCw,
                        {{0.0, 0.0, 0.0}, -1.0}, '4'),
              Primitive("spin-ccw", Kind::kSpinCcw,
                        {{0.0, 0.0, 0.0}, 1.0}, '5'),
              Primitive("stop-switch", Kind::kStopAndSwitch,
                        {{0.0, 0.0, 0.0}, 0.0}, '6'),
          },
  };
  return std::make_shared<const lpp::SafetyCapabilityProfile>(
      lpp::SafetyCapabilityProfile{
          .content_ref =
              Ref("benchmark-wheel-capability", 1U, 'd'),
          .content = std::move(wheel),
      });
}

[[nodiscard]] lpp::CorridorConfig CorridorConfig() {
  return {
      .maximum_regions = 64U,
      .maximum_inflation_iterations = 4096U,
      .maximum_halfplanes_per_region = 256U,
      .maximum_split_depth = 8U,
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
      .maximum_adaptive_samples = 64U,
      .minimum_parameter_step = 0.01,
      .maximum_forward_passes = 4U,
      .maximum_backward_passes = 4U,
      .enable_jerk_smoothing = false,
      .maximum_jerk_smoothing_iterations = 0U,
  };
}

[[nodiscard]] std::shared_ptr<const lpp::PlannerAlgorithmConfig>
MakeConfig() {
  return std::make_shared<const lpp::PlannerAlgorithmConfig>(
      lpp::PlannerAlgorithmConfig{
          .content_ref =
              Ref("benchmark-algorithm-config", 1U, 'e'),
          .time_equivalence_tolerance =
              {.value = std::chrono::milliseconds{100}},
          .max_input_skew =
              {.value = std::chrono::seconds{1}},
          .error_bound_model_id = "benchmark-error-bound",
          .projection_cache_capacity = 2U,
          .ara_star =
              {
                  .initial_epsilon = 2.0,
                  .epsilon_decrement = 0.5,
                  .target_epsilon = 1.0,
                  .resource_caps =
                      {
                          .maximum_expanded_states = 128U,
                          .maximum_reopened_states = 128U,
                          .maximum_generated_candidates = 4U,
                          .maximum_open_states = 128U,
                          .maximum_memory_bytes = 2'097'152U,
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
                  .maximum_graph_nodes = 128U,
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
                  .maximum_inference_evaluations = 4096U,
                  .maximum_absolute_energy_correction = 0.25,
                  .maximum_absolute_nonfatal_risk_correction =
                      0.25,
              },
          .learned_cost_model_ref =
              Ref("benchmark-learned-model", 1U, '9'),
          .deterministic_execution =
              {
                  .fixed_thread_count = 1U,
                  .stable_candidate_order = true,
                  .preallocated_memory_pools = true,
              },
      });
}

[[nodiscard]] std::shared_ptr<const lpp::LearnedCostSnapshot>
MakeLearned(const lpp::ImmutableMapSnapshot& map) {
  const std::size_t count = map.geometry().CellCount();
  lpp::LearnedCostSnapshotInput input{
      .snapshot_ref =
          Ref("benchmark-learned-" +
                  std::to_string(map.geometry().width),
              1U, 'f'),
      .model_ref = Ref("benchmark-learned-model", 1U, '9'),
      .source_map_snapshot_ref = map.snapshot_ref(),
      .source_map_revision = map.map_revision(),
      .frame_id = "map",
      .feature_contract_id = "benchmark-features",
      .output_contract_id = "benchmark-soft-cost",
      .generated_at =
          {.clock_id = "benchmark",
           .tick = std::chrono::nanoseconds{99}},
      .certified_energy_correction_bounds = {-0.25, 0.25},
      .certified_nonfatal_risk_correction_bounds = {-0.25, 0.25},
      .energy_correction = std::vector<float>(count, 0.125F),
      .nonfatal_risk_correction =
          std::vector<float>(count, 0.0625F),
  };
  auto created = lpp::LearnedCostSnapshot::Create(input, count);
  return lpp::IsOk(created)
             ? std::get<
                   std::shared_ptr<const lpp::LearnedCostSnapshot>>(
                   std::move(created))
             : nullptr;
}

template <class T>
[[nodiscard]] std::shared_ptr<const T> OpaqueObject() {
  auto owner = std::make_shared<const std::max_align_t>();
  const auto* opaque =
      reinterpret_cast<const T*>(owner.get());
  return std::shared_ptr<const T>{std::move(owner), opaque};
}

struct BenchmarkScenario final {
  lpp::PlanningRequest request;
  std::size_t map_size{};
};

[[nodiscard]] BenchmarkScenario MakeScenario(std::size_t size) {
  auto map = MakeMap(size);
  auto capability = MakeCapability();
  auto config = MakeConfig();
  auto learned = map ? MakeLearned(*map) : nullptr;
  const std::int32_t row =
      static_cast<std::int32_t>(size / 2U);
  const auto& wheel =
      std::get<lpp::WheeledCapability>(capability->content);
  lpp::ResolvedCapabilityBindings bindings{
      .motion_model =
          {.content_ref = wheel.motion_model_ref,
           .object = OpaqueObject<lpp::MotionModel>()},
      .analytic_cost_model =
          {.content_ref = wheel.analytic_cost_model_ref,
           .object = OpaqueObject<lpp::AnalyticCostModel>()},
  };
  lpp::PlanningRequest request{
      .request_id = "shared-core-benchmark-" + std::to_string(size),
      .request_time =
          {.clock_id = "benchmark",
           .tick = std::chrono::nanoseconds{100}},
      .state_time =
          {.clock_id = "benchmark",
           .tick = std::chrono::nanoseconds{100}},
      .frame_id = "map",
      .platform_type = lpp::PlatformType::kWheeled,
      .current_state =
          lpp::WheeledOrLeggedState{
              .position_m =
                  {1.5, static_cast<double>(row) + 0.5, 0.0},
              .yaw_rad = 0.0,
              .linear_velocity_mps = {0.0, 0.0, 0.0},
              .yaw_rate_radps = 0.0,
              .error_bounds = ErrorBounds(),
          },
      .goal =
          {
              .goal_id = "benchmark-goal",
              .target =
                  lpp::PointGoal{
                      .position_m =
                          {static_cast<double>(size) - 0.5,
                           static_cast<double>(row) + 0.5, 0.0},
                      .position_tolerance_m = 0.10,
                  },
          },
      .map_snapshot = std::move(map),
      .safety_capability = std::move(capability),
      .algorithm_config = std::move(config),
      .capability_bindings = std::move(bindings),
      .learned_cost_snapshot =
          lpp::LearnedCostSnapshotBinding{
              .snapshot_ref = learned->snapshot_ref(),
              .registry_handle = "benchmark-learned-data",
              .resolved_snapshot = std::move(learned),
          },
  };
  return {.request = std::move(request), .map_size = size};
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

class FakeAdapter final {
 public:
  FakeAdapter(const lpp::SafeProjection& projection,
              lpp::Cell start,
              lpp::Cell terminal)
      : projection_{projection},
        terminal_{terminal} {
    for (std::int32_t x = start.x; x <= terminal.x; ++x) {
      cells_.push_back({x, start.y});
      centerline_.push_back(
          {.s_m = static_cast<double>(x - start.x),
           .position_m =
               {static_cast<double>(x) + 0.5,
                static_cast<double>(start.y) + 0.5}});
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
        {.stable_edge_id = "benchmark-route-b",
         .successor = {.key = 2U, .cell = terminal_},
         .edge = Edge(std::chrono::milliseconds{2050},
                      {1.0, 0.20, 0.10})},
        {.stable_edge_id = "benchmark-route-a",
         .successor = {.key = 1U, .cell = terminal_},
         .edge = Edge(std::chrono::seconds{2},
                      {2.0, 0.10, 0.20})},
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
    return {.value =
                state.key == 0U ? std::chrono::seconds{1}
                                : std::chrono::nanoseconds{0}};
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
  [[nodiscard]] FakeEdge Edge(
      std::chrono::nanoseconds duration,
      lpp::SecondaryCostVector costs) const {
    return {
        .duration = {.value = duration},
        .secondary_costs = costs,
        .sampled_cells = cells_,
        .centerline = centerline_,
    };
  }

  const lpp::SafeProjection& projection_;
  lpp::Cell terminal_;
  std::vector<lpp::Cell> cells_;
  std::vector<lpp::PathControlPoint> centerline_;
};

using ProjectionCache =
    lpp::DeterministicCache<lpp::SafeProjectionCacheKey,
                            lpp::SafeProjection>;
using SearchResult = lpp::SearchResult<FakeState, FakeEdge>;

[[nodiscard]] lpp::SafeProjectionCacheKey CacheKey(
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
  return {
      .x = static_cast<std::int32_t>(std::floor(state.position_m.x)),
      .y = static_cast<std::int32_t>(std::floor(state.position_m.y)),
  };
}

[[nodiscard]] std::vector<lpp::CandidateScore> Scores(
    const SearchResult& search) {
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

struct PipelineSummary final {
  bool valid{};
  bool cache_hit{};
  std::size_t candidate_count{};
  std::size_t corridor_cell_count{};
  std::size_t expanded_states{};
};

[[nodiscard]] PipelineSummary RunPlannerCall(
    const lpp::PlanningRequest& request,
    ProjectionCache& cache) {
  if (!lpp::SemanticValidator{}.Validate(request).ok()) {
    return {};
  }

  const auto key = CacheKey(request);
  auto projection = cache.Get(key);
  const bool cache_hit = projection != nullptr;
  if (!projection) {
    auto built = lpp::BuildSafeProjection(
        {.map = request.map_snapshot,
         .capability = request.safety_capability,
         .algorithm_config = request.algorithm_config,
         .learned_cost =
             request.learned_cost_snapshot->resolved_snapshot});
    if (!lpp::IsOk(built)) {
      return {};
    }
    cache.Publish(
        key, std::make_shared<const lpp::SafeProjection>(
                 std::get<lpp::SafeProjection>(std::move(built))));
    projection = cache.Get(key);
    if (!projection) {
      return {};
    }
  }

  const lpp::Cell start = StartCell(request);
  auto terminals = lpp::ResolveTerminal(
      {.projection = *projection,
       .goal_region = request.goal,
       .start_cell = start,
       .capability = *request.safety_capability,
       .algorithm_config = *request.algorithm_config});
  if (!lpp::IsOk(terminals)) {
    return {};
  }
  const auto& resolved =
      std::get<lpp::ResolvedTerminalSet>(terminals);
  if (resolved.kind != lpp::TerminalKind::kSafeFrontier ||
      resolved.candidates.empty()) {
    return {};
  }

  const lpp::Cell terminal = resolved.candidates.front().cell;
  const FakeAdapter adapter{*projection, start, terminal};
  const SearchResult search =
      lpp::RunAraStar<FakeState, FakeEdge>(
          adapter,
          {.start = {.key = 0U, .cell = start},
           .limits =
               request.algorithm_config->ara_star.resource_caps},
          request.algorithm_config->ara_star);
  if (search.status != lpp::SearchStatus::kSolved ||
      search.candidates.size() != 2U) {
    return {};
  }

  const auto scores = Scores(search);
  auto pool = lpp::CandidateRanker::BuildTimeEquivalentPool(
      scores, request.algorithm_config->time_equivalence_tolerance);
  if (!lpp::IsOk(pool)) {
    return {};
  }
  const auto& ranked = std::get<lpp::TimeEquivalentPool>(pool);
  if (ranked.ordered_candidates.size() != 2U) {
    return {};
  }
  const auto selected = std::ranges::find_if(
      search.candidates,
      [&](const auto& candidate) {
        return candidate.stable_path_id ==
               ranked.ordered_candidates.front()
                   .stable_candidate_id;
      });
  if (selected == search.candidates.end() ||
      selected->edges.size() != 1U) {
    return {};
  }

  const auto& corridor_config =
      request.algorithm_config->wheeled.corridor;
  const auto corridor = lpp::BuildConvexCorridor(
      {.projection = *projection,
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
  const bool final_invariants =
      projection->source_map() == request.map_snapshot &&
      projection->capability_ref() ==
          request.safety_capability->content_ref &&
      projection->soft_cost_source ==
          lpp::SoftCostSource::kAnalyticPlusPinnedLearned &&
      corridor.status == lpp::CorridorStatus::kCertified &&
      !corridor.cells.empty() &&
      std::ranges::all_of(
          search.candidates, [](const auto& candidate) {
            return candidate.fully_hard_validated;
          });
  return {
      .valid = final_invariants,
      .cache_hit = cache_hit,
      .candidate_count = ranked.ordered_candidates.size(),
      .corridor_cell_count = corridor.cells.size(),
      .expanded_states = search.expansions,
  };
}

[[nodiscard]] bool WarmProjectionCache(
    const lpp::PlanningRequest& request,
    ProjectionCache& cache) {
  auto built = lpp::BuildSafeProjection(
      {.map = request.map_snapshot,
       .capability = request.safety_capability,
       .algorithm_config = request.algorithm_config,
       .learned_cost =
           request.learned_cost_snapshot->resolved_snapshot});
  if (!lpp::IsOk(built)) {
    return false;
  }
  cache.Publish(
      CacheKey(request),
      std::make_shared<const lpp::SafeProjection>(
          std::get<lpp::SafeProjection>(std::move(built))));
  return true;
}

struct SampleSeries final {
  std::size_t map_size{};
  bool cache_hit{};
  std::vector<std::int64_t> durations_ns;
};

std::mutex g_samples_mutex;
std::map<std::string, SampleSeries> g_samples;

void RecordSample(std::string_view profile,
                  std::size_t map_size,
                  bool cache_hit,
                  std::int64_t duration_ns) {
  const std::lock_guard lock{g_samples_mutex};
  auto [entry, inserted] = g_samples.try_emplace(
      std::string{profile},
      SampleSeries{.map_size = map_size,
                   .cache_hit = cache_hit,
                   .durations_ns = {}});
  static_cast<void>(inserted);
  entry->second.durations_ns.push_back(duration_ns);
}

void RunProfile(benchmark::State& state,
                std::string_view profile,
                std::size_t map_size,
                bool expect_cache_hit) {
  const BenchmarkScenario scenario = MakeScenario(map_size);
  ProjectionCache hit_cache{
      scenario.request.algorithm_config->projection_cache_capacity};
  if (expect_cache_hit &&
      !WarmProjectionCache(scenario.request, hit_cache)) {
    state.SkipWithError("failed to warm projection cache");
    return;
  }

  PipelineSummary last;
  for (auto iteration : state) {
    static_cast<void>(iteration);
    std::unique_ptr<ProjectionCache> miss_cache;
    ProjectionCache* active_cache = &hit_cache;
    if (!expect_cache_hit) {
      miss_cache = std::make_unique<ProjectionCache>(
          scenario.request.algorithm_config
              ->projection_cache_capacity);
      active_cache = miss_cache.get();
    }

    const auto started = std::chrono::steady_clock::now();
    last = RunPlannerCall(scenario.request, *active_cache);
    const auto finished = std::chrono::steady_clock::now();
    const auto elapsed =
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            finished - started);
    state.SetIterationTime(
        std::chrono::duration<double>(elapsed).count());
    RecordSample(profile, map_size, expect_cache_hit,
                 elapsed.count());
    benchmark::DoNotOptimize(last.candidate_count);
    benchmark::ClobberMemory();
    if (!last.valid ||
        last.cache_hit != expect_cache_hit) {
      state.SkipWithError(
          "shared-core pipeline invariant failed");
      break;
    }
  }

  state.counters["map_width"] = static_cast<double>(map_size);
  state.counters["map_height"] = static_cast<double>(map_size);
  state.counters["map_cells"] =
      static_cast<double>(map_size * map_size);
  state.counters["candidate_count"] =
      static_cast<double>(last.candidate_count);
  state.counters["corridor_cell_count"] =
      static_cast<double>(last.corridor_cell_count);
  state.counters["expanded_states"] =
      static_cast<double>(last.expanded_states);
  state.counters["max_expanded_states"] =
      static_cast<double>(
          scenario.request.algorithm_config->ara_star.resource_caps
              .maximum_expanded_states);
  state.counters["max_open_states"] =
      static_cast<double>(
          scenario.request.algorithm_config->ara_star.resource_caps
              .maximum_open_states);
  state.counters["fixed_thread_count"] =
      static_cast<double>(
          scenario.request.algorithm_config
              ->deterministic_execution.fixed_thread_count);
  state.counters["cache_hit_profile"] =
      expect_cache_hit ? 1.0 : 0.0;
  state.counters["benchmark_profile_schema_revision"] = 1.0;
  state.SetLabel(
      std::string{"profile_schema="} +
      std::string{kProfileSchema} + ";build_id=" +
      std::string{kBuildId} + ";dependencies=" +
      std::string{kDependencyIds});
}

void SmallCacheMiss(benchmark::State& state) {
  RunProfile(state, "SmallCacheMiss", 8U, false);
}

void SmallCacheHit(benchmark::State& state) {
  RunProfile(state, "SmallCacheHit", 8U, true);
}

void MediumCacheMiss(benchmark::State& state) {
  RunProfile(state, "MediumCacheMiss", 16U, false);
}

void MediumCacheHit(benchmark::State& state) {
  RunProfile(state, "MediumCacheHit", 16U, true);
}

void LargeCacheMiss(benchmark::State& state) {
  RunProfile(state, "LargeCacheMiss", 24U, false);
}

void LargeCacheHit(benchmark::State& state) {
  RunProfile(state, "LargeCacheHit", 24U, true);
}

#define LPP_REGISTER_SHARED_CORE_BENCHMARK(function_name) \
  BENCHMARK(function_name)                                \
      ->Iterations(kIterationsPerProfile)                 \
      ->UseManualTime()                                   \
      ->Unit(benchmark::kMillisecond)

LPP_REGISTER_SHARED_CORE_BENCHMARK(SmallCacheMiss);
LPP_REGISTER_SHARED_CORE_BENCHMARK(SmallCacheHit);
LPP_REGISTER_SHARED_CORE_BENCHMARK(MediumCacheMiss);
LPP_REGISTER_SHARED_CORE_BENCHMARK(MediumCacheHit);
LPP_REGISTER_SHARED_CORE_BENCHMARK(LargeCacheMiss);
LPP_REGISTER_SHARED_CORE_BENCHMARK(LargeCacheHit);

#undef LPP_REGISTER_SHARED_CORE_BENCHMARK

[[nodiscard]] std::int64_t QuantileNearestRank(
    std::vector<std::int64_t> sorted,
    double probability) {
  if (sorted.empty()) {
    return 0LL;
  }
  std::ranges::sort(sorted);
  const auto rank = static_cast<std::size_t>(
      std::ceil(probability *
                static_cast<double>(sorted.size())));
  const std::size_t index =
      std::min(sorted.size() - 1U,
               rank == 0U ? 0U : rank - 1U);
  return sorted[index];
}

[[nodiscard]] Json PercentileReport() {
  Json profiles = Json::array();
  const std::lock_guard lock{g_samples_mutex};
  for (const auto& [name, series] : g_samples) {
    const std::int64_t p50 =
        QuantileNearestRank(series.durations_ns, 0.50);
    const std::int64_t p95 =
        QuantileNearestRank(series.durations_ns, 0.95);
    const std::int64_t p99 =
        QuantileNearestRank(series.durations_ns, 0.99);
    profiles.push_back(
        {{"profile", name},
         {"map_size", series.map_size},
         {"cache_mode",
          series.cache_hit ? "HIT" : "MISS"},
         {"sample_count", series.durations_ns.size()},
         {"p50_ns", std::to_string(p50)},
         {"p95_ns", std::to_string(p95)},
         {"p99_ns", std::to_string(p99)},
         {"p95_latency_target_ns",
          std::to_string(kP95TargetNs)},
         {"p95_latency_target_met", p95 < kP95TargetNs}});
  }
  return {
      {"schema_version",
       "path-planner-v3-shared-core-benchmark-percentiles/v1"},
      {"metric_scope", "EXPERIMENT_ONLY"},
      {"planner_result_independent_of_latency_target", true},
      {"benchmark_profile_schema", kProfileSchema},
      {"build_id", kBuildId},
      {"dependency_ids", kDependencyIds},
      {"profiles", std::move(profiles)},
  };
}

[[nodiscard]] std::optional<std::string> BenchmarkOutputPath(
    int argc,
    char** argv) {
  constexpr std::string_view prefix = "--benchmark_out=";
  for (int index = 1; index < argc; ++index) {
    const std::string_view argument{argv[index]};
    if (argument.starts_with(prefix)) {
      return std::string{argument.substr(prefix.size())};
    }
    if (argument == "--benchmark_out" && index + 1 < argc) {
      return std::string{argv[index + 1]};
    }
  }
  return std::nullopt;
}

[[nodiscard]] bool WritePercentileReport(
    const std::optional<std::string>& benchmark_output) {
  const std::string payload = PercentileReport().dump(2);
  if (!benchmark_output.has_value()) {
    std::cout << payload << '\n';
    return true;
  }
  const std::string report_path =
      *benchmark_output + ".percentiles.json";
  std::ofstream output{report_path, std::ios::binary};
  if (!output.is_open()) {
    std::cerr << "cannot write percentile report: "
              << report_path << '\n';
    return false;
  }
  output << payload << '\n';
  return output.good();
}

}  // namespace

int main(int argc, char** argv) {
  const auto benchmark_output = BenchmarkOutputPath(argc, argv);
  benchmark::Initialize(&argc, argv);
  if (benchmark::ReportUnrecognizedArguments(argc, argv)) {
    return 1;
  }
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return WritePercentileReport(benchmark_output) ? 0 : 2;
}
