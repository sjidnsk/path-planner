#include "fixtures/system/planner_v3_system_fixture.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <numbers>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "lunar_path_planner/v3/crypto/sha256.hpp"
#include "lunar_path_planner/v3/hopper/hopper_config.hpp"

namespace lunar::planning::v3::system_test {
namespace {

using namespace std::chrono_literals;

constexpr std::size_t kGroundWidth = 12U;
constexpr std::size_t kGroundHeight = 8U;

[[nodiscard]] ContentRef Ref(
    std::string id, const char digit) {
  return {
      .id = std::move(id),
      .revision = 1U,
      .content_hash = std::string(64U, digit),
  };
}

[[nodiscard]] AxisAlignedBox3 ZeroBox() {
  return {
      .center = {},
      .half_extent = {},
  };
}

[[nodiscard]] WheeledOrLeggedErrorBounds ZeroGroundError() {
  return {
      .position_bound_m = ZeroBox(),
      .yaw_bound_rad = {},
      .linear_velocity_bound_mps = ZeroBox(),
      .yaw_rate_bound_radps = {},
  };
}

[[nodiscard]] HopperErrorBounds ZeroHopperError() {
  return {
      .position_bound_m = ZeroBox(),
      .orientation_bound = {},
      .linear_velocity_bound_mps = ZeroBox(),
      .angular_velocity_bound_radps = ZeroBox(),
  };
}

[[nodiscard]] ConvexPolytope3 BoxPolytope(
    const double half_x, const double half_y,
    const double minimum_z, const double maximum_z) {
  return {
      .halfspaces =
          {
              {{1.0, 0.0, 0.0}, half_x},
              {{-1.0, 0.0, 0.0}, half_x},
              {{0.0, 1.0, 0.0}, half_y},
              {{0.0, -1.0, 0.0}, half_y},
              {{0.0, 0.0, 1.0}, maximum_z},
              {{0.0, 0.0, -1.0}, -minimum_z},
          },
  };
}

template <class T>
[[nodiscard]] std::shared_ptr<const T> OpaqueResolvedObject() {
  auto owner = std::make_shared<int>(1);
  const auto* opaque =
      reinterpret_cast<const T*>(owner.get());
  return std::shared_ptr<const T>(
      std::move(owner), opaque);
}

[[nodiscard]] MapSnapshotInput BaseMapInput(
    const PlatformType platform_type,
    const MapScenario map_scenario) {
  const bool hopper = platform_type == PlatformType::kHopper;
  const std::size_t width = hopper ? 16U : kGroundWidth;
  const std::size_t height = hopper ? 12U : kGroundHeight;
  const double resolution_m = hopper ? 0.5 : 1.0;
  const std::size_t count = width * height;
  const char platform_digit =
      platform_type == PlatformType::kWheeled
          ? '1'
          : (platform_type == PlatformType::kLegged ? '2' : '3');

  MapSnapshotInput input{
      .snapshot_ref =
          Ref("system-map-" +
                  std::to_string(
                      static_cast<int>(platform_type)) +
                  "-" +
                  std::to_string(
                      static_cast<int>(map_scenario)),
              platform_digit),
      .map_revision = 1U,
      .immutable_data_handle =
          "system-map-handle-" +
          std::to_string(static_cast<int>(platform_type)) +
          "-" +
          std::to_string(static_cast<int>(map_scenario)),
      .source_time =
          ClockStamp{"mission", std::chrono::nanoseconds{100}},
      .bounds =
          {
              .minimum_m = {0.0, 0.0, -2.0},
              .maximum_m =
                  {
                      static_cast<double>(width) * resolution_m,
                      static_cast<double>(height) * resolution_m,
                      8.0,
                  },
          },
      .geometry =
          {
              .width = width,
              .height = height,
              .resolution_m = resolution_m,
              .origin_m = {0.0, 0.0},
              .frame_id = "map",
          },
      .layer_manifest =
          {
              {LayerKind::kKnownMask, Ref("system-known", '4')},
              {LayerKind::kElevation, Ref("system-elevation", '5')},
              {LayerKind::kTerrainNormal, Ref("system-normal", '6')},
              {LayerKind::kRoughness, Ref("system-roughness", '7')},
              {LayerKind::kHardObstacle, Ref("system-obstacle", '8')},
              {LayerKind::kConfidence, Ref("system-confidence", '9')},
          },
      .known_mask = std::vector<std::uint8_t>(count, 1U),
      .elevation_m = std::vector<float>(count, 0.0F),
      .normal_x = std::vector<float>(count, 0.0F),
      .normal_y = std::vector<float>(count, 0.0F),
      .normal_z = std::vector<float>(count, 1.0F),
      .roughness_m = std::vector<float>(count, 0.0F),
      .hard_obstacle_mask =
          std::vector<std::uint8_t>(count, 0U),
      .confidence = std::vector<float>(count, 1.0F),
  };

  if (!hopper &&
      map_scenario == MapScenario::kKnownObstacleAtGoal) {
    input.hard_obstacle_mask[3U * width + 4U] = 1U;
  }
  if (!hopper &&
      map_scenario ==
          MapScenario::kUnknownGoalWithSafeFrontier) {
    for (std::size_t y = 0U; y < height; ++y) {
      for (std::size_t x = 6U; x < width; ++x) {
        const std::size_t index = y * width + x;
        input.known_mask[index] = 0U;
        input.confidence[index] = 0.0F;
      }
    }
  }
  return input;
}

[[nodiscard]] std::shared_ptr<const ImmutableMapSnapshot>
MakeMap(
    const PlatformType platform_type,
    const MapScenario map_scenario) {
  auto created = ImmutableMapSnapshot::Create(
      BaseMapInput(platform_type, map_scenario));
  if (!IsOk(created)) {
    const Error& error = std::get<Error>(created);
    throw std::runtime_error{
        error.field_path + ": " + error.message};
  }
  return std::get<
      std::shared_ptr<const ImmutableMapSnapshot>>(
      std::move(created));
}

[[nodiscard]] WheelMotionPrimitive WheelPrimitive(
    std::string id, const WheelMotionPrimitive::Kind kind,
    const PoseXyzYaw relative_end_pose,
    const std::chrono::nanoseconds duration,
    const char digest_digit) {
  return {
      .primitive_id = std::move(id),
      .kind = kind,
      .relative_end_pose = relative_end_pose,
      .nominal_duration = DurationNanoseconds{duration},
      .swept_geometry_ref =
          Ref(
              "system-wheel-sweep-" +
                  std::string(1U, digest_digit),
              digest_digit),
  };
}

[[nodiscard]] SafetyCapabilityProfile WheelCapability() {
  using Kind = WheelMotionPrimitive::Kind;
  return {
      .content_ref = Ref("system-wheel-capability", 'a'),
      .content =
          WheeledCapability{
              .frame_id = "map",
              .collision_envelope =
                  {
                      .vertices_xy_m =
                          {
                              {-0.2, -0.2},
                              {0.2, -0.2},
                              {0.2, 0.2},
                              {-0.2, 0.2},
                          },
                      .minimum_z_m = -0.1,
                      .maximum_z_m = 0.5,
                  },
              .motion_model_ref =
                  Ref("system-wheel-motion", 'b'),
              .analytic_cost_model_ref =
                  Ref("system-wheel-cost", 'c'),
              .hard_limits =
                  {
                      .maximum_forward_speed_mps = 1.0,
                      .maximum_reverse_speed_mps = 0.8,
                      .maximum_spin_rate_radps = 1.0,
                      .maximum_forward_acceleration_mps2 = 1.0,
                      .maximum_braking_deceleration_mps2 = 1.0,
                      .maximum_yaw_acceleration_radps2 = 1.0,
                      .maximum_lateral_acceleration_mps2 = 1.0,
                      .maximum_drive_curvature_per_m = 1.0,
                      .maximum_slope_rad = 0.5,
                      .minimum_clearance_m = 0.0,
                  },
              .certified_state_error_bounds =
                  ZeroGroundError(),
              .motion_primitives =
                  {
                      WheelPrimitive(
                          "forward", Kind::kDriveForwardLine,
                          {{1.0, 0.0, 0.0}, 0.0}, 1s, '0'),
                      WheelPrimitive(
                          "forward-arc", Kind::kDriveForwardArc,
                          {{1.0, 0.0, 0.0},
                           std::numbers::pi / 2.0},
                          1s, '1'),
                      WheelPrimitive(
                          "reverse", Kind::kDriveReverseLine,
                          {{-1.0, 0.0, 0.0}, 0.0}, 1s, '2'),
                      WheelPrimitive(
                          "reverse-arc", Kind::kDriveReverseArc,
                          {{-1.0, 0.0, 0.0},
                           -std::numbers::pi / 2.0},
                          1s, '3'),
                      WheelPrimitive(
                          "spin-cw", Kind::kSpinCw,
                          {{0.0, 0.0, 0.0},
                           -std::numbers::pi / 2.0},
                          500ms, '4'),
                      WheelPrimitive(
                          "spin-ccw", Kind::kSpinCcw,
                          {{0.0, 0.0, 0.0},
                           std::numbers::pi / 2.0},
                          500ms, '5'),
                      WheelPrimitive(
                          "stop-switch", Kind::kStopAndSwitch,
                          {{0.0, 0.0, 0.0}, 0.0},
                          100ms, '6'),
                  },
          },
  };
}

[[nodiscard]] LeggedBodyPrimitive LeggedPrimitive(
    std::string id, const LeggedBodyPrimitive::Kind kind,
    const Vec3 displacement, const double yaw_change_rad,
    const char digest_digit) {
  return {
      .primitive_id = std::move(id),
      .kind = kind,
      .body_frame_displacement_m = displacement,
      .yaw_change_rad = yaw_change_rad,
      .nominal_duration = DurationNanoseconds{2s},
      .sampled_body_sweep_ref =
          Ref(
              "system-legged-sweep-" +
                  std::string(1U, digest_digit),
              digest_digit),
  };
}

[[nodiscard]] SafetyCapabilityProfile LeggedCapabilityProfile() {
  using Kind = LeggedBodyPrimitive::Kind;
  return {
      .content_ref = Ref("system-legged-capability", 'd'),
      .content =
          LeggedCapability{
              .frame_id = "map",
              .reference_point_id = "base_link",
              .reference_point_definition =
                  LeggedCapability::ReferencePointDefinition::
                      kFixedNominalCom,
              .collision_envelope =
                  {
                      .body_frame_halfspaces =
                          BoxPolytope(
                              0.2, 0.2, -0.3, 0.3),
                  },
              .motion_model_ref =
                  Ref("system-legged-motion", 'e'),
              .analytic_cost_model_ref =
                  Ref("system-legged-cost", 'f'),
              .terrain_thresholds =
                  {
                      .maximum_slope_rad = 0.5,
                      .maximum_roughness_m = 0.2,
                      .maximum_step_height_m = 0.3,
                      .maximum_gap_width_m = 0.4,
                      .minimum_confidence = 0.8,
                      .minimum_body_clearance_m = 0.0,
                      .minimum_body_height_m = 0.4,
                      .maximum_body_height_m = 0.6,
                  },
              .body_velocity_limits =
                  {
                      .forward_mps = {-0.5, 0.5},
                      .lateral_mps = {-0.5, 0.5},
                      .vertical_mps = {-0.1, 0.1},
                      .yaw_rate_radps = {-1.0, 1.0},
                      .linear_acceleration_mps2 = 0.5,
                      .yaw_acceleration_radps2 = 1.0,
                  },
              .certified_state_error_bounds =
                  ZeroGroundError(),
              .motion_primitives =
                  {
                      LeggedPrimitive(
                          "forward", Kind::kForward,
                          {1.0, 0.0, 0.0}, 0.0, '1'),
                      LeggedPrimitive(
                          "backward", Kind::kBackward,
                          {-1.0, 0.0, 0.0}, 0.0, '2'),
                      LeggedPrimitive(
                          "left", Kind::kLateral,
                          {0.0, 1.0, 0.0}, 0.0, '3'),
                      LeggedPrimitive(
                          "right", Kind::kLateral,
                          {0.0, -1.0, 0.0}, 0.0, '4'),
                      LeggedPrimitive(
                          "spin", Kind::kSpin,
                          {0.0, 0.0, 0.0},
                          std::numbers::pi / 2.0, '5'),
                  },
              .feasibility_scope =
                  LeggedCapability::FeasibilityScope::
                      kBodyGeometryAndTerrainThresholdsOnly,
              .footstep_feasibility_guaranteed = false,
          },
  };
}

[[nodiscard]] SafetyCapabilityProfile HopperCapabilityProfile() {
  return {
      .content_ref = Ref("system-hopper-capability", '1'),
      .content =
          HopperCapability{
              .frame_id = "map",
              .collision_envelope =
                  {
                      .body_frame_halfspaces =
                          BoxPolytope(
                              0.35, 0.25, -0.5, 0.5),
                  },
              .motion_model_ref =
                  Ref("system-hopper-motion", '2'),
              .analytic_cost_model_ref =
                  Ref("system-hopper-cost", '3'),
              .gravity_model_ref =
                  Ref("system-lunar-gravity", '4'),
              .landing_terrain_thresholds =
                  {
                      .maximum_slope_rad = 0.55,
                      .maximum_roughness_m = 0.1,
                      .maximum_plane_residual_m = 0.05,
                      .minimum_overhead_clearance_m = 0.0,
                      .minimum_lateral_clearance_m = 0.0,
                      .minimum_landing_region_area_m2 = 0.2,
                  },
              .launch_limits =
                  {
                      .maximum_launch_speed_mps = 8.0,
                      .maximum_launch_impulse_newton_seconds = 100.0,
                      .minimum_flight_time =
                          DurationNanoseconds{500ms},
                      .maximum_flight_time =
                          DurationNanoseconds{10s},
                      .maximum_landing_speed_mps = 8.0,
                      .minimum_downward_impact_speed_mps = 0.1,
                      .minimum_landing_clearance_m = 0.0,
                  },
              .attitude_envelope =
                  {
                      .maximum_angular_speed_radps = 2.0,
                      .maximum_angular_acceleration_radps2 = 4.0,
                      .maximum_initial_angular_speed_radps = 0.2,
                      .minimum_settle_guard =
                          DurationNanoseconds{100ms},
                  },
              .certified_state_error_bounds =
                  ZeroHopperError(),
          },
  };
}

[[nodiscard]] std::shared_ptr<const SafetyCapabilityProfile>
MakeCapability(const PlatformType platform_type) {
  SafetyCapabilityProfile capability =
      platform_type == PlatformType::kWheeled
          ? WheelCapability()
          : (platform_type == PlatformType::kLegged
                 ? LeggedCapabilityProfile()
                 : HopperCapabilityProfile());
  return std::make_shared<const SafetyCapabilityProfile>(
      std::move(capability));
}

[[nodiscard]] CorridorConfig ValidCorridor() {
  return {
      .maximum_regions = 16U,
      .maximum_inflation_iterations = 64U,
      .maximum_halfplanes_per_region = 16U,
      .maximum_split_depth = 4U,
      .minimum_overlap_m = 0.05,
      .sampling_spacing_m = 0.25,
  };
}

[[nodiscard]] SmoothingConfig ValidSmoothing() {
  return {
      .maximum_scp_iterations = 2U,
      .maximum_trust_region_reductions = 2U,
      .initial_trust_region_m = 0.2,
      .minimum_trust_region_m = 0.01,
      .constraint_tolerance = 1.0e-6,
      .maximum_time_increase = DurationNanoseconds{2s},
  };
}

[[nodiscard]] TimeScalingConfig ValidTimeScaling() {
  return {
      .maximum_adaptive_samples = 128U,
      .minimum_parameter_step = 0.01,
      .maximum_forward_passes = 2U,
      .maximum_backward_passes = 2U,
      .enable_jerk_smoothing = false,
      .maximum_jerk_smoothing_iterations = 0U,
  };
}

[[nodiscard]] std::shared_ptr<const PlannerAlgorithmConfig>
MakeAlgorithmConfig(const PlatformType) {
  PlannerAlgorithmConfig config{
      .content_ref =
          Ref("system-algorithm-config", 'a'),
      .time_equivalence_tolerance =
          DurationNanoseconds{100ms},
      .max_input_skew = DurationNanoseconds{1s},
      .error_bound_model_id = "system-error-bound",
      .projection_cache_capacity = 8U,
      .ara_star =
          {
              .initial_epsilon = 2.0,
              .epsilon_decrement = 0.5,
              .target_epsilon = 1.0,
              .resource_caps =
                  {
                      .maximum_expanded_states = 2048U,
                      .maximum_reopened_states = 2048U,
                      .maximum_generated_candidates = 16U,
                      .maximum_open_states = 2048U,
                      .maximum_memory_bytes = 1U << 20U,
                  },
          },
      .wheeled =
          {
              .state_lattice =
                  {
                      .xy_resolution_m = 1.0,
                      .yaw_bin_count = 4U,
                      .maximum_terminal_candidates = 16U,
                  },
              .corridor = ValidCorridor(),
              .smoothing = ValidSmoothing(),
              .time_scaling = ValidTimeScaling(),
              .continuous_validation_maximum_subdivisions = 16U,
          },
      .legged =
          {
              .pose_lattice =
                  {
                      .xy_resolution_m = 1.0,
                      .yaw_bin_count = 4U,
                      .maximum_terminal_candidates = 16U,
                  },
              .maximum_height_interval_splits = 8U,
              .corridor = ValidCorridor(),
              .smoothing = ValidSmoothing(),
              .time_scaling = ValidTimeScaling(),
              .continuous_validation_maximum_subdivisions = 16U,
          },
      .hopper =
          {
              .maximum_landing_regions = 16U,
              .maximum_graph_nodes = 16U,
              .maximum_graph_out_degree = 4U,
              .yaw_partition_count = 4U,
              .support_direction_count = 16U,
              .maximum_nominal_aim_points_per_region = 8U,
              .maximum_full_certification_attempts = 16U,
              .maximum_interval_subdivision_depth = 8U,
              .maximum_collision_subdivision_depth = 8U,
              .maximum_root_iterations = 64U,
              .maximum_flight_tube_sections = 64U,
              .landing_region_inflation_iterations = 16U,
              .landing_region_maximum_split_depth = 4U,
              .landing_region_maximum_vertices = 16U,
          },
      .learned_cost_policy =
          {
              .mode = LearnedCostPolicy::Mode::kDisabled,
              .maximum_inference_evaluations = 0U,
              .maximum_absolute_energy_correction = 0.0,
              .maximum_absolute_nonfatal_risk_correction = 0.0,
          },
      .deterministic_execution =
          {
              .fixed_thread_count = 1U,
              .stable_candidate_order = true,
              .preallocated_memory_pools = true,
          },
  };
  return std::make_shared<const PlannerAlgorithmConfig>(
      std::move(config));
}

[[nodiscard]] ResolvedCapabilityBindings GroundBindings(
    const SafetyCapabilityProfile& capability) {
  const auto refs = std::visit(
      [](const auto& content) {
        return std::pair{
            content.motion_model_ref,
            content.analytic_cost_model_ref};
      },
      capability.content);
  return {
      .motion_model =
          {
              .content_ref = refs.first,
              .object = OpaqueResolvedObject<MotionModel>(),
          },
      .analytic_cost_model =
          {
              .content_ref = refs.second,
              .object =
                  OpaqueResolvedObject<AnalyticCostModel>(),
          },
  };
}

[[nodiscard]] ResolvedCapabilityBindings HopperBindings(
    const SafetyCapabilityProfile& capability) {
  const auto& hopper =
      std::get<HopperCapability>(capability.content);
  auto gravity = std::make_shared<const GravityModel>(
      GravityModel{
          .content_ref = hopper.gravity_model_ref,
          .frame_id = "map",
          .nominal_acceleration_mps2 = {0.0, 0.0, -1.62},
          .acceleration_error_mps2 =
              {
                  .center = {},
                  .half_extent = {0.0, 0.0, 0.01},
              },
          .spatial_validity_m =
              {
                  .center = {},
                  .half_extent =
                      {1000.0, 1000.0, 1000.0},
              },
          .valid_from =
              ClockStamp{
                  "mission", std::chrono::nanoseconds{0}},
          .valid_until =
              ClockStamp{"mission", std::chrono::hours{24}},
      });
  auto error =
      std::make_shared<const DeterministicErrorModel>(
          DeterministicErrorModel{
              .content_ref =
                  Ref("system-hopper-error", '5'),
              .initial_position_error_m =
                  {
                      .center = {},
                      .half_extent = {0.01, 0.01, 0.01},
                  },
              .initial_velocity_error_mps =
                  {
                      .center = {},
                      .half_extent = {0.01, 0.01, 0.01},
                  },
              .launch_execution_velocity_error_mps =
                  {
                      .center = {},
                      .half_extent = {0.01, 0.01, 0.01},
                  },
              .gravity_error_mps2 =
                  {
                      .center = {},
                      .half_extent = {0.0, 0.0, 0.01},
                  },
              .landing_plane_origin_error_m =
                  {
                      .center = {},
                      .half_extent = {0.01, 0.01, 0.01},
                  },
              .landing_plane_normal_error =
                  RotationVectorBall{0.01},
              .landing_plane_residual_error_m =
                  SymmetricScalarInterval{0.0, 0.01},
          });
  auto actuator =
      std::make_shared<const ActuatorOrImpulseProfile>(
          ActuatorOrImpulseProfile{
              .content_ref =
                  Ref("system-hopper-actuator", '6'),
              .platform_mass_kg = 10.0,
              .launch_preparation_time =
                  DurationNanoseconds{100ms},
              .landing_settle_time =
                  DurationNanoseconds{200ms},
              .nominal_landing_center_normal_offset_m = 0.5,
          });
  auto rotation =
      std::make_shared<const BodyRotationEnvelope>(
          BodyRotationEnvelope{
              .content_ref =
                  Ref("system-hopper-rotation", '7'),
              .arbitrary_attitude_body_envelope =
                  BoxPolytope(0.45, 0.45, -0.45, 0.45),
          });

  return {
      .motion_model =
          {
              .content_ref = hopper.motion_model_ref,
              .object = OpaqueResolvedObject<MotionModel>(),
          },
      .analytic_cost_model =
          {
              .content_ref =
                  hopper.analytic_cost_model_ref,
              .object =
                  OpaqueResolvedObject<AnalyticCostModel>(),
          },
      .gravity_model =
          ResolvedBinding<GravityModel>{
              gravity->content_ref, gravity},
      .error_model =
          ResolvedBinding<DeterministicErrorModel>{
              error->content_ref, error},
      .actuator_or_impulse_profile =
          ResolvedBinding<ActuatorOrImpulseProfile>{
              actuator->content_ref, actuator},
      .body_rotation_envelope =
          ResolvedBinding<BodyRotationEnvelope>{
              rotation->content_ref, rotation},
  };
}

[[nodiscard]] ResolvedCapabilityBindings MakeBindings(
    const SafetyCapabilityProfile& capability,
    const PlatformType platform_type) {
  return platform_type == PlatformType::kHopper
             ? HopperBindings(capability)
             : GroundBindings(capability);
}

class TestContractObject final
    : public ImmutableContractObject {
 public:
  TestContractObject(
      ContentRef ref, const ContractObjectKind kind)
      : ref_(std::move(ref)), kind_(kind) {}

  [[nodiscard]] ContentRef content_ref() const override {
    return ref_;
  }

  [[nodiscard]] ContractObjectKind kind() const override {
    return kind_;
  }

 private:
  ContentRef ref_;
  ContractObjectKind kind_;
};

class TestCertificate final : public CertificationObject {
 public:
  TestCertificate(
      ContentRef ref, CertificationProvenance provenance)
      : ref_(std::move(ref)),
        provenance_(std::move(provenance)) {}

  [[nodiscard]] ContentRef content_ref() const override {
    return ref_;
  }

  [[nodiscard]] ContractObjectKind kind() const override {
    return ContractObjectKind::kCertification;
  }

  [[nodiscard]] const CertificationProvenance& provenance()
      const override {
    return provenance_;
  }

 private:
  ContentRef ref_;
  CertificationProvenance provenance_;
};

[[nodiscard]] bool SameRef(
    const ContentRef& left, const ContentRef& right) {
  return left == right;
}

class FixedSystemRegistry final
    : public ContractObjectRegistry {
 public:
  FixedSystemRegistry(
      std::shared_ptr<const ImmutableMapSnapshot> map,
      std::shared_ptr<const SafetyCapabilityProfile> capability,
      std::shared_ptr<const PlannerAlgorithmConfig> config,
      ResolvedCapabilityBindings bindings,
      const RegistryFault fault)
      : map_(std::move(map)),
        capability_(std::move(capability)),
        config_(std::move(config)),
        bindings_(std::move(bindings)),
        fault_(fault) {}

  [[nodiscard]] std::shared_ptr<const ImmutableMapSnapshot>
  FindMapSnapshot(
      const ContentRef& ref,
      const std::string_view handle) const override {
    return map_ && SameRef(map_->snapshot_ref(), ref) &&
                   map_->immutable_data_handle() == handle
               ? map_
               : nullptr;
  }

  [[nodiscard]]
  std::shared_ptr<const SafetyCapabilityProfile>
  FindSafetyCapability(
      const ContentRef& ref) const override {
    if (fault_ == RegistryFault::kMissingCapability) {
      return nullptr;
    }
    return capability_ &&
                   SameRef(capability_->content_ref, ref)
               ? capability_
               : nullptr;
  }

  [[nodiscard]] std::shared_ptr<const PlannerAlgorithmConfig>
  FindAlgorithmConfig(
      const ContentRef& ref) const override {
    return config_ && SameRef(config_->content_ref, ref)
               ? config_
               : nullptr;
  }

  [[nodiscard]] std::shared_ptr<const LearnedCostSnapshot>
  FindLearnedCost(
      const ContentRef&, std::string_view) const override {
    return nullptr;
  }

  [[nodiscard]] Result<ResolvedCapabilityBindings>
  ResolveCapabilityBindings(
      const SafetyCapabilityProfile& profile) const override {
    if (!capability_ ||
        profile.content_ref != capability_->content_ref) {
      return Error{
          ErrorCode::kMissingRegistryObject,
          "safety_capability_ref",
          "system fixture capability is absent",
      };
    }
    ResolvedCapabilityBindings result = bindings_;
    if (fault_ ==
        RegistryFault::kCapabilityBindingsMismatch) {
      result.motion_model.content_ref =
          Ref("mismatched-motion-model", '0');
    }
    return result;
  }

  [[nodiscard]] Result<
      std::shared_ptr<const ImmutableContractObject>>
  Resolve(
      const ContentRef& ref,
      const ContractObjectKind expected_kind) const override {
    if (expected_kind ==
        ContractObjectKind::kCertification) {
      std::vector<ContentRef> inputs{
          bindings_.motion_model.content_ref,
          bindings_.analytic_cost_model.content_ref,
      };
      const auto append =
          [&inputs](const auto& optional_binding) {
            if (optional_binding.has_value()) {
              inputs.push_back(
                  optional_binding->content_ref);
            }
          };
      append(bindings_.gravity_model);
      append(bindings_.error_model);
      append(bindings_.actuator_or_impulse_profile);
      append(bindings_.body_rotation_envelope);
      append(bindings_.attitude_tightening_table);
      return std::shared_ptr<const ImmutableContractObject>(
          std::make_shared<const TestCertificate>(
              ref,
              CertificationProvenance{
                  .source_map_snapshot_ref =
                      map_->snapshot_ref(),
                  .source_safety_capability_ref =
                      capability_->content_ref,
                  .source_algorithm_config_ref =
                      config_->content_ref,
                  .input_refs = std::move(inputs),
                  .certification_purpose =
                      "system-integration-fixture",
              }));
    }

    if (bindings_.gravity_model.has_value() &&
        bindings_.gravity_model->content_ref == ref &&
        expected_kind == ContractObjectKind::kGravityModel) {
      return std::shared_ptr<const ImmutableContractObject>(
          std::make_shared<const TestContractObject>(
              ref, expected_kind));
    }
    if (bindings_.error_model.has_value() &&
        bindings_.error_model->content_ref == ref &&
        expected_kind ==
            ContractObjectKind::kDeterministicErrorModel) {
      return std::shared_ptr<const ImmutableContractObject>(
          std::make_shared<const TestContractObject>(
              ref, expected_kind));
    }
    if (bindings_.actuator_or_impulse_profile.has_value() &&
        bindings_.actuator_or_impulse_profile->content_ref == ref &&
        expected_kind ==
            ContractObjectKind::kActuatorOrImpulseProfile) {
      return std::shared_ptr<const ImmutableContractObject>(
          std::make_shared<const TestContractObject>(
              ref, expected_kind));
    }
    if (bindings_.body_rotation_envelope.has_value() &&
        bindings_.body_rotation_envelope->content_ref == ref &&
        expected_kind ==
            ContractObjectKind::kBodyRotationEnvelope) {
      return std::shared_ptr<const ImmutableContractObject>(
          std::make_shared<const TestContractObject>(
              ref, expected_kind));
    }
    if (bindings_.attitude_tightening_table.has_value() &&
        bindings_.attitude_tightening_table->content_ref == ref &&
        expected_kind ==
            ContractObjectKind::kAttitudeTighteningTable) {
      return std::shared_ptr<const ImmutableContractObject>(
          std::make_shared<const TestContractObject>(
              ref, expected_kind));
    }
    return Error{
        ErrorCode::kMissingRegistryObject,
        "content_ref",
        "system fixture contract object is absent",
    };
  }

 private:
  std::shared_ptr<const ImmutableMapSnapshot> map_;
  std::shared_ptr<const SafetyCapabilityProfile> capability_;
  std::shared_ptr<const PlannerAlgorithmConfig> config_;
  ResolvedCapabilityBindings bindings_;
  RegistryFault fault_;
};

[[nodiscard]] GoalRegion GoalFor(
    const PlatformType platform_type,
    const MapScenario map_scenario) {
  if (platform_type == PlatformType::kHopper) {
    return {
        .goal_id = "system-hopper-goal",
        .target =
            PointGoal{
                .position_m = {4.0, 3.0, 0.0},
                .position_tolerance_m = 0.5,
            },
        .optional_yaw_interval =
            CircularYawInterval{
                .start_rad = 0.0,
                .span_rad = 0.0,
            },
    };
  }
  const double goal_x =
      map_scenario ==
              MapScenario::kUnknownGoalWithSafeFrontier
          ? 8.5
          : 4.5;
  return {
      .goal_id =
          platform_type == PlatformType::kWheeled
              ? "system-wheel-goal"
              : "system-legged-goal",
      .target =
          PointGoal{
              .position_m = {goal_x, 3.5, 0.0},
              .position_tolerance_m = 0.2,
          },
  };
}

[[nodiscard]] PlatformState StateFor(
    const PlatformType platform_type) {
  if (platform_type == PlatformType::kHopper) {
    return HopperState{
        .position_m = {3.0, 3.0, 0.5},
        .orientation_body_to_frame =
            {1.0, 0.0, 0.0, 0.0},
        .linear_velocity_mps = {},
        .angular_velocity_radps = {},
        .error_bounds = ZeroHopperError(),
    };
  }
  return WheeledOrLeggedState{
      .position_m =
          {
              2.5,
              3.5,
              platform_type == PlatformType::kLegged
                  ? 0.5
                  : 0.0,
          },
      .yaw_rad = 0.0,
      .linear_velocity_mps = {},
      .yaw_rate_radps = 0.0,
      .error_bounds = ZeroGroundError(),
  };
}

}  // namespace

SystemScenario MakeSystemScenario(
    const PlatformType platform_type,
    const MapScenario map_scenario,
    const RegistryFault registry_fault) {
  auto map = MakeMap(platform_type, map_scenario);
  auto capability = MakeCapability(platform_type);
  auto config = MakeAlgorithmConfig(platform_type);
  ResolvedCapabilityBindings bindings =
      MakeBindings(*capability, platform_type);
  auto registry =
      std::make_shared<const FixedSystemRegistry>(
          map, capability, config, bindings, registry_fault);
  PlanningRequest request{
      .request_id =
          "system-request-" +
          std::to_string(static_cast<int>(platform_type)) +
          "-" +
          std::to_string(static_cast<int>(map_scenario)),
      .request_time =
          ClockStamp{
              "mission", std::chrono::nanoseconds{100}},
      .state_time =
          ClockStamp{
              "mission", std::chrono::nanoseconds{100}},
      .frame_id = "map",
      .platform_type = platform_type,
      .current_state = StateFor(platform_type),
      .goal = GoalFor(platform_type, map_scenario),
      .map_snapshot = std::move(map),
      .safety_capability = std::move(capability),
      .algorithm_config = std::move(config),
      .capability_bindings = std::move(bindings),
  };
  const std::size_t cache_capacity =
      request.algorithm_config->projection_cache_capacity;
  return {
      .request = std::move(request),
      .registry = std::move(registry),
      .projection_cache =
          std::make_unique<SafeProjectionCache>(
              cache_capacity),
  };
}

std::unique_ptr<PlannerV3> MakePlanner(
    SystemScenario& scenario) {
  static const SemanticValidator validator;
  return MakeDefaultPlannerV3(
      validator, *scenario.registry,
      *scenario.projection_cache);
}

PreviousExecutionContext MakeGroundExecutionContext(
    const SystemScenario& scenario,
    const bool stale_source_map) {
  ContentRef source_map =
      scenario.request.map_snapshot->snapshot_ref();
  if (stale_source_map) {
    source_map = Ref("stale-system-map", 'f');
  }
  return {
      .active_bundle_ref =
          Ref("active-system-bundle", 'e'),
      .active_bundle_handle = "active-system-bundle-handle",
      .commit_boundary =
          TimeCommitBoundary{
              DurationNanoseconds{500ms}},
      .execution_cursor =
          TimeExecutionCursor{
              .offset = DurationNanoseconds{100ms},
          },
      .controller_status = ControllerStatus::kExecuting,
      .source_map_snapshot_ref = std::move(source_map),
      .source_capability_ref =
          scenario.request.safety_capability->content_ref,
  };
}

Result<Sha256Digest> ResponseHash(
    const PlanningResponse& response) {
  return Sha256Hex(
      JsonCodec::EncodePlanningResponse(response));
}

std::string DescribeIssues(
    const ValidationReport& report) {
  std::ostringstream description;
  for (const ValidationIssue& issue : report.issues) {
    description << issue.field_path << " ["
                << issue.reason_code << "]: "
                << issue.message << '\n';
  }
  return description.str();
}

}  // namespace lunar::planning::v3::system_test
