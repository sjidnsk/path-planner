#include "hopper_test_fixtures.hpp"

#include <utility>

namespace lunar::planning::v3::hopper_test {

using namespace std::chrono_literals;

ContentRef Ref(std::string id, const char digit) {
  return {
      .id = std::move(id),
      .revision = 1U,
      .content_hash = std::string(64U, digit),
  };
}

AxisAlignedBox3 Box(const Vec3 center, const Vec3 half_extent) {
  return {.center = center, .half_extent = half_extent};
}

DeterministicVectorSet3 ZeroVectorSet() {
  return Box();
}

HopperErrorBounds ZeroHopperError() {
  return {
      .position_bound_m = ZeroVectorSet(),
      .orientation_bound = RotationVectorBall{0.0},
      .linear_velocity_bound_mps = ZeroVectorSet(),
      .angular_velocity_bound_radps = ZeroVectorSet(),
  };
}

ConvexPolytope3 UnitBoxPolytope(
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

SafetyCapabilityProfile ValidCapability() {
  return {
      .content_ref = Ref("hopper-capability", 'a'),
      .content =
          HopperCapability{
              .frame_id = "map",
              .collision_envelope =
                  BodyConvexPolytope{
                      .body_frame_halfspaces =
                          UnitBoxPolytope(0.35, 0.25, -0.5, 0.5),
                  },
              .motion_model_ref = Ref("hopper-motion", 'b'),
              .analytic_cost_model_ref = Ref("hopper-cost", 'c'),
              .gravity_model_ref = Ref("lunar-gravity", 'd'),
              .landing_terrain_thresholds =
                  {
                      .maximum_slope_rad = 0.55,
                      .maximum_roughness_m = 0.10,
                      .maximum_plane_residual_m = 0.05,
                      .minimum_overhead_clearance_m = 0.10,
                      .minimum_lateral_clearance_m = 0.10,
                      .minimum_landing_region_area_m2 = 0.20,
                  },
              .launch_limits =
                  {
                      .maximum_launch_speed_mps = 8.0,
                      .maximum_launch_impulse_newton_seconds = 100.0,
                      .minimum_flight_time = DurationNanoseconds{500ms},
                      .maximum_flight_time = DurationNanoseconds{10s},
                      .maximum_landing_speed_mps = 8.0,
                      .minimum_downward_impact_speed_mps = 0.1,
                      .minimum_landing_clearance_m = 0.05,
                  },
              .attitude_envelope =
                  {
                      .maximum_angular_speed_radps = 2.0,
                      .maximum_angular_acceleration_radps2 = 4.0,
                      .maximum_initial_angular_speed_radps = 0.2,
                      .minimum_settle_guard = DurationNanoseconds{100ms},
                  },
              .attitude_tightening_table_ref =
                  Ref("attitude-tightening", 'e'),
              .certified_state_error_bounds = ZeroHopperError(),
          },
  };
}

PlannerAlgorithmConfig ValidAlgorithm() {
  PlannerAlgorithmConfig config{};
  config.content_ref = Ref("algorithm", 'f');
  config.time_equivalence_tolerance = DurationNanoseconds{50ms};
  config.max_input_skew = DurationNanoseconds{1s};
  config.error_bound_model_id = "hopper-error";
  config.projection_cache_capacity = 4U;
  config.ara_star = {
      .initial_epsilon = 2.0,
      .epsilon_decrement = 0.5,
      .target_epsilon = 1.0,
      .resource_caps =
          {
              .maximum_expanded_states = 256U,
              .maximum_reopened_states = 128U,
              .maximum_generated_candidates = 32U,
              .maximum_open_states = 256U,
              .maximum_memory_bytes = 1U << 20U,
          },
  };
  config.hopper = {
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
  };
  config.deterministic_execution = {
      .fixed_thread_count = 1U,
  };
  return config;
}

ResolvedCapabilityBindings ValidBindings() {
  const auto gravity = std::make_shared<GravityModel>(
      GravityModel{
          .content_ref = Ref("lunar-gravity", 'd'),
          .frame_id = "map",
          .nominal_acceleration_mps2 = {0.0, 0.0, -1.62},
          .acceleration_error_mps2 =
              Box({}, {0.0, 0.0, 0.01}),
          .spatial_validity_m =
              Box({}, {1000.0, 1000.0, 1000.0}),
          .valid_from =
              ClockStamp{"mission", std::chrono::nanoseconds{0}},
          .valid_until =
              ClockStamp{"mission", std::chrono::hours{24}},
      });
  const auto error_model =
      std::make_shared<DeterministicErrorModel>(
          DeterministicErrorModel{
              .content_ref = Ref("hopper-error", '6'),
              .initial_position_error_m =
                  Box({}, {0.01, 0.01, 0.01}),
              .initial_velocity_error_mps =
                  Box({}, {0.01, 0.01, 0.01}),
              .launch_execution_velocity_error_mps =
                  Box({}, {0.01, 0.01, 0.01}),
              .gravity_error_mps2 =
                  Box({}, {0.0, 0.0, 0.01}),
              .landing_plane_origin_error_m =
                  Box({}, {0.01, 0.01, 0.01}),
              .landing_plane_normal_error =
                  RotationVectorBall{0.01},
              .landing_plane_residual_error_m =
                  SymmetricScalarInterval{0.0, 0.01},
          });
  const auto actuator =
      std::make_shared<ActuatorOrImpulseProfile>(
          ActuatorOrImpulseProfile{
              .content_ref = Ref("hopper-actuator", '7'),
              .platform_mass_kg = 10.0,
              .launch_preparation_time =
                  DurationNanoseconds{100ms},
              .landing_settle_time =
                  DurationNanoseconds{200ms},
              .nominal_landing_center_normal_offset_m = 0.5,
          });
  const auto rotation =
      std::make_shared<BodyRotationEnvelope>(
          BodyRotationEnvelope{
              .content_ref = Ref("hopper-rotation", '8'),
              .arbitrary_attitude_body_envelope =
                  UnitBoxPolytope(0.45, 0.45, -0.45, 0.45),
          });
  const auto table =
      std::make_shared<AttitudeTighteningTable>(
          AttitudeTighteningTable{
              .content_ref = Ref("attitude-tightening", 'e'),
              .entries =
                  {
                      {0.5, 1.8, 3.5},
                      {3.141592653589793, 1.5, 3.0},
                  },
          });

  ResolvedCapabilityBindings bindings{};
  bindings.motion_model.content_ref = Ref("hopper-motion", 'b');
  bindings.analytic_cost_model.content_ref = Ref("hopper-cost", 'c');
  bindings.gravity_model =
      ResolvedBinding<GravityModel>{gravity->content_ref, gravity};
  bindings.error_model = ResolvedBinding<DeterministicErrorModel>{
      error_model->content_ref, error_model};
  bindings.actuator_or_impulse_profile =
      ResolvedBinding<ActuatorOrImpulseProfile>{
          actuator->content_ref, actuator};
  bindings.body_rotation_envelope =
      ResolvedBinding<BodyRotationEnvelope>{
          rotation->content_ref, rotation};
  bindings.attitude_tightening_table =
      ResolvedBinding<AttitudeTighteningTable>{
          table->content_ref, table};
  return bindings;
}

std::shared_ptr<const ImmutableMapSnapshot> FlatMap(
    const std::size_t width, const std::size_t height,
    const double resolution_m) {
  const std::size_t count = width * height;
  MapSnapshotInput input{
      .snapshot_ref = Ref("hopper-map", '9'),
      .map_revision = 1U,
      .immutable_data_handle = "hopper-map-handle",
      .source_time =
          ClockStamp{"mission", std::chrono::nanoseconds{100}},
      .bounds =
          {
              .minimum_m = {0.0, 0.0, -2.0},
              .maximum_m =
                  {static_cast<double>(width) * resolution_m,
                   static_cast<double>(height) * resolution_m,
                   8.0},
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
              {LayerKind::kKnownMask, Ref("known", '1')},
              {LayerKind::kElevation, Ref("elevation", '2')},
              {LayerKind::kTerrainNormal, Ref("normal", '3')},
              {LayerKind::kRoughness, Ref("roughness", '4')},
              {LayerKind::kHardObstacle, Ref("obstacle", '5')},
              {LayerKind::kConfidence, Ref("confidence", '6')},
              {LayerKind::kEsdf, Ref("esdf", '7')},
              {LayerKind::kStaticSpeedLimit, Ref("speed", '8')},
          },
      .known_mask = std::vector<std::uint8_t>(count, 1U),
      .elevation_m = std::vector<float>(count, 0.0F),
      .normal_x = std::vector<float>(count, 0.0F),
      .normal_y = std::vector<float>(count, 0.0F),
      .normal_z = std::vector<float>(count, 1.0F),
      .roughness_m = std::vector<float>(count, 0.01F),
      .hard_obstacle_mask = std::vector<std::uint8_t>(count, 0U),
      .confidence = std::vector<float>(count, 1.0F),
      .esdf_m = std::vector<float>(count, 10.0F),
      .static_speed_limit_mps = std::vector<float>(count, 10.0F),
  };
  auto result = ImmutableMapSnapshot::Create(input);
  if (!IsOk(result)) {
    return nullptr;
  }
  return std::get<
      std::shared_ptr<const ImmutableMapSnapshot>>(std::move(result));
}

}  // namespace lunar::planning::v3::hopper_test
