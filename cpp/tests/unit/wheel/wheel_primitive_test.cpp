#include <algorithm>
#include <chrono>
#include <ranges>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "lunar_path_planner/v3/wheel/wheel_primitive.hpp"

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

[[nodiscard]] WheelMotionPrimitive Primitive(
    std::string id,
    WheelMotionPrimitive::Kind kind,
    PoseXyzYaw relative_end_pose) {
  return {
      .primitive_id = std::move(id),
      .kind = kind,
      .relative_end_pose = relative_end_pose,
      .nominal_duration = DurationNanoseconds{500ms},
      .swept_geometry_ref = Ref("sweep", 'b'),
  };
}

[[nodiscard]] SafetyCapabilityProfile MakeWheelCapabilityFixture() {
  WheeledCapability wheel{
      .frame_id = "map",
      .collision_envelope =
          {
              .vertices_xy_m =
                  {{-0.6, -0.4}, {0.6, -0.4},
                   {0.6, 0.4}, {-0.6, 0.4}},
              .minimum_z_m = -0.2,
              .maximum_z_m = 0.8,
          },
      .motion_model_ref = Ref("wheel-motion", 'c'),
      .analytic_cost_model_ref = Ref("wheel-cost", 'd'),
      .hard_limits =
          {
              .maximum_forward_speed_mps = 1.0,
              .maximum_reverse_speed_mps = 0.7,
              .maximum_spin_rate_radps = 1.2,
              .maximum_forward_acceleration_mps2 = 0.8,
              .maximum_braking_deceleration_mps2 = 1.0,
              .maximum_yaw_acceleration_radps2 = 1.5,
              .maximum_lateral_acceleration_mps2 = 0.6,
              .maximum_drive_curvature_per_m = 1.0,
              .maximum_slope_rad = 0.5,
              .minimum_clearance_m = 0.1,
          },
      .motion_primitives =
          {
              Primitive("forward-line",
                        WheelMotionPrimitive::Kind::kDriveForwardLine,
                        {{1.0, 0.0, 0.0}, 0.0}),
              Primitive("forward-arc",
                        WheelMotionPrimitive::Kind::kDriveForwardArc,
                        {{0.8, 0.2, 0.0}, 0.4}),
              Primitive("reverse-line",
                        WheelMotionPrimitive::Kind::kDriveReverseLine,
                        {{-0.7, 0.0, 0.0}, 0.0}),
              Primitive("reverse-arc",
                        WheelMotionPrimitive::Kind::kDriveReverseArc,
                        {{-0.6, 0.2, 0.0}, -0.35}),
              Primitive("spin-cw",
                        WheelMotionPrimitive::Kind::kSpinCw,
                        {{0.0, 0.0, 0.0}, -0.5}),
              Primitive("spin-ccw",
                        WheelMotionPrimitive::Kind::kSpinCcw,
                        {{0.0, 0.0, 0.0}, 0.5}),
              Primitive("switch",
                        WheelMotionPrimitive::Kind::kStopAndSwitch,
                        {{0.0, 0.0, 0.0}, 0.0}),
          },
  };
  return {
      .content_ref = Ref("wheel-capability", 'a'),
      .content = std::move(wheel),
  };
}

[[nodiscard]] std::vector<PrimitiveId> PrimitiveIds(
    const WheelPrimitiveCatalog& catalog) {
  std::vector<PrimitiveId> ids;
  for (const auto& primitive : catalog.All()) {
    ids.push_back(primitive.id);
  }
  return ids;
}

TEST(WheelPrimitiveCatalogTest, RejectsIncompleteRequiredModes) {
  auto profile = MakeWheelCapabilityFixture();
  auto& specs =
      std::get<WheeledCapability>(profile.content).motion_primitives;
  std::erase_if(specs, [](const WheelMotionPrimitive& primitive) {
    return primitive.kind ==
           WheelMotionPrimitive::Kind::kDriveReverseLine ||
           primitive.kind ==
           WheelMotionPrimitive::Kind::kDriveReverseArc;
  });

  const auto result = WheelPrimitiveCatalog::Create(profile);

  ASSERT_FALSE(IsOk(result));
  EXPECT_EQ(std::get<Error>(result).code,
            ErrorCode::kInvalidArgument);
  EXPECT_EQ(std::get<Error>(result).field_path,
            "/content/motion_primitives");
}

TEST(WheelPrimitiveCatalogTest, RejectsNonPositiveDuration) {
  auto profile = MakeWheelCapabilityFixture();
  std::get<WheeledCapability>(profile.content)
      .motion_primitives.front()
      .nominal_duration = DurationNanoseconds{0ns};

  const auto result = WheelPrimitiveCatalog::Create(profile);

  ASSERT_FALSE(IsOk(result));
  EXPECT_EQ(std::get<Error>(result).field_path,
            "/content/motion_primitives/nominal_duration_ns");
}

TEST(WheelPrimitiveCatalogTest,
     StableOrderDoesNotDependOnInputOrder) {
  auto a = MakeWheelCapabilityFixture();
  auto b = a;
  auto& reversed =
      std::get<WheeledCapability>(b.content).motion_primitives;
  std::ranges::reverse(reversed);

  const auto catalog_a = WheelPrimitiveCatalog::Create(a);
  const auto catalog_b = WheelPrimitiveCatalog::Create(b);

  ASSERT_TRUE(IsOk(catalog_a));
  ASSERT_TRUE(IsOk(catalog_b));
  EXPECT_EQ(PrimitiveIds(std::get<WheelPrimitiveCatalog>(catalog_a)),
            PrimitiveIds(std::get<WheelPrimitiveCatalog>(catalog_b)));
  EXPECT_EQ(std::get<WheelPrimitiveCatalog>(catalog_a).content_ref(),
            std::get<WheelPrimitiveCatalog>(catalog_b).content_ref());
}

TEST(WheelPrimitiveCatalogTest,
     StartModeCanEnterForwardReverseAndBothSpinModes) {
  const auto result =
      WheelPrimitiveCatalog::Create(MakeWheelCapabilityFixture());
  ASSERT_TRUE(IsOk(result));
  const auto outgoing =
      std::get<WheelPrimitiveCatalog>(result).Outgoing(
          WheelMotionMode::kStart);

  const auto contains_target =
      [outgoing](WheelMotionMode mode) {
        return std::ranges::any_of(
            outgoing, [mode](const WheelPrimitive& primitive) {
              return primitive.target_mode == mode;
            });
      };
  EXPECT_TRUE(contains_target(WheelMotionMode::kForward));
  EXPECT_TRUE(contains_target(WheelMotionMode::kReverse));
  EXPECT_TRUE(
      contains_target(WheelMotionMode::kSpinClockwise));
  EXPECT_TRUE(
      contains_target(WheelMotionMode::kSpinCounterClockwise));
}

TEST(WheelPrimitiveCatalogTest,
     GeneratedModeSwitchesAreStationaryAndKeepCertification) {
  const auto result =
      WheelPrimitiveCatalog::Create(MakeWheelCapabilityFixture());
  ASSERT_TRUE(IsOk(result));
  const auto stop_edges =
      std::get<WheelPrimitiveCatalog>(result).Outgoing(
          WheelMotionMode::kForward);
  const auto stop = std::ranges::find_if(
      stop_edges, [](const WheelPrimitive& primitive) {
        return primitive.kind == WheelPrimitiveKind::kModeSwitch &&
               primitive.target_mode == WheelMotionMode::kStart;
      });

  ASSERT_NE(stop, stop_edges.end());
  EXPECT_DOUBLE_EQ(stop->relative_end.position_m.x, 0.0);
  EXPECT_DOUBLE_EQ(stop->relative_end.position_m.y, 0.0);
  EXPECT_DOUBLE_EQ(stop->relative_end.yaw_rad, 0.0);
  EXPECT_EQ(stop->capability_primitive_id, "switch");
  EXPECT_EQ(stop->sweep.validation_ref.id, "sweep");

  const auto restart_edges =
      std::get<WheelPrimitiveCatalog>(result).Outgoing(
          WheelMotionMode::kStart);
  const auto restart = std::ranges::find_if(
      restart_edges, [](const WheelPrimitive& primitive) {
        return primitive.kind == WheelPrimitiveKind::kModeSwitch &&
               primitive.target_mode == WheelMotionMode::kReverse;
      });
  ASSERT_NE(restart, restart_edges.end());
  EXPECT_EQ(restart->capability_primitive_id, "switch");
}

}  // namespace
}  // namespace lunar::planning::v3
