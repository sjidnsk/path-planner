#include <algorithm>
#include <chrono>
#include <concepts>
#include <limits>
#include <string>
#include <variant>

#include <gtest/gtest.h>

#include "lunar_path_planner/v3/contracts/platform_reference.hpp"
#include "lunar_path_planner/v3/legged/body_motion_primitive.hpp"

namespace lunar::planning::v3 {
namespace {

ContentRef Ref(std::string id, char fill) {
  return ContentRef{
      .id = std::move(id),
      .revision = 1U,
      .content_hash = std::string(64U, fill),
  };
}

LeggedBodyPrimitive Primitive(
    std::string id, LeggedBodyPrimitive::Kind kind,
    Vec3 displacement, double yaw_change_rad) {
  return LeggedBodyPrimitive{
      .primitive_id = std::move(id),
      .kind = kind,
      .body_frame_displacement_m = displacement,
      .yaw_change_rad = yaw_change_rad,
      .nominal_duration =
          DurationNanoseconds{std::chrono::seconds{1}},
      .sampled_body_sweep_ref = Ref("sweep", 'c'),
  };
}

SafetyCapabilityProfile MakeLeggedCapabilityFixture() {
  LeggedCapability capability;
  capability.frame_id = "map";
  capability.reference_point_id = "base_link";
  capability.collision_envelope.body_frame_halfspaces.halfspaces = {
      Halfspace3{{1.0, 0.0, 0.0}, 0.4},
      Halfspace3{{-1.0, 0.0, 0.0}, 0.4},
      Halfspace3{{0.0, 1.0, 0.0}, 0.25},
      Halfspace3{{0.0, -1.0, 0.0}, 0.25},
      Halfspace3{{0.0, 0.0, 1.0}, 0.5},
      Halfspace3{{0.0, 0.0, -1.0}, 0.5},
  };
  capability.motion_model_ref = Ref("motion", 'd');
  capability.analytic_cost_model_ref = Ref("cost", 'e');
  capability.terrain_thresholds = LeggedTerrainThresholds{
      .maximum_slope_rad = 0.5,
      .maximum_roughness_m = 0.1,
      .maximum_step_height_m = 0.2,
      .maximum_gap_width_m = 0.25,
      .minimum_confidence = 0.8,
      .minimum_body_clearance_m = 0.15,
      .minimum_body_height_m = 0.35,
      .maximum_body_height_m = 0.65,
  };
  capability.body_velocity_limits = LeggedBodyVelocityLimits{
      .forward_mps = {-0.3, 0.6},
      .lateral_mps = {-0.25, 0.25},
      .vertical_mps = {-0.1, 0.1},
      .yaw_rate_radps = {-0.5, 0.5},
      .linear_acceleration_mps2 = 0.4,
      .yaw_acceleration_radps2 = 0.6,
  };
  capability.motion_primitives = {
      Primitive("forward", LeggedBodyPrimitive::Kind::kForward,
                {0.5, 0.0, 0.0}, 0.0),
      Primitive("backward", LeggedBodyPrimitive::Kind::kBackward,
                {-0.3, 0.0, 0.0}, 0.0),
      Primitive("lateral-left", LeggedBodyPrimitive::Kind::kLateral,
                {0.0, 0.2, 0.0}, 0.0),
      Primitive("lateral-right", LeggedBodyPrimitive::Kind::kLateral,
                {0.0, -0.2, 0.0}, 0.0),
      Primitive("spin", LeggedBodyPrimitive::Kind::kSpin,
                {0.0, 0.0, 0.0}, 0.5),
  };
  return SafetyCapabilityProfile{
      .content_ref = Ref("legged-capability", 'a'),
      .content = std::move(capability),
  };
}

template <class T>
concept HasFootstepMember = requires(T value) {
  value.footsteps;
};

static_assert(!HasFootstepMember<BodyMotionPrimitive>);
static_assert(!HasFootstepMember<LeggedBodyReference>);

TEST(BodyMotionPrimitiveTest, RejectsMissingFixedReferencePoint) {
  auto profile = MakeLeggedCapabilityFixture();
  std::get<LeggedCapability>(profile.content).reference_point_id.clear();

  const auto result = BodyMotionPrimitiveCatalog::Create(profile);

  ASSERT_FALSE(IsOk(result));
  EXPECT_EQ(std::get<Error>(result).code, ErrorCode::kInvalidArgument);
}

TEST(BodyMotionPrimitiveTest, RequiresBothLateralDirectionsAndSpin) {
  auto profile = MakeLeggedCapabilityFixture();
  auto& primitives =
      std::get<LeggedCapability>(profile.content).motion_primitives;
  primitives.erase(primitives.begin() + 3);

  const auto result = BodyMotionPrimitiveCatalog::Create(profile);

  ASSERT_FALSE(IsOk(result));
  EXPECT_EQ(std::get<Error>(result).field_path,
            "safety_capability.content.motion_primitives");
}

TEST(BodyMotionPrimitiveTest, SortsByKindThenStablePrimitiveId) {
  auto profile = MakeLeggedCapabilityFixture();
  auto& primitives =
      std::get<LeggedCapability>(profile.content).motion_primitives;
  std::reverse(primitives.begin(), primitives.end());

  const auto result = BodyMotionPrimitiveCatalog::Create(profile);

  ASSERT_TRUE(IsOk(result));
  const auto& catalog = std::get<BodyMotionPrimitiveCatalog>(result);
  ASSERT_EQ(catalog.ordered_primitives().size(), 5U);
  EXPECT_EQ(catalog.ordered_primitives()[0].kind,
            BodyMotionKind::kForward);
  EXPECT_EQ(catalog.ordered_primitives()[1].kind,
            BodyMotionKind::kBackward);
  EXPECT_EQ(catalog.ordered_primitives()[2].kind,
            BodyMotionKind::kLateralLeft);
  EXPECT_EQ(catalog.ordered_primitives()[3].kind,
            BodyMotionKind::kLateralRight);
  EXPECT_EQ(catalog.ordered_primitives()[4].kind,
            BodyMotionKind::kSpin);
  EXPECT_EQ(catalog.content_ref(), profile.content_ref);
}

TEST(BodyMotionPrimitiveTest, RejectsNonFinitePrimitiveMotion) {
  auto profile = MakeLeggedCapabilityFixture();
  std::get<LeggedCapability>(profile.content)
      .motion_primitives.front()
      .body_frame_displacement_m.x =
      std::numeric_limits<double>::infinity();

  const auto result = BodyMotionPrimitiveCatalog::Create(profile);

  ASSERT_FALSE(IsOk(result));
  EXPECT_EQ(std::get<Error>(result).code, ErrorCode::kInvalidArgument);
}

}  // namespace
}  // namespace lunar::planning::v3
