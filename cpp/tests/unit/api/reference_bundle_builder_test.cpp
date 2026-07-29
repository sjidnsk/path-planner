#include <chrono>
#include <memory>
#include <string>
#include <utility>
#include <variant>

#include <gtest/gtest.h>

#include "lunar_path_planner/v3/api/reference_bundle_builder.hpp"
#include "lunar_path_planner/v3/codec/json_codec.hpp"
#include "lunar_path_planner/v3/map/immutable_snapshot.hpp"

namespace lpp = lunar::planning::v3;
namespace {

lpp::ContentRef Ref(std::string id, char digit) {
  return {
      std::move(id),
      1U,
      std::string(64U, digit),
  };
}

std::shared_ptr<const lpp::ImmutableMapSnapshot> MakeMap(
    const lpp::ContentRef& ref) {
  const lpp::MapSnapshotInput input{
      .snapshot_ref = ref,
      .map_revision = ref.revision,
      .immutable_data_handle = "bundle-builder-map",
      .source_time =
          {
              .clock_id = "mission",
              .tick = std::chrono::nanoseconds{0},
          },
      .bounds =
          {
              .minimum_m = {0.0, 0.0, -1.0},
              .maximum_m = {1.0, 1.0, 1.0},
          },
      .geometry =
          {
              .width = 1U,
              .height = 1U,
              .resolution_m = 1.0,
              .origin_m = {},
              .frame_id = "map",
          },
      .layer_manifest =
          {
              {lpp::LayerKind::kKnownMask, Ref("known", '1')},
              {lpp::LayerKind::kElevation, Ref("elevation", '2')},
              {lpp::LayerKind::kTerrainNormal, Ref("normal", '3')},
              {lpp::LayerKind::kRoughness, Ref("roughness", '4')},
              {lpp::LayerKind::kHardObstacle, Ref("obstacle", '5')},
              {lpp::LayerKind::kConfidence, Ref("confidence", '6')},
          },
      .known_mask = {1U},
      .elevation_m = {0.0F},
      .normal_x = {0.0F},
      .normal_y = {0.0F},
      .normal_z = {1.0F},
      .roughness_m = {0.0F},
      .hard_obstacle_mask = {0U},
      .confidence = {1.0F},
  };
  auto created = lpp::ImmutableMapSnapshot::Create(input);
  if (!lpp::IsOk(created)) {
    return nullptr;
  }
  return std::get<
      std::shared_ptr<const lpp::ImmutableMapSnapshot>>(
      std::move(created));
}

lpp::WheeledReference MakeWheelReference() {
  const lpp::DurationNanoseconds zero{
      std::chrono::nanoseconds{0}};
  const lpp::DurationNanoseconds one{
      std::chrono::seconds{1}};
  lpp::WheeledReference reference{
      .reference_id = "wheel-reference",
      .reference_hash = std::string(64U, '0'),
      .reference_time_origin =
          {
              .clock_id = "mission",
              .tick = std::chrono::nanoseconds{0},
          },
      .segments =
          {
              lpp::SpinSegment{
                  .segment_id = "spin-1",
                  .time_interval = {zero, one},
                  .fixed_position_m = {},
                  .unwrapped_yaw_rad =
                      {
                          .value_semantics = "unwrapped_yaw_rad",
                          .segments =
                              {
                                  {
                                      .start_offset = zero,
                                      .end_offset = one,
                                      .coefficients =
                                          {0.0, 0.0, 0.0, 0.0},
                                  },
                              },
                      },
              },
          },
      .safe_stop_anchor =
          {
              .anchor_id = "wheel-stop",
              .pose = {},
              .terrain_certification_ref =
                  Ref("wheel-stop-certificate", '7'),
          },
  };
  const auto digest =
      lpp::CanonicalReferenceHash(lpp::PlatformReference{reference});
  EXPECT_TRUE(lpp::IsOk(digest));
  if (lpp::IsOk(digest)) {
    reference.reference_hash =
        std::get<lpp::Sha256Digest>(digest);
  }
  return reference;
}

lpp::BundleBuildRequest MakeWheelBuildRequest() {
  const lpp::WheeledReference wheel = MakeWheelReference();
  const lpp::PlatformReference platform_reference{wheel};
  const lpp::RouteSkeletonContent route{
      .source_reference_id = wheel.reference_id,
      .source_reference_hash = wheel.reference_hash,
      .waypoints = {{{0.0, 0.0, 0.0}, 0.0}},
  };
  const lpp::ReferenceViewContent committed{
      .role = lpp::ReferenceViewContent::Role::kCommittedPrefix,
      .source_reference_id = wheel.reference_id,
      .source_reference_hash = wheel.reference_hash,
      .selector = lpp::SegmentViewSelector{0U, 1U},
  };
  const lpp::ReferenceViewContent preview{
      .role = lpp::ReferenceViewContent::Role::kPreview,
      .source_reference_id = wheel.reference_id,
      .source_reference_hash = wheel.reference_hash,
      .selector = lpp::SegmentViewSelector{1U, 1U},
  };
  return {
      .bundle_id = "wheel-bundle",
      .bundle_revision = 1U,
      .source_request_id = "request",
      .source_map_snapshot_ref = Ref("map", 'a'),
      .source_safety_capability_ref = Ref("capability", 'b'),
      .source_algorithm_config_ref = Ref("config", 'c'),
      .platform_reference = platform_reference,
      .route_skeleton = {"route", route},
      .committed_prefix = {"committed", committed},
      .preview = {"preview", preview},
  };
}

lpp::PlanningRequest MakeRootRequest(
    const lpp::BundleBuildRequest& build) {
  auto capability =
      std::make_shared<lpp::SafetyCapabilityProfile>();
  capability->content_ref =
      build.source_safety_capability_ref;
  auto config =
      std::make_shared<lpp::PlannerAlgorithmConfig>();
  config->content_ref = build.source_algorithm_config_ref;
  return {
      .request_id = build.source_request_id,
      .request_time =
          {
              .clock_id = "mission",
              .tick = std::chrono::nanoseconds{0},
          },
      .state_time =
          {
              .clock_id = "mission",
              .tick = std::chrono::nanoseconds{0},
          },
      .frame_id = "map",
      .platform_type = lpp::PlatformType::kWheeled,
      .current_state = lpp::WheeledOrLeggedState{},
      .map_snapshot = MakeMap(build.source_map_snapshot_ref),
      .safety_capability = std::move(capability),
      .algorithm_config = std::move(config),
  };
}

lpp::ReferenceBundle MakeActiveBundle(
    const lpp::BundleBuildRequest& request) {
  const auto route_hash =
      lpp::CanonicalComponentHash(request.route_skeleton.content);
  const auto committed_hash =
      lpp::CanonicalComponentHash(request.committed_prefix.content);
  const auto preview_hash =
      lpp::CanonicalComponentHash(request.preview.content);
  EXPECT_TRUE(lpp::IsOk(route_hash));
  EXPECT_TRUE(lpp::IsOk(committed_hash));
  EXPECT_TRUE(lpp::IsOk(preview_hash));
  return {
      .bundle_id = "active",
      .bundle_revision = 1U,
      .bundle_hash = std::string(64U, 'd'),
      .source_request_id = request.source_request_id,
      .source_map_snapshot_ref = request.source_map_snapshot_ref,
      .source_safety_capability_ref =
          request.source_safety_capability_ref,
      .source_algorithm_config_ref =
          request.source_algorithm_config_ref,
      .platform_type = lpp::PlatformType::kWheeled,
      .platform_reference = request.platform_reference,
      .route_skeleton =
          {
              request.route_skeleton.component_id,
              std::get<lpp::Sha256Digest>(route_hash),
              request.route_skeleton.content,
          },
      .committed_prefix =
          {
              request.committed_prefix.component_id,
              std::get<lpp::Sha256Digest>(committed_hash),
              request.committed_prefix.content,
          },
      .preview =
          {
              request.preview.component_id,
              std::get<lpp::Sha256Digest>(preview_hash),
              request.preview.content,
          },
  };
}

lpp::HopperReference MakeHashableHopperReference() {
  lpp::HopperReference hopper{
      .reference_id = "hopper-reference",
      .reference_hash = std::string(64U, '0'),
      .reference_time_origin =
          {
              .clock_id = "mission",
              .tick = std::chrono::nanoseconds{0},
          },
      .ground_hold_anchor =
          {
              .anchor_id = "hold",
          },
      .next_landing_region =
          {
              .region_id = "region",
              .frame_id = "map",
          },
      .jump_boundary =
          {
              .boundary_id = "jump",
              .ballistic_flight_time =
                  lpp::DurationNanoseconds{
                      std::chrono::seconds{1}},
          },
  };
  const auto hash =
      lpp::CanonicalReferenceHash(lpp::PlatformReference{hopper});
  EXPECT_TRUE(lpp::IsOk(hash))
      << (lpp::IsOk(hash) ? std::string{}
                          : std::get<lpp::Error>(hash).field_path +
                                ": " +
                                std::get<lpp::Error>(hash).message);
  if (lpp::IsOk(hash)) {
    hopper.reference_hash = std::get<lpp::Sha256Digest>(hash);
  }
  return hopper;
}

}  // namespace

TEST(ReferenceBundleBuilderTest,
     ComponentHashesUseCanonicalContentOnly) {
  const auto request = MakeWheelBuildRequest();
  const auto first =
      lpp::CanonicalComponentHash(request.route_skeleton.content);
  const auto second =
      lpp::CanonicalComponentHash(request.route_skeleton.content);
  ASSERT_TRUE(lpp::IsOk(first));
  ASSERT_TRUE(lpp::IsOk(second));
  EXPECT_EQ(std::get<lpp::Sha256Digest>(first),
            std::get<lpp::Sha256Digest>(second));

  auto changed = request.route_skeleton.content;
  changed.waypoints.front().position_m.x = 0.25;
  const auto changed_hash = lpp::CanonicalComponentHash(changed);
  ASSERT_TRUE(lpp::IsOk(changed_hash));
  EXPECT_NE(std::get<lpp::Sha256Digest>(first),
            std::get<lpp::Sha256Digest>(changed_hash));
}

TEST(ReferenceBundleBuilderTest,
     BundleHashOmitsOnlyTheTopLevelHash) {
  const auto request = MakeWheelBuildRequest();
  auto bundle = MakeActiveBundle(request);
  const auto first = lpp::CanonicalBundleHash(bundle);
  ASSERT_TRUE(lpp::IsOk(first));

  bundle.bundle_hash = std::string(64U, 'e');
  const auto second = lpp::CanonicalBundleHash(bundle);
  ASSERT_TRUE(lpp::IsOk(second));
  EXPECT_EQ(std::get<lpp::Sha256Digest>(first),
            std::get<lpp::Sha256Digest>(second));

  ++bundle.bundle_revision;
  const auto changed = lpp::CanonicalBundleHash(bundle);
  ASSERT_TRUE(lpp::IsOk(changed));
  EXPECT_NE(std::get<lpp::Sha256Digest>(first),
            std::get<lpp::Sha256Digest>(changed));
}

TEST(ReferenceBundleBuilderTest,
     RejectsReusedIdWithDifferentCanonicalContent) {
  auto build = MakeWheelBuildRequest();
  lpp::PlanningRequest request = MakeRootRequest(build);
  lpp::EmptyContractObjectRegistry registry;
  build.active_bundle = MakeActiveBundle(build);
  build.bundle_id = "replacement";
  build.supersedes_bundle_id = build.active_bundle->bundle_id;
  build.route_skeleton.content.waypoints.front().position_m.x = 0.5;

  const auto result = lpp::BuildReferenceBundle(
      build, lpp::ReferenceActivationContext{request, registry});

  ASSERT_FALSE(lpp::IsOk(result));
  EXPECT_EQ(std::get<lpp::Error>(result).field_path,
            "component_id");
}

TEST(ReferenceBundleBuilderTest,
     HopperCommitsHoldAndPreviewsOnlyNextHop) {
  const lpp::HopperReference hopper =
      MakeHashableHopperReference();
  lpp::BundleBuildRequest build{
      .bundle_id = "hopper-bundle",
      .bundle_revision = 1U,
      .source_request_id = "request",
      .source_map_snapshot_ref = Ref("map", 'a'),
      .source_safety_capability_ref = Ref("capability", 'b'),
      .source_algorithm_config_ref = Ref("config", 'c'),
      .platform_reference = hopper,
      .route_skeleton =
          {
              "route",
              {
                  .source_reference_id = hopper.reference_id,
                  .source_reference_hash = hopper.reference_hash,
              },
          },
      .committed_prefix =
          {
              "committed",
              {
                  .role =
                      lpp::ReferenceViewContent::Role::
                          kCommittedPrefix,
                  .source_reference_id = hopper.reference_id,
                  .source_reference_hash = hopper.reference_hash,
                  .selector = lpp::JumpViewSelector{
                      .boundary_id = hopper.jump_boundary.boundary_id,
                      .scope =
                          lpp::JumpViewSelector::Scope::kNextHop,
                  },
              },
          },
      .preview =
          {
              "preview",
              {
                  .role =
                      lpp::ReferenceViewContent::Role::kPreview,
                  .source_reference_id = hopper.reference_id,
                  .source_reference_hash = hopper.reference_hash,
                  .selector = lpp::JumpViewSelector{
                      .boundary_id = hopper.jump_boundary.boundary_id,
                      .scope =
                          lpp::JumpViewSelector::Scope::kNextHop,
                  },
              },
          },
  };
  lpp::PlanningRequest request{};
  lpp::EmptyContractObjectRegistry registry;

  const auto result = lpp::BuildReferenceBundle(
      build, lpp::ReferenceActivationContext{request, registry});

  ASSERT_FALSE(lpp::IsOk(result));
  EXPECT_EQ(std::get<lpp::Error>(result).field_path,
            "committed_prefix.content.selector");
}

TEST(ReferenceBundleBuilderTest,
     RejectsReferenceHashThatDoesNotBindPayload) {
  auto build = MakeWheelBuildRequest();
  auto wheel =
      std::get<lpp::WheeledReference>(build.platform_reference);
  wheel.reference_hash = std::string(64U, 'f');
  build.platform_reference = wheel;
  lpp::PlanningRequest request{};
  lpp::EmptyContractObjectRegistry registry;

  const auto result = lpp::BuildReferenceBundle(
      build, lpp::ReferenceActivationContext{request, registry});

  ASSERT_FALSE(lpp::IsOk(result));
  EXPECT_EQ(std::get<lpp::Error>(result).field_path,
            "platform_reference.reference_hash");
}
