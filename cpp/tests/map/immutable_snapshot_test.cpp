#include <algorithm>
#include <chrono>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <variant>

#include <gtest/gtest.h>

#include "lunar_path_planner/v3/map/immutable_snapshot.hpp"

namespace lpp = lunar::planning::v3;

namespace {

lpp::ContentRef MakeRef(std::string id, std::uint32_t revision,
                        char digest_digit) {
  return {
      .id = std::move(id),
      .revision = revision,
      .content_hash = std::string(64U, digest_digit),
  };
}

lpp::MapSnapshotInput MakeMinimalMapInput() {
  lpp::MapSnapshotInput input{
      .snapshot_ref = MakeRef("map-snapshot", 42U, 'a'),
      .map_revision = 42U,
      .immutable_data_handle = "map-registry-handle",
      .source_time =
          {.clock_id = "mission", .tick = std::chrono::nanoseconds{1234}},
      .bounds =
          {
              .minimum_m = {0.0, 0.0, -1.0},
              .maximum_m = {1.0, 1.0, 1.0},
          },
      .geometry =
          {
              .width = 2U,
              .height = 2U,
              .resolution_m = 0.5,
              .origin_m = {0.0, 0.0},
              .frame_id = "map",
          },
      .layer_manifest =
          {
              {lpp::LayerKind::kKnownMask,
               MakeRef("known", 7U, '1')},
              {lpp::LayerKind::kElevation,
               MakeRef("elevation", 7U, '2')},
              {lpp::LayerKind::kTerrainNormal,
               MakeRef("normal", 7U, '3')},
              {lpp::LayerKind::kRoughness,
               MakeRef("roughness", 7U, '4')},
              {lpp::LayerKind::kHardObstacle,
               MakeRef("obstacle", 7U, '5')},
              {lpp::LayerKind::kConfidence,
               MakeRef("confidence", 7U, '6')},
          },
      .known_mask = {1U, 1U, 0U, 1U},
      .elevation_m = {0.0F, 0.1F, -0.1F, 0.2F},
      .normal_x = {0.0F, 0.0F, 0.0F, 0.0F},
      .normal_y = {0.0F, 0.0F, 0.0F, 0.0F},
      .normal_z = {1.0F, 1.0F, 1.0F, 1.0F},
      .roughness_m = {0.01F, 0.02F, 0.03F, 0.04F},
      .hard_obstacle_mask = {0U, 1U, 0U, 0U},
      .confidence = {1.0F, 0.8F, 0.0F, 0.5F},
  };
  return input;
}

std::shared_ptr<const lpp::ImmutableMapSnapshot> RequireSnapshot(
    const lpp::MapSnapshotInput& input) {
  auto result = lpp::ImmutableMapSnapshot::Create(input);
  EXPECT_TRUE(lpp::IsOk(result));
  if (!lpp::IsOk(result)) {
    return nullptr;
  }
  return std::get<std::shared_ptr<const lpp::ImmutableMapSnapshot>>(
      std::move(result));
}

void RemoveLayer(lpp::MapSnapshotInput& input, lpp::LayerKind kind) {
  std::erase_if(input.layer_manifest,
                [kind](const lpp::LayerManifestEntry& entry) {
                  return entry.layer_kind == kind;
                });
}

}  // namespace

TEST(ImmutableMapSnapshot, CopiesMutableInputBeforePublicationAndOutlivesIt) {
  std::shared_ptr<const lpp::ImmutableMapSnapshot> snapshot;
  {
    auto input = MakeMinimalMapInput();
    snapshot = RequireSnapshot(input);
    ASSERT_NE(snapshot, nullptr);

    input.layer_manifest.front().content_ref.id = "mutated";
    input.known_mask[0] = 0U;
    input.elevation_m[0] = 99.0F;
    input.normal_z[0] = 0.0F;
    input.roughness_m[0] = 99.0F;
    input.hard_obstacle_mask[0] = 1U;
    input.confidence[0] = 0.0F;
  }

  ASSERT_NE(snapshot, nullptr);
  EXPECT_EQ(snapshot->LayerIdentity(lpp::LayerKind::kKnownMask)->get().id,
            "known");
  EXPECT_EQ(snapshot->KnownMask()[0], 1U);
  EXPECT_FLOAT_EQ(snapshot->ElevationMeters()[0], 0.0F);
  EXPECT_FLOAT_EQ(snapshot->SurfaceNormals().z[0], 1.0F);
  EXPECT_FLOAT_EQ(snapshot->RoughnessMeters()[0], 0.01F);
  EXPECT_EQ(snapshot->HardObstacleMask()[0], 0U);
  EXPECT_FLOAT_EQ(snapshot->Confidence()[0], 1.0F);
}

TEST(ImmutableMapSnapshot, RequiresEveryMandatoryLayer) {
  const lpp::LayerKind required[] = {
      lpp::LayerKind::kKnownMask,     lpp::LayerKind::kElevation,
      lpp::LayerKind::kTerrainNormal, lpp::LayerKind::kRoughness,
      lpp::LayerKind::kHardObstacle,  lpp::LayerKind::kConfidence,
  };

  for (const auto kind : required) {
    auto input = MakeMinimalMapInput();
    RemoveLayer(input, kind);
    EXPECT_FALSE(lpp::IsOk(lpp::ImmutableMapSnapshot::Create(input)));
  }
}

TEST(ImmutableMapSnapshot, RejectsDuplicateLayerKindsAndIdentities) {
  {
    auto input = MakeMinimalMapInput();
    input.layer_manifest.push_back(
        {lpp::LayerKind::kKnownMask, MakeRef("other-known", 8U, 'b')});
    EXPECT_FALSE(lpp::IsOk(lpp::ImmutableMapSnapshot::Create(input)));
  }
  {
    auto input = MakeMinimalMapInput();
    input.layer_manifest[1].content_ref =
        input.layer_manifest[0].content_ref;
    EXPECT_FALSE(lpp::IsOk(lpp::ImmutableMapSnapshot::Create(input)));
  }
  {
    auto input = MakeMinimalMapInput();
    input.layer_manifest[1].content_ref.id =
        input.layer_manifest[0].content_ref.id;
    input.layer_manifest[1].content_ref.revision =
        input.layer_manifest[0].content_ref.revision;
    input.layer_manifest[1].content_ref.content_hash =
        std::string(64U, 'f');
    EXPECT_FALSE(lpp::IsOk(lpp::ImmutableMapSnapshot::Create(input)));
  }
}

TEST(ImmutableMapSnapshot, RejectsIncorrectMandatoryLayerCellCounts) {
  {
    auto input = MakeMinimalMapInput();
    input.known_mask.pop_back();
    EXPECT_FALSE(lpp::IsOk(lpp::ImmutableMapSnapshot::Create(input)));
  }
  {
    auto input = MakeMinimalMapInput();
    input.elevation_m.pop_back();
    EXPECT_FALSE(lpp::IsOk(lpp::ImmutableMapSnapshot::Create(input)));
  }
  {
    auto input = MakeMinimalMapInput();
    input.normal_y.pop_back();
    EXPECT_FALSE(lpp::IsOk(lpp::ImmutableMapSnapshot::Create(input)));
  }
  {
    auto input = MakeMinimalMapInput();
    input.roughness_m.pop_back();
    EXPECT_FALSE(lpp::IsOk(lpp::ImmutableMapSnapshot::Create(input)));
  }
  {
    auto input = MakeMinimalMapInput();
    input.hard_obstacle_mask.pop_back();
    EXPECT_FALSE(lpp::IsOk(lpp::ImmutableMapSnapshot::Create(input)));
  }
  {
    auto input = MakeMinimalMapInput();
    input.confidence.pop_back();
    EXPECT_FALSE(lpp::IsOk(lpp::ImmutableMapSnapshot::Create(input)));
  }
}

TEST(ImmutableMapSnapshot, RejectsZeroOverflowedOrNonFiniteGeometry) {
  {
    auto input = MakeMinimalMapInput();
    input.geometry.width = 0U;
    EXPECT_FALSE(lpp::IsOk(lpp::ImmutableMapSnapshot::Create(input)));
  }
  {
    auto input = MakeMinimalMapInput();
    input.geometry.height = 0U;
    EXPECT_FALSE(lpp::IsOk(lpp::ImmutableMapSnapshot::Create(input)));
  }
  {
    auto input = MakeMinimalMapInput();
    input.geometry.width = std::numeric_limits<std::size_t>::max();
    input.geometry.height = 2U;
    EXPECT_FALSE(lpp::IsOk(lpp::ImmutableMapSnapshot::Create(input)));
  }
  {
    auto input = MakeMinimalMapInput();
    input.geometry.resolution_m = 0.0;
    EXPECT_FALSE(lpp::IsOk(lpp::ImmutableMapSnapshot::Create(input)));
  }
  {
    auto input = MakeMinimalMapInput();
    input.geometry.resolution_m =
        std::numeric_limits<double>::infinity();
    EXPECT_FALSE(lpp::IsOk(lpp::ImmutableMapSnapshot::Create(input)));
  }
  {
    auto input = MakeMinimalMapInput();
    input.geometry.origin_m.x =
        std::numeric_limits<double>::quiet_NaN();
    EXPECT_FALSE(lpp::IsOk(lpp::ImmutableMapSnapshot::Create(input)));
  }
  {
    auto input = MakeMinimalMapInput();
    input.geometry.frame_id.clear();
    EXPECT_FALSE(lpp::IsOk(lpp::ImmutableMapSnapshot::Create(input)));
  }
}

TEST(ImmutableMapSnapshot, RejectsInvalidOpaqueRegistryHandle) {
  auto input = MakeMinimalMapInput();
  input.immutable_data_handle.clear();
  EXPECT_FALSE(lpp::IsOk(lpp::ImmutableMapSnapshot::Create(input)));
}

TEST(ImmutableMapSnapshot, RejectsNonFiniteOrInvertedBounds) {
  {
    auto input = MakeMinimalMapInput();
    input.bounds.minimum_m.z =
        std::numeric_limits<double>::quiet_NaN();
    EXPECT_FALSE(lpp::IsOk(lpp::ImmutableMapSnapshot::Create(input)));
  }
  {
    auto input = MakeMinimalMapInput();
    input.bounds.minimum_m.x = 2.0;
    EXPECT_FALSE(lpp::IsOk(lpp::ImmutableMapSnapshot::Create(input)));
  }
}

TEST(ImmutableMapSnapshot, RejectsNonFiniteLayerValues) {
  {
    auto input = MakeMinimalMapInput();
    input.elevation_m[1] = std::numeric_limits<float>::infinity();
    EXPECT_FALSE(lpp::IsOk(lpp::ImmutableMapSnapshot::Create(input)));
  }
  {
    auto input = MakeMinimalMapInput();
    input.normal_x[1] = std::numeric_limits<float>::quiet_NaN();
    EXPECT_FALSE(lpp::IsOk(lpp::ImmutableMapSnapshot::Create(input)));
  }
  {
    auto input = MakeMinimalMapInput();
    input.roughness_m[1] = std::numeric_limits<float>::quiet_NaN();
    EXPECT_FALSE(lpp::IsOk(lpp::ImmutableMapSnapshot::Create(input)));
  }
}

TEST(ImmutableMapSnapshot, RejectsNonBinaryMasksAndOutOfRangeConfidence) {
  {
    auto input = MakeMinimalMapInput();
    input.known_mask[0] = 2U;
    EXPECT_FALSE(lpp::IsOk(lpp::ImmutableMapSnapshot::Create(input)));
  }
  {
    auto input = MakeMinimalMapInput();
    input.hard_obstacle_mask[0] = 2U;
    EXPECT_FALSE(lpp::IsOk(lpp::ImmutableMapSnapshot::Create(input)));
  }
  {
    auto input = MakeMinimalMapInput();
    input.confidence[0] = -0.01F;
    EXPECT_FALSE(lpp::IsOk(lpp::ImmutableMapSnapshot::Create(input)));
  }
  {
    auto input = MakeMinimalMapInput();
    input.confidence[0] = 1.01F;
    EXPECT_FALSE(lpp::IsOk(lpp::ImmutableMapSnapshot::Create(input)));
  }
  {
    auto input = MakeMinimalMapInput();
    input.confidence[0] = std::numeric_limits<float>::quiet_NaN();
    EXPECT_FALSE(lpp::IsOk(lpp::ImmutableMapSnapshot::Create(input)));
  }
}

TEST(ImmutableMapSnapshot, RejectsNormalsOutsideUnitTolerance) {
  {
    auto input = MakeMinimalMapInput();
    input.normal_z[0] = 0.99F;
    EXPECT_FALSE(lpp::IsOk(lpp::ImmutableMapSnapshot::Create(input)));
  }
  {
    auto input = MakeMinimalMapInput();
    input.normal_x[0] = 0.6F;
    input.normal_z[0] = 0.8F;
    EXPECT_TRUE(lpp::IsOk(lpp::ImmutableMapSnapshot::Create(input)));
  }
}

TEST(ImmutableMapSnapshot, OptionalLayersRequireMatchingManifestAndCellData) {
  {
    const auto snapshot = RequireSnapshot(MakeMinimalMapInput());
    ASSERT_NE(snapshot, nullptr);
    EXPECT_TRUE(snapshot->EsdfMeters().empty());
    EXPECT_TRUE(snapshot->StaticSpeedLimitMps().empty());
    EXPECT_FALSE(
        snapshot->LayerIdentity(lpp::LayerKind::kEsdf).has_value());
  }
  {
    auto input = MakeMinimalMapInput();
    input.layer_manifest.push_back(
        {lpp::LayerKind::kEsdf, MakeRef("esdf", 7U, '7')});
    input.esdf_m = {1.0F, 2.0F, 3.0F, 4.0F};
    const auto snapshot = RequireSnapshot(input);
    ASSERT_NE(snapshot, nullptr);
    EXPECT_EQ(snapshot->EsdfMeters().size(), 4U);
    EXPECT_FLOAT_EQ(snapshot->EsdfMeters()[3], 4.0F);
  }
  {
    auto input = MakeMinimalMapInput();
    input.layer_manifest.push_back(
        {lpp::LayerKind::kEsdf, MakeRef("esdf", 7U, '7')});
    EXPECT_FALSE(lpp::IsOk(lpp::ImmutableMapSnapshot::Create(input)));
  }
  {
    auto input = MakeMinimalMapInput();
    input.esdf_m = {1.0F, 2.0F, 3.0F, 4.0F};
    EXPECT_FALSE(lpp::IsOk(lpp::ImmutableMapSnapshot::Create(input)));
  }
  {
    auto input = MakeMinimalMapInput();
    input.layer_manifest.push_back(
        {lpp::LayerKind::kStaticSpeedLimit,
         MakeRef("speed-limit", 7U, '8')});
    input.static_speed_limit_mps = {0.5F, 0.6F, 0.7F, 0.8F};
    const auto snapshot = RequireSnapshot(input);
    ASSERT_NE(snapshot, nullptr);
    EXPECT_FLOAT_EQ(snapshot->StaticSpeedLimitMps()[2], 0.7F);
  }
}

TEST(ImmutableMapSnapshot, RejectsInvalidOptionalLayerValues) {
  {
    auto input = MakeMinimalMapInput();
    input.layer_manifest.push_back(
        {lpp::LayerKind::kEsdf, MakeRef("esdf", 7U, '7')});
    input.esdf_m = {1.0F, 2.0F,
                    std::numeric_limits<float>::quiet_NaN(), 4.0F};
    EXPECT_FALSE(lpp::IsOk(lpp::ImmutableMapSnapshot::Create(input)));
  }
  {
    auto input = MakeMinimalMapInput();
    input.layer_manifest.push_back(
        {lpp::LayerKind::kStaticSpeedLimit,
         MakeRef("speed-limit", 7U, '8')});
    input.static_speed_limit_mps = {0.5F, 0.6F, -0.1F, 0.8F};
    EXPECT_FALSE(lpp::IsOk(lpp::ImmutableMapSnapshot::Create(input)));
  }
}

TEST(ImmutableMapSnapshot, ExposesAllSnapshotMetadataWithoutRepair) {
  const auto input = MakeMinimalMapInput();
  const auto snapshot = RequireSnapshot(input);
  ASSERT_NE(snapshot, nullptr);

  EXPECT_EQ(snapshot->snapshot_ref(), input.snapshot_ref);
  EXPECT_EQ(snapshot->map_revision(), 42U);
  EXPECT_EQ(snapshot->immutable_data_handle(), "map-registry-handle");
  EXPECT_EQ(snapshot->frame_id(), "map");
  EXPECT_EQ(snapshot->source_time().clock_id, "mission");
  EXPECT_EQ(snapshot->source_time().tick.count(), 1234);
  EXPECT_DOUBLE_EQ(snapshot->bounds().minimum_m.z, -1.0);
  EXPECT_DOUBLE_EQ(snapshot->bounds().maximum_m.x, 1.0);
  EXPECT_EQ(snapshot->geometry().width, 2U);
  EXPECT_EQ(snapshot->geometry().height, 2U);
  EXPECT_DOUBLE_EQ(snapshot->geometry().resolution_m, 0.5);
  EXPECT_DOUBLE_EQ(snapshot->geometry().origin_m.x, 0.0);
  EXPECT_EQ(snapshot->layer_manifest().size(), 6U);
  ASSERT_TRUE(
      snapshot->LayerIdentity(lpp::LayerKind::kTerrainNormal).has_value());
  EXPECT_EQ(
      snapshot->LayerIdentity(lpp::LayerKind::kTerrainNormal)->get().revision,
      7U);
}

TEST(ImmutableMapSnapshot, ManifestHashIsCanonicalStableAndComplete) {
  auto first_input = MakeMinimalMapInput();
  auto reordered_input = MakeMinimalMapInput();
  std::reverse(reordered_input.layer_manifest.begin(),
               reordered_input.layer_manifest.end());

  const auto first = RequireSnapshot(first_input);
  const auto reordered = RequireSnapshot(reordered_input);
  ASSERT_NE(first, nullptr);
  ASSERT_NE(reordered, nullptr);
  EXPECT_EQ(first->LayerManifestHash(), reordered->LayerManifestHash());
  EXPECT_EQ(first->LayerManifestHash().size(), 64U);
  EXPECT_TRUE(std::ranges::all_of(
      first->LayerManifestHash(), [](char value) {
        return (value >= '0' && value <= '9') ||
               (value >= 'a' && value <= 'f');
      }));

  auto changed_id_input = MakeMinimalMapInput();
  changed_id_input.layer_manifest[0].content_ref.id = "known-v2";
  const auto changed_id = RequireSnapshot(changed_id_input);
  ASSERT_NE(changed_id, nullptr);
  EXPECT_NE(first->LayerManifestHash(), changed_id->LayerManifestHash());

  auto changed_revision_input = MakeMinimalMapInput();
  changed_revision_input.layer_manifest[0].content_ref.revision = 8U;
  const auto changed_revision = RequireSnapshot(changed_revision_input);
  ASSERT_NE(changed_revision, nullptr);
  EXPECT_NE(first->LayerManifestHash(),
            changed_revision->LayerManifestHash());

  auto changed_digest_input = MakeMinimalMapInput();
  changed_digest_input.layer_manifest[0].content_ref.content_hash =
      std::string(64U, 'f');
  const auto changed_digest = RequireSnapshot(changed_digest_input);
  ASSERT_NE(changed_digest, nullptr);
  EXPECT_NE(first->LayerManifestHash(), changed_digest->LayerManifestHash());
}
