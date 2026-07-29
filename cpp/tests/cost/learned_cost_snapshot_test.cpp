#include <array>
#include <chrono>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <variant>
#include <vector>

#include <gtest/gtest.h>

#include "lunar_path_planner/v3/cost/learned_cost_snapshot.hpp"

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

lpp::MapSnapshotInput MakeMapInput() {
  return {
      .snapshot_ref = MakeRef("map-snapshot", 7U, 'a'),
      .map_revision = 7U,
      .immutable_data_handle = "map-registry-handle",
      .source_time =
          {.clock_id = "mission", .tick = std::chrono::nanoseconds{10}},
      .bounds =
          {
              .minimum_m = {0.0, 0.0, -1.0},
              .maximum_m = {2.0, 2.0, 1.0},
          },
      .geometry =
          {
              .width = 2U,
              .height = 2U,
              .resolution_m = 1.0,
              .origin_m = {0.0, 0.0},
              .frame_id = "map",
          },
      .layer_manifest =
          {
              {lpp::LayerKind::kKnownMask, MakeRef("known", 1U, '1')},
              {lpp::LayerKind::kElevation,
               MakeRef("elevation", 1U, '2')},
              {lpp::LayerKind::kTerrainNormal,
               MakeRef("normal", 1U, '3')},
              {lpp::LayerKind::kRoughness,
               MakeRef("roughness", 1U, '4')},
              {lpp::LayerKind::kHardObstacle,
               MakeRef("obstacle", 1U, '5')},
              {lpp::LayerKind::kConfidence,
               MakeRef("confidence", 1U, '6')},
          },
      .known_mask = {1U, 1U, 1U, 1U},
      .elevation_m = {0.0F, 0.0F, 0.0F, 0.0F},
      .normal_x = {0.0F, 0.0F, 0.0F, 0.0F},
      .normal_y = {0.0F, 0.0F, 0.0F, 0.0F},
      .normal_z = {1.0F, 1.0F, 1.0F, 1.0F},
      .roughness_m = {0.0F, 0.0F, 0.0F, 0.0F},
      .hard_obstacle_mask = {0U, 0U, 0U, 0U},
      .confidence = {1.0F, 1.0F, 1.0F, 1.0F},
  };
}

std::shared_ptr<const lpp::ImmutableMapSnapshot> MakeMap() {
  auto result = lpp::ImmutableMapSnapshot::Create(MakeMapInput());
  EXPECT_TRUE(lpp::IsOk(result));
  if (!lpp::IsOk(result)) {
    return nullptr;
  }
  return std::get<std::shared_ptr<const lpp::ImmutableMapSnapshot>>(
      std::move(result));
}

lpp::LearnedCostSnapshotInput MakeLearnedInput() {
  return {
      .snapshot_ref = MakeRef("learned-snapshot", 3U, 'b'),
      .model_ref = MakeRef("learned-model", 5U, 'c'),
      .source_map_snapshot_ref = MakeRef("map-snapshot", 7U, 'a'),
      .source_map_revision = 7U,
      .frame_id = "map",
      .feature_contract_id = "feature-contract-v1",
      .output_contract_id = "output-contract-v1",
      .generated_at =
          {.clock_id = "mission", .tick = std::chrono::nanoseconds{9}},
      .certified_energy_correction_bounds = {-1.0, 1.0},
      .certified_nonfatal_risk_correction_bounds = {-0.5, 0.5},
      .energy_correction = {0.1F, 0.8F, -0.8F, 0.25F},
      .nonfatal_risk_correction = {0.05F, 0.4F, -0.4F, 0.1F},
  };
}

std::shared_ptr<const lpp::LearnedCostSnapshot> MakeLearned() {
  auto result = lpp::LearnedCostSnapshot::Create(MakeLearnedInput(), 4U);
  EXPECT_TRUE(lpp::IsOk(result));
  if (!lpp::IsOk(result)) {
    return nullptr;
  }
  return std::get<std::shared_ptr<const lpp::LearnedCostSnapshot>>(
      std::move(result));
}

lpp::PlannerAlgorithmConfig MakeConfig() {
  lpp::PlannerAlgorithmConfig config{};
  config.learned_cost_policy.mode =
      lpp::LearnedCostPolicy::Mode::kOptionalBoundedSoftCost;
  config.learned_cost_policy.maximum_absolute_energy_correction = 0.25;
  config.learned_cost_policy
      .maximum_absolute_nonfatal_risk_correction = 0.10;
  config.learned_cost_model_ref = MakeRef("learned-model", 5U, 'c');
  return config;
}

constexpr std::array<float, 4U> kAnalyticEnergy{
    1.0F, 2.0F, 3.0F, 4.0F};
constexpr std::array<float, 4U> kAnalyticRisk{
    0.2F, 0.3F, 0.4F, 0.5F};

lpp::ResolvedSoftCost RequireResolved(
    lpp::Result<lpp::ResolvedSoftCost> result) {
  EXPECT_TRUE(lpp::IsOk(result));
  if (!lpp::IsOk(result)) {
    return {};
  }
  return std::get<lpp::ResolvedSoftCost>(std::move(result));
}

void ExpectInvalidAdmission(const lpp::LearnedCostSnapshotInput& input,
                            std::size_t expected_cell_count = 4U) {
  const auto result =
      lpp::LearnedCostSnapshot::Create(input, expected_cell_count);
  ASSERT_FALSE(lpp::IsOk(result));
  EXPECT_EQ(std::get<lpp::Error>(result).code,
            lpp::ErrorCode::kInvalidArgument);
}

}  // namespace

TEST(LearnedCostSnapshot, CopiesPinnedValuesAndMetadataBeforePublication) {
  auto input = MakeLearnedInput();
  auto result = lpp::LearnedCostSnapshot::Create(input, 4U);
  ASSERT_TRUE(lpp::IsOk(result));
  const auto snapshot =
      std::get<std::shared_ptr<const lpp::LearnedCostSnapshot>>(result);

  input.snapshot_ref.id = "mutated";
  input.model_ref.id = "mutated";
  input.source_map_snapshot_ref.id = "mutated";
  input.frame_id = "mutated";
  input.energy_correction[0] = 99.0F;
  input.nonfatal_risk_correction[0] = 99.0F;

  EXPECT_EQ(snapshot->snapshot_ref(), MakeRef("learned-snapshot", 3U, 'b'));
  EXPECT_EQ(snapshot->model_ref(), MakeRef("learned-model", 5U, 'c'));
  EXPECT_EQ(snapshot->source_map_snapshot_ref(),
            MakeRef("map-snapshot", 7U, 'a'));
  EXPECT_EQ(snapshot->source_map_revision(), 7U);
  EXPECT_EQ(snapshot->frame_id(), "map");
  EXPECT_DOUBLE_EQ(snapshot->certified_energy_correction_bounds().lower,
                   -1.0);
  EXPECT_DOUBLE_EQ(
      snapshot->certified_nonfatal_risk_correction_bounds().upper, 0.5);
  EXPECT_FLOAT_EQ(snapshot->EnergyCorrection()[0], 0.1F);
  EXPECT_FLOAT_EQ(snapshot->NonfatalRiskCorrection()[0], 0.05F);
}

TEST(LearnedCostSnapshot, RejectsEveryInvalidReferenceAndIdentifier) {
  {
    auto input = MakeLearnedInput();
    input.snapshot_ref.id.clear();
    ExpectInvalidAdmission(input);
  }
  {
    auto input = MakeLearnedInput();
    input.model_ref.revision = 0U;
    ExpectInvalidAdmission(input);
  }
  {
    auto input = MakeLearnedInput();
    input.source_map_snapshot_ref.content_hash = "not-a-sha256";
    ExpectInvalidAdmission(input);
  }
  {
    auto input = MakeLearnedInput();
    input.source_map_revision = 0U;
    ExpectInvalidAdmission(input);
  }
  {
    auto input = MakeLearnedInput();
    input.frame_id = " invalid";
    ExpectInvalidAdmission(input);
  }
  {
    auto input = MakeLearnedInput();
    input.feature_contract_id.clear();
    ExpectInvalidAdmission(input);
  }
  {
    auto input = MakeLearnedInput();
    input.output_contract_id = "invalid identifier";
    ExpectInvalidAdmission(input);
  }
  {
    auto input = MakeLearnedInput();
    input.generated_at.clock_id.clear();
    ExpectInvalidAdmission(input);
  }
}

TEST(LearnedCostSnapshot, RejectsZeroAndWrongExpectedShape) {
  ExpectInvalidAdmission(MakeLearnedInput(), 0U);
  {
    auto input = MakeLearnedInput();
    input.energy_correction.pop_back();
    ExpectInvalidAdmission(input);
  }
  {
    auto input = MakeLearnedInput();
    input.nonfatal_risk_correction.push_back(0.0F);
    ExpectInvalidAdmission(input);
  }
}

TEST(LearnedCostSnapshot, RejectsUnorderedOrNonFiniteCertifiedIntervals) {
  {
    auto input = MakeLearnedInput();
    input.certified_energy_correction_bounds = {1.0, -1.0};
    ExpectInvalidAdmission(input);
  }
  {
    auto input = MakeLearnedInput();
    input.certified_nonfatal_risk_correction_bounds.lower =
        -std::numeric_limits<double>::infinity();
    ExpectInvalidAdmission(input);
  }
  {
    auto input = MakeLearnedInput();
    input.certified_energy_correction_bounds.upper =
        std::numeric_limits<double>::quiet_NaN();
    ExpectInvalidAdmission(input);
  }
}

TEST(LearnedCostSnapshot, RejectsOutOfRangeAndNonFiniteCorrections) {
  {
    auto input = MakeLearnedInput();
    input.energy_correction[0] = 1.01F;
    ExpectInvalidAdmission(input);
  }
  {
    auto input = MakeLearnedInput();
    input.nonfatal_risk_correction[1] = -0.51F;
    ExpectInvalidAdmission(input);
  }
  {
    auto input = MakeLearnedInput();
    input.energy_correction[0] =
        std::numeric_limits<float>::infinity();
    ExpectInvalidAdmission(input);
  }
  {
    auto input = MakeLearnedInput();
    input.nonfatal_risk_correction[1] =
        std::numeric_limits<float>::quiet_NaN();
    ExpectInvalidAdmission(input);
  }
}

TEST(BoundedSoftCost, NullSnapshotFallsBackToFiniteAnalyticCost) {
  const auto map = MakeMap();
  ASSERT_NE(map, nullptr);
  const auto resolved = RequireResolved(lpp::ComposeBoundedSoftCost(
      kAnalyticEnergy, kAnalyticRisk, nullptr, *map, MakeConfig()));

  EXPECT_EQ(resolved.source, lpp::SoftCostSource::kAnalyticOnly);
  EXPECT_EQ(resolved.energy,
            std::vector<float>(kAnalyticEnergy.begin(),
                               kAnalyticEnergy.end()));
  EXPECT_EQ(resolved.nonfatal_risk,
            std::vector<float>(kAnalyticRisk.begin(), kAnalyticRisk.end()));
  EXPECT_EQ(resolved.fallback_reason_code, "learned_snapshot_missing");
}

TEST(BoundedSoftCost, DisabledPolicyFallsBackToAnalyticCost) {
  const auto map = MakeMap();
  const auto learned = MakeLearned();
  ASSERT_NE(map, nullptr);
  ASSERT_NE(learned, nullptr);
  auto config = MakeConfig();
  config.learned_cost_policy.mode =
      lpp::LearnedCostPolicy::Mode::kDisabled;
  const auto resolved = RequireResolved(lpp::ComposeBoundedSoftCost(
      kAnalyticEnergy, kAnalyticRisk, learned, *map, config));

  EXPECT_EQ(resolved.source, lpp::SoftCostSource::kAnalyticOnly);
  EXPECT_EQ(resolved.fallback_reason_code, "learned_cost_disabled");
}

TEST(BoundedSoftCost, EveryPinnedIdentityMismatchHasStableFallback) {
  const auto map = MakeMap();
  ASSERT_NE(map, nullptr);

  {
    auto input = MakeLearnedInput();
    input.source_map_snapshot_ref = MakeRef("other-map", 7U, 'd');
    const auto result = lpp::LearnedCostSnapshot::Create(input, 4U);
    ASSERT_TRUE(lpp::IsOk(result));
    const auto resolved = RequireResolved(lpp::ComposeBoundedSoftCost(
        kAnalyticEnergy, kAnalyticRisk,
        std::get<std::shared_ptr<const lpp::LearnedCostSnapshot>>(result),
        *map, MakeConfig()));
    EXPECT_EQ(resolved.fallback_reason_code,
              "learned_map_snapshot_mismatch");
  }
  {
    auto input = MakeLearnedInput();
    input.source_map_revision = 8U;
    const auto result = lpp::LearnedCostSnapshot::Create(input, 4U);
    ASSERT_TRUE(lpp::IsOk(result));
    const auto resolved = RequireResolved(lpp::ComposeBoundedSoftCost(
        kAnalyticEnergy, kAnalyticRisk,
        std::get<std::shared_ptr<const lpp::LearnedCostSnapshot>>(result),
        *map, MakeConfig()));
    EXPECT_EQ(resolved.fallback_reason_code,
              "learned_map_revision_mismatch");
  }
  {
    auto input = MakeLearnedInput();
    input.frame_id = "other-map-frame";
    const auto result = lpp::LearnedCostSnapshot::Create(input, 4U);
    ASSERT_TRUE(lpp::IsOk(result));
    const auto resolved = RequireResolved(lpp::ComposeBoundedSoftCost(
        kAnalyticEnergy, kAnalyticRisk,
        std::get<std::shared_ptr<const lpp::LearnedCostSnapshot>>(result),
        *map, MakeConfig()));
    EXPECT_EQ(resolved.fallback_reason_code, "learned_frame_mismatch");
  }
}

TEST(BoundedSoftCost, UnconfiguredOrWrongModelFallsBackToAnalyticCost) {
  const auto map = MakeMap();
  const auto learned = MakeLearned();
  ASSERT_NE(map, nullptr);
  ASSERT_NE(learned, nullptr);

  {
    auto config = MakeConfig();
    config.learned_cost_model_ref.reset();
    const auto resolved = RequireResolved(lpp::ComposeBoundedSoftCost(
        kAnalyticEnergy, kAnalyticRisk, learned, *map, config));
    EXPECT_EQ(resolved.fallback_reason_code, "learned_model_unconfigured");
  }
  {
    auto config = MakeConfig();
    config.learned_cost_model_ref = MakeRef("other-model", 1U, 'd');
    const auto resolved = RequireResolved(lpp::ComposeBoundedSoftCost(
        kAnalyticEnergy, kAnalyticRisk, learned, *map, config));
    EXPECT_EQ(resolved.fallback_reason_code, "learned_model_mismatch");
  }
}

TEST(BoundedSoftCost, LearnedShapeMismatchFallsBackToAnalyticCost) {
  const auto map = MakeMap();
  auto input = MakeLearnedInput();
  input.energy_correction.pop_back();
  input.nonfatal_risk_correction.pop_back();
  const auto created = lpp::LearnedCostSnapshot::Create(input, 3U);
  ASSERT_TRUE(lpp::IsOk(created));
  const auto learned =
      std::get<std::shared_ptr<const lpp::LearnedCostSnapshot>>(created);

  const auto resolved = RequireResolved(lpp::ComposeBoundedSoftCost(
      kAnalyticEnergy, kAnalyticRisk, learned, *map, MakeConfig()));
  EXPECT_EQ(resolved.source, lpp::SoftCostSource::kAnalyticOnly);
  EXPECT_EQ(resolved.fallback_reason_code, "learned_shape_mismatch");
}

TEST(BoundedSoftCost, TightensCorrectionsByCertifiedAndPolicyBounds) {
  const auto map = MakeMap();
  const auto learned = MakeLearned();
  ASSERT_NE(map, nullptr);
  ASSERT_NE(learned, nullptr);

  const auto resolved = RequireResolved(lpp::ComposeBoundedSoftCost(
      kAnalyticEnergy, kAnalyticRisk, learned, *map, MakeConfig()));
  ASSERT_EQ(resolved.source,
            lpp::SoftCostSource::kAnalyticPlusPinnedLearned);
  EXPECT_FALSE(resolved.fallback_reason_code.has_value());
  EXPECT_FLOAT_EQ(resolved.energy[0], 1.1F);
  EXPECT_FLOAT_EQ(resolved.energy[1], 2.25F);
  EXPECT_FLOAT_EQ(resolved.energy[2], 2.75F);
  EXPECT_FLOAT_EQ(resolved.energy[3], 4.25F);
  EXPECT_FLOAT_EQ(resolved.nonfatal_risk[0], 0.25F);
  EXPECT_FLOAT_EQ(resolved.nonfatal_risk[1], 0.4F);
  EXPECT_FLOAT_EQ(resolved.nonfatal_risk[2], 0.3F);
  EXPECT_FLOAT_EQ(resolved.nonfatal_risk[3], 0.6F);
}

TEST(BoundedSoftCost, InvalidPolicyBoundsFallBackWithoutClampingConfig) {
  const auto map = MakeMap();
  const auto learned = MakeLearned();
  ASSERT_NE(map, nullptr);
  ASSERT_NE(learned, nullptr);

  for (const double invalid :
       {-1.0, std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::quiet_NaN()}) {
    auto config = MakeConfig();
    config.learned_cost_policy.maximum_absolute_energy_correction =
        invalid;
    const auto resolved = RequireResolved(lpp::ComposeBoundedSoftCost(
        kAnalyticEnergy, kAnalyticRisk, learned, *map, config));
    EXPECT_EQ(resolved.source, lpp::SoftCostSource::kAnalyticOnly);
    EXPECT_EQ(resolved.fallback_reason_code,
              "learned_policy_bounds_invalid");
  }
}

TEST(BoundedSoftCost, RejectsInvalidAnalyticInputInsteadOfManufacturingCost) {
  const auto map = MakeMap();
  const auto learned = MakeLearned();
  ASSERT_NE(map, nullptr);
  ASSERT_NE(learned, nullptr);

  {
    auto energy = kAnalyticEnergy;
    energy[1] = std::numeric_limits<float>::infinity();
    const auto result = lpp::ComposeBoundedSoftCost(
        energy, kAnalyticRisk, learned, *map, MakeConfig());
    ASSERT_FALSE(lpp::IsOk(result));
    const auto& error = std::get<lpp::Error>(result);
    EXPECT_EQ(error.code, lpp::ErrorCode::kInvalidArgument);
    EXPECT_EQ(error.message, "analytic_cost_invalid");
  }
  {
    const std::array<float, 3U> wrong_shape{1.0F, 2.0F, 3.0F};
    const auto result = lpp::ComposeBoundedSoftCost(
        wrong_shape, kAnalyticRisk, learned, *map, MakeConfig());
    ASSERT_FALSE(lpp::IsOk(result));
    EXPECT_EQ(std::get<lpp::Error>(result).message,
              "analytic_cost_invalid");
  }
}

TEST(BoundedSoftCost, OverflowedCompositionFallsBackAtomically) {
  auto input = MakeLearnedInput();
  input.certified_energy_correction_bounds = {
      -static_cast<double>(std::numeric_limits<float>::max()),
      static_cast<double>(std::numeric_limits<float>::max())};
  input.energy_correction = {
      std::numeric_limits<float>::max(), 0.0F, 0.0F, 0.0F};
  const auto created = lpp::LearnedCostSnapshot::Create(input, 4U);
  ASSERT_TRUE(lpp::IsOk(created));
  const auto learned =
      std::get<std::shared_ptr<const lpp::LearnedCostSnapshot>>(created);
  const auto map = MakeMap();
  ASSERT_NE(map, nullptr);
  auto config = MakeConfig();
  config.learned_cost_policy.maximum_absolute_energy_correction =
      static_cast<double>(std::numeric_limits<float>::max());
  const std::array<float, 4U> analytic{
      std::numeric_limits<float>::max(), 2.0F, 3.0F, 4.0F};

  const auto resolved = RequireResolved(lpp::ComposeBoundedSoftCost(
      analytic, kAnalyticRisk, learned, *map, config));
  EXPECT_EQ(resolved.source, lpp::SoftCostSource::kAnalyticOnly);
  EXPECT_EQ(resolved.fallback_reason_code,
            "learned_composition_non_finite");
  EXPECT_EQ(resolved.energy,
            std::vector<float>(analytic.begin(), analytic.end()));
}

TEST(BoundedSoftCost, RepeatedCompositionIsDeterministic) {
  const auto map = MakeMap();
  const auto learned = MakeLearned();
  ASSERT_NE(map, nullptr);
  ASSERT_NE(learned, nullptr);

  const auto first = lpp::ComposeBoundedSoftCost(
      kAnalyticEnergy, kAnalyticRisk, learned, *map, MakeConfig());
  const auto second = lpp::ComposeBoundedSoftCost(
      kAnalyticEnergy, kAnalyticRisk, learned, *map, MakeConfig());
  ASSERT_TRUE(lpp::IsOk(first));
  ASSERT_TRUE(lpp::IsOk(second));
  const auto& first_value = std::get<lpp::ResolvedSoftCost>(first);
  const auto& second_value = std::get<lpp::ResolvedSoftCost>(second);
  EXPECT_EQ(first_value.energy, second_value.energy);
  EXPECT_EQ(first_value.nonfatal_risk, second_value.nonfatal_risk);
  EXPECT_EQ(first_value.source, second_value.source);
  EXPECT_EQ(first_value.fallback_reason_code,
            second_value.fallback_reason_code);
}

// The compile-time API shape is intentional: primary traversal time, hard
// feasibility masks, and model-loading/inference callbacks are not arguments
// to ComposeBoundedSoftCost.
