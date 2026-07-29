#include <array>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <variant>

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include "lunar_path_planner/v3/codec/jcs_canonicalizer.hpp"
#include "lunar_path_planner/v3/codec/json_codec.hpp"
#include "lunar_path_planner/v3/map/immutable_snapshot.hpp"

namespace lpp = lunar::planning::v3;

#if defined(LPP_V3_SCHEMA_CONFORMANCE_STANDALONE)
int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
#endif

namespace {

constexpr std::string_view kMapHash =
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
constexpr std::string_view kCapabilityHash =
    "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
constexpr std::string_view kConfigHash =
    "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";

struct ValidFixture final {
  std::string_view request_path;
  std::string_view response_path;
  lpp::PlatformType platform_type;
};

constexpr std::array<ValidFixture, 3U> kValidFixtures{{
    {"valid/wheel-request.json", "valid/wheel-response.json",
     lpp::PlatformType::kWheeled},
    {"valid/legged-request.json", "valid/legged-response.json",
     lpp::PlatformType::kLegged},
    {"valid/hopper-request.json", "valid/hopper-response.json",
     lpp::PlatformType::kHopper},
}};

constexpr std::array<std::string_view, 2U> kInvalidRequests{{
    "invalid/runtime-deadline.json",
    "invalid/platform-state-mismatch.json",
}};

constexpr std::array<std::string_view, 4U> kInvalidResponses{{
    "invalid/hold-with-bundle.json",
    "invalid/component-hash-conflict.json",
    "invalid/zero-ballistic-flight-time.json",
    "invalid/activation-provenance-mismatch.json",
}};

std::filesystem::path FixtureRoot() {
  return std::filesystem::path{LPP_V3_SCHEMA_ROOT}.parent_path().parent_path() /
         "tests" / "fixtures" / "v3" / "schema";
}

std::string ReadFixture(std::string_view relative_path) {
  const auto path = FixtureRoot() / relative_path;
  std::ifstream input{path, std::ios::binary};
  if (!input.is_open()) {
    throw std::runtime_error{"cannot open fixture: " + path.string()};
  }
  return {std::istreambuf_iterator<char>{input},
          std::istreambuf_iterator<char>{}};
}

std::string RepeatedHash(char digit) { return std::string(64U, digit); }

lpp::ContentRef MakeRef(std::string id, char hash_digit) {
  return {std::move(id), 1U, RepeatedHash(hash_digit)};
}

std::shared_ptr<const lpp::ImmutableMapSnapshot> MakeMapSnapshot() {
  lpp::MapSnapshotInput input{
      .snapshot_ref = {"map-snapshot-1", 1U, std::string{kMapHash}},
      .map_revision = 1U,
      .immutable_data_handle = "map-handle-1",
      .source_time = {"mission-clock", std::chrono::nanoseconds{90}},
      .bounds = {{-10.0, -10.0, -1.0}, {10.0, 10.0, 2.0}},
      .geometry = {.width = 1U,
                   .height = 1U,
                   .resolution_m = 0.25,
                   .origin_m = {-10.0, -10.0},
                   .frame_id = "map"},
      .layer_manifest =
          {
              {lpp::LayerKind::kKnownMask, MakeRef("known-mask-1", '1')},
              {lpp::LayerKind::kElevation, MakeRef("elevation-1", '2')},
              {lpp::LayerKind::kTerrainNormal,
               MakeRef("terrain-normal-1", '3')},
              {lpp::LayerKind::kRoughness, MakeRef("roughness-1", '4')},
              {lpp::LayerKind::kHardObstacle, MakeRef("hard-obstacle-1", '5')},
              {lpp::LayerKind::kConfidence, MakeRef("confidence-1", '6')},
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
  auto snapshot = lpp::ImmutableMapSnapshot::Create(input);
  if (!lpp::IsOk(snapshot)) {
    throw std::runtime_error{std::get<lpp::Error>(snapshot).message};
  }
  return std::get<std::shared_ptr<const lpp::ImmutableMapSnapshot>>(snapshot);
}

class FixtureContractObject final : public lpp::ImmutableContractObject {
public:
  FixtureContractObject(lpp::ContentRef ref, lpp::ContractObjectKind kind)
      : ref_{std::move(ref)}, kind_{kind} {}

  [[nodiscard]] lpp::ContentRef content_ref() const override { return ref_; }

  [[nodiscard]] lpp::ContractObjectKind kind() const override { return kind_; }

private:
  lpp::ContentRef ref_;
  lpp::ContractObjectKind kind_;
};

class FixtureRegistry final : public lpp::ContractObjectRegistry {
public:
  FixtureRegistry()
      : map_snapshot_{MakeMapSnapshot()},
        wheel_capability_{MakeCapability("wheel-capability-1")},
        legged_capability_{MakeCapability("legged-capability-1")},
        hopper_capability_{MakeCapability("hopper-capability-1")},
        algorithm_config_{std::make_shared<lpp::PlannerAlgorithmConfig>()} {
    algorithm_config_->content_ref = {"planner-config-1", 1U,
                                      std::string{kConfigHash}};
  }

  [[nodiscard]] std::shared_ptr<const lpp::ImmutableMapSnapshot>
  FindMapSnapshot(const lpp::ContentRef &ref,
                  std::string_view handle) const override {
    if (ref == map_snapshot_->snapshot_ref() &&
        handle == map_snapshot_->immutable_data_handle()) {
      return map_snapshot_;
    }
    return nullptr;
  }

  [[nodiscard]] std::shared_ptr<const lpp::SafetyCapabilityProfile>
  FindSafetyCapability(const lpp::ContentRef &ref) const override {
    for (const auto &capability :
         {wheel_capability_, legged_capability_, hopper_capability_}) {
      if (capability->content_ref == ref) {
        return capability;
      }
    }
    return nullptr;
  }

  [[nodiscard]] std::shared_ptr<const lpp::PlannerAlgorithmConfig>
  FindAlgorithmConfig(const lpp::ContentRef &ref) const override {
    return algorithm_config_->content_ref == ref ? algorithm_config_ : nullptr;
  }

  [[nodiscard]] std::shared_ptr<const lpp::LearnedCostSnapshot>
  FindLearnedCost(const lpp::ContentRef &, std::string_view) const override {
    return nullptr;
  }

  [[nodiscard]] lpp::Result<lpp::ResolvedCapabilityBindings>
  ResolveCapabilityBindings(
      const lpp::SafetyCapabilityProfile &) const override {
    return lpp::ResolvedCapabilityBindings{};
  }

  [[nodiscard]]
  lpp::Result<std::shared_ptr<const lpp::ImmutableContractObject>>
  Resolve(const lpp::ContentRef &ref,
          lpp::ContractObjectKind kind) const override {
    return std::make_shared<const FixtureContractObject>(ref, kind);
  }

private:
  static std::shared_ptr<lpp::SafetyCapabilityProfile>
  MakeCapability(std::string id) {
    auto capability = std::make_shared<lpp::SafetyCapabilityProfile>();
    capability->content_ref = {std::move(id), 1U, std::string{kCapabilityHash}};
    return capability;
  }

  std::shared_ptr<const lpp::ImmutableMapSnapshot> map_snapshot_;
  std::shared_ptr<lpp::SafetyCapabilityProfile> wheel_capability_;
  std::shared_ptr<lpp::SafetyCapabilityProfile> legged_capability_;
  std::shared_ptr<lpp::SafetyCapabilityProfile> hopper_capability_;
  std::shared_ptr<lpp::PlannerAlgorithmConfig> algorithm_config_;
};

std::string Canonicalize(std::string_view payload) {
  const auto canonical =
      lpp::JcsCanonicalizer::Canonicalize(nlohmann::json::parse(payload));
  if (!lpp::IsOk(canonical)) {
    throw std::runtime_error{std::get<lpp::Error>(canonical).message};
  }
  return std::get<std::string>(canonical);
}

std::string ErrorSummary(const lpp::Error &error) {
  return error.field_path + ": " + error.message;
}

} // namespace

TEST(SchemaCodecConformanceTest,
     ValidRequestsRoundTripThroughTheSharedTypedCodecCanonically) {
  FixtureRegistry registry;

  for (const auto &fixture : kValidFixtures) {
    const std::string payload = ReadFixture(fixture.request_path);
    const auto decoded =
        lpp::JsonCodec::DecodePlanningRequest(payload, registry);
    ASSERT_TRUE(lpp::IsOk(decoded))
        << fixture.request_path << ": "
        << ErrorSummary(std::get<lpp::Error>(decoded));

    const auto &request = std::get<lpp::PlanningRequest>(decoded);
    EXPECT_EQ(request.platform_type, fixture.platform_type)
        << fixture.request_path;

    const std::string first_encoded =
        lpp::JsonCodec::EncodePlanningRequest(request);
    EXPECT_EQ(Canonicalize(first_encoded), Canonicalize(payload))
        << fixture.request_path;

    const auto decoded_again =
        lpp::JsonCodec::DecodePlanningRequest(first_encoded, registry);
    ASSERT_TRUE(lpp::IsOk(decoded_again))
        << fixture.request_path << ": "
        << ErrorSummary(std::get<lpp::Error>(decoded_again));
    const std::string second_encoded = lpp::JsonCodec::EncodePlanningRequest(
        std::get<lpp::PlanningRequest>(decoded_again));
    EXPECT_EQ(Canonicalize(second_encoded), Canonicalize(first_encoded))
        << fixture.request_path;
  }
}

TEST(SchemaCodecConformanceTest,
     ValidResponsesRoundTripThroughTheSharedTypedCodecCanonically) {
  FixtureRegistry registry;

  for (const auto &fixture : kValidFixtures) {
    const auto request = lpp::JsonCodec::DecodePlanningRequest(
        ReadFixture(fixture.request_path), registry);
    ASSERT_TRUE(lpp::IsOk(request)) << fixture.request_path;
    const lpp::ReferenceActivationContext context{
        .request = std::get<lpp::PlanningRequest>(request),
        .registry = registry,
    };

    const std::string payload = ReadFixture(fixture.response_path);
    const auto decoded =
        lpp::JsonCodec::DecodePlanningResponse(payload, context);
    ASSERT_TRUE(lpp::IsOk(decoded))
        << fixture.response_path << ": "
        << ErrorSummary(std::get<lpp::Error>(decoded));

    const std::string first_encoded = lpp::JsonCodec::EncodePlanningResponse(
        std::get<lpp::PlanningResponse>(decoded));
    EXPECT_EQ(Canonicalize(first_encoded), Canonicalize(payload))
        << fixture.response_path;

    const auto decoded_again =
        lpp::JsonCodec::DecodePlanningResponse(first_encoded, context);
    ASSERT_TRUE(lpp::IsOk(decoded_again))
        << fixture.response_path << ": "
        << ErrorSummary(std::get<lpp::Error>(decoded_again));
    const std::string second_encoded = lpp::JsonCodec::EncodePlanningResponse(
        std::get<lpp::PlanningResponse>(decoded_again));
    EXPECT_EQ(Canonicalize(second_encoded), Canonicalize(first_encoded))
        << fixture.response_path;
  }
}

TEST(SchemaCodecConformanceTest,
     InvalidRequestFixturesAreRejectedBeforeRegistryResolutionCanSucceed) {
  FixtureRegistry registry;
  for (const std::string_view fixture : kInvalidRequests) {
    const auto decoded =
        lpp::JsonCodec::DecodePlanningRequest(ReadFixture(fixture), registry);
    EXPECT_FALSE(lpp::IsOk(decoded)) << fixture;
  }
}

TEST(SchemaCodecConformanceTest,
     InvalidResponseFixturesAreRejectedByTheSharedTypedCodec) {
  FixtureRegistry registry;
  lpp::PlanningRequest request{};
  const lpp::ReferenceActivationContext context{
      .request = request,
      .registry = registry,
  };
  for (const std::string_view fixture : kInvalidResponses) {
    const auto decoded =
        lpp::JsonCodec::DecodePlanningResponse(ReadFixture(fixture), context);
    EXPECT_FALSE(lpp::IsOk(decoded)) << fixture;
  }
}
