#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <variant>

#include <gtest/gtest.h>

#include "integration/wheel/wheel_planner_test_fixture.hpp"
#include "lunar_path_planner/v3/codec/json_codec.hpp"
#include "lunar_path_planner/v3/wheel/wheel_planner.hpp"

namespace lunar::planning::v3 {
namespace {

[[nodiscard]] double EvaluatePolynomial(
    const CubicPolynomialSegment& segment,
    DurationNanoseconds offset) {
  const double seconds =
      std::chrono::duration<double>(
          offset.value - segment.start_offset.value)
          .count();
  const auto& c = segment.coefficients;
  return ((c[3] * seconds + c[2]) * seconds + c[1]) *
             seconds +
         c[0];
}

[[nodiscard]] std::optional<double> StartRate(
    const WheeledSegment& segment) {
  return std::visit(
      [](const auto& concrete) {
        using Segment = std::decay_t<decltype(concrete)>;
        if constexpr (std::is_same_v<Segment, DriveSegment>) {
          return EvaluateMonotoneTimeScalingDerivative(
              concrete.time_scaling,
              concrete.time_interval.start_offset);
        } else {
          return EvaluateScalarTrajectoryDerivative(
              concrete.unwrapped_yaw_rad,
              concrete.time_interval.start_offset);
        }
      },
      segment);
}

[[nodiscard]] std::optional<double> EndRate(
    const WheeledSegment& segment) {
  return std::visit(
      [](const auto& concrete) {
        using Segment = std::decay_t<decltype(concrete)>;
        if constexpr (std::is_same_v<Segment, DriveSegment>) {
          return EvaluateMonotoneTimeScalingDerivative(
              concrete.time_scaling,
              concrete.time_interval.end_offset);
        } else {
          return EvaluateScalarTrajectoryDerivative(
              concrete.unwrapped_yaw_rad,
              concrete.time_interval.end_offset);
        }
      },
      segment);
}

[[nodiscard]] bool Contiguous(
    const WheeledReference& reference) {
  if (reference.segments.empty()) {
    return false;
  }
  std::chrono::nanoseconds prior_end{};
  for (std::size_t index = 0U;
       index < reference.segments.size(); ++index) {
    const auto interval = std::visit(
        [](const auto& segment) {
          return segment.time_interval;
        },
        reference.segments[index]);
    if (index == 0U &&
        interval.start_offset.value.count() != 0) {
      return false;
    }
    if (index > 0U && interval.start_offset.value != prior_end) {
      return false;
    }
    if (interval.end_offset.value <=
        interval.start_offset.value) {
      return false;
    }
    prior_end = interval.end_offset.value;
  }
  return true;
}

[[nodiscard]] bool DriveScalingsMonotone(
    const WheeledReference& reference) {
  for (const auto& segment : reference.segments) {
    const auto* drive = std::get_if<DriveSegment>(&segment);
    if (drive == nullptr) {
      continue;
    }
    double previous = -1.0;
    for (const auto& polynomial :
         drive->time_scaling.segments) {
      const auto duration =
          polynomial.end_offset.value -
          polynomial.start_offset.value;
      for (std::int64_t sample = 0; sample <= 8; ++sample) {
        const auto offset = DurationNanoseconds{
            polynomial.start_offset.value +
            duration * sample / 8};
        const double value =
            EvaluatePolynomial(polynomial, offset);
        if (!std::isfinite(value) ||
            value + 1.0e-9 < previous) {
          return false;
        }
        previous = value;
      }
    }
    if (std::abs(previous - 1.0) > 1.0e-7) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool BoundariesAreStopped(
    const WheeledReference& reference) {
  for (const auto& segment : reference.segments) {
    const auto start = StartRate(segment);
    const auto finish = EndRate(segment);
    if (!start.has_value() || !finish.has_value() ||
        std::abs(*start) > 1.0e-8 ||
        std::abs(*finish) > 1.0e-8) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool EndsAtAnchor(
    const WheeledReference& reference) {
  if (reference.segments.empty()) {
    return false;
  }
  PoseXyzYaw endpoint;
  const auto& final = reference.segments.back();
  if (const auto* drive = std::get_if<DriveSegment>(&final)) {
    endpoint = std::visit(
        [](const auto& path) {
          using Path = std::decay_t<decltype(path)>;
          if constexpr (std::is_same_v<
                            Path,
                            ClampedCubicBSplinePath>) {
            return path.control_points.back();
          } else {
            return path.primitives.back().end_pose;
          }
        },
        drive->geometric_path);
  } else {
    const auto& spin = std::get<SpinSegment>(final);
    const auto yaw = EvaluateScalarTrajectory(
        spin.unwrapped_yaw_rad,
        spin.time_interval.end_offset);
    if (!yaw.has_value()) {
      return false;
    }
    endpoint = {
        .position_m = spin.fixed_position_m,
        .yaw_rad = *yaw,
    };
  }
  const auto& anchor = reference.safe_stop_anchor.pose;
  return std::hypot(
             endpoint.position_m.x - anchor.position_m.x,
             endpoint.position_m.y - anchor.position_m.y) <=
             1.0e-8 &&
         std::abs(endpoint.position_m.z -
                  anchor.position_m.z) <= 1.0e-8 &&
         std::abs(endpoint.yaw_rad - anchor.yaw_rad) <= 1.0e-8;
}

TEST(WheelReferencePropertyTest,
     ThousandFixedSeedsRespectReferenceInvariants) {
  WheelPlanner planner;
  for (std::uint64_t seed = 0U; seed < 1000U; ++seed) {
    auto fixture = test::MakeFixture((seed & 1U) != 0U);
    fixture.request.request_id += "-" + std::to_string(seed);

    const auto result =
        planner.Plan(fixture.request, fixture.terminal);

    ASSERT_TRUE(IsOk(result)) << "seed=" << seed;
    const auto& reference =
        std::get<WheeledReference>(result);
    EXPECT_TRUE(Contiguous(reference)) << "seed=" << seed;
    EXPECT_TRUE(DriveScalingsMonotone(reference))
        << "seed=" << seed;
    EXPECT_TRUE(BoundariesAreStopped(reference))
        << "seed=" << seed;
    EXPECT_TRUE(EndsAtAnchor(reference)) << "seed=" << seed;
    const auto hash =
        CanonicalReferenceHash(PlatformReference{reference});
    ASSERT_TRUE(IsOk(hash)) << "seed=" << seed;
    EXPECT_EQ(reference.reference_hash,
              std::get<Sha256Digest>(hash))
        << "seed=" << seed;
  }
}

TEST(WheelReferencePropertyTest,
     SameInputIsDeterministicAcrossOneHundredRuns) {
  auto fixture = test::MakeFixture();
  WheelPlanner planner;
  std::string expected_hash;
  for (std::size_t repeat = 0U; repeat < 100U; ++repeat) {
    const auto result =
        planner.Plan(fixture.request, fixture.terminal);
    ASSERT_TRUE(IsOk(result));
    const auto& reference =
        std::get<WheeledReference>(result);
    if (repeat == 0U) {
      expected_hash = reference.reference_hash;
    }
    EXPECT_EQ(reference.reference_hash, expected_hash);
  }
}

}  // namespace
}  // namespace lunar::planning::v3
