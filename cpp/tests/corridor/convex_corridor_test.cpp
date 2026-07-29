#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "lunar_path_planner/v3/corridor/convex_corridor.hpp"

namespace lpp = lunar::planning::v3;

namespace {

constexpr double kGeometryTolerance = 1.0e-9;

std::size_t Index(std::size_t x, std::size_t y,
                  std::size_t width) {
  return y * width + x;
}

lpp::SafeProjection OpenProjection(std::size_t width = 12U,
                                   std::size_t height = 9U,
                                   float clearance_m = 2.4F) {
  const std::size_t count = width * height;
  lpp::SafeProjection projection;
  projection.geometry = {
      .width = width,
      .height = height,
      .resolution_m = 1.0,
      .origin_m = {0.0, 0.0},
      .frame_id = "map",
  };
  projection.known_mask.assign(count, 1U);
  projection.hard_feasible_mask.assign(count, 1U);
  projection.esdf_clearance_m.assign(count, clearance_m);
  return projection;
}

lpp::PathControlPoint Point(double s_m, double x_m,
                            double y_m) {
  return {
      .s_m = s_m,
      .position_m = {x_m, y_m},
  };
}

lpp::CorridorTightening Tightening(
    double footprint_m = 0.25, double tracking_m = 0.10,
    double additional_m = 0.05,
    std::vector<lpp::HalfPlane2> platform_planes = {}) {
  return {
      .footprint_support_radius_m = footprint_m,
      .tracking_error_bound_m = tracking_m,
      .additional_margin_m = additional_m,
      .platform_half_planes = std::move(platform_planes),
  };
}

lpp::CorridorRequest Request(
    const lpp::SafeProjection& projection,
    std::span<const lpp::PathControlPoint> centerline,
    lpp::CorridorTightening tightening = Tightening(),
    std::size_t max_planes = 64U,
    std::size_t max_iterations = 256U) {
  return {
      .projection = projection,
      .validated_centerline = centerline,
      .tightening = std::move(tightening),
      .max_planes = max_planes,
      .max_iterations = max_iterations,
  };
}

double Evaluation(const lpp::HalfPlane2& plane,
                  const lpp::Vec2& point) {
  return plane.outward_unit_normal.x * point.x +
         plane.outward_unit_normal.y * point.y -
         plane.upper_offset_m;
}

bool Contains(const lpp::ConvexCorridorCell& cell,
              const lpp::Vec2& point) {
  return std::ranges::all_of(
      cell.half_planes, [&](const lpp::HalfPlane2& plane) {
        return Evaluation(plane, point) <= kGeometryTolerance;
      });
}

std::vector<lpp::Vec2> Vertices(
    std::span<const lpp::HalfPlane2> planes) {
  std::vector<lpp::Vec2> vertices;
  for (std::size_t first = 0U; first < planes.size(); ++first) {
    for (std::size_t second = first + 1U;
         second < planes.size(); ++second) {
      const auto& a = planes[first];
      const auto& b = planes[second];
      const double determinant =
          a.outward_unit_normal.x * b.outward_unit_normal.y -
          a.outward_unit_normal.y * b.outward_unit_normal.x;
      if (std::abs(determinant) <= kGeometryTolerance) {
        continue;
      }
      const lpp::Vec2 vertex{
          .x = (a.upper_offset_m * b.outward_unit_normal.y -
                a.outward_unit_normal.y * b.upper_offset_m) /
               determinant,
          .y = (a.outward_unit_normal.x * b.upper_offset_m -
                a.upper_offset_m * b.outward_unit_normal.x) /
               determinant,
      };
      if (std::ranges::all_of(
              planes, [&](const lpp::HalfPlane2& plane) {
                return Evaluation(plane, vertex) <=
                       kGeometryTolerance;
              })) {
        vertices.push_back(vertex);
      }
    }
  }
  return vertices;
}

bool IntersectionsOverlap(const lpp::ConvexCorridorCell& first,
                          const lpp::ConvexCorridorCell& second) {
  std::vector<lpp::HalfPlane2> combined = first.half_planes;
  combined.insert(combined.end(), second.half_planes.begin(),
                  second.half_planes.end());
  return !Vertices(combined).empty();
}

double PositiveXBound(const lpp::ConvexCorridorCell& cell) {
  const auto plane = std::ranges::find_if(
      cell.half_planes, [](const lpp::HalfPlane2& candidate) {
        return std::abs(candidate.outward_unit_normal.x - 1.0) <=
                   kGeometryTolerance &&
               std::abs(candidate.outward_unit_normal.y) <=
                   kGeometryTolerance;
      });
  EXPECT_NE(plane, cell.half_planes.end());
  return plane == cell.half_planes.end()
             ? std::numeric_limits<double>::quiet_NaN()
             : plane->upper_offset_m;
}

void ExpectEquivalent(const lpp::CorridorResult& first,
                      const lpp::CorridorResult& second) {
  ASSERT_EQ(first.status, second.status);
  ASSERT_EQ(first.fallback, second.fallback);
  ASSERT_EQ(first.reason_code, second.reason_code);
  ASSERT_EQ(first.iterations, second.iterations);
  ASSERT_EQ(first.cells.size(), second.cells.size());
  for (std::size_t index = 0U; index < first.cells.size(); ++index) {
    const auto& a = first.cells[index];
    const auto& b = second.cells[index];
    EXPECT_EQ(a.stable_cell_id, b.stable_cell_id);
    EXPECT_DOUBLE_EQ(a.centerline_s_begin, b.centerline_s_begin);
    EXPECT_DOUBLE_EQ(a.centerline_s_end, b.centerline_s_end);
    ASSERT_EQ(a.half_planes.size(), b.half_planes.size());
    for (std::size_t plane = 0U; plane < a.half_planes.size();
         ++plane) {
      EXPECT_DOUBLE_EQ(
          a.half_planes[plane].outward_unit_normal.x,
          b.half_planes[plane].outward_unit_normal.x);
      EXPECT_DOUBLE_EQ(
          a.half_planes[plane].outward_unit_normal.y,
          b.half_planes[plane].outward_unit_normal.y);
      EXPECT_DOUBLE_EQ(a.half_planes[plane].upper_offset_m,
                       b.half_planes[plane].upper_offset_m);
    }
  }
}

}  // namespace

TEST(ConvexCorridor,
     CoversCenterlineWithBoundedOverlappingCertifiedCells) {
  const auto projection = OpenProjection();
  const std::vector centerline{
      Point(0.0, 2.5, 4.5), Point(2.0, 4.5, 4.5),
      Point(4.0, 6.5, 4.5), Point(6.0, 8.5, 4.5),
  };

  const auto result =
      lpp::BuildConvexCorridor(Request(projection, centerline));

  ASSERT_EQ(result.status, lpp::CorridorStatus::kCertified);
  ASSERT_FALSE(result.cells.empty());
  EXPECT_EQ(result.fallback, lpp::CorridorFallback::kNone);
  EXPECT_EQ(result.reason_code, "certified");
  for (const auto& sample : centerline) {
    EXPECT_TRUE(std::ranges::any_of(
        result.cells, [&](const lpp::ConvexCorridorCell& cell) {
          return sample.s_m + kGeometryTolerance >=
                     cell.centerline_s_begin &&
                 sample.s_m <=
                     cell.centerline_s_end + kGeometryTolerance &&
                 Contains(cell, sample.position_m);
        }));
  }
  for (const auto& cell : result.cells) {
    EXPECT_LE(cell.half_planes.size(), 64U);
    const auto vertices = Vertices(cell.half_planes);
    EXPECT_GE(vertices.size(), 3U);
    for (const auto& plane : cell.half_planes) {
      EXPECT_TRUE(std::isfinite(plane.outward_unit_normal.x));
      EXPECT_TRUE(std::isfinite(plane.outward_unit_normal.y));
      EXPECT_TRUE(std::isfinite(plane.upper_offset_m));
      EXPECT_NEAR(
          std::hypot(plane.outward_unit_normal.x,
                     plane.outward_unit_normal.y),
          1.0, kGeometryTolerance);
      if (plane.outward_unit_normal.x == 0.0) {
        EXPECT_FALSE(
            std::signbit(plane.outward_unit_normal.x));
      }
      if (plane.outward_unit_normal.y == 0.0) {
        EXPECT_FALSE(
            std::signbit(plane.outward_unit_normal.y));
      }
      if (plane.upper_offset_m == 0.0) {
        EXPECT_FALSE(std::signbit(plane.upper_offset_m));
      }
    }
  }
  for (std::size_t index = 1U; index < result.cells.size();
       ++index) {
    EXPECT_TRUE(IntersectionsOverlap(result.cells[index - 1U],
                                     result.cells[index]));
  }
}

TEST(ConvexCorridor,
     ReturnsNoPartialCorridorWhenPinchedCertificationFails) {
  const auto projection = OpenProjection(10U, 5U, 0.75F);
  const std::vector centerline{
      Point(0.0, 2.5, 2.5),
      Point(1.0, 6.5, 2.5),
  };

  const auto result =
      lpp::BuildConvexCorridor(Request(
          projection, centerline, Tightening(0.1, 0.0, 0.0)));

  EXPECT_EQ(result.status,
            lpp::CorridorStatus::kFallbackRequired);
  EXPECT_TRUE(result.cells.empty());
  EXPECT_EQ(
      result.fallback,
      lpp::CorridorFallback::kUseDiscreteValidatedPrimitives);
  EXPECT_EQ(result.reason_code,
            "adjacent_cells_do_not_overlap");
}

TEST(ConvexCorridor,
     RejectsUnknownAndObstacleCenterlineCellsAtomically) {
  auto unknown = OpenProjection();
  unknown.known_mask[Index(3U, 4U, 12U)] = 0U;
  unknown.hard_feasible_mask[Index(3U, 4U, 12U)] = 0U;
  const std::vector unknown_centerline{
      Point(0.0, 2.5, 4.5),
      Point(1.0, 3.5, 4.5),
  };
  const auto unknown_result = lpp::BuildConvexCorridor(
      Request(unknown, unknown_centerline));
  EXPECT_EQ(unknown_result.status,
            lpp::CorridorStatus::kFallbackRequired);
  EXPECT_TRUE(unknown_result.cells.empty());
  EXPECT_EQ(unknown_result.reason_code,
            "centerline_not_hard_feasible");

  auto obstacle = OpenProjection();
  obstacle.hard_feasible_mask[Index(6U, 4U, 12U)] = 0U;
  obstacle.esdf_clearance_m[Index(6U, 4U, 12U)] = 0.0F;
  const std::vector obstacle_centerline{
      Point(0.0, 5.5, 4.5),
      Point(1.0, 6.5, 4.5),
  };
  const auto obstacle_result = lpp::BuildConvexCorridor(
      Request(obstacle, obstacle_centerline));
  EXPECT_EQ(obstacle_result.status,
            lpp::CorridorStatus::kFallbackRequired);
  EXPECT_TRUE(obstacle_result.cells.empty());
  EXPECT_EQ(obstacle_result.reason_code,
            "centerline_not_hard_feasible");
}

TEST(ConvexCorridor,
     AppliesWheelAndLeggedEnvelopeTighteningDeterministically) {
  const auto projection = OpenProjection();
  const std::vector centerline{Point(0.0, 5.5, 4.5)};

  const auto wheel = lpp::BuildConvexCorridor(Request(
      projection, centerline, Tightening(0.25, 0.10, 0.05)));
  const auto legged = lpp::BuildConvexCorridor(Request(
      projection, centerline, Tightening(0.65, 0.10, 0.05)));

  ASSERT_EQ(wheel.status, lpp::CorridorStatus::kCertified);
  ASSERT_EQ(legged.status, lpp::CorridorStatus::kCertified);
  ASSERT_EQ(wheel.cells.size(), 1U);
  ASSERT_EQ(legged.cells.size(), 1U);
  EXPECT_NEAR(PositiveXBound(wheel.cells.front()) -
                  PositiveXBound(legged.cells.front()),
              0.40, kGeometryTolerance);
}

TEST(ConvexCorridor,
     SortsForbiddenCellsAndPlatformPlanesIndependentOfInputOrder) {
  auto forward = OpenProjection(10U, 9U, 3.2F);
  auto reverse = forward;
  const std::vector<std::pair<std::size_t, std::size_t>>
      forbidden{{7U, 4U}, {6U, 6U}, {6U, 2U}};
  for (const auto& [x, y] : forbidden) {
    forward.hard_feasible_mask[Index(x, y, 10U)] = 0U;
    forward.esdf_clearance_m[Index(x, y, 10U)] = 0.0F;
  }
  for (auto item = forbidden.rbegin(); item != forbidden.rend();
       ++item) {
    reverse.hard_feasible_mask[
        Index(item->first, item->second, 10U)] = 0U;
    reverse.esdf_clearance_m[
        Index(item->first, item->second, 10U)] = 0.0F;
  }
  const std::vector centerline{Point(0.0, 4.5, 4.5)};
  const std::vector<lpp::HalfPlane2> first_planes{
      {{0.0, 1.0}, 7.75},
      {{1.0, 0.0}, 8.25},
  };
  const std::vector<lpp::HalfPlane2> reversed_planes{
      first_planes.rbegin(), first_planes.rend()};

  const auto first = lpp::BuildConvexCorridor(Request(
      forward, centerline,
      Tightening(0.2, 0.1, 0.0, first_planes)));
  const auto second = lpp::BuildConvexCorridor(Request(
      reverse, centerline,
      Tightening(0.2, 0.1, 0.0, reversed_planes)));

  ASSERT_EQ(first.status, lpp::CorridorStatus::kCertified);
  ExpectEquivalent(first, second);
  ASSERT_EQ(first.cells.size(), 1U);
  EXPECT_GT(first.cells.front().half_planes.size(), 4U);
}

TEST(ConvexCorridor, EnforcesPlaneAndIterationBoundsWithoutPrefix) {
  auto projection = OpenProjection(10U, 9U, 3.2F);
  for (const auto& [x, y] :
       std::vector<std::pair<std::size_t, std::size_t>>{
           {7U, 4U}, {6U, 6U}, {6U, 2U}}) {
    projection.hard_feasible_mask[Index(x, y, 10U)] = 0U;
    projection.esdf_clearance_m[Index(x, y, 10U)] = 0.0F;
  }
  const std::vector centerline{Point(0.0, 4.5, 4.5)};

  const auto plane_limited = lpp::BuildConvexCorridor(Request(
      projection, centerline, Tightening(0.2, 0.1, 0.0),
      4U, 256U));
  EXPECT_EQ(plane_limited.status,
            lpp::CorridorStatus::kFallbackRequired);
  EXPECT_TRUE(plane_limited.cells.empty());
  EXPECT_EQ(plane_limited.reason_code,
            "plane_limit_exceeded");

  const auto iteration_limited =
      lpp::BuildConvexCorridor(Request(
          projection, centerline,
          Tightening(0.2, 0.1, 0.0), 64U, 1U));
  EXPECT_EQ(iteration_limited.status,
            lpp::CorridorStatus::kFallbackRequired);
  EXPECT_TRUE(iteration_limited.cells.empty());
  EXPECT_EQ(iteration_limited.reason_code,
            "iteration_limit_exceeded");
  EXPECT_EQ(iteration_limited.iterations, 1U);
}

TEST(ConvexCorridor, RejectsStructurallyInvalidRequests) {
  const auto projection = OpenProjection();
  const std::vector valid{Point(0.0, 4.5, 4.5)};
  const std::vector duplicate_s{
      Point(0.0, 4.5, 4.5),
      Point(0.0, 5.5, 4.5),
  };
  const std::vector unsorted{
      Point(1.0, 4.5, 4.5),
      Point(0.0, 5.5, 4.5),
  };
  const std::vector nonfinite{
      Point(0.0, 4.5,
            std::numeric_limits<double>::infinity()),
  };

  for (const auto& request :
       std::vector<lpp::CorridorRequest>{
           Request(projection,
                   std::span<const lpp::PathControlPoint>{}),
           Request(projection, duplicate_s),
           Request(projection, unsorted),
           Request(projection, nonfinite),
           Request(projection, valid, Tightening(-0.1, 0.0, 0.0)),
           Request(projection, valid, Tightening(
                                          0.0, 0.0, 0.0,
                                          {{{2.0, 0.0}, 3.0}})),
           Request(projection, valid, Tightening(), 0U, 10U),
           Request(projection, valid, Tightening(), 10U, 0U),
       }) {
    const auto result = lpp::BuildConvexCorridor(request);
    EXPECT_EQ(result.status,
              lpp::CorridorStatus::kInvalidRequest);
    EXPECT_TRUE(result.cells.empty());
    EXPECT_EQ(
        result.fallback,
        lpp::CorridorFallback::kUseDiscreteValidatedPrimitives);
  }

  auto malformed_projection = projection;
  malformed_projection.hard_feasible_mask.pop_back();
  const auto malformed = lpp::BuildConvexCorridor(
      Request(malformed_projection, valid));
  EXPECT_EQ(malformed.status,
            lpp::CorridorStatus::kInvalidRequest);
  EXPECT_TRUE(malformed.cells.empty());
  EXPECT_EQ(malformed.reason_code, "invalid_projection");
}

TEST(ConvexCorridor,
     RejectsNoncanonicalHalfPlanesAndInsufficientClearance) {
  const auto projection = OpenProjection();
  const std::vector centerline{Point(0.0, 4.5, 4.5)};
  const double negative_zero = -0.0;
  const auto noncanonical = lpp::BuildConvexCorridor(Request(
      projection, centerline,
      Tightening(0.0, 0.0, 0.0,
                 {{{1.0, negative_zero}, 8.0}})));
  EXPECT_EQ(noncanonical.status,
            lpp::CorridorStatus::kInvalidRequest);
  EXPECT_EQ(noncanonical.reason_code,
            "invalid_platform_half_plane");

  auto low_clearance = projection;
  low_clearance.esdf_clearance_m[
      Index(4U, 4U, projection.geometry.width)] = 0.4F;
  const auto insufficient = lpp::BuildConvexCorridor(Request(
      low_clearance, centerline, Tightening(0.4, 0.0, 0.0)));
  EXPECT_EQ(insufficient.status,
            lpp::CorridorStatus::kFallbackRequired);
  EXPECT_TRUE(insufficient.cells.empty());
  EXPECT_EQ(insufficient.reason_code,
            "insufficient_clearance");
}

TEST(ConvexCorridor, ProducesStableUniqueIdsAndRepeatableGeometry) {
  const auto projection = OpenProjection();
  const std::vector centerline{
      Point(0.0, 2.5, 4.5), Point(2.0, 4.5, 4.5),
      Point(4.0, 6.5, 4.5), Point(6.0, 8.5, 4.5),
  };

  const auto first =
      lpp::BuildConvexCorridor(Request(projection, centerline));
  const auto second =
      lpp::BuildConvexCorridor(Request(projection, centerline));

  ASSERT_EQ(first.status, lpp::CorridorStatus::kCertified);
  ExpectEquivalent(first, second);
  std::vector<std::string> ids;
  for (const auto& cell : first.cells) {
    EXPECT_FALSE(cell.stable_cell_id.empty());
    ids.push_back(cell.stable_cell_id);
  }
  std::ranges::sort(ids);
  EXPECT_EQ(std::ranges::adjacent_find(ids), ids.end());
}
