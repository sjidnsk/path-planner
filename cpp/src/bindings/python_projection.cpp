#include "lunar_path_planner/v3/bindings/python_projection.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numbers>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace lunar::planning::v3 {
namespace {

constexpr std::size_t kSplineSamplesPerSpan = 64U;
constexpr std::size_t kMaximumGeometrySamples = 262144U;
constexpr std::size_t kMaximumProjectionCells = 1000000U;
constexpr double kContinuityToleranceM = 1.0e-8;
constexpr double kYawContinuityToleranceRad = 1.0e-8;
constexpr double kRasterStepFraction = 0.25;

[[nodiscard]] Error Invalid(std::string field_path,
                            std::string message) {
  return {
      .code = ErrorCode::kInvalidArgument,
      .field_path = std::move(field_path),
      .message = std::move(message),
  };
}

[[nodiscard]] Error Resource(std::string field_path,
                             std::string message) {
  return {
      .code = ErrorCode::kResourceLimit,
      .field_path = std::move(field_path),
      .message = std::move(message),
  };
}

[[nodiscard]] bool FinitePose(const PoseXyzYaw& pose) noexcept {
  return std::isfinite(pose.position_m.x) &&
         std::isfinite(pose.position_m.y) &&
         std::isfinite(pose.position_m.z) &&
         std::isfinite(pose.yaw_rad);
}

[[nodiscard]] bool SamePosition(const PoseXyzYaw& lhs,
                                const PoseXyzYaw& rhs) noexcept {
  return std::abs(lhs.position_m.x - rhs.position_m.x) <=
             kContinuityToleranceM &&
         std::abs(lhs.position_m.y - rhs.position_m.y) <=
             kContinuityToleranceM &&
         std::abs(lhs.position_m.z - rhs.position_m.z) <=
             kContinuityToleranceM;
}

[[nodiscard]] double NormalizeYaw(double yaw_rad) noexcept {
  double normalized =
      std::remainder(yaw_rad, 2.0 * std::numbers::pi);
  if (normalized >= std::numbers::pi) {
    normalized -= 2.0 * std::numbers::pi;
  }
  if (normalized < -std::numbers::pi) {
    normalized += 2.0 * std::numbers::pi;
  }
  return normalized == 0.0 ? 0.0 : normalized;
}

[[nodiscard]] Result<GridCell> WorldToCell(
    const Vec3& point,
    const GridGeometry& geometry) {
  if (!std::isfinite(point.x) || !std::isfinite(point.y)) {
    return Invalid("platform_reference.geometric_path",
                   "NONFINITE_DIAGNOSTIC_PATH_POINT");
  }
  const double cell_x =
      std::floor((point.x - geometry.origin_m.x) /
                 geometry.resolution_m);
  const double cell_y =
      std::floor((point.y - geometry.origin_m.y) /
                 geometry.resolution_m);
  if (!std::isfinite(cell_x) || !std::isfinite(cell_y) ||
      cell_x < 0.0 || cell_y < 0.0 ||
      cell_x >= static_cast<double>(geometry.width) ||
      cell_y >= static_cast<double>(geometry.height)) {
    return Invalid("grid_geometry",
                   "PLATFORM_REFERENCE_OUTSIDE_DIAGNOSTIC_GRID");
  }
  return GridCell{
      .x = static_cast<std::size_t>(cell_x),
      .y = static_cast<std::size_t>(cell_y),
  };
}

[[nodiscard]] Result<bool> AppendCell(
    const Vec3& point,
    const GridGeometry& geometry,
    std::vector<GridCell>& cells) {
  const auto converted = WorldToCell(point, geometry);
  if (!IsOk(converted)) {
    return std::get<Error>(converted);
  }
  const GridCell cell = std::get<GridCell>(converted);
  if (cells.empty() || cells.back() != cell) {
    if (cells.size() >= kMaximumProjectionCells) {
      return Resource("path_cells",
                      "DIAGNOSTIC_PROJECTION_CELL_LIMIT");
    }
    cells.push_back(cell);
  }
  return true;
}

[[nodiscard]] Result<bool> AppendLineCells(
    const Vec3& start,
    const Vec3& finish,
    const GridGeometry& geometry,
    std::vector<GridCell>& cells) {
  const double distance_m =
      std::hypot(finish.x - start.x, finish.y - start.y);
  if (!std::isfinite(distance_m)) {
    return Invalid("platform_reference.geometric_path",
                   "NONFINITE_DIAGNOSTIC_PATH_LENGTH");
  }
  if (distance_m <= kContinuityToleranceM) {
    return AppendCell(finish, geometry, cells);
  }
  const double maximum_step_m =
      geometry.resolution_m * kRasterStepFraction;
  const double raw_steps = std::ceil(distance_m / maximum_step_m);
  if (!std::isfinite(raw_steps) ||
      raw_steps >
          static_cast<double>(kMaximumProjectionCells)) {
    return Resource("path_cells",
                    "DIAGNOSTIC_PROJECTION_SAMPLE_LIMIT");
  }
  const std::size_t steps =
      std::max<std::size_t>(
          1U, static_cast<std::size_t>(raw_steps));
  for (std::size_t index = 0U; index <= steps; ++index) {
    const double alpha =
        static_cast<double>(index) /
        static_cast<double>(steps);
    const Vec3 point{
        start.x + alpha * (finish.x - start.x),
        start.y + alpha * (finish.y - start.y),
        start.z + alpha * (finish.z - start.z),
    };
    const auto appended = AppendCell(point, geometry, cells);
    if (!IsOk(appended)) {
      return std::get<Error>(appended);
    }
  }
  return true;
}

[[nodiscard]] Result<PoseXyzYaw> EvaluateSpline(
    const ClampedCubicBSplinePath& spline,
    double parameter) {
  constexpr std::size_t kDegree = 3U;
  if (spline.control_points.size() < 4U ||
      spline.knots.size() !=
          spline.control_points.size() + kDegree + 1U ||
      !std::isfinite(parameter)) {
    return Invalid("platform_reference.geometric_path",
                   "INVALID_DIAGNOSTIC_SPLINE");
  }
  for (const auto& pose : spline.control_points) {
    if (!FinitePose(pose)) {
      return Invalid("platform_reference.geometric_path",
                     "INVALID_DIAGNOSTIC_SPLINE");
    }
  }
  for (std::size_t index = 0U; index < spline.knots.size();
       ++index) {
    if (!std::isfinite(spline.knots[index]) ||
        (index > 0U &&
         spline.knots[index] < spline.knots[index - 1U])) {
      return Invalid("platform_reference.geometric_path",
                     "INVALID_DIAGNOSTIC_SPLINE");
    }
  }

  const std::size_t control_count = spline.control_points.size();
  const double domain_start = spline.knots[kDegree];
  const double domain_finish = spline.knots[control_count];
  if (!(domain_finish > domain_start) ||
      parameter < domain_start || parameter > domain_finish) {
    return Invalid("platform_reference.geometric_path",
                   "INVALID_DIAGNOSTIC_SPLINE_DOMAIN");
  }

  std::size_t span = control_count - 1U;
  if (parameter < domain_finish) {
    for (std::size_t index = kDegree;
         index < control_count; ++index) {
      if (parameter >= spline.knots[index] &&
          parameter < spline.knots[index + 1U]) {
        span = index;
        break;
      }
    }
  }

  std::array<PoseXyzYaw, 4U> work{};
  for (std::size_t index = 0U; index <= kDegree; ++index) {
    work[index] =
        spline.control_points[span - kDegree + index];
  }
  for (std::size_t level = 1U; level <= kDegree; ++level) {
    for (std::size_t index = kDegree; index >= level; --index) {
      const std::size_t knot_index =
          span - kDegree + index;
      const double denominator =
          spline.knots[knot_index + kDegree - level + 1U] -
          spline.knots[knot_index];
      const double alpha =
          denominator > 0.0
              ? (parameter - spline.knots[knot_index]) /
                    denominator
              : 0.0;
      const auto blend = [alpha](double lhs, double rhs) {
        return (1.0 - alpha) * lhs + alpha * rhs;
      };
      work[index] = {
          .position_m =
              {
                  blend(work[index - 1U].position_m.x,
                        work[index].position_m.x),
                  blend(work[index - 1U].position_m.y,
                        work[index].position_m.y),
                  blend(work[index - 1U].position_m.z,
                        work[index].position_m.z),
              },
          .yaw_rad =
              blend(work[index - 1U].yaw_rad,
                    work[index].yaw_rad),
      };
      if (index == level) {
        break;
      }
    }
  }
  if (!FinitePose(work[kDegree])) {
    return Invalid("platform_reference.geometric_path",
                   "NONFINITE_DIAGNOSTIC_SPLINE_SAMPLE");
  }
  return work[kDegree];
}

[[nodiscard]] Result<std::vector<PoseXyzYaw>> SampleSpline(
    const ClampedCubicBSplinePath& spline) {
  if (spline.control_points.size() < 4U) {
    return Invalid("platform_reference.geometric_path",
                   "INVALID_DIAGNOSTIC_SPLINE");
  }
  if (spline.control_points.size() - 3U >
          kMaximumGeometrySamples / kSplineSamplesPerSpan) {
    return Resource("platform_reference.geometric_path",
                    "DIAGNOSTIC_SPLINE_SAMPLE_LIMIT");
  }
  const std::size_t sample_count =
      (spline.control_points.size() - 3U) *
          kSplineSamplesPerSpan +
      1U;
  const double domain_start =
      spline.knots.size() > 3U ? spline.knots[3U] : 0.0;
  const double domain_finish =
      spline.knots.size() > spline.control_points.size()
          ? spline.knots[spline.control_points.size()]
          : 0.0;
  if (!std::isfinite(domain_start) ||
      !std::isfinite(domain_finish) ||
      !(domain_finish > domain_start)) {
    return Invalid("platform_reference.geometric_path",
                   "INVALID_DIAGNOSTIC_SPLINE_DOMAIN");
  }

  std::vector<PoseXyzYaw> samples;
  samples.reserve(sample_count);
  for (std::size_t index = 0U; index < sample_count; ++index) {
    const double alpha =
        static_cast<double>(index) /
        static_cast<double>(sample_count - 1U);
    const double parameter =
        domain_start + alpha * (domain_finish - domain_start);
    const auto pose = EvaluateSpline(spline, parameter);
    if (!IsOk(pose)) {
      return std::get<Error>(pose);
    }
    samples.push_back(std::get<PoseXyzYaw>(pose));
  }
  return samples;
}

[[nodiscard]] Result<std::vector<PoseXyzYaw>>
SamplePrimitiveChain(const ValidatedPrimitiveChain& chain) {
  if (chain.primitives.empty()) {
    return Invalid("platform_reference.geometric_path",
                   "EMPTY_DIAGNOSTIC_PRIMITIVE_CHAIN");
  }
  if (chain.primitives.size() >
      (kMaximumGeometrySamples - 1U) / 2U) {
    return Resource("platform_reference.geometric_path",
                    "DIAGNOSTIC_PRIMITIVE_SAMPLE_LIMIT");
  }
  std::vector<PoseXyzYaw> samples;
  samples.reserve(chain.primitives.size() * 2U);
  for (std::size_t index = 0U;
       index < chain.primitives.size(); ++index) {
    const auto& primitive = chain.primitives[index];
    if (!FinitePose(primitive.start_pose) ||
        !FinitePose(primitive.end_pose)) {
      return Invalid("platform_reference.geometric_path",
                     "INVALID_DIAGNOSTIC_PRIMITIVE");
    }
    if (index > 0U) {
      const auto& previous = chain.primitives[index - 1U];
      if (!SamePosition(previous.end_pose,
                        primitive.start_pose) ||
          std::abs(previous.end_pose.yaw_rad -
                   primitive.start_pose.yaw_rad) >
              kYawContinuityToleranceRad) {
        return Invalid("platform_reference.geometric_path",
                       "DISCONTINUOUS_DIAGNOSTIC_PRIMITIVE_CHAIN");
      }
    }
    if (samples.empty()) {
      samples.push_back(primitive.start_pose);
    }
    samples.push_back(primitive.end_pose);
  }
  return samples;
}

[[nodiscard]] Result<std::vector<PoseXyzYaw>>
SampleGeometricPath(const GeometricPath& path) {
  return std::visit(
      [](const auto& concrete)
          -> Result<std::vector<PoseXyzYaw>> {
        using Path = std::decay_t<decltype(concrete)>;
        if constexpr (
            std::is_same_v<Path, ClampedCubicBSplinePath>) {
          return SampleSpline(concrete);
        } else {
          return SamplePrimitiveChain(concrete);
        }
      },
      path);
}

[[nodiscard]] Result<bool> AppendGeometrySamples(
    const std::vector<PoseXyzYaw>& samples,
    const GridGeometry& geometry,
    PpoPathProjection& projection,
    std::optional<PoseXyzYaw>& previous_pose) {
  if (samples.empty()) {
    return Invalid("platform_reference.geometric_path",
                   "EMPTY_DIAGNOSTIC_PATH");
  }
  if (previous_pose.has_value()) {
    if (!SamePosition(*previous_pose, samples.front()) ||
        std::abs(previous_pose->yaw_rad -
                 samples.front().yaw_rad) >
            kYawContinuityToleranceRad) {
      return Invalid("platform_reference.geometric_path",
                     "DISCONTINUOUS_DIAGNOSTIC_PATH");
    }
  } else {
    const auto appended = AppendCell(
        samples.front().position_m, geometry,
        projection.path_cells);
    if (!IsOk(appended)) {
      return std::get<Error>(appended);
    }
  }
  for (std::size_t index = 1U; index < samples.size(); ++index) {
    const auto& left = samples[index - 1U];
    const auto& right = samples[index];
    const double distance_m =
        std::hypot(right.position_m.x - left.position_m.x,
                   right.position_m.y - left.position_m.y);
    if (!std::isfinite(distance_m) ||
        projection.path_length_m >
            std::numeric_limits<double>::max() - distance_m) {
      return Invalid("platform_reference.geometric_path",
                     "NONFINITE_DIAGNOSTIC_PATH_LENGTH");
    }
    projection.path_length_m += distance_m;
    const auto appended = AppendLineCells(
        left.position_m, right.position_m, geometry,
        projection.path_cells);
    if (!IsOk(appended)) {
      return std::get<Error>(appended);
    }
  }
  previous_pose = samples.back();
  projection.final_yaw_rad =
      NormalizeYaw(samples.back().yaw_rad);
  return true;
}

[[nodiscard]] Result<PoseXyzYaw> SpinEndpoint(
    const SpinSegment& spin,
    bool finish) {
  if (!std::isfinite(spin.fixed_position_m.x) ||
      !std::isfinite(spin.fixed_position_m.y) ||
      !std::isfinite(spin.fixed_position_m.z) ||
      spin.unwrapped_yaw_rad.segments.empty()) {
    return Invalid("platform_reference.segments",
                   "INVALID_DIAGNOSTIC_SPIN_SEGMENT");
  }
  const auto& polynomial =
      finish ? spin.unwrapped_yaw_rad.segments.back()
             : spin.unwrapped_yaw_rad.segments.front();
  if (polynomial.end_offset.value <
      polynomial.start_offset.value) {
    return Invalid("platform_reference.segments",
                   "INVALID_DIAGNOSTIC_SPIN_INTERVAL");
  }
  for (const double coefficient : polynomial.coefficients) {
    if (!std::isfinite(coefficient)) {
      return Invalid("platform_reference.segments",
                     "INVALID_DIAGNOSTIC_SPIN_POLYNOMIAL");
    }
  }
  const double seconds =
      finish
          ? std::chrono::duration<double>(
                polynomial.end_offset.value -
                polynomial.start_offset.value)
                .count()
          : 0.0;
  const auto& c = polynomial.coefficients;
  const double yaw =
      ((c[3] * seconds + c[2]) * seconds + c[1]) *
          seconds +
      c[0];
  if (!std::isfinite(yaw)) {
    return Invalid("platform_reference.segments",
                   "NONFINITE_DIAGNOSTIC_SPIN_YAW");
  }
  return PoseXyzYaw{
      .position_m = spin.fixed_position_m,
      .yaw_rad = yaw,
  };
}

[[nodiscard]] Result<PpoPathProjection> ProjectWheel(
    const WheeledReference& reference,
    const GridGeometry& geometry,
    std::string bundle_id) {
  if (reference.segments.empty()) {
    return Invalid("platform_reference.segments",
                   "EMPTY_WHEEL_REFERENCE");
  }
  PpoPathProjection projection{
      .source_bundle_id = std::move(bundle_id),
  };
  std::optional<PoseXyzYaw> previous_pose;
  for (const auto& segment : reference.segments) {
    const auto appended = std::visit(
        [&](const auto& concrete) -> Result<bool> {
          using Segment = std::decay_t<decltype(concrete)>;
          if constexpr (std::is_same_v<Segment, DriveSegment>) {
            const auto samples =
                SampleGeometricPath(concrete.geometric_path);
            if (!IsOk(samples)) {
              return std::get<Error>(samples);
            }
            return AppendGeometrySamples(
                std::get<std::vector<PoseXyzYaw>>(samples),
                geometry, projection, previous_pose);
          } else {
            const auto start = SpinEndpoint(concrete, false);
            const auto finish = SpinEndpoint(concrete, true);
            if (!IsOk(start)) {
              return std::get<Error>(start);
            }
            if (!IsOk(finish)) {
              return std::get<Error>(finish);
            }
            const std::vector<PoseXyzYaw> samples{
                std::get<PoseXyzYaw>(start),
                std::get<PoseXyzYaw>(finish),
            };
            return AppendGeometrySamples(
                samples, geometry, projection, previous_pose);
          }
        },
        segment);
    if (!IsOk(appended)) {
      return std::get<Error>(appended);
    }
  }
  if (projection.path_cells.empty() ||
      !std::isfinite(projection.path_length_m) ||
      !std::isfinite(projection.final_yaw_rad)) {
    return Invalid("platform_reference",
                   "INVALID_WHEEL_DIAGNOSTIC_PROJECTION");
  }
  projection.executable_reference_present = true;
  return projection;
}

[[nodiscard]] Result<PpoPathProjection> ProjectLegged(
    const LeggedBodyReference& reference,
    const GridGeometry& geometry,
    std::string bundle_id) {
  const auto samples =
      SampleGeometricPath(reference.geometric_path);
  if (!IsOk(samples)) {
    return std::get<Error>(samples);
  }
  PpoPathProjection projection{
      .source_bundle_id = std::move(bundle_id),
  };
  std::optional<PoseXyzYaw> previous_pose;
  const auto appended = AppendGeometrySamples(
      std::get<std::vector<PoseXyzYaw>>(samples), geometry,
      projection, previous_pose);
  if (!IsOk(appended)) {
    return std::get<Error>(appended);
  }
  if (projection.path_cells.empty() ||
      !std::isfinite(projection.path_length_m) ||
      !std::isfinite(projection.final_yaw_rad)) {
    return Invalid("platform_reference",
                   "INVALID_LEGGED_DIAGNOSTIC_PROJECTION");
  }
  projection.executable_reference_present = true;
  return projection;
}

[[nodiscard]] PpoPathProjection NonExecutableProjection(
    std::string bundle_id) {
  return {
      .path_cells = {},
      .path_length_m = 0.0,
      .final_yaw_rad = 0.0,
      .executable_reference_present = false,
      .source_bundle_id = std::move(bundle_id),
  };
}

[[nodiscard]] std::optional<Error> ValidateGrid(
    const GridGeometry& geometry) {
  if (geometry.width == 0U || geometry.height == 0U ||
      geometry.CellCount() == 0U) {
    return Invalid("grid_geometry",
                   "INVALID_DIAGNOSTIC_GRID_EXTENT");
  }
  if (!std::isfinite(geometry.resolution_m) ||
      geometry.resolution_m <= 0.0) {
    return Invalid("grid_geometry.resolution_m",
                   "INVALID_DIAGNOSTIC_GRID_RESOLUTION");
  }
  if (!std::isfinite(geometry.origin_m.x) ||
      !std::isfinite(geometry.origin_m.y)) {
    return Invalid("grid_geometry.origin_m",
                   "INVALID_DIAGNOSTIC_GRID_ORIGIN");
  }
  if (geometry.frame_id.empty()) {
    return Invalid("grid_geometry.frame_id",
                   "DIAGNOSTIC_GRID_FRAME_REQUIRED");
  }
  return std::nullopt;
}

}  // namespace

Result<PpoPathProjection> ProjectForPpoDiagnostics(
    const PlanningResponse& response,
    const GridGeometry& grid_geometry) {
  if (const auto grid_error = ValidateGrid(grid_geometry);
      grid_error.has_value()) {
    return *grid_error;
  }

  if (response.execution_directive !=
      ExecutionDirective::kActivateNewBundle) {
    return NonExecutableProjection(
        response.active_bundle_ref.has_value()
            ? response.active_bundle_ref->id
            : std::string{});
  }
  if (!response.new_reference_bundle.has_value()) {
    return Invalid("new_reference_bundle",
                   "DIAGNOSTIC_ACTIVATION_BUNDLE_REQUIRED");
  }

  const ReferenceBundle& bundle =
      *response.new_reference_bundle;
  const bool type_matches =
      (bundle.platform_type == PlatformType::kWheeled &&
       std::holds_alternative<WheeledReference>(
           bundle.platform_reference)) ||
      (bundle.platform_type == PlatformType::kLegged &&
       std::holds_alternative<LeggedBodyReference>(
           bundle.platform_reference)) ||
      (bundle.platform_type == PlatformType::kHopper &&
       std::holds_alternative<HopperReference>(
           bundle.platform_reference));
  if (!type_matches) {
    return Invalid("new_reference_bundle.platform_reference",
                   "DIAGNOSTIC_PLATFORM_VARIANT_MISMATCH");
  }

  return std::visit(
      [&](const auto& reference) -> Result<PpoPathProjection> {
        using Reference = std::decay_t<decltype(reference)>;
        if constexpr (
            std::is_same_v<Reference, WheeledReference>) {
          return ProjectWheel(
              reference, grid_geometry, bundle.bundle_id);
        } else if constexpr (
            std::is_same_v<Reference, LeggedBodyReference>) {
          return ProjectLegged(
              reference, grid_geometry, bundle.bundle_id);
        } else {
          // A ballistic jump has no continuous ground path. The
          // landing region remains in the authoritative bundle and is
          // deliberately not collapsed to a fake executable cell path.
          return NonExecutableProjection(bundle.bundle_id);
        }
      },
      bundle.platform_reference);
}

}  // namespace lunar::planning::v3
