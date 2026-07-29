#pragma once

#include <compare>
#include <cstddef>
#include <string>
#include <vector>

#include "lunar_path_planner/v3/contracts/planning_response.hpp"
#include "lunar_path_planner/v3/map/immutable_snapshot.hpp"

namespace lunar::planning::v3 {

struct GridCell final {
  std::size_t x{};
  std::size_t y{};

  auto operator<=>(const GridCell&) const = default;
};

struct PpoPathProjection final {
  std::vector<GridCell> path_cells;
  double path_length_m{};
  double final_yaw_rad{};
  bool executable_reference_present{false};
  std::string source_bundle_id;

  bool operator==(const PpoPathProjection&) const = default;
};

// This is a diagnostic projection only. It never changes a planning
// response, never resolves an external bundle, and never turns a route
// skeleton into an executable path.
[[nodiscard]] Result<PpoPathProjection> ProjectForPpoDiagnostics(
    const PlanningResponse& response,
    const GridGeometry& grid_geometry);

}  // namespace lunar::planning::v3
