#include <benchmark/benchmark.h>

#include "../tests/integration/wheel/wheel_planner_test_fixture.hpp"
#include "lunar_path_planner/v3/wheel/wheel_planner.hpp"

namespace lunar::planning::v3 {
namespace {

static void BM_WheelDeclaredSuite(
    benchmark::State& state) {
  const auto fixture = test::MakeFixture();
  WheelPlanner planner;
  for (auto _ : state) {
    static_cast<void>(_);
    auto result =
        planner.Plan(fixture.request, fixture.terminal);
    benchmark::DoNotOptimize(result);
  }
}

BENCHMARK(BM_WheelDeclaredSuite)->UseRealTime();

}  // namespace
}  // namespace lunar::planning::v3

BENCHMARK_MAIN();
