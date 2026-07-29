#include <benchmark/benchmark.h>

#include "../tests/integration/legged/legged_test_fixture.hpp"

namespace lunar::planning::v3 {
namespace {

static void BM_LeggedDeclaredSuite(
    benchmark::State& state) {
  const auto fixture = legged_test::MakeFixture();
  const LeggedPlanner planner;
  for (auto _ : state) {
    static_cast<void>(_);
    auto result =
        planner.Plan(fixture.request, fixture.terminal);
    benchmark::DoNotOptimize(result);
  }
  state.counters["experimental_p95_target_ms"] = 1000.0;
  state.SetLabel(
      "latency_target_is_a_reported_metric_not_a_planner_deadline");
}

BENCHMARK(BM_LeggedDeclaredSuite)
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);

}  // namespace
}  // namespace lunar::planning::v3

BENCHMARK_MAIN();
