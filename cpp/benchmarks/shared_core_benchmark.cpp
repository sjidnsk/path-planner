#include <benchmark/benchmark.h>

#include <cstdint>

namespace {

void SharedCoreHarness(benchmark::State& state) {
  std::uint64_t value = 0;
  for (auto iteration : state) {
    static_cast<void>(iteration);
    benchmark::DoNotOptimize(value);
    ++value;
  }
}

BENCHMARK(SharedCoreHarness);

}  // namespace
