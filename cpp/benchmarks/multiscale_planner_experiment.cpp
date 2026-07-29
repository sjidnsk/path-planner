#include "multiscale_experiment.hpp"

#include <filesystem>
#include <iostream>
#include <string>
#include <string_view>

namespace {

[[nodiscard]] int Run(const int argc, char** argv) {
  std::filesystem::path output_root;
  constexpr std::string_view prefix = "--output-root=";
  for (int index = 1; index < argc; ++index) {
    const std::string_view argument{argv[index]};
    if (argument == "--output-root" && index + 1 < argc) {
      output_root = argv[++index];
    } else if (argument.starts_with(prefix)) {
      output_root =
          std::string{argument.substr(prefix.size())};
    } else if (argument == "--help" || argument == "-h") {
      std::cout
          << "usage: lpp_v3_multiscale_experiment "
             "--output-root <D:/path>\n";
      return 0;
    } else {
      std::cerr << "unrecognized argument: " << argument << '\n';
      return 2;
    }
  }
  if (output_root.empty()) {
    std::cerr << "--output-root is required\n";
    return 2;
  }
  const auto result =
      lunar::planning::v3::RunMultiscaleExperiment(
          {}, output_root);
  if (!lunar::planning::v3::IsOk(result)) {
    const auto& error =
        std::get<lunar::planning::v3::Error>(result);
    std::cerr << error.field_path << ": " << error.message << '\n';
    return 1;
  }
  const auto& paths =
      std::get<lunar::planning::v3::ExperimentRunPaths>(result);
  std::cout << "manifest: " << paths.manifest_json.string() << '\n'
            << "summary: " << paths.summary_json.string() << '\n';
  return 0;
}

}  // namespace

int main(const int argc, char** argv) {
  return Run(argc, argv);
}
