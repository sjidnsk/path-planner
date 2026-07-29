# 路径规划 v3 C++ 构建与测试

## 构建边界

路径规划 v3 是独立、opt-in 的 C++20 实现，源码位于 `path-planner/cpp`。
它不会替换既有 Python 默认 A*，也不连接 executor。构建产物默认写入
`D:/xunce/build/path-planner-v3`，不写入源码目录。

实验中的“单次调用 P95 小于 1 秒”只由 benchmark runner 统计，不是规划器
运行时 deadline，也不会触发提前返回。

## 当前 Windows 工具链

当前工作站已经验证以下组合：

| 工具 | 版本或固定值 | 当前路径 |
|---|---:|---|
| MSVC | 19.44.35228 | `D:/APP/VSBuildTools` |
| CMake | 4.4.0，项目最低要求 3.28 | `D:/APP/CMake/bin/cmake.exe` |
| Ninja | 1.13.2 | `D:/APP/Ninja/ninja.exe` |
| PowerShell | 7.5.4 | `D:/APP/PowerShell/7.5.4-portable/pwsh.exe` |
| vcpkg | baseline `56bb2411609227288b70117ead2c47585ba07713` | `D:/CodexDownloads/vcpkg` |

依赖版本由 `vcpkg.json` 和其中的 `overrides` 固定。默认构建包含 Eigen、
nlohmann_json、double-conversion、picosha2、GoogleTest 和 Google
Benchmark；OSQP 1.0.0 是可选 feature。

## Windows 配置

在 PowerShell 中从 `path-planner/cpp` 执行：

```powershell
$env:PATH = @(
  'D:\APP\CMake\bin'
  'D:\APP\Ninja'
  'D:\APP\PowerShell\7.5.4-portable'
  'D:\CodexDownloads\tools\7zip-portable\full'
  $env:PATH
) -join ';'

$configure = @'
call "D:\APP\VSBuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul &&
set "VCPKG_ROOT=D:\CodexDownloads\vcpkg" &&
set "VCPKG_DOWNLOADS=D:\CodexDownloads\vcpkg-downloads" &&
set "VCPKG_DEFAULT_BINARY_CACHE=D:\CodexDownloads\vcpkg-bincache" &&
set "VCPKG_FORCE_SYSTEM_BINARIES=1" &&
cmake --preset windows-msvc-debug
'@

cmd.exe /d /c $configure
```

必须在 `vcvars64.bat` 之后重新设置 `VCPKG_ROOT`。当前 Visual Studio
环境脚本可能覆盖该变量；顺序反过来会让 CMake 使用错误的 vcpkg checkout。

若已有 build cache 曾绑定其他 vcpkg、编译器或 Ninja，可只清除这三个 CMake
cache 项后重新配置，不删除构建目录：

```powershell
$configure = @'
call "D:\APP\VSBuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul &&
set "VCPKG_ROOT=D:\CodexDownloads\vcpkg" &&
set "VCPKG_DOWNLOADS=D:\CodexDownloads\vcpkg-downloads" &&
set "VCPKG_DEFAULT_BINARY_CACHE=D:\CodexDownloads\vcpkg-bincache" &&
set "VCPKG_FORCE_SYSTEM_BINARIES=1" &&
cmake --preset windows-msvc-debug -U Z_VCPKG_ROOT_DIR -U CMAKE_MAKE_PROGRAM -U CMAKE_CXX_COMPILER
'@

cmd.exe /d /c $configure
```

## 构建与测试

默认 Debug：

```powershell
cmake --build --preset windows-msvc-debug --parallel 4
ctest --preset windows-msvc-debug --no-tests=error --output-on-failure
```

Release：

```powershell
cmake --preset windows-msvc-release
cmake --build --preset windows-msvc-release --parallel 4
ctest --preset windows-msvc-release --no-tests=error --output-on-failure
```

可选 OSQP：

```powershell
cmake --preset windows-msvc-debug-osqp
cmake --build --preset windows-msvc-debug-osqp --parallel 4
ctest --preset windows-msvc-debug-osqp `
  --no-tests=error --output-on-failure
```

OSQP 的 vcpkg port 固定为 1.0.0。其上游生成的 CMake package-version
文件会报告 `0.0.0`，因此版本固定由 manifest/override 完成，项目使用
`find_package(osqp CONFIG REQUIRED)`，不要把它改成不可靠的 `EXACT` 查找。

## 安装与 CMake 消费

安装到 D 盘：

```powershell
cmake --install D:/xunce/build/path-planner-v3/windows-msvc-release `
  --prefix D:/xunce/install/path-planner-v3/release
```

消费工程应使用同一份 vcpkg 固定依赖，并配置安装前缀：

```cmake
find_package(LunarPathPlannerV3 CONFIG REQUIRED)
target_link_libraries(my_planner PRIVATE LunarPathPlannerV3::api)
```

公开目标还包括 `contracts`、`common`、`wheel`、`legged` 和 `hopper`，
完整名称均使用 `LunarPathPlannerV3::` 前缀。

## 分层测试前缀

可以用 CTest 前缀运行局部门：

```powershell
ctest --test-dir D:/xunce/build/path-planner-v3/windows-msvc-debug `
  -R '^lpp_v3_codec\.' --no-tests=error --output-on-failure

ctest --test-dir D:/xunce/build/path-planner-v3/windows-msvc-debug `
  -R '^lpp_v3_api\.' --no-tests=error --output-on-failure
```

平台层接入后对应前缀为 `lpp_v3_wheel.`、`lpp_v3_legged.` 和
`lpp_v3_hopper.`。所有过滤式测试必须带 `--no-tests=error`，避免拼错正则时
出现“零测试也成功”的假阳性。

## 基准输出

Release benchmark 输出写入 D 盘。共享核心基准示例：

```powershell
& 'D:\xunce\build\path-planner-v3\windows-msvc-release\lpp_v3_shared_core_benchmark.exe' `
  --benchmark_repetitions=30 `
  --benchmark_report_aggregates_only=true `
  --benchmark_out_format=json `
  --benchmark_out=D:/xunce/out/path-planner-v3/shared-core.json
```

benchmark wrapper 可以读取 `steady_clock`；运行时规划器、能力配置和请求合同
均不得携带 clock deadline、剩余时长或 benchmark profile。
