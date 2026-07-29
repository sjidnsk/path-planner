# Planner V3：G1 派生多尺度重复规划延迟实验

## 结论

本实验以“双门槛实验”G1 验证集中的三个场景为冻结参考，构造带有明确障碍物的
十米级、百米级和千米级环境，并对轮式、足式和飞跃式平台执行 810 次正式规划调用。
全部调用均返回与场景约定一致的结果，三个平台合并全部尺度与场景后的端到端
P95 均小于 1000 ms：

| 平台 | 正式调用数 | P50 / ms | P95 / ms | 最大值 / ms | 正确结果率 |
|---|---:|---:|---:|---:|---:|
| 轮式 | 270 | 14.9742 | 31.3721 | 34.3592 | 100% |
| 足式 | 270 | 17.2362 | 37.5169 | 47.9466 | 100% |
| 飞跃式 | 270 | 23.5281 | 69.4211 | 79.5194 | 100% |

因此，在本实验的 G1 派生环境—平台联合约束地图、场景矩阵和测试机器上，
Planner V3 满足“单次调用 P95 小于 1 秒”的实验指标。最慢单次调用为
79.5194 ms。该结论是当前基准工作负载下的性能证据，不外推为任意地图、
任意硬件或任意障碍密度下的最坏情况保证。

## G1 场景来源与障碍物合同

场景只引用 G1 的 `validation3` 验证集，不使用 `Test-Q24` 或 `Unseen-24`
正式评测队列。每个尺度使用相同的三个冻结参考：

| 实验场景 | G1 冻结参考 | `scenario_hash` | 原始障碍占比 |
|---|---|---|---:|
| `G1_LOW_KNOWN` | `validation/scenario-0027/standard-proxy/v1` | `564835143c269d1ef4fb8d01cb7517cf4efb674bb1554c36467a1d59eb39f16c` | 0.6683% |
| `G1_MEDIUM_KNOWN` | `validation/scenario-0007/standard-proxy/v1` | `6f754169bb3c4837978e3e852258493f4b1c971e8a59148a9a8269fb306a95bc` | 1.3489% |
| `G1_HIGH_FRONTIER` | `validation/scenario-0116/standard-proxy/v1` | `49e4254dc8e5602be67cc2e9eb68af65093ec049b463e3581edfc9b5f352b554` | 3.8147% |

实验的场景派生合同为：

- `derivation=g1_validation_reference_scaled/v1`；
- `synthetic_source_kind=synthetic_terrain_obstacle_proxy/v1`；
- `physical_obstacle_cells_written=false`。

障碍物由 G1 冻结种子确定性生成，表现为 5～9 栅格单元的岩石代理簇，并在三个
物理尺度上保持对应参考的障碍占比。轮式与足式仅保留约 ±1 栅格宽的认证通道，
因此规划器必须绕开代理障碍，不存在大面积人工空走廊。

这些地图是“G1 派生的缩放代理场景”，不是原始 128 m × 128 m G1 地图的直接复制，
也不构成 G1 覆盖能力结论。图中的障碍只能称为合成地形障碍代理，不能称为
`physical_obstacle_cells`。

## 实验矩阵

实验采用 `3 个尺度 × 3 个 G1 派生场景 × 3 个平台 = 27` 个组合。
每个组合执行 1 次冷启动、3 次预热和 30 次正式计时：

- 冷启动调用：27 次，单独记录，不进入 P95；
- 预热调用：81 次，不进入统计；
- 正式调用：810 次，全部进入 P50、P95、最大值和正确结果率统计。

地图与参考任务如下：

| 尺度 | 物理范围 | 任务栅格 | 轮式/足式名义路程 | 飞跃式单次跃迁 |
|---|---|---|---:|---:|
| 十米级 | 12 m × 12 m | 24 × 24，0.5 m/格 | 约 8 m | 约 4 m |
| 百米级 | 120 m × 80 m | 60 × 40，2 m/格 | 约 80 m | 约 8 m |
| 千米级 | 1200 m × 200 m | 240 × 40，5 m/格 | 约 500 m | 约 10 m |

轮式和足式共享同一环境 `Known/Hard` 掩码，但分别应用平台能力约束并输出带
时序的机体/质心几何路径与速度参考。`G1_HIGH_FRONTIER` 中，两种地面平台只规划
到安全前沿并返回 `SAFE_FRONTIER`。

飞跃式平台输出纯弹道单次跃迁参考：确定的下一着陆状态、包含 yaw 的姿态目标、
已知区内的着陆范围以及飞跃边界条件。其起终点由 G1 派生障碍场中的高净空位置
确定；即使在高障碍前沿场景，着陆范围也不得进入未知区。

## 计时边界

每次正式样本的计时从调用 `PlannerV3::Plan(request)` 之前立即开始，到
`JsonCodec::EncodePlanningResponse(response)` 完成之后结束。因此统计覆盖规划
计算和响应 JSON 序列化，不包含场景构造、预热、文件写入和绘图。

`1000 ms` 仅在全部调用自然结束后作为实验统计指标使用，不是 deadline、
timeout、搜索预算、提前取消条件或结果选择条件。即使某次调用超过 1 秒，
运行器也会完成该调用及其余计划调用，再计算最终 P95。

## 分尺度结果

下表给出每个“尺度 × 平台”汇总 90 次正式调用后的统计：

| 尺度 | 平台 | P50 / ms | P95 / ms | 最大值 / ms |
|---|---|---:|---:|---:|
| 十米级 | 轮式 | 10.0356 | 10.9003 | 11.2136 |
| 十米级 | 足式 | 11.8441 | 13.0174 | 15.4109 |
| 十米级 | 飞跃式 | 15.6766 | 22.2859 | 23.4874 |
| 百米级 | 轮式 | 14.9742 | 15.8057 | 16.8191 |
| 百米级 | 足式 | 17.2362 | 22.6736 | 28.5992 |
| 百米级 | 飞跃式 | 23.5281 | 31.4974 | 32.8737 |
| 千米级 | 轮式 | 23.1588 | 32.7251 | 34.3592 |
| 千米级 | 足式 | 36.3791 | 39.7840 | 47.9466 |
| 千米级 | 飞跃式 | 59.7094 | 74.9293 | 79.5194 |

27 个组合均为 30/30 正确，总体正确结果率为 100%。最慢组合是
“千米级 × `G1_MEDIUM_KNOWN` × 飞跃式”，其 P50 为 61.9071 ms、
P95 为 78.8338 ms、最大值为 79.5194 ms。

## 测试环境

- 操作系统：Microsoft Windows 11 家庭中文版，版本 10.0.26200；
- 处理器：AMD Ryzen 9 9955HX，16 核 32 线程；
- 构建：CMake Release；
- 编译器：MSVC 19.44（`compiler_version=1944`）；
- 运行器：`lpp_v3_multiscale_experiment.exe`。

## 产物与复现

正式实验产物位于：

- `D:/xunce/out/planner-v3-multiscale/manifest.json`：构建信息、G1 派生来源、
  计时边界和 27 组合矩阵；
- `D:/xunce/out/planner-v3-multiscale/latency-samples.jsonl`：837 条原始样本，
  其中 27 条冷启动、810 条正式样本；
- `D:/xunce/out/planner-v3-multiscale/summary.json`：场景、尺度和平台统计；
- `D:/xunce/out/planner-v3-multiscale/responses.jsonl`：27 个代表性规划响应；
- `D:/xunce/out/planner-v3-multiscale/figures/scenario-overview.png`：
  3 × 3 有障碍仿真案例场景图；
- `D:/xunce/out/planner-v3-multiscale/figures/latency-results.png`：
  P50、P95、最大值、正确结果率及 1000 ms 指标线。

运行正式实验：

```powershell
$env:PATH = "D:\xunce\build\path-planner-v3\windows-msvc-release\vcpkg_installed\x64-windows\bin;$env:PATH"
D:\xunce\build\path-planner-v3\windows-msvc-release\lpp_v3_multiscale_experiment.exe `
  --output-root D:\xunce\out\planner-v3-multiscale
```

根据实验产物重新绘图：

```powershell
python scripts/plot_planner_v3_multiscale.py `
  --input-root D:/xunce/out/planner-v3-multiscale `
  --output-dir D:/xunce/out/planner-v3-multiscale/figures
```
