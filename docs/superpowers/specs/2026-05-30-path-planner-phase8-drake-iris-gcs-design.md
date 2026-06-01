# path-planner Phase 8 Drake IRIS/GCS 框架设计

日期：2026-05-30

## 1. 目标

Phase 8 把当前规划器推进到 Drake IRIS/GCS 框架原型阶段。目标不是立即替换
现有 A*、postprocess、tracking simulation 或 fixed-corridor optimizer，也不是
直接声明 Ackermann 可执行轨迹，而是在保持现有 fallback 链稳定的前提下，
建立 2D workspace safe-region、region graph 和可选 Drake backend 的设计边界。

Phase 8 输出应包括：

- pydrake 知识结构和本项目映射；
- IRIS/GCS region 和 graph 的内部模型设计；
- Drake backend optional dependency 策略；
- route JSON 候选报告契约；
- CLI、诊断和测试方案；
- 明确 fallback 链。
- 明确 rover 运动学/Ackermann 不属于 Phase 8 的硬求解目标，但必须作为后续
  feasibility layer 预留。

## 2. 当前基础

Phase 8 复用现有模块，不重写前置链路：

- `CostGrid` 是标准地图输入。
- `PlanningGrid` 和 platform-aware A* 负责基础可达性搜索。
- `postprocess` 负责 corridor、smoothing、curvature 和 trackable path。
- `tracking` 提供 low-speed pure-pursuit simulation baseline。
- `optimization` 提供 fixed-corridor continuous optimization baseline。

这些能力继续作为 Drake backend 的输入、对照组和 fallback。

## 3. IRIS 族选择

Drake 中与 IRIS 相关的 Python API 不止一种。Phase 8 必须先明确选择，避免把
workspace safe-region、configuration-space region 和 Ackermann 可执行轨迹混为一谈。

### 3.1 原始 `Iris`

`pydrake.geometry.optimization.Iris` 输入 convex obstacles、sample point 和 bounded
domain，输出一个 `HPolyhedron`。它适合当前项目的 2D 月面 workspace safe-region
生成：

- sample point：A* 或 smoothed path waypoint。
- domain：地图局部窗口、corridor box 或 path 周围 local box。
- obstacles：由 `inflated_passable_mask` 的 blocked cells 转换出的凸障碍物集合。
- output：可作为 GCS vertex 的 2D convex region。

### 3.2 `IrisNp`

`pydrake.geometry.optimization.IrisNp` 在 `MultibodyPlant` 的 configuration space
中寻找 collision-free region，需要 plant、context 和 SceneGraph/collision geometry。
它更适合机械臂、多体机器人或已经建立完整 C-space 碰撞模型的系统。

当前 `path-planner` 还没有 `(x, y, theta)` 状态空间、Ackermann 模型或
MultibodyPlant/SceneGraph pipeline，因此 Phase 8 不选 `IrisNp` 作为主路径。

### 3.3 `IrisNp2`、`IrisZo` 和 clique-cover 变体

`pydrake.planning` 中还提供 `IrisNp2`、`IrisZo` 和
`IrisInConfigurationSpaceFromCliqueCover` 等 configuration-space / sampled IRIS
变体。这些更适合对复杂 C-space 做采样覆盖和碰撞检查，工程前提是已有可信的
robot model 与 collision checker。

Phase 8 不使用这些变体。它们可以留到后续 rover motion model、姿态空间和
碰撞几何成熟后再评估。

### 3.4 Phase 8 选择

Phase 8 应优先选择 original 2D workspace `Iris`，路线为：

```text
2D workspace grid_box baseline
  -> original Iris over convex obstacle primitives
  -> region graph / GCS geometric path prototype
```

不选择：

```text
IrisNp / IrisNp2 / IrisZo as primary backend
```

原因是当前项目的数据结构、地图输入和诊断链路都是 2D grid / point-center safe
mask，最贴近原始 `Iris` 的 workspace region generation。

## 4. 非凸月表障碍物处理

原始 `Iris` 要求输入的每个 obstacle primitive 是 convex set，但月表障碍物整体
可以是非凸的。解决方式不是强迫月表障碍物整体凸化，而是把 unsafe space 表示
为多个凸障碍物的并集。

Phase 8 采用以下建模策略：

1. 使用现有 `inflated_passable_mask` 定义车辆中心点不可进入区域。
2. 将每个 blocked cell 转换为 2D axis-aligned convex box。
3. 将相邻 blocked cells 可选地合并为较大的 axis-aligned rectangles，减少
   obstacle 数量。
4. 不使用 connected component 的单个 convex hull 作为默认策略，因为它会过度
   保守，可能封死凹口、窄通道和真实可通行区域。
5. IRIS 只在 path waypoint 周围的 local domain 中运行，不在整张月面图上一次性
   运行。
6. IRIS 输出的 `HPolyhedron` 必须通过 grid sampling / cell intersection 校验；
   如果校验发现 region 穿越 inflated unsafe cells，则该 region invalid，并回退到
   `grid_box` region。

因此，Phase 8 的 `iris_region_report` 必须记录：

- obstacle primitive count；
- obstacle source，例如 `inflated_blocked_cells` 或 `merged_blocked_rectangles`；
- domain source；
- validation status；
- invalid / fallback reason。

## 5. Drake 设计映射

Drake 相关 API 只允许进入可选 backend 层。核心模型不直接依赖 `pydrake`。

推荐映射：

- `HPolyhedron`：表示 IRIS 或 box approximation 产生的 convex region。
- `Point`：表示 start、goal 或 sample point。
- `Iris` / `IrisOptions`：从 sample point 和 convex obstacle primitives 生成 local convex region。
- `GraphOfConvexSets`：底层 region graph 和 shortest path 研究接口。
- `GcsTrajectoryOptimization`：优先用于连续 geometric trajectory backend 原型。
- `MathematicalProgram` / `Solve`：只在 backend 内部使用。
- `PiecewisePolynomial` / `BsplineTrajectory`：只作为后端输出表示，route JSON 中必须采样后序列化。

`GraphOfConvexSets` 和 `GcsTrajectoryOptimization` 在 Drake 中属于 experimental
能力，因此本项目不能把它们作为默认必需路径。

## 6. Rover 运动学与 Ackermann 边界

月球探索车必须考虑运动学可行性，但 Phase 8 不应把所有 rover 都默认建模为
Ackermann 车辆。

原因：

- 月球车可能是六轮 rocker-bogie、多轮独立驱动、部分轮独立转向或 skid-steer
  近似平台。
- 严格 Ackermann 适用于 car-like steering，但不一定适用于真实月球车底盘。
- 当前 `yutu2` 配置中的 `min_turning_radius` 仍可能是 assumed/disabled 参数，不适合
  直接作为硬事实。
- 现有系统已有曲率和 turning-radius 后验检查，但还没有把 `(x, y, theta)`、
  方向连续性、倒车能力、原地转向能力或 skid 模型纳入搜索/优化状态。

Phase 8 的边界：

- IRIS/GCS 输出仍是 2D geometric safe-region / sampled path candidate。
- 不声明 GCS 输出为 Ackermann-feasible trajectory。
- 保留 `curvature_report`、`trackable_path` 和 tracking simulation 作为后验可执行性诊断。
- 设计上预留后续 `motion_feasibility_report`，但不在 Phase 8 中强制实现完整
  Ackermann optimizer。

建议后续 Phase 9 引入 rover motion feasibility layer：

- `motion_model`: `point`、`curvature_bounded`、`ackermann`、`skid_steer`、
  `differential` 或 `custom`。
- `min_turning_radius_m`
- `can_reverse`
- `can_turn_in_place`
- `max_heading_change_deg`
- `feasibility_status`
- `violation_indices`
- `fallback_reason`

## 7. 候选输出契约

Phase 8 不改变现有顶层语义：

- `trajectory_kind` 仍为 `geometric_path`。
- `reachable` 仍由基础规划链路决定。
- `postprocess`、`tracking_simulation_report`、`trajectory_optimization_report`
  保持兼容。

Phase 8.1 已实现 `region_graph_report` 的 Drake-free baseline。Phase 8.2 已实现
可选 `workspace_iris` backend 的 `iris_region_report` 原型。`gcs_trajectory_report`
仍作为候选设计，不要求本轮实现：

- `iris_region_report`
  - `backend`
  - `status`
  - `region_count`
  - `seed_source`
  - `domain_source`
  - `obstacle_source`
  - `obstacle_count`
  - `validation_status`
  - `failure_reason`
  - `fallback_used`
- `region_graph_report`
  - `status`
  - `vertex_count`
  - `edge_count`
  - `region_source`
  - `graphviz_available`
  - `failure_reason`
- `gcs_trajectory_report`
  - `backend`
  - `status`
  - `solver_status`
  - `region_source`
  - `sampled_path`
  - `path_cost`
  - `motion_feasibility_status`
  - `warnings`
  - `fallback_status`

这些报告必须是可选字段，缺失时不影响旧消费者读取 route JSON。

`motion_feasibility_status` 在 Phase 8 中只能是 `not_evaluated`、`diagnostic_only`
或已有曲率检查派生结果，不能宣称完整 Ackermann 可行。

## 8. Phase 8.1 Region Graph Baseline

Phase 8.1 落地一个不依赖 Drake 的 region graph baseline，用于稳定内部模型和
JSON 报告契约。

新增 Drake-free 模型：

- `ObstaclePrimitive`
- `ConvexRegion`
- `RegionEdge`
- `RegionGraph`
- `RegionGraphReport`

`SampledTrajectory` 本阶段不新增代码模型；它只为后续 original `Iris` + GCS
trajectory backend 输出预留，等实际需要序列化 Drake trajectory sample 时再落地。

实际行为：

- 从 `postprocess.corridor` 的 sections 生成 `grid_box` regions。
- 从 `inflated_passable_mask` / footprint-safe mask 生成 `blocked_cell_box`
  obstacle primitives。
- 相邻、重叠或接触的 regions 生成 `grid_adjacency` edges。
- route JSON 顶层可选加入 `region_graph_report`。
- `region_graph_report.motion_feasibility_status` 当前为 `diagnostic_only` 或
  `not_evaluated`，不得解释为 Ackermann 可行性证明。

Phase 8.1 仍不实现：

- GCS solver；
- `motion_feasibility_report`；
- Ackermann / skid-steer / differential motion backend。

## 8.2 Phase 8.2 Optional Workspace Iris Backend Prototype

Phase 8.2 在 optional backend 边界内落地 original 2D workspace `Iris`
region generation 原型。该阶段只生成和验证 2D convex safe regions，不替代
A*、postprocess、tracking simulation 或 fixed-corridor optimizer。

新增行为：

- 新增 `path_planner.drake_backend`，所有 `pydrake` import 限制在该 backend
  或 `pytest.mark.drake` 测试中。
- CLI 新增 `--drake-iris-regions`，启用后 route JSON 顶层可选输出
  `iris_region_report`。
- backend 标记为 `workspace_iris`，只调用 `pydrake.geometry.optimization.Iris`，
  不调用 `IrisNp`、`IrisNp2`、`IrisZo` 或 clique-cover C-space IRIS。
- seed 来自 `postprocess.corridor` section center。
- local domain 来自对应 `grid_box` corridor section。
- obstacles 复用 footprint-safe mask 生成的 `blocked_cell_box` primitives。
- IRIS 输出的 `HPolyhedron` 序列化为普通 numeric `A`、`b` arrays 和 bounds，
  不把 Drake object 写入 JSON。
- 每个 region 通过 grid cell-center sampling 做 unsafe-cell validation；若
  sample 落入 unsafe cell，则该 region invalid 并 fallback 到对应 `grid_box`
  region。

`iris_region_report` 至少包含：

- `backend`
- `status`
- `region_count`
- `seed_source`
- `domain_source`
- `obstacle_source`
- `obstacle_count`
- `validation_status`
- `failure_status`
- `failure_reason`
- `fallback_used`
- `regions`

Phase 8.2 failure / fallback status 至少覆盖：

- `backend_unavailable`
- `invalid_region_input`
- `infeasible`
- `solver_error`
- `fallback_used`

Phase 8.2 仍不实现：

- full GCS graph search；
- `GcsTrajectoryOptimization` sampled trajectory backend；
- `motion_feasibility_report`；
- Ackermann / skid-steer / differential motion backend；
- production Drake backend；
- 闭环控制命令流。

## 8.3 Phase 8.3 IRIS-Backed Region Graph And Diagnostics Readiness

Phase 8.3 把 Phase 8.2 的 `iris_region_report` 接入 `region_graph_report`，
使 IRIS regions 成为可检查、可比较、可回退的 region graph 输入候选。本阶段仍不
实现 GCS solver，也不声明任何车辆运动学可执行性。

新增行为：

- `build_region_graph_report(...)` 可接收 `iris_region_report`。
- 当 `iris_region_report.status == "ok"`、所有 regions 都是 valid `iris`
  regions，且 first-to-last graph connected 时，`region_graph_report.region_source`
  为 `iris`。
- IRIS-backed graph edge 使用 conservative sampled connectivity：基于 region
  sampled cell bounds 的 overlap / touch 判断，edge source 为
  `sampled_connectivity`。
- 当 Drake unavailable、IRIS report fallback、invalid region、empty region 或
  start-goal graph disconnected 时，`region_graph_report` 回退到现有
  `grid_box` graph。
- 回退只影响 region graph diagnostic，不改变 `reachable`、`trajectory_kind`、
  `geometric_path`、postprocess 或 fixed-corridor optimizer。

`region_graph_report.quality_metrics` 新增：

- `requested_region_source`
- `graph_source`
- `iris_region_count`
- `grid_fallback_region_count`
- `invalid_region_count`
- `fallback_ratio`
- `connected_component_count`
- `start_goal_connected`
- `fallback_used`
- `fallback_reason`

Diagnostics 新增 `IRIS / Region Graph Summary`，并明确说明：

- IRIS graph 是 2D workspace safe-region diagnostic；
- 不是 GCS trajectory；
- 不是 Ackermann / skid-steer feasibility proof。

Phase 8.3 仍不实现：

- full GCS graph search；
- `GcsTrajectoryOptimization` sampled trajectory backend；
- `motion_feasibility_report`；
- Ackermann / skid-steer / differential motion backend；
- production Drake backend；
- 闭环控制命令流。

## 9. Fallback 策略

Phase 8 必须实现或至少在设计上固定以下 fallback 链：

```text
Drake unavailable
  -> Drake backend failed or infeasible
  -> current fixed-corridor optimizer
  -> postprocess smoothed_path
  -> raw A* geometric_path
```

每一级 fallback 都要记录原因。Drake 失败不得把成功的 A* 结果改成 unreachable。

## 10. CLI 与依赖策略

现有 `--optimize-trajectory` 继续表示 fixed-corridor optimizer。

未来 Drake backend 使用独立入口，例如：

- `--drake-iris-regions`
- `--trajectory-backend fixed-corridor`
- `--trajectory-backend drake-gcs`
- 或 `--drake-gcs`

`pydrake` 不加入默认依赖。Drake 专项测试使用 pytest marker：

```bash
conda run -n lunar-explorer env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -m drake
```

默认测试在缺少 `pydrake` 时应 skip Drake 专项测试，而不是失败。

## 11. 非目标

Phase 8 Framework First 不做：

- 不实现完整 Ackermann 轨迹优化。
- 不把所有 rover 默认视为 Ackermann 车辆。
- 不使用 `IrisNp`、`IrisNp2`、`IrisZo` 作为主路径。
- 不把 Drake 作为默认运行依赖。
- 不移除现有 fixed-corridor optimizer。
- 不把 grid corridor 声明为 IRIS region。
- 不把 Drake trajectory object 直接写入 JSON。
- 不提供闭环控制命令流。

## 12. 验收标准

- pydrake 知识结构文档存在，并包含官方链接和本项目映射。
- Phase 8 spec 和 implementation plan 存在。
- Phase 8.1 region graph baseline 输出可选 `region_graph_report`，且不改变
  `trajectory_kind` 或 `reachable` 语义。
- Phase 8 spec 明确选择 2D workspace `Iris`，并说明不优先采用 `IrisNp`、
  `IrisNp2` 或 `IrisZo`。
- Phase 8 spec 说明非凸月表障碍物通过 convex obstacle primitives 表示。
- Phase 8 spec 说明 rover 运动学必须进入后续可行性层，但 Phase 8 不宣称
  Ackermann 可执行轨迹。
- Phase 8.2 `iris_region_report` 只在 `--drake-iris-regions` 启用时输出，且
  缺失时旧 JSON 消费者不受影响。
- `pydrake` import 只出现在 optional backend 或 Drake marker 测试中。
- Drake unavailable 或 region validation 失败时回退到 `grid_box` regions，不改变
  `reachable` 或 `trajectory_kind`。
- README 明确 Phase 8 是 framework prototype，不是已完成 full Drake backend。
- 默认 `python3 -m pytest` 在没有 pydrake 的环境中通过。
- `conda run -n lunar-explorer env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -m drake`
  能验证关键 pydrake API 可导入。

## 13. 下一阶段系统闭环验证

Phase 8 后的下一阶段不是直接实现 GCS solver 或 rover motion-feasibility backend，
而是把 `dev-platform-constraints -> model-explorer -> path-planner` 半真实 JSON
闭环固化为可信实验入口。

主验收链路：

```bash
bash scripts/run_path_feedback_validation.sh --scenario-set all --diagnostic-profile all --top-k 3
```

该链路应确认：

- path feedback summary 使用 sidecar 的真实 `cost` / `passable_mask`，
  `open_grid_fallback_used = false`；
- `path-feedback-summary/v1` 保留目标选择变化、路径失败、重规划、安全违规、
  fixed-corridor optimization fallback、IRIS 状态、region graph 来源/回退/断连和
  场景组聚合；
- stress / mixed-stress 场景可以稳定解释失败、绕行和 replan 原因；
- `path-planner-route/v1` 语义不变，`trajectory_kind` 仍为 `geometric_path`；
- `region_graph_report` 和 `iris_region_report` 仍为诊断字段，不宣称是 GCS
  trajectory，也不宣称输出 Ackermann/skid-steer feasible trajectory。
