# pydrake 知识结构与 path-planner 映射

日期：2026-05-30

本文整理 Drake 官方 pydrake 文档中与 `path-planner` 后续 IRIS/GCS
框架相关的知识结构。本文只作为 Repo Docs 工程参考，不更新 Notion
导出副本。

官方参考：

- pydrake 总索引：https://drake.mit.edu/pydrake/index.html
- geometry optimization：https://drake.mit.edu/pydrake/pydrake.geometry.optimization.html
- planning：https://drake.mit.edu/pydrake/pydrake.planning.html
- solvers：https://drake.mit.edu/pydrake/pydrake.solvers.html
- trajectories：https://drake.mit.edu/pydrake/pydrake.trajectories.html
- pip 安装：https://drake.mit.edu/pip.html

## 1. 环境与依赖边界

Drake 的 Python 绑定通过 `pydrake` 暴露。当前项目的统一开发环境是
`lunar-explorer` Conda 环境，该环境应作为 Drake 专项验证入口。

设计映射：

- `path-planner` 默认依赖仍只包含 NumPy、Matplotlib 和 pytest。
- `pydrake` 不进入默认依赖；Drake 后端作为 optional backend。
- 默认 `python3 -m pytest` 不能因为未安装 `pydrake` 失败。
- Drake 专项验证使用：

```bash
conda run -n lunar-explorer env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -m drake
```

`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` 用于避免环境中无关 pytest 插件影响本项目的
Drake API 探测。

当前阶段不做：

- 不要求所有开发者本地全局 Python 都安装 Drake。
- 不把现有 fixed-corridor optimizer 改成 Drake 硬依赖实现。

## 2. ConvexSet 与 HPolyhedron

Drake 在 `pydrake.geometry.optimization` 中提供 convex set 相关类型。
与本项目最相关的是：

- `HPolyhedron`：半空间表示的凸多面体，形式为 `{x | A x <= b}`。
- `Point`：点集，可作为 start/goal 或 region seed 的简单 convex set。
- `ConvexSet`：Drake convex set 抽象基类族。

设计映射：

- 当前 `corridor` 的 grid section 不是 convex set。
- 当前 `corridor_boxes` 可以作为 `HPolyhedron` 的近似来源，但必须显式标记为
  box approximation。
- 后续 region 模型应保留 `source` 字段，例如 `grid_box`、`iris`、`manual`。

当前阶段不做：

- 不把所有 grid cell 精确转换为复杂多面体。
- 不声明现有 corridor 等价于 IRIS region。

## 3. IRIS 族

当前 `lunar-explorer` 环境中的 pydrake 暴露了多个 IRIS 相关接口：

- `pydrake.geometry.optimization.Iris`
- `pydrake.geometry.optimization.IrisNp`
- `pydrake.planning.IrisNp2`
- `pydrake.planning.IrisZo`
- `pydrake.planning.IrisInConfigurationSpaceFromCliqueCover`

这些接口的适用对象不同，不能只按名字都称为“IRIS”后直接替换使用。

### 3.1 原始 `Iris`

Drake 的原始 `Iris` 属于 `pydrake.geometry.optimization`，核心输入包括：

- convex obstacles；
- sample point；
- domain `HPolyhedron`；
- `IrisOptions`。

官方文档描述 IRIS 通过迭代方式寻找 obstacle-free convex region，输出
`HPolyhedron`。

设计映射：

- A* 或 smoothed path 上的 waypoint 可作为 IRIS sample point。
- 当前 inflated safe mask 和 obstacle cells 可用于构造 conservative obstacles。
- 当前 corridor box 可作为 IRIS domain 或 fallback region。
- `iris_region_report` 应记录 sample、domain、status、region count、failure reason。

该方法最贴合当前 `path-planner`，因为当前规划空间是 2D 月面 workspace，车辆
footprint 已经通过 inflated passable mask 转换为 point-center safe region。

### 3.2 `IrisNp`

`IrisNp` 在 `MultibodyPlant` 的 configuration space 中寻找 collision-free region。
它需要 plant、context，以及连接到 SceneGraph 的碰撞几何。它适合已经建立 C-space
和碰撞检查器的机器人系统。

当前项目还没有 `(x, y, theta)` 状态空间、Ackermann backend 或
MultibodyPlant/SceneGraph pipeline，因此不应把 `IrisNp` 作为 Phase 8 主路径。

### 3.3 `IrisNp2`、`IrisZo` 和 clique-cover

`pydrake.planning` 下的 `IrisNp2`、`IrisZo` 和
`IrisInConfigurationSpaceFromCliqueCover` 偏向 configuration-space / sampled
region generation。它们更适合复杂机器人 C-space 覆盖，而不是当前的 2D grid
safe-region 原型。

Phase 8 可记录这些能力，但不使用它们作为默认路线。

当前阶段不做：

- 不保证从 lunar grid 自动生成高质量 convex obstacles。
- 不在没有可验证 obstacle 建模前宣称 IRIS region 是完整月面安全区域。
- 不优先使用 C-space IRIS 变体。

## 3.4 月表非凸障碍物与 convex obstacle primitives

原始 `Iris` 要求每个 obstacle primitive 是 convex set，但不要求整个月表障碍物
整体是凸的。非凸 unsafe space 可以表示为多个凸障碍物的并集。

推荐路线：

- 使用现有 `inflated_passable_mask` 作为车辆中心点不可进入区域。
- v1 将每个 blocked cell 转换为 2D axis-aligned convex box。
- v1.5 将相邻 blocked cells 合并为 axis-aligned rectangles，降低 obstacle 数量。
- 不默认使用 connected component 的单个 convex hull，因为它会封死凹口、窄通道
  和真实可通行区域。
- 每个 IRIS seed 只在 local domain 内运行，避免整图 obstacle 数量过大。
- 对输出 `HPolyhedron` 做 grid sampling / cell intersection 校验；失败时回退到
  `grid_box` region。

## 4. GraphOfConvexSets

`GraphOfConvexSets` 位于 `pydrake.geometry.optimization`。官方文档说明它实现
"Shortest Paths in Graphs of Convex Sets" 设计模式：每个 vertex 关联一个
convex set，edge 包含连续变量上的 convex costs 和 constraints，可求解图上的
shortest path 类问题。

重要事实：

- `GraphOfConvexSets` 是 Drake 中的 experimental feature。
- 典型方法包括 `AddVertex`、`AddEdge`、`SolveShortestPath`、
  `SolveConvexRestriction`、`GetGraphvizString`。

设计映射：

- `region_graph_report` 对应本项目层的 region adjacency graph。
- 每个 safe region 可以映射为 GCS vertex。
- region overlap 或可连通关系可以映射为 GCS edge。
- GCS 失败时必须回退到 current optimizer 或 postprocess path。

当前阶段不做：

- 不直接暴露 Drake 内部 `MathematicalProgram` 变量作为公共契约。
- 不将 GCS relaxation 的解无条件视为最终可执行路径。

## 5. GcsTrajectoryOptimization

`pydrake.planning.GcsTrajectoryOptimization` 是 Drake 提供的面向运动规划的
GCS 简化接口。官方文档说明它用于 obstacle-around motion planning，并支持：

- `AddRegions`
- `AddEdges`
- `AddPathLengthCost`
- `AddPathEnergyCost`
- `AddTimeCost`
- `AddVelocityBounds`
- `AddContinuityConstraints`
- `SolvePath`
- `SolveConvexRestriction`
- `NormalizeSegmentTimes`
- `UnwrapToContinuousTrajectory`

重要事实：

- `GcsTrajectoryOptimization` 也是 experimental feature。
- 输出更接近连续轨迹候选，但不等价于闭环控制命令。

设计映射：

- 对 `path-planner`，优先把它作为 Phase 8 的 Drake trajectory backend 研究对象。
- `gcs_trajectory_report` 应记录 backend、region source、solver status、path cost、
  trajectory sample、warnings 和 fallback status。
- 顶层 `trajectory_kind` 继续保持 `geometric_path`，直到项目真正提供稳定的
  executable trajectory contract。

当前阶段不做：

- 不用 `--optimize-trajectory` 直接切换到 GCS；该参数继续代表现有
  fixed-corridor optimizer。
- 不把 GCS 输出直接当作 Ackermann-feasible trajectory。

## 6. MathematicalProgram 与 solvers

`pydrake.solvers` 提供 `MathematicalProgram`、`Solve`、solver options 和求解结果
相关接口。GCS 相关类内部会构造优化问题，但上层接口不一定暴露全部内部变量。

设计映射：

- 本项目需要记录 solver status、success/failure、warnings、fallback reason。
- 未来如果直接使用 `MathematicalProgram`，应放在独立 backend 内，避免污染
  `core`、`search`、`postprocess`。

当前阶段不做：

- 不创建手写 Drake optimization program 作为主路径。
- 不把 solver-specific option 暴露成稳定 JSON schema。

## 7. Trajectories

`pydrake.trajectories` 提供 trajectory 表示，例如：

- `PiecewisePolynomial`
- `BsplineTrajectory`

设计映射：

- Drake backend 的连续输出需要采样为现有 JSON 可读结构。
- `trackable_path` 可以作为采样后的工程接口，但不能反向证明原 trajectory
  满足全部车辆运动学约束。
- 诊断图可先显示 sampled trajectory，不必序列化 Drake 对象本身。

当前阶段不做：

- 不把 Drake trajectory object 直接写入 route JSON。
- 不改变现有 `trackable_path` 的兼容字段。

## 8. Rover 运动学与 Ackermann 边界

月球探索车必须考虑运动学可行性，但不能未经平台构型确认就默认使用严格
Ackermann 模型。月球车可能是 rocker-bogie、多轮独立驱动、部分轮独立转向或
skid-steer 近似平台。

设计映射：

- Phase 8 的 IRIS/GCS 输出是 2D geometric safe-region / sampled path candidate。
- GCS 输出不自动等价于 Ackermann-feasible trajectory。
- 当前 `curvature_report`、`trackable_path` 和 tracking simulation 只能提供后验
  可执行性诊断。
- 后续应引入 `motion_feasibility_report`，让 `ackermann` 成为 motion model 的
  一个选项，而不是默认事实。

建议后续 motion model 字段：

- `point`
- `curvature_bounded`
- `ackermann`
- `skid_steer`
- `differential`
- `custom`

## 9. 对 path-planner 的阶段结论

当前项目已经具备：

- platform-aware A*；
- postprocess corridor 和 smoothing；
- trackable path；
- pure-pursuit tracking simulation baseline；
- fixed-corridor continuous optimization baseline。

因此 Phase 8 应进入 IRIS/GCS framework prototype，而不是直接替换现有规划器。
IRIS 路线应优先选择原始 `Iris` over 2D convex obstacle primitives，C-space IRIS
变体留到 rover motion model 和碰撞几何成熟后再评估。
推荐 fallback 链：

```text
Drake unavailable
  -> Drake backend failed or infeasible
  -> current fixed-corridor optimizer
  -> postprocess smoothed_path
  -> raw A* geometric_path
```

该链保证 Drake 研究不会破坏现有可联调、可诊断的规划输出。
