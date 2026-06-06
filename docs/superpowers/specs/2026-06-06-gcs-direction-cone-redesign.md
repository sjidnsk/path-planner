# path-planner GCS Direction Cone Redesign

日期：2026-06-06

## 1. 目标

本设计围绕根仓库 `docs/算法设计与系统架构报告.md` 中的 GCS 轨迹优化目标，收敛
`path-planner` 当前实现中的 GCS 约束和成本合同。本阶段只采用 `direction_cone`
曲率思路，不引入 hard 曲率回退。

当前 `path-planner` 不要求复刻报告中的 A*、局部走廊或凸区域实现。前置链路只需
为 GCS 提供可用的参考路径、可通行安全域、区域连通关系、地形成本和失败诊断。
现有 A*、postprocess、tracking 和 fixed-corridor optimizer 主链路保持稳定。

## 2. 当前实现

当前 GCS 相关路径位于 `src/path_planner/drake_backend/`：

- `gcs_trajectory.py` 调用可选 `pydrake.planning.GcsTrajectoryOptimization`，
  从 `convex_region_sequence_report` 生成 2D sampled trajectory。
- `gcs_candidate.py` 将 GCS sampled points 作为几何候选，与 A* baseline 和
  postprocess 结果比较。
- `gcs_motion_feasibility.py` 对 sampled points 做曲率、heading 和 turning-radius
  后验诊断。
- `gcs_curvature_constrained_candidate.py` 是现有采样点修复诊断，不等同于 GCS
  solver 内部的 direction_cone 约束。
- route JSON 保持 `schema_version=path-planner-route/v1`、
  `trajectory_kind=geometric_path` 和 `reachable` 语义不变。

## 3. 变量与约束合同

GCS 的核心输入是 `ConvexRegionSequenceReport`：

- 每个 region 提供 2D H-polyhedron、seed point 和 world bounds。
- 相邻 region 的 overlap/portal 决定 GCS 是否 ready。
- start/goal 由首尾 region seed 接入。

本阶段明确以下约束合同：

- 区域包含：GCS trajectory sampled points 必须复检 passable mask；碰撞样本使
  `gcs_trajectory_success=false` 或候选不可用。
- 起终点固定：当前 backend 使用首尾 region seed 作为 start/goal。
- 连通性：`convex_region_sequence_report.gcs_ready` 和相邻 region overlap 是
  求解前置条件。
- `direction_cone`：根据 sampled trajectory 和 region seed reference 构造
  离散方向锥诊断参数，包括 tangent `t`、normal `n`、前向下界 `rho`、方向锥
  `eta`、region width 和 risk flags。
- motion feasibility：曲率、heading、turning-radius 仍是 sampled points 的后验
  诊断，不宣称完整车辆动力学可执行性。

当前 Drake wrapper 尚未把 `direction_cone` 线性约束注入 solver。因此报告必须显式
写出 `backend_enforced=false` 和
`fallback_reason=direction_cone_backend_constraint_not_supported`。这不是 hard
曲率回退，也不能解释为已求解完整 Ackermann-feasible trajectory。

## 4. 成本合同

GCS 报告新增结构化成本摘要：

- `path_length`
- `terrain_path_cost`
- `high_cost_exposure`
- `energy_proxy`
- `smoothness_proxy`

GCS candidate 报告在此基础上增加：

- `baseline_path_cost`
- `postprocess_path_cost`
- `cost_delta_vs_baseline`
- `cost_delta_vs_postprocess`

候选选择不能只看 solver success。以下情况必须阻止 route replacement：

- sampled trajectory collision
- `direction_cone` 未评估
- `direction_cone` violation
- motion feasibility 已评估且 infeasible
- candidate cost dominated
- duplicate baseline
- no quality gain

## 5. JSON 契约

所有新增字段都是 additive：

- `gcs_trajectory_constraint_summary`
- `gcs_trajectory_cost_summary`
- `gcs_candidate_constraint_summary`
- `gcs_candidate_cost_summary`

这些字段不改变旧字段含义。缺失或不可用 backend 必须通过 `attempted`、
`success/evaluated`、`selected`、`fallback_reason` 和 summary 字段解释，而不是
静默失败。

## 6. 已实现、后验项与 Future Hook

已实现：

- direction_cone 参数和风险诊断摘要。
- GCS trajectory cost summary。
- GCS candidate cost summary。
- direction_cone 未评估/违反时的候选阻断。
- motion infeasible 时的候选阻断。
- pydrake 不可用路径的未评估 summary。

后验项：

- sampled point collision check。
- curvature、heading、turning-radius feasibility。
- high-cost exposure 和 smoothness proxy。

Future hook：

- 将 direction_cone 线性约束直接注入 Drake GCS/Bezier 或项目自有连续优化 backend。
- 用 Bezier/B-spline derivative control points 替代当前 sampled segment proxy。
- 将 region portal 宽度和支持宽度用于更严格的 `rho` 下界估计。

## 7. 非目标

- 不宣称输出 Ackermann-feasible trajectory。
- 不使用 IrisNp、IrisNp2、IrisZo 或 C-space IRIS。
- 不加入 hard 曲率回退路径。
- 不重写 A*、postprocess、tracking 主链路。
- 不破坏 `model-explorer-contract/v1`、`path-planner-request/v1`、
  `path-planner-route/v1` 的稳定语义。
