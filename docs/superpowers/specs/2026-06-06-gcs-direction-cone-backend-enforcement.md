# GCS Direction-Cone Backend Enforcement v1

日期：2026-06-06

## 1. 目标

本阶段把上一阶段的 `direction_cone` 诊断和候选阻断合同推进为可验证的 backend
enforcement 原型。实现目标不是完整 GCS graph search，也不是 Ackermann 可执行轨迹；
它是在固定 `convex_region_sequence` 上，用 Drake `MathematicalProgram` 求解一条满足
方向锥线性约束的 2D geometric candidate。

本阶段继续保持：

- 不加入 hard 曲率回退。
- 不使用 IrisNp、IrisNp2、IrisZo 或 C-space IRIS。
- 不重写 A*、postprocess、tracking 主链路。
- 不改变 `path-planner-route/v1` 的 `trajectory_kind=geometric_path` 和 `reachable` 语义。

## 2. API 审计结论

`pydrake.planning.GcsTrajectoryOptimization` 当前在本项目可用的 Python API 包括：

- `AddRegions`
- `AddEdges`
- `AddPathLengthCost`
- `AddPathEnergyCost`
- `AddVelocityBounds`
- `AddNonlinearDerivativeBounds`
- `SolvePath`
- `SolveConvexRestriction`

它没有直接暴露本项目需要的 per-edge `direction_cone` 线性约束注入接口。因此本阶段
选择 Drake `MathematicalProgram` 作为 enforcing backend，明确命名为
`pydrake_direction_cone_program`，避免把它描述成完整 GCS backend。

## 3. 变量

给定一个已验证的 `ConvexRegionSequenceReport`：

- 每个 region 生成一个连续 2D 决策点 `p_i = [x_i, y_i]`。
- 首点固定到首 region seed。
- 末点固定到末 region seed。
- 输出轨迹由求解后的 waypoint polyline 按 `sample_count` 重采样。

该变量选择是最小原型。后续若接入 Bezier/B-spline，可把 `p_i` 替换为控制点，并把
direction-cone 约束施加到一阶/二阶导数控制点上。

## 4. 约束

已 backend-enforced 的约束：

- Region containment：每个 `p_i` 满足对应 region 的 H-polyhedron `A p_i <= b`。
- Start/goal fixed：首尾点使用 bounding-box equality 约束固定。
- Direction cone per edge：对每个相邻 region seed 构造参考切向 `t` 和法向 `n`，
  对 `d_i = p_{i+1} - p_i` 加入：
  - `t^T d_i >= rho`
  - `n^T d_i - eta * t^T d_i <= 0`
  - `-n^T d_i - eta * t^T d_i <= 0`

其中 `eta = tan(45 deg)`，`rho` 使用相邻 seed 距离的保守比例下界。

后验检查仍保留：

- sampled point collision check。
- sampled direction-cone summary 的 violation count。
- motion feasibility 的 curvature、heading、turning-radius 诊断。

## 5. 成本

backend objective 包含：

- segment length quadratic proxy。
- region 内低 cost anchor tracking proxy，用于让 terrain/high-cost 信息影响解。
- second-difference smoothness proxy。

route JSON 继续输出：

- `gcs_trajectory_cost_summary`
- `gcs_candidate_cost_summary`

其中 terrain path cost、high-cost exposure 和 baseline/postprocess delta 仍以 sampled path
后验计算为准。

## 6. 报告合同

`GcsTrajectoryReport.backend` 可以是：

- `pydrake_direction_cone_program`
- 旧值 `pydrake_gcs` 仅保留为兼容枚举，不作为本阶段默认成功路径。

只有当 MathematicalProgram 成功求解、实际添加 direction-cone 线性约束，并且 sampled
summary 没有 violation 时，报告才设置：

- `gcs_trajectory_constraint_summary.backend_enforced=true`
- `enforcing_backend=pydrake_mathematical_program`
- `fallback_reason=null`
- `solver_constraint_count > 0`

如果 pydrake 不可用、region 输入无效、reference degenerate 或 solver 失败，报告必须
保持 `backend_enforced=false`，并通过 `fallback_reason`/`result_status` 解释失败。
不得静默回退到无 direction-cone 约束的 GCS trajectory。

## 7. Candidate 选择

GCS geometric candidate 只有同时满足以下条件才可 `selected=true`：

- GCS trajectory success。
- sampled path collision-free。
- `direction_cone.evaluated=true`。
- `direction_cone.backend_enforced=true`。
- `direction_cone.violation_count=0`。
- motion feasibility 未评估或评估为 feasible。
- cost 不劣于 baseline/postprocess，并通过已有 quality gate。

若 direction-cone 只有诊断、未 enforce、未评估或 violation，必须阻断候选替换。

## 8. 非目标与剩余风险

非目标：

- 不宣称 Ackermann-feasible trajectory。
- 不实现完整 GCS graph branch-and-bound、rounding 或 fixed-path re-solve。
- 不加入 hard 曲率 fallback。
- 不改变上游 request/route 稳定契约。

剩余风险：

- 当前变量是 region waypoint，不是 Bezier/B-spline derivative control point。
- `rho` 和低 cost anchor 是保守原型参数，后续需要基于 portal/support width 收紧。
- terrain/high-cost 通过 anchor proxy 影响优化，不是连续 terrain field 积分成本。
