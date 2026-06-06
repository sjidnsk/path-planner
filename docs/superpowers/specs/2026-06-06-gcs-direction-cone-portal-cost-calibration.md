# GCS Direction-Cone Portal and Cost Calibration v1

日期：2026-06-06

## 1. 目标

本阶段在已落地的 `pydrake_direction_cone_program` 上增强约束参数和成本解释。
实现范围仍是固定 `convex_region_sequence` 上的 Drake `MathematicalProgram` 2D
geometric candidate，不是完整 GCS graph search，也不是 Ackermann 可执行轨迹。

## 2. 约束参数设计

每条相邻 region seed edge 继续使用参考切向 `t`、法向 `n` 和方向锥线性约束：

- `t^T d_i >= rho`
- `n^T d_i - eta * t^T d_i <= 0`
- `-n^T d_i - eta * t^T d_i <= 0`

本阶段新增 `rho` 的解释来源：

- seed distance lower bound：相邻 seed 距离的保守比例。
- portal/overlap width：相邻 region 交叠或接触界面沿法向的宽度。
- support width：两个 region 沿法向支撑宽度的较小值。

solver 和 summary 使用同一个 `direction_cone_edge_parameters(...)` helper。这样
`backend_enforced=true` 时，报告里的 `rho_lower_bound_m` 与实际添加到 solver 的
前向约束一致。

## 3. 诊断合同

`gcs_trajectory_constraint_summary` 继续保持 additive，并新增：

- `rho_lower_bound_min_m`
- `rho_source_counts`
- `portal_width_min_m`
- `support_width_min_m`
- `constraint_tightness_min`

每个 parameter 新增：

- `rho_lower_bound_m`
- `rho_source`
- `rho_seed_distance_m`
- `rho_portal_m`
- `rho_support_m`
- `portal_width_m`
- `support_width_m`
- `constraint_tightness`

退化 portal/overlap 不静默通过。summary 会通过 risk flags 标注
`degenerate_portal_width`、`region_portal_gap` 或 `degenerate_support_width`。

## 4. 成本解释

GCS trajectory 和 candidate cost summary 继续以后验 sampled path 为准。本阶段新增：

- `terrain_cost_source`
- `high_cost_threshold`
- `sampled_cell_count`
- `blocked_sample_count`

candidate cost summary 还新增：

- `candidate_decision`
- `decision_reason`
- `quality_gate`

这样 candidate 被选中或被阻断时，summary 本身能解释 baseline delta、postprocess
delta、duplicate baseline 和 improvement epsilon 的质量门槛。

## 5. Candidate 选择边界

本阶段不放宽候选替换条件。以下情况仍必须阻断 route replacement：

- direction cone 未评估。
- direction cone 未 backend-enforced。
- direction cone violation。
- sampled trajectory collision。
- motion feasibility infeasible。
- cost dominated。
- duplicate baseline。
- no quality gain。

## 6. 非目标

- 不实现完整 GCS graph branch-and-bound、rounding 或 graph search。
- 不把变量升级为 Bezier/B-spline control points。
- 不加入 hard 曲率 fallback。
- 不使用 IrisNp、IrisNp2、IrisZo 或 C-space IRIS。
- 不重写 A*、postprocess、tracking 主链路。
- 不改变 `path-planner-route/v1`、`path-planner-request/v1`、
  `model-explorer-contract/v1` 的稳定语义。
- 不宣称输出 Ackermann-feasible trajectory。

## 7. 剩余风险

- waypoint 变量仍弱于 derivative-control-point formulation。
- `rho` 的 portal/support calibration 仍是工程保守下界，需要更多地图 fixture 校准。
- terrain/high-cost 对 solver 的影响仍通过 low-cost anchor proxy 表达，不是连续场积分。
- 可选 Drake backend 不可用时只能输出 not-evaluated summary。
