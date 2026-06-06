# GCS Direction-Cone Scenario Matrix and Contract Sync v1

日期：2026-06-06

## 1. 目标

本阶段把 `pydrake_direction_cone_program` 的验证从单点 smoke 推进到场景矩阵级
证据。目标是让 selected/blocked、fallback reason、direction-cone 参数和成本门槛能
从 route JSON 或稳定 summary 中解释清楚。

## 2. 场景矩阵

矩阵覆盖以下 case：

- open corridor 成功候选。
- cost dominated 高代价候选。
- duplicate baseline。
- sampled trajectory collision。
- direction cone 未评估。
- direction cone 未 backend-enforced。
- direction cone violation。
- motion infeasible turn。
- degenerate portal。
- pydrake unavailable。

每个 case 都记录 expected outcome 和 expected decision reason。若实际 outcome 或
reason 不匹配，summary 的 `expectation_failures` 必须列出 case id 和 mismatch fields。

## 3. Summary 合同

新增轻量 summary helper：

- `schema_version=gcs_direction_cone_scenario_matrix/v1`
- `case_count`
- `selected_count`
- `blocked_count`
- `decision_reason_counts`
- `fallback_reason_counts`
- `expectation_failures`
- `cases`

每个 case 摘要保留：

- `case_id`
- `outcome`
- `selected`
- `fallback_reason`
- `decision_reason`
- `trajectory_attempted`
- `trajectory_success`
- `direction_cone_status`
- `direction_cone_backend_enforced`
- `direction_cone_violation_count`
- `direction_cone_risk_flags`
- `rho_source_counts`
- `portal_width_min_m`
- `support_width_min_m`
- `constraint_tightness_min`
- `candidate_decision`
- `quality_gate`
- `cost_delta_vs_baseline`
- `high_cost_exposure`

## 4. CLI Contract Sync

CLI JSON remains the preferred acceptance surface. When GCS candidate flags are enabled, route JSON
must expose:

- `gcs_trajectory_constraint_summary`
- `rho_source_counts`
- `portal_width_min_m`
- `support_width_min_m`
- `constraint_tightness_min`
- `gcs_candidate_cost_summary`
- `candidate_decision`
- `decision_reason`
- `quality_gate`

Missing `pydrake` remains an explicit not-evaluated path, not a silent fallback to an unconstrained
candidate.

## 5. README Sync

README must describe the current state as a fixed-sequence
`pydrake_direction_cone_program` with additive GCS reports. It must not describe
`gcs_trajectory_report` as a future-only report, and it must not imply complete GCS graph search or
Ackermann-feasible trajectory output.

## 6. 非目标

- 不实现完整 GCS graph branch-and-bound、rounding 或 graph search。
- 不升级为 Bezier/B-spline control points。
- 不加入 hard 曲率 fallback。
- 不使用 IrisNp、IrisNp2、IrisZo 或 C-space IRIS。
- 不重写 A*、postprocess、tracking 主链路。
- 不改变 `path-planner-route/v1`、`path-planner-request/v1`、
  `model-explorer-contract/v1` 的稳定语义。
- 不宣称输出 Ackermann-feasible trajectory。
