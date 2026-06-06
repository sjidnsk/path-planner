# GCS Motion-Feasibility Batch Gate v1

日期：2026-06-06

## 1. 目标

本阶段把 `--gcs-motion-feasibility` 从单点 route JSON 诊断推进为可回归的
CLI batch evidence。batch 必须通过真实 `path_planner.cli` 生成 route JSON，再从
route 字段汇总 heading、curvature、turning-radius、candidate blocking 和 expected
outcome。

该阶段是 sampled 2D geometric candidate 的后验 gate，不是 Ackermann trajectory
optimizer。

## 2. Summary 合同

新增 summary schema：

- `schema_version=gcs_motion_feasibility_cli_batch/v1`
- `case_count`
- `feasible_count`
- `infeasible_count`
- `diagnostic_only_count`
- `candidate_selected_count`
- `candidate_blocked_count`
- `decision_reason_counts`
- `fallback_reason_counts`
- `expectation_failures`
- `cases`

每个 case 摘要保留：

- `case_id`
- `outcome`
- `decision_reason`
- `fallback_reason`
- `trajectory_attempted`
- `trajectory_success`
- `trajectory_reason`
- `motion_evaluated`
- `motion_status`
- `motion_fallback_reason`
- `motion_model`
- `min_turning_radius_m`
- `max_heading_change_deg`
- `curvature_violation_count`
- `heading_violation_count`
- `violation_indices`
- `sample_count`
- `path_length`
- `max_observed_curvature`
- `min_observed_turning_radius_m`
- `max_observed_heading_change_deg`
- `candidate_selected`
- `candidate_available`
- `candidate_fallback_reason`
- `motion_gate_blocked_candidate`

## 3. CLI 场景矩阵

内置 batch 覆盖：

- `straight_feasible`
- `gentle_turn_feasible`
- `sharp_turn_blocked`
- `tight_radius_blocked`
- `direction_cone_selected_but_motion_blocked`
- `pydrake_unavailable`

命令：

```bash
python -m path_planner.drake_backend.gcs_cli_batch \
  --batch-kind motion-feasibility \
  --output-dir outputs/gcs-motion-batch \
  --summary-json outputs/gcs-motion-batch/summary.json
```

默认不传 `--batch-kind` 时，旧的
`gcs_direction_cone_cli_scenario_batch/v1` 行为保持不变。

## 4. 非目标

- 不实现完整 GCS graph branch-and-bound、rounding 或 graph search。
- 不升级为 Bezier/B-spline control points。
- 不加入 hard 曲率 fallback。
- 不使用 IrisNp、IrisNp2、IrisZo 或 C-space IRIS。
- 不重写 A*、postprocess、tracking 主链路。
- 不改变 `path-planner-route/v1`、`path-planner-request/v1`、
  `model-explorer-contract/v1` 的稳定语义。
- 不宣称输出 Ackermann-feasible trajectory。
