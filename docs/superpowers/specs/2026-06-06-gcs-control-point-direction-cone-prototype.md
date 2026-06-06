# GCS Control-Point Direction-Cone Prototype v1

日期：2026-06-06

## 1. 目标

本阶段把固定 `convex_region_sequence` 上的 direction-cone 原型从 region waypoint
推进到 control-point derivative proxy。实现目标是新增一个明确 opt-in 的
`pydrake_control_point_direction_cone_program`，让 route JSON 能区分 waypoint backend
和 control-point backend，并能审计 derivative direction-cone 约束、control-point
containment 和成本项。

该阶段仍输出 sampled 2D geometric candidate diagnostic，不输出 Ackermann-feasible
trajectory。

## 2. 约束设计

对每个 region 建立 2D control point `cp_i`：

- region containment：`A_i cp_i <= b_i`
- start/goal fixed：首尾 control point 固定到首尾 seed
- derivative proxy：`d_i = cp_{i+1} - cp_i`
- direction cone：

```text
t_i^T d_i >= rho_i
n_i^T d_i - eta_i * t_i^T d_i <= 0
-n_i^T d_i - eta_i * t_i^T d_i <= 0
```

其中 `rho_i` 仍来自 seed distance、portal width 和 support width 的保守组合。

## 3. 成本设计

当前 objective 只保留能解释的二次项：

- `segment_length_quadratic`
- `low_cost_anchor_quadratic`
- `control_point_second_difference_quadratic`

route JSON 的 candidate decision 仍由 sampled path 后验 cost/collision/motion gate
决定，而不是只看 solver success。

## 4. Route JSON 证据

`gcs_trajectory_report/v1` 保持 schema 不变，通过字段区分 backend：

- `gcs_trajectory_backend=pydrake_control_point_direction_cone_program`
- `gcs_trajectory_constraint_summary.trajectory_parameterization=control_point_derivative_proxy`
- `control_point_count`
- `derivative_proxy=successive_control_point_difference`
- `derivative_constraint_count`
- `control_point_region_containment_count`
- `start_goal_constraint_count`
- `objective_terms`

CLI opt-in：

```bash
python -m path_planner.cli \
  --input examples/demo_map_corridor.json \
  --output-json outputs/demo/control-point-route.json \
  --output-dir outputs/demo-control-point \
  --drake-iris-regions \
  --gcs-control-point-candidate \
  --gcs-motion-feasibility
```

## 5. 非目标

- 不实现完整 GCS graph branch-and-bound、rounding 或 graph search。
- 不实现 production Bezier/B-spline trajectory optimizer。
- 不加入 hard 曲率 fallback。
- 不使用 IrisNp、IrisNp2、IrisZo 或 C-space IRIS。
- 不重写 A*、postprocess、tracking 主链路。
- 不改变 `path-planner-route/v1`、`path-planner-request/v1`、
  `model-explorer-contract/v1` 的稳定语义。
- 不宣称输出 Ackermann-feasible trajectory。
