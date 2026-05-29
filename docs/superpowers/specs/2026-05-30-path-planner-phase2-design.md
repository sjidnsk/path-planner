# path-planner 第二阶段设计

日期：2026-05-30

## 1. 目标

第二阶段把第一阶段的 A* 几何路径规划器推进为“可执行性评估前端”。系统仍然输出 `geometric_path`，但会在 A* 路径之后增加轻量后处理：

- 从原始 A* 路径生成基于 `passable_mask` 的 corridor；
- 对路径做 line-of-sight shortcut，得到不穿越不可通行区域的 `smoothed_path`；
- 对原始或平滑路径做曲率后验检查；
- 在 JSON、CLI 和诊断报告中暴露 `raw_path`、`corridor`、`smoothed_path`、`curvature_report`、`fallback_status`；
- 保留 A* 原始路径作为稳定 fallback。

第二阶段不实现 GCS、IRIS、Ackermann 优化器、Drake 或服务化 API。

## 2. 模块边界

新增 `path_planner.postprocess` 模块，负责 A* 之后的可执行性评估前处理。

建议文件：

- `postprocess/corridor.py`
  - 根据 `CostGrid` 和路径 cell 生成 corridor。
  - corridor 是每个路径 cell 周围可通行邻域的保守描述，不是凸优化走廊。
- `postprocess/smoothing.py`
  - 使用 grid line-of-sight shortcut 抽稀路径。
  - 平滑失败时返回原路径并记录 fallback 原因。
- `postprocess/curvature.py`
  - 基于连续三点估计转角、转弯半径和曲率违规。
  - 只做后验检查，不修改车辆运动学。
- `postprocess/pipeline.py`
  - 串联 corridor、smoothing 和 curvature check。
  - 输出统一的 `PostprocessResult`。

## 3. 输出契约

保持 Phase 1 顶层字段不变：

- `reachable`
- `geometric_path`
- `path_cost`
- `diagnostics`
- `failure_reason`

新增字段放在 `postprocess` 下：

- `raw_path`
  - A* 原始 cell/world 路径，与 `geometric_path` 等价但语义更明确。
- `corridor`
  - `status`
  - `radius_cells`
  - `sections`
  - `failure_reason`
- `smoothed_path`
  - `status`
  - `cells`
  - `world`
  - `fallback_reason`
- `curvature_report`
  - `is_feasible`
  - `max_curvature`
  - `min_turning_radius`
  - `violation_indices`
  - `summary`
- `fallback_status`
  - `used_raw_path`
  - `reason`

`postprocess` 失败不能让成功的 A* 结果变成不可达；失败时必须保留 `geometric_path`。

## 4. Corridor 策略

第一版 corridor 使用 grid 层面的保守近邻区域：

1. 每个路径 cell 必须在地图内且可通行。
2. 以 `radius_cells` 搜索周围 cell。
3. 只纳入可通行 cell。
4. 若某个路径点自身不可通行，则 corridor 失败。
5. corridor section 记录中心点和可通行邻域边界。

该 corridor 用于诊断和后续优化器输入准备，不声明为凸区域。

## 5. Smoothing 策略

第一版平滑采用 line-of-sight shortcut：

1. 从原路径起点开始，尽可能连接到最远可视路径点。
2. line-of-sight 经过的 grid cell 必须全部可通行。
3. 若无法产生更短路径，返回原路径，状态为 `unchanged`。
4. 若输入路径为空或包含不可通行点，返回 fallback。

该策略只减少不必要折点，不做曲线拟合。

## 6. 曲率后验检查

曲率检查基于离散路径三点转角：

1. 三点共线时曲率为 0。
2. 对非共线三点，用外接圆半径估算曲率。
3. 若曲率大于 `max_curvature`，记录中间点索引为违规点。
4. 报告 `min_turning_radius` 和 `max_curvature`。

这只是可执行性后验检查，不等价于 Ackermann 轨迹优化。

## 7. 验收标准

- 单元测试覆盖 corridor、smoothing、curvature 和 pipeline。
- CLI 输出包含兼容旧字段和新增 `postprocess` 字段。
- PNG/HTML 诊断能显示 raw path、smoothed path 和曲率报告摘要。
- `python -m pytest` 通过。
- CLI demo 能生成 `outputs/demo/route.json`、`diagnostics.png`、`diagnostics.html`。
- 文档明确 Phase 2 仍不是完整车辆轨迹优化。
