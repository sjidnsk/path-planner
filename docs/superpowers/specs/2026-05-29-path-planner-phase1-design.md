# path-planner 第一阶段设计

日期：2026-05-29

## 1. 背景与目标

`path-planner` 是 `a_gcs_ws-2.0.1` 的重新开发版本，但不是源码迁移项目。旧项目只作为算法目标、系统边界、失败经验和文档参考；新版不复制旧源码，不把旧项目作为运行依赖，也不沿用旧项目的混乱包结构。

第一阶段目标是实现一个可联调、可测试、可展示的月面 A* 路径规划器：

- 在内部使用干净的数据协议；
- 能通过 adapter 消费外部 `cost` 和 `passable_mask`；
- 支持从月面语义图层合成基础代价图；
- 用 2D A* 输出几何路径骨架；
- 通过 CLI 输出 JSON；
- 生成 PNG/HTML 诊断四联图；
- 为后续 `model-explorer`、`dev-platform-constraints`、Corridor、GCS/Ackermann 接入预留接口。

第一阶段输出的是 `geometric_path`，不是车辆可执行轨迹。

## 2. 设计原则

1. 从零实现：不复制 `a_gcs_ws-2.0.1` 源码，不依赖旧包。
2. 分层清晰：搜索层只依赖标准化后的 `cost` 与 `passable_mask`。
3. 外部隔离：通过 adapter 接入其他项目，不让外部实验字段污染内部模型。
4. 先可用再优化：第一阶段不实现 GCS/Ackermann，不依赖 Drake。
5. 失败可解释：所有失败路径必须返回明确原因和诊断信息。
6. Windows 可验证：第一阶段验收命令应能在当前 Windows 环境中轻量运行。

## 3. 第一阶段范围

### 3.1 模块

包名使用 `path_planner`，第一阶段包含：

- `path_planner.core`
  - 基础数据结构和结果协议。
  - 地图规格、坐标转换、规划请求、规划结果、失败原因。
- `path_planner.costmap`
  - 可选代价图合成器。
  - 从 `slope`、`roughness`、`illumination`、`confidence` 等语义层生成 `cost` 和 `passable_mask`。
- `path_planner.adapters`
  - 外部契约适配层。
  - 第一阶段预留 `DevPlatformAdapter` 和 JSON 输入适配。
- `path_planner.search`
  - 2D A* 搜索。
  - 使用 8 邻接和 corner-cut 防护。
- `path_planner.diagnostics`
  - PNG/HTML 诊断四联图。
  - 展示 `Cost + Path`、`Passable Mask`、`Expanded Nodes`、`Summary Metrics`。
- `path_planner.cli`
  - CLI demo。
  - 读取 demo JSON 或合成示例输入，输出路径 JSON 和诊断报告。

### 3.2 阶段外内容

第一阶段不做：

- 不实现 GCS、IRIS、Ackermann 轨迹优化。
- 不创建 GCS/Ackermann 空壳模块。
- 不实现探索目标选择、观测更新或探索主循环。
- 不依赖 Drake。
- 不把 `a_gcs_ws-2.0.1` 作为源码或运行依赖。

## 4. 核心数据结构

### 4.1 `GridSpec`

描述地图坐标协议：

- `width`
- `height`
- `resolution`
- `origin`
- `frame_id`

职责：

- 校验地图尺寸；
- 执行 cell/world 坐标转换；
- 判断 cell 是否越界。

### 4.2 `CostGrid`

标准搜索输入：

- `spec: GridSpec`
- `cost: ndarray`
- `passable_mask: ndarray[bool]`
- `metadata: dict`

约束：

- `cost` 必须为二维数组；
- `cost` 与 `passable_mask` 尺寸一致；
- 可通行格子的代价必须是非负有限值；
- 不可通行格子可有有限高代价，但不会被搜索扩展。

### 4.3 `PlanRequest`

规划请求：

- `start`
- `goal`
- `neighbor_policy`
- `corner_cut_policy`
- `max_iterations`

第一阶段 `start` 和 `goal` 以 cell 坐标为主。接口可预留 world 坐标转换，但搜索层内部统一使用 cell。

### 4.4 `PlanResult`

规划结果：

- `success`
- `path_cells`
- `path_world`
- `total_cost`
- `expanded_count`
- `failure_reason`
- `diagnostics`

语义要求：

- `path_cells` 是几何路径骨架；
- `path_world` 由 `GridSpec` 转换得到；
- `total_cost` 是搜索代价，不等价于车辆能耗或轨迹优化成本；
- `failure_reason` 失败时必须非空。

## 5. A* 算法行为

第一阶段搜索器采用 2D A*：

- 邻接：8-neighbor。
- 启发式：octile distance。
- 斜向代价：基础距离乘以相邻格代价，斜向距离为 `sqrt(2)`。
- corner-cut 防护：斜向移动时，禁止从两个不可通行正交邻格形成的障碍夹角中穿过。

失败原因枚举：

- `invalid_input`
- `invalid_cost`
- `start_out_of_bounds`
- `goal_out_of_bounds`
- `start_blocked`
- `goal_blocked`
- `unreachable`
- `max_iterations`

诊断信息至少包含：

- 扩展节点数量；
- 最大 frontier 大小；
- 路径节点数；
- 路径长度；
- 代价统计；
- 邻接策略；
- corner-cut 策略；
- 运行时间。

## 6. Costmap 设计

搜索层只消费 `cost` 和 `passable_mask`。语义图层进入规划的方式由 `costmap` 层负责。

第一阶段支持两类输入：

1. 已生成的 `cost` 和 `passable_mask`。
2. 多层月面语义输入，包含可选：
   - `slope`
   - `roughness`
   - `illumination`
   - `confidence`
   - `obstacle`
   - `valid_mask`

合成策略：

- 超过平台能力或无效区域进入硬约束，写入 `passable_mask = false`。
- 坡度、崎岖度、低光照、低可信度进入软代价。
- 输出非负有限 `cost`。

第一阶段只实现基础权重合成，不做复杂能耗模型。

## 7. 外部接口预留

### 7.1 与 `dev-platform-constraints`

预留 `DevPlatformAdapter`：

- 接收 `cost`、`passable_mask` 或等价硬约束输出；
- 可选接收 `slope`、`roughness`、`illumination`、`confidence` 等语义层；
- 转换为内部 `CostGrid`；
- 保留外部图层摘要到 `metadata`，供诊断报告使用。

该 adapter 只负责格式转换，不复制 `dev-platform-constraints` 的建模逻辑。

### 7.2 与 `model-explorer`

预留 `ExplorerPlanningPort`：

- `plan_route(request) -> route_result`
- `evaluate_route(request) -> route_evaluation`

第一阶段以 Python API 和 JSON contract 的形式预留，不做服务化 API。

响应字段应包含：

- `reachable`
- `geometric_path`
- `path_cost`
- `diagnostics`
- `failure_reason`

`model-explorer` 可用这些字段做目标可达性判断、路径代价反馈和重规划触发，但目标选择、观测更新、探索主循环仍不属于 `path-planner` 第一阶段职责。

### 7.3 Schema 版本

`path-planner` 维护自己的 schema 版本，例如：

```text
path-planner-route/v1
```

内部 schema 不依赖外部实验字段。外部字段通过 adapter 转换。

## 8. CLI 与诊断报告

CLI 第一阶段能力：

- 读取 demo JSON 输入；
- 或生成内置示例地图；
- 运行 A*；
- 输出路径 JSON；
- 可选写入 PNG/HTML 诊断报告。

诊断四联图：

1. `Cost + Path`
2. `Passable Mask`
3. `Expanded Nodes`
4. `Summary Metrics`

HTML 报告应包含规划摘要、失败原因、代价统计和输入元数据。PNG 用于快速查看，HTML 用于报告与调试。

## 9. 测试与验收

测试范围：

- `core`
  - 坐标转换；
  - 越界检查；
  - cost/mask 形状和数值校验。
- `costmap`
  - 已有 cost/mask 输入；
  - 简单语义图层合成；
  - 硬约束排除。
- `search`
  - 空地图可达；
  - 障碍绕行；
  - 加权代价偏好；
  - corner-cut 防护；
  - 起点阻塞；
  - 终点阻塞；
  - 完全不可达；
  - 非法代价。
- `cli/diagnostics`
  - JSON 输出；
  - PNG 生成；
  - HTML 生成；
  - 失败案例报告。

验收标准：

- 单元测试通过；
- CLI demo 能输出路径 JSON；
- CLI demo 能生成诊断 PNG/HTML；
- 至少包含一个失败案例测试；
- 文档明确说明第一阶段输出是 `geometric_path`，不是车辆可执行轨迹。

## 10. 后续路线

P1：Core + Costmap + A* + Diagnostics。

P2：Corridor + smoothing + curvature post-check。

P3：GCS/Ackermann optimizer + fallback chain。

后续 GCS/Ackermann 应作为高质量轨迹优化器接入，而不是第一阶段唯一可用规划器。若 GCS 求解失败，系统应保留 A* 几何路径、轻量平滑结果、失败原因和建议修复动作。
