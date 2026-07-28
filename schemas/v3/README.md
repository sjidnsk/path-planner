# 路径规划 v3 JSON Schema

本目录是路径规划 v3 的语言无关线协议合同。所有 schema 使用 JSON Schema Draft 2020-12，并通过稳定 URN `$id` 相互引用。运行时 C++ 不读取 schema 文件；它使用与本合同一一对应的显式 codec 和语义校验器。schema 文件用于接口评审、离线记录、跨语言一致性测试和 CI 校验。

## 文件职责

| 文件 | `$id` | 职责 |
|---|---|---|
| `common.schema.json` | `urn:lunar-path-planner:v3:common` | 公共标量、向量、状态、确定性误差集合、时间、样条和多项式 |
| `planning-request.schema.json` | `urn:lunar-path-planner:v3:planning-request` | 单次不可变规划请求 |
| `planning-response.schema.json` | `urn:lunar-path-planner:v3:planning-response` | 规划结果与执行指令 |
| `reference-bundle.schema.json` | `urn:lunar-path-planner:v3:reference-bundle` | 原子激活 bundle、组件视图、有效性和生成证据 |
| `references/wheeled.schema.json` | `urn:lunar-path-planner:v3:reference:wheeled` | 轮式行驶/自旋分段参考 |
| `references/legged.schema.json` | `urn:lunar-path-planner:v3:reference:legged` | 足式固定机体参考点的几何与时序参考 |
| `references/hopper.schema.json` | `urn:lunar-path-planner:v3:reference:hopper` | 下一安全着陆范围、唯一飞跃边界和认证飞行管 |
| `safety-capability-profile.schema.json` | `urn:lunar-path-planner:v3:safety-capability-profile` | 三平台强制安全能力 |
| `planner-algorithm-config.schema.json` | `urn:lunar-path-planner:v3:planner-algorithm-config` | 搜索、走廊、连续化和有限资源配置 |
| `benchmark-profile.schema.json` | `urn:lunar-path-planner:v3:benchmark-profile` | 仅实验使用的固定性能测试配置 |
| `benchmark-report.schema.json` | `urn:lunar-path-planner:v3:benchmark-report` | 仅实验使用的时延统计报告 |

## 版本、引用与哈希

`schema_version` 表示线协议修订，例如 `path-planner-v3-planning-request/v1`。它和设计代际 v3 是两个维度。

`ContentRef` 的固定结构为：

```json
{
  "id": "content-id",
  "revision": 1,
  "content_hash": "64-character-lowercase-sha256"
}
```

所有 `content_hash` 使用 RFC 8785 JSON Canonicalization Scheme（JCS）规范化后计算 SHA-256：

- `SafetyCapabilityProfile`、`PlannerAlgorithmConfig` 和 `BenchmarkProfile` 的 `content_ref.content_hash`，仅以同一文档的 `content` 对象为哈希输入，因此没有自引用。
- bundle 子组件的 `component_hash`，仅以组件的 `content` 对象为哈希输入。
- `bundle_hash` 以整个 bundle 为哈希输入，但计算前省略顶层 `bundle_hash`。
- 三个平台参考的 `reference_hash` 以整个平台参考为哈希输入，但计算前省略顶层 `reference_hash`。
- 其他 `ContentRef` 的哈希输入由该内容自身的 schema 定义；未定义时不得猜测或使用文件原始字节哈希。

JCS 输入必须是合法 I-JSON：不允许 NaN、正负无穷、重复对象键或依赖实现的数字表示。

## 时间表示

所有在线接口中的 int64 纳秒时刻和时长均编码为十进制字符串。这样 JavaScript、Python 和 C++ 不会因 IEEE-754 整数精度上限产生不同值。

```json
{
  "clock_id": "mission_monotonic",
  "tick_ns": "1723456789012345678"
}
```

C++ codec 将其严格解析为 `std::chrono::nanoseconds`，并拒绝：

- 超出 int64 范围；
- 前导 `+`；
- 非零值的前导零；
- 小数、指数或单位后缀。

相对时间区间使用 `start_offset_ns` 和 `end_offset_ns`。轮式/足式默认相对外层
`reference_time_origin`；Hopper flight-tube 和 landing window 显式相对
`JumpBoundary.ballistic_time_origin = BALLISTIC_LAUNCH_EVENT`。区间顺序由语义
校验器验证。

## 几何与时序真值

轮式和足式 `GeometricPath` 只能是：

1. `clamped_cubic_bspline`：明确给出三次 B 样条 knots 和 `[x,y,z,yaw]` 控制点；
2. `validated_primitive_chain`：明确引用能力 profile 中的运动原语及验证证书。

不得把任意采样点序列伪装成连续权威路径。
轮式 `STOP_AND_SWITCH` 是零相对位姿、非零时长的认证换向/停启原语；在
`validated_primitive_chain` 回退中必须保留其 kind、duration 和 validation ref。

`s(t)` 和自旋 `yaw(t)` 使用分段三次多项式。每段系数按局部时间
\(\tau=(t-t_0)\) 的秒数解释，在线区间端点仍使用纳秒十进制字符串。轮式显式速度和 yaw 角速度只能位于 `derived_caches`，并保持非权威。

飞跃式平面区域必须同时给出：

- `origin_m`；
- 单位 `normal`；
- 平面内正交基 `basis_u`、`basis_v`；
- `vertices_uv`；
- `winding = CCW`。

语义校验器必须验证基向量正交、右手性、凸性、顶点无自交以及所有相关多边形使用一致平面。

`HopperReference` 的着陆时间窗与线速度边界只保存在
`predicted_landing_footprint` 中。着陆 yaw 还必须满足
`footprint.landing_yaw_interval ⊆
attitude_boundary.target_attitude_set.allowed_yaw_interval ⊆
next_landing_region.allowed_yaw_interval`；任一圆周区间子集关系失败都拒绝整条引用。

`CircularYawInterval` 采用唯一规范形式：

```json
{
  "representation": "canonical_ccw",
  "start_rad": -1.0,
  "span_rad": 2.0,
  "closed": true
}
```

`start_rad` 归一化到 `[-pi, pi)`，`span_rad` 位于 `[0, 2*pi]`。不存在 `wrap` 布尔值。

## 确定性误差集合

状态误差和飞跃传播误差不是概率或信念状态。每个集合都必须带显式 `shape`：

- `axis_aligned_box`；
- `euclidean_ball`；
- `symmetric_interval`；
- `rotation_vector_ball`。

集合的单位由承载字段名确定，例如 `_m`、`_mps`、`_rad`、`_radps`。负半宽、负半轴或负半径必须拒绝；数组分量非负等 JSON Schema 难以表达的规则由 C++ 语义校验器执行。

## 运行时与基准隔离

`PlanningResponse` 不允许 `benchmark_profile_ref` 或 `benchmark_report_ref`。`call_diagnostics.api_latency_ns` 只是原始观测值，不包含目标比较或结果独立性判定。`P95 < 1 s` 只存在于 `BenchmarkProfile` 和 `BenchmarkReport`：

- 不构成规划截止时间；
- 不触发运行时降级；
- 不改变搜索终止、候选排序或 bundle 内容；
- 只在 `BenchmarkReport` 产生目标比较和实验验收结果。

`BenchmarkProfile.platform_suites` 与 `BenchmarkReport.platform_results` 均恰好包含
一个 `WHEELED`、一个 `LEGGED` 和一个 `HOPPER` 条目。每个 suite/result 自己携带
对应平台的 `safety_capability_ref`；report 顶层只携带三平台共享的
`algorithm_config_ref` 与 `benchmark_profile_ref`，其中 algorithm ref 从
`BenchmarkProfile.content.algorithm_config_ref` 原样复制。P95 字段统一命名为
`p95_latency_target_ns` 和 `p95_latency_target_met`。

只有 `NEW_REFERENCE_READY` 或 `SAFE_FRONTIER_REFERENCE_READY` 能与
`ACTIVATE_NEW_BUNDLE` 组合并发布 `new_reference_bundle`。只有
`CONTINUE_ACTIVE_BUNDLE` 或 `CONTINUE_COMMITTED_JUMP` 携带
`active_bundle_ref`；`ACTIVE_REFERENCE_INVALIDATED` 禁止继续活动 bundle。
`HOLD_STATIONARY` 不发布 `new_reference_bundle`。`SAFE_DEAD_END` 只能作为
`reason_code`，不是 `planning_outcome`，也不是飞跃图物理安全状态。

## 固定输入边界

请求只通过 `ContentRef` 绑定 `SafetyCapabilityProfile` 和 `PlannerAlgorithmConfig`。二者必须在调用前解析、校验并固定。

可选学习代价输入在请求中只携带 `LearnedCostSnapshotBinding`：

- `snapshot_ref` 绑定不可变内容版本与哈希；
- `registry_handle` 指向进程内已解析对象；
- `ready_before_request = true`；
- `hard_feasibility_authority = NONE`。

完整 `LearnedCostSnapshot` 不进入线协议，而是固定的 C++ registry 对象。它至少包含 snapshot/model/map revision、frame、feature contract、output contract、输出边界和定长向量，并必须在请求进入规划器前完成加载、哈希校验与合同校验：

- 仅能提供受限软代价；
- 不可用或输出越界时使用解析代价；
- 规划调用期间不得重新加载、在线更新或查询远程模型。

地图和既有 bundle 的 `_handle` 字段是进程内只读 registry key，不是内存地址、文件路径或远程 URL。

`PlannerAlgorithmConfig` 固定最大输入时间偏差 `max_input_skew_ns`、确定性误差模型 `error_bound_model_id` 和投影缓存容量 `projection_cache_capacity`；可选 `learned_cost_model_ref` 只允许与请求所绑定 snapshot 的 registry 内容一致。跨文档一致性由 C++ 语义校验器检查。

飞跃能力的 `landing_terrain_thresholds` 固定坡度、粗糙度、单平面残差、
顶/侧净空和非退化着陆域面积硬阈值。`attitude_envelope` 是认证的保守任意轴
基础包络；可选 `attitude_tightening_table_ref` 只能收紧，semantic validator
必须证明查表结果从不放宽基础包络。飞跃算法配置分别固定图节点、yaw 分区、
支撑方向、着陆域切分、区间求根、碰撞细分和飞行管 section 上限，不能用一个
模糊上限代替多个不同终止量。

## 原子 bundle 与组件

`bundle_id + bundle_revision` 唯一标识可激活 bundle。只有 bundle 能被激活、替换或回滚。

`route_skeleton`、`committed_prefix` 和 `preview` 各带 `component_id + component_hash`，但组件不能单独激活。两种执行视图只包含对同一个内联 `platform_reference` 的 selector，并通过 `source_reference_id + source_reference_hash` 绑定，避免形成第二份权威路径。

飞跃 `JUMP_READY` bundle 的 committed prefix 使用 `GROUND_HOLD` selector 指向
同一 `HopperReference.ground_hold_anchor`，preview 使用
`JUMP_BOUNDARY/NEXT_HOP` 指向唯一边界。ground-hold anchor 的线速度和角速度
必须为零且带地形证书；边界锁定状态只进入既有执行上下文，不修改 bundle 内容。
锁定和发射不属于 whole-bundle invalidation condition；它们关闭替换资格，但
`JUMP_COMMITTED/IN_FLIGHT` 仍通过 `CONTINUE_COMMITTED_JUMP` 使用同一 bundle。

`generation_evidence` 只记录候选 ID、生成模式、稳定终止原因和可内容寻址证据；ARA* 图、走廊、QP 工作集和投影 cache 等过程内结构不进入运行时接口。

本次 API 调用的 `call_diagnostics` 只属于 `PlanningResponse`；bundle 不复制调用诊断，也不记录 benchmark 结果。

语义校验器必须进一步验证：

- 子组件源 reference 与 bundle 内 reference 完全一致；
- `committed_prefix.content.role = COMMITTED_PREFIX`；
- `preview.content.role = PREVIEW`；
- Hopper ground-hold selector 只出现在 committed prefix 且锚点 ID、零速度和
  地形证书与同一 reference 一致；
- bundle 源地图必须逐字段等于 validity 地图和 Hopper flight-tube 源地图；
  bundle 源能力必须逐字段等于 validity 能力，算法配置必须等于本次请求绑定；
- `response.request_id`、`bundle.source_request_id` 与激活请求 ID 相同，平台、
  frame 和 reference-time provenance 也必须与该请求一致；
- Hopper boundary 的 gravity、actuator/impulse、body-rotation envelope 和
  footprint/tube error model 必须命中本次请求开始前固定的已解析能力依赖；
- 请求当前 Hopper 状态集合必须包含于 ground-hold 允许集合，anchor 与 launch
  boundary 的位置/姿态连续，速度跃迁及误差映射由固定 actuator/impulse profile
  认证；
- `ballistic_flight_time_ns > 0` 且位于能力 min/max 内；tube 的 offset 相对
  `BALLISTIC_LAUNCH_EVENT` 连续覆盖 `[0,T]`，landing window 包含 `T`；
- `USED_BOUNDED_SOFT_COST` 时 generation evidence 必须携带与请求逐字段相同的
  learned snapshot ref；`DISABLED/FELL_BACK_TO_ANALYTIC` 时必须不携带；
- 每个 terrain/physical/attitude/validation-summary certificate 必须解析为带
  `CertificationProvenance` 的只读证书对象；其 map/capability/config 和
  gravity/error/actuator/body-envelope 输入必须与本次激活上下文完全一致；
- 不混用其他 bundle 的组件；
- 允许复用旧 committed component 时，其 ID、hash 和 content 必须全部不变。

## C++ 映射边界

推荐手写不可变合同类型和显式 codec，不从 JSON Schema 直接生成核心规划类型：

```cpp
Result<PlanningRequest> JsonCodec::DecodePlanningRequest(
    std::string_view payload,
    const ContractObjectRegistry& registry);

Result<PlanningResponse> JsonCodec::DecodePlanningResponse(
    std::string_view payload,
    const ReferenceActivationContext& activation_context);

std::string JsonCodec::EncodePlanningRequest(
    const PlanningRequest& request);

std::string JsonCodec::EncodePlanningResponse(
    const PlanningResponse& response);

ValidationReport SemanticValidator::Validate(
    const PlanningRequest& request) const;

ValidationReport SemanticValidator::Validate(
    const PlanningResponse& response) const;

ValidationReport SemanticValidator::ValidateForActivation(
    const ReferenceBundle& bundle,
    const ReferenceActivationContext& context) const;

ValidationReport SemanticValidator::ValidateForActivation(
    const PlanningResponse& response,
    const ReferenceActivationContext& context) const;
```

`Result<T>` 固定为 `std::variant<T, Error>`，不再引入第二套 result 类型。
平台多态使用 `std::variant`，可选字段使用 `std::optional`，纳秒包装为
`DurationNanoseconds{std::chrono::nanoseconds}`。codec 负责字段、枚举、数字类型、
未知字段和 schema version；semantic validator 负责跨字段、几何、时间、哈希和安全不变量。

运行时不从磁盘加载 schema，也不通过通用 schema 引擎反射构造规划对象。
无 `ReferenceActivationContext` 的局部 `Validate` 不能授予执行权；只有
`ValidateForActivation` 完成 registry 解析、provenance join、能力边界和状态连续性
校验后，仲裁器才可返回 `ACTIVATE_NEW_BUNDLE`。

## 必须由语义校验器补充的规则

JSON Schema 结构校验通过后，至少还要验证：

- 所有 JSON 数字有限；
- ContentRef、bundle、reference 和组件哈希正确；
- 请求 frame、地图 frame、能力 frame 和状态 frame 绑定一致；
- `platform_type` 与 state/reference/profile variant 一致；
- `request_time`、`state_time` 与地图 `source_time` 的 clock 一致，输入偏差不超过 `max_input_skew_ns`；
- 时间区间有序且多项式连续；
- B 样条 knot/control-point 数量、clamping 和参数域正确；
- `s(t)` 连续、单调、从 0 到 1；
- 轮式前进/倒车/自旋模式切换速度边界；
- 足式固定参考点属于对应能力 profile；
- `footstep_feasibility_guaranteed = false`；
- 飞跃姿态四元数归一化；
- 可选姿态查表相对认证任意轴基础包络只收紧不放宽；
- 能力 `minimum_flight_time_ns <= maximum_flight_time_ns`，maximum 必须正；
- `JumpBoundary.ballistic_flight_time_ns` 必须正且位于能力区间内；
- per-call learned usage 为 `DISABLED` 时 diagnostics snapshot ref 不得存在；
  `USED_BOUNDED_SOFT_COST` 或 `FELL_BACK_TO_ANALYTIC` 时必须等于请求固定 ref，
  即使本次未生成 bundle；
- 飞行管时间覆盖完整且按序；
- 落点外包加安全裕量包含在 `NextLandingRegion`；
- `JumpBoundary` 是唯一下一跳命令；
- `committed_prefix` 和 `preview` 是同一权威 reference 的视图；
- `HOLD_STATIONARY` 不携带新 bundle；
- benchmark 目标不影响运行时结果。

## 离线 Schema 校验

校验器必须预注册本目录全部 URN；不得尝试从网络解析 `$ref`。推荐在 CI 中使用 Python `jsonschema` 的 `Draft202012Validator`：

```powershell
$env:PYTHONPATH = "path-planner/src"
python -m pytest -q path-planner/tests/test_v3_schema.py
```

同一组 valid/invalid fixtures 应由 Python Draft 2020-12 校验器和 C++ codec/semantic validator 共同消费，防止两套合同漂移。
