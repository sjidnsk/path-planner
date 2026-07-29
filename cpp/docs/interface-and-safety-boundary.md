# 路径规划 v3 C++ 接口与安全边界

## 公开入口

应用层通过 `lunar::planning::v3::PlannerV3::Plan(const PlanningRequest&)`
调用统一规划器。`MakeDefaultPlannerV3` 组装共享核心与轮式、足式、飞跃式三个
平台层；请求中的不可变地图、安全能力和算法配置均由强引用绑定。

安装后的 CMake package 名为 `LunarPathPlannerV3`，公开目标为：

- `LunarPathPlannerV3::contracts`
- `LunarPathPlannerV3::common`
- `LunarPathPlannerV3::wheel`
- `LunarPathPlannerV3::legged`
- `LunarPathPlannerV3::hopper`
- `LunarPathPlannerV3::api`

## 平台输出

- 轮式平台输出带时序的 `x/y/yaw` 几何与速度参考，允许前进、倒车和原地旋转。
- 足式平台只输出机体/质心 `x/y/z/yaw` 与时序参考；落足、步态和接触力由下游
  运动控制器决定，本模块不声明足步可行。
- 飞跃式平台只认证常量局部重力下的下一次纯弹道飞跃；输出确定性凸着陆范围、
  唯一起跳边界、四元数姿态边界和角速度边界。发射后不能重定向，后续飞跃仅为
  预览。

## 执行与性能边界

- C++ v3 是独立 opt-in 实现，不替换既有 Python 默认 A*。
- 本模块不连接 executor，不发布 checkpoint，不启动 canary。
- 只有通过激活校验的 READY 响应可携带新 reference bundle；
  `HOLD_STATIONARY` 不携带 bundle。
- 规划器内部没有 1 秒 deadline、超时中断或基于剩余时间的语义分支。
  “P95 小于 1 秒”仅由 Release benchmark 在全部调用自然结束后统计。
