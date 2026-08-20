# super_ik_tsinghua_1dqp

基于清华论文《针对关节限位优化的 7 自由度机械臂逆运动学解法》(胡奎等,清华大学学报 2020,60(12):1007-1015)的 **7-DoF S-R-S 机械臂解析逆运动学库**。

臂角参数化闭式解 + 解析可行域 + 加权二次优化避限位;C++17 核心 + pybind11 绑定 + ROS2 实时桥接。

## 特性

- **臂角 ψ 参数化闭式 IK**:S-R-S 构型,8 分支解由分支恒等式$(\sigma_4, \sigma_2, \sigma_6)$精确区分,单支求解计算量为全枚举的$\frac{1}{8}$
- **解析可行域**:由旋转恒等式 $R_3(\psi) = R_{\psi−\psi_0}\cdot R_3(\psi_0)$构造关节模型(零采样),cos 型/atan2 型关节的限位→臂角映射用事件-单元法**精确求根**(论文式 13-17),非采样近似
- **加权二次优化避限位**(论文式 19-33):权重函数 $w(x)=\frac{bx}{(e^{a(1−x)}−1)}$(a=2.38, b=2.28),差分线性化 + 一维二次函数 $\psi_{opt}=−\frac{B}{2A}$ + 可行域投影
- **连续性双路径选解**:目标静止沿 $q_{prev}$ 臂角展开(最优区域重合);目标连续移动沿锚点 $\psi_{prev}$ 展开;目标跳变/冷启动走 **8 分支解析全局重规划**(跨分支择优)
- **实时性能**:单点 0.017 ms(论文宣称 <0.1ms 的 6 倍),1000 随机可达目标 100% 成功、机器精度位姿误差、0% 限位违规

## 性能(Benchmark,`examples/benchmark.py`)

单点 IK(1000 随机可达目标,ψ_prev=0 冷启动最坏场景):

| 指标 | 值 |
|---|---|
| 成功率 | 1000/1000 (100%) |
| 时延 mean / p50 | 0.398 / 0.386 ms |
| 时延 p90 / p99 / max | 0.663 / 0.790 / 0.844 ms |
| 吞吐 | 2368 Hz |
| 位姿误差(Frobenius)mean | 2.2e-15(机器精度) |

连续轨迹(200 点,可达点):avg 0.012 ms/点,最大关节跳变 0.047 rad。
对比:论文宣称单点 <0.1 ms,本实现连续性场景中位数 ~0.03 ms(轨迹传续时远快于上述冷启动最坏值)。

## 构建

依赖:Eigen3、pybind11(仅 Python 绑定)、CMake ≥ 3.16、C++17。

```bash
# C++ 库 + demo + 测试
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
ctest --test-dir build            # 4 个测试套件

# Python 绑定 (可选)
pip install .
```

## 使用

### C++

```cpp
#include "super_ik_tsinghua_1dqp/solver.hpp"
#include "super_ik_tsinghua_1dqp/analytic.hpp"

super_ik_tsinghua_1dqp::Solver solver;
Eigen::Matrix4d target;                 // 目标位姿 (link7 系)
Eigen::Matrix<double, 7, 1> q_prev;     // 上一时刻关节角
double psi_prev = 0.0;                  // 上一时刻臂角 (首帧可传任意值)

auto res = solver.IkArmAngle(target, q_prev, psi_prev);
// res.q_best: 最优解 (恒在限位内); res.report: 可行域/ψ_opt/quad 系数等诊断
```

公共解析 API(`analytic.hpp`):`BuildJointModels` / `CosJointFeasibleIntervals` / `AtanJointFeasibleIntervals` / `ClosedFormIkAll` / `ClosedFormIkBranch` / `RecoverArmAngle` / `BranchOf`。

### Python

```python
from super_ik_tsinghua_1dqp import Solver, PaperParams

solver = Solver()                      # 默认 Nero DH 参数
T = solver.fk(q)                       # 正运动学
out = solver.solve_trajectory([T], q_prev, psi_init=0.0)
q_best = out["q_list"][0]              # 最优解
```

### ROS2 实时仿真(RUN.md 流程)

仿真包为独立 submodule,克隆时需 `--recursive`(或事后 `git submodule update --init`):

```bash
git clone --recursive https://github.com/vanstrong12138/robotic_arm_kinematics.git
# 仿真包位于 sim_ws/src/agx_arm_sim (agilexrobotics/agx_arm_sim)

# 1. 编译仿真包
cd sim_ws && colcon build && source install/setup.bash

# 2. 仿真显示 (use_gui:=false 避免 joint_state_publisher 与 IK 桥接竞争 /joint_states)
ros2 launch agx_arm_description display.launch.py arm_type:=nero use_gui:=false

# 3. IK 桥接 (发布 /joint_states)
python3 examples/ik_joint_state_publisher_1dqp.py

# 4. 交互式拖拽目标 (rviz 中拖动 marker)
python3 examples/interactive_target_marker.py
```

拖动测试:工作空间为以基座为球心半径 0.58 m 的球(腕心),超出范围 `IK failed` 属正常(正确拒绝不可达目标)。

### Benchmark

```bash
python3 examples/benchmark.py   # 单点 IK + 连续轨迹基准
```

## 算法说明

1. **闭式 IK**(论文 1.2 节):给定目标位姿求 S/W/$\theta_4$(余弦定理),臂角$\psi$决定肘点 E 在 SW 轴圆上的位置,几何法解$q_1\dots q_3$、矩阵法解$q_5\dots q_7$,共 8 支解 =$(\sigma_4, \sigma_2, \sigma_6)$三比特。
2. **解析可行域**(论文 2 节):$R_3(\psi) = A_s\sin{\psi} + B_s\cos{\psi} + C_s$ ($A_s$/$B_s$/$C_s$ 由种子处一次 FK + 反对称阵构造),各关节角与$\psi$的映射由此解析提取;$\cos$型用$\sigma$分支符号传播($\sigma$在$\cos{\theta}=±1$处翻转),$atan2$型用导数零点(论文式 16)+ 分支切割 + 奇异点 + 限位水平线四类事件精确求根;全部 7 关节区间求交(论文式 31-32)。
3. **加权二次优化**(论文 3 节):目标函数 $\sum w_i(x_i)(q_i(\psi)−q_{i,t})^2$经差分线性化$(\Delta \psi=1e-3)$变为二次函数,$\psi_{opt}=−\frac{B}{2A}$,投影到可行域。
4. **选解**:分支恒等式$(\sigma_4, \sigma_2, \sigma_6)$作为分支"身份证";连续性判定(位姿匹配 + 臂角连续)选择局部路径(沿$q_{prev}$臂角展开)或全局路径(8 分支解析重规划)。

## 项目结构

```
src/solver.cpp                     核心: 解析模型/区间/选解主流程
include/super_ik_tsinghua_1dqp/
  analytic.hpp                     公共解析 API (模型/区间/分支 IK/臂角反解)
  solver.hpp types.hpp             Solver 接口与数据结构
tests/                             4 个测试套件
examples/
  ik_joint_state_publisher_1dqp.py ROS2 桥接 (marker → IK → joint_states)
  interactive_target_marker.py     交互式拖拽目标
  benchmark.py                     性能基准
  python_demo.py                   快速示例
app/main.cpp                       C++ demo
doc/benchmark_comparison.md        与 pinocchio-lite 的对比
```

## 已知限制

- 解析模型恒等式针对本仓库默认 DH$(\alpha_1\dots \alpha_6 = \frac{\pi}{2})$;其他 DH 自动回退采样+拟合路径
- 冷启动(目标跳变)走全局路径,单点 ~0.4 ms(1ms 控制周期内)
- rviz 可视化在无 GPU 的远程/虚拟显示环境可能不渲染模型(数据流正常)
