#pragma once

// 解析臂角-关节模型(论文《针对关节限位优化的 7 自由度机械臂逆运动学解法》第 2 节方法)。
//
// 论文做法:给定末端位姿后,0R3(ψ) = As·sinψ + Bs·cosψ + Cs(式 5)、
// 0R7(ψ) = Aw·sinψ + Bw·cosψ + Cw(式 9),各关节角与臂角的映射由此解析提取:
//   cos 型关节(θ2、θ6):cosθi = a·sinψ + b·cosψ + c
//   atan2 型关节(θ1、θ3、θ5、θ7):θi = atan2(an·sinψ + bn·cosψ + cn,
//                                            ad·sinψ + bd·cosψ + cd)
// 本实现按本仓库 DH 约定(α1..α6 = π/2)数值验证的恒等式解析构造这些系数,
// 不做任何采样拟合;恒等式在校验失败时(非本 DH 构型)由调用方回退。

#include "super_ik_tsinghua_1dqp/types.hpp"

#include <array>
#include <utility>
#include <vector>

namespace super_ik_tsinghua_1dqp {

// 单个关节的解析模型。is_cos 为 true 时用 a/b/c,否则用 an..cd。
struct JointModel {
  bool is_cos { false };
  double a { 0.0 }, b { 0.0 }, c { 0.0 };
  double an { 0.0 }, bn { 0.0 }, cn { 0.0 };
  double ad { 0.0 }, bd { 0.0 }, cd { 0.0 };
  // 分支在锚点 ψ0 处的关节角(θ 域,含 theta_offset),用于常数型/校验。
  double theta_ref { 0.0 };
  // 锚点臂角 ψ0(σ 分支符号传播的起点)。
  double psi_ref { 0.0 };
};

// 7 个关节的模型集合。ok=false 表示 DH 构型不匹配本解析恒等式。
struct JointModelSet {
  std::array<JointModel, 7> joint;
  bool ok { false };
};

// 由种子分支 (psi0, q0) 解析构造全部 7 个关节的模型。
// q0 必须是 target 在 psi0 处的闭式 IK 解(分支锚点),通常来自
// FindNearestFeasibleSeed 的结果。
JointModelSet BuildJointModels(
    const Mat4& target, double psi0, const Vec7& q0, const NeroParams& p);

// 关节 i 的 θ 域限位:[q_lo + theta_offset, q_hi + theta_offset]。
std::pair<double, double> ThetaLimits(const NeroParams& p, int i);

// cos 型关节的精确可行臂角区间:{ψ ∈ (−π, π] : cosθ(ψ) ∈ cos([th_lo, th_hi])}。
std::vector<std::pair<double, double>> CosJointFeasibleIntervals(
    const JointModel& m, double th_lo, double th_hi);

// atan2 型关节的精确可行臂角区间:{ψ ∈ (−π, π] : atan2(u(ψ), v(ψ)) ∈ [th_lo, th_hi]},
// 限位区间按 2π 周期性拆分为至多两段弧。
std::vector<std::pair<double, double>> AtanJointFeasibleIntervals(
    const JointModel& m, double th_lo, double th_hi);

// 分支恒等式: 闭式 IK 的 8 支解 = (σ4, σ2, σ6) 三个二值位的组合。
//   σ4 = sign(q4)        肘部上/下 (q4 = ±θ4_abs)
//   σ2 = sign(sinθ2)     肩上/下
//   σ6 = sign(sinθ6)     腕上/下
// 非奇异点上每支恒等分支的解存在且唯一, 是"选解 = 查表"的基础。
struct BranchId {
  int s4 {0};
  int s2 {0};
  int s6 {0};
  bool operator==(const BranchId& o) const {
    return s4 == o.s4 && s2 == o.s2 && s6 == o.s6;
  }
};

// 从闭式解 q 读出其分支恒等式。
BranchId BranchOf(const Vec7& q, const NeroParams& p);

// 调试/测试辅助:给定臂角 ψ 的闭式 IK 全部分支解(不过滤关节限位)。
std::vector<Vec7> ClosedFormIkAll(const Mat4& target, double psi, const NeroParams& p);

// 指定恒等分支的闭式 IK(单支求解, 内部枚举按分支剪枝, 计算量约为全枚举的 1/8)。
// 分支不存在(退化/奇异)时返回空。
std::optional<Vec7> ClosedFormIkBranch(
    const Mat4& target, double psi, const NeroParams& p, const BranchId& branch);

// 由闭式解 q 反解其臂角 ψ(肘心绕 SW 轴的角度, 论文 3.1 节"初始位置反解臂角")。
// 用于轨迹初始化: SolveTrajectory 的 psi_init 应传入首姿态自身的臂角。
double RecoverArmAngle(const Mat4& target, const Vec7& q, const NeroParams& p);

// 角度回绕到 (−π, π]。
double WrapToPi(double x);
Vec7 WrapToPi(const Vec7& q);

}  // namespace super_ik_tsinghua_1dqp
