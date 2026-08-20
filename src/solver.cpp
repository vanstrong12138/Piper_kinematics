#include "super_ik_tsinghua_1dqp/solver.hpp"
#include "super_ik_tsinghua_1dqp/analytic.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>

namespace super_ik_tsinghua_1dqp {

constexpr double kPi = 3.14159265358979323846;
constexpr double kMaxWeight = 1e6;

double WrapToPi(double x) {
  double y = std::fmod(x + kPi, 2.0 * kPi);
  if (y < 0.0) y += 2.0 * kPi;
  return y - kPi;
}

Vec7 WrapToPi(const Vec7& q) {
  Vec7 out;
  for (int i = 0; i < 7; ++i) out[i] = WrapToPi(q[i]);
  return out;
}

namespace {

struct Interval {
  double start {};
  double end {};
};

struct BranchSamples {
  std::vector<double> psi;
  std::vector<Vec7> q;
};

struct CosPsiModel {
  double a {0.0};
  double b {0.0};
  double c {0.0};
  bool ok {false};
};

struct AtanPsiModel {
  double an {0.0};
  double bn {0.0};
  double cn {0.0};
  double ad {0.0};
  double bd {0.0};
  double cd {0.0};
  bool ok {false};
};

struct LinearizedModel {
  bool ok {false};
  double psi0 {0.0};
  Vec7 q0 {Vec7::Zero()};
  Vec7 k {Vec7::Zero()};
};

Eigen::Matrix3d RotX(double t) {
  const double c = std::cos(t);
  const double s = std::sin(t);
  Eigen::Matrix3d r;
  r << 1.0, 0.0, 0.0,
       0.0, c, -s,
       0.0, s, c;
  return r;
}

Eigen::Matrix3d RotZ(double t) {
  const double c = std::cos(t);
  const double s = std::sin(t);
  Eigen::Matrix3d r;
  r << c, -s, 0.0,
       s, c, 0.0,
       0.0, 0.0, 1.0;
  return r;
}

Mat4 DhA(double theta, double d, double a_prev, double alpha_prev) {
  Mat4 t = Mat4::Identity();
  t.block<3, 3>(0, 0) = RotX(alpha_prev) * RotZ(theta);
  t.block<3, 1>(0, 3) = Eigen::Vector3d(
      a_prev, -std::sin(alpha_prev) * d, std::cos(alpha_prev) * d);
  return t;
}

double PoseErrorNorm(const Mat4& cur, const Mat4& des) {
  const Eigen::Vector3d dp = cur.block<3, 1>(0, 3) - des.block<3, 1>(0, 3);
  const Eigen::Matrix3d r = cur.block<3, 3>(0, 0);
  const Eigen::Matrix3d rd = des.block<3, 3>(0, 0);
  const Eigen::Vector3d re = 0.5 * (
      r.col(0).cross(rd.col(0)) +
      r.col(1).cross(rd.col(1)) +
      r.col(2).cross(rd.col(2)));
  Eigen::Matrix<double, 6, 1> e;
  e << dp, re;
  return e.norm();
}

bool WithinLimits(const Vec7& q, const Eigen::Matrix<double, 7, 2>& limits) {
  for (int i = 0; i < 7; ++i) {
    if (q[i] < limits(i, 0) - 1e-8 || q[i] > limits(i, 1) + 1e-8) return false;
  }
  return true;
}

void ComputeSweFromTarget(
    const Mat4& t07, const NeroParams& p,
    Eigen::Vector3d* s, Eigen::Vector3d* w, double* q4_abs) {
  const Eigen::Matrix3d r = t07.block<3, 3>(0, 0);
  const Eigen::Vector3d p_target = t07.block<3, 1>(0, 3);
  const Eigen::Vector3d z7 = r.col(2);
  const Eigen::Vector3d o7 = p_target - p.post_transform_d8 * z7;
  *w = o7 - p.d_i[6] * z7;
  *s = Eigen::Vector3d(0.0, 0.0, p.d_i[0]);

  const double l_sw = (*w - *s).norm();
  const double l_se = std::abs(p.d_i[2]);
  const double l_ew = std::abs(p.d_i[4]);
  if (l_sw < 1e-10) {
    *q4_abs = std::numeric_limits<double>::quiet_NaN();
    return;
  }
  const double c4 = (l_sw * l_sw - l_se * l_se - l_ew * l_ew) / (2.0 * l_se * l_ew);
  if (c4 < -1.0 || c4 > 1.0) {
    *q4_abs = std::numeric_limits<double>::quiet_NaN();
    return;
  }
  *q4_abs = std::acos(std::clamp(c4, -1.0, 1.0));
}

std::optional<Eigen::Vector3d> ElbowFromArmAngle(
    const Eigen::Vector3d& s,
    const Eigen::Vector3d& w,
    double psi,
    const NeroParams& p) {
  const double l_se = std::abs(p.d_i[2]);
  const double l_ew = std::abs(p.d_i[4]);
  const Eigen::Vector3d sw = w - s;
  const double l_sw = sw.norm();
  if (l_sw < 1e-12) return std::nullopt;
  const Eigen::Vector3d u_sw = sw / l_sw;
  const double x = (l_se * l_se - l_ew * l_ew + l_sw * l_sw) / (2.0 * l_sw);
  const double r2 = l_se * l_se - x * x;
  if (r2 < -1e-10) return std::nullopt;
  const double rc = std::sqrt(std::max(0.0, r2));
  const Eigen::Vector3d c = s + x * u_sw;

  Eigen::Vector3d t = s.cross(u_sw);
  if (t.norm() < 1e-10) t = Eigen::Vector3d(1.0, 0.0, 0.0).cross(u_sw);
  if (t.norm() < 1e-10) t = Eigen::Vector3d(0.0, 1.0, 0.0).cross(u_sw);
  const Eigen::Vector3d e1 = t.normalized();
  const Eigen::Vector3d e2 = u_sw.cross(e1).normalized();
  return c + rc * (std::cos(psi) * e1 + std::sin(psi) * e2);
}

std::vector<Vec7> SolveQ123FromSwe(
    const Eigen::Vector3d& e,
    const Eigen::Vector3d& w,
    double q4,
    const NeroParams& p,
    std::optional<int> sigma2 = std::nullopt) {
  std::vector<Vec7> out;
  const double d0 = p.d_i[0];
  const double d2 = p.d_i[2];
  const double d4 = p.d_i[4];
  if (std::abs(d2) < 1e-12 || std::abs(d4) < 1e-12) return out;

  const double ex = e.x();
  const double ey = e.y();
  const double ez = e.z();
  const double rho = std::hypot(ex, ey);
  double c2 = (ez - d0) / d2;
  if (c2 < -1.0 - 1e-8 || c2 > 1.0 + 1e-8) return out;
  c2 = std::clamp(c2, -1.0, 1.0);
  const double s2_abs = std::sqrt(std::max(0.0, 1.0 - c2 * c2));
  if (rho > std::abs(d2) + 1e-7) return out;

  Eigen::Vector3d col2 = -(w - e) / d4;
  if (col2.norm() < 1e-10) return out;
  col2.normalize();
  const double u1 = col2.x();
  const double u2 = col2.y();
  const double u3 = col2.z();
  const double s4 = std::sin(q4);
  const double c4 = std::cos(q4);
  if (std::abs(s4) < 1e-8) return out;

  for (double s2 : {s2_abs, -s2_abs}) {
    if (std::abs(s2) < 1e-10) continue;
    if (sigma2.has_value()) {
      // 注: SolveQ123FromSwe 返回的 q2 是关节角 (主循环加 theta_offset 才得
      // DH 角 θ2)。σ2 = sign(sinθ2) = sign(sin(q2 + theta_offset[1]))。
      const double q2 = std::atan2(s2, c2);
      if ((std::sin(q2 + p.theta_offset[1]) >= 0.0 ? 1 : -1) != *sigma2) continue;
    }
    double c1 = -ex / (d2 * s2);
    double s1 = -ey / (d2 * s2);
    const double n1 = std::hypot(c1, s1);
    if (n1 < 1e-12) continue;
    c1 /= n1;
    s1 /= n1;
    const double q1 = std::atan2(s1, c1);
    const double q2 = std::atan2(s2, c2);

    const double b1 = (s2 * c1 * c4 - u1) / s4;
    const double b2 = (u2 - s1 * s2 * c4) / s4;
    double s3 = s1 * b1 + c1 * b2;
    double c3 = (std::abs(c2) > 1e-8) ? ((-c1 * b1 + s1 * b2) / c2) : ((u3 + c2 * c4) / (s2 * s4));
    const double n3 = std::hypot(s3, c3);
    if (n3 < 1e-12) continue;
    s3 /= n3;
    c3 /= n3;
    Vec7 q = Vec7::Zero();
    q[0] = q1;
    q[1] = q2;
    q[2] = std::atan2(s3, c3);
    out.push_back(q);
  }
  return out;
}

std::vector<Eigen::Vector3d> Extract567(
    const Mat4& t47, const NeroParams& p, std::optional<int> sigma6 = std::nullopt) {
  std::vector<Eigen::Vector3d> out;
  const double c6 = std::clamp(t47(1, 2), -1.0, 1.0);
  for (double sgn : {1.0, -1.0}) {
    const double s6 = sgn * std::sqrt(std::max(0.0, 1.0 - c6 * c6));
    if (std::abs(s6) < 1e-8) continue;
    if (sigma6.has_value()) {
      // 注: Extract567 返回的 q6 = atan2(s6, c6) 即 DH 角 θ6 (主循环减
      // theta_offset 才得关节角)。σ6 = sign(sinθ6) = sign(s6)。
      if ((s6 >= 0.0 ? 1 : -1) != *sigma6) continue;
    }
    out.emplace_back(
        std::atan2(t47(2, 2) / s6, t47(0, 2) / s6),
        std::atan2(s6, c6),
        std::atan2(t47(1, 1) / s6, -t47(1, 0) / s6));
  }
  return out;
}

Vec7 NormalizeJointPosition(const Vec7& q, const NeroParams& p) {
  Vec7 x = Vec7::Zero();
  for (int i = 0; i < 7; ++i) {
    const double ql = p.joint_limits(i, 0);
    const double qu = p.joint_limits(i, 1);
    x[i] = 2.0 * (q[i] - (qu + ql) * 0.5) / (qu - ql);
  }
  return x;
}

double WeightFromNormalized(double x) {
  constexpr double a = 2.38;
  constexpr double b = 2.28;
  if (x >= 0.0) {
    if (x >= 1.0) return kMaxWeight;
    return std::min(kMaxWeight, b * x / (std::exp(a * (1.0 - x)) - 1.0));
  }
  if (x <= -1.0) return kMaxWeight;
  return std::min(kMaxWeight, -b * x / (std::exp(a * (1.0 + x)) - 1.0));
}

Vec7 WeightLimitsFromPrev(const Vec7& q_prev, const NeroParams& p) {
  const Vec7 x_prev = NormalizeJointPosition(q_prev, p);
  Vec7 w = Vec7::Zero();
  for (int i = 0; i < 7; ++i) w[i] = WeightFromNormalized(x_prev[i]);
  return w;
}

std::vector<int> DangerJointIndices(const Vec7& q_prev, const NeroParams& p, double danger_threshold) {
  std::vector<int> idx;
  const Vec7 x_prev = NormalizeJointPosition(q_prev, p);
  for (int i = 0; i < 7; ++i) {
    if (std::abs(x_prev[i]) >= danger_threshold) idx.push_back(i);
  }
  return idx;
}

double DistanceToReference(const Vec7& q, const Vec7& q_ref) {
  return WrapToPi(q - q_ref).norm();
}

// 闭式 IK 全部分支解(不过滤关节限位)——解析关节模型构造与暴力真值校验使用。
// 可选分支过滤: 内部枚举按 (σ4, σ2, σ6) 三级剪枝, 计算量约为全枚举的 1/8。
std::vector<Vec7> IkOneArmAngleUnfiltered(
    const Mat4& target, double psi, const NeroParams& p,
    const std::optional<BranchId>& branch = std::nullopt) {
  std::vector<Vec7> sols;
  Eigen::Vector3d s, w;
  double q4_abs = 0.0;
  ComputeSweFromTarget(target, p, &s, &w, &q4_abs);
  if (!std::isfinite(q4_abs)) return sols;
  const auto e = ElbowFromArmAngle(s, w, psi, p);
  if (!e.has_value()) return sols;

  Mat4 t_post = Mat4::Identity();
  t_post(2, 3) = p.post_transform_d8;
  const Mat4 t_chain = target * t_post.inverse();

  std::vector<double> q4s{q4_abs, -q4_abs};
  if (branch.has_value()) q4s = {branch->s4 > 0 ? q4_abs : -q4_abs};
  const std::optional<int> sigma2 = branch.has_value()
      ? std::optional<int>(branch->s2)
      : std::nullopt;
  const std::optional<int> sigma6 = branch.has_value()
      ? std::optional<int>(branch->s6)
      : std::nullopt;

  for (double q4 : q4s) {
    auto q123_sols = SolveQ123FromSwe(*e, w, q4, p, sigma2);
    for (const auto& q123 : q123_sols) {
      const double th1 = q123[0] + p.theta_offset[0];
      const double th2 = q123[1] + p.theta_offset[1];
      const double th3 = q123[2] + p.theta_offset[2];
      const double th4 = q4 + p.theta_offset[3];
      Mat4 t04 = Mat4::Identity();
      const double th_raw[4] = {th1, th2, th3, th4};
      for (int i = 0; i < 4; ++i) t04 = t04 * DhA(th_raw[i], p.d_i[i], p.a_prev[i], p.alpha_prev[i]);
      const auto q567_sols = Extract567(t04.inverse() * t_chain, p, sigma6);
      for (const auto& q567 : q567_sols) {
        Vec7 theta_raw;
        theta_raw << th1, th2, th3, th4, q567[0], q567[1], q567[2];
        Vec7 q = WrapToPi(theta_raw - p.theta_offset);
        sols.push_back(q);
      }
    }
  }
  return sols;
}

std::vector<Vec7> IkOneArmAngleInternal(const Mat4& target, double psi, const NeroParams& p) {
  std::vector<Vec7> sols;
  for (const auto& q : IkOneArmAngleUnfiltered(target, psi, p)) {
    if (WithinLimits(q, p.joint_limits)) sols.push_back(q);
  }
  return sols;
}

std::optional<Vec7> SelectBranchNearest(const Mat4& target, double psi, const Vec7& q_ref, const NeroParams& p) {
  const auto sols = IkOneArmAngleInternal(target, psi, p);
  if (sols.empty()) return std::nullopt;
  int best_idx = 0;
  double best_cost = std::numeric_limits<double>::infinity();
  for (int i = 0; i < static_cast<int>(sols.size()); ++i) {
    const double cost = DistanceToReference(sols[i], q_ref);
    if (cost < best_cost) {
      best_cost = cost;
      best_idx = i;
    }
  }
  return sols[best_idx];
}

std::optional<std::pair<double, Vec7>> FindNearestFeasibleSeed(
    const Mat4& target,
    double psi_ref,
    const Vec7& q_ref,
    const NeroParams& p,
    const PaperParams& paper) {
  auto q = SelectBranchNearest(target, psi_ref, q_ref, p);
  if (q.has_value()) return std::make_pair(psi_ref, *q);

  std::optional<std::pair<double, Vec7>> best;
  double best_cost = std::numeric_limits<double>::infinity();
  const int kSamples = std::max(9, paper.psi_seed_samples);
  for (int i = 0; i < kSamples; ++i) {
    const double psi = -kPi + (2.0 * kPi * static_cast<double>(i)) / static_cast<double>(kSamples - 1);
    auto cand = SelectBranchNearest(target, psi, q_ref, p);
    if (!cand.has_value()) continue;
    const double cost = std::abs(WrapToPi(psi - psi_ref)) + 0.1 * DistanceToReference(*cand, q_ref);
    if (cost < best_cost) {
      best_cost = cost;
      best = std::make_pair(psi, *cand);
    }
  }
  return best;
}

std::vector<double> BuildPsiGrid(double step) {
  std::vector<double> psi;
  for (double p = -kPi; p <= kPi + 1e-12; p += step) psi.push_back(std::min(kPi, p));
  if (psi.empty() || std::abs(psi.back() - kPi) > 1e-9) psi.push_back(kPi);
  return psi;
}

BranchSamples TrackContinuousBranchSamples(
    const Mat4& target,
    double psi_ref,
    const Vec7& q_ref,
    const NeroParams& p,
    const PaperParams& paper) {
  BranchSamples out;
  const auto grid = BuildPsiGrid(paper.feasible_scan_step);
  if (grid.empty()) return out;
  int ref_idx = 0;
  double best_dist = std::numeric_limits<double>::infinity();
  for (int i = 0; i < static_cast<int>(grid.size()); ++i) {
    const double d = std::abs(grid[i] - psi_ref);
    if (d < best_dist) {
      best_dist = d;
      ref_idx = i;
    }
  }
  auto q0 = SelectBranchNearest(target, grid[ref_idx], q_ref, p);
  if (!q0.has_value()) return out;
  out.psi.push_back(grid[ref_idx]);
  out.q.push_back(*q0);

  Vec7 ref = *q0;
  for (int i = ref_idx + 1; i < static_cast<int>(grid.size()); ++i) {
    auto q = SelectBranchNearest(target, grid[i], ref, p);
    if (!q.has_value()) break;
    out.psi.push_back(grid[i]);
    out.q.push_back(*q);
    ref = *q;
  }

  ref = *q0;
  std::vector<double> rev_psi;
  std::vector<Vec7> rev_q;
  for (int i = ref_idx - 1; i >= 0; --i) {
    auto q = SelectBranchNearest(target, grid[i], ref, p);
    if (!q.has_value()) break;
    rev_psi.push_back(grid[i]);
    rev_q.push_back(*q);
    ref = *q;
  }
  for (int i = static_cast<int>(rev_psi.size()) - 1; i >= 0; --i) {
    out.psi.insert(out.psi.begin(), rev_psi[i]);
    out.q.insert(out.q.begin(), rev_q[i]);
  }
  return out;
}

bool IsCosJointIndex(int idx) {
  return idx == 1 || idx == 5;
}

CosPsiModel FitCosPsiModel(const BranchSamples& samples, int joint_idx) {
  CosPsiModel model;
  if (samples.psi.size() < 3) return model;
  Eigen::MatrixXd a(samples.psi.size(), 3);
  Eigen::VectorXd b(samples.psi.size());
  for (int i = 0; i < static_cast<int>(samples.psi.size()); ++i) {
    a(i, 0) = std::sin(samples.psi[i]);
    a(i, 1) = std::cos(samples.psi[i]);
    a(i, 2) = 1.0;
    b(i) = std::cos(samples.q[i][joint_idx]);
  }
  const Eigen::Vector3d x = a.colPivHouseholderQr().solve(b);
  model.a = x[0];
  model.b = x[1];
  model.c = x[2];
  model.ok = true;
  return model;
}

AtanPsiModel FitAtanPsiModel(const BranchSamples& samples, int joint_idx) {
  AtanPsiModel model;
  if (samples.psi.size() < 6) return model;
  Eigen::MatrixXd a(2 * samples.psi.size(), 6);
  Eigen::VectorXd b(2 * samples.psi.size());
  for (int i = 0; i < static_cast<int>(samples.psi.size()); ++i) {
    const double psi = samples.psi[i];
    const double q = samples.q[i][joint_idx];
    a.row(2 * i) << std::sin(psi), std::cos(psi), 1.0, 0.0, 0.0, 0.0;
    a.row(2 * i + 1) << 0.0, 0.0, 0.0, std::sin(psi), std::cos(psi), 1.0;
    b[2 * i] = std::sin(q);
    b[2 * i + 1] = std::cos(q);
  }
  const Eigen::Matrix<double, 6, 1> x = a.colPivHouseholderQr().solve(b);
  model.an = x[0];
  model.bn = x[1];
  model.cn = x[2];
  model.ad = x[3];
  model.bd = x[4];
  model.cd = x[5];
  model.ok = true;
  return model;
}

std::vector<double> SolveTrigEquation(double a, double b, double c) {
  std::vector<double> roots;
  const double r = std::hypot(a, b);
  if (r < 1e-12) return roots;
  const double rhs = -c / r;
  if (rhs < -1.0 - 1e-9 || rhs > 1.0 + 1e-9) return roots;
  const double phi = std::atan2(b, a);
  const double y = std::asin(std::clamp(rhs, -1.0, 1.0));
  roots.push_back(WrapToPi(y - phi));
  roots.push_back(WrapToPi((kPi - y) - phi));
  std::sort(roots.begin(), roots.end());
  roots.erase(std::unique(roots.begin(), roots.end(), [](double x, double y) {
    return std::abs(x - y) < 1e-8;
  }), roots.end());
  return roots;
}

std::vector<Interval> IntersectIntervals(const std::vector<Interval>& a, const std::vector<Interval>& b) {
  if (a.empty() || b.empty()) return {};
  std::vector<Interval> out;
  for (const auto& ia : a) {
    for (const auto& ib : b) {
      const double s = std::max(ia.start, ib.start);
      const double e = std::min(ia.end, ib.end);
      if (s <= e + 1e-10) out.push_back({s, e});
    }
  }
  return out;
}

std::vector<Interval> MergeIntervals(std::vector<Interval> intervals, double gap_tol = 1e-4) {
  if (intervals.empty()) return intervals;
  std::sort(intervals.begin(), intervals.end(), [](const Interval& lhs, const Interval& rhs) {
    if (lhs.start == rhs.start) return lhs.end < rhs.end;
    return lhs.start < rhs.start;
  });
  std::vector<Interval> merged;
  merged.push_back(intervals.front());
  for (int i = 1; i < static_cast<int>(intervals.size()); ++i) {
    auto& back = merged.back();
    if (intervals[i].start <= back.end + gap_tol) {
      back.end = std::max(back.end, intervals[i].end);
    } else {
      merged.push_back(intervals[i]);
    }
  }
  return merged;
}

std::vector<Interval> SampleIntervals(
    const std::function<double(double)>& eval,
    double q_lo,
    double q_hi,
    const std::vector<double>& extra_cuts = {}) {
  std::vector<double> cuts = {-kPi, kPi};
  cuts.insert(cuts.end(), extra_cuts.begin(), extra_cuts.end());
  const int dense_n = 720;
  for (int i = 0; i <= dense_n; ++i) {
    const double psi = -kPi + (2.0 * kPi * static_cast<double>(i)) / static_cast<double>(dense_n);
    const double q = eval(psi);
    if (std::abs(WrapToPi(q - q_lo)) < 0.02 || std::abs(WrapToPi(q - q_hi)) < 0.02) cuts.push_back(psi);
  }
  std::sort(cuts.begin(), cuts.end());
  cuts.erase(std::unique(cuts.begin(), cuts.end(), [](double x, double y) {
    return std::abs(x - y) < 1e-5;
  }), cuts.end());
  std::vector<Interval> out;
  for (int i = 0; i + 1 < static_cast<int>(cuts.size()); ++i) {
    const double s = cuts[i];
    const double e = cuts[i + 1];
    const double mid = 0.5 * (s + e);
    const double q = eval(mid);
    if (q >= q_lo - 1e-7 && q <= q_hi + 1e-7) out.push_back({s, e});
  }
  return out;
}

std::vector<Interval> BuildCosJointIntervals(const CosPsiModel& model, double q_lo, double q_hi) {
  auto eval = [&](double psi) {
    const double c = model.a * std::sin(psi) + model.b * std::cos(psi) + model.c;
    return std::acos(std::clamp(c, -1.0, 1.0));
  };
  return SampleIntervals(eval, q_lo, q_hi);
}

std::vector<Interval> BuildAtanJointIntervals(const AtanPsiModel& model, double q_lo, double q_hi) {
  const double at = model.cn * model.bd - model.bn * model.cd;
  const double bt = model.an * model.cd - model.cn * model.ad;
  const double ct = model.an * model.bd - model.bn * model.ad;
  auto eval = [&](double psi) {
    const double u = model.an * std::sin(psi) + model.bn * std::cos(psi) + model.cn;
    const double v = model.ad * std::sin(psi) + model.bd * std::cos(psi) + model.cd;
    return WrapToPi(std::atan2(u, v));
  };
  return SampleIntervals(eval, q_lo, q_hi, SolveTrigEquation(at, bt, ct));
}

// 回退路径: 采样+最小二乘拟合的可行域(用于 BuildJointModels 校验失败的构型)。
std::vector<Interval> BuildDangerJointFeasibleIntervalsLegacy(
    const Mat4& target,
    const Vec7& q_prev,
    double psi_prev,
    const NeroParams& p,
    const PaperParams& paper) {
  auto seed = FindNearestFeasibleSeed(target, psi_prev, q_prev, p, paper);
  if (!seed.has_value()) return {};
  const auto danger_joint_idx = DangerJointIndices(q_prev, p, paper.danger_threshold);
  if (danger_joint_idx.empty()) return {{-kPi, kPi}};
  const BranchSamples samples = TrackContinuousBranchSamples(target, seed->first, seed->second, p, paper);
  if (samples.psi.size() < 6) return {};

  std::vector<Interval> feasible {{-kPi, kPi}};
  for (int joint_idx : danger_joint_idx) {
    std::vector<Interval> joint_intervals;
    if (IsCosJointIndex(joint_idx)) {
      const auto model = FitCosPsiModel(samples, joint_idx);
      if (!model.ok) return {};
      joint_intervals = BuildCosJointIntervals(model, p.joint_limits(joint_idx, 0), p.joint_limits(joint_idx, 1));
    } else {
      const auto model = FitAtanPsiModel(samples, joint_idx);
      if (!model.ok) return {};
      joint_intervals = BuildAtanJointIntervals(model, p.joint_limits(joint_idx, 0), p.joint_limits(joint_idx, 1));
    }
    feasible = IntersectIntervals(feasible, joint_intervals);
    if (feasible.empty()) return {};
  }
  return MergeIntervals(std::move(feasible));
}

double ProjectPsiToFeasibleSet(double psi, const std::vector<Interval>& feasible) {
  for (const auto& seg : feasible) {
    if (psi >= seg.start && psi <= seg.end) return psi;
  }
  double best = feasible.front().start;
  double best_dist = std::numeric_limits<double>::infinity();
  for (const auto& seg : feasible) {
    for (double cand : {seg.start, seg.end}) {
      const double d = std::abs(WrapToPi(psi - cand));
      if (d < best_dist) {
        best_dist = d;
        best = cand;
      }
    }
  }
  return best;
}

// 差分线性化 (论文式 28-29): k = (q(ψ0+Δψ) − q0)/Δψ, 同一恒等分支上求取。
// 分支退化时返回 ok=false, 由调用方转入全局路径。
LinearizedModel LinearizeJointAroundPsi(
    const Mat4& target,
    double psi0,
    const Vec7& q0,
    const BranchId& branch,
    const NeroParams& p,
    const PaperParams& paper) {
  LinearizedModel out;
  auto q1 = ClosedFormIkBranch(target, WrapToPi(psi0 + paper.delta_psi), p, branch);
  if (!q1.has_value()) return out;
  out.ok = true;
  out.psi0 = psi0;
  out.q0 = q0;
  out.k = WrapToPi(*q1 - q0) / paper.delta_psi;
  return out;
}

std::vector<std::pair<double, double>> ToPairIntervals(const std::vector<Interval>& intervals) {
  std::vector<std::pair<double, double>> out;
  for (const auto& seg : intervals) out.emplace_back(seg.start, seg.end);
  return out;
}

}  // namespace

// 任意参数下的正运动学(全 8 个变换, 含单位阵)。
std::vector<Mat4> FkAllWith(const Vec7& q, const NeroParams& p) {
  std::vector<Mat4> out;
  out.reserve(8);
  Mat4 t = Mat4::Identity();
  out.push_back(t);
  for (int i = 0; i < 7; ++i) {
    t = t * DhA(q[i] + p.theta_offset[i], p.d_i[i], p.a_prev[i], p.alpha_prev[i]);
    out.push_back(t);
  }
  return out;
}

// ===================== 解析臂角-关节模型(论文第 2 节) =====================
// 论文方法: 给定末端位姿后 0R3(ψ) = As·sinψ + Bs·cosψ + Cs (式 5)、
// 0R7(ψ) = Aw·sinψ + Bw·cosψ + Cw (式 9), 各关节角由矩阵元素解析提取。
// 本实现按本仓库 DH 约定 (α1..α6 = π/2) 推导并经数值验证的恒等式构造,
// 不做任何采样拟合; 构型不匹配时 BuildJointModels 返回 ok=false 由调用方回退。

namespace {

// 把 θ 域限位拆成 (−π, π] 内的至多 2 段弧(θ 按 2π 周期性回绕)。
std::vector<Interval> SplitThetaRange(double th_lo, double th_hi) {
  std::vector<Interval> arcs;
  const double lo = WrapToPi(th_lo);
  const double hi = WrapToPi(th_hi);
  if (lo <= hi + 1e-12) {
    arcs.push_back({lo, hi});
  } else {
    arcs.push_back({lo, kPi});
    arcs.push_back({-kPi, hi});
  }
  return arcs;
}

// 把可能跨 ±π 接缝的 ψ 区间规范化到 (−π, π] 并合并。
std::vector<Interval> CanonicalizeIntervals(std::vector<Interval> ivs) {
  std::vector<Interval> out;
  for (auto iv : ivs) {
    if (iv.end - iv.start >= 2.0 * kPi - 1e-9) {
      out.push_back({-kPi, kPi});
      continue;
    }
    const double s = WrapToPi(iv.start);
    const double e = WrapToPi(iv.end);
    if (s <= e + 1e-12) {
      out.push_back({s, e});
    } else {
      out.push_back({s, kPi});
      out.push_back({-kPi, e});
    }
  }
  return MergeIntervals(std::move(out));
}

}  // namespace

std::pair<double, double> ThetaLimits(const NeroParams& p, int i) {
  return {p.joint_limits(i, 0) + p.theta_offset[i],
          p.joint_limits(i, 1) + p.theta_offset[i]};
}

std::vector<Vec7> ClosedFormIkAll(const Mat4& target, double psi, const NeroParams& p) {
  return IkOneArmAngleUnfiltered(target, psi, p);
}

BranchId BranchOf(const Vec7& q, const NeroParams& p) {
  BranchId id;
  id.s4 = q[3] >= 0.0 ? 1 : -1;
  id.s2 = std::sin(q[1] + p.theta_offset[1]) >= 0.0 ? 1 : -1;
  id.s6 = std::sin(q[5] + p.theta_offset[5]) >= 0.0 ? 1 : -1;
  return id;
}

std::optional<Vec7> ClosedFormIkBranch(
    const Mat4& target, double psi, const NeroParams& p, const BranchId& branch) {
  const auto sols = IkOneArmAngleUnfiltered(target, psi, p, branch);
  if (sols.empty()) return std::nullopt;
  return sols.front();
}

double RecoverArmAngle(const Mat4& target, const Vec7& q, const NeroParams& p) {
  // 肘心 E 绕 SW 轴在圆上运动 (论文式 3 的几何), ψ = atan2((E−C)·e2, (E−C)·e1)
  Eigen::Vector3d s, w;
  double q4_abs = 0.0;
  ComputeSweFromTarget(target, p, &s, &w, &q4_abs);
  if (!std::isfinite(q4_abs)) return 0.0;
  const Eigen::Vector3d e = FkAllWith(q, p)[3].block<3, 1>(0, 3);
  const Eigen::Vector3d sw = w - s;
  const double l_sw = sw.norm();
  if (l_sw < 1e-12) return 0.0;
  const Eigen::Vector3d u_sw = sw / l_sw;
  const double l_se = std::abs(p.d_i[2]);
  const double l_ew = std::abs(p.d_i[4]);
  const double x = (l_se * l_se - l_ew * l_ew + l_sw * l_sw) / (2.0 * l_sw);
  const Eigen::Vector3d c = s + x * u_sw;
  Eigen::Vector3d t = s.cross(u_sw);
  if (t.norm() < 1e-10) t = Eigen::Vector3d(1.0, 0.0, 0.0).cross(u_sw);
  if (t.norm() < 1e-10) t = Eigen::Vector3d(0.0, 1.0, 0.0).cross(u_sw);
  const Eigen::Vector3d e1 = t.normalized();
  const Eigen::Vector3d e2 = u_sw.cross(e1).normalized();
  return std::atan2((e - c).dot(e2), (e - c).dot(e1));
}

JointModelSet BuildJointModels(
    const Mat4& target, double psi0, const Vec7& q0, const NeroParams& p) {
  JointModelSet out;
  Eigen::Vector3d s, w;
  double q4_abs = 0.0;
  ComputeSweFromTarget(target, p, &s, &w, &q4_abs);
  if (!std::isfinite(q4_abs)) return out;
  const Eigen::Vector3d u = (w - s).normalized();
  Eigen::Matrix3d k;
  k << 0.0, -u.z(), u.y(),
       u.z(), 0.0, -u.x(),
       -u.y(), u.x(), 0.0;

  const Eigen::Matrix3d r3_0 = FkAllWith(q0, p)[3].block<3, 3>(0, 0);
  const double th4 = q0[3] + p.theta_offset[3];

  // R3(ψ) = R_{ψ−ψ0}·R3(ψ0), R_Δ = I + sinΔ·[u×] + (1−cosΔ)·[u×]² (论文式 3)
  // → R3(ψ) = As·sinψ + Bs·cosψ + Cs (论文式 5)
  const Eigen::Matrix3d k2 = k * k;
  const Eigen::Matrix3d k_r3 = k * r3_0;
  const Eigen::Matrix3d k2_r3 = k2 * r3_0;
  const double c0 = std::cos(psi0);
  const double s0 = std::sin(psi0);
  const Eigen::Matrix3d as = c0 * k_r3 - s0 * k2_r3;
  const Eigen::Matrix3d bs = -s0 * k_r3 - c0 * k2_r3;
  const Eigen::Matrix3d cs = r3_0 + k2_r3;

  // 腕部: R47(ψ) = R43·R3ᵀ(ψ)·R07 = Aw·sinψ + Bw·cosψ + Cw (论文式 9)
  const Eigen::Matrix3d r43 = (RotX(p.alpha_prev[3]) * RotZ(th4)).transpose();
  const Eigen::Matrix3d r07 = target.block<3, 3>(0, 0);
  const Eigen::Matrix3d aw = r43 * as.transpose() * r07;
  const Eigen::Matrix3d bw = r43 * bs.transpose() * r07;
  const Eigen::Matrix3d cw = r43 * cs.transpose() * r07;

  // 分支符号 (论文式 7-8 / 11-12 中的 sgn 因子, 分支内恒定)
  const double sig2 = std::sin(q0[1] + p.theta_offset[1]) >= 0.0 ? 1.0 : -1.0;
  const double sig6 = std::sin(q0[5] + p.theta_offset[5]) >= 0.0 ? 1.0 : -1.0;

  // 恒等式校验: 非本 DH(α1..α6 = π/2)构型时 ok=false, 由调用方回退。
  // 本 DH 下数值验证的恒等式 (solver.cpp 测试覆盖):
  //   cosθ2 = −(R3)₃₃, θ1 = atan2(σ2·(R3)₂₃, σ2·(R3)₁₃), θ3 = atan2(−σ2·(R3)₃₂, σ2·(R3)₃₁)
  //   cosθ6 = (R47)₁₂, θ5 = atan2(σ6·(R47)₂₂, σ6·(R47)₀₂), θ7 = atan2(σ6·(R47)₁₁, −σ6·(R47)₁₀)
  {
    const Eigen::Matrix3d r47_0 = r43 * r3_0.transpose() * r07;
    const double th1 = q0[0] + p.theta_offset[0];
    const double th2 = q0[1] + p.theta_offset[1];
    const double th3 = q0[2] + p.theta_offset[2];
    const double th5 = q0[4] + p.theta_offset[4];
    const double th6 = q0[5] + p.theta_offset[5];
    const double th7 = q0[6] + p.theta_offset[6];
    double err = std::abs(std::cos(th2) + r3_0(2, 2));
    err += std::abs(WrapToPi(th1 - std::atan2(sig2 * r3_0(1, 2), sig2 * r3_0(0, 2))));
    err += std::abs(WrapToPi(th3 - std::atan2(-sig2 * r3_0(2, 1), sig2 * r3_0(2, 0))));
    err += std::abs(std::cos(th6) - r47_0(1, 2));
    err += std::abs(WrapToPi(th5 - std::atan2(sig6 * r47_0(2, 2), sig6 * r47_0(0, 2))));
    err += std::abs(WrapToPi(th7 - std::atan2(sig6 * r47_0(1, 1), -sig6 * r47_0(1, 0))));
    if (err > 1e-8) return out;
  }

  for (int i = 0; i < 7; ++i) {
    out.joint[i].theta_ref = q0[i] + p.theta_offset[i];
    out.joint[i].psi_ref = psi0;
  }

  // cos 型关节: cosθ2 = −(R3)₃₃, cosθ4 = 常数, cosθ6 = (R47)₁₂
  out.joint[1].is_cos = true;
  out.joint[1].a = -as(2, 2);
  out.joint[1].b = -bs(2, 2);
  out.joint[1].c = -cs(2, 2);
  out.joint[3].is_cos = true;
  out.joint[3].a = 0.0;
  out.joint[3].b = 0.0;
  out.joint[3].c = std::cos(th4);
  out.joint[5].is_cos = true;
  out.joint[5].a = aw(1, 2);
  out.joint[5].b = bw(1, 2);
  out.joint[5].c = cw(1, 2);

  // atan2 型关节: θ1、θ3(肩部), θ5、θ7(腕部)
  out.joint[0].is_cos = false;
  out.joint[0].an = sig2 * as(1, 2);
  out.joint[0].bn = sig2 * bs(1, 2);
  out.joint[0].cn = sig2 * cs(1, 2);
  out.joint[0].ad = sig2 * as(0, 2);
  out.joint[0].bd = sig2 * bs(0, 2);
  out.joint[0].cd = sig2 * cs(0, 2);
  out.joint[2].is_cos = false;
  out.joint[2].an = -sig2 * as(2, 1);
  out.joint[2].bn = -sig2 * bs(2, 1);
  out.joint[2].cn = -sig2 * cs(2, 1);
  out.joint[2].ad = sig2 * as(2, 0);
  out.joint[2].bd = sig2 * bs(2, 0);
  out.joint[2].cd = sig2 * cs(2, 0);
  out.joint[4].is_cos = false;
  out.joint[4].an = sig6 * aw(2, 2);
  out.joint[4].bn = sig6 * bw(2, 2);
  out.joint[4].cn = sig6 * cw(2, 2);
  out.joint[4].ad = sig6 * aw(0, 2);
  out.joint[4].bd = sig6 * bw(0, 2);
  out.joint[4].cd = sig6 * cw(0, 2);
  out.joint[6].is_cos = false;
  out.joint[6].an = sig6 * aw(1, 1);
  out.joint[6].bn = sig6 * bw(1, 1);
  out.joint[6].cn = sig6 * cw(1, 1);
  out.joint[6].ad = -sig6 * aw(1, 0);
  out.joint[6].bd = -sig6 * bw(1, 0);
  out.joint[6].cd = -sig6 * cw(1, 0);

  out.ok = true;
  return out;
}

std::vector<std::pair<double, double>> CosJointFeasibleIntervals(
    const JointModel& m, double th_lo, double th_hi) {
  std::vector<std::pair<double, double>> out;
  const auto arcs = SplitThetaRange(th_lo, th_hi);
  const double r = std::hypot(m.a, m.b);
  if (r < 1e-12) {
    // 常数关节 (θ4): 分支角在限位内 → 全域, 否则空
    const double th = WrapToPi(m.theta_ref);
    for (const auto& arc : arcs) {
      if (th >= arc.start - 1e-9 && th <= arc.end + 1e-9) {
        out.emplace_back(-kPi, kPi);
        break;
      }
    }
    return out;
  }
  // cosθ = a·sinψ + b·cosψ + c 无法区分 θ 的镜像符号 (cos 为偶函数),
  // 因此逐单元追踪分支符号 σ = sign(sinθ): σ 在 cosθ = ±1 (θ 过 0/±π) 处翻转;
  // 相切退化 (模型常数 c = ±1) 时翻转可忽略。
  const double sigma0 = std::sin(m.theta_ref) >= 0.0 ? 1.0 : -1.0;
  struct Event {
    double psi;
    bool flips;
  };
  std::vector<Event> events{{-kPi, false}, {kPi, false}};
  for (double level : {1.0, -1.0}) {
    for (double psi : SolveTrigEquation(m.a, m.b, m.c - level)) {
      events.push_back({psi, std::abs(m.c - level) > 1e-9});
    }
  }
  // 限位弧段端点水平线 cosθ = cosθ* (约束边界)
  for (const auto& arc : arcs) {
    for (double th_star : {arc.start, arc.end}) {
      for (double psi : SolveTrigEquation(m.a, m.b, m.c - std::cos(th_star))) {
        events.push_back({psi, false});
      }
    }
  }
  std::sort(events.begin(), events.end(), [](const Event& a, const Event& b) {
    return a.psi < b.psi;
  });
  std::vector<Event> ev;
  for (const auto& e : events) {
    if (!ev.empty() && std::abs(ev.back().psi - e.psi) < 1e-9) {
      ev.back().flips = ev.back().flips || e.flips;
    } else {
      ev.push_back(e);
    }
  }
  // 锚点 ψ_ref 所在单元 σ = σ0, 向两侧传播 (跨翻转事件取反)
  int c0 = 0;
  for (int i = 0; i + 1 < static_cast<int>(ev.size()); ++i) {
    if (m.psi_ref >= ev[i].psi - 1e-12 && m.psi_ref <= ev[i + 1].psi + 1e-12) {
      c0 = i;
      break;
    }
  }
  std::vector<double> cell_sigma(ev.size() - 1, sigma0);
  cell_sigma[c0] = sigma0;
  double sgn = sigma0;
  for (int i = c0 + 1; i + 1 < static_cast<int>(ev.size()); ++i) {
    sgn *= ev[i].flips ? -1.0 : 1.0;
    cell_sigma[i] = sgn;
  }
  sgn = sigma0;
  for (int i = c0 - 1; i >= 0; --i) {
    sgn *= ev[i + 1].flips ? -1.0 : 1.0;
    cell_sigma[i] = sgn;
  }
  for (int i = 0; i + 1 < static_cast<int>(ev.size()); ++i) {
    const double s = ev[i].psi;
    const double e = ev[i + 1].psi;
    if (e - s < 1e-12) continue;
    const double mid = 0.5 * (s + e);
    const double c_mid = m.a * std::sin(mid) + m.b * std::cos(mid) + m.c;
    const double th_mid = cell_sigma[i] * std::acos(std::clamp(c_mid, -1.0, 1.0));
    for (const auto& arc : arcs) {
      if (th_mid >= arc.start - 1e-9 && th_mid <= arc.end + 1e-9) {
        out.emplace_back(s, e);
        break;
      }
    }
  }
  return out;
}

std::vector<std::pair<double, double>> AtanJointFeasibleIntervals(
    const JointModel& m, double th_lo, double th_hi) {
  std::vector<std::pair<double, double>> out;
  const auto arcs = SplitThetaRange(th_lo, th_hi);
  auto u_of = [&m](double psi) { return m.an * std::sin(psi) + m.bn * std::cos(psi) + m.cn; };
  auto v_of = [&m](double psi) { return m.ad * std::sin(psi) + m.bd * std::cos(psi) + m.cd; };
  const bool u_const = std::abs(m.an) < 1e-12 && std::abs(m.bn) < 1e-12;
  const bool v_const = std::abs(m.ad) < 1e-12 && std::abs(m.bd) < 1e-12;
  if (u_const && v_const) {
    // 常数关节: θ = atan2(cn, cd), 在限位内 → 全域, 否则空
    const double th = std::atan2(m.cn, m.cd);
    for (const auto& arc : arcs) {
      if (th >= arc.start - 1e-9 && th <= arc.end + 1e-9) {
        out.emplace_back(-kPi, kPi);
        break;
      }
    }
    return out;
  }
  // 事件: 导数零点(论文式 16-17)、atan2 分支切割(u=0 且 v<0)、奇异点(u=v=0)、
  // 以及限位弧段端点的水平线交叉 θ(ψ)=θ*。单元内 θ(ψ) 连续单调且不穿越限位
  // 边界, 中点判定即精确结果。
  std::vector<double> events{-kPi, kPi};
  const double at = m.cn * m.bd - m.bn * m.cd;
  const double bt = m.an * m.cd - m.cn * m.ad;
  const double ct = m.an * m.bd - m.bn * m.ad;
  if (std::abs(at) + std::abs(bt) > 1e-12) {
    for (double psi : SolveTrigEquation(at, bt, ct)) events.push_back(psi);
  }
  // 限位边界: θ(ψ) = θ* ⟺ v·sinθ* − u·cosθ* = 0 且 (u,v)·(sinθ*,cosθ*) > 0
  for (const auto& arc : arcs) {
    for (double th_star : {arc.start, arc.end}) {
      const double ss = std::sin(th_star);
      const double cc = std::cos(th_star);
      const double A = m.ad * ss - m.an * cc;
      const double B = m.bd * ss - m.bn * cc;
      const double C = m.cd * ss - m.cn * cc;
      if (std::abs(A) + std::abs(B) < 1e-12) continue;
      for (double psi : SolveTrigEquation(A, B, C)) {
        if (u_of(psi) * ss + v_of(psi) * cc > 1e-12) events.push_back(psi);
      }
    }
  }
  if (u_const) {
    if (std::abs(m.cn) < 1e-12 && !v_const) {
      // u ≡ 0: θ 在 v 过零点处跳 π
      for (double psi : SolveTrigEquation(m.ad, m.bd, m.cd)) events.push_back(psi);
    }
  } else {
    for (double psi : SolveTrigEquation(m.an, m.bn, m.cn)) {
      if (v_of(psi) < -1e-12) events.push_back(psi);  // θ 跨 ±π 接缝
    }
  }
  if (v_const && std::abs(m.cd) < 1e-12) {
    // v ≡ 0: θ 在 u 过零点处跳 π
    for (double psi : SolveTrigEquation(m.an, m.bn, m.cn)) events.push_back(psi);
  }
  // 奇异点 (分支切换处): u = 0 与 v = 0 的公共根
  {
    const auto ru = u_const ? std::vector<double>{} : SolveTrigEquation(m.an, m.bn, m.cn);
    const auto rv = v_const ? std::vector<double>{} : SolveTrigEquation(m.ad, m.bd, m.cd);
    for (double p1 : ru) {
      for (double p2 : rv) {
        if (std::abs(WrapToPi(p1 - p2)) < 1e-9) events.push_back(p1);
      }
    }
  }
  std::sort(events.begin(), events.end());
  events.erase(std::unique(events.begin(), events.end(), [](double a, double b) {
    return std::abs(a - b) < 1e-9;
  }), events.end());
  for (int i = 0; i + 1 < static_cast<int>(events.size()); ++i) {
    const double mid = 0.5 * (events[i] + events[i + 1]);
    const double th = std::atan2(u_of(mid), v_of(mid));
    for (const auto& arc : arcs) {
      if (th >= arc.start - 1e-9 && th <= arc.end + 1e-9) {
        out.emplace_back(events[i], events[i + 1]);
        break;
      }
    }
  }
  return out;
}

// 全部 7 关节精确可行臂角区间之交 (论文式 31-32)。
// 解析模型下各关节区间计算代价可忽略, 无需论文第 3.3 节的危险关节筛选。
std::vector<Interval> BuildFeasibleIntervals(
    const Mat4& target, const Vec7& q_seed, double psi_seed, const JointModelSet& models,
    const NeroParams& p) {
  std::vector<Interval> feasible{{-kPi, kPi}};
  for (int i = 0; i < 7; ++i) {
    const auto [lo, hi] = ThetaLimits(p, i);
    std::vector<Interval> iv;
    if (models.joint[i].is_cos) {
      for (const auto& pr : CosJointFeasibleIntervals(models.joint[i], lo, hi)) {
        iv.push_back({pr.first, pr.second});
      }
    } else {
      for (const auto& pr : AtanJointFeasibleIntervals(models.joint[i], lo, hi)) {
        iv.push_back({pr.first, pr.second});
      }
    }
    feasible = IntersectIntervals(feasible, iv);
    if (feasible.empty()) return {};
  }
  return MergeIntervals(std::move(feasible));
}

// 全局路径: 目标跳变/分支退化时的 8 分支解析重规划。
// 对 ψ0 处枚举的每个分支锚点构造解析模型与精确可行域, 在可行域内搜索
// 加权目标函数极小值(每段 33 点等距采样粗选 + 黄金分割局部细化),
// 跨分支取目标函数最小者。全程零限位过滤近似, 各分支解恒在限位内。
std::optional<std::pair<double, Vec7>> GlobalAnalyticSolve(
    const Mat4& target,
    double psi0,
    const std::vector<Vec7>& anchors,
    const Vec7& q_prev,
    const Vec7& w,
    const NeroParams& p,
    const PaperParams& paper) {
  std::optional<std::pair<double, Vec7>> best;
  double best_cost = std::numeric_limits<double>::infinity();
  for (const auto& anchor : anchors) {
    const BranchId branch = BranchOf(anchor, p);
    const auto models = BuildJointModels(target, psi0, anchor, p);
    if (!models.ok) continue;
    const auto feasible = BuildFeasibleIntervals(target, anchor, psi0, models, p);
    if (feasible.empty()) continue;
    auto eval = [&](double psi) -> double {
      auto q = ClosedFormIkBranch(target, psi, p, branch);
      if (!q.has_value()) return std::numeric_limits<double>::infinity();
      double cost = 0.0;
      for (int j = 0; j < 7; ++j) {
        const double dq = WrapToPi((*q)[j] - q_prev[j]);
        cost += w[j] * dq * dq;
      }
      return cost;
    };
    for (const auto& seg : feasible) {
      const double span = seg.end - seg.start;
      if (span < 1e-9) continue;
      const int kN = 33;
      double psi_best = seg.start;
      double cost_best = std::numeric_limits<double>::infinity();
      for (int i = 0; i < kN; ++i) {
        const double psi = seg.start + span * static_cast<double>(i) / static_cast<double>(kN - 1);
        const double c = eval(psi);
        if (c < cost_best) {
          cost_best = c;
          psi_best = psi;
        }
      }
      if (!std::isfinite(cost_best)) continue;
      // 黄金分割细化最佳采样点邻域(跨度 = 采样间距, 段内裁剪;
      // 25 次迭代精度 0.618^25 ≈ 7e-6 rad, 远超目标函数需要)
      double lo = std::max(seg.start, psi_best - span / static_cast<double>(kN - 1));
      double hi = std::min(seg.end, psi_best + span / static_cast<double>(kN - 1));
      constexpr double kGr = 0.6180339887498949;
      double x1 = hi - kGr * (hi - lo);
      double x2 = lo + kGr * (hi - lo);
      double f1 = eval(x1);
      double f2 = eval(x2);
      for (int iter = 0; iter < 25 && hi - lo > 1e-7; ++iter) {
        if (f1 < f2) {
          hi = x2;
          x2 = x1;
          f2 = f1;
          x1 = hi - kGr * (hi - lo);
          f1 = eval(x1);
        } else {
          lo = x1;
          x1 = x2;
          f1 = f2;
          x2 = lo + kGr * (hi - lo);
          f2 = eval(x2);
        }
      }
      const double psi_refine = 0.5 * (lo + hi);
      const double c_refine = std::min(f1, f2);
      const double psi_final = c_refine < cost_best ? psi_refine : psi_best;
      const double cost_final = std::min(c_refine, cost_best);
      if (cost_final < best_cost) {
        auto q = ClosedFormIkBranch(target, psi_final, p, branch);
        if (q.has_value()) {
          best_cost = cost_final;
          best = std::make_pair(psi_final, *q);
        }
      }
    }
  }
  return best;
}

NeroParams NeroParams::Default() {
  NeroParams p;
  p.a_prev << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  p.alpha_prev << 0.0, kPi / 2.0, kPi / 2.0, kPi / 2.0, kPi / 2.0, kPi / 2.0, kPi / 2.0;
  p.d_i << 0.138, 0.0, 0.31, 0.0, 0.27, 0.0, 0.0235;
  p.theta_offset << 0.0, -kPi, -kPi, -kPi, kPi / 2.0, kPi / 2.0, 0.0;
  p.joint_limits <<
      -2.705261, 2.705261,
      -1.745330, 1.745330,
      -2.757621, 2.757621,
      -1.012291, 2.146755,
      -2.757621, 2.757621,
      -0.733039, 0.959932,
      -1.570797, 1.570797;
  // Default IK target is link7 origin (no virtual TCP offset).
  p.post_transform_d8 = 0.0;
  return p;
}

Solver::Solver(NeroParams params, PaperParams paper)
    : params_(std::move(params)), paper_(paper) {}

std::vector<Mat4> Solver::FkAll(const Vec7& q) const {
  return FkAllWith(q, params_);
}

Mat4 Solver::Fk(const Vec7& q) const {
  Mat4 t = FkAll(q).back();
  Mat4 t_post = Mat4::Identity();
  t_post(2, 3) = params_.post_transform_d8;
  return t * t_post;
}

IkResult Solver::IkArmAngle(const Mat4& target, const Vec7& q_prev, double psi_prev) const {
  IkResult res;
  res.report.method = "paper_1d_qp";
  res.report.psi_prev = psi_prev;
  res.report.delta_psi = paper_.delta_psi;
  res.report.danger_joint_indices = DangerJointIndices(q_prev, params_, paper_.danger_threshold);
  const Vec7 w = WeightLimitsFromPrev(q_prev, params_);

  // 全局路径(目标跳变/连续性失效/分支退化时): 8 分支解析模型 + 可行域 + 跨分支择优。
  auto run_global = [&](double psi0, const std::vector<Vec7>& anchors) -> bool {
    auto global = GlobalAnalyticSolve(target, psi0, anchors, q_prev, w, params_, paper_);
    if (!global.has_value()) return false;
    // 防御性限位复查: 解析区间在数值边界(奇异/相切邻域)可能含越限点,
    // 全局路径返回前强制校验, 失败则重试相邻 ψ 或放弃。
    if (!WithinLimits(global->second, params_.joint_limits)) {
      bool ok = false;
      for (double dpsi : {0.01, -0.01, 0.02, -0.02}) {
        auto q = ClosedFormIkBranch(target, WrapToPi(global->first + dpsi), params_,
                                    BranchOf(global->second, params_));
        if (q.has_value() && WithinLimits(*q, params_.joint_limits)) {
          global->second = *q;
          ok = true;
          break;
        }
      }
      if (!ok) return false;
    }
    res.report.fallback_used = true;
    res.report.psi_selected = global->first;
    res.q_best = global->second;
    res.report.pose_err_best = PoseErrorNorm(Fk(*res.q_best), target);
    res.report.success = true;
    return true;
  };
  auto try_global = [&](double psi0) -> bool {
    auto anchors = ClosedFormIkAll(target, psi0, params_);
    if (anchors.empty()) {
      // 闭式解退化(奇异邻域): 微移 ψ0 重试一次
      psi0 = WrapToPi(psi0 + 0.01);
      anchors = ClosedFormIkAll(target, psi0, params_);
    }
    if (anchors.empty()) return false;  // 目标不可达
    return run_global(psi0, anchors);
  };
  auto try_global_with_anchors = [&](double psi0, const std::vector<Vec7>& anchors) -> bool {
    if (anchors.empty()) return try_global(psi0);
    return run_global(psi0, anchors);
  };

  // 1. 连续性判定, 分两种情况:
  //    路径 A (目标静止/回环): q_prev 是当前目标的解 (位姿差 ≈ 0), 局部路径
  //        围绕 q_prev 自身臂角 ψ_q 展开 (锚点 = q_prev 本身, 最优区域重合)。
  //    路径 B (目标连续移动): q_prev 是上一目标的解, 相邻目标差一个插补步长;
  //        锚点 = ψ_prev 处枚举并认领 q_prev 分支, 用锚点解与 q_prev 的关节差
  //        度量连续性 (ψ_prev 传续, 连续轨迹该差必然小), 局部路径围绕 ψ_prev 展开。
  //    两者都不满足 → 全局路径 (目标跳变/冷启动)。
  const double pose_d = (Fk(q_prev) - target).cwiseAbs().maxCoeff();

  // 局部路径公共部分: 给定锚点 (psi0, anchor, branch), 模型/区间/线性化/二次/执行
  auto run_local = [&](double psi0, const Vec7& anchor, const BranchId& branch) -> bool {
    const auto models = BuildJointModels(target, psi0, anchor, params_);
    if (!models.ok) {
      // DH 构型与解析恒等式不匹配: 回退采样+拟合路径 (仅覆盖非默认 DH 的
      // 可行域; 线性化/选解仍用通用闭式 IK)。
      auto feasible_legacy = BuildDangerJointFeasibleIntervalsLegacy(
          target, anchor, psi0, params_, paper_);
      res.report.feasible_intervals = ToPairIntervals(feasible_legacy);
      if (feasible_legacy.empty()) return false;
      auto q1 = SelectBranchNearest(target, WrapToPi(psi0 + paper_.delta_psi), anchor, params_);
      if (!q1.has_value()) return false;
      const Vec7 k = WrapToPi(*q1 - anchor) / paper_.delta_psi;
      double a = 0.0, b = 0.0, c = 0.0;
      for (int i = 0; i < 7; ++i) {
        const double d = WrapToPi(anchor[i] - q_prev[i]) - k[i] * psi0;
        a += w[i] * k[i] * k[i];
        b += 2.0 * w[i] * k[i] * d;
        c += w[i] * d * d;
      }
      double psi_opt = psi_prev;
      if (std::abs(a) > 1e-12) psi_opt = -b / (2.0 * a);
      const double psi_final = ProjectPsiToFeasibleSet(WrapToPi(psi_opt), feasible_legacy);
      res.report.psi_selected = psi_final;
      auto q_best = SelectBranchNearest(target, psi_final, anchor, params_);
      if (!q_best.has_value()) return false;
      res.q_best = *q_best;
      res.report.pose_err_best = PoseErrorNorm(Fk(*q_best), target);
      res.report.success = true;
      return true;
    }
    auto feasible = BuildFeasibleIntervals(target, anchor, psi0, models, params_);
    res.report.feasible_intervals = ToPairIntervals(feasible);
    if (feasible.empty()) return false;
    const auto linearized = LinearizeJointAroundPsi(target, psi0, anchor, branch, params_, paper_);
    if (!linearized.ok) return false;
    // 二次函数系数 (论文式 33; d 做 wrap 修正)
    double a = 0.0;
    double b = 0.0;
    double c = 0.0;
    for (int i = 0; i < 7; ++i) {
      const double d = WrapToPi(linearized.q0[i] - q_prev[i]) - linearized.k[i] * linearized.psi0;
      a += w[i] * linearized.k[i] * linearized.k[i];
      b += 2.0 * w[i] * linearized.k[i] * d;
      c += w[i] * d * d;
    }
    res.report.quad_A = a;
    res.report.quad_B = b;
    res.report.quad_C = c;

    double psi_opt = psi_prev;
    if (std::abs(a) > 1e-12) psi_opt = -b / (2.0 * a);
    psi_opt = WrapToPi(psi_opt);
    res.report.psi_opt = psi_opt;
    const double psi_final = ProjectPsiToFeasibleSet(psi_opt, feasible);
    res.report.psi_selected = psi_final;

    // 单支执行: 恒等分支在可行域内必存在且限位内 (解析区间保证)
    auto q_best = ClosedFormIkBranch(target, psi_final, params_, branch);
    if (!q_best.has_value()) {
      // 分支退化(奇异邻域): 微移 ψ 重试
      for (double dpsi : {0.01, -0.01, 0.02, -0.02}) {
        q_best = ClosedFormIkBranch(target, WrapToPi(psi_final + dpsi), params_, branch);
        if (q_best.has_value()) {
          res.report.psi_selected = WrapToPi(psi_final + dpsi);
          break;
        }
      }
    }
    if (!q_best.has_value()) return false;
    for (int i = 0; i < 7; ++i) {
      if ((*q_best)[i] < params_.joint_limits(i, 0) - 1e-6 ||
          (*q_best)[i] > params_.joint_limits(i, 1) + 1e-6) {
        return false;
      }
    }
    res.q_best = *q_best;
    res.report.pose_err_best = PoseErrorNorm(Fk(*q_best), target);
    res.report.success = true;
    return true;
  };

  if (pose_d <= 1e-4) {
    // 路径 A: 目标静止/回环
    const double psi_q = RecoverArmAngle(target, q_prev, params_);
    const double jump = std::abs(WrapToPi(psi_prev - psi_q));
    if (jump <= paper_.branch_jump_threshold) {
      const BranchId branch = BranchOf(q_prev, params_);
      auto anchor = ClosedFormIkBranch(target, psi_q, params_, branch);
      if (anchor.has_value() && WithinLimits(*anchor, params_.joint_limits)) {
        if (run_local(psi_q, *anchor, branch)) return res;
      }
    }
  } else {
    // 路径 B: 目标连续移动 — 锚点 = ψ_prev 处枚举, 认领 q_prev 分支
    double psi0 = WrapToPi(psi_prev);
    auto anchors = ClosedFormIkAll(target, psi0, params_);
    if (anchors.empty()) {
      // 闭式解退化(奇异邻域): 微移 ψ0 重试一次
      psi0 = WrapToPi(psi0 + 0.01);
      anchors = ClosedFormIkAll(target, psi0, params_);
    }
    if (!anchors.empty()) {
      const BranchId branch_prev = BranchOf(q_prev, params_);
      std::optional<Vec7> anchor;
      for (const auto& q : anchors) {
        if (BranchOf(q, params_) == branch_prev) {
          anchor = q;
          break;
        }
      }
      if (!anchor.has_value()) {
        // 认领失败(分支退化): 取距 q_prev 最近者, 显式换分支
        double best_d = std::numeric_limits<double>::infinity();
        for (const auto& q : anchors) {
          const double d = (WrapToPi(q - q_prev)).norm();
          if (d < best_d) {
            best_d = d;
            anchor = q;
          }
        }
      }
      const BranchId branch = BranchOf(*anchor, params_);
      const double jump = (WrapToPi(*anchor - q_prev)).cwiseAbs().maxCoeff();
      if (jump <= paper_.branch_jump_threshold) {
        if (run_local(psi0, *anchor, branch)) return res;
      }
      // 跳变检测失败(关节差大): 落到全局路径, 用锚点集合
      try_global_with_anchors(psi0, anchors);
      return res;
    }
  }

  // 2. 全局路径 (目标跳变/冷启动/局部失败)
  try_global(WrapToPi(psi_prev));
  return res;
}

TrajectoryResult Solver::SolveTrajectory(
    const std::vector<Mat4>& targets,
    const Vec7& q_init,
    double psi_init) const {
  TrajectoryResult out;
  Vec7 q_prev = q_init;
  double psi_prev = psi_init;
  for (const auto& target : targets) {
    const IkResult res = IkArmAngle(target, q_prev, psi_prev);
    out.q_list.push_back(res.q_best);
    out.reports.push_back(res.report);
    if (!res.q_best.has_value()) {
      out.psi_list.push_back(psi_prev);
      continue;
    }
    q_prev = *res.q_best;
    psi_prev = res.report.psi_selected;
    out.psi_list.push_back(psi_prev);
  }
  return out;
}

}  // namespace super_ik_tsinghua_1dqp
