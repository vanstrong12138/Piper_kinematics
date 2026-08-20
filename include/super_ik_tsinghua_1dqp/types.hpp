#pragma once

#include <Eigen/Dense>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace super_ik_tsinghua_1dqp {

using Vec7 = Eigen::Matrix<double, 7, 1>;
using Mat4 = Eigen::Matrix4d;

struct NeroParams {
  Eigen::Matrix<double, 7, 1> a_prev;
  Eigen::Matrix<double, 7, 1> alpha_prev;
  Eigen::Matrix<double, 7, 1> d_i;
  Eigen::Matrix<double, 7, 1> theta_offset;
  Eigen::Matrix<double, 7, 2> joint_limits;
  double post_transform_d8 { 0.0 };

  static NeroParams Default();
};

struct PaperParams {
  double delta_psi { 1e-3 };
  double danger_threshold { 0.8 };
  double feasible_scan_step { 0.01 };
  int psi_seed_samples { 33 };
  double psi_recover_window { 0.15 };
  int psi_recover_samples { 25 };
  bool enable_global_fallback { true };
  int global_fallback_samples { 121 };
  double branch_jump_threshold { 0.4 };
};

struct IkReport {
  std::string method;
  bool success { false };
  double pose_err_best { 0.0 };
  double psi_prev { 0.0 };
  double psi_selected { 0.0 };
  double psi_opt { 0.0 };
  double delta_psi { 0.0 };
  double quad_A { 0.0 };
  double quad_B { 0.0 };
  double quad_C { 0.0 };
  bool fallback_used { false };
  std::vector<int> danger_joint_indices;
  std::vector<std::pair<double, double>> feasible_intervals;
};

struct IkResult {
  std::optional<Vec7> q_best;
  IkReport report;
};

struct TrajectoryResult {
  std::vector<std::optional<Vec7>> q_list;
  std::vector<double> psi_list;
  std::vector<IkReport> reports;
};

}  // namespace super_ik_tsinghua_1dqp
