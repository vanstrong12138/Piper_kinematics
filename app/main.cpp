#include "super_ik_tsinghua_1dqp/solver.hpp"

#include <chrono>
#include <iostream>

int main() {
  using namespace super_ik_tsinghua_1dqp;
  constexpr double kPi = 3.14159265358979323846;
  Solver solver;

  Vec7 q_true;
  q_true << 0.2, -0.4, 0.5, 0.3, -0.2, 0.1, 0.2;
  Vec7 q_prev = q_true;
  q_prev[0] -= 0.05;
  q_prev[2] += 0.03;

  const Mat4 target = solver.Fk(q_true);

  const auto t0 = std::chrono::high_resolution_clock::now();
  auto ik = solver.IkArmAngle(target, q_prev, 0.0);
  const auto t1 = std::chrono::high_resolution_clock::now();
  const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

  std::cout << "=== super_ik_tsinghua_1dqp demo ===\n";
  std::cout << "method: " << ik.report.method << "\n";
  std::cout << "success: " << (ik.report.success ? "true" : "false") << "\n";
  std::cout << "pose_err_best: " << ik.report.pose_err_best << "\n";
  std::cout << "psi_selected: " << ik.report.psi_selected << "\n";
  std::cout << "psi_opt: " << ik.report.psi_opt << "\n";
  std::cout << "elapsed_ms: " << ms << "\n";
  if (ik.q_best.has_value()) {
    std::cout << "q_best: " << ik.q_best->transpose() << "\n";
  }

  std::vector<Mat4> traj_targets;
  traj_targets.reserve(40);
  for (int i = 0; i < 40; ++i) {
    const double s = static_cast<double>(i) / 39.0;
    Mat4 t = target;
    t(0, 3) += 0.05 * std::cos(2.0 * kPi * s);
    t(1, 3) += 0.05 * std::sin(2.0 * kPi * s);
    traj_targets.push_back(t);
  }

  auto tr = solver.SolveTrajectory(traj_targets, q_true, 0.0);
  int ok = 0;
  for (const auto& q : tr.q_list) ok += q.has_value() ? 1 : 0;
  std::cout << "trajectory success: " << ok << "/" << traj_targets.size() << "\n";
  return 0;
}
