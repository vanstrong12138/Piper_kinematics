#pragma once

#include "super_ik_tsinghua_1dqp/types.hpp"

#include <vector>

namespace super_ik_tsinghua_1dqp {

class Solver {
 public:
  explicit Solver(
      NeroParams params = NeroParams::Default(),
      PaperParams paper = PaperParams {});

  const NeroParams& params() const { return params_; }
  const PaperParams& paper() const { return paper_; }

  Mat4 Fk(const Vec7& q) const;
  std::vector<Mat4> FkAll(const Vec7& q) const;

  IkResult IkArmAngle(const Mat4& target, const Vec7& q_prev, double psi_prev) const;

  TrajectoryResult SolveTrajectory(
      const std::vector<Mat4>& targets,
      const Vec7& q_init,
      double psi_init) const;

 private:
  NeroParams params_;
  PaperParams paper_;
};

}  // namespace super_ik_tsinghua_1dqp
