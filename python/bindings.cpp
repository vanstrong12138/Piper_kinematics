#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "super_ik_tsinghua_1dqp/solver.hpp"

namespace py = pybind11;

namespace {

py::dict ReportToDict(const super_ik_tsinghua_1dqp::IkReport& r) {
  py::dict d;
  d["method"] = r.method;
  d["success"] = r.success;
  d["pose_err_best"] = r.pose_err_best;
  d["psi_prev"] = r.psi_prev;
  d["psi_selected"] = r.psi_selected;
  d["psi_opt"] = r.psi_opt;
  d["delta_psi"] = r.delta_psi;
  d["quad_A"] = r.quad_A;
  d["quad_B"] = r.quad_B;
  d["quad_C"] = r.quad_C;
  d["fallback_used"] = r.fallback_used;
  d["danger_joint_indices"] = r.danger_joint_indices;
  d["feasible_intervals"] = r.feasible_intervals;
  return d;
}

py::object OptionalQToPy(const std::optional<super_ik_tsinghua_1dqp::Vec7>& q) {
  if (!q.has_value()) return py::none();
  return py::cast(*q);
}

}  // namespace

PYBIND11_MODULE(_super_ik_tsinghua_1dqp, m) {
  m.doc() = "super_ik_tsinghua_1dqp: explicit feasible intervals + 1D QP IK (C++)";

  py::class_<super_ik_tsinghua_1dqp::NeroParams>(m, "NeroParams")
      .def(py::init<>())
      .def_static("default_params", &super_ik_tsinghua_1dqp::NeroParams::Default)
      .def_readwrite("a_prev", &super_ik_tsinghua_1dqp::NeroParams::a_prev)
      .def_readwrite("alpha_prev", &super_ik_tsinghua_1dqp::NeroParams::alpha_prev)
      .def_readwrite("d_i", &super_ik_tsinghua_1dqp::NeroParams::d_i)
      .def_readwrite("theta_offset", &super_ik_tsinghua_1dqp::NeroParams::theta_offset)
      .def_readwrite("joint_limits", &super_ik_tsinghua_1dqp::NeroParams::joint_limits)
      .def_readwrite("post_transform_d8", &super_ik_tsinghua_1dqp::NeroParams::post_transform_d8);

  py::class_<super_ik_tsinghua_1dqp::PaperParams>(m, "PaperParams")
      .def(py::init<>())
      .def_readwrite("delta_psi", &super_ik_tsinghua_1dqp::PaperParams::delta_psi)
      .def_readwrite("danger_threshold", &super_ik_tsinghua_1dqp::PaperParams::danger_threshold)
      .def_readwrite("feasible_scan_step", &super_ik_tsinghua_1dqp::PaperParams::feasible_scan_step)
      .def_readwrite("psi_seed_samples", &super_ik_tsinghua_1dqp::PaperParams::psi_seed_samples)
      .def_readwrite("psi_recover_window", &super_ik_tsinghua_1dqp::PaperParams::psi_recover_window)
      .def_readwrite("psi_recover_samples", &super_ik_tsinghua_1dqp::PaperParams::psi_recover_samples)
      .def_readwrite("enable_global_fallback", &super_ik_tsinghua_1dqp::PaperParams::enable_global_fallback)
      .def_readwrite("global_fallback_samples", &super_ik_tsinghua_1dqp::PaperParams::global_fallback_samples);

  py::class_<super_ik_tsinghua_1dqp::Solver>(m, "Solver")
      .def(py::init<super_ik_tsinghua_1dqp::NeroParams, super_ik_tsinghua_1dqp::PaperParams>(),
           py::arg("params") = super_ik_tsinghua_1dqp::NeroParams::Default(),
           py::arg("paper") = super_ik_tsinghua_1dqp::PaperParams{})
      .def("fk", &super_ik_tsinghua_1dqp::Solver::Fk, py::arg("q"))
      .def("fk_all", &super_ik_tsinghua_1dqp::Solver::FkAll, py::arg("q"))
      .def("ik_arm_angle", [](const super_ik_tsinghua_1dqp::Solver& s,
                              const super_ik_tsinghua_1dqp::Mat4& target,
                              const super_ik_tsinghua_1dqp::Vec7& q_prev,
                              double psi_prev) {
        auto r = s.IkArmAngle(target, q_prev, psi_prev);
        py::dict out;
        out["q_best"] = OptionalQToPy(r.q_best);
        out["report"] = ReportToDict(r.report);
        return out;
      }, py::arg("target"), py::arg("q_prev"), py::arg("psi_prev") = 0.0)
      .def("solve_trajectory", [](const super_ik_tsinghua_1dqp::Solver& s,
                                  const std::vector<super_ik_tsinghua_1dqp::Mat4>& targets,
                                  const super_ik_tsinghua_1dqp::Vec7& q_init,
                                  double psi_init) {
        auto tr = s.SolveTrajectory(targets, q_init, psi_init);
        py::list q_list;
        for (const auto& q : tr.q_list) q_list.append(OptionalQToPy(q));
        py::list psi_list;
        for (double psi : tr.psi_list) psi_list.append(psi);
        py::list reports;
        for (const auto& rep : tr.reports) reports.append(ReportToDict(rep));
        py::dict out;
        out["q_list"] = q_list;
        out["psi_list"] = psi_list;
        out["reports"] = reports;
        return out;
      }, py::arg("targets"), py::arg("q_init"), py::arg("psi_init") = 0.0);
}
