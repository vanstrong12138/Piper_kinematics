import numpy as np

from super_ik_tsinghua_1dqp import Solver


def max_jump(q_list):
    jumps = []
    for i in range(1, len(q_list)):
        if q_list[i] is None or q_list[i - 1] is None:
            continue
        dq = (q_list[i] - q_list[i - 1] + np.pi) % (2.0 * np.pi) - np.pi
        jumps.append(float(np.max(np.abs(dq))))
    return float(np.max(jumps)) if jumps else float("nan")


def main():
    np.set_printoptions(precision=6, suppress=True)
    solver = Solver()

    q_true = np.array([0.2, -0.4, 0.5, 0.3, -0.2, 0.1, 0.2], dtype=float)
    q_prev = q_true.copy()
    q_prev[0] -= 0.05
    q_prev[2] += 0.03

    # Single-point IK
    T = solver.fk(q_true)
    ik_out = solver.ik_arm_angle(T, q_prev, psi_prev=0.0)
    print("=== Single-point IK ===")
    print("report:", ik_out["report"])
    print("q_best:", ik_out["q_best"])
    print()

    # Build a small circular target trajectory
    traj = []
    base = T.copy()
    for i in range(60):
        s = i / 59.0
        t = base.copy()
        t[0, 3] += 0.05 * np.cos(2.0 * np.pi * s)
        t[1, 3] += 0.05 * np.sin(2.0 * np.pi * s)
        traj.append(t)

    out = solver.solve_trajectory(traj, q_true, psi_init=0.0)
    ok = sum(1 for q in out["q_list"] if q is not None)
    print("=== Trajectory ===")
    print(f"success: {ok}/{len(traj)}, max_jump(rad): {max_jump(out['q_list']):.6f}")


if __name__ == "__main__":
    main()

