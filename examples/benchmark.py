import time
import numpy as np

from super_ik_tsinghua_1dqp import Solver


def wrap_to_pi(x):
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def max_jump(q_list):
    jumps = []
    for i in range(1, len(q_list)):
        if q_list[i] is None or q_list[i - 1] is None:
            continue
        dq = wrap_to_pi(q_list[i] - q_list[i - 1])
        jumps.append(float(np.max(np.abs(dq))))
    return float(np.max(jumps)) if jumps else float("nan")


def percentile_ms(arr_ms, p):
    if len(arr_ms) == 0:
        return float("nan")
    return float(np.percentile(np.asarray(arr_ms, dtype=float), p))


def benchmark_single_ik(solver: Solver, n_tests: int = 1000, seed: int = 42):
    rng = np.random.default_rng(seed)
    lo = np.array([-2.705261, -1.745330, -2.757621, -1.012291, -2.757621, -0.733039, -1.570797], dtype=float)
    hi = np.array([2.705261, 1.745330, 2.757621, 2.146755, 2.757621, 0.959932, 1.570797], dtype=float)

    times_ms = []
    succ = 0
    pose_errs = []

    t_all0 = time.perf_counter()
    for _ in range(n_tests):
        q_true = rng.uniform(lo, hi)
        q_prev = np.clip(q_true + rng.normal(0.0, 0.05, size=7), lo, hi)
        T = solver.fk(q_true)

        t0 = time.perf_counter()
        out = solver.ik_arm_angle(T, q_prev, psi_prev=0.0)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        times_ms.append(dt_ms)

        q_best = out["q_best"]
        if q_best is not None:
            succ += 1
            T_rec = solver.fk(q_best)
            pose_errs.append(float(np.linalg.norm(T_rec - T)))

    total_s = time.perf_counter() - t_all0
    tps = n_tests / total_s if total_s > 0.0 else float("nan")
    return {
        "n_tests": n_tests,
        "success": succ,
        "success_rate": succ / max(1, n_tests),
        "mean_ms": float(np.mean(times_ms)) if times_ms else float("nan"),
        "p50_ms": percentile_ms(times_ms, 50),
        "p90_ms": percentile_ms(times_ms, 90),
        "p99_ms": percentile_ms(times_ms, 99),
        "max_ms": float(np.max(times_ms)) if times_ms else float("nan"),
        "throughput_hz": tps,
        "pose_err_normF_mean": float(np.mean(pose_errs)) if pose_errs else float("nan"),
    }


def benchmark_trajectory(solver: Solver, n_pts: int = 200):
    q_seed = np.array([0.2, -0.4, 0.5, 0.3, -0.2, 0.1, 0.2], dtype=float)
    T_base = solver.fk(q_seed)
    traj = []
    for i in range(n_pts):
        s = i / max(1, n_pts - 1)
        T = T_base.copy()
        T[0, 3] += 0.06 * np.cos(2.0 * np.pi * s)
        T[1, 3] += 0.06 * np.sin(2.0 * np.pi * s)
        traj.append(T)

    t0 = time.perf_counter()
    out = solver.solve_trajectory(traj, q_seed, psi_init=0.0)
    total_ms = (time.perf_counter() - t0) * 1000.0
    ok = sum(1 for q in out["q_list"] if q is not None)
    return {
        "n_pts": n_pts,
        "success": ok,
        "success_rate": ok / max(1, n_pts),
        "total_ms": total_ms,
        "avg_ms_per_pt": total_ms / max(1, n_pts),
        "max_jump_rad": max_jump(out["q_list"]),
    }


def main():
    np.set_printoptions(precision=6, suppress=True)
    solver = Solver()

    print("==== Benchmark: Single IK ====")
    s = benchmark_single_ik(solver, n_tests=1000, seed=42)
    print(
        f"tests={s['n_tests']}, success={s['success']} ({s['success_rate']*100:.2f}%), "
        f"mean={s['mean_ms']:.3f}ms, p50={s['p50_ms']:.3f}ms, "
        f"p90={s['p90_ms']:.3f}ms, p99={s['p99_ms']:.3f}ms, max={s['max_ms']:.3f}ms, "
        f"throughput={s['throughput_hz']:.2f} Hz"
    )
    print(f"pose_err_normF_mean={s['pose_err_normF_mean']:.3e}")
    print()

    print("==== Benchmark: Trajectory ====")
    t = benchmark_trajectory(solver, n_pts=200)
    print(
        f"success={t['success']}/{t['n_pts']} ({t['success_rate']*100:.2f}%), "
        f"total={t['total_ms']:.2f}ms, avg={t['avg_ms_per_pt']:.3f}ms/pt, "
        f"max_jump={t['max_jump_rad']:.6f}rad"
    )


if __name__ == "__main__":
    main()

