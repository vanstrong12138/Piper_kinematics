#!/usr/bin/env python3
"""Subscribe target pose, run 1D-QP IK, publish JointState."""

from collections import deque
import time
from typing import List, Optional

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import JointState
from tf2_ros import Buffer, TransformException, TransformListener

from super_ik_tsinghua_1dqp import PaperParams, Solver


class IkJointStatePublisher1DQP(Node):
    """Bridge node: interactive marker pose -> 1D-QP IK -> joint_states."""

    def __init__(self) -> None:
        super().__init__("ik_joint_state_publisher_1dqp")

        self.declare_parameter("target_pose_topic", "/ik_target_pose")
        self.declare_parameter("joint_state_topic", "/joint_states")
        self.declare_parameter("ik_base_frame", "base_link")
        self.declare_parameter("default_target_frame", "base_link")
        self.declare_parameter("republish_rate_hz", 20.0)
        self.declare_parameter("trajectory_window_size", 50)
        self.declare_parameter("delta_psi", 1e-3)
        self.declare_parameter("danger_threshold", 0.8)
        self.declare_parameter("feasible_scan_step", 0.01)
        self.declare_parameter("psi_seed_samples", 33)
        self.declare_parameter("psi_recover_window", 0.15)
        self.declare_parameter("psi_recover_samples", 25)
        self.declare_parameter("enable_global_fallback", True)
        self.declare_parameter("global_fallback_samples", 121)
        # TCP offset definition: pose of TCP frame expressed in link7 frame.
        self.declare_parameter("tcp_in_link7_xyz", [0.0, 0.0, 0.0])
        self.declare_parameter("tcp_in_link7_rpy", [0.0, 0.0, 0.0])
        self.declare_parameter("ik_perf_log_hz", 1.0)
        # 仿真场景: 启动即发布零位 joint_states 使 TF 立即可用;
        # 真实机械臂场景建议置 false (首帧发布实际关节角前不发布)。
        self.declare_parameter("initial_publish_zero", True)

        self._target_pose_topic = str(self.get_parameter("target_pose_topic").value)
        self._joint_state_topic = str(self.get_parameter("joint_state_topic").value)
        self._ik_base_frame = str(self.get_parameter("ik_base_frame").value)
        self._default_target_frame = str(
            self.get_parameter("default_target_frame").value
        )
        self._republish_rate_hz = float(self.get_parameter("republish_rate_hz").value)
        self._ik_perf_log_hz = max(
            0.1, float(self.get_parameter("ik_perf_log_hz").value)
        )
        self._initial_publish_zero = bool(
            self.get_parameter("initial_publish_zero").value
        )
        self._trajectory_window_size = max(
            1, int(self.get_parameter("trajectory_window_size").value)
        )
        tcp_xyz = [float(v) for v in self.get_parameter("tcp_in_link7_xyz").value]
        tcp_rpy = [float(v) for v in self.get_parameter("tcp_in_link7_rpy").value]
        self._tcp_xyz = tcp_xyz
        self._tcp_rpy = tcp_rpy
        self._T_link7_tcp = self._xyz_rpy_to_transform(self._tcp_xyz, self._tcp_rpy)
        self._T_tcp_link7 = np.linalg.inv(self._T_link7_tcp)

        self._paper = PaperParams()
        self._paper.delta_psi = float(self.get_parameter("delta_psi").value)
        self._paper.danger_threshold = float(
            self.get_parameter("danger_threshold").value
        )
        self._paper.feasible_scan_step = float(
            self.get_parameter("feasible_scan_step").value
        )
        self._paper.psi_seed_samples = int(self.get_parameter("psi_seed_samples").value)
        self._paper.psi_recover_window = float(
            self.get_parameter("psi_recover_window").value
        )
        self._paper.psi_recover_samples = int(
            self.get_parameter("psi_recover_samples").value
        )
        self._paper.enable_global_fallback = bool(
            self.get_parameter("enable_global_fallback").value
        )
        self._paper.global_fallback_samples = int(
            self.get_parameter("global_fallback_samples").value
        )

        self._joint_names: List[str] = [
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
            "joint6",
            "joint7",
        ]

        self._solver = Solver(paper=self._paper)
        self._q_prev = np.zeros(7, dtype=float)
        self._psi_prev = 0.0
        self._last_q: Optional[np.ndarray] = (
            np.zeros(7, dtype=float) if self._initial_publish_zero else None
        )
        self._target_history = deque(maxlen=self._trajectory_window_size)
        self._ik_solve_count = 0
        self._ik_solve_fail_count = 0
        self._ik_solve_dt_sum = 0.0
        self._ik_log_prev_t = time.perf_counter()

        self._joint_pub = self.create_publisher(JointState, self._joint_state_topic, 10)
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        # TF lookup 缓存: 拖动场景 target 帧基本恒为 ik_base_frame (恒等) 或
        # 少数静态帧; 每次消息都查 TF buffer 是高频下的主要开销, 缓存后大幅降低。
        # key: source_frame; value: (T_ikbase_source, 缓存时刻)
        self._tf_cache: dict = {}
        self._tf_cache_ttl = 0.1  # 秒; 动态帧在窗口内近似为常数 (拖动场景足够)
        self._target_sub = self.create_subscription(
            PoseStamped, self._target_pose_topic, self._on_target_pose, 10
        )

        timer_dt = 1.0 / max(self._republish_rate_hz, 1.0)
        self._timer = self.create_timer(timer_dt, self._republish_last_joint_state)

        self.get_logger().info(
            f"IK bridge (1DQP): {self._target_pose_topic} -> IK -> {self._joint_state_topic}"
        )
        self.get_logger().info(
            f"IK base frame: {self._ik_base_frame}, default target frame: {self._default_target_frame}"
        )
        self.get_logger().info(
            f"Using super_ik_tsinghua_1dqp, trajectory_window_size={self._trajectory_window_size}"
        )
        self.get_logger().info(
            "TCP offset in link7 frame: "
            f"xyz={self._tcp_xyz}, rpy={self._tcp_rpy} (rad)"
        )
        self.get_logger().info(
            f"IK perf logging enabled: {self._ik_perf_log_hz:.2f} Hz"
        )

    def _lookup_tf(self, source_frame: str, stamp_msg) -> np.ndarray:
        """带缓存的 TF lookup: 返回 T_ikbase_source。

        - source_frame == ik_base_frame: 恒等变换, 零开销 (拖动场景主路径)。
        - 其他 frame: 缓存最近一次结果 (TTL 0.1s), 动态帧在窗口内近似为常数;
          拖动/交互场景目标帧基本不变, 该近似对位姿精度无实际影响。
        """
        if source_frame == self._ik_base_frame:
            return np.eye(4)
        now = time.perf_counter()
        cached = self._tf_cache.get(source_frame)
        if cached is not None and (now - cached[1]) < self._tf_cache_ttl:
            return cached[0]
        stamp = (
            Time.from_msg(stamp_msg) if (stamp_msg.sec or stamp_msg.nanosec) else Time()
        )
        tf_msg = self._tf_buffer.lookup_transform(
            self._ik_base_frame,
            source_frame,
            stamp,
            timeout=Duration(seconds=0.1),
        )
        T = self._transform_stamped_to_matrix(tf_msg)
        self._tf_cache[source_frame] = (T, now)
        return T

    def _on_target_pose(self, msg: PoseStamped) -> None:
        source_frame = (
            msg.header.frame_id if msg.header.frame_id else self._default_target_frame
        )
        T_source_target = self._pose_to_transform(msg.pose)
        if T_source_target is None:
            self.get_logger().warning(
                "Ignore target pose: invalid quaternion norm too small"
            )
            return

        try:
            T_ikbase_source = self._lookup_tf(source_frame, msg.header.stamp)
        except TransformException as exc:
            self.get_logger().warning(
                f"TF lookup failed ({source_frame} -> {self._ik_base_frame}): {exc}"
            )
            return
        # Incoming target is interpreted as desired TCP pose.
        # IK solver target is link7 pose, so convert by T_link7 = T_tcp * inv(T_link7_tcp).
        T_ikbase_tcp = T_ikbase_source @ T_source_target
        T_ikbase_link7 = T_ikbase_tcp @ self._T_tcp_link7
        self._target_history.append(T_ikbase_link7.copy())

        t0 = time.perf_counter()
        out = self._solver.solve_trajectory(
            list(self._target_history),
            self._q_prev,
            psi_init=float(self._psi_prev),
        )
        solve_dt = time.perf_counter() - t0
        self._ik_solve_count += 1
        self._ik_solve_dt_sum += solve_dt
        q_list = out.get("q_list", [])
        psi_list = out.get("psi_list", [])
        reports = out.get("reports", [])
        q_best = q_list[-1] if q_list else None
        psi_best = psi_list[-1] if psi_list else self._psi_prev
        report = reports[-1] if reports else {}
        if q_best is None:
            self._ik_solve_fail_count += 1
            self.get_logger().warning(
                "IK failed "
                f"(method={report.get('method')}, psi_prev={report.get('psi_prev')}, "
                f"fallback_used={report.get('fallback_used')})"
            )
            self._maybe_log_ik_perf()
            return

        q_best_np = np.array(q_best, dtype=float).reshape(7)
        self._q_prev = q_best_np
        self._psi_prev = float(psi_best)
        self._last_q = q_best_np
        self._publish_joint_state(q_best_np, self._ik_base_frame)
        self._maybe_log_ik_perf()

    def _maybe_log_ik_perf(self) -> None:
        now = time.perf_counter()
        log_period = 1.0 / self._ik_perf_log_hz
        elapsed = now - self._ik_log_prev_t
        if elapsed < log_period or self._ik_solve_count == 0:
            return
        avg_solve_ms = 1000.0 * self._ik_solve_dt_sum / self._ik_solve_count
        solve_rate_hz = self._ik_solve_count / elapsed
        fail_ratio = 100.0 * self._ik_solve_fail_count / self._ik_solve_count
        self.get_logger().info(
            "IK speed: "
            f"{solve_rate_hz:.1f} Hz, avg solve {avg_solve_ms:.3f} ms, "
            f"fail {self._ik_solve_fail_count}/{self._ik_solve_count} ({fail_ratio:.1f}%)"
        )
        self._ik_solve_count = 0
        self._ik_solve_fail_count = 0
        self._ik_solve_dt_sum = 0.0
        self._ik_log_prev_t = now

    def _publish_joint_state(self, q: np.ndarray, frame_id: str) -> None:
        out = JointState()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = frame_id if frame_id else self._ik_base_frame
        out.name = self._joint_names
        out.position = [float(v) for v in q.tolist()]
        out.velocity = []
        out.effort = []
        self._joint_pub.publish(out)

    def _republish_last_joint_state(self) -> None:
        if self._last_q is None:
            return
        self._publish_joint_state(self._last_q, self._ik_base_frame)

    @staticmethod
    def _quat_to_rot(
        qx: float, qy: float, qz: float, qw: float
    ) -> Optional[np.ndarray]:
        n = qx * qx + qy * qy + qz * qz + qw * qw
        if n < 1e-12:
            return None
        s = 2.0 / n
        xx, yy, zz = qx * qx * s, qy * qy * s, qz * qz * s
        xy, xz, yz = qx * qy * s, qx * qz * s, qy * qz * s
        wx, wy, wz = qw * qx * s, qw * qy * s, qw * qz * s
        R = np.eye(3, dtype=float)
        R[0, 0] = 1.0 - (yy + zz)
        R[0, 1] = xy - wz
        R[0, 2] = xz + wy
        R[1, 0] = xy + wz
        R[1, 1] = 1.0 - (xx + zz)
        R[1, 2] = yz - wx
        R[2, 0] = xz - wy
        R[2, 1] = yz + wx
        R[2, 2] = 1.0 - (xx + yy)
        return R

    def _pose_to_transform(self, pose) -> Optional[np.ndarray]:
        R = self._quat_to_rot(
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        )
        if R is None:
            return None
        T = np.eye(4, dtype=float)
        T[:3, :3] = R
        T[0, 3] = float(pose.position.x)
        T[1, 3] = float(pose.position.y)
        T[2, 3] = float(pose.position.z)
        return T

    def _transform_stamped_to_matrix(self, tf_msg) -> np.ndarray:
        R = self._quat_to_rot(
            tf_msg.transform.rotation.x,
            tf_msg.transform.rotation.y,
            tf_msg.transform.rotation.z,
            tf_msg.transform.rotation.w,
        )
        if R is None:
            raise TransformException("Invalid TF quaternion")
        T = np.eye(4, dtype=float)
        T[:3, :3] = R
        T[0, 3] = float(tf_msg.transform.translation.x)
        T[1, 3] = float(tf_msg.transform.translation.y)
        T[2, 3] = float(tf_msg.transform.translation.z)
        return T

    @staticmethod
    def _xyz_rpy_to_transform(xyz: List[float], rpy: List[float]) -> np.ndarray:
        x, y, z = xyz
        roll, pitch, yaw = rpy
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)
        Rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=float)
        Ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=float)
        Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=float)
        T = np.eye(4, dtype=float)
        T[:3, :3] = Rz @ Ry @ Rx
        T[0, 3] = x
        T[1, 3] = y
        T[2, 3] = z
        return T


def main(args=None) -> None:
    rclpy.init(args=args)
    node = IkJointStatePublisher1DQP()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
