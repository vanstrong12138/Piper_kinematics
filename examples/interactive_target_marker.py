#!/usr/bin/env python3
"""MoveIt-like interactive marker for specifying a target pose."""

from typing import Optional

import rclpy
from geometry_msgs.msg import Pose, PoseStamped
from interactive_markers.interactive_marker_server import InteractiveMarkerServer
from rclpy.node import Node
from rclpy.time import Time
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import InteractiveMarker
from visualization_msgs.msg import InteractiveMarkerControl
from visualization_msgs.msg import Marker


class InteractiveTargetPoseTool(Node):
    """Interactive marker tool that publishes dragged target poses."""

    def __init__(self) -> None:
        super().__init__("interactive_target_pose_tool")

        self.declare_parameter("frame_id", "base_link")
        self.declare_parameter("ee_link", "link7")
        self.declare_parameter("marker_scale", 0.25)
        self.declare_parameter("publish_topic", "/ik_target_pose")
        self.declare_parameter("publish_rate_hz", 2000.0)

        self._frame_id = str(self.get_parameter("frame_id").value)
        self._ee_link = str(self.get_parameter("ee_link").value)
        self._marker_scale = float(self.get_parameter("marker_scale").value)
        self._publish_topic = str(self.get_parameter("publish_topic").value)
        self._publish_rate_hz = float(self.get_parameter("publish_rate_hz").value)

        self._pose_pub = self.create_publisher(PoseStamped, self._publish_topic, 10)
        self._server = InteractiveMarkerServer(self, "ik_target_marker")
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        self._marker_name = "ik_goal"
        self._current_pose: Optional[Pose] = None
        self._synced_from_link7 = False
        self._default_pose = Pose()
        self._default_pose.position.x = 0.35
        self._default_pose.position.y = 0.0
        self._default_pose.position.z = 0.35
        self._default_pose.orientation.w = 1.0

        self._create_interactive_marker()

        timer_dt = 1.0 / max(self._publish_rate_hz, 1.0)
        self._timer = self.create_timer(timer_dt, self._publish_pose_timer)
        self._init_pose_timer = self.create_timer(
            0.2, self._try_sync_initial_pose_from_tf
        )
        self.get_logger().info(
            f"Interactive target marker ready. Drag in RViz and read from {self._publish_topic}"
        )

    def _create_interactive_marker(self) -> None:
        int_marker = InteractiveMarker()
        int_marker.header.frame_id = self._frame_id
        int_marker.name = self._marker_name
        int_marker.description = "IK Target Pose"
        int_marker.scale = self._marker_scale
        int_marker.pose = self._default_pose

        # Visible axis body
        body_control = InteractiveMarkerControl()
        body_control.always_visible = True
        body_control.interaction_mode = InteractiveMarkerControl.NONE

        body_control.markers.append(self._make_axis_marker("x"))
        body_control.markers.append(self._make_axis_marker("y"))
        body_control.markers.append(self._make_axis_marker("z"))
        int_marker.controls.append(body_control)

        # 6DoF controls (rotate + move on x/y/z)
        int_marker.controls.append(self._make_control("rotate_x", 1.0, 0.0, 0.0, True))
        int_marker.controls.append(self._make_control("move_x", 1.0, 0.0, 0.0, False))
        int_marker.controls.append(self._make_control("rotate_y", 0.0, 1.0, 0.0, True))
        int_marker.controls.append(self._make_control("move_y", 0.0, 1.0, 0.0, False))
        int_marker.controls.append(self._make_control("rotate_z", 0.0, 0.0, 1.0, True))
        int_marker.controls.append(self._make_control("move_z", 0.0, 0.0, 1.0, False))

        self._server.insert(int_marker, feedback_callback=self._on_feedback)
        self._server.applyChanges()
        self._current_pose = int_marker.pose

    @staticmethod
    def _make_control(
        name: str, x: float, y: float, z: float, is_rotate: bool
    ) -> InteractiveMarkerControl:
        control = InteractiveMarkerControl()
        control.name = name
        control.orientation.w = 1.0
        control.orientation.x = x
        control.orientation.y = y
        control.orientation.z = z
        control.interaction_mode = (
            InteractiveMarkerControl.ROTATE_AXIS
            if is_rotate
            else InteractiveMarkerControl.MOVE_AXIS
        )
        return control

    def _make_axis_marker(self, axis: str) -> Marker:
        marker = Marker()
        marker.type = Marker.ARROW
        marker.scale.x = self._marker_scale * 0.45
        marker.scale.y = self._marker_scale * 0.08
        marker.scale.z = self._marker_scale * 0.08
        marker.color.a = 0.95

        if axis == "x":
            marker.color.r = 1.0
            marker.color.g = 0.1
            marker.color.b = 0.1
            marker.pose.orientation.w = 1.0
        elif axis == "y":
            marker.color.r = 0.1
            marker.color.g = 1.0
            marker.color.b = 0.1
            marker.pose.orientation.w = 0.70710678
            marker.pose.orientation.z = 0.70710678
        else:
            marker.color.r = 0.1
            marker.color.g = 0.3
            marker.color.b = 1.0
            marker.pose.orientation.w = 0.70710678
            marker.pose.orientation.y = -0.70710678

        return marker

    def _on_feedback(self, feedback) -> None:
        self._current_pose = feedback.pose
        self._publish_pose(feedback.pose)

    def _publish_pose_timer(self) -> None:
        if self._current_pose is None:
            return
        self._publish_pose(self._current_pose)

    def _publish_pose(self, pose: Pose) -> None:
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self._frame_id
        msg.pose = pose
        self._pose_pub.publish(msg)

    def _try_sync_initial_pose_from_tf(self) -> None:
        if self._synced_from_link7:
            return
        try:
            tf_msg = self._tf_buffer.lookup_transform(
                self._frame_id,
                self._ee_link,
                Time(),
            )
        except TransformException:
            return

        pose = Pose()
        pose.position.x = tf_msg.transform.translation.x
        pose.position.y = tf_msg.transform.translation.y
        pose.position.z = tf_msg.transform.translation.z
        pose.orientation = tf_msg.transform.rotation

        self._server.setPose(self._marker_name, pose)
        self._server.applyChanges()
        self._current_pose = pose
        self._synced_from_link7 = True
        self.get_logger().info(
            f"Marker initial pose synced from TF: {self._frame_id} -> {self._ee_link}"
        )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = InteractiveTargetPoseTool()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
