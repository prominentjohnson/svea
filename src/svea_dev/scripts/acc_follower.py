#! /usr/bin/env python3

import ast
import math

import numpy as np
from nav_msgs.msg import Odometry
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSHistoryPolicy, QoSReliabilityPolicy

from svea_core import rosonic as rx
from svea_core.controllers.pure_pursuit import PurePursuitController
from svea_core.interfaces import ActuationInterface, LocalizationInterface
from svea_core.utils import PlaceMarker


qos_profile = QoSProfile(
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.VOLATILE,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
)


class acc_follower(rx.Node):
    """Follow the waypoint path while adapting speed to a leader vehicle."""

    DELTA_TIME = 0.1
    TRAJ_LEN = 20

    points = rx.Parameter([
        '[-2.3, -7.1]',
        '[-0.166667, -3.966667]',
        '[1.966667, -0.833333]',
        '[4.1, 2.3]',
        '[6.233333, 5.433333]',
        '[8.366667, 8.566667]',
        '[10.5, 11.7]',
        '[8.9, 12.8]',
        '[7.3, 13.9]',
        '[5.7, 15.0]',
        '[3.583333, 11.833333]',
        '[1.466667, 8.666667]',
        '[-0.65, 5.5]',
        '[-2.766667, 2.333333]',
        '[-4.883333, -0.833333]',
        '[-7.0, -4.0]',
        '[-5.433333, -5.033333]',
        '[-3.866667, -6.066667]',
    ])
    leader_odom_topic = rx.Parameter('/vehicle1/odometry/local')
    target_velocity = rx.Parameter(0.6)
    max_velocity = rx.Parameter(1.0)
    min_gap = rx.Parameter(0.8)
    time_headway = rx.Parameter(1.2)
    gap_gain = rx.Parameter(0.35)
    relative_speed_gain = rx.Parameter(0.6)
    cruise_gain = rx.Parameter(0.1)
    max_accel = rx.Parameter(0.4)
    max_decel = rx.Parameter(0.8)
    leader_timeout = rx.Parameter(1.0)

    actuation = ActuationInterface()
    localizer = LocalizationInterface()
    mark = PlaceMarker()

    def on_startup(self):
        if isinstance(self.points[0], str):
            self._points = [ast.literal_eval(point) for point in self.points]
        else:
            self._points = self.points

        self.path_points = np.asarray(self._points, dtype=float)
        self.path_points_closed = np.vstack([self.path_points, self.path_points[0]])
        path_deltas = np.diff(self.path_points_closed, axis=0)
        self.path_segment_lengths = np.linalg.norm(path_deltas, axis=1)
        self.path_s = np.concatenate(([0.0], np.cumsum(self.path_segment_lengths)))
        self.path_length = self.path_s[-1]

        self.leader_state = None
        self.leader_last_time = None
        self.last_velocity_cmd = 0.0
        self.curr = 0

        self.controller = PurePursuitController()
        self.controller.target_velocity = self.target_velocity

        x, y, _, _ = self.localizer.get_state()
        self.goal = self._points[self.curr]
        self.mark.marker('follower_goal', 'blue', self.goal)
        self.update_traj(x, y)

        self.create_subscription(
            Odometry,
            self.leader_odom_topic,
            self.leader_odom_callback,
            qos_profile,
        )
        self.create_timer(self.DELTA_TIME, self.loop)
        self.get_logger().info(f"ACC follower subscribed to leader topic: {self.leader_odom_topic}")

    def leader_odom_callback(self, msg):
        self.leader_state = self.odom_to_state(msg)
        self.leader_last_time = self.get_clock().now()

    def loop(self):
        ego_state = self.localizer.get_state()
        x, y, _, _ = ego_state

        if self.controller.is_finished:
            self.update_goal()
            self.update_traj(x, y)

        steering = self.controller.compute_steering(ego_state)
        velocity = self.compute_acc_velocity(ego_state)
        self.actuation.send_control(steering, velocity)

    def compute_acc_velocity(self, ego_state):
        if self.leader_state is None or self.is_leader_stale():
            return self.rate_limit_velocity(0.0)

        ego_x, ego_y, _, ego_speed = ego_state
        leader_x, leader_y, _, leader_speed = self.leader_state

        gap = self.compute_path_gap((ego_x, ego_y), (leader_x, leader_y))
        desired_gap = self.min_gap + self.time_headway * max(ego_speed, 0.0)
        spacing_error = gap - desired_gap
        relative_speed = leader_speed - ego_speed

        acceleration_cmd = (
            self.gap_gain * spacing_error
            + self.relative_speed_gain * relative_speed
            + self.cruise_gain * (self.target_velocity - ego_speed)
        )
        acceleration_cmd = self.clip(acceleration_cmd, -self.max_decel, self.max_accel)

        target_speed = self.last_velocity_cmd + acceleration_cmd * self.DELTA_TIME
        target_speed = self.clip(target_speed, 0.0, self.max_velocity)
        return self.rate_limit_velocity(target_speed)

    def compute_path_gap(self, ego_xy, leader_xy):
        ego_s = self.project_to_path_s(ego_xy)
        leader_s = self.project_to_path_s(leader_xy)
        gap = leader_s - ego_s
        if gap < 0.0:
            gap += self.path_length
        return gap

    def project_to_path_s(self, xy):
        point = np.asarray(xy, dtype=float)
        best_distance = float('inf')
        best_s = 0.0

        for idx, segment_length in enumerate(self.path_segment_lengths):
            if segment_length <= 1e-9:
                continue

            start = self.path_points_closed[idx]
            end = self.path_points_closed[idx + 1]
            segment = end - start
            t = np.dot(point - start, segment) / (segment_length * segment_length)
            t = self.clip(t, 0.0, 1.0)
            projection = start + t * segment
            distance = np.linalg.norm(point - projection)

            if distance < best_distance:
                best_distance = distance
                best_s = self.path_s[idx] + t * segment_length

        return best_s

    def is_leader_stale(self):
        if self.leader_last_time is None:
            return True
        age = (self.get_clock().now() - self.leader_last_time).nanoseconds / 1e9
        return age > self.leader_timeout

    def rate_limit_velocity(self, target_velocity):
        delta = target_velocity - self.last_velocity_cmd
        max_up = self.max_accel * self.DELTA_TIME
        max_down = self.max_decel * self.DELTA_TIME
        delta = self.clip(delta, -max_down, max_up)
        self.last_velocity_cmd += delta
        return self.last_velocity_cmd

    def update_goal(self):
        self.curr += 1
        self.curr %= len(self._points)
        self.goal = self._points[self.curr]
        self.controller.is_finished = False
        self.mark.marker('follower_goal', 'blue', self.goal)

    def update_traj(self, x, y):
        xs = np.linspace(x, self.goal[0], self.TRAJ_LEN)
        ys = np.linspace(y, self.goal[1], self.TRAJ_LEN)
        self.controller.traj_x = xs
        self.controller.traj_y = ys

    @staticmethod
    def odom_to_state(msg):
        yaw = acc_follower.yaw_from_quaternion(msg.pose.pose.orientation)
        return (
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            yaw,
            msg.twist.twist.linear.x,
        )

    @staticmethod
    def yaw_from_quaternion(q):
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    @staticmethod
    def clip(value, lower, upper):
        return max(lower, min(upper, value))


if __name__ == '__main__':
    acc_follower.main()
