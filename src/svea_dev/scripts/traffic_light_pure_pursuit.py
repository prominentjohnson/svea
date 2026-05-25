#! /usr/bin/env python3

import ast

import numpy as np
from std_msgs.msg import String

from svea_core import rosonic as rx
from svea_core.controllers.pure_pursuit import PurePursuitController
from svea_core.interfaces import ActuationInterface, LocalizationInterface
from svea_core.utils import PlaceMarker


class traffic_light_pure_pursuit(rx.Node):
    """Pure pursuit leader that stops before a red traffic light."""

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
    target_velocity = rx.Parameter(0.6)
    traffic_state_topic = rx.Parameter('/light_a/state')
    light_x = rx.Parameter(3.34)
    light_y = rx.Parameter(0.90)
    stop_offset = rx.Parameter(0.7)
    slowdown_distance = rx.Parameter(2.0)
    stop_states = rx.Parameter(['Rd', 'YR'])

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

        self.light_s = self.project_to_path_s((self.light_x, self.light_y))
        self.stop_s = (self.light_s - self.stop_offset) % self.path_length
        self.light_state = None
        self.curr = 0

        self.controller = PurePursuitController()
        self.controller.target_velocity = self.target_velocity

        x, y, _, _ = self.localizer.get_state()
        self.goal = self._points[self.curr]
        self.mark.marker('leader_goal', 'blue', self.goal)
        self.update_traj(x, y)

        self.create_subscription(String, self.traffic_state_topic, self.traffic_state_callback, 1)
        self.create_timer(self.DELTA_TIME, self.loop)
        self.get_logger().info(
            f"Traffic-light pure pursuit subscribed to {self.traffic_state_topic}; "
            f"stop line is {self.stop_offset:.2f} m before light"
        )

    def traffic_state_callback(self, msg):
        self.light_state = msg.data

    def loop(self):
        state = self.localizer.get_state()
        x, y, _, _ = state

        if self.controller.is_finished:
            self.update_goal()
            self.update_traj(x, y)

        self.controller.target_velocity = self.compute_allowed_velocity(state)
        steering, velocity = self.controller.compute_control(state)
        self.actuation.send_control(steering, velocity)

    def compute_allowed_velocity(self, state):
        if self.light_state not in self.stop_states:
            return self.target_velocity

        ego_s = self.project_to_path_s((state[0], state[1]))
        distance_to_stop = self.stop_s - ego_s
        if distance_to_stop < 0.0:
            distance_to_stop += self.path_length

        if distance_to_stop > self.slowdown_distance:
            return self.target_velocity

        if distance_to_stop <= 0.15:
            return 0.0

        return self.target_velocity * distance_to_stop / self.slowdown_distance

    def update_goal(self):
        self.curr += 1
        self.curr %= len(self._points)
        self.goal = self._points[self.curr]
        self.controller.is_finished = False
        self.mark.marker('leader_goal', 'blue', self.goal)

    def update_traj(self, x, y):
        xs = np.linspace(x, self.goal[0], self.TRAJ_LEN)
        ys = np.linspace(y, self.goal[1], self.TRAJ_LEN)
        self.controller.traj_x = xs
        self.controller.traj_y = ys

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

    @staticmethod
    def clip(value, lower, upper):
        return max(lower, min(upper, value))


if __name__ == '__main__':
    traffic_light_pure_pursuit.main()
