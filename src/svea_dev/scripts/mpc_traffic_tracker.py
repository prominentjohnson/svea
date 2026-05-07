#! /usr/bin/env python3

import numpy as np

from rclpy.clock import Clock
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from std_msgs.msg import Float32, Float32MultiArray, String

from svea_core import rosonic as rx
from svea_core.controllers.mpc import MPC
from svea_core.interfaces import ActuationInterface, LocalizationInterface


qos_trajectory = QoSProfile(
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
)


class mpc_traffic_tracker(rx.Node):
    """Track a time-parameterized trajectory from planning_traffic_light.py.

    Expected trajectory rows:
        [t, x, y, yaw, v, a]

    The MPC receives a horizon of [x, y, yaw, v] reference states sampled by
    time, not by nearest path index. This preserves traffic-light waiting
    segments where position is constant and v_ref is zero.
    """

    mpc_freq = rx.Parameter(10.0)
    trajectory_topic = rx.Parameter("/planned_trajectory/data")
    traffic_state_topic = rx.Parameter("/light_a/state")
    traffic_time_left_topic = rx.Parameter("/light_a/time_left")
    sync_with_traffic_light = rx.Parameter(True)
    sync_state = rx.Parameter("Rd")
    sync_state_duration = rx.Parameter(5.0)
    sync_tolerance = rx.Parameter(0.3)
    hold_position_tolerance = rx.Parameter(0.25)

    actuation = ActuationInterface()
    localizer = LocalizationInterface()

    trajectory_sub = rx.Subscriber(Float32MultiArray, trajectory_topic, qos_trajectory)
    traffic_state_sub = rx.Subscriber(String, traffic_state_topic)
    traffic_time_left_sub = rx.Subscriber(Float32, traffic_time_left_topic)

    def on_startup(self):
        self.controller = MPC(self)
        self.mpc_dt = 1.0 / float(self.mpc_freq)
        self.reference_dt = float(self.controller.dt)
        self.prediction_horizon = self.controller.current_horizon

        self.trajectory = None
        self.trajectory_start_time = None
        self.last_control_time = Clock().now().nanoseconds / 1e9
        self.steering = 0.0
        self.velocity = 0.0
        self.started = False
        self.finished = False
        self.execution_started = False
        self.light_state = None
        self.light_time_left = None

        self.create_timer(self.mpc_dt, self.loop)
        self.get_logger().info("Waiting for /planned_trajectory/data")

    @trajectory_sub
    def trajectory_callback(self, msg):
        data = np.array(msg.data, dtype=float)
        if data.size == 0 or data.size % 6 != 0:
            self.get_logger().error("Received malformed planned trajectory")
            return

        trajectory = data.reshape((-1, 6))
        if trajectory.shape[0] < self.prediction_horizon + 1:
            self.get_logger().error("Planned trajectory is shorter than the MPC horizon")
            return

        trajectory[:, 3] = np.unwrap(trajectory[:, 3])
        self.trajectory = trajectory
        self.trajectory_start_time = None
        self.last_control_time = Clock().now().nanoseconds / 1e9
        self.finished = False
        self.started = False
        self.execution_started = False
        self.get_logger().info(
            f"Loaded planned trajectory with {trajectory.shape[0]} points, "
            f"duration {trajectory[-1, 0]:.2f}s"
        )

    @traffic_state_sub
    def traffic_state_callback(self, msg):
        self.light_state = msg.data

    @traffic_time_left_sub
    def traffic_time_left_callback(self, msg):
        self.light_time_left = msg.data

    def loop(self):
        if self.trajectory is None:
            self.actuation.send_control(0.0, 0.0)
            return

        now = Clock().now().nanoseconds / 1e9
        if not self.execution_started:
            self.actuation.send_control(0.0, 0.0)
            if self.traffic_phase_is_aligned():
                self.trajectory_start_time = now
                self.last_control_time = now
                self.execution_started = True
                self.get_logger().info("Traffic-light phase aligned; starting trajectory tracking")
            return

        elapsed = now - self.trajectory_start_time
        measured_dt = max(0.0, now - self.last_control_time)
        self.last_control_time = now

        state = self.localizer.get_state()
        if not self.started:
            self.velocity = state[3]
            self.started = True

        if elapsed >= self.trajectory[-1, 0]:
            self.finished = True
            self.actuation.send_control(0.0, 0.0)
            return

        reference = self.sample_reference(elapsed)
        try:
            steering_rate, acceleration = self.controller.compute_control(
                [state[0], state[1], state[2], state[3], self.steering],
                reference,
            )
        except Exception as exc:
            self.get_logger().warn(f"MPC solve failed, stopping: {exc}")
            self.actuation.send_control(0.0, 0.0)
            return

        self.steering += steering_rate * measured_dt
        self.velocity += acceleration * measured_dt

        if self.should_hold_stop(elapsed, state):
            self.actuation.send_control(0.0, 0.0)
            return

        self.actuation.send_control(self.steering, self.velocity)

    def traffic_phase_is_aligned(self):
        if not self.sync_with_traffic_light:
            return True
        if self.light_state is None or self.light_time_left is None:
            return False
        return (
            self.light_state == self.sync_state
            and self.light_time_left >= float(self.sync_state_duration) - float(self.sync_tolerance)
        )

    def sample_reference(self, elapsed):
        query_times = elapsed + self.reference_dt * np.arange(self.prediction_horizon + 1)
        t = self.trajectory[:, 0]

        x_ref = np.interp(query_times, t, self.trajectory[:, 1])
        y_ref = np.interp(query_times, t, self.trajectory[:, 2])
        yaw_ref = np.interp(query_times, t, self.trajectory[:, 3])
        v_ref = np.interp(query_times, t, self.trajectory[:, 4])

        return np.vstack((x_ref, y_ref, yaw_ref, v_ref))

    def should_hold_stop(self, elapsed, state):
        ref_now = self.sample_point(elapsed)
        distance = np.linalg.norm(np.array(state[:2]) - ref_now[1:3])
        return ref_now[4] <= 0.02 and distance <= float(self.hold_position_tolerance)

    def sample_point(self, elapsed):
        t = self.trajectory[:, 0]
        return np.array(
            [
                elapsed,
                np.interp(elapsed, t, self.trajectory[:, 1]),
                np.interp(elapsed, t, self.trajectory[:, 2]),
                np.interp(elapsed, t, self.trajectory[:, 3]),
                np.interp(elapsed, t, self.trajectory[:, 4]),
                np.interp(elapsed, t, self.trajectory[:, 5]),
            ]
        )


if __name__ == "__main__":
    mpc_traffic_tracker.main()
