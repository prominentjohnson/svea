#! /usr/bin/env python3

import math
import os
from dataclasses import dataclass

import casadi as ca
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
from visualization_msgs.msg import Marker
import tf_transformations as tf


@dataclass
class PlanResult:
    objective: float
    final_time: float
    crossing_time: float
    crossing_index: int
    green_window: tuple[float, float]
    trajectory: np.ndarray
    station_trajectory: np.ndarray
    light_s: float
    road_length: float


class PlanningTrafficLight(Node):
    """Global 1D traffic-light planner projected onto a straight map segment.

    The optimized trajectory is published as rows with fields:
        [t, x, y, yaw, v, a]
    on /planned_trajectory/data. A nav_msgs/Path is also published for
    visualization on /planned_trajectory/path.
    """

    FIELDS = ("t", "x", "y", "yaw", "v", "a")

    def __init__(self):
        super().__init__("planning_traffic_light")

        self.declare_parameter("start_x", -2.3)
        self.declare_parameter("start_y", -7.1)
        self.declare_parameter("goal_x", 10.5)
        self.declare_parameter("goal_y", 11.7)
        self.declare_parameter("light_position", 0.4)
        self.declare_parameter("light_position_is_ratio", True)

        self.declare_parameter("num_steps", 70)
        self.declare_parameter("candidate_stride", 2)
        self.declare_parameter("min_final_time", 2.0)
        self.declare_parameter("max_final_time", 22.0)
        self.declare_parameter("start_velocity", 0.0)
        self.declare_parameter("goal_velocity", 0.0)
        self.declare_parameter("velocity_max", 1.2)
        self.declare_parameter("acceleration_min", -0.6)
        self.declare_parameter("acceleration_max", 0.6)

        self.declare_parameter("red_time", 5.0)
        self.declare_parameter("yellow_time", 2.0)
        self.declare_parameter("green_time", 5.0)
        self.declare_parameter("allow_yellow_crossing", False)
        self.declare_parameter("green_entry_margin", 1.0)
        self.declare_parameter("green_exit_margin", 0.5)

        self.declare_parameter("time_weight", 1.0)
        self.declare_parameter("energy_weight", 0.2)
        self.declare_parameter("jerk_weight", 0.02)
        self.declare_parameter("output_csv", "")
        self.declare_parameter("output_dir", "/tmp/svea_traffic_plans")
        self.declare_parameter("speed_plot_x", -7.4)
        self.declare_parameter("speed_plot_y", -18.0)
        self.declare_parameter("speed_plot_time_scale", 0.25)
        self.declare_parameter("speed_plot_speed_scale", 2.0)

        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.path_pub = self.create_publisher(Path, "/planned_trajectory/path", qos)
        self.data_pub = self.create_publisher(Float32MultiArray, "/planned_trajectory/data", qos)
        self.speed_plot_pub = self.create_publisher(Marker, "/planned_trajectory/speed_plot", qos)

        self.plan = None
        self._planning_timer = self.create_timer(0.5, self.plan_once)

    def plan_once(self):
        self._planning_timer.cancel()
        self.get_logger().info("Starting global traffic-light trajectory optimization...")
        self.plan = self.solve_best_plan()
        self.publish_plan()
        self.write_csv_if_requested()
        try:
            self.write_plots()
        except Exception as exc:
            self.get_logger().error(f"Failed to write plan plots: {exc}")

        self.get_logger().info(
            "Best plan: T=%.2fs, crossing t=%.2fs at index %d, green window=[%.2f, %.2f]"
            % (
                self.plan.final_time,
                self.plan.crossing_time,
                self.plan.crossing_index,
                self.plan.green_window[0],
                self.plan.green_window[1],
            )
        )

    def solve_best_plan(self) -> PlanResult:
        start = self.get_xy("start")
        goal = self.get_xy("goal")
        road = goal - start
        length = float(np.linalg.norm(road))
        if length <= 1e-6:
            raise ValueError("Start and goal must be separated.")

        light_position = self.get_parameter("light_position").value
        if self.get_parameter("light_position_is_ratio").value:
            light_s = float(light_position) * length
        else:
            light_s = float(light_position)
        light_s = min(max(light_s, 0.0), length)

        num_steps = int(self.get_parameter("num_steps").value)
        candidate_stride = max(1, int(self.get_parameter("candidate_stride").value))
        windows = self.green_windows()
        candidate_count = len(windows) * len(range(1, num_steps - 1, candidate_stride))
        self.get_logger().info(
            f"Road length={length:.2f} m, light_s={light_s:.2f} m, "
            f"testing {candidate_count} traffic-light crossing candidates"
        )

        best = None
        for window in windows:
            self.get_logger().info(f"Trying green window [{window[0]:.2f}, {window[1]:.2f}]")
            for crossing_index in range(1, num_steps - 1, candidate_stride):
                try:
                    result = self.solve_candidate(length, light_s, crossing_index, window)
                except RuntimeError:
                    continue
                if best is None or result.objective < best.objective:
                    best = result

        if best is None:
            raise RuntimeError(
                "No feasible trajectory found. Try increasing max_final_time, velocity_max, "
                "or acceleration limits."
            )

        unit = road / length
        yaw = math.atan2(unit[1], unit[0])
        t = best.trajectory[:, 0]
        s = best.trajectory[:, 1]
        v = best.trajectory[:, 2]
        a = best.trajectory[:, 3]
        xy = start[None, :] + s[:, None] * unit[None, :]
        full_traj = np.column_stack((t, xy[:, 0], xy[:, 1], np.full_like(t, yaw), v, a))

        return PlanResult(
            objective=best.objective,
            final_time=best.final_time,
            crossing_time=best.crossing_time,
            crossing_index=best.crossing_index,
            green_window=best.green_window,
            trajectory=full_traj,
            station_trajectory=best.trajectory,
            light_s=light_s,
            road_length=length,
        )

    def solve_candidate(self, road_length: float, light_s: float, crossing_index: int, green_window):
        num_steps = int(self.get_parameter("num_steps").value)
        min_final_time = float(self.get_parameter("min_final_time").value)
        max_final_time = float(self.get_parameter("max_final_time").value)
        v0 = float(self.get_parameter("start_velocity").value)
        vf = float(self.get_parameter("goal_velocity").value)
        vmax = float(self.get_parameter("velocity_max").value)
        amin = float(self.get_parameter("acceleration_min").value)
        amax = float(self.get_parameter("acceleration_max").value)

        w_time = float(self.get_parameter("time_weight").value)
        w_energy = float(self.get_parameter("energy_weight").value)
        w_jerk = float(self.get_parameter("jerk_weight").value)

        opti = ca.Opti()
        s = opti.variable(num_steps + 1)
        v = opti.variable(num_steps + 1)
        a = opti.variable(num_steps)
        final_time = opti.variable()
        dt = final_time / num_steps

        opti.subject_to(min_final_time <= final_time)
        opti.subject_to(final_time <= max_final_time)

        opti.subject_to(s[0] == 0.0)
        opti.subject_to(v[0] == v0)
        opti.subject_to(s[num_steps] == road_length)
        opti.subject_to(v[num_steps] == vf)
        opti.subject_to(opti.bounded(0.0, s, road_length))
        opti.subject_to(opti.bounded(0.0, v, vmax))
        opti.subject_to(opti.bounded(amin, a, amax))

        for k in range(num_steps):
            opti.subject_to(s[k + 1] == s[k] + dt * v[k])
            opti.subject_to(v[k + 1] == v[k] + dt * a[k])

        opti.subject_to(s[crossing_index] <= light_s)
        opti.subject_to(s[crossing_index + 1] >= light_s)

        crossing_time = crossing_index * final_time / num_steps
        opti.subject_to(green_window[0] <= crossing_time)
        opti.subject_to(crossing_time <= green_window[1])

        energy = ca.sumsqr(a) * dt
        if num_steps > 1:
            jerk = ca.sumsqr(a[1:] - a[:-1]) / dt
        else:
            jerk = 0.0
        objective = w_time * final_time + w_energy * energy + w_jerk * jerk
        opti.minimize(objective)

        light_ratio = max(light_s / road_length, 1e-3)
        crossing_time_guess = green_window[0] + 0.5 * (green_window[1] - green_window[0])
        guess_time = crossing_time_guess / light_ratio
        guess_time = min(max(guess_time, min_final_time), max_final_time)
        opti.set_initial(final_time, guess_time)
        opti.set_initial(s, np.linspace(0.0, road_length, num_steps + 1))
        opti.set_initial(v, max(0.05, min(vmax * 0.5, road_length / guess_time)))
        opti.set_initial(a, 0.0)

        opti.solver("ipopt", {"ipopt.print_level": 0, "print_time": 0})
        sol = opti.solve()

        final_time_value = float(sol.value(final_time))
        t_values = np.linspace(0.0, final_time_value, num_steps + 1)
        s_values = np.array(sol.value(s)).reshape(-1)
        v_values = np.array(sol.value(v)).reshape(-1)
        a_values = np.r_[np.array(sol.value(a)).reshape(-1), 0.0]
        traj = np.column_stack((t_values, s_values, v_values, a_values))

        return PlanResult(
            objective=float(sol.value(objective)),
            final_time=final_time_value,
            crossing_time=float(crossing_index * final_time_value / num_steps),
            crossing_index=crossing_index,
            green_window=green_window,
            trajectory=traj,
            station_trajectory=traj,
            light_s=light_s,
            road_length=road_length,
        )

    def green_windows(self):
        red = float(self.get_parameter("red_time").value)
        yellow = float(self.get_parameter("yellow_time").value)
        green = float(self.get_parameter("green_time").value)
        max_time = float(self.get_parameter("max_final_time").value)
        allow_yellow = bool(self.get_parameter("allow_yellow_crossing").value)
        entry_margin = float(self.get_parameter("green_entry_margin").value)
        exit_margin = float(self.get_parameter("green_exit_margin").value)

        cycle = red + yellow + green + yellow
        windows = []
        cycle_start = 0.0
        while cycle_start <= max_time:
            if allow_yellow:
                start = cycle_start + red
                end = cycle_start + red + yellow + green
            else:
                start = cycle_start + red + yellow
                end = cycle_start + red + yellow + green
            start += entry_margin
            end -= exit_margin
            if start < end:
                windows.append((start, end))
            cycle_start += cycle
        return windows

    def get_xy(self, prefix: str) -> np.ndarray:
        return np.array(
            [
                float(self.get_parameter(f"{prefix}_x").value),
                float(self.get_parameter(f"{prefix}_y").value),
            ]
        )

    def publish_plan(self):
        path = Path()
        path.header.frame_id = "map"
        path.header.stamp = self.get_clock().now().to_msg()

        for row in self.plan.trajectory:
            _, x, y, yaw, _, _ = row
            pose = PoseStamped()
            pose.header = path.header
            pose.pose.position.x = float(x)
            pose.pose.position.y = float(y)
            qx, qy, qz, qw = tf.quaternion_from_euler(0.0, 0.0, float(yaw))
            pose.pose.orientation.x = qx
            pose.pose.orientation.y = qy
            pose.pose.orientation.z = qz
            pose.pose.orientation.w = qw
            path.poses.append(pose)
        self.path_pub.publish(path)

        msg = Float32MultiArray()
        msg.layout.dim = [
            MultiArrayDimension(label="points", size=self.plan.trajectory.shape[0], stride=self.plan.trajectory.size),
            MultiArrayDimension(label="fields:t,x,y,yaw,v,a", size=len(self.FIELDS), stride=len(self.FIELDS)),
        ]
        msg.data = self.plan.trajectory.astype(np.float32).reshape(-1).tolist()
        self.data_pub.publish(msg)
        self.publish_speed_plot()

    def publish_speed_plot(self):
        x0 = float(self.get_parameter("speed_plot_x").value)
        y0 = float(self.get_parameter("speed_plot_y").value)
        time_scale = float(self.get_parameter("speed_plot_time_scale").value)
        speed_scale = float(self.get_parameter("speed_plot_speed_scale").value)

        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "planned_speed"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.08
        marker.color.r = 0.1
        marker.color.g = 0.6
        marker.color.b = 1.0
        marker.color.a = 1.0

        for t, _, _, _, v, _ in self.plan.trajectory:
            point = PoseStamped().pose.position
            point.x = x0 + float(t) * time_scale
            point.y = y0 + float(v) * speed_scale
            point.z = 0.2
            marker.points.append(point)

        self.speed_plot_pub.publish(marker)

    def write_csv_if_requested(self):
        output_csv = self.get_parameter("output_csv").value
        if not output_csv:
            return
        header = ",".join(self.FIELDS)
        np.savetxt(output_csv, self.plan.trajectory, delimiter=",", header=header, comments="")
        self.get_logger().info(f"Wrote optimized trajectory to {output_csv}")

    def write_plots(self):
        output_dir = self.get_parameter("output_dir").value
        if not output_dir:
            return

        os.makedirs(output_dir, exist_ok=True)
        traj = self.plan.trajectory
        station_traj = self.plan.station_trajectory
        t = traj[:, 0]
        s = station_traj[:, 1]
        x = traj[:, 1]
        y = traj[:, 2]
        v = traj[:, 4]
        a = traj[:, 5]
        light_s = self.plan.light_s

        overview_path = os.path.join(output_dir, "state_time_overview.png")
        fig, axs = plt.subplots(3, 1, figsize=(9, 8), sharex=True)

        axs[0].plot(t, s, linewidth=2.0, label="position s")
        axs[0].axhline(light_s, color="tab:red", linestyle=":", label="traffic light position")
        axs[0].set_ylabel("position s [m]")
        axs[0].legend(loc="best")

        axs[1].plot(t, v, linewidth=2.0, label="speed v")
        axs[1].set_ylabel("speed [m/s]")
        axs[1].legend(loc="best")

        axs[2].plot(t, a, linewidth=2.0, color="tab:orange", label="acceleration a")
        axs[2].set_xlabel("time [s]")
        axs[2].set_ylabel("acceleration [m/s^2]")
        axs[2].legend(loc="best")

        for ax in axs:
            ax.axvline(self.plan.crossing_time, color="tab:red", linestyle="--", label="_nolegend_")
            ax.axvspan(self.plan.green_window[0], self.plan.green_window[1], color="tab:green", alpha=0.15)
            ax.grid(True, alpha=0.3)

        fig.suptitle("Optimized traffic-light trajectory")
        fig.tight_layout()
        fig.savefig(overview_path, dpi=160)
        plt.close(fig)

        speed_path = os.path.join(output_dir, "speed_time.png")
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(t, v, linewidth=2.0, label="planned speed")
        ax.axvline(self.plan.crossing_time, color="tab:red", linestyle="--", label="traffic light crossing")
        ax.axvspan(self.plan.green_window[0], self.plan.green_window[1], color="tab:green", alpha=0.15, label="green window")
        ax.set_xlabel("time [s]")
        ax.set_ylabel("speed [m/s]")
        ax.set_title("Planned speed profile")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(speed_path, dpi=160)
        plt.close(fig)

        position_path = os.path.join(output_dir, "position_time.png")
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(t, s, linewidth=2.0, label="planned position")
        ax.axhline(light_s, color="tab:red", linestyle=":", label="traffic light position")
        ax.axvline(self.plan.crossing_time, color="tab:red", linestyle="--", label="traffic light crossing")
        ax.axvspan(self.plan.green_window[0], self.plan.green_window[1], color="tab:green", alpha=0.15, label="green window")
        ax.set_xlabel("time [s]")
        ax.set_ylabel("position along road [m]")
        ax.set_title("Planned position profile")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(position_path, dpi=160)
        plt.close(fig)

        accel_path = os.path.join(output_dir, "acceleration_time.png")
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(t, a, linewidth=2.0, color="tab:orange")
        ax.axvline(self.plan.crossing_time, color="tab:red", linestyle="--")
        ax.axvspan(self.plan.green_window[0], self.plan.green_window[1], color="tab:green", alpha=0.15)
        ax.set_xlabel("time [s]")
        ax.set_ylabel("acceleration [m/s^2]")
        ax.set_title("Planned acceleration profile")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(accel_path, dpi=160)
        plt.close(fig)

        xy_path = os.path.join(output_dir, "xy_path.png")
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(x, y, linewidth=2.0)
        ax.scatter([x[0], x[-1]], [y[0], y[-1]], c=["tab:blue", "tab:purple"], label="start / goal")
        ax.scatter([np.interp(self.plan.crossing_time, t, x)], [np.interp(self.plan.crossing_time, t, y)], c="tab:red", label="traffic light")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_title("Planned map path")
        ax.axis("equal")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(xy_path, dpi=160)
        plt.close(fig)

        csv_path = os.path.join(output_dir, "trajectory.csv")
        np.savetxt(csv_path, traj, delimiter=",", header=",".join(self.FIELDS), comments="")

        self.get_logger().info(f"Wrote plan plots to {output_dir}")


def main(args=None):
    rclpy.init(args=args)
    node = PlanningTrafficLight()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
